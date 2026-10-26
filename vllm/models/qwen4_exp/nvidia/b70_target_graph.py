# SPDX-License-Identifier: Apache-2.0
"""C1 target replay with native row collectives and live hybrid-state metadata."""

import copy
import dataclasses
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

_GRAPH_CHECK = os.environ.get("B70_QSA_GRAPH_CHECK", "").strip() in ("1", "true", "on")


def state_routes(runner, metadata):
    context = runner.vllm_config.compilation_config.static_forward_context
    grouped = {}
    for name, meta in metadata.items():
        cache = getattr(context[name], "kv_cache", None)
        indices = getattr(meta, "spec_state_indices_tensor", None)
        if not isinstance(cache, (tuple, list)) or indices is None:
            continue
        indices = indices.flatten().long()
        for tensor in cache:
            key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            if key not in grouped:
                grouped[key] = (name, tensor, [])
            grouped[key][2].append(indices)
    return [
        (name, tensor, torch.cat(indices).unique())
        for name, tensor, indices in grouped.values()
    ]


def qsa_side_rows(runner, metadata):
    """B70-QSA-Q: the QSA side-cache rows this step writes (B70_QSA_GRAPH_CHECK).

    The side caches are not state routes: their writes are expected to be
    idempotent for a repeated 8-row verify step. This lets the capture prove
    it on the live batch instead of assuming it.
    """
    from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata

    context = runner.vllm_config.compilation_config.static_forward_context
    found = []
    for name, meta in metadata.items():
        if not isinstance(meta, QSAForwardMetadata):
            continue
        cache = getattr(context.get(name), "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or not cache.numel():
            continue
        slots = meta.slot_mapping[: meta.num_actual_tokens].long()
        page = cache.shape[1]
        safe = slots.clamp_min(0)
        found.append((name, cache, safe // page, safe % page, slots >= 0))
    return found


def qsa_side_snapshot(side):
    return [
        (torch.where(valid.view(-1, 1, 1), cache[blocks, offsets], 0).clone(),
         cache[0].clone())
        for _, cache, blocks, offsets, valid in side
    ]


# B70-C1-R1: the capture snapshots (``initial`` and ``expected_state``) live in host
# RAM, moved through the device in chunks of at most this many bytes. On the device
# they were 2 x (requests x (1 + K) state blocks x every GDN layer): ~0.45 GiB per
# request per rank at TP2, plus one transient gather copy, none of it seen by the
# memory profiler; TP2 warm-up (3 requests) OOMed at the second gather. Bytes are copied exactly, so every check below is unchanged.
SNAPSHOT_CHUNK_BYTES = 256 << 20


def _rows_per_chunk(tensor):
    row = tensor[0].numel() * tensor.element_size() if tensor.shape[0] else 1
    return max(1, SNAPSHOT_CHUNK_BYTES // max(row, 1))


def save(routes):
    values = []
    for _, tensor, indices in routes:
        host = torch.empty((indices.numel(), *tensor.shape[1:]), dtype=tensor.dtype)
        step = _rows_per_chunk(tensor)
        for i in range(0, indices.numel(), step):
            host[i:i + step].copy_(tensor[indices[i:i + step]])
        values.append(host)
    return values


def restore(routes, values):
    for (_, tensor, indices), value in zip(routes, values):
        step = _rows_per_chunk(tensor)
        for i in range(0, indices.numel(), step):
            tensor[indices[i:i + step]] = value[i:i + step].to(tensor.device)


def state_equal(routes, values):
    """Exact comparison of the live state rows with a host snapshot, chunk by chunk."""
    for (_, tensor, indices), value in zip(routes, values):
        step = _rows_per_chunk(tensor)
        for i in range(0, indices.numel(), step):
            if not torch.equal(value[i:i + step].to(tensor.device), tensor[indices[i:i + step]]):
                return False
    return True


def capture_reserve_bytes(runner) -> int:
    """Device bytes a target-graph capture needs beyond the step itself (B70-C1-R1):
    one gathered chunk plus one chunk staged back, while the snapshots are in host RAM.
    0 when the C1 target-graph path cannot run on this engine."""
    spec = getattr(runner, "speculative_config", None)
    try:
        enabled = (
            spec is not None
            and spec.method == "dflash"
            and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
            and runner.device.type == "xpu"
            and runner.model_config.hf_text_config.model_type == "qwen4_exp_text"
            and os.getenv("VLLM_XPU_C1_TARGET_GRAPH", "1") == "1"
        )
    except AttributeError:
        return 0
    return 2 * SNAPSHOT_CHUNK_BYTES if enabled else 0


def clone_output(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return type(value)(clone_output(x) for x in value)


def compare(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return all(compare(x, y) for x, y in zip(a, b))


@contextmanager
def native_rows(width):
    from vllm.distributed import get_tp_group

    comm = get_tp_group().device_communicator
    original = comm.all_reduce

    def reduce(x):
        if x.ndim == 2 and x.shape[0] == width and x.dtype == torch.bfloat16:
            if (
                comm.world_size == 4
                and width % 8 == 0
                and x.shape[1] == 2560
                and os.getenv("CCL_SYCL_ALLREDUCE_LL_THRESHOLD") == "65536"
                and os.getenv("VLLM_XPU_C1_PARTITIONED_REDUCE", "1") == "1"
            ):
                # The native ring changes its BF16 summation order by partition.
                # Keep each row's quarters in the same rank-owned partition.
                # Full 8/16/24/32-row batches preserve this partition order.
                packed = x.reshape(width, 4, -1).transpose(0, 1).contiguous()
                dist.all_reduce(packed, group=comm.device_group)
                return packed.transpose(0, 1).contiguous().reshape_as(x)
            return torch.cat([original(row) for row in x.split(1)])
        return original(x)

    comm.all_reduce = reduce
    try:
        yield
    finally:
        comm.all_reduce = original


def stage(value, memo, bound):
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif isinstance(value, dict):
        result = {k: stage(v, memo, bound) for k, v in value.items()}
    elif dataclasses.is_dataclass(value):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            v = getattr(value, field.name)
            setattr(result, field.name, stage(v, memo, bound))
        if hasattr(result, "max_seq_len"):
            result.max_seq_len = bound
    elif isinstance(value, (tuple, list)):
        result = type(value)(stage(v, memo, bound) for v in value)
    else:
        return value
    memo[id(value)] = result
    return result


def update(destination, source, seen):
    """Validate the schema, then submit required device copies together."""
    destinations, sources = [], []
    def visit(destination, source):
        if id(destination) in seen:
            return
        seen.add(id(destination))
        if isinstance(destination, torch.Tensor):
            assert destination.shape == source.shape and destination.dtype == source.dtype
            assert destination.device == source.device
            if destination.device.type == "xpu":
                destinations.append(destination)
                sources.append(source)
            else:
                destination.copy_(source)
        elif isinstance(destination, dict):
            assert destination.keys() == source.keys()
            for key in destination:
                visit(destination[key], source[key])
        elif dataclasses.is_dataclass(destination):
            for field in dataclasses.fields(destination):
                if field.name == "max_seq_len":
                    assert getattr(source, field.name) <= destination.max_seq_len
                    continue
                visit(getattr(destination, field.name), getattr(source, field.name))
        elif isinstance(destination, (tuple, list)):
            assert len(destination) == len(source)
            for a, b in zip(destination, source):
                visit(a, b)
        else:
            assert destination == source, (destination, source)
    visit(destination, source)
    if destinations:
        torch._foreach_copy_(destinations, sources)


@contextmanager
def staged_context(context, metadata, slots, padding):
    previous = context.attn_metadata, context.slot_mapping, context.is_padding
    context.attn_metadata, context.slot_mapping, context.is_padding = (
        metadata,
        slots,
        padding,
    )
    try:
        yield
    finally:
        context.attn_metadata, context.slot_mapping, context.is_padding = previous


class TargetGraphs:
    def __init__(self):
        self.entries = {}
        self.disabled = set()
        self.replays = 0
        self.fallbacks = 0
        self.fallback_reasons = {}
        self.pool = torch.xpu.graph_pool_handle()

    def fallback(self, reason, key):
        self.fallbacks += 1
        count = self.fallback_reasons.get(reason, 0) + 1
        self.fallback_reasons[reason] = count
        if count == 1 or count % 1024 == 0:
            logger.warning("Target graph fallback: key=%s reason=%s count=%d", key, reason, count)

    def run(self, runner, batch, metadata, inputs, eager):
        # Uniform eight-row verification for one through four requests.
        if batch.num_tokens != 8 * batch.num_reqs:
            return eager(runner, batch, metadata, inputs)
        if os.getenv("VLLM_XPU_DFLASH_FIXED_SPLITS", "0") in ("32", "64"):
            # The target split schedule no longer depends on this host bound.
            # One graph follows live device lengths across context boundaries.
            bound = runner.model_config.max_model_len
        else:
            bound = max(
                512, 1 << (int(batch.seq_lens_cpu_upper_bound[0]) - 1).bit_length()
            )
        key = (batch.num_reqs, bound)
        if key in self.disabled:
            self.fallback("disabled", key)
            return eager(runner, batch, metadata, inputs)
        context = get_forward_context()
        # The runner's forward context includes target AND draft metadata.
        # Only bind consumers in this target model: draft softmax workspaces
        # and mappings are neither inputs nor outputs of the target graph.
        names = getattr(self, "target_context_names", None)
        if names is None:
            modules = set(runner.model.modules())
            names = self.target_context_names = frozenset(
                name for name, module in runner.vllm_config.compilation_config.static_forward_context.items()
                if module in modules
            )
            assert names and all(name in names for name in metadata if name.startswith("language_model."))
        target_metadata = {name: value for name, value in metadata.items() if name in names}
        target_slots = {name: value for name, value in context.slot_mapping.items() if name in names}
        current = (inputs, target_metadata, target_slots, context.is_padding)
        entry = self.entries.get(key)
        if entry is not None:
            try:
                update(entry[2], current, set())
            except (AssertionError, TypeError, AttributeError) as exc:
                self.fallback(type(exc).__name__ + ": " + str(exc), key)
                return eager(runner, batch, metadata, inputs)
            entry[0].replay()
            self.replays += 1
            if self.replays % 1024 == 0:
                logger.info("Target graph counters: replays=%d fallbacks=%d key=%s",
                            self.replays, self.fallbacks, key)
            return entry[1]

        # Capture must never advance the accepted-prefix state. Several layers
        # share cache arenas: union every layer's indices before saving them.
        routes = state_routes(runner, metadata)
        side = qsa_side_rows(runner, target_metadata) if _GRAPH_CHECK else []
        side_initial = qsa_side_snapshot(side)
        initial = save(routes)
        expected = clone_output(eager(runner, batch, metadata, inputs))
        expected_state = save(routes)
        side_expected = qsa_side_snapshot(side)
        restore(routes, initial)
        static = stage(current, {}, bound)
        model_inputs, model_metadata, slots, padding = static
        try:
            with (
                staged_context(context, model_metadata, slots, padding),
                native_rows(batch.num_tokens),
            ):
                runner.model(**model_inputs)
                torch.xpu.synchronize()
                restore(routes, initial)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, pool=self.pool):
                    output = runner.model(**model_inputs)
            restore(routes, initial)
            del initial
            graph.replay()
            torch.xpu.synchronize()
            # Check every state exactly without another full observed-state copy.
            passed = compare(expected, output) and state_equal(routes, expected_state)
            if side:
                side_replayed = qsa_side_snapshot(side)
                bad = [
                    side[i][0]
                    for i, ((rows, null), (rows2, null2), (_, null0)) in enumerate(
                        zip(side_expected, side_replayed, side_initial)
                    )
                    if not (torch.equal(rows, rows2) and torch.equal(null2, null0))
                ]
                logger.info(
                    "B70-QSA-GRAPH-CHECK: %d QSA side caches, %d differ after "
                    "replay: %s", len(side), len(bad), bad[:4]
                )
                passed = passed and not bad
            verdict = torch.tensor(int(passed), dtype=torch.int32)
            dist.all_reduce(
                verdict, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group
            )
            if verdict.item():
                self.entries[key] = (graph, output, static)
                self.replays += 1
                logger.info(
                    "Target graph ready: requests=%d, width=%d, context bound=%d",
                    batch.num_reqs, batch.num_tokens, bound
                )
                return output
            logger.warning(
                "C1 target graph differs from eager at bound %d; using eager", bound
            )
        except Exception:
            logger.exception("C1 target graph capture failed at bound %d", bound)
        restore(routes, expected_state)
        self.disabled.add(key)
        self.fallback("capture_or_equality_failure", key)
        return expected
