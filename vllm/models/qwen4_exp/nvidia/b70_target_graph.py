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


def save(routes):
    return [tensor[indices].clone() for _, tensor, indices in routes]


def restore(routes, values):
    for (_, tensor, indices), value in zip(routes, values):
        tensor[indices] = value


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
                and width == 8
                and x.shape[1] == 2560
                and os.getenv("CCL_SYCL_ALLREDUCE_LL_THRESHOLD") == "65536"
                and os.getenv("VLLM_XPU_C1_PARTITIONED_REDUCE", "1") == "1"
            ):
                # The native ring changes its BF16 summation order by partition.
                # Keep each row's quarters in the same rank-owned partition.
                # Both 5 KiB rows and this 40 KiB batch use the LL ring path.
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
    if id(destination) in seen:
        return
    seen.add(id(destination))
    if isinstance(destination, torch.Tensor):
        assert destination.shape == source.shape and destination.dtype == source.dtype
        destination.copy_(source)
    elif isinstance(destination, dict):
        assert destination.keys() == source.keys()
        for key in destination:
            update(destination[key], source[key], seen)
    elif dataclasses.is_dataclass(destination):
        for field in dataclasses.fields(destination):
            if field.name == "max_seq_len":
                assert getattr(source, field.name) <= destination.max_seq_len
                continue
            update(getattr(destination, field.name), getattr(source, field.name), seen)
    elif isinstance(destination, (tuple, list)):
        assert len(destination) == len(source)
        for a, b in zip(destination, source):
            update(a, b, seen)
    else:
        assert destination == source, (destination, source)


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
        self.pool = torch.xpu.graph_pool_handle()

    def run(self, runner, batch, metadata, inputs, eager):
        # The runner admits one uniform C1 verification. Partial blocks use eager.
        if batch.num_tokens != 8:
            return eager(runner, batch, metadata, inputs)
        bound = max(512, 1 << (int(batch.seq_lens_cpu_upper_bound[0]) - 1).bit_length())
        if bound in self.disabled:
            self.fallbacks += 1
            return eager(runner, batch, metadata, inputs)
        context = get_forward_context()
        current = (inputs, metadata, context.slot_mapping, context.is_padding)
        entry = self.entries.get(bound)
        if entry is not None:
            try:
                update(entry[2], current, set())
            except (AssertionError, TypeError, AttributeError):
                self.fallbacks += 1
                return eager(runner, batch, metadata, inputs)
            entry[0].replay()
            self.replays += 1
            return entry[1]

        # Capture must never advance the accepted-prefix state. Several layers
        # share cache arenas: union every layer's indices before saving them.
        routes = state_routes(runner, metadata)
        initial = save(routes)
        expected = clone_output(eager(runner, batch, metadata, inputs))
        expected_state = save(routes)
        restore(routes, initial)
        static = stage(current, {}, bound)
        model_inputs, model_metadata, slots, padding = static
        try:
            with (
                staged_context(context, model_metadata, slots, padding),
                native_rows(8),
            ):
                runner.model(**model_inputs)
                torch.xpu.synchronize()
                restore(routes, initial)
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, pool=self.pool):
                    output = runner.model(**model_inputs)
            restore(routes, initial)
            graph.replay()
            torch.xpu.synchronize()
            observed_state = save(routes)
            passed = compare(expected, output) and all(
                torch.equal(a, b) for a, b in zip(expected_state, observed_state)
            )
            verdict = torch.tensor(int(passed), dtype=torch.int32)
            dist.all_reduce(
                verdict, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group
            )
            if verdict.item():
                self.entries[bound] = (graph, output, static)
                self.replays += 1
                logger.info(
                    "C1 native target graph ready: width=8, context bound=%d", bound
                )
                return output
            logger.warning(
                "C1 target graph differs from eager at bound %d; using eager", bound
            )
        except Exception:
            logger.exception("C1 target graph capture failed at bound %d", bound)
        restore(routes, expected_state)
        self.disabled.add(bound)
        self.fallbacks += 1
        return expected
