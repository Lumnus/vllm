# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0040: routed experts split between VRAM (hot) and pinned host RAM (cold).

For INT4 W4A16 MoE on XPU (XPUExpertsWNA16). Each rank keeps the
``B70_MOE_HOT_EXPERTS`` most-routed of its local experts per layer as the
layer's ordinary parameters (in VRAM) and the rest in pinned host memory,
which the same grouped-GEMM kernel reads through a UVA view (zero copy over
PCIe). Both calls see the full ``topk_ids``; each maps only its own experts
(``expert_map`` -1 elsewhere, the mechanism expert parallelism already uses),
and the two outputs are summed. The split is exact: a row's result does not
depend on where its experts live.

Load path (no full-size host or device copy of a layer ever exists):
  * ``create_weights`` is called with the hot count, so the device params hold
    only hot experts;
  * the weight loader writes a cold expert into a pageable CPU twin of the
    same param (GPTQ layout);
  * after loading, each twin is repacked into the kernel layout (transpose,
    unsigned -> signed int4) on the device, 16 experts at a time, copied into
    pinned memory and dropped (the host only copies: an all-core CPU repack on
    both ranks at once was the heaviest burst of the whole boot).

Switches: B70_MOE_HOTCOLD=1, B70_MOE_HOT_EXPERTS=<per layer per rank>,
B70_MOE_HOTCOLD_RANKINGS=<json: layer -> global expert ids, hottest first>.
Off (default): nothing here runs.
"""

import json
import os
import re
from functools import cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Per-expert params of the GPTQ MoE method that get a cold twin.
_EXPERT_PARAMS = ("w13_qweight", "w13_scales", "w13_qzeros", "w13_bias",
                  "w2_qweight", "w2_scales", "w2_qzeros", "w2_bias")


def hotcold_enabled() -> bool:
    return os.environ.get("B70_MOE_HOTCOLD", "0") == "1"


@cache
def _rankings() -> dict[int, list[int]]:
    path = os.environ.get("B70_MOE_HOTCOLD_RANKINGS", "")
    if not path:
        return {}
    with open(path) as handle:
        raw = json.load(handle)
    return {int(k): [int(e) for e in v] for k, v in raw.items()}


class _PinnedArena:
    """Pinned host memory carved from 1 GiB chunks (first fit, 4 KiB aligned).

    torch's caching host allocator rounds every pinned block up to a power of
    two (a 197 MB cold w13 takes 256 MB), so per-tensor pinning wastes ~30 %
    of the cold experts' RAM. Chunks of exactly 2^30 B are not rounded.
    """

    CHUNK = 1 << 30
    ALIGN = 4096

    def __init__(self) -> None:
        self.chunks: list[torch.Tensor] = []
        self.used: list[int] = []

    def empty(self, shape, dtype: torch.dtype) -> torch.Tensor:
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        if nbytes > self.CHUNK:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        for i, used in enumerate(self.used):
            if used + nbytes <= self.CHUNK:
                break
        else:
            self.chunks.append(torch.empty(self.CHUNK, dtype=torch.uint8, pin_memory=True))
            self.used.append(0)
            i, used = len(self.chunks) - 1, 0
        self.used[i] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        return self.chunks[i][used : used + nbytes].view(dtype).view(shape)

    @property
    def gib(self) -> float:
        return len(self.chunks) * self.CHUNK / 2**30


_ARENA = _PinnedArena()
_FINALIZED: list[str] = []
_COUNTED: list["HotColdPlan"] = []  # plans with a route counter (B70_MOE_ROUTE_COUNT)
_DUMP = {"t": 0.0}


_PLANS: dict[int, "HotColdPlan"] = {}  # finalized plans by layer index


def cold_prefetch_min_tokens() -> int:
    """0 = off; else batches with at least this many tokens run the cold experts from VRAM staging."""
    if os.environ.get("B70_MOE_COLD_PREFETCH", "0") != "1":
        return 0
    return max(64, int(os.environ.get("B70_MOE_COLD_PREFETCH_MIN_TOKENS", "512")))


class _ColdPrefetcher:
    """Two VRAM slots, each holding one layer's cold experts (kernel layout).

    Large batches (prefill) run the cold experts from a slot instead of reading
    pinned host memory through UVA: layer L computes from one slot while a
    copy stream DMAs layer L+1's cold experts into the other. Events order the
    two streams: a slot is refilled only after the GEMM that read it, and a
    GEMM starts only after its slot's copy. The math is unchanged (same kernel,
    same bytes); only where the kernel reads the weights from differs."""

    def __init__(self) -> None:
        self.bufs: list[dict[str, torch.Tensor]] | None = None
        self.owner: list[int | None] = [None, None]
        self.ready: list = [None, None]
        self.free: list = [None, None]
        self.last = 1
        self.stream = None

    def _ensure(self, plan: "HotColdPlan", device) -> None:
        if self.bufs is not None:
            return
        self.bufs = [{k: torch.empty(t.shape, dtype=t.dtype, device=device)
                      for k, t in plan.cold.items()} for _ in range(2)]
        self.stream = torch.xpu.Stream(device)
        gib = 2 * sum(t.numel() * t.element_size() for t in plan.cold.values()) / 2**30
        logger.info("B70 hot/cold: cold prefetch on (batches >= %d tokens), 2 VRAM slots, %.2f GiB",
                    cold_prefetch_min_tokens(), gib)

    def _load(self, slot: int, plan: "HotColdPlan") -> None:
        compute = torch.xpu.current_stream()
        with torch.xpu.stream(self.stream):
            if self.free[slot] is not None:
                self.stream.wait_event(self.free[slot])
            for key, src in plan.cold.items():
                self.bufs[slot][key].copy_(src, non_blocking=True)
            event = torch.xpu.Event()
            event.record(self.stream)
        self.ready[slot] = event
        self.owner[slot] = plan.layer_idx
        del compute

    def acquire(self, plan: "HotColdPlan", device) -> dict[str, torch.Tensor]:
        self._ensure(plan, device)
        if plan.layer_idx in self.owner:
            slot = self.owner.index(plan.layer_idx)
        else:
            slot = 1 - self.last
            self._load(slot, plan)
        torch.xpu.current_stream().wait_event(self.ready[slot])
        self.last = slot
        return self.bufs[slot]

    def release_and_prefetch(self, plan: "HotColdPlan") -> None:
        slot = self.last
        event = torch.xpu.Event()
        event.record(torch.xpu.current_stream())
        self.free[slot] = event
        keys = sorted(_PLANS)
        nxt = _PLANS[keys[(keys.index(plan.layer_idx) + 1) % len(keys)]]
        if nxt.layer_idx not in self.owner:
            self._load(1 - slot, nxt)


_PREFETCH = _ColdPrefetcher()


def route_count_enabled() -> bool:
    return os.environ.get("B70_MOE_ROUTE_COUNT", "0") == "1"


def maybe_dump_routes() -> None:
    """Log the cold share of routes and write the cumulative counts (every
    B70_MOE_ROUTE_COUNT_S seconds). Called by the runner on the host right
    after its per-step sync, so the .cpu() below does not stall the GPU."""
    import time

    if not _COUNTED:
        return
    now = time.monotonic()
    period = float(os.environ.get("B70_MOE_ROUTE_COUNT_S", "60"))
    if _DUMP["t"] == 0.0:
        _DUMP["t"] = now
        return
    if now - _DUMP["t"] < period:
        return
    _DUMP["t"] = now
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = 0
    out, local_all, cold_all, per_layer = {}, 0, 0, []
    for plan in _COUNTED:
        counts = plan.route_counts[:-1].cpu()
        local = (plan.hot_map >= 0).cpu() | (plan.cold_map >= 0).cpu()
        cold = (plan.cold_map >= 0).cpu()
        n_local, n_cold = int(counts[local].sum()), int(counts[cold].sum())
        local_all += n_local
        cold_all += n_cold
        per_layer.append((n_cold / n_local) if n_local else 0.0)
        match = _LAYER_RE.search(plan.layer_name)
        out[match.group(1) if match else plan.layer_name] = counts.tolist()
    share = cold_all / local_all if local_all else 0.0
    worst = sorted(range(len(per_layer)), key=lambda i: -per_layer[i])[:3]
    logger.info("B70 routes rank %d: %d local routes, cold share %.3f (layers min %.3f max %.3f; "
                "worst %s)", rank, local_all, share, min(per_layer), max(per_layer),
                ", ".join(f"{i}:{per_layer[i]:.2f}" for i in worst))
    path = os.environ.get("B70_MOE_ROUTE_COUNT_PATH", "")
    if path:
        with open(f"{path}-rank{rank}.json", "w") as handle:
            json.dump(out, handle)


def _pin(t: torch.Tensor) -> torch.Tensor:
    pinned = _ARENA.empty(t.shape, t.dtype)
    pinned.copy_(t)
    return pinned


def _repack_int4_pinned(qweight: torch.Tensor, device: torch.device) -> torch.Tensor:
    """GPTQ int32 [E, in/8, out] (CPU) -> signed-int4 bytes [E, out, in/2] in pinned memory.

    The transforms of int_wna16._process_weights_xpu (transpose, int32 -> uint8 view)
    and vllm_xpu_kernels.implement_zp, run on ``device`` a few experts at a time: the
    host only copies, and the device temporaries stay at a few hundred MB.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    e, k8, n = qweight.shape
    out = _ARENA.empty((e, n, k8 * 4), torch.uint8)
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        packed = chunk.transpose(1, 2).contiguous().view(torch.uint8)
        out[i : i + step].copy_(implement_zp(packed))
    return out


class HotColdPlan:
    """Which local experts of one layer live where, plus the cold storage."""

    def __init__(self, layer_name: str, local_ids: list[int], global_ids: list[int],
                 num_global: int, hot_local: list[int]):
        self.layer_name = layer_name
        self.num_global = num_global
        hot_set = set(hot_local)
        self.hot_local = sorted(hot_set)
        self.cold_local = [i for i in local_ids if i not in hot_set]
        # local id -> (is_cold, slot)
        self.slot: dict[int, tuple[bool, int]] = {}
        for s, i in enumerate(self.hot_local):
            self.slot[i] = (False, s)
        for s, i in enumerate(self.cold_local):
            self.slot[i] = (True, s)
        self.local_to_global = dict(zip(local_ids, global_ids))
        self.twins: dict[str, torch.nn.Parameter] = {}
        self._param_names: dict[int, str] = {}
        self.cold: dict[str, torch.Tensor] = {}  # pinned, kernel layout
        self.cold_views: dict[str, torch.Tensor] = {}
        self.hot_map: torch.Tensor | None = None
        self.cold_map: torch.Tensor | None = None
        self._impls = None

    @property
    def num_hot(self) -> int:
        return len(self.hot_local)

    @property
    def num_cold(self) -> int:
        return len(self.cold_local)

    # -- load ---------------------------------------------------------------

    def attach_twins(self, layer: torch.nn.Module) -> None:
        """A pageable CPU twin [num_cold, ...] for every per-expert param."""
        for name, param in layer.named_parameters(recurse=False):
            if name not in _EXPERT_PARAMS or param.ndim == 0 or param.shape[0] != self.num_hot:
                continue
            twin = torch.nn.Parameter(
                # explicit CPU: the model is built under a default-device context
                torch.zeros((self.num_cold, *param.shape[1:]), dtype=param.dtype,
                            device="cpu"),
                requires_grad=False,
            )
            for key, value in vars(param).items():
                if not key.startswith("_"):
                    setattr(twin, key, value)
            self.twins[name] = twin
            self._param_names[id(param)] = name

    def route(self, param: torch.nn.Parameter, local_id: int):
        """(param, slot) the loader should write expert ``local_id`` into."""
        is_cold, slot = self.slot[local_id]
        if not is_cold:
            return param, slot
        name = self._param_names.get(id(param))
        if name is None:
            raise RuntimeError(
                f"B70 hot/cold: {self.layer_name}: no cold twin for a param of "
                f"shape {tuple(param.shape)}"
            )
        return self.twins[name], slot

    # -- after loading ---------------------------------------------------------

    def finalize(self, layer: torch.nn.Module, moe_kernel) -> None:
        """Repack the cold twins into pinned kernel-layout tensors; wire the kernel."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        for bias in ("w13_bias", "w2_bias"):
            if getattr(layer, bias, None) is not None:
                raise NotImplementedError(f"B70 hot/cold: {bias} is not supported")
        twins, self.twins, self._param_names = self.twins, {}, {}
        # Same transforms as int_wna16._process_weights_xpu, then implement_zp.
        for key in ("w13", "w2"):
            self.cold[key] = _repack_int4_pinned(
                twins.pop(f"{key}_qweight").data, layer.w13_qweight.device
            )
            scales = twins.pop(f"{key}_scales").data.transpose(1, 2).contiguous()
            self.cold[f"{key}_scales"] = _pin(scales.to(getattr(layer, f"{key}_scales").dtype))
            del scales
        del twins  # qzeros: symmetric int4 has none on XPU
        for key, tensor in self.cold.items():
            self.cold_views[key] = get_accelerator_view_from_cpu_tensor(tensor)

        device = layer.w13_qweight.device
        hot_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        cold_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        for local_id, (is_cold, slot) in self.slot.items():
            (cold_map if is_cold else hot_map)[self.local_to_global[local_id]] = slot
        self.hot_map = hot_map.to(device)
        self.cold_map = cold_map.to(device)
        match = _LAYER_RE.search(self.layer_name)
        self.layer_idx = int(match.group(1)) if match else len(_PLANS)
        _PLANS[self.layer_idx] = self
        self.route_counts = None
        if route_count_enabled():
            # one bucket per global expert + one for negative (padding) ids
            self.route_counts = torch.zeros(self.num_global + 1, dtype=torch.int64,
                                            device=device)
            _COUNTED.append(self)

        experts = getattr(moe_kernel, "fused_experts", None)
        if experts is None or not hasattr(experts, "b70_apply_hotcold"):
            raise NotImplementedError(
                f"B70 hot/cold needs XPUExpertsWNA16, got {type(experts).__name__}"
            )
        experts.b70_hotcold = self
        gib = sum(t.numel() * t.element_size() for t in self.cold.values()) / 2**30
        logger.debug("B70 hot/cold: %s: %d hot in VRAM, %d cold pinned (%.2f GiB); "
                     "arena %.0f GiB", self.layer_name, self.num_hot, self.num_cold,
                     gib, _ARENA.gib)
        _FINALIZED.append(self.layer_name)
        if len(_FINALIZED) % 12 == 0 or os.environ.get("B70_MOE_HOTCOLD_MEMLOG") == "1":
            free, _ = torch.xpu.mem_get_info(device)
            logger.info("B70 hot/cold: %d layers split (%d hot in VRAM, %d cold); "
                        "cold pinned arena %.0f GiB; device: torch allocated %.2f GiB, "
                        "reserved %.2f GiB, free %.2f GiB", len(_FINALIZED), self.num_hot,
                        self.num_cold, _ARENA.gib,
                        torch.xpu.memory_allocated(device) / 2**30,
                        torch.xpu.memory_reserved(device) / 2**30, free / 2**30)

    # -- forward ---------------------------------------------------------------

    def apply_cold(self, cold, output, hidden_states, topk_weights, topk_ids) -> None:
        """The cold experts' share: UVA reads (decode, graphs) or a VRAM slot (large batches)."""
        min_tokens = cold_prefetch_min_tokens()
        if not min_tokens or hidden_states.shape[0] < min_tokens:
            cold.apply(output=output, hidden_states=hidden_states,
                       topk_weights=topk_weights, topk_ids=topk_ids)
            return
        import copy

        buf = _PREFETCH.acquire(self, hidden_states.device)
        staged = copy.copy(cold)
        staged.w13, staged.w2 = buf["w13"].view(torch.int8), buf["w2"].view(torch.int8)
        staged.gemm1_wei_scales, staged.gemm2_wei_scales = buf["w13_scales"], buf["w2_scales"]
        staged.apply(output=output, hidden_states=hidden_states,
                     topk_weights=topk_weights, topk_ids=topk_ids)
        _PREFETCH.release_and_prefetch(self)

    def impls(self, w1, w1_scale, w2, w2_scale, topk, activation, extra):
        """(hot, cold) XpuFusedMoe, built on first use like the stock path."""
        if self._impls is None:
            from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

            common = dict(n_experts_per_token=topk, activation=activation,
                          ep_rank=0, ep_size=2, **extra)
            hot = XpuFusedMoe(w13=w1, w13_scales=w1_scale, w13_bias=None, w2=w2,
                              w2_scales=w2_scale, w2_bias=None,
                              num_experts=self.num_hot, expert_map=self.hot_map,
                              **common)
            cw13, cw2 = self.cold_views["w13"], self.cold_views["w2"]
            for t in (cw13, cw2):
                t.xpu_fused_moe = True  # already signed int4
            cold = XpuFusedMoe(w13=cw13, w13_scales=self.cold_views["w13_scales"],
                               w13_bias=None, w2=cw2,
                               w2_scales=self.cold_views["w2_scales"], w2_bias=None,
                               num_experts=self.num_cold, expert_map=self.cold_map,
                               **common)
            # sizes come from the uint8 view at init; the kernel reads int4 as int8
            cold.w13, cold.w2 = cw13.view(torch.int8), cw2.view(torch.int8)
            self._impls = (hot, cold)
        return self._impls


def maybe_plan(layer_name: str, moe_config, expert_map_manager,
               quant_config) -> HotColdPlan | None:
    """A plan for this RoutedExperts layer, or None (switch off / not applicable)."""
    if not hotcold_enabled() or quant_config is None:
        return None
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        return None
    match = _LAYER_RE.search(layer_name)
    if match is None or "mtp" in layer_name:
        return None  # the MTP draft layer stays fully in VRAM
    layer_idx = int(match.group(1))
    num_global = moe_config.num_experts
    local_of = {g: expert_map_manager.map_global_to_local(g) for g in range(num_global)}
    pairs = sorted((loc, g) for g, loc in local_of.items() if loc >= 0)
    local_ids = [loc for loc, _ in pairs]
    global_ids = [g for _, g in pairs]
    hot_count = int(os.environ.get("B70_MOE_HOT_EXPERTS", str(len(local_ids))))
    hot_count = max(1, min(hot_count, len(local_ids)))
    if hot_count >= len(local_ids):
        return None
    ranked = [g for g in _rankings().get(layer_idx, []) if local_of.get(g, -1) >= 0]
    seen = set(ranked)
    ranked += [g for g in global_ids if g not in seen]  # never observed: by id
    hot_local = [local_of[g] for g in ranked[:hot_count]]
    return HotColdPlan(layer_name, local_ids, global_ids, num_global, hot_local)
