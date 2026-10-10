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
