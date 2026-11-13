# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed expert slots in device and host tiers on XPU, placed by EPLB.

``--cpu-offload-gb`` offloads whole parameters in module order, so for a MoE
model the first layers end up with every expert in host memory and the last
ones with none. With ``--cpu-offload-per-expert`` (and ``--enable-eplb``) the
budget is spent per expert slot instead: every MoE decoder layer keeps the
same number of its local slots in pinned host memory (the slow tier, read by
the kernels through UVA) and the rest in device memory (the fast tier).

The slots are EPLB's physical slots. EPLB decides which logical expert sits
in which slot from its load statistics and moves the bytes; it addresses a
layer's tensors as ``TieredExpertWeight``, one row per slot across both
storages, and its tier step keeps each rank's most-loaded experts in the fast
slots. The slot -> tier row maps are fixed, so the experts here only run the
stock grouped GEMM twice, over the fast and over the slow slots, each with an
``expert_map`` that maps only its own slots (-1 elsewhere), and sum the two
outputs. Every expert stays routable; results equal the all-resident layer up
to the order of that sum.

With an expert store and ``--cpu-offload-pinned-gb`` below the host budget,
the pinned rows hold only part of the slots that are off the card; the others
are *host slots*, in neither map, computed on the CPU from the store
(``xpu_cpu_experts``) and added to the output. The layer then checks every
routed id of its rank against the placement: an expert of the rank is on the
card, in a pinned row or on the CPU, or the step fails.
"""

import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch

from vllm.distributed.eplb.expert_weights import StoreBackedWeight, TieredExpertWeight
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config.expert_store import ExpertStoreConfig
    from vllm.config.offload import OffloadConfig
    from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
        CpuExpertRuntime,
    )
    from vllm.model_executor.offloader.expert_tier_map import (
        PreparedMove,
        StreamOps,
        TierPlacement,
    )
    from vllm.model_executor.offloader.expert_tier_store import StoreSession

logger = init_logger(__name__)

# Decoder layers only: other MoE layers (e.g. MTP) keep their experts in VRAM, so
# the budget below is exact.
_LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")
_EXPERT_PARAMS = (
    "w13_qweight",
    "w13_scales",
    "w13_qzeros",
    "w13_bias",
    "w2_qweight",
    "w2_scales",
    "w2_qzeros",
    "w2_bias",
)
# The kernel tensors EPLB moves, in the order of the fast and the slow impl.
_TIER_FIELDS = (
    ("w13", "w13"),
    ("w2", "w2"),
    ("gemm1_wei_scales", "w13_scales"),
    ("gemm2_wei_scales", "w2_scales"),
)


def _uva_enabled() -> bool:
    """The decision `UVAOffloader` takes: host views are available and not
    disabled with VLLM_WEIGHT_OFFLOADING_DISABLE_UVA."""
    import vllm.envs as envs
    from vllm.utils.platform_utils import is_uva_available

    return is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA


def selects_expert_tiers(offload_config: "OffloadConfig") -> bool:
    """True when the offload settings put expert slots in tiers.

    Raises ValueError when they ask for it while UVA is disabled or
    unavailable: slow slots are only ever read through host views, there is no
    copy fallback as in `UVAOffloader`.
    """
    uva = offload_config.uva
    selected = (
        uva.cpu_offload_per_expert
        and current_platform.is_xpu()
        and offload_config.offload_backend in ("auto", "uva")
        and offload_config.prefetch.offload_group_size == 0
        and uva.cpu_offload_gb > 0
    )
    if selected and not _uva_enabled():
        raise ValueError(
            "--cpu-offload-per-expert reads the slow expert slots from pinned "
            "host memory through UVA, which is disabled "
            "(VLLM_WEIGHT_OFFLOADING_DISABLE_UVA) or unavailable on this "
            "platform. Drop it to offload parameters in module order."
        )
    if uva.cpu_offload_per_expert and not current_platform.is_xpu():
        logger.warning_once(
            "--cpu-offload-per-expert is supported on XPU only; offloading "
            "parameters in module order"
        )
    return bool(selected)


def _tier_config() -> "OffloadConfig | None":
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None or not selects_expert_tiers(config.offload_config):
        return None
    if not config.parallel_config.enable_eplb:
        raise ValueError(
            "--cpu-offload-per-expert places experts with EPLB: add --enable-eplb "
            "(with --enable-expert-parallel), on one rank too."
        )
    return config.offload_config


def expert_bytes(
    quant_method: Any,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
) -> int:
    """One routed expert's packed weights and scales, as
    ``AutoGPTQMoEMethod.create_weights`` lays them out (int32-packed qweight,
    ``params_dtype`` scales per group)."""
    quant_config = quant_method.quant_config
    w13_out = quant_method.moe.w13_num_shards * intermediate_size_per_partition
    pack = quant_config.pack_factor
    groups13 = groups2 = 1
    if quant_config.group_size != -1:
        groups13 = hidden_size // quant_config.group_size
        groups2 = intermediate_size_per_partition // quant_config.group_size
    qweight = (
        hidden_size // pack * w13_out
        + intermediate_size_per_partition // pack * hidden_size
    ) * torch.int32.itemsize
    scales = (groups13 * w13_out + groups2 * hidden_size) * params_dtype.itemsize
    return qweight + scales


def budget_slow_slots(
    gb: float, num_layers: int, expert_bytes: int, num_local: int
) -> int:
    """Slow slots per layer per rank that ``gb`` per rank pays for.

    At least one slot stays fast in each layer.
    """
    slow = int(gb * 2**30 / max(num_layers * expert_bytes, 1))
    return max(0, min(slow, num_local - 1))


class _PinnedArena:
    """Pinned host memory carved from 1 GiB chunks (first fit, 4 KiB aligned).

    The pinned allocator rounds each block up to a power of two, which can waste
    up to half of a layer's slow slots. Chunks of exactly 2^30 B are not
    rounded, so ``--cpu-offload-gb`` is close to the host memory it takes.

    The arena lives as long as the process: releasing a model does not free its
    chunks, and a second model loaded in the same process adds chunks to it.
    """

    CHUNK = 1 << 30
    ALIGN = 4096

    def __init__(self) -> None:
        self.chunks: list[torch.Tensor] = []
        self.used: list[int] = []
        self.total = 0

    def empty(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        for i, used in enumerate(self.used):
            if used + nbytes <= self.chunks[i].numel():
                break
        else:
            size = (
                self.CHUNK if nbytes <= self.CHUNK else 1 << (nbytes - 1).bit_length()
            )
            self.chunks.append(
                torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
            )
            self.used.append(0)
            self.total += size
            i, used = len(self.chunks) - 1, 0
        self.used[i] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        return self.chunks[i][used : used + nbytes].view(dtype).view(shape)

    @property
    def gib(self) -> float:
        return self.total / 2**30


_ARENA = _PinnedArena()

# The layers' tiers in this process once wired to their kernels, for the
# runtime control surface's readout (`live_tiers`).
_LIVE_TIERS: "weakref.WeakSet[ExpertTiers]" = weakref.WeakSet()


def live_tiers() -> list["ExpertTiers"]:
    """The tiered expert layers of this process, in layer name order."""
    return sorted(_LIVE_TIERS, key=lambda t: t.layer_name)


def pinned_arena_gib() -> float:
    """Pinned host memory the tiers' arena holds in this process."""
    return _ARENA.gib


# -- the pinned bound as a runtime setting (`tiers.set host_tier_bound`) -----


def _cpu_tiers() -> list["ExpertTiers"]:
    return [t for t in live_tiers() if t.cpu is not None and t.placement is not None]


def host_tier_bound() -> int | None:
    """Pinned slots per layer per rank now (None: no CPU tier)."""
    tiers = _cpu_tiers()
    return None if not tiers else int(tiers[0].placement.num_pinned)  # type: ignore[union-attr]


def check_host_tier_bound(value: Any) -> str | None:
    tiers = _cpu_tiers()
    if not tiers:
        return "no layer computes experts on the CPU yet"
    if not isinstance(value, int) or isinstance(value, bool):
        return f"pinned slots per layer per rank, an integer; got {value!r}"
    lo, hi = tiers[0].pinned_bound_range()
    if not lo <= value <= hi:
        return (
            f"{value} is outside {lo}..{hi}: the pinned rows were sized at load "
            f"({hi} of them can serve slots, the others are EPLB's spare rows); "
            "a larger bound needs more pinned rows, which only a restart with a "
            "larger --cpu-offload-pinned-gb or host_promote_rows allocates"
        )
    pending = [t.layer_name for t in tiers if t._eplb_pending]
    if pending:
        return f"an EPLB move is prepared on {pending[0]}; ask after it commits"
    return None


def apply_host_tier_bound(value: int) -> None:
    """Every CPU-tier layer of this process to ``value`` pinned slots; on a
    failure the layers already changed go back to the old bound."""
    tiers = _cpu_tiers()
    old = host_tier_bound()
    done: list[ExpertTiers] = []
    try:
        for t in tiers:
            t.set_pinned_bound(int(value))
            done.append(t)
    except BaseException:
        for t in done:
            t.set_pinned_bound(int(old))  # type: ignore[arg-type]
        raise
    if tiers:
        logger.info(
            "CPU experts: %d of the %d slots per layer per rank that are off "
            "the card now hold pinned rows (was %s); the other %d are computed "
            "on the CPU.",
            value,
            tiers[0].num_slow,
            old,
            tiers[0].num_slow - int(value),
        )


_RUNTIME_SETTINGS_REGISTERED = False


def _register_runtime_settings() -> None:
    """The CPU tier's settings that a running engine may change, offered to
    the runtime control surface (vllm/v1/worker/runtime_control.py)."""
    global _RUNTIME_SETTINGS_REGISTERED
    if _RUNTIME_SETTINGS_REGISTERED:
        return
    from vllm.v1.worker.runtime_control import RuntimeSetting, register_tier_setting

    register_tier_setting(
        "host_tier_bound",
        RuntimeSetting(
            get=host_tier_bound,
            check=check_host_tier_bound,
            apply=apply_host_tier_bound,
        ),
    )
    _RUNTIME_SETTINGS_REGISTERED = True


def _repack_int4_pinned(
    qweight: torch.Tensor, device: torch.device, rows: int | None = None
) -> torch.Tensor:
    """GPTQ int32 ``[E, in/8, out]`` (CPU) -> signed-int4 bytes ``[E, out, in/2]``.

    The layout transform of the method's post-load step plus ``implement_zp``,
    run on ``device`` a few experts at a time into pinned host memory, so device
    temporaries stay small. ``rows`` > E leaves that many rows in the tensor,
    the ones beyond E zeroed.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    e, k8, n = qweight.shape
    out = _ARENA.empty((max(e, rows or 0), n, k8 * 4), torch.uint8)
    out[e:].zero_()
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        packed = chunk.transpose(1, 2).contiguous().view(torch.uint8)
        out[i : i + chunk.shape[0]].copy_(implement_zp(packed))
    return out


def _pin(t: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """A pinned copy of ``t``, with ``rows`` rows if more than it has (the
    rows beyond zeroed)."""
    n = max(t.shape[0], rows or 0)
    pinned = _ARENA.empty((n, *t.shape[1:]), t.dtype)
    pinned[: t.shape[0]].copy_(t)
    pinned[t.shape[0] :].zero_()
    return pinned


def _xpu_fused_moe() -> Any:
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    return XpuFusedMoe


def _slot_maps(
    local_of: dict[int, int], num_global: int, num_fast: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical expert id -> row of the fast and of the slow tier, -1 elsewhere."""
    # explicit CPU: filled element by element here, moved to the device after
    fast_map = torch.full((num_global,), -1, dtype=torch.int32, device="cpu")
    slow_map = torch.full((num_global,), -1, dtype=torch.int32, device="cpu")
    for g, slot in local_of.items():
        if slot < 0:
            continue
        if slot < num_fast:
            fast_map[g] = slot
        else:
            slow_map[g] = slot - num_fast
    return fast_map, slow_map


@dataclass
class StoreBinding:
    """How one layer meets the expert store."""

    session: "StoreSession"
    store_layer: int  # the store's layer index: the decoder layer number
    spare_rows: int  # pool rows beyond the host tier's slots
    initial_logical: np.ndarray  # logical expert in each local slot at load
    num_logical: int  # logical experts; physical ids below it are not replicas


def _u8(t: torch.Tensor) -> torch.Tensor:
    """``t`` as bytes, one row per leading index: always a view (a tensor
    that is not contiguous raises here instead of yielding a copy that a
    row write would silently miss)."""
    return t.view(torch.uint8).view(t.shape[0], -1)


class ExpertTiers:
    """One layer's local expert slots, in device memory (fast rows) and in
    pinned host memory (pool rows).

    Without an expert store the tiers are fixed: slots ``[0, num_fast)`` are
    the rows of the layer's device parameters and ``[num_fast, num_local)`` the
    rows of the pinned tensors, in slot order. With one (``store``), a slot's
    tier and row are state of :attr:`placement` and the maps the kernels read
    are rewritten in place; see :meth:`prepare_move` and :meth:`commit_move`.
    """

    def __init__(
        self,
        layer_name: str,
        local_of: dict[int, int],
        num_global: int,
        num_local: int,
        num_fast: int,
        store: StoreBinding | None = None,
        num_pinned: int | None = None,
    ):
        assert 0 < num_fast < num_local
        self.layer_name = layer_name
        self.local_of = local_of  # physical expert id -> local slot, -1 elsewhere
        self.num_global = num_global
        self.num_local = num_local
        self.num_fast = num_fast
        # Slots off the card that hold pinned rows; the rest are host slots.
        pinned = num_local - num_fast if num_pinned is None else num_pinned
        assert 0 < pinned <= num_local - num_fast
        assert pinned == num_local - num_fast or store is not None, (
            "host slots are computed from the expert store"
        )
        self.num_pinned = pinned
        self.cpu: CpuExpertRuntime | None = None
        # Per physical id: the tier of this rank's slot (-1: not this rank's).
        self._tier_of_global = np.full(num_global, -1, dtype=np.int8)
        self._slot_of_global = np.full(num_global, -1, dtype=np.int64)
        for g, slot in local_of.items():
            if slot >= 0:
                self._slot_of_global[g] = slot
        # CPU-served layer calls per slot (decayed), for promotions.
        self._host_count = np.zeros(num_local, dtype=np.int64)
        self._steps = 0
        self._promotion: tuple[int, int, Any, int] | None = None
        # An EPLB move prepared and not yet committed or discarded: tier
        # changes of the layer's own wait for it.
        self._eplb_pending = False
        self._move_lock = threading.Lock()
        # The load of each slot's expert at the last committed move or fill
        # (EPLB's window), and the one of a move prepared and not yet
        # committed: what a change of the pinned bound ranks the slots by.
        self._slot_load: np.ndarray | None = None
        self._prepared_load: np.ndarray | None = None
        self.twins: dict[str, torch.nn.Parameter] = {}
        self._param_names: dict[int, str] = {}
        self.slow: dict[str, torch.Tensor] = {}  # pinned, kernel layout
        self.slow_views: dict[str, torch.Tensor] = {}
        self.fast_map: torch.Tensor | None = None
        self.slow_map: torch.Tensor | None = None
        self._impls: tuple[Any, Any] | None = None
        # Physical expert id of each local slot.
        self.slot_global = [-1] * num_local
        for g, slot in local_of.items():
            if slot >= 0:
                self.slot_global[slot] = g
        self.store_binding = store
        self.placement: TierPlacement | None = None
        self.ops: StreamOps | None = None
        # The logical expert in each local slot; moves keep it current.
        self.slot_logical = (
            None if store is None else np.array(store.initial_logical, dtype=np.int64)
        )
        self.move_log: list[dict[str, float]] = []
        # Slots whose row or expert changed since the last scrub.
        self._touched: set[int] = set()

    @property
    def num_slow(self) -> int:
        return self.num_local - self.num_fast

    @property
    def num_host(self) -> int:
        """Slots computed on the CPU (no row the kernels read)."""
        return self.num_slow - self.num_pinned

    @property
    def promote_rows(self) -> int:
        """Pinned rows beyond the spares for promotions of host slots between
        rearrangements."""
        if self.num_host == 0 or self.store_binding is None:
            return 0
        return int(self.store_binding.session.config.host_promote_rows)

    @property
    def num_pool_rows(self) -> int:
        """Rows of each pinned tensor: the pinned slots, the spares, and the
        rows for promotions of host slots."""
        spare = 0 if self.store_binding is None else self.store_binding.spare_rows
        return self.num_pinned + spare + self.promote_rows

    @property
    def fills_from_store(self) -> bool:
        """The host tier is filled from an existing store, not from the
        checkpoint: the loader skips its slots."""
        b = self.store_binding
        return b is not None and b.session.mode == "fill"

    def attach_twins(self, layer: torch.nn.Module) -> None:
        """A pageable CPU twin ``[num_slow, ...]`` for every per-expert param."""
        if self.fills_from_store:
            return
        for name, param in layer.named_parameters(recurse=False):
            if (
                name not in _EXPERT_PARAMS
                or param.ndim == 0
                or param.shape[0] != self.num_fast
            ):
                continue
            twin = torch.nn.Parameter(
                # explicit CPU: the model is built under a default-device context
                torch.zeros(
                    (self.num_slow, *param.shape[1:]), dtype=param.dtype, device="cpu"
                ),
                requires_grad=False,
            )
            for key, value in vars(param).items():
                if not key.startswith("_"):
                    setattr(twin, key, value)
            self.twins[name] = twin
            self._param_names[id(param)] = name

    def route(
        self, param: torch.nn.Parameter, slot: int
    ) -> tuple[torch.nn.Parameter | None, int]:
        """(param, row) the loader should write local slot ``slot`` into, or
        (None, -1) when the slot is filled from the store instead."""
        if self.store_binding is not None:
            # The model is constructed by now: refuse a store this model
            # cannot fill before any weight is loaded.
            self.store_binding.session.check_layers()
        if slot < self.num_fast:
            return param, slot
        if self.fills_from_store:
            return None, -1
        name = self._param_names.get(id(param))
        if name is None:
            raise RuntimeError(
                f"{self.layer_name}: no slow-slot twin for a param of shape "
                f"{tuple(param.shape)}"
            )
        return self.twins[name], slot - self.num_fast

    def finalize(self, layer: torch.nn.Module, moe_kernel: Any) -> None:
        """Repack the slow twins into pinned kernel-layout tensors; wire the
        kernel."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        for bias in ("w13_bias", "w2_bias"):
            if getattr(layer, bias, None) is not None:
                raise NotImplementedError(
                    f"{self.layer_name}: --cpu-offload-per-expert does not support "
                    f"{bias}"
                )
        experts = getattr(moe_kernel, "fused_experts", None)
        if experts is None or not hasattr(experts, "tiers"):
            raise NotImplementedError(
                "--cpu-offload-per-expert needs XPUExperts, got "
                f"{type(experts).__name__}"
            )
        twins, self.twins, self._param_names = self.twins, {}, {}
        device: torch.device = getattr(layer, "w13_weight").device  # noqa: B009
        binding = self.store_binding
        rows = self.num_pool_rows if binding is not None else None
        if self.fills_from_store:
            # No twins: the host tier's rows come from the store, in the shape
            # of the fast tier's tensors.
            assert rows is not None
            for key in ("w13", "w2"):
                fast_w = getattr(layer, f"{key}_weight").data.view(torch.uint8)
                self.slow[key] = _ARENA.empty((rows, *fast_w.shape[1:]), torch.uint8)
                fast_s = getattr(layer, f"{key}_scales")
                self.slow[f"{key}_scales"] = _ARENA.empty(
                    (rows, *fast_s.shape[1:]), fast_s.dtype
                )
            for tensor in self.slow.values():
                tensor[self.num_pinned :].zero_()
            self._fill_host_tier()
        else:
            for key in ("w13", "w2"):
                self.slow[key] = _repack_int4_pinned(
                    twins.pop(f"{key}_qweight").data, device, rows
                )
                scales = twins.pop(f"{key}_scales").data.transpose(1, 2).contiguous()
                self.slow[f"{key}_scales"] = _pin(
                    scales.to(getattr(layer, f"{key}_scales").dtype), rows
                )
        del twins  # qzeros: symmetric int4 has none on XPU
        for key, tensor in self.slow.items():
            view = get_accelerator_view_from_cpu_tensor(tensor)
            # Host memory seen from the device.
            view._vllm_is_uva_offloaded = True  # type: ignore[attr-defined]
            self.slow_views[key] = view
        fast_map, slow_map = _slot_maps(self.local_of, self.num_global, self.num_fast)
        self.fast_map, self.slow_map = fast_map.to(device), slow_map.to(device)
        if binding is not None:
            self._check_views_alias()
            from vllm.model_executor.offloader.expert_tier_map import (
                TierPlacement,
                TorchStreamOps,
            )

            self.ops = TorchStreamOps.shared(device)
            self.placement = TierPlacement(
                self.num_local,
                self.num_fast,
                binding.spare_rows,
                lambda event: self.ops.done(event),  # type: ignore[union-attr]
                num_pool_slots=self.num_pinned,
                extra_rows=self.promote_rows,
            )
            binding.session.note_pool(
                self.num_pool_rows, self.slot_bytes(), self.num_pinned
            )
            if self.num_host:
                # Host slots are in neither map (the slot-order maps above
                # would point them at spare rows).
                fast_np, slow_np = self.placement.maps(
                    self.slot_global, self.num_global
                )
                self.fast_map = torch.from_numpy(fast_np).to(device)
                self.slow_map = torch.from_numpy(slow_np).to(device)
            self._refresh_tiers()
            if self.num_host:
                self._start_cpu_tier(experts)
        experts.tiers = self
        _LIVE_TIERS.add(self)

    # -- the CPU tier ---------------------------------------------------------

    def _start_cpu_tier(self, experts: Any) -> None:
        """Host slots exist: take the process's CPU tier and read their
        experts into its host cache ahead of the first step (as far as free
        rows go)."""
        from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
            UNIT_TENSORS,
            UnitLayout,
            get_runtime,
        )

        if getattr(experts, "impl_apply_kwargs", {}).get("w4a8"):
            raise NotImplementedError(
                f"{self.layer_name}: CPU experts compute W4A16; this layer's "
                "experts run W4A8 (int8 activations) on the card"
            )
        binding = self.store_binding
        assert binding is not None
        if self.slow["w13_scales"].dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{self.layer_name}: CPU experts compute with bfloat16 "
                f"activations and scales, the layer has {self.slow['w13_scales'].dtype}"
            )
        w13 = self.slow["w13"]
        hidden, inter = int(w13.shape[2]) * 2, int(w13.shape[1]) // 2
        group = hidden // int(self.slow["w13_scales"].shape[2])
        layout = UnitLayout(
            hidden,
            inter,
            group,
            {name: _u8(self.slow[name])[0].numel() for name in UNIT_TENSORS},
        )
        self.cpu = get_runtime(binding.session.config, binding.session.store, layout)
        _register_runtime_settings()
        self._prefetch_host(evict=False)
        logger.info_once(
            "CPU experts: %d of the %d slots per layer per rank that are off "
            "the card hold pinned rows; the other %d are computed on the CPU "
            "from the expert store.",
            self.num_pinned,
            self.num_slow,
            self.num_host,
        )

    def _refresh_tiers(self) -> None:
        """The tier of every physical id of this rank, from the placement:
        what the kernels' maps and the CPU tier serve until the next
        commit."""
        if self.placement is None:
            return
        for s, g in enumerate(self.slot_global):
            self._tier_of_global[g] = self.placement.tier[s]

    def _host_keys(self, slots: Any = None) -> list[tuple[int, int]]:
        assert self.store_binding is not None and self.slot_logical is not None
        if slots is None:
            assert self.placement is not None
            slots = np.flatnonzero(self.placement.is_host())
        layer = self.store_binding.store_layer
        return [(layer, int(self.slot_logical[s])) for s in slots]

    def _prefetch_host(self, slots: Any = None, evict: bool = True) -> None:
        if self.cpu is not None:
            self.cpu.cache.prefetch(self._host_keys(slots), evict=evict)

    def host_calls(
        self, ids: np.ndarray, weights: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, dict[str, int]]:
        """The calls of a step's routing ``ids`` [T, K] (physical ids, the
        unrouted already mapped to a rank's sentinel) that this rank computes
        on the CPU: (store key, token, weight) in token order, top-k order
        inside a token; and the rank's routed calls per tier. Raises when a
        routed id is outside the physical ids: nothing the kernels would skip
        silently passes here."""
        keys, tok, k, counts = self._host_selection(ids)
        return (
            keys,
            tok.astype(np.int32),
            np.ascontiguousarray(weights[tok, k], dtype=np.float32),
            counts,
        )

    def host_pairs(
        self, ids: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray, dict[str, int]]:
        """As :meth:`host_calls`, for a sum that takes each (token, top-k)
        pair's own row (VLLM_XPU_MOE_COMBINE rank/exact): the store keys and
        the flat pair index ``token * top_k + k`` of every call."""
        keys, tok, k, counts = self._host_selection(ids)
        pair = tok.astype(np.int64) * ids.shape[1] + k
        return keys, pair, counts

    def _host_selection(
        self, ids: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, dict[str, int]]:
        from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
            CpuExpertError,
        )
        from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.num_global):
            raise CpuExpertError(
                f"{self.layer_name}: a routed expert id is outside the "
                f"{self.num_global} physical experts (min {int(ids.min())}, max "
                f"{int(ids.max())})"
            )
        tier = self._tier_of_global[ids]
        counts = {
            "card": int((tier == FAST).sum()),
            "pinned": int((tier == POOL).sum()),
            "cpu": int((tier == HOST).sum()),
        }
        tok, k = np.nonzero(tier == HOST)
        if tok.size == 0:
            return [], tok, k, counts
        assert self.slot_logical is not None and self.store_binding is not None
        slots = self._slot_of_global[ids[tok, k]]
        np.add.at(self._host_count, np.unique(slots), 1)
        layer = self.store_binding.store_layer
        keys = [(layer, int(e)) for e in self.slot_logical[slots]]
        return keys, tok, k, counts

    def between_steps(self) -> None:
        """Between two steps, on the thread that enqueues the forward: the
        promotion of a host slot the CPU served often since the last
        rearrangement into a free pinned row (read from the host cache or
        the store in the background, mapped at a later step). EPLB's next
        rearrangement chooses the pinned set again from its load window."""
        cpu = self.cpu
        if cpu is None:
            return
        cfg = cpu.config
        threshold = int(cfg.host_promote_misses)
        if threshold <= 0:
            return
        self._steps += 1
        if self._steps % int(cfg.host_promote_window) == 0:
            self._host_count //= 2
        if self._promotion is not None:
            self._finish_promotion()
            return
        placement = self.placement
        if self._eplb_pending or placement is None:
            return
        hot = np.flatnonzero(placement.is_host() & (self._host_count >= threshold))
        # The spare rows stay free for EPLB's next move, and promotions stay
        # within their own rows above the pinned bound (which the runtime
        # control surface may have lowered: see set_pinned_bound).
        if (
            hot.size == 0
            or placement.pool.free_count <= placement.spare
            or int(placement.is_pool().sum())
            >= placement.num_pinned + self.promote_rows
        ):
            return
        if not self._move_lock.acquire(blocking=False):
            return
        try:
            slot = int(hot[np.argmax(self._host_count[hot])])
            row = placement.pool.acquire(1)[0]
            assert self.slot_logical is not None and self.store_binding is not None
            logical = int(self.slot_logical[slot])
            key = (self.store_binding.store_layer, logical)
            dests = {k: _u8(t)[row] for k, t in self.slow.items()}
            cached = cpu.cache.row_of(key)
            from concurrent.futures import Future

            if cached is not None:
                future: Any = Future()
                src = cpu.cache.dests(cached)
                for name, dst in dests.items():
                    dst.copy_(src[name])
                future.set_result("host cache")
            else:
                future = self.store_binding.session.store.read_into_async(key, dests)
            self._promotion = (slot, row, future, logical)
        finally:
            self._move_lock.release()

    def _finish_promotion(self) -> None:
        from vllm.model_executor.offloader.expert_tier_map import HOST

        assert self._promotion is not None and self.placement is not None
        slot, row, future, logical = self._promotion
        if not future.done():
            return
        self._promotion = None
        placement = self.placement
        if not self._move_lock.acquire(blocking=False):
            placement.pool.give_back_unused([row])
            return
        try:
            assert self.slot_logical is not None
            if (
                future.exception() is not None
                or self._eplb_pending
                or int(self.slot_logical[slot]) != logical
                or placement.tier[slot] != HOST
            ):
                placement.pool.give_back_unused([row])
                return
            # The row is complete (the host wrote it) and no map points at it:
            # map it (R2); the slot leaves the CPU tier at the same point.
            placement.promote_host(slot, row)
            ops = self.ops
            assert ops is not None and self.slow_map is not None
            _, slow_np = placement.maps(self.slot_global, self.num_global)
            ops.compute_write(self.slow_map, slow_np)
            self._refresh_tiers()
            self._host_count[slot] = 0
            self._touched.add(slot)
            self.on_rows_written([])
            assert self.cpu is not None
            self.cpu.stats.promotions += 1
            logger.info_once(
                "CPU experts: a host slot the CPU served often was promoted to "
                "a pinned row between two steps (%s, slot %d, expert %d).",
                self.layer_name,
                slot,
                logical,
            )
        finally:
            self._move_lock.release()

    # -- the pinned bound while serving (runtime control: host_tier_bound) ---

    def pinned_bound_range(self) -> tuple[int, int]:
        """The pinned slots per layer a running engine may move to: at least
        one, at most every pool row but EPLB's spares (the rows were sized at
        load: the load-time bound plus ``host_promote_rows``)."""
        assert self.placement is not None
        return 1, self.placement.max_pinned_capacity()

    def _drop_promotion(self) -> None:
        """Forget a promotion being read: its row goes back once the read
        that may still write it is done."""
        if self._promotion is None:
            return
        assert self.placement is not None
        _, row, future, _ = self._promotion
        self._promotion = None
        if future.done():
            self.placement.pool.give_back_unused([row])
        else:
            self.placement.pool.release([row], future)

    def _fill_rows_cached(
        self, reads: Mapping[int, int], rows: Mapping[int, int]
    ) -> None:
        """:meth:`_fill_rows`, taking the units the CPU tier's host cache
        holds from there instead of the drive."""
        assert self.store_binding is not None
        rest: dict[int, int] = {}
        for slot, logical in reads.items():
            key = (self.store_binding.store_layer, int(logical))
            cached = None if self.cpu is None else self.cpu.cache.row_of(key)
            if cached is None:
                rest[slot] = logical
                continue
            src = self.cpu.cache.dests(cached)  # type: ignore[union-attr]
            for name, dst in ((k, _u8(t)[rows[slot]]) for k, t in self.slow.items()):
                dst.copy_(src[name])
        if rest:
            self._fill_rows(rest, {s: rows[s] for s in rest})

    def set_pinned_bound(self, bound: int) -> dict[str, float]:
        """Between two steps, on the thread that enqueues the forward: serve
        ``bound`` of this layer's off-card slots from pinned rows and the
        others on the CPU. Lowering it moves the coldest pinned slots to the
        CPU (no read; their rows go back to the free rows, which stay
        allocated); raising it reads the hottest host slots into free rows
        (from the host cache where it holds them) before they are mapped.
        Hottest is by the load of the last rearrangement (or fill), then by
        how often the CPU served the slot since. The next rearrangement
        chooses the pinned set with this bound. Raises, with the placement
        unchanged, while an EPLB move is prepared or when the reads fail."""
        from vllm.model_executor.offloader.expert_tier_map import HOST, POOL

        binding, placement, _ = self._require_store()
        if self.cpu is None:
            raise RuntimeError(
                f"{self.layer_name}: the pinned bound moves only in an engine "
                "with experts computed on the CPU (--cpu-offload-pinned-gb)"
            )
        lo, hi = self.pinned_bound_range()
        if not lo <= bound <= hi:
            raise ValueError(
                f"{self.layer_name}: pinned slots per layer must be {lo}..{hi}, "
                f"got {bound}"
            )
        assert self.slot_logical is not None
        start = time.perf_counter()
        with self._move_lock:
            if self._eplb_pending:
                raise RuntimeError(
                    f"{self.layer_name}: an EPLB move is prepared; the bound "
                    "can change once it is committed"
                )
            self._drop_promotion()
            load = (
                np.zeros(self.num_local) if self._slot_load is None else self._slot_load
            )
            # ascending heat: by load, then by the CPU's count
            heat = np.lexsort((self._host_count, load))
            rank = np.empty(self.num_local, dtype=np.int64)
            rank[heat] = np.arange(self.num_local)
            pool = placement.is_pool().copy()
            in_pool = np.flatnonzero(pool)
            if in_pool.size > bound:  # the coldest pinned slots go to the CPU
                cold = in_pool[np.argsort(rank[in_pool], kind="stable")]
                pool[cold[: in_pool.size - bound]] = False
            elif in_pool.size < bound:  # the hottest host slots get rows
                host = np.flatnonzero(placement.tier == HOST)
                hot = host[np.argsort(-rank[host], kind="stable")]
                pool[hot[: bound - in_pool.size]] = True
            old = placement.num_pinned
            plan = placement.plan_from_set(
                self.slot_logical,
                self.slot_logical,
                placement.is_fast(),
                pool,
                check_budget=False,
            )
            move = placement.prepare(
                plan,
                self._fill_rows_cached,
                timeout=binding.session.config.read_timeout_s,
            )
            placement.set_pinned_capacity(bound)
            try:
                self.commit_move(move)
            except BaseException:
                placement.set_pinned_capacity(old)
                raise
        stats = {
            "bound": float(bound),
            "pinned": float(int((placement.tier == POOL).sum())),
            "cpu": float(int((placement.tier == HOST).sum())),
            "units_read": float(len(plan.reads)),
            "ms": (time.perf_counter() - start) * 1e3,
        }
        move.stats.update(stats)
        return stats

    def _check_views_alias(self) -> None:
        """With a store, host threads write the pool rows into the pinned
        tensors and the kernels read them through the device views: the two
        must be the same memory. A view made over a copy (memory the
        accelerator does not take as pinned is copied first) would serve the
        load-time bytes for ever and none of the store's reads. One byte of
        a spare row per tensor settles it at load."""
        last = self.num_pool_rows - 1
        if last < self.num_pinned:
            return  # no spare row to probe
        for key, tensor in self.slow.items():
            host = _u8(tensor)[last]
            saved = int(host[0])
            host[0] = saved ^ 0xA5
            seen = int(_u8(self.slow_views[key])[last][0])
            host[0] = saved
            if seen != saved ^ 0xA5:
                raise RuntimeError(
                    f"{self.layer_name}: the device view of the host tier "
                    f"({key}) is a copy, not the pinned rows themselves: "
                    "experts read from the expert store would never reach the "
                    "kernels. The host tier must be memory the accelerator "
                    "reads in place"
                )
        logger.info_once(
            "Expert store tiers: the device views of the pool rows are the "
            "pinned rows themselves (a byte per tensor written on the host "
            "and read back through the view)."
        )

    def slot_bytes(self) -> int:
        """Bytes of one expert in kernel layout (the four tensors)."""
        return sum(t[0].numel() * t.element_size() for t in self.slow.values())

    # -- the expert store ---------------------------------------------------

    def _fill_host_tier(self) -> None:
        """Read every host-tier slot of the layer from the store into its row."""
        binding = self.store_binding
        assert binding is not None
        store = binding.session.open_reader()
        futures = []
        # The pinned slots only: host slots are read by the CPU tier.
        for slot in range(self.num_fast, self.num_fast + self.num_pinned):
            row = slot - self.num_fast
            dests = {k: _u8(t)[row] for k, t in self.slow.items()}
            logical = int(binding.initial_logical[slot])
            futures.append(store.read_into_async((binding.store_layer, logical), dests))
        for future in futures:
            future.result()

    def _write_units(self) -> None:
        """Build mode: write the layer's experts (fast rows read back from the
        card once, host rows from memory) as units keyed by logical expert;
        replicas are skipped."""
        binding = self.store_binding
        assert binding is not None and self._impls is not None
        fast = self._impls[0]
        host_fast = {
            key: _u8(getattr(fast, impl_field)).cpu()
            for impl_field, key in _TIER_FIELDS
        }
        units = []
        for slot in range(self.num_local):
            if self.slot_global[slot] >= binding.num_logical:
                continue  # a replica: its logical expert is written by its owner
            if slot < self.num_fast:
                tensors = {k: t[slot] for k, t in host_fast.items()}
            else:
                row = slot - self.num_fast
                tensors = {k: _u8(t)[row] for k, t in self.slow.items()}
            units.append((int(binding.initial_logical[slot]), tensors))
        binding.session.write_units(binding.store_layer, units)

    def _after_impls(self) -> None:
        binding = self.store_binding
        if binding is None:
            return
        if binding.session.mode == "build":
            self._write_units()
        else:
            self._check_store_against_this_boot()
        binding.session.layer_done(self.layer_name)

    def _check_store_against_this_boot(self) -> None:
        """Fill mode: the fast rows were just laid out by this tree and these
        kernels from the checkpoint. A few of them must equal the store's
        units for the same experts, or the store was built from another
        checkpoint or under another kernel layout, and the host rows it
        filled would not match the fast rows."""
        from vllm.model_executor.offloader.expert_store import StoreMismatchError

        binding = self.store_binding
        assert binding is not None and self._impls is not None
        store = binding.session.store
        fast = self._impls[0]
        for slot in sorted({0, self.num_fast // 2, self.num_fast - 1}):
            if self.slot_global[slot] >= binding.num_logical:
                continue  # a replica
            logical = int(binding.initial_logical[slot])
            with store.get((binding.store_layer, logical)) as entry:
                unit = entry.tensors()
                for impl_field, key in _TIER_FIELDS:
                    row = _u8(getattr(fast, impl_field))[slot].cpu()
                    want = unit[key].reshape(-1).view(torch.uint8)
                    if not torch.equal(row, want):
                        raise StoreMismatchError(
                            f"{self.layer_name}: the expert store's unit of "
                            f"expert {logical} ({key}) differs from the row "
                            "this start built from the checkpoint: the store "
                            "was built from another checkpoint or by kernels "
                            "with another weight layout. Start once with "
                            "--expert-store-config build set to 'rebuild'"
                        )
        logger.info_once(
            "Expert store: the store's units equal the fast rows this start "
            "built from the checkpoint (three experts per layer compared)."
        )

    def impls(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk: int,
        activation: str,
        gemm1_clamp_limit: float | None,
    ) -> tuple[Any, Any]:
        """(fast, slow) ``XpuFusedMoe``, built once."""
        if self._impls is None:
            XpuFusedMoe = _xpu_fused_moe()
            # ep_size > 1 only makes XpuFusedMoe index through the expert_map
            # given here.
            common: dict[str, Any] = {
                "n_experts_per_token": topk,
                "activation": activation,
                "ep_rank": 0,
                "ep_size": 2,
                "gemm1_clamp_limit": gemm1_clamp_limit,
            }
            fast = XpuFusedMoe(
                w13=w1,
                w13_scales=w1_scale,
                w13_bias=None,
                w2=w2,
                w2_scales=w2_scale,
                w2_bias=None,
                num_experts=self.num_fast,
                expert_map=self.fast_map,
                **common,
            )
            sw13, sw2 = self.slow_views["w13"], self.slow_views["w2"]
            # already signed int4, in kernel layout
            sw13.xpu_fused_moe = True  # type: ignore[attr-defined]
            slow = XpuFusedMoe(
                w13=sw13,
                w13_scales=self.slow_views["w13_scales"],
                w13_bias=None,
                w2=sw2,
                w2_scales=self.slow_views["w2_scales"],
                w2_bias=None,
                num_experts=self.num_pool_rows,
                expert_map=self.slow_map,
                **common,
            )
            # Sizes come from the uint8 view; the kernel reads packed int4 as int8.
            slow.w13, slow.w2 = sw13.view(torch.int8), sw2.view(torch.int8)
            slow.w13._vllm_is_uva_offloaded = True
            slow.w2._vllm_is_uva_offloaded = True
            self._impls = (fast, slow)
            self._after_impls()
        return self._impls

    def eplb_weights(self) -> list[TieredExpertWeight]:
        """The tensors the two impls read, as EPLB's per-slot rows."""
        assert self._impls is not None, "the impls are built at weight processing"
        fast, slow = self._impls
        weights: list[TieredExpertWeight] = []
        for impl_field, _ in _TIER_FIELDS:
            parts = [getattr(impl, impl_field) for impl in (fast, slow)]
            # The spare rows beyond the pinned slots are not EPLB's; the host
            # slots have no row (they count as slots without a tensor).
            parts[1] = parts[1][: self.num_pinned]
            parts = [p.view(p.shape[0], -1) for p in parts]
            if parts[0].dtype != parts[1].dtype:  # packed int4: int8 vs uint8
                parts = [p.view(torch.uint8) for p in parts]
            if self.store_binding is not None:
                # placed by this object, through the expert store
                weights.append(
                    StoreBackedWeight(
                        parts, self.on_rows_written, self, unbacked=self.num_host
                    )
                )
            else:
                weights.append(TieredExpertWeight(parts, self.on_rows_written))
        return weights

    # -- moving slots between tiers while serving --------------------------

    def _require_store(self) -> "tuple[StoreBinding, TierPlacement, StreamOps]":
        if (
            self.store_binding is None
            or self.placement is None
            or self.ops is None
            or self._impls is None
        ):
            raise RuntimeError(
                f"{self.layer_name}: tier moves need an expert store and built impls"
            )
        return self.store_binding, self.placement, self.ops

    def _fill_rows(self, reads: Mapping[int, int], rows: Mapping[int, int]) -> None:
        """Read the units of ``reads`` (slot -> logical expert) into the pool
        rows ``rows`` (slot -> row), all in flight at once."""
        from concurrent.futures import TimeoutError as ReadTimeout

        from vllm.model_executor.offloader.expert_store import ExpertStoreError

        binding, _, _ = self._require_store()
        store = binding.session.store
        timeout = binding.session.config.read_timeout_s
        pending: list[tuple[int, tuple[int, int], Any]] = []
        try:
            for slot, logical in reads.items():
                row = rows[slot]
                key = (binding.store_layer, int(logical))
                dests = {k: _u8(t)[row] for k, t in self.slow.items()}
                pending.append((row, key, store.read_into_async(key, dests)))
            deadline = time.monotonic() + timeout
            for _, _, future in pending:
                future.result(max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            # Reads not started are cancelled; the ones under way may still
            # write their rows, which the placement then holds until they
            # are done.
            in_flight = {
                row: future
                for row, _, future in pending
                if not future.cancel() and not future.done()
            }
            if isinstance(exc, ReadTimeout):
                stuck = [key for _, key, future in pending if not future.done()]
                error = ExpertStoreError(
                    f"{self.layer_name}: {len(stuck)} of {len(pending)} unit "
                    f"reads did not return within {timeout:g} s (first: unit "
                    f"{stuck[0] if stuck else '?'}): the drive or the store's "
                    "reader threads are stuck"
                )
                error.rows_in_flight = in_flight  # type: ignore[attr-defined]
                raise error from None
            exc.rows_in_flight = in_flight  # type: ignore[attr-defined]
            raise

    def _copy_rows(self, slot: int, pool_row: int, fast_row: int) -> None:
        """Enqueue the copy of one expert's bytes from a pool row into a fast
        row on the side stream."""
        _, _, ops = self._require_store()
        assert self._impls is not None
        fast = self._impls[0]
        for impl_field, key in _TIER_FIELDS:
            dst = _u8(getattr(fast, impl_field))[fast_row]
            ops.side_copy(dst, _u8(self.slow[key])[pool_row])

    def prepare_move(
        self,
        new_logical: np.ndarray,
        slot_load: np.ndarray,
        hysteresis: float,
        *,
        timeout: float = 5.0,
    ) -> "PreparedMove":
        """Plan the move to the experts ``new_logical`` (per local slot) with
        the tier choice for ``slot_load`` (the load of the expert each slot
        will hold), and read what it needs from the store into free pool rows.
        Runs on any thread; touches no device state. Commit the result with
        :meth:`commit_move` between two steps, on the thread that enqueues the
        forward."""
        _, placement, _ = self._require_store()
        assert self.slot_logical is not None
        t0 = time.perf_counter()
        plan = placement.plan(self.slot_logical, new_logical, slot_load, hysteresis)
        move = placement.prepare(plan, self._fill_rows, timeout=timeout)
        self._prepared_load = np.asarray(slot_load, dtype=np.float64).copy()
        move.stats = {
            "units": float(len(plan.reads)),
            "bytes": float(len(plan.reads) * self.slot_bytes()),
            "prepare_ms": (time.perf_counter() - t0) * 1e3,
        }
        return move

    def commit_move(self, move: "PreparedMove") -> None:
        """Make a prepared move the placement: enqueue its copies and the map
        writes (see :meth:`TierPlacement.commit` for the stream order)."""
        from vllm.model_executor.offloader.expert_tier_map import HOST

        _, placement, ops = self._require_store()
        t0 = time.perf_counter()
        was_host = placement.is_host().copy()
        placement.commit(
            move,
            ops,
            slot_global=self.slot_global,
            num_global=self.num_global,
            fast_map=self.fast_map,
            slow_map=self.slow_map,
            copy_rows=self._copy_rows,
            on_pool_rows_changed=lambda: self.on_rows_written([]),
        )
        self.slot_logical = move.plan.new_logical.copy()
        if self._prepared_load is not None:
            self._slot_load, self._prepared_load = self._prepared_load, None
        plan = move.plan
        self._touched.update(plan.changed, plan.demoted, plan.to_fast)
        self._refresh_tiers()
        if self.cpu is not None:
            is_host = placement.tier == HOST
            self.cpu.stats.promotions += int((was_host & ~is_host).sum())
            self.cpu.stats.demotions += int((~was_host & is_host).sum())
            # New host slots and host slots with a new expert: read ahead.
            changed = np.zeros(self.num_local, dtype=bool)
            changed[plan.changed] = True
            self._prefetch_host(np.flatnonzero(is_host & (~was_host | changed)))
        move.stats["commit_ms"] = (time.perf_counter() - t0) * 1e3
        self.move_log.append(move.stats)

    def _set_fast(self, new_fast: np.ndarray) -> "PreparedMove":
        _, placement, _ = self._require_store()
        assert self.slot_logical is not None
        plan = placement.plan_from_set(self.slot_logical, self.slot_logical, new_fast)
        move = placement.prepare(plan, self._fill_rows)
        move.stats = {"units": float(len(plan.reads))}
        self.commit_move(move)
        return move

    # -- EPLB's ExpertMover ---------------------------------------------------

    def _local_view(
        self, phys2log: np.ndarray, logical_load: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(logical expert, load) of each local slot under the placement
        ``phys2log`` of the layer's physical slots."""
        assert self.store_binding is not None
        phys2log = np.asarray(phys2log)
        slots = np.array(self.slot_global)
        logical = phys2log[slots].astype(np.int64)
        if (logical < 0).any():
            raise NotImplementedError(
                "empty expert slots (elastic EP) are not supported"
            )
        counts = np.bincount(
            phys2log[phys2log >= 0], minlength=self.store_binding.num_logical
        )
        load = np.asarray(logical_load, dtype=np.float64)
        return logical, load[logical] / np.maximum(counts[logical], 1)

    def prepare(
        self,
        old_phys2log: np.ndarray,
        new_phys2log: np.ndarray,
        logical_load: np.ndarray,
        hysteresis: float,
    ) -> "PreparedMove":
        old_logical, _ = self._local_view(old_phys2log, logical_load)
        new_logical, slot_load = self._local_view(new_phys2log, logical_load)
        assert self.slot_logical is not None
        if not np.array_equal(old_logical, self.slot_logical):
            raise RuntimeError(
                f"{self.layer_name}: EPLB's placement and the tiers' disagree "
                f"({old_logical.tolist()} vs {self.slot_logical.tolist()})"
            )
        # From here until the commit or the discard the placement is EPLB's:
        # the layer's own promotions wait.
        with self._move_lock:
            self._eplb_pending = True
            try:
                return self.prepare_move(new_logical, slot_load, hysteresis)
            except BaseException:
                self._eplb_pending = False
                raise

    def commit(self, prepared: "PreparedMove") -> None:
        try:
            self.commit_move(prepared)
        finally:
            self._eplb_pending = False

    def discard(self, prepared: "PreparedMove") -> None:
        """A prepared move that will not be committed: its pool rows were only
        written, never mapped, so they are free at once."""
        self._eplb_pending = False
        self._prepared_load = None
        _, placement, _ = self._require_store()
        placement.pool.give_back_unused(prepared.pool_dest.values())

    @staticmethod
    def prepared_stats(prepared: "PreparedMove") -> dict[str, float]:
        return {
            k: prepared.stats[k]
            for k in ("units", "bytes", "prepare_ms")
            if k in prepared.stats
        }

    def fill_to(
        self, phys2log: np.ndarray, logical_load: np.ndarray, hysteresis: float
    ) -> None:
        """Before serving: take the placement ``phys2log`` and read every slot
        from the store into the row it serves from. The hottest slots get the
        fast rows (copied through the spare pool rows), the rest get pool rows
        in slot order. Nothing reads the rows yet, so no ordering is needed."""
        import time as _time

        from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

        binding, placement, ops = self._require_store()
        # The pinned capacity is the load-time one unless the runtime control
        # surface moved it (set_pinned_bound).
        pinned = placement.num_pinned
        spare_rows = list(range(pinned, self.num_pool_rows))
        assert spare_rows, "filling the fast rows stages them through spare rows"
        logical, load = self._local_view(phys2log, logical_load)
        order = np.argsort(-load, kind="stable")
        is_fast = np.zeros(self.num_local, dtype=bool)
        is_fast[order[: self.num_fast]] = True
        # The next hottest hold the pinned rows; with host slots the coldest
        # are computed on the CPU.
        is_pool = np.zeros(self.num_local, dtype=bool)
        is_pool[order[self.num_fast : self.num_fast + pinned]] = True
        self._slot_load = np.asarray(load, dtype=np.float64).copy()
        if (
            np.array_equal(logical, self.slot_logical)
            and np.array_equal(is_fast, placement.is_fast())
            and np.array_equal(is_pool, placement.is_pool())
        ):
            return  # the rows already hold this placement
        tier = np.full(self.num_local, HOST, dtype=np.int8)
        tier[is_fast] = FAST
        tier[is_pool] = POOL
        row = np.full(self.num_local, -1, dtype=np.int32)
        row[is_fast] = np.arange(self.num_fast)
        row[is_pool] = np.arange(int(is_pool.sum()))
        start = _time.perf_counter()
        pool_slots = np.flatnonzero(is_pool).tolist()
        self._fill_rows(
            {s: int(logical[s]) for s in pool_slots},
            {s: int(row[s]) for s in pool_slots},
        )
        fast_slots = np.flatnonzero(is_fast).tolist()
        for i in range(0, len(fast_slots), len(spare_rows)):
            chunk = fast_slots[i : i + len(spare_rows)]
            staging = {s: spare_rows[j] for j, s in enumerate(chunk)}
            self._fill_rows({s: int(logical[s]) for s in chunk}, staging)
            for s in chunk:
                self._copy_rows(s, staging[s], int(row[s]))
            done = ops.record_side()
            while not ops.done(done):
                _time.sleep(0.0005)
        placement.reset_to(tier, row)
        fast_np, slow_np = placement.maps(self.slot_global, self.num_global)
        assert self.fast_map is not None and self.slow_map is not None
        ops.compute_write(self.fast_map, fast_np)
        ops.compute_write(self.slow_map, slow_np)
        self.slot_logical = logical.copy()
        self._touched.update(range(self.num_local))
        self._refresh_tiers()
        self._prefetch_host(evict=False)
        self.on_rows_written([])
        self.move_log.append(
            {
                "fill_ms": (_time.perf_counter() - start) * 1e3,
                "units": float(len(logical)),
            }
        )

    def nonresident_physical(self) -> list[int]:
        """Physical expert ids this rank owns that no tier holds (the EPLB
        residency mask keeps the router away from them). Empty while every
        owned slot is in a fast row, a pool row or on the CPU, which is the
        invariant of the tiers: a host slot is served, not missing."""
        from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

        placement = self.placement
        if placement is None:
            return []
        held = np.isin(placement.tier, (FAST, POOL, HOST))
        return [g for g, slot in self.local_of.items() if slot >= 0 and not held[slot]]

    @property
    def scrub_every(self) -> int:
        """The debug setting: scrub after every Nth rearrangement (0: off)."""
        binding = self.store_binding
        return 0 if binding is None else int(binding.session.config.scrub_every)

    def scrub(self, phys2log: np.ndarray | None = None, full: bool = True) -> dict:
        """Compare what the kernels of this layer read with the expert store.

        For each slot checked, the bytes of the row that serves it, read the
        way the kernels read them (a fast row back from the device, a pool
        row through the device view, not from the pinned tensor), must be the
        store's unit of the expert the slot holds. The two maps on the device
        must equal the placement, and with ``phys2log`` (EPLB's map of the
        layer) the experts must be the ones EPLB routes to these slots.
        ``full`` checks every slot, otherwise the slots moves touched since
        the last scrub.

        Blocking (device reads). Safe on a serving engine between two steps,
        on the thread that enqueues the forward: it reads mapped rows only,
        which nothing writes, and waits for the commits queued before it.
        Returns ``slots``, ``mismatches``, ``maps_equal``, ``placement_equal``
        and ``first`` (the first problem as text, or None).
        """
        from vllm.model_executor.offloader.expert_tier_map import FAST, POOL

        binding, placement, _ = self._require_store()
        assert self._impls is not None and self.slot_logical is not None
        assert self.fast_map is not None and self.slow_map is not None
        store = binding.session.store
        fast = self._impls[0]
        problems: list[str] = []
        want_fast, want_slow = placement.maps(self.slot_global, self.num_global)
        maps_equal = bool(
            np.array_equal(self.fast_map.cpu().numpy(), want_fast)
            and np.array_equal(self.slow_map.cpu().numpy(), want_slow)
        )
        if not maps_equal:
            problems.append(
                f"{self.layer_name}: the tier maps on the device differ from "
                "the placement"
            )
        placement_equal = True
        if phys2log is not None:
            routed = np.asarray(phys2log)[np.array(self.slot_global)]
            placement_equal = bool(np.array_equal(routed, self.slot_logical))
            if not placement_equal:
                slot = int(np.flatnonzero(routed != self.slot_logical)[0])
                problems.append(
                    f"{self.layer_name}: slot {slot} holds expert "
                    f"{int(self.slot_logical[slot])}, EPLB routes expert "
                    f"{int(routed[slot])} to it"
                )
        slots = range(self.num_local) if full else sorted(self._touched)
        whole: dict[tuple[int, str], torch.Tensor] = {}
        if full:  # one device read per tensor instead of one per row
            for impl_field, key in _TIER_FIELDS:
                whole[(FAST, key)] = _u8(getattr(fast, impl_field)).cpu()
                whole[(POOL, key)] = _u8(self.slow_views[key]).cpu()
        from vllm.model_executor.offloader.expert_tier_map import HOST

        mismatches = 0
        for slot in slots:
            tier, row = int(placement.tier[slot]), int(placement.row[slot])
            tensors = {}
            if tier == HOST:
                # Computed on the CPU from its host cache: check the cached
                # unit if it is there (the drive's copy is the store itself).
                if self.cpu is None:
                    continue
                key = (binding.store_layer, int(self.slot_logical[slot]))
                cached = self.cpu.cache.row_of(key)
                if cached is None:
                    continue
                if not store.holds_unit(key, self.cpu.cache.dests(cached)):
                    mismatches += 1
                    problems.append(
                        f"{self.layer_name}: slot {slot} (expert {key[1]}, "
                        "host cache) does not hold the store's unit"
                    )
                continue
            for impl_field, key in _TIER_FIELDS:
                if full:
                    tensors[key] = whole[(tier, key)][row]
                elif tier == FAST:
                    tensors[key] = _u8(getattr(fast, impl_field))[row].cpu()
                else:
                    tensors[key] = _u8(self.slow_views[key])[row].cpu()
            logical = int(self.slot_logical[slot])
            if not store.holds_unit((binding.store_layer, logical), tensors):
                mismatches += 1
                problems.append(
                    f"{self.layer_name}: slot {slot} (expert {logical}, "
                    f"{'fast' if tier == FAST else 'pool'} row {row}) does not "
                    "hold the store's unit"
                )
        self._touched.clear()
        return {
            "slots": len(slots),
            "mismatches": mismatches,
            "maps_equal": maps_equal,
            "placement_equal": placement_equal,
            "first": problems[0] if problems else None,
        }

    def swap(self, promote: int, demote: int) -> "PreparedMove":
        """Put the pool slot ``promote`` in a fast row and the fast slot
        ``demote`` in a pool row, in one commit (design 5.3): ``demote``'s
        bytes are read from the store into a spare pool row, ``promote``'s are
        copied from its pool row into the fast row ``demote`` leaves."""
        _, placement, _ = self._require_store()
        new_fast = placement.is_fast().copy()
        if not new_fast[demote] or new_fast[promote]:
            raise ValueError(
                "swap needs a fast slot to demote and a pool slot to promote"
            )
        new_fast[demote], new_fast[promote] = False, True
        return self._set_fast(new_fast)

    def promote(self, slot: int) -> "PreparedMove":
        """Put the pool slot ``slot`` in a free fast row (one a demotion left)."""
        _, placement, _ = self._require_store()
        if placement.tier[slot] == 0:
            raise ValueError(f"slot {slot} is already in a fast row")
        if not placement.fast_free:
            raise RuntimeError(
                "no free fast row: demote a slot first, or swap two slots"
            )
        new_fast = placement.is_fast().copy()
        new_fast[slot] = True
        return self._set_fast(new_fast)

    def demote(self, slot: int) -> "PreparedMove":
        """Put the fast slot ``slot`` in a pool row, its bytes read from the
        store; the fast row it leaves is free."""
        _, placement, _ = self._require_store()
        if placement.tier[slot] != 0:
            raise ValueError(f"slot {slot} is not in a fast row")
        new_fast = placement.is_fast().copy()
        new_fast[slot] = False
        return self._set_fast(new_fast)

    def on_rows_written(self, slots: list[int]) -> None:
        """EPLB wrote slow slots (on the current stream, between steps):
        copies of the slow tier staged ahead of time are stale."""
        from vllm.model_executor.offloader.base import get_offloader

        get_offloader().invalidate_staged()


def remap_unrouted(
    topk_ids: torch.Tensor, num_global: int, ep_rank: int
) -> torch.Tensor:
    """The stock remap of ``topk_ids == -1`` to an expert this rank doesn't own.

    That expert is -1 in both maps, so both kernel calls skip the row.
    """
    sentinel = num_global - 1 if ep_rank == 0 else 0
    return torch.where(topk_ids < 0, topk_ids.new_full((), sentinel), topk_ids)


def _store_config() -> "ExpertStoreConfig | None":
    """The expert store setting if it is on."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    store = getattr(config, "expert_store_config", None)
    return store if store is not None and store.enabled else None


def slot_tensor_bytes(
    quant_method: Any,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
) -> dict[str, int]:
    """Bytes of each of the four tensors of one expert in kernel layout
    (signed int4 pairs per byte, ``params_dtype`` scales), in unit order."""
    quant_config = quant_method.quant_config
    w13_out = quant_method.moe.w13_num_shards * intermediate_size_per_partition
    groups13 = groups2 = 1
    if quant_config.group_size != -1:
        groups13 = hidden_size // quant_config.group_size
        groups2 = intermediate_size_per_partition // quant_config.group_size
    itemsize = params_dtype.itemsize
    return {
        "w13": w13_out * hidden_size // 2,
        "w2": hidden_size * intermediate_size_per_partition // 2,
        "w13_scales": groups13 * w13_out * itemsize,
        "w2_scales": groups2 * hidden_size * itemsize,
    }


def kernel_layout_unit(
    w13_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_scales: torch.Tensor,
    scales_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """One expert's GPTQ tensors as the checkpoint holds them (int32
    ``[in/8, out]`` qweight, ``[groups, out]`` scales, ``w13`` already
    stacked over its shards) as the four tensors of the expert's store unit,
    computed on the host: the bytes the tiers write at the first load, for a
    slot of either tier. For building a store outside the engine (the unit
    source of the store's build tool)."""
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    unit: dict[str, torch.Tensor] = {}
    scales = {"w13": w13_scales, "w2": w2_scales}
    for key, qweight in (("w13", w13_qweight), ("w2", w2_qweight)):
        packed = qweight.cpu().transpose(0, 1).contiguous().view(torch.uint8)
        unit[key] = implement_zp(packed)
    for key in ("w13", "w2"):
        unit[f"{key}_scales"] = (
            scales[key].cpu().transpose(0, 1).contiguous().to(scales_dtype)
        )
    return unit


def _layout_fingerprint() -> str:
    """What the kernels' int4 transform does to every byte value. It is part
    of the store's identity: a kernels build whose transform differs sees
    another identity, whatever its version string says. A change of the row
    layout that this probe cannot see is caught at load, when a store that
    fills the host tier is compared with the rows this boot laid out
    (``ExpertTiers._check_store_against_this_boot``).

    Computed on the host whatever the default device: the engine calls it
    while the model is built under the accelerator's default-device context,
    the store's build tool calls it on a host without one, and both must get
    the same bytes. The ``torch.device("cpu")`` context also covers any
    tensor the kernels' function creates without a device."""
    import hashlib

    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    with torch.device("cpu"):
        probe = implement_zp(torch.arange(256, dtype=torch.uint8, device="cpu"))
        raw = probe.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _ep_barrier() -> None:
    if not torch.distributed.is_initialized():
        return
    from vllm.distributed.parallel_state import get_ep_group

    get_ep_group().barrier()


def _store_binding(
    store_config: "ExpertStoreConfig",
    layer_name: str,
    moe_config: Any,
    quant_method: Any,
    moe_quant_params: dict[str, Any],
    local_of: dict[int, int],
    num_global: int,
    num_layers: int,
) -> StoreBinding:
    """The layer's place in the process's expert store session (created by the
    first layer): schema and identity from the model, build or fill from the
    manifest."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.offloader.expert_tier_store import (
        StoreSession,
        build_schema,
        checkpoint_identity,
        get_session,
        store_identity,
    )
    from vllm.utils.platform_utils import is_pin_memory_available

    vllm_config = get_current_vllm_config()
    num_logical = int(moe_config.num_logical_experts)
    match = _LAYER_RE.search(layer_name)
    assert match is not None
    schema = build_schema(
        num_layers,
        num_logical,
        slot_tensor_bytes(
            quant_method,
            moe_quant_params["hidden_size"],
            moe_quant_params["intermediate_size_per_partition"],
            moe_quant_params["params_dtype"],
        ),
    )
    identity = store_identity(
        schema, checkpoint_identity(vllm_config.model_config), _layout_fingerprint()
    )
    session = get_session(
        (identity, tuple(store_config.dirs)),
        lambda: StoreSession(
            store_config,
            schema,
            identity,
            rank=int(moe_config.ep_rank),
            num_ranks=int(moe_config.ep_size),
            barrier=_ep_barrier,
            pin=store_config.pool_pinned and is_pin_memory_available(),
        ),
    )
    session.register(layer_name)
    # EPLB's initial placement: the routed experts in order, then the
    # redundant ones cycling over them.
    physical_to_logical = list(range(num_logical)) + [
        i % num_logical for i in range(num_global - num_logical)
    ]
    num_local = sum(1 for slot in local_of.values() if slot >= 0)
    initial = np.full(num_local, -1, dtype=np.int64)
    for g, slot in local_of.items():
        if slot >= 0:
            initial[slot] = physical_to_logical[g]
    spare = store_config.spare_rows_for(
        vllm_config.parallel_config.eplb_config.max_moves_per_rank
    )
    return StoreBinding(session, int(match.group(1)), spare, initial, num_logical)


def maybe_tier(
    layer_name: str,
    moe_config: Any,
    expert_map_manager: Any,
    quant_method: Any,
    moe_quant_params: dict[str, Any],
) -> ExpertTiers | None:
    """Tiers for this layer, or None: the setting is off or doesn't apply."""
    config = _tier_config()
    if config is None:
        return None
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

    if (
        not isinstance(quant_method, AutoGPTQMoEMethod)
        or quant_method.quant_config.weight_bits != 4
    ):
        logger.warning_once(
            "--cpu-offload-per-expert supports int4 GPTQ experts only; %s "
            "experts stay in VRAM",
            type(quant_method).__name__,
        )
        return None
    if _LAYER_RE.search(layer_name) is None:
        logger.info_once(
            "--cpu-offload-per-expert: %s is not a decoder layer; its experts "
            "stay in VRAM",
            layer_name,
        )
        return None
    num_global = moe_config.num_experts  # logical + redundant: EPLB's slots
    local_of = {g: expert_map_manager.map_global_to_local(g) for g in range(num_global)}
    num_local = sum(1 for slot in local_of.values() if slot >= 0)
    from vllm.config import get_current_vllm_config

    num_layers = get_current_vllm_config().model_config.hf_text_config.num_hidden_layers
    bytes_each = expert_bytes(
        quant_method,
        moe_quant_params["hidden_size"],
        moe_quant_params["intermediate_size_per_partition"],
        moe_quant_params["params_dtype"],
    )
    num_slow = budget_slow_slots(
        config.uva.cpu_offload_gb, num_layers, bytes_each, num_local
    )
    if num_slow == 0:
        return None
    num_pinned = pinned_slots(config, num_layers, bytes_each, num_slow)
    logger.info_once(
        "--cpu-offload-per-expert: --cpu-offload-gb %s: %d of %d expert slots per "
        "layer per rank in host memory (%d B each, %d layers); EPLB places the "
        "experts",
        config.uva.cpu_offload_gb,
        num_slow,
        num_local,
        bytes_each,
        num_layers,
    )
    binding = None
    store_config = _store_config()
    if store_config is not None:
        binding = _store_binding(
            store_config,
            layer_name,
            moe_config,
            quant_method,
            moe_quant_params,
            local_of,
            num_global,
            num_layers,
        )
    if num_pinned < num_slow:
        if binding is None:
            raise ValueError(
                "--cpu-offload-pinned-gb computes the experts beyond the pinned "
                "rows on the CPU from the expert store: add --expert-store-config"
            )
        if binding.session.mode != "fill":
            raise ValueError(
                "--cpu-offload-pinned-gb serves the experts beyond the pinned rows "
                "from an existing expert store; this start would build it "
                f"({binding.session.mode}): build the store first (one start "
                "without --cpu-offload-pinned-gb, or the store's build tool)"
            )
        logger.info_once(
            "--cpu-offload-pinned-gb %s: %d of the %d expert slots per layer per "
            "rank that are off the card in pinned host memory; %d computed on "
            "the CPU from the expert store",
            config.uva.cpu_offload_pinned_gb,
            num_pinned,
            num_slow,
            num_slow - num_pinned,
        )
    return ExpertTiers(
        layer_name,
        local_of,
        num_global,
        num_local,
        num_local - num_slow,
        binding,
        num_pinned=num_pinned,
    )


def pinned_slots(
    config: "OffloadConfig", num_layers: int, expert_bytes: int, num_slow: int
) -> int:
    """Slots per layer per rank in pinned rows: all ``num_slow`` off the card,
    or what ``--cpu-offload-pinned-gb`` per rank pays for (at least one)."""
    gb = config.uva.cpu_offload_pinned_gb
    if gb is None:
        return num_slow
    pinned = int(gb * 2**30 / max(num_layers * expert_bytes, 1))
    if pinned < 1:
        raise ValueError(
            f"--cpu-offload-pinned-gb {gb} pays for no expert slot per layer "
            f"({num_layers} layers of {expert_bytes} B experts): raise it to at "
            f"least {num_layers * expert_bytes / 2**30:.3f}"
        )
    return min(pinned, num_slow)
