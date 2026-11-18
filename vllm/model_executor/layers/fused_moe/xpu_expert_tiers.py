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
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch

from vllm.distributed.eplb.expert_weights import StoreBackedWeight, TieredExpertWeight
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.xpu_expert_temperature import (
    CLASS_KEYS,
    class_readout,
    classify,
)
from vllm.platforms import current_platform
from vllm.v1 import runtime_debug

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

# The weight group the kernels' W4A8 grouped GEMM takes (moe_w4a8.GROUP).
W4A8_GROUP = 128

# Routed ids read as unsigned for the one-pass range check of host_selection.
_UNSIGNED = {
    np.dtype(np.int32): np.dtype(np.uint32),
    np.dtype(np.int64): np.dtype(np.uint64),
}


@dataclass
class HostSelection:
    """The calls of a layer call that this rank computes on the CPU, as
    arrays: ``pair`` the flat (token * top_k + k) index of each call, in
    token order and top-k order inside a token; ``logical`` its logical
    expert; ``layer`` the store layer (store keys are ``(layer, logical)``);
    ``counts`` the rank's routed calls per tier. Arrays listed by the
    runtime's select op are views of its kept buffers, valid until its next
    listing."""

    pair: np.ndarray
    logical: np.ndarray
    layer: int
    counts: dict[str, int]
    # the distinct logical experts in order of first use, when the lister
    # had them at hand (the select op); None: derived from ``logical``
    distinct: list[int] | None = None

    def keys(self) -> list[tuple[int, int]]:
        return [(self.layer, int(e)) for e in self.logical]


# Decoder layers only: other MoE layers (e.g. MTP) keep their experts in VRAM, so
# the budget below is exact.
_LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")


@dataclass(frozen=True)
class ExpertFormat:
    """How a quantization method holds int4 group-quantized routed experts
    between loading and the tiers' repack: its parameter names and the axis
    order of what the loader writes. After the method's post-load step both
    formats are the same bytes (``[E, out, in/2]`` uint4 pairs with the
    zero point 8, low nibble first; ``[E, out, groups]`` scales), which the
    kernels and the tiers take; only the way there differs."""

    name: str
    weights: tuple[str, str]  # w13, w2: int32, 8 nibbles per word along in
    scales: tuple[str, str]  # w13, w2: one scale per group of ``in``
    # Loaded as ``[E, in/8, out]`` / ``[E, groups, out]`` (GPTQ), else as
    # ``[E, out, in/8]`` / ``[E, out, groups]`` (compressed-tensors).
    k_first: bool
    # Tensors per expert that the loader writes and the repack drops.
    extra: tuple[str, ...] = ()


GPTQ_INT4 = ExpertFormat(
    "gptq",
    ("w13_qweight", "w2_qweight"),
    ("w13_scales", "w2_scales"),
    k_first=True,
    extra=("w13_qzeros", "w2_qzeros", "w13_bias", "w2_bias"),
)
CT_PACKED_INT4 = ExpertFormat(
    "compressed-tensors pack-quantized",
    ("w13_weight_packed", "w2_weight_packed"),
    ("w13_weight_scale", "w2_weight_scale"),
    k_first=False,
    extra=("w13_weight_shape", "w2_weight_shape", "w13_bias", "w2_bias"),
)
_FORMATS = (GPTQ_INT4, CT_PACKED_INT4)
# Every per-expert parameter that gets a slow-slot twin, in either format.
_EXPERT_PARAMS = tuple(
    dict.fromkeys(name for f in _FORMATS for name in (*f.weights, *f.scales, *f.extra))
)
# The card rows' parameters at construction, in the order of _TIER_FIELDS
# (each by the names the quantization methods give it).
_RESERVE_PARAMS = (
    ("w13_qweight", "w13_weight_packed", "w13_weight"),
    ("w2_qweight", "w2_weight_packed", "w2_weight"),
    ("w13_scales", "w13_weight_scale"),
    ("w2_scales", "w2_weight_scale"),
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
            "parameters in module order",
            scope="process",
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


def _int4_geometry(quant_method: Any) -> tuple[int, int]:
    """(group size, values per int32) of a method's int4 experts: GPTQ keeps
    them on its quant config, compressed-tensors on the method itself."""
    if hasattr(quant_method, "weight_quant"):
        return int(quant_method.group_size), int(quant_method.packed_factor)
    quant_config = quant_method.quant_config
    return int(quant_config.group_size), int(getattr(quant_config, "pack_factor", 8))


def expert_bytes(
    quant_method: Any,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
) -> int:
    """One routed expert's packed weights and scales, as
    ``AutoGPTQMoEMethod.create_weights`` (or the compressed-tensors WNA16
    method, the same numbers N-first) lays them out: int32-packed weights,
    ``params_dtype`` scales per group."""
    w13_out = quant_method.moe.w13_num_shards * intermediate_size_per_partition
    group_size, pack = _int4_geometry(quant_method)
    groups13 = groups2 = 1
    if group_size != -1:
        groups13 = hidden_size // group_size
        groups2 = intermediate_size_per_partition // group_size
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


class StageRows:
    """Pinned rows in the layers' kernel layout that a layer call fills with
    the CPU-tier experts it routes many tokens to, and the map (physical id
    -> row, -1 elsewhere) the card's kernel call over them reads. Shared by
    the layers of a process: a layer writes them only after the host has
    seen its own routing, which the stream orders after every kernel of the
    previous layer, so no kernel still reads what is written (R1).

    With ``device_rows`` > 0 (``stage_device_rows``) also that many rows in
    device memory, filled by DMA on a side stream (:meth:`on_side`), so the
    card's call reads device memory instead of host memory through UVA (where
    each tile of tokens reads the expert's weights across PCIe). The DMA
    reads the host cache directly where it is registered with the driver,
    else the pinned rows (:meth:`to_device`, batch by batch); the ordering
    of the pinned rows above holds for them too, since a layer's call waits
    for its DMA."""

    # rows per DMA batch: the host copies the next batch into the pinned rows
    # while the previous one crosses to the card
    DMA_BATCH = 16

    def __init__(
        self,
        like: dict[str, torch.Tensor],
        rows: int,
        num_global: int,
        device: torch.device,
        device_rows: int = 0,
    ):
        from vllm.utils.platform_utils import is_pin_memory_available
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        self.num_rows = rows
        self.host = {
            k: _ARENA.empty((rows, *t.shape[1:]), t.dtype) for k, t in like.items()
        }
        self.views: dict[str, torch.Tensor] = {}
        for key, tensor in self.host.items():
            view = get_accelerator_view_from_cpu_tensor(tensor)
            view._vllm_is_uva_offloaded = True  # type: ignore[attr-defined]
            self.views[key] = view
        self.map = torch.full((num_global,), -1, dtype=torch.int32, device=device)
        self.map_host = torch.full(
            (num_global,), -1, dtype=torch.int32, pin_memory=is_pin_memory_available()
        )
        # device staging rows (stage_device_rows): allocated here, at load
        self.device_rows = min(int(device_rows), rows)
        self.dev: dict[str, torch.Tensor] = {}
        self._stream: Any = None
        self._mod: Any = None
        # the side stream's event after a layer call's last DMA, and the
        # compute stream's after the card's call that read the rows
        self.ready: Any = None
        self.free: Any = None
        if self.device_rows:
            self.dev = {
                k: torch.empty(
                    (self.device_rows, *t.shape[1:]), dtype=t.dtype, device=device
                )
                for k, t in like.items()
            }
            if device.type != "cpu":
                self._mod = torch.get_device_module(device)
                self._stream = self._mod.Stream(device)

    def rows_u8(self, key: str) -> torch.Tensor:
        return _u8(self.host[key])

    @property
    def device_gib(self) -> float:
        return sum(t.numel() * t.element_size() for t in self.dev.values()) / 2**30

    def begin_dma(self) -> None:
        """A layer call starts filling the device rows: the DMAs wait until
        the card's previous call over them is done (``free``), not for the
        kernels enqueued since, so they overlap this layer's own experts."""
        self.ready = None
        if self._stream is not None and self.free is not None:
            self._stream.wait_event(self.free)

    def to_device(self, lo: int, hi: int) -> None:
        """DMA of pinned rows ``lo..hi-1`` into the same device rows, one
        contiguous copy per tensor, on the side stream."""

        def copy() -> None:
            for key, dev in self.dev.items():
                _u8(dev)[lo:hi].copy_(_u8(self.host[key])[lo:hi], non_blocking=True)

        self.on_side(copy)

    def on_side(self, enqueue: Any) -> None:
        """Run ``enqueue`` (copies into the device rows) on the side stream
        and record ``ready`` after it (CPU, in tests: run it)."""
        if self._stream is None:
            enqueue()
            return
        with self._mod.stream(self._stream):
            enqueue()
            self.ready = self._mod.Event()
            self.ready.record(self._stream)

    def end_dma(self) -> None:
        """The compute stream waits for the layer call's last DMA (the
        card's call over the rows is enqueued after this)."""
        if self._stream is not None and self.ready is not None:
            self._mod.current_stream().wait_event(self.ready)

    def read_done(self) -> None:
        """Recorded on the compute stream right after the card's call over
        the device rows: the next layer call's DMAs wait for it."""
        if self._stream is not None:
            self.free = self._mod.Event()
            self.free.record(self._mod.current_stream())


class DeviceStagedCall:
    """The card's call over the device staging rows: ``apply`` runs the
    kernel and then marks the rows free for the next layer call's DMAs."""

    def __init__(self, impl: Any, rows: StageRows):
        self.impl = impl
        self.rows = rows

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        out = self.impl.apply(*args, **kwargs)
        self.rows.read_done()
        return out


_STAGE_ROWS: StageRows | None = None


def reset_stage_rows() -> None:
    """Forget the process's staging rows (tests; a new model load)."""
    global _STAGE_ROWS
    _STAGE_ROWS = None


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
    """Every CPU-tier layer of this process to ``value`` pinned slots. Only
    the capacity changes here; the slots move at EPLB's next step, in a tier
    pass it runs because a layer asks for one (EPLB is the one writer of the
    placement)."""
    tiers = _cpu_tiers()
    old = host_tier_bound()
    if tiers and tiers[0].pinned_release:
        _apply_bound_releasing(tiers, int(value), int(old))  # type: ignore[arg-type]
        return
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
            "CPU experts: the pinned bound is %d of the %d slots per layer per "
            "rank that are off the card (was %s); EPLB's next step runs a tier "
            "pass that moves the slots to it.",
            value,
            tiers[0].num_slow,
            old,
        )


def _set_bound_and_rows(t: "ExpertTiers", bound: int, arena: "_PinnedArena") -> None:
    """One layer to ``bound`` pinned slots with its pinned rows laid out
    for it. EPLB is the one writer of the placement, so neither direction
    moves a slot here: raising lays out more rows and then raises the
    capacity (EPLB's next tier pass reads the hottest CPU slots into them);
    lowering lowers the capacity and lays out fewer rows only once EPLB's
    tier pass has moved the slots over the bound to the CPU (at once when
    they already fit)."""
    assert t.placement is not None
    rows = t.pool_rows_for(bound)
    if rows > t.num_pool_rows:
        t._release_rows_to = None
        t.relayout_pool_rows(rows, arena)
        t.set_pinned_bound(bound)
    else:
        t.set_pinned_bound(bound)
        t._release_rows_to = None if rows == t.num_pool_rows else rows
        t.release_pinned_rows_if_due(arena)


def _apply_bound_releasing(tiers: list["ExpertTiers"], value: int, old: int) -> None:
    """``apply_host_tier_bound`` with ``pinned_release``: every layer's
    pinned rows are laid out again, layer by layer, in a fresh arena; the
    old arena's chunks go back to the system as soon as no layer holds rows
    in them (torch's pinned cache is emptied after each layer). On a failure
    the layers already changed (and the failing one) go back to ``old``."""
    global _ARENA
    old_gib = _ARENA.gib
    old_arena = _ARENA
    arena = type(old_arena)()
    _ARENA = arena
    # From here the old chunks live as long as some layer's rows do.
    for name in ("chunks", "used"):
        held = getattr(old_arena, name, None)
        if held is not None:
            held.clear()
    start = time.perf_counter()
    done: list[ExpertTiers] = []
    try:
        for t in tiers:
            done.append(t)
            _set_bound_and_rows(t, value, arena)
            _empty_host_cache()
    except BaseException:
        for t in reversed(done):
            try:
                _set_bound_and_rows(t, old, arena)
            except Exception:  # noqa: BLE001 - the first error is raised
                logger.exception(
                    "%s: could not go back to %d pinned slots", t.layer_name, old
                )
        _empty_host_cache()
        raise
    waiting = sum(t._release_rows_to is not None for t in tiers)
    if waiting:
        logger.info(
            "CPU experts: the pinned bound is %d of the %d slots per layer per "
            "rank that are off the card (was %d); EPLB's next step runs a tier "
            "pass that moves the slots to it, and %d layers then lay their "
            "pinned rows out for it (%d per layer), giving the host memory "
            "back (%.2f GiB pinned now).",
            value,
            tiers[0].num_slow,
            old,
            waiting,
            tiers[0].pool_rows_for(value),
            old_gib,
        )
        return
    logger.info(
        "CPU experts: the pinned bound is %d of the %d slots per layer per "
        "rank that are off the card (was %d); EPLB's next step runs a tier "
        "pass that moves the slots to it. Pinned rows laid out again: %d per "
        "layer, pinned host memory %.2f -> %.2f GiB in %.2f s.",
        value,
        tiers[0].num_slow,
        old,
        tiers[0].num_pool_rows,
        old_gib,
        arena.gib,
        time.perf_counter() - start,
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


def tier_classes_state(
    hot: float, warm: float, hysteresis: float
) -> dict[str, Any] | None:
    """Per layer of this rank, the experts that are hot, warm and cold by
    EPLB's load window (the load the last pass planned with, over the mean
    expert's) against the rows each tier has; the sums say whether the
    pinned bound (or the card tier) is short. None without tiered layers
    that EPLB has placed by a load yet."""
    tiers = [t for t in live_tiers() if t.placement is not None]
    per_layer = [t.class_readout(hot, warm, hysteresis) for t in tiers]
    pairs = [(t, r) for t, r in zip(tiers, per_layer) if r is not None]
    if not pairs:
        return None
    rows = [r for _, r in pairs]
    totals = {k: sum(r[k] for r in rows) for k in CLASS_KEYS}
    return {
        "thresholds": {"hot": hot, "warm": warm, "hysteresis": hysteresis},
        "layers": len(rows),
        "totals": totals,
        "per_layer_min_max": {
            k: [min(r[k] for r in rows), max(r[k] for r in rows)]
            for k in ("hot", "warm", "cold", "rows_short")
        },
        "layers_short": sum(1 for r in rows if r["rows_short"] > 0),
        "card_rows": rows[0]["card_rows"],
        "pinned_rows": rows[0]["pinned_rows"],
        "per_layer": {t.layer_name: r for t, r in pairs},
    }


def _repack_int4_pinned(
    qweight: torch.Tensor,
    device: torch.device,
    rows: int | None = None,
    k_first: bool = True,
) -> torch.Tensor:
    """int32 ``[E, in/8, out]`` (GPTQ, ``k_first``) or ``[E, out, in/8]``
    (compressed-tensors) on the CPU -> signed-int4 bytes ``[E, out, in/2]``.

    The layout transform of the method's post-load step plus ``implement_zp``,
    run on ``device`` a few experts at a time into pinned host memory, so device
    temporaries stay small. ``rows`` > E leaves that many rows in the tensor,
    the ones beyond E zeroed.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    if k_first:
        e, k8, n = qweight.shape
    else:
        e, n, k8 = qweight.shape
    out = _ARENA.empty((max(e, rows or 0), n, k8 * 4), torch.uint8)
    out[e:].zero_()
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        if k_first:
            chunk = chunk.transpose(1, 2)
        packed = chunk.contiguous().view(torch.uint8)
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
        expert_format: ExpertFormat = GPTQ_INT4,
    ):
        assert 0 < num_fast < num_local
        self.layer_name = layer_name
        # The parameter names and loaded axis order of the method's experts.
        self.expert_format = expert_format
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
        # The last moves' stats (bounded: EPLB's tier passes commit a move per
        # layer per pass) and how many were committed.
        self.move_log: deque[dict[str, float]] = deque(maxlen=512)
        self.moves_committed = 0
        # Slots whose row or expert changed since the last scrub.
        self._touched: set[int] = set()
        # The mean load of a logical expert in the window the last move or
        # fill planned with (temperature = slot load / this), and the one of a
        # move prepared and not yet committed.
        self._load_mean: float | None = None
        self._prepared_mean: float | None = None
        # The classes the last readout gave (its hysteresis).
        self._cls: np.ndarray | None = None
        # The capacity of a tier changed (the pinned bound): EPLB's next step
        # runs a tier pass for it. Set and read between steps only.
        self.tier_pass_wanted = False
        self._topk: int | None = None
        self._impl_common: dict[str, Any] | None = None
        self._stage_impl: Any = None
        self._stage_device_impl: DeviceStagedCall | None = None
        # Whether the layer has host slots (computed on the CPU), decided at
        # load; the slot counts may move later, this does not.
        self._host_tier = pinned < num_local - num_fast
        # The card tier's rows in a reserved device range (fast_rows_max):
        # None while the card tier is fixed. See :meth:`resize_fast`.
        self.fast_rows: Any = None
        # The range the card rows were put in at construction
        # (reserve_card_rows), until the move at weight processing.
        self._early_rows: Any = None
        self._num_fast_at_load = num_fast
        self._fast_capacity = 0
        # Pinned rows per layer laid out again at run time (pinned_release);
        # None: the rows sized at load.
        self._pool_rows: int | None = None
        # With pinned_release, after a lower bound: the rows to lay out once
        # EPLB's tier pass has moved the slots over the bound to the CPU
        # (:meth:`release_pinned_rows_if_due`); None: nothing waits.
        self._release_rows_to: int | None = None
        # Pinned rows beyond the spares that let the card tier shrink below
        # its size at load in a layer without host slots (fast_rows_min).
        self._extra_pool_rows = 0
        self._layer_ref: Any = None
        # EPLB's store-backed weights made from this layer (eplb_weights).
        self._eplb_parts: list[Any] = []
        cfg = None if store is None else getattr(store.session, "config", None)
        if cfg is not None and int(getattr(cfg, "fast_rows_max", 0) or 0) > 0:
            self._fast_capacity = max(num_fast, min(int(cfg.fast_rows_max), num_local))
            floor = getattr(cfg, "fast_rows_min", None)
            if not self._host_tier and floor is not None and floor < num_fast:
                self._extra_pool_rows = num_fast - int(floor)

    @property
    def num_slow(self) -> int:
        return self.num_local - self.num_fast

    @property
    def num_host(self) -> int:
        """Slots computed on the CPU (no row the kernels read) when every
        pinned row the bound allows is used."""
        if self.placement is not None:
            return int(self.placement.num_host)
        return self.num_slow - self.num_pinned

    @property
    def promote_rows(self) -> int:
        """Pinned rows beyond the bound at load and the spares, which a
        larger bound (``tiers.set host_tier_bound``) may take at run time;
        EPLB's tier pass moves the slots into them."""
        if not self._host_tier or self.store_binding is None:
            return 0
        return int(self.store_binding.session.config.host_promote_rows)

    @property
    def num_pool_rows(self) -> int:
        """Rows of each pinned tensor: the pinned slots, the spares, and the
        rows a larger pinned bound may take (``host_promote_rows``) and,
        without host slots, the rows a card tier that shrinks below its
        load size needs; laid out again at run time with pinned_release."""
        if self._pool_rows is not None:
            return self._pool_rows
        spare = 0 if self.store_binding is None else self.store_binding.spare_rows
        return self.num_pinned + spare + self.promote_rows + self._extra_pool_rows

    @property
    def fills_from_store(self) -> bool:
        """The host tier is filled from an existing store, not from the
        checkpoint: the loader skips its slots."""
        b = self.store_binding
        return b is not None and b.session.mode == "fill"

    def attach_twins(self, layer: torch.nn.Module) -> None:
        """A pageable CPU twin ``[num_slow, ...]`` for every per-expert param
        (and, with ``fast_rows_max``, the card rows put in their range)."""
        self.reserve_card_rows(layer)
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
        # For the card tier's move into a reserved range (impls): the layer's
        # parameters that hold its rows.
        try:
            self._layer_ref = weakref.ref(layer)
        except TypeError:  # a test's stand-in layer
            self._layer_ref = None
        device: torch.device = getattr(layer, "w13_weight").device  # noqa: B009
        binding = self.store_binding
        rows = self.num_pool_rows if binding is not None else None
        fmt = self.expert_format
        if self.fills_from_store:
            # No twins: the host tier's rows come from the store, in the shape
            # of the fast tier's tensors.
            assert rows is not None
            for key, scales_name in zip(("w13", "w2"), fmt.scales):
                fast_w = getattr(layer, f"{key}_weight").data.view(torch.uint8)
                self.slow[key] = _ARENA.empty((rows, *fast_w.shape[1:]), torch.uint8)
                fast_s = getattr(layer, scales_name)
                self.slow[f"{key}_scales"] = _ARENA.empty(
                    (rows, *fast_s.shape[1:]), fast_s.dtype
                )
            for tensor in self.slow.values():
                tensor[self.num_pinned :].zero_()
            self._fill_host_tier()
        else:
            for key, weight_name, scales_name in zip(
                ("w13", "w2"), fmt.weights, fmt.scales
            ):
                self.slow[key] = _repack_int4_pinned(
                    twins.pop(weight_name).data, device, rows, k_first=fmt.k_first
                )
                scales = twins.pop(scales_name).data
                if fmt.k_first:
                    scales = scales.transpose(1, 2)
                self.slow[f"{key}_scales"] = _pin(
                    scales.contiguous().to(getattr(layer, scales_name).dtype), rows
                )
        # qzeros: symmetric int4 has none on XPU; weight_shape: the shapes the
        # loader checked, not kernel material
        del twins
        if fmt is not GPTQ_INT4:
            logger.info_once(
                "Expert tiers: %s int4 experts; pinned rows laid out from the "
                "loaded [E, out, in/8] words without a transpose, the bytes the "
                "card rows get.",
                fmt.name,
            )
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
                extra_rows=self.promote_rows + self._extra_pool_rows,
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
                if int(getattr(binding.session.config, "stage_device_rows", 0)):
                    # Device staging rows are taken at load, before the KV
                    # cache is sized, so the KV cache gives their memory; the
                    # host cache is registered for DMA into them now too.
                    self.stage_rows(device)
                    self.cpu.direct_dma()
        experts.tiers = self
        _LIVE_TIERS.add(self)

    # -- the CPU tier ---------------------------------------------------------

    def _cpu_w4a8(self, experts: Any, hidden: int, inter: int, group: int) -> bool:
        """Whether the card runs this layer's experts W4A8, so the CPU must
        too: the experts ask for it (``XPUExpertsW4A8Int``) and the kernels'
        W4A8 GEMM takes the layer by ``vllm_xpu_kernels.moe_w4a8.eligible``'s
        rules (SiLU, no clamp, 128-element groups, N and K multiples of 64).
        Otherwise the card keeps W4A16 for the layer, and so does the CPU."""
        if not getattr(experts, "impl_apply_kwargs", {}).get("w4a8"):
            return False
        moe = getattr(experts, "moe_config", None)
        activation = getattr(getattr(moe, "activation", None), "value", "silu")
        why = None
        if activation != "silu" or getattr(experts, "gemm1_clamp_limit", None):
            why = f"activation {activation} with a clamp or not SiLU"
        elif group != W4A8_GROUP:
            why = f"group size {group}, not {W4A8_GROUP}"
        elif hidden % 64 or inter % 64:
            why = f"hidden {hidden} / intermediate {inter} not multiples of 64"
        if why is not None:
            logger.warning_once(
                "CPU experts: W4A8 requested, but the card's W4A8 GEMM does not "
                "take this layer (%s): the card keeps W4A16, and so does the CPU.",
                why,
            )
            return False
        return True

    def _start_cpu_tier(self, experts: Any) -> None:
        """Host slots exist: take the process's CPU tier and read their
        experts into its host cache ahead of the first step (as far as free
        rows go)."""
        from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
            UNIT_TENSORS,
            UnitLayout,
            get_runtime,
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
        self.cpu = get_runtime(
            binding.session.config,
            binding.session.store,
            layout,
            w4a8=self._cpu_w4a8(experts, hidden, inter, group),
        )
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
        sel = self.host_selection(ids)
        return sel.keys(), sel.pair, sel.counts

    def _host_selection(
        self, ids: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, dict[str, int]]:
        sel = self.host_selection(ids)
        tok, k = np.divmod(sel.pair, ids.shape[1])
        return sel.keys(), tok, k, sel.counts

    def host_selection(self, ids: np.ndarray) -> "HostSelection":
        """The calls of a step's routing ``ids`` [T, K] (physical ids) that
        this rank computes on the CPU, as arrays (no per-call Python): the
        flat pair index ``token * K + k`` of each, in token order and top-k
        order inside a token, its logical expert, the store layer, and the
        rank's routed calls per tier. Raises when a routed id is outside the
        physical ids: nothing the kernels would skip silently passes here."""
        from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
            CpuExpertError,
        )
        from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

        flat = ids.reshape(-1)
        # One pass checks the range: a negative id read as unsigned is above
        # every table index, so the gather raises for both ends.
        unsigned = _UNSIGNED.get(flat.dtype)
        try:
            if unsigned is None:
                if flat.size and int(flat.min()) < 0:
                    raise IndexError
                tier = self._tier_of_global[flat]
            else:
                tier = self._tier_of_global[flat.view(unsigned)]
        except IndexError:
            raise CpuExpertError(
                f"{self.layer_name}: a routed expert id is outside the "
                f"{self.num_global} physical experts (min {int(ids.min())}, max "
                f"{int(ids.max())})"
            ) from None
        # tier values -1 (another rank's), FAST, POOL, HOST -> bins 0..3
        bins = np.bincount(tier.view(np.uint8) + np.uint8(1), minlength=4)
        counts = {
            "card": int(bins[FAST + 1]),
            "pinned": int(bins[POOL + 1]),
            "cpu": int(bins[HOST + 1]),
        }
        pair = np.flatnonzero(tier == HOST)
        binding = self.store_binding
        layer = 0 if binding is None else binding.store_layer
        if pair.size == 0:
            logical = pair
        else:
            assert self.slot_logical is not None and binding is not None
            logical = np.asarray(self.slot_logical)[self._slot_of_global[flat[pair]]]
        return HostSelection(pair, logical.astype(np.int64, copy=False), layer, counts)

    # -- prefill staging of CPU-tier experts (stage_min_calls) ----------------

    def stage_rows(self, device: torch.device) -> "StageRows":
        """The process's staging rows (made at the first use, sized
        ``stage_rows`` or this layer's CPU slots)."""
        global _STAGE_ROWS
        if _STAGE_ROWS is None:
            binding = self.store_binding
            cfg = None if binding is None else binding.session.config
            want = 0 if cfg is None else int(cfg.stage_rows)
            assert self.placement is not None
            rows = want or max(1, self.num_host)
            on_device = 0 if cfg is None else int(getattr(cfg, "stage_device_rows", 0))
            _STAGE_ROWS = StageRows(
                self.slow, rows, self.num_global, device, device_rows=on_device
            )
            logger.info(
                "CPU experts: %d staging rows (%.2f GiB pinned) for the experts "
                "of the CPU tier that a large step routes many tokens to; the "
                "card computes those from the rows.",
                rows,
                rows * self.slot_bytes() / 2**30,
            )
            if _STAGE_ROWS.device_rows:
                logger.info(
                    "CPU experts: %d device staging rows (%.2f GiB on the card, "
                    "reserved at load, so the KV cache gives it): a layer call "
                    "that stages copies its staged experts there by DMA on a side "
                    "stream, and the card's call reads them from device memory "
                    "(stage_device_rows).",
                    _STAGE_ROWS.device_rows,
                    _STAGE_ROWS.device_gib,
                )
        return _STAGE_ROWS

    def stage_impl(self, rows: "StageRows") -> Any:
        """This layer's kernel call over the staging rows (built once; the
        rows and their map are shared by the layers, which run one at a
        time)."""
        if self._stage_impl is None:
            assert self._impl_common is not None
            XpuFusedMoe = _xpu_fused_moe()
            sw13, sw2 = rows.views["w13"], rows.views["w2"]
            sw13.xpu_fused_moe = True  # type: ignore[attr-defined]
            impl = XpuFusedMoe(
                w13=sw13,
                w13_scales=rows.views["w13_scales"],
                w13_bias=None,
                w2=sw2,
                w2_scales=rows.views["w2_scales"],
                w2_bias=None,
                num_experts=rows.num_rows,
                expert_map=rows.map,
                **self._impl_common,
            )
            impl.w13, impl.w2 = sw13.view(torch.int8), sw2.view(torch.int8)
            impl.w13._vllm_is_uva_offloaded = True
            impl.w2._vllm_is_uva_offloaded = True
            self._stage_impl = impl
        return self._stage_impl

    def stage_device_impl(self, rows: "StageRows") -> DeviceStagedCall:
        """This layer's kernel call over the device staging rows
        (``stage_device_rows``): the same kernel as :meth:`stage_impl`, the
        same map, built over the device rows (its ``num_experts`` is their
        number, so the map's entries must stay below it); built once."""
        if self._stage_device_impl is None:
            assert self._impl_common is not None and rows.device_rows
            XpuFusedMoe = _xpu_fused_moe()
            dw13, dw2 = rows.dev["w13"], rows.dev["w2"]
            dw13.xpu_fused_moe = True  # type: ignore[attr-defined]
            impl = XpuFusedMoe(
                w13=dw13,
                w13_scales=rows.dev["w13_scales"],
                w13_bias=None,
                w2=dw2,
                w2_scales=rows.dev["w2_scales"],
                w2_bias=None,
                num_experts=rows.device_rows,
                expert_map=rows.map,
                **self._impl_common,
            )
            impl.w13, impl.w2 = dw13.view(torch.int8), dw2.view(torch.int8)
            self._stage_device_impl = DeviceStagedCall(impl, rows)
        return self._stage_device_impl

    # -- temperature and classes (readout of EPLB's load window) -------------

    def temperature(self) -> np.ndarray | None:
        """Per slot, the load of its expert in the window EPLB's last move or
        fill planned with, over the mean logical expert's: 1.0 is an average
        expert. None before EPLB has placed the layer by a load. The same
        signal and the same order EPLB's plan cuts into tiers."""
        if self._slot_load is None or not self._load_mean:
            return None
        return self._slot_load / self._load_mean

    def class_readout(
        self, hot: float, warm: float, hysteresis: float
    ) -> dict[str, Any] | None:
        """The layer's experts by class (thresholds over :meth:`temperature`)
        against the rows each tier has (see ``class_readout``)."""
        temp = self.temperature()
        placement = self.placement
        if temp is None or placement is None:
            return None
        self._cls = classify(temp, self._cls, hot, warm, hysteresis)
        out = class_readout(
            self._cls, placement.tier, self.num_fast, placement.num_pinned
        )
        out["temp_max"] = round(float(temp.max()), 3) if temp.size else 0.0
        return out

    # -- the pinned bound while serving (runtime control: host_tier_bound) ---

    def pinned_bound_range(self) -> tuple[int, int]:
        """The pinned slots per layer a running engine may move to: at least
        one, at most every pool row but EPLB's spares (the rows were sized at
        load: the load-time bound plus ``host_promote_rows``)."""
        assert self.placement is not None
        if self.pinned_release:
            # the rows are laid out again for the bound (relayout_pool_rows)
            return 1, max(1, int(self.placement.num_slow))
        return 1, self.placement.max_pinned_capacity()

    @property
    def pinned_release(self) -> bool:
        """The pinned rows follow the bound (``pinned_release``): a lower
        bound gives their host memory back, a higher one lays out more."""
        b = self.store_binding
        return (
            self._host_tier
            and b is not None
            and bool(getattr(b.session.config, "pinned_release", False))
        )

    def pool_rows_for(self, bound: int) -> int:
        """Pinned rows per layer that serve ``bound`` slots: the bound, EPLB's
        spare rows and the promotion rows."""
        assert self.store_binding is not None
        return int(bound) + self.store_binding.spare_rows + self.promote_rows

    def release_pinned_rows_if_due(self, arena: "_PinnedArena | None" = None) -> bool:
        """With ``pinned_release`` after a lower bound: once no more slots
        hold pinned rows than the bound (EPLB's tier pass moved the rest to
        the CPU), lay the rows out for the bound, giving the others' host
        memory back. A row relayout: no slot changes its tier or its expert.
        Called where the bound is lowered and after each EPLB commit of the
        layer, between two steps on the forward's thread. True when the rows
        were laid out; a failure is logged and tried again at the next
        commit."""
        from vllm.model_executor.offloader.expert_tier_map import POOL

        target = self._release_rows_to
        placement = self.placement
        if target is None or placement is None:
            return False
        if target == self.num_pool_rows:
            self._release_rows_to = None
            return False
        used = int((placement.tier == POOL).sum())
        if used + placement.spare > target:
            return False  # EPLB's tier pass has not moved them yet
        try:
            stats = self.relayout_pool_rows(target, _ARENA if arena is None else arena)
        except Exception as e:  # noqa: BLE001 - serving goes on; retried
            logger.warning(
                "%s: the pinned rows could not be laid out for the bound yet "
                "(%s); tried again at EPLB's next commit of the layer.",
                self.layer_name,
                e,
            )
            return False
        self._release_rows_to = None
        _empty_host_cache()
        if runtime_debug.on("tiers"):
            runtime_debug.emit(
                "tiers",
                logger,
                "%s: pinned rows %d -> %d after EPLB's pass (%.1f ms)",
                self.layer_name,
                int(stats["rows_from"]),
                int(stats["rows_to"]),
                stats["ms"],
            )
        if not any(t._release_rows_to is not None for t in _cpu_tiers()):
            logger.info(
                "CPU experts: every layer's pinned rows are laid out for the "
                "bound (%d per layer); pinned host memory now %.2f GiB.",
                target,
                pinned_arena_gib(),
            )
        return True

    def relayout_pool_rows(self, num_rows: int, arena: "_PinnedArena") -> dict:
        """Between two steps, on the thread that enqueues the forward, in a
        layer with host slots (the MoE op then runs outside the captured
        graphs, so the slow kernel's tensors may move): lay the pinned rows
        out again in ``num_rows`` rows of ``arena``. The rows that serve
        slots are copied (host to host) to rows ``0..k-1``; the slow kernel,
        its device views and the slow map are switched to the new rows; the
        device finishes every kernel that read the old rows before this
        returns, and the old rows are dropped (their chunk goes back to the
        system once no layer holds rows in it). Raises, with the layer as it
        was, when the rows in use do not fit."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        _, placement, ops = self._require_store()
        assert self._impls is not None and self.slow_map is not None
        start = time.perf_counter()
        old_rows = self.num_pool_rows
        with self._move_lock:
            if self._eplb_pending:
                raise RuntimeError(
                    f"{self.layer_name}: an EPLB move is prepared; the pinned "
                    "rows can be laid out again once it is committed"
                )
            # Rows in quarantine may still be read by the device or written
            # by a host read; both finish before the rows move.
            deadline = time.monotonic() + 30.0
            while _quarantined(placement) and time.monotonic() < deadline:
                time.sleep(0.001)
            if _quarantined(placement):
                raise RuntimeError(
                    f"{self.layer_name}: pinned rows still in use after 30 s"
                )
            saved = (placement.row.copy(), placement.pool, placement.num_pool_rows)
            saved_extra = placement.extra_rows
            moved = placement.relayout_pool(num_rows)
            try:
                new: dict[str, torch.Tensor] = {}
                for key, old in self.slow.items():
                    t = arena.empty((num_rows, *old.shape[1:]), old.dtype)
                    t.zero_()
                    for src, dst in moved.items():
                        t[dst].copy_(old[src])
                    new[key] = t
                views = {}
                for key, t in new.items():
                    view = get_accelerator_view_from_cpu_tensor(t)
                    view._vllm_is_uva_offloaded = True  # type: ignore[attr-defined]
                    views[key] = view
            except BaseException:
                placement.row, placement.pool, placement.num_pool_rows = saved
                placement.extra_rows = saved_extra
                raise
            old_slow = self.slow
            self.slow, self.slow_views = new, views
            self._pool_rows = num_rows
            slow = self._impls[1]
            sw13, sw2 = views["w13"], views["w2"]
            slow.w13, slow.w2 = sw13.view(torch.int8), sw2.view(torch.int8)
            slow.w13._vllm_is_uva_offloaded = True
            slow.w2._vllm_is_uva_offloaded = True
            slow.gemm1_wei_scales = views["w13_scales"]
            slow.gemm2_wei_scales = views["w2_scales"]
            slow.num_experts = num_rows
            slow.local_experts_num = num_rows
            _, slow_np = placement.maps(self.slot_global, self.num_global)
            ops.compute_write(self.slow_map, slow_np)
            self._check_views_alias()
            self.on_rows_written([])
            _forget_staged_slow()
            # EPLB reads only the shapes of a store-backed layer's weights;
            # their pinned part must not keep the old rows alive.
            for ref in self._eplb_parts:
                weight = ref()
                if weight is None or len(weight.parts) < 2:
                    continue
                part = weight.parts[1]
                weight.parts = (
                    weight.parts[0],
                    torch.empty(part.shape, dtype=part.dtype, device="meta"),
                    *weight.parts[2:],
                )
            # Nothing reads the old rows once the device is past this point.
            _device_sync()
            del old_slow
        return {
            "rows_from": float(old_rows),
            "rows_to": float(num_rows),
            "ms": (time.perf_counter() - start) * 1e3,
        }

    def _fill_rows_cached(
        self, reads: Mapping[int, int], rows: Mapping[int, int]
    ) -> None:
        """:meth:`_fill_rows`, taking the units the CPU tier's host cache
        holds from there (a host copy, the cache row held meanwhile) instead
        of the drive. Runs where EPLB prepares a move (its worker thread)."""
        assert self.store_binding is not None
        rest: dict[int, int] = {}
        for slot, logical in reads.items():
            key = (self.store_binding.store_layer, int(logical))
            dests = {k: _u8(t)[rows[slot]] for k, t in self.slow.items()}
            if self.cpu is None or not self.cpu.cache.copy_unit_out(key, dests):
                rest[slot] = logical
        if rest:
            self._fill_rows(rest, {s: rows[s] for s in rest})

    def set_pinned_bound(self, bound: int) -> dict[str, float]:
        """Between two steps: from now on ``bound`` of this layer's off-card
        slots may hold pinned rows. Nothing moves here: the layer asks EPLB
        for a tier pass (``tier_pass_wanted``), which plans the pinned set
        with the new capacity from its load window and moves the slots like
        any of its moves (lowering reads nothing, raising reads the hottest
        CPU slots). Raises while an EPLB move is prepared."""
        from vllm.model_executor.offloader.expert_tier_map import POOL

        _, placement, _ = self._require_store()
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
        with self._move_lock:
            if self._eplb_pending:
                raise RuntimeError(
                    f"{self.layer_name}: an EPLB move is prepared; the bound "
                    "can change once it is committed"
                )
            old = int(placement.num_pinned)
            placement.set_pinned_capacity(bound)
            self.tier_pass_wanted = True
        if runtime_debug.on("tiers"):
            runtime_debug.emit(
                "tiers",
                logger,
                "%s: pinned bound %d -> %d (capacity; EPLB's tier pass moves "
                "the slots)",
                self.layer_name,
                old,
                bound,
            )
        return {
            "bound": float(bound),
            "pinned_now": float(int((placement.tier == POOL).sum())),
        }

    # -- the card tier's size while serving (fast_rows_max) -----------------

    def fast_range(self) -> tuple[int, int]:
        """The card rows per layer a running engine may move to. At most the
        reserved rows. At least one with host slots (a slot leaving the card
        can go to the CPU), else as many as keep every slot off the card in a
        pinned row with EPLB's spare rows still free."""
        assert self.placement is not None
        cfg = None if self.store_binding is None else self.store_binding.session.config
        floor = None if cfg is None else getattr(cfg, "fast_rows_min", None)
        if self._host_tier:
            lo = int(floor) if floor else 1
        else:
            p = self.placement
            lo = max(
                int(floor) if floor else self._num_fast_at_load,
                self.num_local - (p.num_pool_rows - p.spare),
                1,
            )
        return min(lo, self._fast_capacity), self._fast_capacity

    def _heat_rank(self) -> np.ndarray:
        """Each slot's place in the heat order (0 coldest): by the load of
        EPLB's last move or fill (its window), the one signal its tier passes
        cut by; equal loads keep slot order."""
        load = np.zeros(self.num_local) if self._slot_load is None else self._slot_load
        rank = np.empty(self.num_local, dtype=np.int64)
        rank[np.argsort(load, kind="stable")] = np.arange(self.num_local)
        return rank

    def _copy_fast(self, src: int, dst: int) -> None:
        """Enqueue the copy of one card row into another on the side stream."""
        _, _, ops = self._require_store()
        assert self._impls is not None
        fast = self._impls[0]
        for impl_field, _ in _TIER_FIELDS:
            rows = _u8(getattr(fast, impl_field))
            ops.side_copy(rows[dst], rows[src])

    def _apply_sets(
        self,
        new_fast: np.ndarray,
        new_pool: np.ndarray,
        timeout: float,
        stats: dict[str, float],
    ) -> None:
        """One move to the tier sets given (the experts stay in their
        slots): units read from the host cache or the store into pinned rows,
        pinned rows copied to card rows, the maps switched; see
        :meth:`commit_move`."""
        placement = self.placement
        assert placement is not None and self.slot_logical is not None
        plan = placement.plan_from_set(
            self.slot_logical,
            self.slot_logical,
            new_fast,
            new_pool,
            check_budget=False,
        )
        move = placement.prepare(plan, self._fill_rows_cached, timeout=timeout)
        move.stats = {"units": float(len(plan.reads))}
        self.commit_move(move)
        stats["units_read"] += len(plan.reads)
        stats["commits"] += 1
        stats["to_card"] += len(plan.to_fast)
        stats["off_card"] += len(plan.demoted)

    def _fast_rounds(self, target: int, timeout: float, stats: dict) -> None:
        placement = self.placement
        assert placement is not None
        while int(placement.is_fast().sum()) != target:
            # The choice is the placement's (the tier-choice rules EPLB's
            # moves use), by EPLB's load; this method only commits it.
            sets = placement.plan_card_round(
                self._heat_rank(),
                target,
                placement.pool.free_count + placement.pool.quarantined,
            )
            if sets is None:
                from vllm.model_executor.offloader.expert_tier_map import (
                    RowsUnavailable,
                )

                raise RowsUnavailable(
                    f"{self.layer_name}: no move toward {target} card slots "
                    f"fits the free pinned rows ({placement.pool.free_count} "
                    f"free, {placement.pool.quarantined} in quarantine)"
                )
            self._apply_sets(*sets, timeout, stats)

    def _pool_rounds(self, timeout: float, stats: dict) -> None:
        """With host slots: the pinned rows serve the bound's number of slots
        again, the hottest off the card (a grow took some pinned slots to the
        card; a shrink may have left fewer than the bound)."""
        placement = self.placement
        assert placement is not None
        if not self._host_tier:
            return
        while True:
            new_pool = placement.plan_pinned_refill(
                self._heat_rank(),
                placement.pool.free_count + placement.pool.quarantined,
            )
            if new_pool is None:
                return
            self._apply_sets(placement.is_fast(), new_pool, timeout, stats)

    def resize_fast(self, target: int, *, timeout: float | None = None) -> dict:
        """Between two steps, on the thread that enqueues the forward: hold
        ``target`` of this layer's slots in card rows. A grow maps the rows'
        memory (``FastRows.commit``; the caller has made room for it) and
        moves the hottest slots off the card in; a shrink moves the coldest
        out, moves the slots left in rows at or above ``target`` down
        (card-to-card copies), drops those rows and releases their memory to
        the device (whole objects, after the device has finished every
        kernel that read them). Bytes enter card rows only from pinned rows
        or the store. Raises while an EPLB move is prepared; on a failure
        part-way the caller asks for the old size again (moves are whole:
        every committed move left the maps and the rows consistent)."""
        binding, placement, ops = self._require_store()
        rows = self.fast_rows
        if rows is None:
            raise RuntimeError(
                f"{self.layer_name}: the card tier is fixed in this engine "
                "(--expert-store-config fast_rows_max)"
            )
        lo, hi = self.fast_range()
        if not lo <= target <= hi:
            raise ValueError(
                f"{self.layer_name}: card rows per layer must be {lo}..{hi}, "
                f"got {target}"
            )
        if timeout is None:
            timeout = float(binding.session.config.read_timeout_s)
        assert self.fast_map is not None
        start = time.perf_counter()
        stats: dict[str, float] = dict.fromkeys(
            ("units_read", "commits", "to_card", "off_card", "relocated"), 0.0
        )
        with self._move_lock:
            if self._eplb_pending:
                raise RuntimeError(
                    f"{self.layer_name}: an EPLB move is prepared; the card "
                    "tier can change once it is committed"
                )
            old = int(placement.num_fast)
            mapped_before = rows.mapped_bytes
            if target > old:
                rows.commit(target)
                placement.grow_fast_rows(target)
                self.num_fast = target
                self._fast_rounds(target, timeout, stats)
                self._pool_rounds(timeout, stats)
            elif target < old or int(placement.is_fast().sum()) != target:
                self._fast_rounds(target, timeout, stats)
                stats["relocated"] = placement.compact_fast(
                    target,
                    ops,
                    slot_global=self.slot_global,
                    num_global=self.num_global,
                    fast_map=self.fast_map,
                    copy_fast=self._copy_fast,
                )
                placement.shrink_fast_rows(target)
                self.num_fast = target
                # Waits for the device: the rows dropped were read last by
                # kernels enqueued before the map writes above.
                rows.release_to(target)
                self._pool_rounds(timeout, stats)
        stats.update(
            rows_from=float(old),
            rows_to=float(target),
            mapped_before=float(mapped_before),
            mapped_after=float(rows.mapped_bytes),
            ms=(time.perf_counter() - start) * 1e3,
        )
        self.move_log.append(dict(stats))
        if runtime_debug.on("tiers"):
            runtime_debug.emit(
                "tiers",
                logger,
                "%s: card rows %d -> %d: %d commits, %d onto the card, %d off "
                "it, %d moved down, %d units read, %.1f ms",
                self.layer_name,
                old,
                target,
                int(stats["commits"]),
                int(stats["to_card"]),
                int(stats["off_card"]),
                int(stats["relocated"]),
                int(stats["units_read"]),
                stats["ms"],
            )
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
            key: _u8(getattr(fast, impl_field))[: self.num_fast].cpu()
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
            self._topk = int(topk)
            self._impl_common = dict(common)
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
            if self._fast_capacity:
                self._adopt_fast_rows(fast)
            self._after_impls()
        return self._impls

    def _adopt_fast_rows(self, fast: Any) -> None:
        """Move the card tier's rows into a device range reserved for
        ``fast_rows_max`` rows (``FastRows``) and point the fast kernel at the
        whole range: its ``num_experts`` is the capacity from here on, so the
        graphs captured later stay valid whatever the number of rows in use.
        Rows above the ones in use have no memory behind them and no map entry
        points at them; the kernels read an expert's weights only for experts
        with routed rows. The tensors keep their identity (their ``data`` is
        rebound), so the layer's parameters and the kernel see the new rows."""
        from vllm.model_executor.layers.fused_moe import xpu_fast_tier
        from vllm.model_executor.layers.fused_moe.xpu_fast_tier import FastRows

        assert self.store_binding is not None, "a movable card tier needs the store"
        cfg = self.store_binding.session.config
        tensors = [getattr(fast, impl_field) for impl_field, _ in _TIER_FIELDS]
        expected = sum(1 for t in live_tiers() if t._fast_capacity)
        early, self._early_rows = self._early_rows, None
        if early is not None and early.holds(tensors):
            # Put in the range at construction (reserve_card_rows) and kept
            # there by loading and post-load processing: point the kernel at
            # the capacity, nothing to copy or free.
            self._point_at_capacity(fast, tensors, early)
            xpu_fast_tier.moves_done(expected, early.buffer.device)
            return
        if early is not None:
            # Post-load processing put the rows in new storage: the range
            # reserved at construction goes, the move below makes a new one.
            early.release_to(0)
            early.buffer.free()
            xpu_fast_tier.forget_move(self.layer_name)
        old_storages = xpu_fast_tier.OldStorages(tensors)
        free_before = xpu_fast_tier.free_before_move(tensors[0].device)
        rows = FastRows(tensors, self._fast_capacity, int(cfg.fast_chunk_mib) << 20)
        rows.commit(self.num_fast)
        views = rows.views(tensors)
        old = {}
        for tensor, view in zip(tensors, views):
            view[: self.num_fast].copy_(tensor)
            old[tensor.data_ptr()] = view
        layer = None if self._layer_ref is None else self._layer_ref()
        others = [] if layer is None else list(layer.parameters(recurse=False))
        for tensor in [*tensors, *others]:
            view = old.get(tensor.data_ptr())
            if view is not None and tensor.shape[0] == self.num_fast:
                rows_u8 = view.view(torch.uint8).view(view.shape[0], -1)
                tensor.data = rows_u8.view(tensor.dtype).view(
                    view.shape[0], *tensor.shape[1:]
                )
        fast.num_experts = self._fast_capacity
        fast.local_experts_num = self._fast_capacity
        _device_sync()
        del tensor, view, views, old, others
        # The rows' old storage goes back to the device.
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        self.fast_rows = rows
        # Every old storage must be gone now; one another tensor still holds
        # is released in place (else it stays on the device for good).
        xpu_fast_tier.finish_move(
            self.layer_name,
            old_storages,
            rows.mapped_bytes,
            rows.buffer.device,
            free_before,
            expected,
        )
        self._log_range(rows)

    def _point_at_capacity(self, fast: Any, tensors: list[Any], rows: Any) -> None:
        """The kernel tensors (and the layer's parameters on the same rows)
        rebound to views of the whole reserved range, the fast kernel built
        over the capacity."""
        views = {
            t.data_ptr(): v
            for t, v in zip(tensors, rows.views_as(tensors, self._fast_capacity))
        }
        layer = None if self._layer_ref is None else self._layer_ref()
        others = [] if layer is None else list(layer.parameters(recurse=False))
        for tensor in [*tensors, *others]:
            view = views.get(tensor.data_ptr())
            if view is not None and tensor.shape[0] == self.num_fast:
                rows_u8 = view.view(torch.uint8).view(view.shape[0], -1)
                tensor.data = rows_u8.view(tensor.dtype).view(
                    view.shape[0], *tensor.shape[1:]
                )
        fast.num_experts = self._fast_capacity
        fast.local_experts_num = self._fast_capacity
        self.fast_rows = rows
        self._log_range(rows)

    def reserve_card_rows(self, layer: torch.nn.Module) -> None:
        """With ``fast_rows_max``: put the card rows' parameters into their
        reserved range at construction, before the weights load, so that
        their bytes never live in torch's allocator: freed there after
        loading they end in segments shared with tensors that live on, which
        torch cannot give back and the KV cache would pay for. Loading and
        post-load processing then write into the range (the weights keep
        their storage through both); `_adopt_fast_rows` finds them there.
        Parameters it cannot place (another set of names or row sizes) are
        left to the move at weight processing."""
        from vllm.model_executor.layers.fused_moe import xpu_fast_tier
        from vllm.model_executor.layers.fused_moe.xpu_fast_tier import FastRows

        if not self._fast_capacity or self.store_binding is None:
            return
        params: list[torch.nn.Parameter] = []
        for names in _RESERVE_PARAMS:
            found = [
                p
                for p in (getattr(layer, n, None) for n in names)
                if isinstance(p, torch.nn.Parameter)
                and not p.is_meta
                and p.ndim > 0
                and p.shape[0] == self.num_fast
                and p.is_contiguous()
            ]
            if not found:
                return
            params.append(found[0])
        device = params[0].device
        if device.type not in ("cuda", "xpu") and FastRows.device_override is None:
            return
        cfg = self.store_binding.session.config
        old = xpu_fast_tier.OldStorages(params)
        free_before = xpu_fast_tier.free_before_move(device)
        rows = FastRows(params, self._fast_capacity, int(cfg.fast_chunk_mib) << 20)
        rows.commit(self.num_fast)
        for param, view in zip(params, rows.views_as(params, self.num_fast)):
            view.copy_(param.data)
            param.data = view
        _device_sync()
        xpu_fast_tier.finish_move(
            self.layer_name,
            old,
            rows.mapped_bytes,
            rows.buffer.device,
            free_before,
            expected_layers=1 << 30,  # the line comes at weight processing
            when="at construction",
        )
        self._early_rows = rows

    def _log_range(self, rows: Any) -> None:
        logger.info_once(
            "Expert tiers: the card tier of each layer sits in a device range "
            "reserved for %d rows (%d in use, %.2f GiB mapped per layer); it "
            "grows and shrinks while serving (runtime control tiers.fast) "
            "between %d and %d rows.",
            self._fast_capacity,
            self.num_fast,
            rows.mapped_bytes / 2**30,
            self.fast_range()[0],
            self._fast_capacity,
        )

    def eplb_weights(self) -> list[TieredExpertWeight]:
        """The tensors the two impls read, as EPLB's per-slot rows."""
        assert self._impls is not None, "the impls are built at weight processing"
        fast, slow = self._impls
        weights: list[TieredExpertWeight] = []
        for impl_field, _ in _TIER_FIELDS:
            parts = [getattr(impl, impl_field) for impl in (fast, slow)]
            # The spare rows beyond the pinned slots are not EPLB's; the host
            # slots have no row (they count as slots without a tensor). A
            # card tier in a reserved range has rows beyond the ones in use.
            pinned = (
                self.num_pinned if self.placement is None else self.placement.num_pinned
            )
            parts[0] = parts[0][: self.num_fast]
            parts[1] = parts[1][:pinned]
            parts = [p.view(p.shape[0], -1) for p in parts]
            if parts[0].dtype != parts[1].dtype:  # packed int4: int8 vs uint8
                parts = [p.view(torch.uint8) for p in parts]
            if self.store_binding is not None:
                # placed by this object, through the expert store
                weight = StoreBackedWeight(
                    parts,
                    self.on_rows_written,
                    self,
                    unbacked=self.num_local - self.num_fast - pinned,
                )
                # Its pinned part holds the rows' host memory (the device
                # view owns the host tensor): a relayout swaps it out.
                self._eplb_parts.append(weakref.ref(weight))
                weights.append(weight)
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
        read_budget: int | None = None,
        min_gain: float | None = None,
    ) -> "PreparedMove":
        """Plan the move to the experts ``new_logical`` (per local slot) with
        the tier choice for ``slot_load`` (the load of the expert each slot
        will hold), and read what it needs into free pool rows (from the CPU
        tier's host cache where it holds the expert, else the store), at most
        ``read_budget`` experts (default: the spare rows), each tier swap
        passing the gain test ``min_gain`` (EPLB's tier pass). Runs on any
        thread; touches no device state. Commit the result with
        :meth:`commit_move` between two steps, on the thread that enqueues
        the forward."""
        _, placement, _ = self._require_store()
        assert self.slot_logical is not None
        t0 = time.perf_counter()
        plan = placement.plan(
            self.slot_logical,
            new_logical,
            slot_load,
            hysteresis,
            read_budget=read_budget,
            **({"min_gain": min_gain} if min_gain else {}),
        )
        move = placement.prepare(plan, self._fill_rows_cached, timeout=timeout)
        self._prepared_load = np.asarray(slot_load, dtype=np.float64).copy()
        move.stats = {
            "units": float(len(plan.reads)),
            "bytes": float(len(plan.reads) * self.slot_bytes()),
            "card_bytes": float(len(plan.to_fast) * self.slot_bytes()),
            "changed": float(len(plan.changed)),
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
        self.moves_committed += 1
        plan = move.plan
        if self._prepared_load is not None:
            self._slot_load, self._prepared_load = self._prepared_load, None
            if self._prepared_mean:
                self._load_mean, self._prepared_mean = self._prepared_mean, None
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
        if runtime_debug.on("tiers"):
            tier = placement.tier
            runtime_debug.emit(
                "tiers",
                logger,
                "%s: commit: %d experts changed, %d onto the card, %d off it, "
                "%d units read; card %d, pinned %d, CPU %d",
                self.layer_name,
                len(plan.changed),
                len(plan.to_fast),
                len(plan.demoted),
                len(plan.reads),
                int((tier == 0).sum()),
                int((tier == 1).sum()),
                int((tier == 2).sum()),
            )

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
        read_budget: int | None = None,
        min_gain: float | None = None,
    ) -> "PreparedMove":
        """EPLB's mover: the move from ``old_phys2log`` to ``new_phys2log``
        with the tiers cut from ``logical_load`` (its window). With the two
        maps equal it is a tier pass: no expert of the rank changes, the
        slots move between the card, the pinned rows and the CPU, at most
        ``read_budget`` experts read, each swap passing the gain test
        ``min_gain`` (``EPLBConfig.tier_min_gain``)."""
        old_logical, _ = self._local_view(old_phys2log, logical_load)
        new_logical, slot_load = self._local_view(new_phys2log, logical_load)
        assert self.slot_logical is not None
        if not np.array_equal(old_logical, self.slot_logical):
            raise RuntimeError(
                f"{self.layer_name}: EPLB's placement and the tiers' disagree "
                f"({old_logical.tolist()} vs {self.slot_logical.tolist()})"
            )
        # From here until the commit or the discard the placement is EPLB's:
        # a change of the pinned bound waits.
        with self._move_lock:
            self._eplb_pending = True
            self.tier_pass_wanted = False  # this move plans with the capacity
            load = np.asarray(logical_load, dtype=np.float64)
            self._prepared_mean = float(load.mean()) if load.size else None
            try:
                return self.prepare_move(
                    new_logical,
                    slot_load,
                    hysteresis,
                    read_budget=read_budget,
                    min_gain=min_gain,
                )
            except BaseException:
                self._eplb_pending = False
                raise

    def tier_changes_wanted(
        self,
        phys2log: np.ndarray,
        logical_load: np.ndarray,
        hysteresis: float,
        min_gain: float | None,
        budget: int | None,
    ) -> int:
        """About how many of this rank's slots a tier pass would move now,
        by the window ``logical_load`` (``TierPlacement.tier_changes_wanted``);
        0 for a layer without the store's placement."""
        if self.placement is None or self.store_binding is None:
            return 0
        _, slot_load = self._local_view(phys2log, logical_load)
        return self.placement.tier_changes_wanted(
            slot_load, hysteresis, min_gain, budget
        )

    def commit(self, prepared: "PreparedMove") -> None:
        try:
            self.commit_move(prepared)
        finally:
            self._eplb_pending = False
        if self._release_rows_to is not None:
            self.release_pinned_rows_if_due()

    def discard(self, prepared: "PreparedMove") -> None:
        """A prepared move that will not be committed: its pool rows were only
        written, never mapped, so they are free at once."""
        self._eplb_pending = False
        self._prepared_load = None
        self._prepared_mean = None
        _, placement, _ = self._require_store()
        placement.pool.give_back_unused(prepared.pool_dest.values())

    @staticmethod
    def prepared_stats(prepared: "PreparedMove") -> dict[str, float]:
        return {
            k: prepared.stats[k]
            for k in ("units", "bytes", "card_bytes", "changed", "prepare_ms")
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
        mean = np.asarray(logical_load, dtype=np.float64)
        self._load_mean = float(mean.mean()) if mean.size else None
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
                # the rows in use: a reserved range has no memory above them
                whole[(FAST, key)] = _u8(getattr(fast, impl_field))[
                    : placement.num_fast
                ].cpu()
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


def _device_sync() -> None:
    """Wait for the accelerator (nothing to wait for without one)."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def _empty_host_cache() -> None:
    """Give the pinned host memory torch caches and no tensor holds back to
    the system (torch.accelerator.empty_host_cache, where torch has it)."""
    empty = getattr(torch.accelerator, "empty_host_cache", None)
    if empty is not None and torch.accelerator.is_available():
        empty()


def _quarantined(placement: "TierPlacement") -> int:
    placement.pool.free_count  # noqa: B018 - reaps the rows whose event is done
    return placement.pool.quarantined


def _forget_staged_slow() -> None:
    """Drop the stager's device copies of the slow tier: their shape was the
    old row count (a new one is made for the next large step)."""
    from vllm.model_executor.offloader.base import get_offloader

    stager = getattr(get_offloader(), "expert_stager", None)
    for name in ("_buffers", "_rings"):
        table = getattr(stager, name, None)
        if not isinstance(table, dict):
            continue
        for key in [k for k in table if str(k[0]).startswith("slow.")]:
            del table[key]


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
    w13_out = quant_method.moe.w13_num_shards * intermediate_size_per_partition
    group_size, _ = _int4_geometry(quant_method)
    groups13 = groups2 = 1
    if group_size != -1:
        groups13 = hidden_size // group_size
        groups2 = intermediate_size_per_partition // group_size
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
    k_first: bool = True,
) -> dict[str, torch.Tensor]:
    """One expert's tensors as the checkpoint holds them (``w13`` already
    stacked over its shards) as the four tensors of the expert's store unit,
    computed on the host: the bytes the tiers write at the first load, for a
    slot of either tier. For building a store outside the engine (the unit
    source of the store's build tool).

    ``k_first``: GPTQ, int32 ``[in/8, out]`` qweight and ``[groups, out]``
    scales; else compressed-tensors pack-quantized, ``[out, in/8]`` and
    ``[out, groups]``, which are already the kernels' axis order."""
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    def order(t: torch.Tensor) -> torch.Tensor:
        t = t.cpu()
        return (t.transpose(0, 1) if k_first else t).contiguous()

    unit: dict[str, torch.Tensor] = {}
    scales = {"w13": w13_scales, "w2": w2_scales}
    for key, qweight in (("w13", w13_qweight), ("w2", w2_qweight)):
        unit[key] = implement_zp(order(qweight).view(torch.uint8))
    for key in ("w13", "w2"):
        unit[f"{key}_scales"] = order(scales[key]).to(scales_dtype)
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


def tier_expert_format(quant_method: Any) -> tuple[ExpertFormat | None, str]:
    """(the format, "") when the tiers can hold this method's experts, else
    (None, why not).

    1. GPTQ / AutoRound int4 (``AutoGPTQMoEMethod``, 4 bits): as before.
    2. compressed-tensors ``pack-quantized`` int4 (``CompressedTensorsWNA16MoEMethod``
       on the XPU backend; also what a checkpoint that declares int8 per-token
       activations, or the ``int8_per_token`` serve-time setting, runs as):
       symmetric, group strategy, no expert bias. Its words are the GPTQ words
       transposed (``[out, in/8]``, value + 8, low nibble first), so after the
       method's post-load step the card rows hold the same bytes as for GPTQ,
       and the pinned rows, the store and the CPU tier take them unchanged.
    """
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

    if isinstance(quant_method, AutoGPTQMoEMethod):
        if quant_method.quant_config.weight_bits != 4:
            return None, f"{quant_method.quant_config.weight_bits}-bit weights"
        return GPTQ_INT4, ""
    try:
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
            CompressedTensorsWNA16MoEMethod,
        )
    except ImportError:  # compressed-tensors is not installed
        return None, ""
    if not isinstance(quant_method, CompressedTensorsWNA16MoEMethod):
        return None, ""
    why = None
    if quant_method.num_bits != 4:
        why = f"{quant_method.num_bits}-bit weights"
    elif not quant_method.symmetric:
        why = "asymmetric weights (zero points)"
    elif quant_method.strategy != "group" or int(quant_method.group_size) <= 0:
        why = f"strategy {quant_method.strategy}, not group"
    elif quant_method.wna16_backend != WNA16MoEBackend.XPU:
        why = f"backend {quant_method.wna16_backend}, not XPU"
    elif getattr(quant_method.moe, "has_bias", False):
        why = "experts with a bias"
    if why is not None:
        return None, why
    return CT_PACKED_INT4, ""


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
    expert_format, refusal = tier_expert_format(quant_method)
    if expert_format is None:
        logger.warning_once(
            "--cpu-offload-per-expert supports int4 GPTQ and compressed-tensors "
            "pack-quantized int4 experts only; %s experts stay in VRAM%s",
            type(quant_method).__name__,
            f" ({refusal})" if refusal else "",
        )
        return None
    if expert_format is not GPTQ_INT4:
        logger.info_once(
            "--cpu-offload-per-expert: %s int4 experts (group %d) enter the "
            "expert tiers",
            expert_format.name,
            _int4_geometry(quant_method)[0],
        )
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
        expert_format=expert_format,
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
