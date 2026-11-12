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
"""

import time
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
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        for i, used in enumerate(self.used):
            if used + nbytes <= self.chunks[i].numel():
                break
        else:
            size = (
                self.CHUNK if nbytes <= self.CHUNK else 1 << (nbytes - 1).bit_length()
            )
            self.chunks.append(torch.empty(size, dtype=torch.uint8, pin_memory=True))
            self.used.append(0)
            self.total += size
            i, used = len(self.chunks) - 1, 0
        self.used[i] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        return self.chunks[i][used : used + nbytes].view(dtype).view(shape)

    @property
    def gib(self) -> float:
        return self.total / 2**30


_ARENA = _PinnedArena()


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
    fast_map = torch.full((num_global,), -1, dtype=torch.int32)
    slow_map = torch.full((num_global,), -1, dtype=torch.int32)
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
    ):
        assert 0 < num_fast < num_local
        self.layer_name = layer_name
        self.local_of = local_of  # physical expert id -> local slot, -1 elsewhere
        self.num_global = num_global
        self.num_local = num_local
        self.num_fast = num_fast
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
    def num_pool_rows(self) -> int:
        """Rows of each pinned tensor: the host tier's slots and the spares."""
        spare = 0 if self.store_binding is None else self.store_binding.spare_rows
        return self.num_slow + spare

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
                tensor[self.num_slow :].zero_()
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
            )
            binding.session.note_pool(
                self.num_pool_rows, self.slot_bytes(), self.num_slow
            )
        experts.tiers = self

    def _check_views_alias(self) -> None:
        """With a store, host threads write the pool rows into the pinned
        tensors and the kernels read them through the device views: the two
        must be the same memory. A view made over a copy (memory the
        accelerator does not take as pinned is copied first) would serve the
        load-time bytes for ever and none of the store's reads. One byte of
        a spare row per tensor settles it at load."""
        last = self.num_pool_rows - 1
        if last < self.num_slow:
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
        for slot in range(self.num_fast, self.num_local):
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
            # The spare rows beyond the host tier's slots are not EPLB's.
            parts[1] = parts[1][: self.num_slow]
            parts = [p.view(p.shape[0], -1) for p in parts]
            if parts[0].dtype != parts[1].dtype:  # packed int4: int8 vs uint8
                parts = [p.view(torch.uint8) for p in parts]
            if self.store_binding is not None:
                # placed by this object, through the expert store
                weights.append(StoreBackedWeight(parts, self.on_rows_written, self))
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
        move.stats = {
            "units": float(len(plan.reads)),
            "bytes": float(len(plan.reads) * self.slot_bytes()),
            "prepare_ms": (time.perf_counter() - t0) * 1e3,
        }
        return move

    def commit_move(self, move: "PreparedMove") -> None:
        """Make a prepared move the placement: enqueue its copies and the map
        writes (see :meth:`TierPlacement.commit` for the stream order)."""
        _, placement, ops = self._require_store()
        t0 = time.perf_counter()
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
        plan = move.plan
        self._touched.update(plan.changed, plan.demoted, plan.to_fast)
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
        return self.prepare_move(new_logical, slot_load, hysteresis)

    def commit(self, prepared: "PreparedMove") -> None:
        self.commit_move(prepared)

    def discard(self, prepared: "PreparedMove") -> None:
        """A prepared move that will not be committed: its pool rows were only
        written, never mapped, so they are free at once."""
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

        from vllm.model_executor.offloader.expert_tier_map import FAST, POOL

        binding, placement, ops = self._require_store()
        spare_rows = list(range(self.num_slow, self.num_pool_rows))
        assert spare_rows, "filling the fast rows stages them through spare rows"
        logical, load = self._local_view(phys2log, logical_load)
        is_fast = np.zeros(self.num_local, dtype=bool)
        is_fast[np.argsort(-load, kind="stable")[: self.num_fast]] = True
        if np.array_equal(logical, self.slot_logical) and np.array_equal(
            is_fast, placement.is_fast()
        ):
            return  # the rows already hold this placement
        tier = np.where(is_fast, FAST, POOL).astype(np.int8)
        row = np.zeros(self.num_local, dtype=np.int32)
        row[is_fast] = np.arange(self.num_fast)
        row[~is_fast] = np.arange(self.num_slow)
        start = _time.perf_counter()
        pool_slots = np.flatnonzero(~is_fast).tolist()
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
        self.on_rows_written([])
        self.move_log.append(
            {
                "fill_ms": (_time.perf_counter() - start) * 1e3,
                "units": float(len(logical)),
            }
        )

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
        mismatches = 0
        for slot in slots:
            tier, row = int(placement.tier[slot]), int(placement.row[slot])
            tensors = {}
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


def _layout_fingerprint() -> str:
    """What the kernels' int4 transform does to every byte value. It is part
    of the store's identity: a kernels build whose transform differs sees
    another identity, whatever its version string says. A change of the row
    layout that this probe cannot see is caught at load, when a store that
    fills the host tier is compared with the rows this boot laid out
    (``ExpertTiers._check_store_against_this_boot``)."""
    import hashlib

    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    probe = implement_zp(torch.arange(256, dtype=torch.uint8))
    raw = probe.contiguous().view(torch.uint8).numpy().tobytes()
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
    return ExpertTiers(
        layer_name, local_of, num_global, num_local, num_local - num_slow, binding
    )
