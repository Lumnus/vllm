# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache backing allocation that grows after warmup."""

import copy
import gc
from bisect import bisect_right
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload.layerwise import max_layer_reload_bytes
from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory
from vllm.platforms import current_platform
from vllm.utils.extensible_tensor import ExtensibleTensor
from vllm.utils.math_utils import round_up
from vllm.utils.mem_utils import device_memory_info, format_gib
from vllm.v1.hisparse.binding import rebind_hisparse_device_caches
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.attn_utils import bind_kv_caches
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.utils import layer_kernel_block_size, layer_spec_for_tensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Headroom left after sizing from measured memory, beyond the measured transient
# peak, for what the allocated peak does not show: the caching allocator's
# rounding and fragmentation around it, real steps combining shapes warmup
# exercises separately (a prompt-logprobs chunk alongside a live decode, the
# rejection sampler over a full spec-decode batch), and inputs profiling only
# approximates (a multimodal encoder batch of a modality other than the one
# with the largest feature size). The share scales with the model's working
# set; the floor covers small models whose measured peak is a few hundred MiB
# while their encoder or sampler transients are not. Measured on GB200: the
# first prefill of an 8B model needed over 25% beyond its allocated peak before
# the allocator could serve it, and adversarial workloads reached up to 45%.
# ROCr also maps kernel scratch outside the allocator when a kernel first needs
# it, so ROCm keeps a larger floor. On Arc Pro B70 (4 cards, a hybrid MoE at
# tensor parallel 4) the measured transient peak already covers what the
# first requests need: with no margin the cache grew from 765,379 tokens (floor
# 256 MiB, fraction 0.25) to 801,549, and a stress run (16-stream decode, 2K x
# 16 prefill, needles at 128K and 240K) passed with no preemptions and a lower
# peak than the estimate path, so XPU keeps none.
KV_CACHE_MARGIN_FLOOR_BYTES = (
    3 << 30
    if current_platform.is_rocm()
    else 0
    if current_platform.is_xpu()
    else 1 << 30
)
KV_CACHE_MARGIN_FRACTION = 0.0 if current_platform.is_xpu() else 0.75
# Part of the margin kept for allocations outside torch's caching allocator
# (kernel scratch, NCCL/RCCL buffers, lazily loaded code objects); the
# allocator is capped to leave it free. The rest absorbs torch's fragmentation,
# which scales with the transient where these do not: serving grew them by at
# most 0.17 GiB on MI355 and not at all on GB200.
KV_CACHE_NON_TORCH_RESERVE_BYTES = (2 if current_platform.is_rocm() else 1) << 29
# Share of the headroom warmup leaves uncommitted, on top of the profiled
# activation peak, for transients the profiling run does not exercise.
KV_CACHE_WARMUP_RESERVE_FRACTION = 0.1


class ExtensibleKVCache:
    """Growable backing allocation for the KV cache.

    Reserves address space for `kv_cache_config.num_blocks` and commits a
    prefix of blocks on demand. Blocks keep fixed offsets, so views over the
    reservation and captured CUDA graphs stay valid as more are committed. The
    allocation splits into segments holding num_blocks blocks each: one per
    layer for block-compact layer-outermost layouts (sized by that layer's
    block stride, possibly padded apart by `pad_kv_cache_segments`), one
    overall for block-outermost. Host-resident tensors live outside it.

    With `release_chunk_bytes`, blocks are committed in physical objects of at
    most that size (rounded up to the driver granularity), so that `shrink_to`
    can give the tail of the committed prefix back to the device. Without it a
    commit maps one object per contiguous run, as before.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        device: torch.device,
        layout: KVCacheLayout,
        release_chunk_bytes: int | None = None,
    ):
        tensors = [
            tensor
            for tensor in kv_cache_config.kv_cache_tensors
            if not tensor.host_resident
        ]
        assert tensors, "Extensible KV cache requires at least one KV cache tensor."
        sizes = {tensor.size for tensor in tensors}
        assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
        self.size = sizes.pop()
        self.capacity_blocks = kv_cache_config.num_blocks
        if self.capacity_blocks <= 0:
            raise ValueError(f"Invalid KV cache capacity {self.capacity_blocks}.")
        self._is_block_outermost = layout.is_block_outermost
        self.segment_offsets, self.segment_strides = self._segments(tensors, layout)
        self.bytes_per_block = sum(self.segment_strides)
        self.num_committed_blocks = 0
        # The null block (and a second one under block-outermost layouts, see
        # `allocate`) stays committed.
        self.min_committed_blocks = 2 if self._is_block_outermost else 1
        # Memory to keep free while committing before the final sizing, e.g.
        # the profiled activation peak the warmup steps are about to hit.
        self.reserved_headroom_bytes = 0
        # Blocks warmup may commit: the minimum of `committable_blocks()` over
        # ranks when the cache is created.
        self.warmup_committable_blocks = 0
        # Bytes torch's caching allocator may reserve, set once the cache is
        # sized; None leaves it unlimited.
        self.torch_memory_limit: int | None = None
        self.buffer = ExtensibleTensor(
            self.size,
            device=device,
            segment_capacities=[
                end - start
                for start, end in pairwise([*self.segment_offsets, self.size])
            ],
            max_chunk_bytes=release_chunk_bytes,
        )

    def _segments(
        self, tensors: list[KVCacheTensor], layout: KVCacheLayout
    ) -> tuple[list[int], list[int]]:
        """Segment offsets and block strides covering the allocation.

        Block-compact layer-outermost layouts give each layer a num_blocks-block
        region, which may be padded apart. Otherwise the segments are implied by
        the one block stride.
        """
        if not layout.is_block_compact or layout.is_block_outermost:
            block_strides = {tensor.block_stride for tensor in tensors}
            if len(block_strides) != 1:
                raise ValueError(
                    "KV cache tensors with different block strides need a "
                    "block-compact layer-outermost layout for the extensible KV "
                    "cache."
                )
            stride = block_strides.pop()
            segment_bytes = self.capacity_blocks * stride
            if self.size % segment_bytes != 0:
                raise ValueError(
                    f"KV cache allocation of {self.size} bytes is not a whole "
                    f"number of {self.capacity_blocks}-block segments."
                )
            num_segments = self.size // segment_bytes
            return [i * segment_bytes for i in range(num_segments)], [
                stride
            ] * num_segments
        segments: dict[int, int] = {}
        for tensor in tensors:
            for layer_idx in range(len(tensor.layers)):
                offset = tensor.offset + layer_idx * tensor.layer_stride
                stride = segments.setdefault(offset, tensor.block_stride)
                if stride != tensor.block_stride:
                    raise ValueError(
                        f"KV cache tensors alias offset {offset} with different "
                        "block strides."
                    )
        offsets = sorted(segments)
        strides = [segments[offset] for offset in offsets]
        end = offsets[0]
        for offset, stride in zip(offsets, strides):
            if offset < end:
                raise ValueError("KV cache tensors overlap.")
            end = offset + self.capacity_blocks * stride
        if offsets[0] != 0 or end > self.size:
            raise ValueError("KV cache tensors do not tile the allocation.")
        return offsets, strides

    def allocate(self, size: int) -> torch.Tensor:
        """Allocator hook for allocate_kv_cache; commits the null block.

        Under block-outermost layouts layers' views start inside block 0, so a
        second block keeps at least half a granule mapped past each view's start
        (see `pad_kv_cache_segments` for what reads it).
        """
        assert size == self.size, f"Expected {self.size} bytes, got {size}."
        self.commit(2 if self._is_block_outermost else 1)
        return self.buffer.full_view()

    def commit(
        self, num_blocks: int, defragment: bool = False, shrink: bool = False
    ) -> None:
        """Back the first num_blocks blocks; new blocks are zeroed.

        Grow-only unless shrink is set, which remaps a smaller prefix and
        discards contents. defragment remaps each segment as one driver
        allocation (discarding contents), whatever the release chunk size: UCX
        cannot RDMA a range spanning several allocations.
        """
        num_blocks = min(num_blocks, self.capacity_blocks)
        sizes = [num_blocks * stride for stride in self.segment_strides]
        remap = (
            defragment and not all(self.buffer.segments_backed_by_one_chunk(sizes))
        ) or (shrink and num_blocks < self.num_committed_blocks)
        if remap:
            self.buffer.release_physical()
            self.num_committed_blocks = 0
        if num_blocks <= self.num_committed_blocks:
            return
        self.buffer.resize_segments_(sizes, zero_new=True, whole_runs=defragment)
        self.num_committed_blocks = num_blocks
        logger.debug(
            "Committed %d of %d KV cache blocks (%.2f GiB physical).",
            num_blocks,
            self.capacity_blocks,
            self.physical_bytes / (1 << 30),
        )

    def back(self, num_blocks: int) -> None:
        """Map physical memory for the first num_blocks blocks without
        committing them: nothing that reads the committed prefix changes.

        Lets every rank learn that the memory is there before any commits. If
        this raises, some granules may stay mapped; a later `commit` reuses
        them.
        """
        num_blocks = min(num_blocks, self.capacity_blocks)
        if num_blocks <= self.num_committed_blocks:
            return
        self.buffer.back_segments_(
            [num_blocks * stride for stride in self.segment_strides]
        )

    def blocks_releasable(self, num_bytes: int) -> int | None:
        """Fewest tail blocks whose release gives back at least num_bytes of
        physical memory, or None if no `shrink_to` that keeps the minimum
        committed blocks frees that much.

        Counts what `shrink_to` frees, which is whole release chunks: a chunk
        straddling the new size stays, so the answer can be a little above
        num_bytes / bytes_per_block, and without a release chunk size it is
        often None.
        """
        if num_bytes <= 0:
            return 0
        most = self.num_committed_blocks - self.min_committed_blocks
        if most <= 0 or self._releasable_bytes(self.num_committed_blocks - most) < (
            num_bytes
        ):
            return None
        low, high = 1, most
        while low < high:
            mid = (low + high) // 2
            if self._releasable_bytes(self.num_committed_blocks - mid) >= num_bytes:
                high = mid
            else:
                low = mid + 1
        return low

    def _releasable_bytes(self, num_blocks: int) -> int:
        """Physical bytes `shrink_to(num_blocks)` would give back."""
        return self.buffer.releasable_bytes(
            [num_blocks * stride for stride in self.segment_strides]
        )

    def shrink_to(self, num_blocks: int) -> int:
        """Commit only the first num_blocks blocks and give back the physical
        memory of the whole release chunks above them. Returns the bytes
        released.

        Contents of the kept blocks are untouched; a chunk that straddles the
        new size stays mapped, so the bytes freed can be less than a
        block-proportional share. The caller makes sure nothing reads the
        released blocks any more: the views over the committed prefix are
        rebuilt first (`rebind_committed_views`), the scheduler no longer hands
        the blocks out, and the connector has let go of them. Nothing in
        serving calls this yet.
        """
        if not self.min_committed_blocks <= num_blocks <= self.num_committed_blocks:
            raise ValueError(
                f"Cannot shrink the KV cache from {self.num_committed_blocks} to "
                f"{num_blocks} blocks (minimum {self.min_committed_blocks})."
            )
        if num_blocks == self.num_committed_blocks:
            return 0
        released = self.buffer.shrink_segments_(
            [num_blocks * stride for stride in self.segment_strides]
        )
        self.num_committed_blocks = num_blocks
        logger.debug(
            "Shrank the KV cache to %d of %d blocks, released %.2f GiB "
            "(%.2f GiB physical).",
            num_blocks,
            self.capacity_blocks,
            released / (1 << 30),
            self.physical_bytes / (1 << 30),
        )
        return released

    def committed_views(
        self,
        kv_cache_config: KVCacheConfig,
        layout: KVCacheLayout,
        kernel_block_sizes: list[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Per-layer views whose storage spans exactly the committed blocks.

        `kv_cache_config.num_blocks` must equal the committed count. Connectors
        derive extents from `untyped_storage().nbytes()`, so they must not see the
        reservation. Host-resident layers are omitted. None for layouts whose
        layers span several segments.
        """
        if not layout.is_block_compact:
            return None
        assert kv_cache_config.num_blocks == self.num_committed_blocks
        kv_caches: dict[str, torch.Tensor] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            if tensor.host_resident:
                continue
            group_id, spec = layer_spec_for_tensor(kv_cache_config, tensor)
            if not spec.has_layer_views:
                # Laid out by the layer itself over the raw buffer, as at
                # allocation, now over just the committed blocks.
                assert layout.is_block_outermost
                raw_view = self.buffer.segment_view(0)
                kv_caches.update((name, raw_view) for name in tensor.layers)
                continue
            kernel_block_size = layer_kernel_block_size(group_id, kernel_block_sizes)
            for layer_idx, layer_name in enumerate(tensor.layers):
                segment, offset = self._locate(
                    tensor.offset + layer_idx * tensor.layer_stride
                )
                (view,) = create_kv_cache_views(
                    self.buffer.segment_view(segment),
                    spec,
                    kv_cache_config.num_blocks,
                    layout,
                    replace(tensor, layers=[layer_name], offset=offset),
                    kernel_block_size=kernel_block_size,
                )
                kv_caches[layer_name] = view
        return kv_caches

    def committable_blocks(self) -> int:
        """Blocks that can be committed now without eating into the headroom.

        Bounds what warmup may commit: free memory plus the committed prefix,
        less the sizing margin and the larger of reserved_headroom_bytes and
        a share of the headroom, never below what is already committed nor
        above the capacity.
        """
        free_memory, _ = device_memory_info(self.buffer.device)
        headroom = free_memory + self.physical_bytes
        reserve = max(
            self.reserved_headroom_bytes,
            int(headroom * KV_CACHE_WARMUP_RESERVE_FRACTION),
        )
        margin = max(
            KV_CACHE_MARGIN_FLOOR_BYTES, int(reserve * KV_CACHE_MARGIN_FRACTION)
        )
        budget = headroom - margin - reserve
        return max(self.blocks_within(budget), self.num_committed_blocks)

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes, including granule rounding."""
        return self.buffer.physical_bytes

    def physical_bytes_for(self, num_blocks: int) -> int:
        """Bytes a commit of num_blocks maps, each segment rounded to granules."""
        granule = self.buffer.granularity
        return sum(
            -(-num_blocks * stride // granule) * granule
            for stride in self.segment_strides
        )

    def blocks_within(self, num_bytes: int) -> int:
        """Most blocks whose physical footprint fits in num_bytes."""
        num_blocks = min(num_bytes // self.bytes_per_block, self.capacity_blocks)
        while num_blocks > 0 and self.physical_bytes_for(num_blocks) > num_bytes:
            num_blocks -= 1
        return num_blocks

    def _locate(self, start: int) -> tuple[int, int]:
        """Segment index and offset within it of allocation byte start."""
        segment = bisect_right(self.segment_offsets, start) - 1
        return segment, start - self.segment_offsets[segment]

    def committed_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, layout: KVCacheLayout, num_blocks: int
    ) -> list[KVCacheTensor]:
        """Tensor placements matching committed_views for num_blocks.

        size is the committed total, as every device tensor describes the one
        allocation; host-resident tensors are unchanged. Under block-outermost
        layouts the committed blocks are one prefix of it, so the capacity
        placements still hold. Under layer-outermost layouts each layer's view
        has its own storage over its committed prefix: describe it as a
        one-layer tensor placed at the start of that storage.
        """
        size = num_blocks * self.bytes_per_block
        tensors = [
            tensor if tensor.host_resident else replace(tensor, size=size)
            for tensor in kv_cache_config.kv_cache_tensors
        ]
        if layout.is_block_outermost:
            return tensors
        committed: list[KVCacheTensor] = []
        for tensor in tensors:
            if tensor.host_resident:
                committed.append(tensor)
                continue
            _, spec = layer_spec_for_tensor(kv_cache_config, tensor)
            for layer_idx, layer_name in enumerate(tensor.layers):
                start = tensor.offset + layer_idx * tensor.layer_stride
                segment, offset = self._locate(start)
                if not spec.has_layer_views:
                    # The layer keeps the whole reservation as its storage.
                    offset = start
                committed.append(
                    KVCacheTensor(
                        size=size,
                        layers=[layer_name],
                        layer_stride=num_blocks * tensor.block_stride,
                        block_stride=tensor.block_stride,
                        offset=offset,
                    )
                )
        return committed

    def release_physical(self) -> None:
        """Drop physical pages for sleep; addresses and views stay valid."""
        self.buffer.release_physical()

    def recommit(self) -> None:
        """Map fresh zeroed pages for the blocks committed before release."""
        num_blocks = self.num_committed_blocks
        self.num_committed_blocks = 0
        self.commit(num_blocks)

    def set_torch_memory_limit(self, num_bytes: int | None) -> None:
        """Cap what torch's caching allocator reserves on this device, or lift
        the cap (None). At the cap it releases cached segments before growing,
        rather than taking memory other allocations need."""
        self.torch_memory_limit = num_bytes
        total = torch.accelerator.get_memory_info(self.buffer.device)[1]
        fraction = 1.0 if num_bytes is None else min(num_bytes / total, 1.0)
        # torch.accelerator has no per-process memory fraction.
        device = self.buffer.device
        torch.get_device_module(device).set_per_process_memory_fraction(
            fraction, device
        )

    def free(self) -> None:
        # The process may start another engine, which must not inherit the cap.
        if self.torch_memory_limit is not None:
            self.set_torch_memory_limit(None)
        self.buffer.free()
        self.num_committed_blocks = 0


def registers_kv_cache_memory(vllm_config: "VllmConfig") -> bool:
    """Whether a KV connector registers the committed KV cache for RDMA, which
    needs each segment backed by one physical chunk."""
    kv_transfer_config = vllm_config.kv_transfer_config
    return (
        kv_transfer_config is not None
        and kv_transfer_config.registers_kv_cache_memory()
    )


def release_chunk_bytes(vllm_config: "VllmConfig") -> int | None:
    """Bytes per physical allocation when committing the extensible KV cache in
    release chunks, or None for one allocation per contiguous run.

    A connector that registers the cache for RDMA needs each segment backed by
    one allocation, which chunks rule out, so the setting is not applied there.
    """
    chunk_mib = vllm_config.cache_config.kv_cache_release_chunk_mib
    if chunk_mib is None:
        return None
    if registers_kv_cache_memory(vllm_config):
        logger.warning_once(
            "kv_cache_release_chunk_mib is ignored: the KV connector registers "
            "the KV cache memory, which needs one allocation per segment."
        )
        return None
    return chunk_mib << 20


def pad_kv_cache_segments(
    kv_cache_config: KVCacheConfig,
    layout: KVCacheLayout,
    granule: int,
    *,
    registers_memory: bool = False,
) -> KVCacheConfig:
    """Start every layer's region on a granule.

    Committing the null block then backs each layer's first granule, which
    TMA gathers with out-of-bounds indices touch beyond the rows they address.
    It also lets a defragmenting commit of any block count back each segment
    with one physical chunk, as RDMA registration (registers_memory) requires.

    Only address space grows: commits already map whole granules per segment.
    Block-outermost layouts (one segment at offset 0, see `allocate`) are
    returned as is, and so are other layouts that are not block-compact, unless
    registers_memory.
    """
    if layout.is_block_outermost:
        return kv_cache_config
    if not layout.is_block_compact:
        assert not registers_memory, f"{layout.name} has no per-layer segments."
        return kv_cache_config
    num_blocks = kv_cache_config.num_blocks
    tensors = kv_cache_config.kv_cache_tensors
    assert all(t.layer_stride == num_blocks * t.block_stride for t in tensors)
    # Cache groups alias each other's layer regions.
    regions = {
        tensor.offset + layer_idx * tensor.layer_stride: tensor.layer_stride
        for tensor in tensors
        for layer_idx in range(len(tensor.layers))
    }
    padded_offsets: dict[int, int] = {}
    end = padded_end = 0
    for offset in sorted(regions):
        assert offset == end, "KV cache tensors do not tile the allocation."
        padded_offsets[offset] = padded_end
        end += regions[offset]
        padded_end += round_up(regions[offset], granule)
    padded = copy.copy(kv_cache_config)
    padded.kv_cache_tensors = [
        replace(
            tensor,
            size=padded_end,
            offset=padded_offsets[tensor.offset],
            layer_stride=round_up(tensor.layer_stride, granule),
        )
        for tensor in tensors
    ]
    return padded


def num_committable_kv_blocks(runner: "GPUModelRunner") -> int:
    """Blocks warmup may address: all of them, or, for an extensible cache,
    the count agreed across ranks so that every rank warms up the same shapes."""
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        return kv_cache.warmup_committable_blocks
    return runner.kv_cache_config.num_blocks


def ensure_kv_cache_blocks(runner: "GPUModelRunner", num_blocks: int) -> None:
    """Commit enough of an extensible KV cache to address num_blocks."""
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        kv_cache.commit(num_blocks)


def rebind_committed_views(runner: "GPUModelRunner") -> None:
    """Point the runner at views over the committed blocks of its extensible
    KV cache, the count in `runner.kv_cache_config.num_blocks`.

    The views have the addresses of the allocation they replace, so graphs
    captured against it stay valid. The KV connector is not touched.
    """
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    forward_context = runner.compilation_config.static_forward_context
    committed = kv_cache.committed_views(
        runner.kv_cache_config,
        runner.cache_config.get_resolved_kv_cache_layout(),
        runner.kernel_block_sizes,
    )
    if committed is not None:
        if runner.vllm_config.attention_config.hisparse_config is not None:
            rebind_hisparse_device_caches(
                forward_context=forward_context,
                kv_cache_config=runner.kv_cache_config,
                kv_caches=committed,
            )
        # As in `initialize_kv_cache`, draft groups map their views too.
        attn_groups = [
            group
            for groups in runner.attn_groups
            + getattr(runner.speculator, "attn_groups", [])
            for group in groups
        ]
        bind_kv_caches(
            committed,
            forward_context,
            runner.kv_cache_config,
            runner.vllm_config,
            attn_groups,
        )
        runner.kv_caches_by_layer = {
            name: committed.get(name, view)
            for name, view in runner.kv_caches_by_layer.items()
        }
        runner.kv_caches = [
            cache
            for cache in runner.kv_caches_by_layer.values()
            if cache.device == runner.device
        ]
        if runner.vllm_config.cache_config.use_replayssm:
            from vllm.v1.worker.replayssm_utils import (
                get_replayssm_block_copy_tensors,
                initialize_replayssm_ring_trackers,
            )

            initialize_replayssm_ring_trackers(runner.kv_cache_config, forward_context)
            runner.replayssm_block_copy_tensors = get_replayssm_block_copy_tensors(
                forward_context
            )


def extend_kv_cache(
    runner: "GPUModelRunner", kv_cache_config: KVCacheConfig, num_blocks: int
) -> None:
    """Commit the final size once post-warmup memory is known.

    Afterwards the runner matches one that allocated num_blocks up front:
    views are rebuilt over the committed bytes (same addresses) and the KV
    connector is created against final memory. kv_cache_config is the
    engine's config the cache was allocated from.
    """
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    kv_cache_config = copy.copy(kv_cache_config)
    kv_cache_config.num_blocks = num_blocks
    # Connectors that derive geometry from the config must see the committed
    # storages the rebuilt views will have.
    kv_cache_config.kv_cache_tensors = kv_cache.committed_kv_cache_tensors(
        kv_cache_config,
        runner.cache_config.get_resolved_kv_cache_layout(),
        num_blocks,
    )
    ensure_kv_transfer_initialized(runner.vllm_config, kv_cache_config)
    # Config resolution rejects `custom_mem_pool`; a connector that wants the
    # KV cache in its own pool some other way cannot have it either.
    if (
        has_kv_transfer_group()
        and get_kv_transfer_group().get_mem_pool_context() is not None
    ):
        raise ValueError(
            "The KV connector allocates the KV cache from its own memory pool, "
            "which the extensible KV cache cannot use. Disable it with "
            "--no-enable-extensible-kv-cache."
        )
    if num_blocks < kv_cache.num_committed_blocks:
        # Warmup committed more than the measured budget fits (its transient
        # peak exceeded the reserve); the contents are warmup garbage.
        logger.info(
            "Remapping the KV cache from %d to %d blocks to fit the measured budget.",
            kv_cache.num_committed_blocks,
            num_blocks,
        )
    kv_cache.commit(
        num_blocks,
        defragment=registers_kv_cache_memory(runner.vllm_config),
        shrink=True,
    )
    runner.kv_cache_config.num_blocks = kv_cache.num_committed_blocks
    logger.info(
        "Extended KV cache to %d blocks (%.2f GiB).",
        kv_cache.num_committed_blocks,
        kv_cache.physical_bytes / (1 << 30),
    )
    rebind_committed_views(runner)
    runner.kv_connector = get_kv_connector(
        runner.vllm_config, runner.kv_caches_by_layer
    )


def measure_kv_cache_bytes(
    *,
    init_free_memory: int,
    free_memory: int,
    committed_bytes: int,
    requested_memory: int,
    transient_peak_bytes: int = 0,
    margin_floor_bytes: int = KV_CACHE_MARGIN_FLOOR_BYTES,
    margin_fraction: float = KV_CACHE_MARGIN_FRACTION,
) -> int:
    """Bytes the KV cache may occupy given the memory measured after warmup.

    Everything resident except the committed KV prefix is needed by the engine;
    the cache gets the rest of the budget, capped by the device headroom (free
    memory plus the committed prefix) less a margin of
    `max(margin_floor_bytes, margin_fraction * transient_peak_bytes)`. The
    measured transient_peak_bytes are then kept free in full. The margin
    guards physical memory, so it never reduces an explicit budget that already
    leaves that much free.
    """
    non_kv_used = init_free_memory - free_memory - committed_bytes
    headroom = free_memory + committed_bytes
    margin = max(margin_floor_bytes, int(transient_peak_bytes * margin_fraction))
    available = min(requested_memory - non_kv_used, headroom - margin)
    available -= transient_peak_bytes
    return max(available, 0)


class _KVCacheMeasurement(NamedTuple):
    free_memory: int
    transient_peak: int
    available: int
    kv_cache_bytes: int
    num_blocks: int


def _measure_kv_cache(
    kv_cache: ExtensibleKVCache,
    vllm_config: "VllmConfig",
    device: torch.device,
    *,
    init_free_memory: int,
    requested_memory: int,
) -> _KVCacheMeasurement:
    """Blocks the memory free now fits, by the rule of `measure_kv_cache_bytes`,
    keeping free reserved_headroom_bytes, the larger of the profiled peak and
    warmup's allocated (not reserved) transient peak. Torch's cache is emptied
    first: its blocks may not fit the large workspaces the first real steps
    allocate."""
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    free_memory = current_platform.free_device_memory(device)
    transient_peak = kv_cache.reserved_headroom_bytes
    available = measure_kv_cache_bytes(
        init_free_memory=init_free_memory,
        free_memory=free_memory,
        committed_bytes=kv_cache.physical_bytes,
        requested_memory=requested_memory,
        transient_peak_bytes=transient_peak,
    )
    kv_cache_bytes = reserve_mm_ipc_gpu_memory(
        available,
        vllm_config.model_config.multimodal_config,
        getattr(vllm_config.parallel_config, "_api_process_count", 1),
    )
    return _KVCacheMeasurement(
        free_memory,
        transient_peak,
        available,
        kv_cache_bytes,
        kv_cache.blocks_within(kv_cache_bytes),
    )


def measure_kv_cache_blocks(
    kv_cache: ExtensibleKVCache,
    vllm_config: "VllmConfig",
    device: torch.device,
    *,
    init_free_memory: int,
    requested_memory: int,
    initial_blocks: int | None = None,
) -> int:
    """Size an extensible KV cache from the memory free after warmup.

    Measures as `_measure_kv_cache` does, and commits no more than
    initial_blocks blocks when that is set. Then caps torch's caching allocator
    to leave KV_CACHE_NON_TORCH_RESERVE_BYTES free for allocations outside it.
    """
    free_memory, transient_peak, available, kv_cache_bytes, num_blocks = (
        _measure_kv_cache(
            kv_cache,
            vllm_config,
            device,
            init_free_memory=init_free_memory,
            requested_memory=requested_memory,
        )
    )
    if initial_blocks is not None and initial_blocks < num_blocks:
        logger.info(
            "Committing %d KV cache blocks at startup, not the %d the "
            "measurement allows (extensible_kv_cache_initial_blocks).",
            initial_blocks,
            num_blocks,
        )
        num_blocks = initial_blocks
    # Torch may grow into what stays free once the cache is extended, less the
    # memory kept for multimodal IPC and for allocations outside torch.
    left_free = (
        free_memory
        - (kv_cache.physical_bytes_for(num_blocks) - kv_cache.physical_bytes)
        - (available - kv_cache_bytes)
        - KV_CACHE_NON_TORCH_RESERVE_BYTES
    )
    torch_memory_limit = torch.accelerator.memory_reserved(device) + max(left_free, 0)
    kv_cache.set_torch_memory_limit(torch_memory_limit)
    logger.info(
        "Memory after warmup: %s GiB free, %s GiB committed to the KV cache, "
        "%s GiB kept for transient activations; %d KV cache blocks (%s GiB) "
        "fit within the requested %s GiB. Torch may reserve up to %s GiB.",
        format_gib(free_memory),
        format_gib(kv_cache.physical_bytes),
        format_gib(transient_peak),
        num_blocks,
        format_gib(num_blocks * kv_cache.bytes_per_block),
        format_gib(requested_memory),
        format_gib(torch_memory_limit),
    )
    return num_blocks


def grow_kv_cache(
    runner: "GPUModelRunner",
    target_num_blocks: int,
    *,
    init_free_memory: int,
    requested_memory: int,
    reduce_min: Callable[[int], int],
) -> int:
    """Commit more of an extensible KV cache's reservation while serving.

    Every rank takes part: each measures with the rule of the startup sizing
    and the ranks agree, through reduce_min (the minimum over ranks), on the
    block count n = the least of the target, what every rank's memory fits and
    the capacity. Physical memory is mapped on every rank before any commits, so
    a rank that cannot map leaves all ranks at the current count. Returns the
    count in force afterwards, never below the current one: a target at or below
    it changes nothing.

    Must run where no step is between its forward and its sampling: blocks
    below the old count are untouched, the new blocks are zeroed, and the views
    rebuilt over n blocks keep their addresses, so graphs stay valid.
    """
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    vllm_config = runner.vllm_config
    if vllm_config.attention_config.hisparse_config is not None:
        raise NotImplementedError("Growing the KV cache does not support HiSparse.")
    if not runner.cache_config.get_resolved_kv_cache_layout().is_block_compact:
        raise NotImplementedError(
            "Growing the KV cache needs a block-compact KV cache layout."
        )
    if (
        has_kv_transfer_group()
        and not get_kv_transfer_group().supports_kv_cache_resize()
    ):
        raise NotImplementedError(
            f"The KV connector {type(get_kv_transfer_group()).__name__} does not "
            "support resizing the KV cache."
        )
    old = kv_cache.num_committed_blocks
    try:
        fits = _measure_kv_cache(
            kv_cache,
            vllm_config,
            runner.device,
            init_free_memory=init_free_memory,
            requested_memory=requested_memory,
        ).num_blocks
    except ValueError:
        # Nothing left after the multimodal reservation: this rank fits no more.
        fits = old
    num_blocks = reduce_min(min(target_num_blocks, fits, kv_cache.capacity_blocks))
    if num_blocks <= old:
        return old
    try:
        kv_cache.back(num_blocks)
        backed = 1
    except RuntimeError:
        logger.warning(
            "Could not map memory for %d KV cache blocks.", num_blocks, exc_info=True
        )
        backed = 0
    if not reduce_min(backed):
        # Nothing was committed anywhere; mapped granules stay for the next try.
        logger.warning(
            "Not growing the KV cache from %d to %d blocks: memory could not be "
            "mapped on every rank.",
            old,
            num_blocks,
        )
        return old
    old_bytes = kv_cache.physical_bytes_for(old)
    kv_cache.commit(num_blocks)
    runner.kv_cache_config.num_blocks = num_blocks
    rebind_committed_views(runner)
    if has_kv_transfer_group():
        get_kv_transfer_group().resize_kv_caches(runner.kv_caches_by_layer, num_blocks)
    # Torch must not grow into what the cache now holds.
    torch_memory_limit = kv_cache.torch_memory_limit
    if torch_memory_limit is not None:
        torch_memory_limit = max(
            torch_memory_limit - (kv_cache.physical_bytes_for(num_blocks) - old_bytes),
            0,
        )
        kv_cache.set_torch_memory_limit(torch_memory_limit)
        if torch.accelerator.memory_reserved(runner.device) > torch_memory_limit:
            torch.accelerator.empty_cache()
    # The new blocks are zeroed on the current stream; finish before any step
    # can address them.
    torch.accelerator.synchronize()
    logger.info(
        "Grew KV cache from %d to %d blocks (%.2f GiB to %.2f GiB).",
        old,
        num_blocks,
        old_bytes / (1 << 30),
        kv_cache.physical_bytes_for(num_blocks) / (1 << 30),
    )
    return num_blocks


# A layerwise weight update holds a layer's checkpoint-format copy while weight
# processing (format conversion, padding, shuffles) allocates around as much
# again, or twice that, before the result is copied into place.
WEIGHT_UPDATE_PEAK_LAYER_FACTOR = 3


@contextmanager
def kv_cache_lent_for_weight_update(
    kv_cache: ExtensibleKVCache | None,
    model: torch.nn.Module,
    *,
    reclaimable: bool,
    fail_if_short: bool,
) -> Iterator[None]:
    """Make room for a layerwise weight update of model within the context.

    Only an extensible KV cache (kv_cache, None for a standard one) leaves
    the free memory short, since it keeps free just the transients warmup
    measured. When short, release the cache's physical memory for the
    context if it holds nothing to keep (reclaimable); else fail before any
    weight changes, when not even the largest layer fits and fail_if_short,
    or warn. Torch's memory cap is lifted for the context.
    """
    if kv_cache is None:
        yield
        return
    lent = _release_for_weight_update(
        kv_cache, model, reclaimable=reclaimable, fail_if_short=fail_if_short
    )
    # Weight processing allocates through torch, beyond the cap sized for
    # serving.
    torch_memory_limit = kv_cache.torch_memory_limit
    if torch_memory_limit is not None:
        kv_cache.set_torch_memory_limit(None)
    try:
        yield
    finally:
        if lent or torch_memory_limit is not None:
            torch.accelerator.empty_cache()
        if lent:
            kv_cache.recommit()
            if has_kv_transfer_group():
                get_kv_transfer_group().restore_kv_caches()
        if torch_memory_limit is not None:
            kv_cache.set_torch_memory_limit(torch_memory_limit)


def _release_for_weight_update(
    kv_cache: ExtensibleKVCache,
    model: torch.nn.Module,
    *,
    reclaimable: bool,
    fail_if_short: bool,
) -> bool:
    """Whether kv_cache was released to fit the update; see the caller."""
    if not kv_cache.buffer.physical_bytes:
        # Released by sleep or release_kv_cache_memory(), which also restores it.
        return False
    layer_bytes = max_layer_reload_bytes(model)
    needed = WEIGHT_UPDATE_PEAK_LAYER_FACTOR * layer_bytes
    free = (
        current_platform.free_device_memory()
        + torch.accelerator.memory_reserved()
        - torch.accelerator.memory_allocated()
    )
    if free >= needed:
        return False
    if reclaimable:
        logger.info(
            "Releasing the emptied KV cache's memory for a weight update: "
            "%s GiB free, about %s GiB needed.",
            format_gib(free),
            format_gib(needed),
        )
        if has_kv_transfer_group():
            get_kv_transfer_group().unregister_kv_caches()
        kv_cache.release_physical()
        return True
    message = (
        f"Updating weights layer by layer needs about {format_gib(needed)} GiB "
        f"of free device memory, at least {format_gib(layer_bytes)} GiB for the "
        f"largest layer alone, but only {format_gib(free)} GiB is free beside "
        "the extensible KV cache. Pause generation with the cache cleared "
        "first, or release it with release_kv_cache_memory(), so that the "
        "update can borrow its memory; or leave headroom by setting "
        "gpu_memory_utilization."
    )
    if fail_if_short and free < layer_bytes:
        raise RuntimeError(message)
    logger.warning(message)
    return False
