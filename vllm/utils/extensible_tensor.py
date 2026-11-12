# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growable device byte buffers backed by GPU virtual memory management."""

from __future__ import annotations

import ctypes
import itertools
import sys
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import torch

from vllm.utils.vmm_driver import get_vmm_driver


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ROCm reports a 4 KiB granularity (CUDA 2 MiB); commit in larger units to
# keep the granule bookkeeping small.
_MIN_GRANULARITY = 2 << 20


def granule_size(device_index: int) -> int:
    """Commit granule for device_index: driver granularity, floored at 2 MiB."""
    driver_granularity = get_vmm_driver().granularity(device_index)
    return _round_up(max(driver_granularity, _MIN_GRANULARITY), driver_granularity)


class _VirtualBuffer:
    """One device VA reservation and the physical chunks mapped into it.

    Memory is committed in granules via `ensure_committed_range`; already-mapped
    granules are skipped, so ranges may abut or overlap. A physical object can
    only be released whole, so with `max_chunk_bytes` each unmapped run is
    mapped as objects of at most that size (rounded up to a multiple of the
    granularity), and `release_range` can give back every object wholly inside
    a range. Without it a run is one object.
    """

    def __init__(
        self,
        max_bytes: int,
        device_index: int,
        max_chunk_bytes: int | None = None,
    ) -> None:
        self._driver = get_vmm_driver()
        self._driver.ensure_context(device_index)
        self.device_index = device_index

        self.granularity: int = granule_size(device_index)
        if max_chunk_bytes is not None and max_chunk_bytes <= 0:
            raise ValueError(
                f"max_chunk_bytes must be positive, got {max_chunk_bytes}."
            )
        self.max_chunk_bytes: int | None = (
            None
            if max_chunk_bytes is None
            else _round_up(max_chunk_bytes, self.granularity)
        )
        self.reserved_size: int = _round_up(max(max_bytes, 1), self.granularity)
        self.base_ptr: int = self._driver.reserve(self.reserved_size)

        # Granule indices (VA offset // granularity) that have physical
        # memory mapped.
        self._mapped_granules: set[int] = set()
        # Each entry is (handle, va_offset, size) for one mapped physical chunk.
        self._handles: list[tuple[int, int, int]] = []
        self._freed: bool = False

    @property
    def committed_bytes(self) -> int:
        """Total physically mapped bytes (a multiple of the granularity)."""
        return len(self._mapped_granules) * self.granularity

    def ensure_committed_range(
        self,
        start: int,
        end: int,
        chunk_origin: int = 0,
        whole_runs: bool = False,
    ) -> None:
        """Map physical pages so that the byte range `[start, end)` is backed.

        With `max_chunk_bytes` set, each unmapped run is split into objects at
        multiples of it counted from `chunk_origin` (the granule holding that
        byte), so a caller that commits a region as a prefix gets the same
        objects whatever the sizes it commits in. `whole_runs` maps each run as
        one object regardless.
        """
        if not 0 <= start <= end:
            raise ValueError(f"Invalid range [{start}, {end}).")
        if end > self.reserved_size:
            raise ValueError(
                f"Requested range end {end} exceeds reserved capacity "
                f"{self.reserved_size}."
            )
        if start == end:
            return
        first = start // self.granularity
        last = (end + self.granularity - 1) // self.granularity  # exclusive
        run_start: int | None = None
        for g in range(first, last + 1):
            unmapped = g < last and g not in self._mapped_granules
            if unmapped and run_start is None:
                run_start = g
            elif not unmapped and run_start is not None:
                self._map_run(run_start, g, chunk_origin, whole_runs)
                run_start = None

    def _map_run(
        self, first: int, last: int, chunk_origin: int, whole_run: bool
    ) -> None:
        """Map the unmapped granule run `[first, last)` as one or more objects."""
        size = self.granularity
        chunk = None if whole_run else self.max_chunk_bytes
        step = chunk // size if chunk else 0
        origin = chunk_origin // size
        mapped_to = first
        try:
            while mapped_to < last:
                stop = last
                if step:
                    # Next boundary of the chunk grid strictly after mapped_to.
                    boundary = origin + ((mapped_to - origin) // step + 1) * step
                    stop = min(boundary, last)
                self._map_chunk_at(mapped_to * size, (stop - mapped_to) * size)
                mapped_to = stop
        finally:
            if mapped_to > first:
                # Grant before recording the run as mapped: a failure in a
                # later run must not leave these granules mapped but
                # inaccessible, since later commits would skip them. Objects
                # mapped before a failure in this run are recorded too, so the
                # next commit does not map over them.
                self._grant_access(*self._run_bounds(first, mapped_to))
                self._mapped_granules.update(range(first, mapped_to))

    def _run_bounds(self, first: int, last: int) -> tuple[int, int]:
        """Bounds of the maximal contiguous mapped run covering `[first, last)`."""
        while first - 1 in self._mapped_granules:
            first -= 1
        while last in self._mapped_granules:
            last += 1
        return first, last

    def _grant_access(self, first: int, last: int) -> None:
        """Grant device access over the granule run `[first, last)`.

        Per run rather than per chunk: ROCm rejects a set-access range that starts
        inside an already-mapped region. Re-granting is a cheap no-op.
        """
        self._driver.set_access(
            self.base_ptr + first * self.granularity,
            (last - first) * self.granularity,
            self.device_index,
        )

    def _map_chunk_at(self, offset: int, size: int) -> None:
        """Create one physical chunk of `size` bytes and map it at `offset`."""
        driver = self._driver
        driver.ensure_context(self.device_index)
        try:
            handle = driver.create(size, self.device_index)
        except RuntimeError:
            # The VMM allocator cannot reuse memory idling in torch's
            # caching allocator; return it to the driver and retry once.
            torch.accelerator.empty_cache()
            handle = driver.create(size, self.device_index)

        addr = self.base_ptr + offset
        try:
            driver.map(addr, size, handle)
        except RuntimeError:
            driver.release(handle)
            raise
        # Access is granted by the caller, per contiguous run.
        self._handles.append((handle, offset, size))

    def _objects_within(self, start: int, end: int) -> list[tuple[int, int, int]]:
        """Mapped physical objects lying wholly inside `[start, end)`."""
        return [
            (handle, offset, size)
            for handle, offset, size in self._handles
            if start <= offset and offset + size <= end
        ]

    def releasable_bytes(self, start: int, end: int) -> int:
        """Bytes `release_range(start, end)` would give back."""
        if not 0 <= start <= end:
            raise ValueError(f"Invalid range [{start}, {end}).")
        return sum(size for _, _, size in self._objects_within(start, end))

    def release_range(self, start: int, end: int) -> int:
        """Unmap and release the physical objects wholly inside `[start, end)`.

        An object cannot be released in part, so one that straddles `start` or
        `end` stays mapped: the release rounds inward to whole objects, never
        outward, and bytes outside the range keep their contents. Returns the
        bytes released. The caller guarantees nothing reads or writes the
        range any more; in-flight work on the device is waited for here.

        If a driver call fails, objects already released stay released and the
        bookkeeping matches the device.
        """
        if not 0 <= start <= end:
            raise ValueError(f"Invalid range [{start}, {end}).")
        doomed = self._objects_within(start, end)
        if not doomed:
            return 0
        driver = self._driver
        driver.ensure_context(self.device_index)
        torch.accelerator.synchronize(self.device_index)
        released = 0
        done: set[int] = set()
        try:
            for handle, offset, size in doomed:
                driver.unmap(self.base_ptr + offset, size)
                done.add(handle)
                self._mapped_granules.difference_update(
                    range(
                        offset // self.granularity, (offset + size) // self.granularity
                    )
                )
                released += size
                driver.release(handle)
        finally:
            self._handles = [h for h in self._handles if h[0] not in done]
        return released

    def release_physical(self) -> None:
        """Unmap and release all physical memory, keeping the VA reservation."""
        driver = self._driver
        driver.ensure_context(self.device_index)
        if self._handles:
            torch.accelerator.synchronize(self.device_index)
        for handle, offset, size in self._handles:
            driver.unmap(self.base_ptr + offset, size)
            driver.release(handle)
        self._handles = []
        self._mapped_granules = set()

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        self.release_physical()
        if self.base_ptr:
            self._driver.free_reserved(self.base_ptr, self.reserved_size)
        self.base_ptr = 0

    def __del__(
        self, _suppress: Any = suppress, _is_finalizing: Any = sys.is_finalizing
    ) -> None:
        # Bound as defaults: module globals may be gone at interpreter exit.
        # The process frees its mappings on exit, and the HIP runtime may
        # already be torn down, where a driver call aborts the process.
        if _suppress is None or _is_finalizing():
            return
        with _suppress(Exception):
            self.free()


_K_DL_UINT = 1
_UINT8_BITS = 8


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]

# A private prototype: ctypes.pythonapi.PyCapsule_New is shared, and other
# libraries (e.g. FlashInfer's MNNVL buffers) reset its restype.
_PyCapsule_New = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
)(("PyCapsule_New", ctypes.pythonapi))
_malloc = ctypes.CDLL(None).malloc
_malloc.restype = ctypes.c_void_p
_malloc.argtypes = [ctypes.c_size_t]


def uint8_tensor_from_ptr(ptr: int, num_bytes: int, device_index: int) -> torch.Tensor:
    """Wrap device memory at `ptr` as a uint8 tensor; the caller keeps it mapped.

    The DLManagedTensor is malloc'd and never freed: torch reads it when the
    view is destroyed, which can be after this module's globals are cleared at
    interpreter exit. Its deleter is NULL, so nothing calls back into Python.
    """
    driver = get_vmm_driver()
    if (tensor := driver.uint8_tensor(ptr, num_bytes, device_index)) is not None:
        return tensor
    managed_size = ctypes.sizeof(_DLManagedTensor)
    addr = _malloc(managed_size + ctypes.sizeof(ctypes.c_int64))
    if not addr:
        raise MemoryError("Failed to allocate a DLManagedTensor.")
    ctypes.memset(addr, 0, managed_size + ctypes.sizeof(ctypes.c_int64))
    shape_addr = addr + managed_size
    ctypes.c_int64.from_address(shape_addr).value = num_bytes

    managed = _DLManagedTensor.from_address(addr)
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    device_type = driver.dlpack_device_type
    managed.dl_tensor.device = _DLDevice(device_type, device_index)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(_K_DL_UINT, _UINT8_BITS, 1)
    managed.dl_tensor.shape = ctypes.cast(shape_addr, ctypes.POINTER(ctypes.c_int64))

    capsule = _PyCapsule_New(addr, b"dltensor", None)
    return torch.from_dlpack(capsule)


class ExtensibleTensor:
    """A device byte buffer that grows without moving its base pointer.

    The reservation is divided into contiguous segments whose committed bytes
    form a prefix of each; num_segments equal segments by default, or the
    explicit segment_capacities (in bytes, summing to max_num_bytes).

    With max_chunk_bytes, growing a segment maps its new bytes as physical
    objects of at most that size, aligned to multiples of it from the segment's
    start, and `shrink_segments_` gives back the objects that lie wholly above
    the new size. Without it each grow maps one object per contiguous run and
    a shrink can only free objects a grow made small enough.
    """

    def __init__(
        self,
        max_num_bytes: int,
        device: torch.device | str | int | None = None,
        num_segments: int = 1,
        segment_capacities: Sequence[int] | None = None,
        max_chunk_bytes: int | None = None,
    ) -> None:
        if max_num_bytes < 0:
            raise ValueError("max_num_bytes must be non-negative.")
        if segment_capacities is None:
            if num_segments < 1:
                raise ValueError(f"num_segments must be positive, got {num_segments}.")
            if max_num_bytes % num_segments != 0:
                raise ValueError(
                    f"max_num_bytes ({max_num_bytes}) must be divisible by "
                    f"num_segments ({num_segments})."
                )
            segment_capacities = [max_num_bytes // num_segments] * num_segments
        elif not segment_capacities or any(c <= 0 for c in segment_capacities):
            raise ValueError("segment_capacities must be non-empty and positive.")
        elif sum(segment_capacities) != max_num_bytes:
            raise ValueError(
                f"segment_capacities sum to {sum(segment_capacities)}, not "
                f"max_num_bytes ({max_num_bytes})."
            )

        if device is None:
            device = torch.accelerator.current_device_index()
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type not in ("cuda", "xpu"):
            raise ValueError(
                f"ExtensibleTensor requires a cuda or xpu device, got {dev}."
            )
        self._device_type: str = dev.type
        self._device_index: int = (
            dev.index
            if dev.index is not None
            else torch.accelerator.current_device_index()
        )

        self._max_num_bytes: int = max_num_bytes
        self._segment_capacities: list[int] = list(segment_capacities)
        self._segment_offsets: list[int] = list(
            itertools.accumulate([0, *self._segment_capacities[:-1]])
        )
        self._committed: list[int] = [0] * len(self._segment_capacities)
        self._buffer: _VirtualBuffer = _VirtualBuffer(
            max_num_bytes, self._device_index, max_chunk_bytes
        )

    def full_view(self) -> torch.Tensor:
        """Uint8 view spanning the whole reservation."""
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._max_num_bytes, self._device_index
        )

    def segment_view(self, index: int) -> torch.Tensor:
        """Uint8 tensor over one segment's committed prefix, sized exactly to it."""
        if not 0 <= index < self.num_segments:
            raise IndexError(f"Segment {index} out of range ({self.num_segments}).")
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr + self._segment_offsets[index],
            self._committed[index],
            self._device_index,
        )

    def resize_per_segment_(
        self, bytes_per_segment: int, zero_new: bool = False
    ) -> None:
        """Grow every segment's committed prefix to the same size."""
        self.resize_segments_([bytes_per_segment] * self.num_segments, zero_new)

    def _grown_ranges(self, sizes: Sequence[int]) -> list[tuple[int, int, int]]:
        """(segment offset, start, end) reservation byte ranges that growing each
        segment's committed prefix to `sizes` adds, after checking `sizes`."""
        if len(sizes) != self.num_segments:
            raise ValueError(f"Expected {self.num_segments} sizes, got {len(sizes)}.")
        for old, new, capacity in zip(self._committed, sizes, self._segment_capacities):
            if new < old:
                raise ValueError(
                    f"ExtensibleTensor is grow-only: cannot resize from {old} "
                    f"to {new} bytes per segment."
                )
            if new > capacity:
                raise ValueError(
                    f"Requested {new} bytes per segment exceeds the "
                    f"segment capacity {capacity}."
                )
        return [
            (start, start + old, start + new)
            for start, old, new in zip(self._segment_offsets, self._committed, sizes)
            if new > old
        ]

    def back_segments_(self, sizes: Sequence[int]) -> None:
        """Map physical memory behind each segment's first `sizes[i]` bytes
        without growing the committed prefixes.

        Lets a caller learn whether a resize can be backed, e.g. on every rank
        of a group, before it commits. A later `resize_segments_` to the same
        sizes skips the granules mapped here. If this raises, some granules may
        stay mapped; they hold no committed bytes and later resizes reuse them.
        """
        for origin, start, end in self._grown_ranges(sizes):
            self._buffer.ensure_committed_range(start, end, chunk_origin=origin)

    def resize_segments_(
        self,
        sizes: Sequence[int],
        zero_new: bool = False,
        whole_runs: bool = False,
    ) -> None:
        """Grow each segment's committed prefix; optionally zero the new bytes.

        `whole_runs` maps each contiguous unmapped run as one physical object
        even when the tensor has a chunk size, for a caller that needs a range
        backed by a single allocation.
        """
        grown = self._grown_ranges(sizes)
        for origin, start, end in grown:
            self._buffer.ensure_committed_range(
                start, end, chunk_origin=origin, whole_runs=whole_runs
            )
        self._committed = list(sizes)
        if zero_new and grown:
            full = self.full_view()
            for _, start, end in grown:
                full[start:end].zero_()

    def _shrunk_ranges(self, sizes: Sequence[int]) -> list[tuple[int, int]]:
        """Reservation byte ranges above each segment's new committed size,
        up to its capacity, after checking `sizes`.

        To the capacity rather than to the old committed size, so that a shrink
        also gives back what a failed grow or release left mapped above the
        committed prefix, and repeating a shrink that failed part-way finishes it.
        """
        if len(sizes) != self.num_segments:
            raise ValueError(f"Expected {self.num_segments} sizes, got {len(sizes)}.")
        for old, new in zip(self._committed, sizes):
            if new < 0 or new > old:
                raise ValueError(f"Cannot shrink a segment from {old} to {new} bytes.")
        return [
            (start + new, start + capacity)
            for start, capacity, new in zip(
                self._segment_offsets, self._segment_capacities, sizes
            )
        ]

    def releasable_bytes(self, sizes: Sequence[int]) -> int:
        """Physical bytes `shrink_segments_(sizes)` would give back."""
        return sum(
            self._buffer.releasable_bytes(start, end)
            for start, end in self._shrunk_ranges(sizes)
        )

    def shrink_segments_(self, sizes: Sequence[int]) -> int:
        """Shrink each segment's committed prefix to `sizes[i]` bytes and give
        back the physical objects wholly above it. Returns the bytes released.

        Bytes below the new sizes keep their contents. An object that straddles
        a new size stays mapped (so `physical_bytes` drops by whole objects
        only, and by nothing for a tensor without `max_chunk_bytes` whose
        segment is one object); the bytes above the new size inside it are not
        committed. Objects mapped above the old committed size (by `back_segments_`
        or a failed grow) go too, when wholly above the new size. Growing again
        maps fresh objects for the unmapped part and zeroes (with `zero_new`) all
        bytes from the old committed size.

        The caller guarantees nothing reads or writes the released range any
        more. If a driver call fails the committed sizes are already the new
        ones: some memory above them stays mapped, which a later grow reuses and
        a repeated shrink releases.
        """
        ranges = self._shrunk_ranges(sizes)
        self._committed = list(sizes)
        return sum(self._buffer.release_range(start, end) for start, end in ranges)

    def release_physical(self) -> None:
        """Drop all physical pages but keep the reservation and its views."""
        self._buffer.release_physical()
        self._committed = [0] * self.num_segments

    @property
    def num_bytes(self) -> int:
        """Current committed size in bytes, summed over all segments."""
        return sum(self._committed)

    @property
    def bytes_per_segment(self) -> int:
        """Current committed prefix size of each segment in bytes (equal segments)."""
        return self._uniform(self._committed)

    @property
    def num_segments(self) -> int:
        return len(self._segment_capacities)

    @property
    def segment_capacity_bytes(self) -> int:
        """Maximum size of each segment (equal segments)."""
        return self._uniform(self._segment_capacities)

    @property
    def segment_capacities(self) -> list[int]:
        return list(self._segment_capacities)

    @property
    def segment_offsets(self) -> list[int]:
        """Byte offset of each segment within the reservation."""
        return list(self._segment_offsets)

    @staticmethod
    def _uniform(values: list[int]) -> int:
        if any(v != values[0] for v in values):
            raise ValueError(f"Segments differ: {values}.")
        return values[0]

    @property
    def capacity_bytes(self) -> int:
        return self._buffer.reserved_size

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes, including granule rounding."""
        return self._buffer.committed_bytes

    @property
    def max_chunk_bytes(self) -> int | None:
        """Largest physical object a grow maps, or None for one per run."""
        return self._buffer.max_chunk_bytes

    @property
    def num_physical_chunks(self) -> int:
        """Number of driver allocations currently mapped into the reservation."""
        return len(self._buffer._handles)

    def segments_backed_by_one_chunk(self, sizes: Sequence[int]) -> list[bool]:
        """Whether each segment's first `sizes[i]` bytes lie within a single
        driver allocation, as RDMA registration of the segment requires."""
        chunks = [(offset, offset + size) for _, offset, size in self._buffer._handles]
        return [
            any(start <= offset and offset + size <= end for start, end in chunks)
            for offset, size in zip(self._segment_offsets, sizes)
        ]

    @property
    def granularity(self) -> int:
        return self._buffer.granularity

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def device(self) -> torch.device:
        return torch.device(self._device_type, self._device_index)

    def free(self) -> None:
        self._buffer.free()
        self._committed = [0] * self.num_segments
