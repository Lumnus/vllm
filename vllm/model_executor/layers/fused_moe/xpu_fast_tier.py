# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The card tier of the expert tiers, resized while serving.

With ``--expert-store-config`` ``fast_rows_max`` the card rows of each tiered
MoE layer (``ExpertTiers``) sit in a device address range reserved for that
many rows (:class:`FastRows`, an ``ExtensibleTensor`` with one segment per
kernel tensor). The fast kernel is built over the whole range, so its
``num_experts`` and its base pointers never change and the captured graphs
stay valid; only the rows in use have physical memory behind them, and no
map entry ever points above them (the kernels read an expert's weights only
for experts with routed rows).

A running engine moves the number of rows in use on every layer of every
rank at once (runtime control ``tiers.fast``; the engine core's side takes
the device memory from the KV cache first, see
``vllm/v1/engine/fast_tier.py``). This module is the worker's side: the
state, the check and the change, for every layer of this process.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import ExpertTiers

logger = init_logger(__name__)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


class FastRows:
    """One layer's card rows: a device range reserved for ``capacity`` rows
    of each of the layer's kernel tensors, the first ``rows`` of them
    backed by physical memory. Rows are committed and released as a prefix
    of every tensor's segment; a release gives back the physical objects
    (of at most ``chunk_bytes``) wholly above the rows that stay."""

    # The device to reserve on when the tensors' own is not an accelerator
    # (tests run the VMM ops' fake on the CPU).
    device_override: torch.device | None = None

    def __init__(
        self,
        like: Sequence[torch.Tensor],
        capacity: int,
        chunk_bytes: int | None,
    ) -> None:
        from vllm.utils.extensible_tensor import ExtensibleTensor, granule_size

        assert like and capacity > 0
        device = like[0].device
        if device.type not in ("cuda", "xpu") and self.device_override is not None:
            device = self.device_override
        index = device.index if device.index is not None else 0
        granularity = granule_size(index)
        self.capacity = capacity
        self.row_bytes = [t[0].numel() * t.element_size() for t in like]
        self.shapes = [tuple(t.shape[1:]) for t in like]
        self.dtypes = [t.dtype for t in like]
        # Whole granules per segment: each tensor's rows start on a granule.
        caps = [_round_up(capacity * rb, granularity) for rb in self.row_bytes]
        self.offsets = [sum(caps[:i]) for i in range(len(caps))]
        self.buffer = ExtensibleTensor(
            sum(caps),
            device=device,
            segment_capacities=caps,
            max_chunk_bytes=chunk_bytes,
        )
        self.granularity = self.buffer.granularity
        self.rows = 0

    def views(self, like: Sequence[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """One tensor per kernel tensor over the whole reserved range, in the
        shape and dtype of ``like`` with ``capacity`` rows. Rows at or above
        ``rows`` have no memory: nothing may read or write them."""
        full = self.buffer.full_view()
        out = []
        for i, (rb, shape, dtype) in enumerate(
            zip(self.row_bytes, self.shapes, self.dtypes)
        ):
            start = self.offsets[i]
            span = full[start : start + self.capacity * rb]
            out.append(span.view(dtype).view(self.capacity, *shape))
        return out

    def _sizes(self, rows: int) -> list[int]:
        return [rows * rb for rb in self.row_bytes]

    def bytes_to(self, rows: int) -> int:
        """Device bytes a commit to ``rows`` maps beyond what is mapped now
        (whole granules; 0 for a smaller count)."""
        if rows <= self.rows:
            return 0
        g = self.granularity
        return sum(
            _round_up(rows * rb, g) - _round_up(self.rows * rb, g)
            for rb in self.row_bytes
        )

    def releasable_bytes(self, rows: int) -> int:
        """Device bytes a release to ``rows`` gives back (whole objects)."""
        if rows >= self.rows:
            return 0
        return self.buffer.releasable_bytes(self._sizes(rows))

    def commit(self, rows: int) -> None:
        """Map memory behind the first ``rows`` rows of every tensor. All or
        nothing: on a failure what this call mapped is given back."""
        if rows <= self.rows:
            return
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed the {self.capacity} reserved")
        try:
            self.buffer.resize_segments_(self._sizes(rows))
        except BaseException:
            # Objects mapped above the committed prefix before the failure.
            self.buffer.shrink_segments_(self._sizes(self.rows))
            raise
        self.rows = rows

    def release_to(self, rows: int) -> int:
        """Drop the rows from ``rows`` on and give their memory back; waits
        for the device first. Returns the bytes released."""
        if rows >= self.rows:
            return 0
        released = self.buffer.shrink_segments_(self._sizes(rows))
        self.rows = rows
        return released

    @property
    def mapped_bytes(self) -> int:
        return self.buffer.physical_bytes


# --------------------------------------------------------------------------
# Every layer of this process (the worker's side of `tiers.fast`)
# --------------------------------------------------------------------------


def resizable_tiers() -> list["ExpertTiers"]:
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import live_tiers

    return [
        t for t in live_tiers() if t.fast_rows is not None and t.placement is not None
    ]


def mapped_bytes() -> int:
    """Device bytes the card tiers of this process have mapped."""
    return sum(t.fast_rows.mapped_bytes for t in resizable_tiers())


def rows_now() -> int | None:
    tiers = resizable_tiers()
    return None if not tiers else int(tiers[0].placement.num_fast)  # type: ignore[union-attr]


def rows_range() -> tuple[int, int] | None:
    tiers = resizable_tiers()
    if not tiers:
        return None
    ranges = [t.fast_range() for t in tiers]
    return max(lo for lo, _ in ranges), min(hi for _, hi in ranges)


def placement_digest() -> str:
    """Eight hex digits over the tier (card, pinned, CPU) of every slot of
    every layer: two equal digests mean the same placement, so the same
    sums (a resize and its reverse with EPLB paused can be compared)."""
    import hashlib

    h = hashlib.sha1(usedforsecurity=False)
    for t in resizable_tiers():
        h.update(t.layer_name.encode())
        h.update(t.placement.tier.tobytes())  # type: ignore[union-attr]
    return h.hexdigest()[:8]


def bytes_to(rows: int) -> int:
    """Device bytes every layer together maps for ``rows`` rows."""
    return sum(t.fast_rows.bytes_to(rows) for t in resizable_tiers())


def state(targets: Sequence[int] = ()) -> dict[str, Any] | None:
    """The readout of the card tier (part ``tiers`` of the runtime control
    state), with the bytes a grow to each of ``targets`` maps."""
    tiers = resizable_tiers()
    if not tiers:
        return None
    lo, hi = rows_range()  # type: ignore[misc]
    first = tiers[0].fast_rows
    return {
        "rows": rows_now(),
        "rows_per_layer": sorted({int(t.placement.num_fast) for t in tiers}),  # type: ignore[union-attr]
        "capacity": first.capacity,
        "min": lo,
        "max": hi,
        "layers": len(tiers),
        "mapped_bytes": mapped_bytes(),
        "chunk_bytes": first.buffer.max_chunk_bytes,
        "row_bytes_all_layers": sum(sum(t.fast_rows.row_bytes) for t in tiers),
        "bytes_to": {str(int(r)): bytes_to(int(r)) for r in targets},
        "placement": placement_digest(),
        "last": tiers[-1].move_log[-1] if tiers[-1].move_log else None,
    }


def _free_device_bytes(device: torch.device) -> int:
    """Free device memory plus torch's cache that holds nothing."""
    from vllm.platforms import current_platform

    free = int(current_platform.free_device_memory(device))
    reclaimable = max(
        torch.accelerator.memory_reserved(device)
        - torch.accelerator.memory_allocated(device),
        0,
    )
    return free + reclaimable


def device_memory(device: torch.device) -> tuple[int, int, int, int]:
    """Free device memory, and torch's allocated, peak allocated (since the
    KV cache's sizing) and reserved bytes on ``device``."""
    from vllm.platforms import current_platform

    stats = torch.accelerator.memory_stats(device)
    return (
        int(current_platform.free_device_memory(device)),
        int(stats.get("allocated_bytes.all.current", 0)),
        int(stats.get("allocated_bytes.all.peak", 0)),
        int(torch.accelerator.memory_reserved(device)),
    )


def grow_refusal(
    need: int,
    credit: int,
    *,
    free: int,
    reclaimable: int,
    floor: int,
    cap_room: int | None,
) -> str | None:
    """Why ``need`` more bytes of card rows may not be mapped now, or None.

    The rows are mapped from free device memory only: torch's unused cache
    is not room for them (a cached segment that holds one live block cannot
    be given back). ``credit`` is what the KV cache released for this resize
    within the same operation: those bytes were the KV cache's and go to
    the rows, torch's share stays as it was. Bytes beyond the credit must
    leave torch the floor the KV cache's grow leaves it
    (``grow_torch_floor_bytes``), on the device (free memory plus its unused
    cache) and under its cap (``cap_room``: the budget less the KV cache,
    the other holders, torch's reservation and the floor; None without a
    cap)."""
    if need <= 0:
        return None
    gib = 2**30
    if need > free:
        return (
            f"the device has {free / gib:.2f} GiB free and the rows need "
            f"{need / gib:.2f} GiB: the KV cache must give them back first"
        )
    beyond = max(need - credit, 0)
    if beyond <= 0:
        return None
    room = free + reclaimable - floor
    if beyond > room:
        return (
            f"{beyond / gib:.2f} GiB more card rows would leave torch less than "
            f"its floor of {floor / gib:.2f} GiB (the KV cache grow floor; "
            f"{max(room, 0) / gib:.2f} GiB above it): the KV cache must give "
            "them back first"
        )
    if cap_room is not None and beyond > cap_room:
        return (
            f"{beyond / gib:.2f} GiB more card rows would put torch's cap below "
            f"what it holds and its floor of {floor / gib:.2f} GiB "
            f"({max(cap_room, 0) / gib:.2f} GiB to spare): the KV cache must "
            "give them back first"
        )
    return None


def check(rows: Any, memory: bool = True) -> str | None:
    """Why this rank cannot move its card tier to ``rows`` now, or None.
    ``memory`` false leaves the device memory out (asked before the KV cache
    gives it)."""
    tiers = resizable_tiers()
    if not tiers:
        return (
            "the card tier is fixed in this engine: start with "
            "--expert-store-config fast_rows_max (an expert store and "
            "--cpu-offload-per-expert)"
        )
    if not isinstance(rows, int) or isinstance(rows, bool):
        return f"card rows per layer, an integer; got {rows!r}"
    lo, hi = rows_range()  # type: ignore[misc]
    if not lo <= rows <= hi:
        why = (
            "the reserved rows (fast_rows_max)"
            if rows > hi
            else "the floor (fast_rows_min; without experts computed on the CPU "
            "every slot off the card needs a pinned row)"
        )
        return f"{rows} is outside {lo}..{hi}: {why}"
    pending = [t.layer_name for t in tiers if t._eplb_pending]
    if pending:
        return f"an EPLB move is prepared on {pending[0]}; ask after it commits"
    now = rows_now()
    if memory and now is not None and rows > now:
        need = bytes_to(rows)
        device = tiers[0].fast_rows.buffer.device
        have = _free_device_bytes(device)
        if have < need:
            return (
                f"the device has {have / 2**30:.2f} GiB free (with torch's "
                f"unused cache) and {rows - now} more card rows per layer need "
                f"{need / 2**30:.2f} GiB: the KV cache must give them back first"
            )
    return None


def apply(rows: int) -> dict[str, Any]:
    """Every layer of this process to ``rows`` card rows. A grow maps every
    layer's rows first (all or nothing), then moves the slots; a failure
    part-way puts every layer back to the old count, and the error is
    raised."""
    import time

    tiers = resizable_tiers()
    old = rows_now()
    assert old is not None
    start = time.perf_counter()
    before = mapped_bytes()
    digest_before = placement_digest()
    totals: dict[str, float] = {}
    if rows > old:
        committed: list[ExpertTiers] = []
        try:
            for t in tiers:
                t.fast_rows.commit(rows)
                committed.append(t)
        except BaseException:
            for t in committed:
                t.fast_rows.release_to(old)
            raise
    done: list[ExpertTiers] = []
    try:
        for t in tiers:
            done.append(t)
            stats = t.resize_fast(rows)
            for k, v in stats.items():
                if k in ("units_read", "commits", "to_card", "off_card", "relocated"):
                    totals[k] = totals.get(k, 0.0) + v
    except BaseException:
        for t in reversed(done):
            try:
                t.resize_fast(old)
            except Exception:  # noqa: BLE001 - the first error is raised
                logger.exception(
                    "%s: could not go back to %d card rows", t.layer_name, old
                )
        for t in tiers:
            if t.fast_rows.rows > old:
                t.fast_rows.release_to(old)
        raise
    after = mapped_bytes()
    digest = placement_digest()
    seconds = time.perf_counter() - start
    logger.info(
        "Fast tier: card rows per layer %d -> %d on %d layers in %.2f s: %d "
        "slots onto the card, %d off it, %d moved down, %d units read; %.2f "
        "GiB mapped (%+.2f GiB); placement %s -> %s.",
        old,
        rows,
        len(tiers),
        seconds,
        int(totals.get("to_card", 0)),
        int(totals.get("off_card", 0)),
        int(totals.get("relocated", 0)),
        int(totals.get("units_read", 0)),
        after / 2**30,
        (after - before) / 2**30,
        digest_before,
        digest,
    )
    return {
        "rows": [old, rows],
        "placement": [digest_before, digest],
        "mapped_before": before,
        "mapped_after": after,
        "released_bytes": max(before - after, 0),
        "taken_bytes": max(after - before, 0),
        "seconds": seconds,
        **{k: int(v) for k, v in totals.items()},
    }


# --------------------------------------------------------------------------
# The move into the reserved range at load: what it gave back
# --------------------------------------------------------------------------


class OldStorages:
    """The storages a layer's card rows leave when they move into the
    reserved range (weak references: holding them would keep them alive).
    After the move each must be gone; one still alive is held by a tensor
    somewhere else, which would keep its device memory for good and leave
    the KV cache smaller by it at sizing."""

    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        from torch.multiprocessing.reductions import StorageWeakRef

        self.refs = []
        # storages that are the tensor's own (not a part of a larger buffer)
        self.own: set[int] = set()
        for t in tensors:
            storage = t.untyped_storage()
            self.refs.append(
                (StorageWeakRef(storage), int(storage.nbytes()), storage.data_ptr())
            )
            if t.storage_offset() == 0 and storage.nbytes() == (
                t.numel() * t.element_size()
            ):
                self.own.add(storage.data_ptr())
            del storage
        self.bytes = sum(n for _, n, _ in self.refs)

    def alive(self) -> list[tuple[Any, int, int]]:
        return [r for r in self.refs if not r[0].expired()]


def _storages_in_use() -> set[int]:
    """Storages the kernels or a layer still read: those of the tiers whose
    rows have not moved yet (their impls' tensors and their layers'
    parameters). An old storage among them is shared, never released."""
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
        _TIER_FIELDS,
        live_tiers,
    )

    ptrs: set[int] = set()
    for t in live_tiers():
        if getattr(t, "fast_rows", None) is not None:
            continue
        tensors: list[Any] = []
        if t._impls is not None:
            tensors += [getattr(t._impls[0], f) for f, _ in _TIER_FIELDS]
        layer = None if t._layer_ref is None else t._layer_ref()
        if layer is not None:
            tensors += list(layer.parameters(recurse=False))
        for x in tensors:
            if isinstance(x, torch.Tensor):
                ptrs.add(x.untyped_storage().data_ptr())
    return ptrs


def _holders(data_ptr: int) -> str:
    """Tensors on the storage at ``data_ptr`` and what refers to them (a
    garbage collector walk: once per process, only when a storage survived)."""
    import gc

    found = []
    for obj in gc.get_objects():
        try:
            # type(), not isinstance: some objects compute __class__
            if not issubclass(type(obj), torch.Tensor):
                continue
            if obj.untyped_storage().data_ptr() != data_ptr:
                continue
        except Exception:  # noqa: BLE001 - any object the walk meets
            continue
        owners = sorted(
            {
                type(r).__name__
                for r in gc.get_referrers(obj)
                if not isinstance(r, list) or len(r) < 1000
            }
        )
        found.append(f"{type(obj).__name__}{tuple(obj.shape)} <- {', '.join(owners)}")
        if len(found) >= 3:
            break
    return "; ".join(found) or "no tensor found (the storage itself is held)"


_ADOPTED: dict[str, Any] = {
    "layers": 0,
    "mapped": 0,
    "old": 0,
    "stale": 0,
    "released": 0,
    "free_delta": 0,
    "free_known": True,
    "logged": False,
    "holders": None,
}


def _free_or_none(device: torch.device) -> int | None:
    try:
        from vllm.platforms import current_platform

        return int(current_platform.free_device_memory(device))
    except Exception:  # noqa: BLE001 - no device to read (tests)
        return None


def free_before_move(device: torch.device) -> int | None:
    """Free device memory before one layer's move (see `finish_move`)."""
    return _free_or_none(device)


def finish_move(
    layer_name: str,
    old: OldStorages,
    mapped: int,
    device: torch.device,
    free_before: int | None,
    expected_layers: int,
) -> None:
    """After a layer's rows moved into its range: give back on the spot any
    old storage another tensor still holds (its bytes cannot be reached by
    the kernels any more, and keeping them would take them from the KV
    cache), then count the move. Once every layer has moved, one line says
    what the moves did to the device: mapped bytes against the storage
    given up and the change of free memory, so a boot shows at once whether
    the card tier costs more than its rows."""
    alive = old.alive()
    stale = sum(n for _, n, _ in alive)
    released = 0
    if alive:
        if _ADOPTED["holders"] is None:
            _ADOPTED["holders"] = _holders(alive[0][2])
        in_use = _storages_in_use()
        for ref, n, ptr in alive:
            if ptr not in old.own or ptr in in_use:
                # part of a larger buffer, or read by a layer that has not
                # moved yet: reported, not released
                continue
            storage = torch.UntypedStorage._new_with_weak_ptr(ref.cdata)
            if storage is None:
                continue
            try:
                storage.resize_(0)
                released += n
            except Exception:  # noqa: BLE001 - reported below
                logger.warning(
                    "%s: an old storage of the card rows (%d bytes) is still "
                    "held and cannot be released in place.",
                    layer_name,
                    n,
                    exc_info=True,
                )
            del storage
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
    free_after = _free_or_none(device)
    a = _ADOPTED
    a["layers"] += 1
    a["mapped"] += mapped
    a["old"] += old.bytes
    a["stale"] += stale
    a["released"] += released
    if free_before is None or free_after is None:
        a["free_known"] = False
    else:
        a["free_delta"] += free_after - free_before
    if alive:
        logger.warning_once(
            "Expert tiers: after the card rows moved into their reserved range, "
            "another tensor still held their old storage (%s); released in "
            "place on every layer where it happens. A reader of that tensor "
            "would read nothing now: it was not the card tier the kernels read.",
            a["holders"],
        )
    if a["layers"] >= expected_layers and not a["logged"]:
        a["logged"] = True
        gib = 2**30
        logger.info(
            "Expert tiers: the card rows of %d layers moved into their reserved "
            "ranges: %.2f GiB mapped, %.2f GiB of storage given up (%.2f GiB of "
            "it held by other tensors, %.2f GiB of that released in place); "
            "device free memory %s over the moves (rows and granules alone: "
            "%+.2f GiB).",
            a["layers"],
            a["mapped"] / gib,
            a["old"] / gib,
            a["stale"] / gib,
            a["released"] / gib,
            f"{a['free_delta'] / gib:+.2f} GiB" if a["free_known"] else "unknown",
            (a["old"] - a["mapped"]) / gib,
        )


def adoption_summary() -> dict[str, Any]:
    return dict(_ADOPTED)


def reset_adoption_summary() -> None:
    _ADOPTED.update(
        layers=0,
        mapped=0,
        old=0,
        stale=0,
        released=0,
        free_delta=0,
        free_known=True,
        logged=False,
        holders=None,
    )


# --------------------------------------------------------------------------
# What the KV cache's torch cap must leave out
# --------------------------------------------------------------------------


def register_with_kv_cache(kv_cache: Any) -> None:
    """Count the card tier's memory, as it moves from now on, among the VMM
    memory of other holders that the KV cache's torch cap leaves out
    (``ExtensibleKVCache.external_mapped_bytes``). The bytes mapped at load
    were measured with the rest of the model when the cache was sized, so
    only the change from here counts."""
    if kv_cache is None or getattr(kv_cache, "_fast_tier_registered", False):
        return
    baseline = mapped_bytes()
    kv_cache.external_mapped_bytes.append(lambda: mapped_bytes() - baseline)
    kv_cache._fast_tier_registered = True
