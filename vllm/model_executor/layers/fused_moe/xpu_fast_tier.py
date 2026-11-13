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


def check(rows: Any) -> str | None:
    """Why this rank cannot move its card tier to ``rows`` now, or None."""
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
    if now is not None and rows > now:
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
    seconds = time.perf_counter() - start
    logger.info(
        "Fast tier: card rows per layer %d -> %d on %d layers in %.2f s: %d "
        "slots onto the card, %d off it, %d moved down, %d units read; %.2f "
        "GiB mapped (%+.2f GiB).",
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
    )
    return {
        "rows": [old, rows],
        "mapped_before": before,
        "mapped_after": after,
        "released_bytes": max(before - after, 0),
        "taken_bytes": max(after - before, 0),
        "seconds": seconds,
        **{k: int(v) for k, v in totals.items()},
    }


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
