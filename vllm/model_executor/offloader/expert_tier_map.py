# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where each local expert slot of a layer lives while serving, and how it moves.

A layer's slots on one rank sit in two places the kernels read: *fast rows* in
device memory and *pool rows* in pinned host memory (read over UVA). Which slot
is in which row is host state here, and two int32 maps on the device
(``fast_map[g]``, ``slow_map[g]``: physical expert id -> row, -1 elsewhere) are
what the kernels read. The maps are rewritten in place while serving.

The weights of an expert never change, so a row is a lease on bytes that live
elsewhere (the expert store): giving a row up is an unmap, never a write-back,
and a row is refilled only from the store or from a pool row.

Ordering rules (R1-R3 of the design) hold for everything below:

* R1: a row is written only when no map entry points at it and every kernel
  enqueued before its unmapping has finished. A fast row is refilled on a side
  stream that waits for an event recorded on the compute stream at the commit
  point; a pool row is written by host threads, so a freed pool row waits in
  quarantine until the event recorded after its unmapping has completed.
* R2: a row is mapped only when complete: the compute stream waits for the
  event recorded after the side stream's copies before the map write.
* R3: all map writes of one commit are enqueued by one thread in one window, so
  no kernel sees a slot in both tiers or in neither.

Nothing here talks to a device directly: streams, events and copies go through
:class:`StreamOps`, so the ordering is testable on a simulated device.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

FAST = 0
POOL = 1


class RowsUnavailable(RuntimeError):
    """Not enough free pool rows (within the wait asked for)."""


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------


class StreamOps(Protocol):
    """The device operations a commit enqueues, on the thread that commits."""

    def record_compute(self) -> Any:
        """An event that completes when everything enqueued so far on the
        compute stream has completed."""

    def side_wait(self, event: Any) -> None:
        """Make the side stream wait for ``event``."""

    def side_copy(self, dst: Any, src: Any) -> None:
        """Enqueue ``dst.copy_(src)`` on the side stream."""

    def record_side(self) -> Any:
        """An event that completes when the side stream's work so far has."""

    def compute_wait(self, event: Any) -> None:
        """Make the compute stream wait for ``event``."""

    def compute_write(self, dst: Any, values: np.ndarray) -> None:
        """Enqueue a copy of ``values`` (int32) into the device tensor ``dst``
        on the compute stream."""

    def done(self, event: Any) -> bool:
        """Whether ``event`` has completed (never blocks)."""


class TorchStreamOps:
    """:class:`StreamOps` on the accelerator's compute stream and one side
    stream; call from the thread that enqueues the forward."""

    def __init__(self, device: Any):
        import torch

        self._torch = torch
        self.mod = torch.get_device_module(device)
        self.side = self.mod.Stream(device)

    def record_compute(self) -> Any:
        event = self.mod.Event()
        event.record(self.mod.current_stream())
        return event

    def side_wait(self, event: Any) -> None:
        self.side.wait_event(event)

    def side_copy(self, dst: Any, src: Any) -> None:
        with self.mod.stream(self.side):
            dst.copy_(src, non_blocking=True)

    def record_side(self) -> Any:
        event = self.mod.Event()
        event.record(self.side)
        return event

    def compute_wait(self, event: Any) -> None:
        self.mod.current_stream().wait_event(event)

    def compute_write(self, dst: Any, values: np.ndarray) -> None:
        from vllm.utils.platform_utils import is_pin_memory_available

        torch = self._torch
        host = torch.empty(
            values.shape, dtype=torch.int32, pin_memory=is_pin_memory_available()
        )
        host.copy_(torch.from_numpy(np.ascontiguousarray(values, dtype=np.int32)))
        dst.copy_(host, non_blocking=True)

    def done(self, event: Any) -> bool:
        return bool(event.query())


# --------------------------------------------------------------------------
# Row allocator with the quarantine rule
# --------------------------------------------------------------------------


class RowAllocator:
    """Free rows of one tier. A row given back is unmapped from the maps but
    kernels enqueued before the unmapping may still read it: it is handed out
    again only once the event recorded after the unmapping has completed.

    ``acquire`` is called by the thread that prepares a move, ``release`` by
    the one that commits it, so both take the lock.
    """

    def __init__(
        self,
        num_rows: int,
        free: Iterable[int],
        done: Callable[[Any], bool],
    ):
        self.num_rows = num_rows
        self._done = done
        self._free: deque[int] = deque(free)
        self._quarantine: list[tuple[int, Any]] = []
        self._lock = threading.Lock()

    def _reap_locked(self) -> None:
        if not self._quarantine:
            return
        keep = []
        for row, event in self._quarantine:
            if event is None or self._done(event):
                self._free.append(row)
            else:
                keep.append((row, event))
        self._quarantine = keep

    @property
    def free_count(self) -> int:
        """Rows that can be taken now."""
        with self._lock:
            self._reap_locked()
            return len(self._free)

    @property
    def quarantined(self) -> int:
        with self._lock:
            return len(self._quarantine)

    def free_rows(self) -> list[int]:
        with self._lock:
            self._reap_locked()
            return list(self._free)

    def release(self, rows: Iterable[int], event: Any) -> None:
        """Give rows back, unmapped as of ``event`` (None: nothing can read
        them)."""
        with self._lock:
            for row in rows:
                self._quarantine.append((int(row), event))
            self._reap_locked()

    def acquire(
        self, n: int, *, timeout: float = 0.0, poll: float = 0.002
    ) -> list[int]:
        """``n`` free rows, waiting up to ``timeout`` seconds for quarantined
        ones to clear. All or nothing."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._reap_locked()
                if len(self._free) >= n:
                    return [self._free.popleft() for _ in range(n)]
                have, waiting = len(self._free), len(self._quarantine)
            if time.monotonic() >= deadline:
                raise RowsUnavailable(
                    f"{n} pool rows needed, {have} free and {waiting} in quarantine"
                )
            time.sleep(poll)

    def give_back_unused(self, rows: Iterable[int]) -> None:
        """Rows taken and never mapped: free again at once."""
        with self._lock:
            self._free.extendleft(reversed([int(r) for r in rows]))


# --------------------------------------------------------------------------
# The set-form tier choice
# --------------------------------------------------------------------------


def choose_fast_set(
    slot_load: np.ndarray,
    is_fast: np.ndarray,
    hysteresis: float,
    *,
    changed: np.ndarray | None = None,
    read_budget: int | None = None,
) -> np.ndarray:
    """The slots that should be in fast rows, as a set that does not permute
    slots.

    Starts from the current set (``is_fast``) and keeps its size. The hottest
    slot outside it replaces the coldest slot inside it while it carries more
    than ``hysteresis`` times that load, so near-equal experts do not trade
    places back and forth: the rule of ``order_slots_by_tier``, applied to a
    set instead of to the slot order.

    ``changed`` marks the slots whose expert is replaced by this move (their
    new expert is read from the store anyway). Every swap that gives up an
    unchanged fast slot costs one extra read; ``read_budget`` caps those.
    """
    fast = np.asarray(is_fast, dtype=bool).copy()
    load = np.asarray(slot_load, dtype=np.float64)
    changed_mask = np.zeros_like(fast) if changed is None else np.asarray(changed, bool)
    slow_idx = np.flatnonzero(~fast)
    fast_idx = np.flatnonzero(fast)
    hot = slow_idx[np.argsort(-load[slow_idx], kind="stable")]
    cold = fast_idx[np.argsort(load[fast_idx], kind="stable")]
    extra = 0
    for s, f in zip(hot, cold):
        if load[s] <= hysteresis * max(load[f], 0.0):
            break
        cost = 0 if changed_mask[f] else 1
        if read_budget is not None and extra + cost > read_budget:
            break
        extra += cost
        fast[s], fast[f] = True, False
    return fast


# --------------------------------------------------------------------------
# One layer's placement on this rank
# --------------------------------------------------------------------------


@dataclass
class MovePlan:
    """What a move changes on one rank, decided from the old and new
    placement and the loads; no rows chosen yet."""

    changed: list[int]  # slots whose expert is replaced
    demoted: list[int]  # unchanged slots leaving the fast tier
    to_fast: list[int]  # slots that end in a fast row (and need a copy there)
    new_fast: np.ndarray  # bool per slot: fast after the move
    new_logical: np.ndarray  # logical expert per slot after the move
    reads: dict[int, int] = field(default_factory=dict)  # slot -> logical to read


@dataclass
class PreparedMove:
    """A plan with its rows: pool rows filled from the store, fast rows chosen.
    Committed by :meth:`TierPlacement.commit`."""

    plan: MovePlan
    pool_dest: dict[int, int]  # slot -> pool row holding its (new) bytes
    fast_dest: dict[int, int]  # slot -> fast row it ends in (for to_fast)
    end_tier: np.ndarray
    end_row: np.ndarray
    free_pool_rows: list[int]  # pool rows unmapped by the commit
    free_fast_rows: list[int]
    stats: dict[str, float] = field(default_factory=dict)


class TierPlacement:
    """Host state of one layer on one rank: for each local slot its tier and
    row, the free pool rows, and the move planning over them.

    The initial placement is the one tiers have at load: slot ``s`` below
    ``num_fast`` is fast row ``s``; the others are pool rows ``0..num_slow-1``;
    the ``spare`` pool rows beyond are free.
    """

    def __init__(
        self,
        num_slots: int,
        num_fast: int,
        spare: int,
        done: Callable[[Any], bool],
    ):
        assert 0 < num_fast < num_slots
        self.num_slots = num_slots
        self.num_fast = num_fast
        self.num_slow = num_slots - num_fast
        self.spare = spare
        self.num_pool_rows = self.num_slow + spare
        self.tier = np.empty(num_slots, dtype=np.int8)
        self.row = np.empty(num_slots, dtype=np.int32)
        self.tier[:num_fast] = FAST
        self.row[:num_fast] = np.arange(num_fast)
        self.tier[num_fast:] = POOL
        self.row[num_fast:] = np.arange(self.num_slow)
        self.pool = RowAllocator(
            self.num_pool_rows, range(self.num_slow, self.num_pool_rows), done
        )
        # Fast rows with no slot: none while the boundary stays where it was
        # at load, one per demoted slot when a slot is demoted alone.
        self.fast_free: list[int] = []

    # -- inspection -------------------------------------------------------

    def is_fast(self) -> np.ndarray:
        return self.tier == FAST

    def check(self) -> None:
        """Every slot is in exactly one row; no row serves two slots; the
        free pool rows are the ones no slot uses."""
        fast_rows = self.row[self.tier == FAST].tolist()
        pool_rows = self.row[self.tier == POOL].tolist()
        assert len(set(fast_rows)) == len(fast_rows), "fast row twice"
        assert len(set(pool_rows)) == len(pool_rows), "pool row twice"
        assert set(fast_rows).isdisjoint(self.fast_free), "fast row used and free"
        assert len(fast_rows) + len(self.fast_free) == self.num_fast, "fast rows lost"
        assert all(0 <= r < self.num_fast for r in fast_rows + self.fast_free)
        assert all(0 <= r < self.num_pool_rows for r in pool_rows)
        free = set(self.pool.free_rows())
        assert free.isdisjoint(pool_rows), "pool row used and free"

    def maps(
        self, slot_global: Sequence[int], num_global: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(fast_map, slow_map) over physical ids; ``slot_global[s]`` is the
        physical id of local slot ``s``."""
        fast_map = np.full(num_global, -1, dtype=np.int32)
        slow_map = np.full(num_global, -1, dtype=np.int32)
        for s, g in enumerate(slot_global):
            if self.tier[s] == FAST:
                fast_map[g] = self.row[s]
            else:
                slow_map[g] = self.row[s]
        return fast_map, slow_map

    # -- planning ---------------------------------------------------------

    def plan(
        self,
        old_logical: np.ndarray,
        new_logical: np.ndarray,
        slot_load: np.ndarray,
        hysteresis: float,
        *,
        tier_moves: bool = True,
    ) -> MovePlan:
        """The move from ``old_logical`` to ``new_logical`` (the expert in
        each local slot) with the tier choice for the new placement.

        ``slot_load`` is the load of the expert each slot will hold, per slot.
        """
        old_logical = np.asarray(old_logical)
        new_logical = np.asarray(new_logical)
        changed_mask = old_logical != new_logical
        n_changed = int(changed_mask.sum())
        if n_changed > self.spare:
            raise RowsUnavailable(
                f"the move replaces {n_changed} experts of the rank and the "
                f"layer has {self.spare} spare rows"
            )
        is_fast = self.is_fast()
        if tier_moves:
            new_fast = choose_fast_set(
                slot_load,
                is_fast,
                hysteresis,
                changed=changed_mask,
                read_budget=self.spare - n_changed,
            )
        else:
            new_fast = is_fast.copy()
        return self.plan_from_set(old_logical, new_logical, new_fast)

    def plan_from_set(
        self, old_logical: np.ndarray, new_logical: np.ndarray, new_fast: np.ndarray
    ) -> MovePlan:
        """The move that puts the experts ``new_logical`` in the slots and
        exactly the slots ``new_fast`` in fast rows."""
        old_logical = np.asarray(old_logical)
        new_logical = np.asarray(new_logical)
        new_fast = np.asarray(new_fast, dtype=bool)
        changed_mask = old_logical != new_logical
        is_fast = self.is_fast()
        changed = np.flatnonzero(changed_mask).tolist()
        demoted = np.flatnonzero(is_fast & ~new_fast & ~changed_mask).tolist()
        to_fast = np.flatnonzero(new_fast & (changed_mask | ~is_fast)).tolist()
        reads = {s: int(new_logical[s]) for s in changed}
        reads.update({s: int(old_logical[s]) for s in demoted})
        if len(reads) > self.spare:
            raise RowsUnavailable(
                f"the move reads {len(reads)} experts and the layer has "
                f"{self.spare} spare rows"
            )
        return MovePlan(changed, demoted, to_fast, new_fast, new_logical.copy(), reads)

    def prepare(
        self,
        plan: MovePlan,
        fill: Callable[[Mapping[int, int], Mapping[int, int]], None],
        *,
        timeout: float = 5.0,
    ) -> PreparedMove:
        """Take pool rows for the plan's reads, call ``fill(reads, rows)``
        (``reads``: slot -> logical expert; ``rows``: slot -> pool row) to put
        the bytes there, and choose fast rows. On a failure of ``fill`` the
        rows go back and the error propagates."""
        slots = sorted(plan.reads)
        rows = self.pool.acquire(len(slots), timeout=timeout)
        pool_dest = dict(zip(slots, rows))
        try:
            fill(plan.reads, pool_dest)
            return self._assign(plan, pool_dest)
        except BaseException:
            self.pool.give_back_unused(rows)
            raise

    def _assign(self, plan: MovePlan, pool_dest: dict[int, int]) -> PreparedMove:
        S = self.num_slots
        changed = set(plan.changed)
        end_tier = self.tier.copy()
        end_row = self.row.copy()
        # Fast rows: a changed slot that stays fast keeps its row; the other
        # slots entering the fast tier take the rows the leaving slots free
        # (and rows that were already free).
        leaving = [
            int(self.row[s])
            for s in range(S)
            if self.tier[s] == FAST and (not plan.new_fast[s] or s in changed)
        ]
        keep_row = {s: int(self.row[s]) for s in plan.to_fast if self.tier[s] == FAST}
        taken = set(keep_row.values())
        available = sorted((set(leaving) | set(self.fast_free)) - taken)
        fast_dest: dict[int, int] = {}
        for s in plan.to_fast:
            if s in keep_row:
                fast_dest[s] = keep_row[s]
            elif available:
                fast_dest[s] = available.pop(0)
            else:
                raise RowsUnavailable("no free fast row for a promotion")
        free_pool: list[int] = []
        for s in range(S):
            was_pool = self.tier[s] == POOL
            if plan.new_fast[s]:
                end_tier[s] = FAST
                end_row[s] = fast_dest[s] if s in fast_dest else self.row[s]
                if was_pool:
                    free_pool.append(int(self.row[s]))
                if s in pool_dest:  # a changed slot's read row was only a source
                    free_pool.append(pool_dest[s])
            else:
                end_tier[s] = POOL
                if s in pool_dest:
                    end_row[s] = pool_dest[s]
                    if was_pool:
                        free_pool.append(int(self.row[s]))
                # unchanged pool slots keep their row
        return PreparedMove(
            plan, pool_dest, fast_dest, end_tier, end_row, free_pool, available
        )

    # -- committing -------------------------------------------------------

    def commit(
        self,
        move: PreparedMove,
        ops: StreamOps,
        *,
        slot_global: Sequence[int],
        num_global: int,
        fast_map: Any,
        slow_map: Any,
        copy_rows: Callable[[int, int, int], None],
        on_pool_rows_changed: Callable[[], None] | None = None,
    ) -> None:
        """Enqueue the move on the device and update the host state.

        Must run on the thread that enqueues the forward, between two steps.
        ``copy_rows(slot, pool_row, fast_row)`` enqueues the copies of one
        slot's bytes from its pool row into its fast row via ``ops``'
        ``side_copy`` (the caller knows the tensors).

        Stream order: [record E1] side waits E1; side copies pool rows into
        the fast rows being overwritten (E1 says every kernel that could read
        them has finished: R1); [record E2]; compute waits E2 (R2); compute
        writes both maps (R3); [record E3]; the pool rows this move unmapped
        go to quarantine until E3.
        """
        plan = move.plan
        if move.fast_dest:
            drained = ops.record_compute()
            ops.side_wait(drained)
            for slot in plan.to_fast:
                src = move.pool_dest.get(slot, int(self.row[slot]))
                copy_rows(slot, src, move.fast_dest[slot])
            filled = ops.record_side()
            ops.compute_wait(filled)
        self.tier, self.row = move.end_tier.copy(), move.end_row.copy()
        self.fast_free = list(move.free_fast_rows)
        fast_np, slow_np = self.maps(slot_global, num_global)
        ops.compute_write(fast_map, fast_np)
        ops.compute_write(slow_map, slow_np)
        unmapped = ops.record_compute()
        if on_pool_rows_changed is not None:
            on_pool_rows_changed()
        self.pool.release(move.free_pool_rows, unmapped)
        self.check()
