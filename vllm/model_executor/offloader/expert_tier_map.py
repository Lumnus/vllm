# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where each local expert slot of a layer lives while serving, and how it moves.

A layer's slots on one rank sit in two places the kernels read: *fast rows* in
device memory and *pool rows* in pinned host memory (read over UVA). Which slot
is in which row is host state here, and two int32 maps on the device
(``fast_map[g]``, ``slow_map[g]``: physical expert id -> row, -1 elsewhere) are
what the kernels read. The maps are rewritten in place while serving.

With a bound on the pinned rows (``num_pool_slots`` below the slots off the
card) a third place exists: *host* slots have no row the kernels read. Both
maps say -1 for them, and the layer computes them on the CPU from the expert
store (its host cache in ordinary memory, the drive behind it).

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
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

FAST = 0
POOL = 1
HOST = 2  # in no row the kernels read: computed on the CPU

# The tier pass's gain test (``EPLBConfig.tier_min_gain``) weighs what a
# routed call costs in each tier beside the card's resident experts against
# what a move costs, in µs, as measured on the serving host and its cards: a
# call from a pinned row through UVA +28; a call on the CPU 100-130 at one
# token; a unit into a pinned row (from the host cache or the drive) about
# 250; a pinned row's unit into a card row about 150.
CALL_US = {FAST: 0.0, POOL: 28.0, HOST: 120.0}
MOVE_US_POOL = 250.0
MOVE_US_CARD = 150.0


def entry_gains(tier: np.ndarray, min_gain: float) -> tuple[np.ndarray, float]:
    """The load difference (calls over the window) a slot must carry over the
    one it replaces to enter the card (per slot, by its tier now), and to
    enter a pinned row from the CPU: the call cost the swap saves over one
    window's load must repay its move ``min_gain`` times."""
    tier = np.asarray(tier)
    to_card = np.where(
        tier == HOST,
        (MOVE_US_POOL + MOVE_US_CARD) / (CALL_US[HOST] - CALL_US[FAST]),
        MOVE_US_CARD / (CALL_US[POOL] - CALL_US[FAST]),
    )
    to_pool = MOVE_US_POOL / (CALL_US[HOST] - CALL_US[POOL])
    return min_gain * to_card, min_gain * to_pool


def _leading_swaps(
    load: np.ndarray,
    out: np.ndarray,
    inn: np.ndarray,
    hysteresis: float,
    need: Any,
) -> int:
    """How many of the pairs (``out[i]`` enters, ``inn[i]`` leaves) a
    set-form choice takes before its first refusal: the outside slot carries
    more than ``hysteresis`` times the inside one's load and at least
    ``need`` (per outside slot, or one number) more."""
    m = min(len(out), len(inn))
    if m == 0:
        return 0
    a, b = load[out[:m]], np.maximum(load[inn[:m]], 0.0)
    ok = a > hysteresis * b
    if need is not None:
        req = need[out[:m]] if isinstance(need, np.ndarray) else need
        ok &= (a - b) >= req
    return m if ok.all() else int(np.argmin(ok))


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

    _shared: "dict[str, TorchStreamOps]" = {}

    @classmethod
    def shared(cls, device: Any) -> "TorchStreamOps":
        """One instance (one side stream) per device: the commits of all layers
        run on the same thread, one after the other."""
        key = str(device)
        if key not in cls._shared:
            cls._shared[key] = cls(device)
        return cls._shared[key]

    def __init__(self, device: Any):
        import torch

        self._torch = torch
        self.mod = torch.get_device_module(device)
        self.side = self.mod.Stream(device)
        # Host sources of map writes still queued: (event behind the copy,
        # the pinned tensor). A commit enqueues its map writes behind a wait,
        # so their copies are pending when this returns; the source must not
        # go back to the pinned allocator (and be handed out again for the
        # next map) before its copy has run.
        self._held: list[tuple[Any, Any]] = []

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
        self._held = [(e, h) for e, h in self._held if not e.query()]
        self._held.append((self.record_compute(), host))

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
            if isinstance(event, Future):
                finished = event.done()  # a host read still writing the row
            else:
                finished = event is None or self._done(event)
            if finished:
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
        them). ``event`` may also be the future of a host read that may still
        write the rows: they are free once it is done."""
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


def choose_pool_set(
    slot_load: np.ndarray,
    candidates: np.ndarray,
    is_pool: np.ndarray,
    capacity: int,
    hysteresis: float,
    max_changes: int | None = None,
    min_delta: float | None = None,
) -> np.ndarray:
    """The slots, among ``candidates`` (the slots off the card after a move),
    that should hold pinned rows: at most ``capacity``, the hottest, keeping
    the current pinned slots (``is_pool``) unless an outside slot carries more
    than ``hysteresis`` times the load of one inside. Free capacity is filled
    with the hottest outside candidates first. ``max_changes`` bounds the
    slots that enter (each is one read); leaving costs nothing.
    ``min_delta``: an entering slot must also carry at least that much more
    load than the slot it replaces (than nothing, into free capacity)."""
    cand = np.asarray(candidates, dtype=bool)
    load = np.asarray(slot_load, dtype=np.float64)
    pool = np.asarray(is_pool, dtype=bool) & cand
    outside = np.flatnonzero(cand & ~pool)
    outside = outside[np.argsort(-load[outside], kind="stable")]
    room = capacity - int(pool.sum())
    if room < 0:  # more pinned slots than capacity: keep the hottest
        inside = np.flatnonzero(pool)
        inside = inside[np.argsort(-load[inside], kind="stable")]
        pool[inside[capacity:]] = False
        room = 0
    left = len(outside) if max_changes is None else max(0, int(max_changes))
    fill = outside if min_delta is None else outside[load[outside] >= min_delta]
    take = fill[: min(room, left)]
    pool[take] = True
    left -= len(take)
    outside = outside[~np.isin(outside, take)]
    inside = np.flatnonzero(pool)
    inside = inside[np.argsort(load[inside], kind="stable")]
    for s, f in zip(outside, inside):
        if left <= 0 or load[s] <= hysteresis * max(load[f], 0.0):
            break
        if min_delta is not None and load[s] - max(load[f], 0.0) < min_delta:
            break
        pool[s], pool[f] = True, False
        left -= 1
    return pool


def choose_fast_set(
    slot_load: np.ndarray,
    is_fast: np.ndarray,
    hysteresis: float,
    *,
    changed: np.ndarray | None = None,
    read_budget: int | None = None,
    max_swaps: int | None = None,
    min_delta: np.ndarray | None = None,
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
    ``max_swaps`` caps the swaps themselves. ``min_delta`` (per slot): the
    entering slot must also carry at least that much more load than the
    leaving one.
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
        if max_swaps is not None and len(np.flatnonzero(fast[slow_idx])) >= max_swaps:
            break
        if load[s] <= hysteresis * max(load[f], 0.0):
            break
        if min_delta is not None and load[s] - max(load[f], 0.0) < min_delta[s]:
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
    # bool per slot: in a pinned row after the move (None: every slot that is
    # not fast, the placement without host slots)
    new_pool: np.ndarray | None = None


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
    ``num_fast`` is fast row ``s``; the next ``num_pool_slots`` (by default
    every other slot) are pool rows ``0..num_pool_slots-1``; the slots after
    them are host slots (no row); the ``spare`` (and ``extra_rows``) pool
    rows beyond are free.
    """

    def __init__(
        self,
        num_slots: int,
        num_fast: int,
        spare: int,
        done: Callable[[Any], bool],
        num_pool_slots: int | None = None,
        extra_rows: int = 0,
    ):
        assert 0 < num_fast < num_slots
        self.num_slots = num_slots
        self.num_fast = num_fast
        self.num_slow = num_slots - num_fast
        pinned = self.num_slow if num_pool_slots is None else num_pool_slots
        assert 0 <= pinned <= self.num_slow
        self.num_pinned = pinned  # slots in pinned rows when the tiers are full
        # The pinned bound as set (num_pinned is it, clamped to the slots off
        # the card while the card tier holds more rows than at load).
        self.pinned_bound = pinned
        self.num_host = self.num_slow - pinned
        # Whether this layer has a host tier at all (decided at load): the
        # pinned capacity may move later (set_pinned_capacity), the planning
        # rules do not.
        self.host_tier = self.num_host > 0
        self.spare = spare
        # Rows beyond the spares, for host slots promoted between moves.
        self.extra_rows = extra_rows
        self.num_pool_rows = pinned + spare + extra_rows
        self.tier = np.empty(num_slots, dtype=np.int8)
        self.row = np.empty(num_slots, dtype=np.int32)
        self.tier[:num_fast] = FAST
        self.row[:num_fast] = np.arange(num_fast)
        self.tier[num_fast : num_fast + pinned] = POOL
        self.row[num_fast : num_fast + pinned] = np.arange(pinned)
        self.tier[num_fast + pinned :] = HOST
        self.row[num_fast + pinned :] = -1
        self.pool = RowAllocator(
            self.num_pool_rows, range(pinned, self.num_pool_rows), done
        )
        # Fast rows with no slot: none while the boundary stays where it was
        # at load, one per demoted slot when a slot is demoted alone.
        self.fast_free: list[int] = []

    def reset_to(self, tier: np.ndarray, row: np.ndarray) -> None:
        """Take a placement as it is (before serving: no kernel reads the
        rows, so nothing is quarantined)."""
        self.tier = np.asarray(tier, dtype=np.int8).copy()
        self.row = np.asarray(row, dtype=np.int32).copy()
        self.fast_free = []
        used = set(self.row[self.tier == POOL].tolist())
        self.pool = RowAllocator(
            self.num_pool_rows,
            [r for r in range(self.num_pool_rows) if r not in used],
            self.pool._done,
        )
        self.check()

    # -- inspection -------------------------------------------------------

    def is_fast(self) -> np.ndarray:
        return self.tier == FAST

    def is_pool(self) -> np.ndarray:
        return self.tier == POOL

    def is_host(self) -> np.ndarray:
        return self.tier == HOST

    def check(self) -> None:
        """Every slot is in exactly one tier, and in one row unless it is a
        host slot; no row serves two slots; the free pool rows are the ones
        no slot uses."""
        assert np.isin(self.tier, (FAST, POOL, HOST)).all(), "slot in no tier"
        assert (self.row[self.tier == HOST] == -1).all(), "host slot with a row"
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
            elif self.tier[s] == POOL:
                slow_map[g] = self.row[s]
            # a host slot is in neither map: the CPU computes it
        return fast_map, slow_map

    def max_pinned_capacity(self) -> int:
        """The most slots that may hold pinned rows: every row but the spares
        (EPLB's), and no more than the slots off the card."""
        return min(self.num_slow, self.num_pool_rows - self.spare)

    def set_pinned_capacity(self, pinned: int) -> None:
        """How many slots the pinned rows serve from now on (a layer with a
        host tier only; its rows were sized at load). Moves nothing: the
        caller moves the slots, and the next rearrangement plans with it."""
        assert self.host_tier, "the pinned capacity moves only with a host tier"
        assert 0 <= pinned <= self.max_pinned_capacity()
        self.num_pinned = pinned
        self.pinned_bound = pinned
        self.num_host = self.num_slow - pinned

    # -- the card tier's row count while serving -------------------------

    def _set_num_fast(self, num_fast: int) -> None:
        self.num_fast = num_fast
        self.num_slow = self.num_slots - num_fast
        if self.host_tier:
            self.num_pinned = min(self.pinned_bound, self.num_slow)
        else:
            self.num_pinned = self.num_slow
        self.num_host = self.num_slow - self.num_pinned

    def grow_fast_rows(self, num_fast: int) -> None:
        """Fast rows ``[self.num_fast, num_fast)`` exist now (the caller
        mapped their memory): they are free, for the moves that follow."""
        assert self.num_fast <= num_fast <= self.num_slots
        self.fast_free = sorted(
            set(self.fast_free) | set(range(self.num_fast, num_fast))
        )
        self._set_num_fast(num_fast)
        self.check()

    def shrink_fast_rows(self, num_fast: int) -> None:
        """Drop fast rows ``[num_fast, self.num_fast)``: none may serve a
        slot (see :meth:`compact_fast`). The caller releases their memory
        once the device is past the last kernel that read them."""
        assert 0 <= num_fast <= self.num_fast
        top = set(range(num_fast, self.num_fast))
        assert top <= set(self.fast_free), (
            "a fast row above the new count still serves a slot"
        )
        self.fast_free = sorted(set(self.fast_free) - top)
        self._set_num_fast(num_fast)
        self.check()

    def plan_card_round(
        self, rank: np.ndarray, target: int, budget: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The tier choice for a change of the card tier's size (the one
        place it is made; the caller only commits it through
        :meth:`plan_from_set`, :meth:`prepare` and :meth:`commit`, the path
        EPLB's moves take). ``rank`` orders the slots by EPLB's load (0
        coldest). The next step toward ``target`` card slots that reads at
        most ``budget`` units into pinned rows:

        - grow: the hottest slots off the card enter free card rows (a pinned
          slot by a copy, a CPU slot through a read);
        - shrink: the coldest card slots leave, into a pinned row (a read; the
          card is never read back) where one is free or the slot is hotter
          than the coldest pinned slot, which then goes to the CPU; else to
          the CPU. Without host slots every leaving slot takes a pinned row.

        Returns (card set, pinned set), or None when nothing fits."""
        fast, pool, host = self.is_fast(), self.is_pool(), self.is_host()
        new_fast, new_pool = fast.copy(), pool.copy()
        n = int(fast.sum())
        reads = 0
        changed = False
        if n < target:
            room = min(target - n, len(self.fast_free))
            outside = np.flatnonzero(~fast)
            for s in outside[np.argsort(-rank[outside], kind="stable")][:room]:
                cost = 1 if host[s] else 0
                if reads + cost > budget:
                    break
                new_fast[s], new_pool[s] = True, False
                reads += cost
                changed = True
        elif n > target:
            inside = np.flatnonzero(fast)
            leaving = inside[np.argsort(rank[inside], kind="stable")][: n - target]
            cap = int(self.num_pinned)
            for s in leaving:
                if not self.host_tier:
                    if reads + 1 > budget:
                        break
                    new_fast[s], new_pool[s] = False, True
                    reads += 1
                    changed = True
                    continue
                in_pool = np.flatnonzero(new_pool)
                coldest = (
                    int(in_pool[np.argmin(rank[in_pool])]) if in_pool.size else None
                )
                if in_pool.size < cap or (
                    coldest is not None and rank[s] > rank[coldest]
                ):
                    if reads + 1 > budget:
                        break
                    if in_pool.size >= cap and coldest is not None:
                        new_pool[coldest] = False  # to the CPU, nothing read
                    new_pool[s] = True
                    reads += 1
                new_fast[s] = False
                changed = True
        return (new_fast, new_pool) if changed else None

    def plan_pinned_refill(self, rank: np.ndarray, budget: int) -> np.ndarray | None:
        """With host slots, after a change of the card tier: the pinned set
        for the bound by EPLB's own pinned-set rule (:func:`choose_pool_set`:
        over the bound the coldest leave, under it the hottest CPU slots
        enter), without its swaps (those are the next rearrangement's, by
        its hysteresis), the units read bounded by ``budget``. None when the
        pinned set stays."""
        if not self.host_tier:
            return None
        pool = self.is_pool()
        cap = int(self.num_pinned)
        count = int(pool.sum())
        if count < cap:
            cap = count + min(cap - count, budget)
        # heat from 1 and an infinite hysteresis: fill and trim, no swaps
        heat = rank.astype(np.float64) + 1.0
        want = choose_pool_set(heat, ~self.is_fast(), pool, cap, float("inf"))
        if np.array_equal(want, pool):
            return None
        return want

    def compact_fast(
        self,
        limit: int,
        ops: StreamOps,
        *,
        slot_global: Sequence[int],
        num_global: int,
        fast_map: Any,
        copy_fast: Callable[[int, int], None],
    ) -> int:
        """Move the slots in fast rows at or above ``limit`` into free rows
        below it, so that every row from ``limit`` on is free. Stream order
        as :meth:`commit`: the side stream waits for the compute stream, then
        copies each slot's row into its new row (``copy_fast(src, dst)``,
        through ``ops.side_copy``; a free row is read by no kernel after the
        wait: R1), the compute stream waits for the copies (R2) and the map
        is rewritten (R3). Returns the slots moved."""
        high = [
            s
            for s in range(self.num_slots)
            if self.tier[s] == FAST and self.row[s] >= limit
        ]
        if not high:
            return 0
        low = sorted(r for r in self.fast_free if r < limit)
        if len(low) < len(high):
            raise RowsUnavailable(
                f"{len(high)} slots sit in fast rows at or above {limit} and "
                f"{len(low)} rows below it are free"
            )
        moves = list(zip(high, low))
        drained = ops.record_compute()
        ops.side_wait(drained)
        for s, dst in moves:
            copy_fast(int(self.row[s]), dst)
        filled = ops.record_side()
        ops.compute_wait(filled)
        freed = []
        for s, dst in moves:
            freed.append(int(self.row[s]))
            self.row[s] = dst
        self.fast_free = sorted(
            (set(self.fast_free) - set(low[: len(moves)])) | set(freed)
        )
        fast_np, _ = self.maps(slot_global, num_global)
        ops.compute_write(fast_map, fast_np)
        self.check()
        return len(moves)

    def relayout_pool(self, num_pool_rows: int) -> dict[int, int]:
        """Lay the pool rows out again in ``num_pool_rows`` rows: the rows
        that serve slots become ``0..k-1`` in slot order, the rest are free.
        Host state only; the caller copies the rows into the new tensors
        (old row -> new row, returned), rewrites the slow map, and must have
        waited for every row in quarantine (nothing reads or writes them)."""
        used = [
            (s, int(self.row[s])) for s in range(self.num_slots) if self.tier[s] == POOL
        ]
        if len(used) + self.spare > num_pool_rows:
            raise RowsUnavailable(
                f"{len(used)} slots in pinned rows and {self.spare} spare rows "
                f"do not fit {num_pool_rows} rows"
            )
        moved = {old: new for new, (_, old) in enumerate(used)}
        for new, (s, _) in enumerate(used):
            self.row[s] = new
        self.extra_rows = num_pool_rows - self.num_pinned - self.spare
        self.num_pool_rows = num_pool_rows
        self.pool = RowAllocator(
            num_pool_rows, range(len(used), num_pool_rows), self.pool._done
        )
        self.check()
        return moved

    # -- planning ---------------------------------------------------------

    def plan(
        self,
        old_logical: np.ndarray,
        new_logical: np.ndarray,
        slot_load: np.ndarray,
        hysteresis: float,
        *,
        tier_moves: bool = True,
        read_budget: int | None = None,
        min_gain: float | None = None,
    ) -> MovePlan:
        """The move from ``old_logical`` to ``new_logical`` (the expert in
        each local slot) with the tier choice for the new placement.

        ``slot_load`` is the load of the expert each slot will hold, per slot.
        ``read_budget`` bounds the experts the move reads (at most the spare
        rows, the default): a move of the experts alone (EPLB bounds it by the
        spare rows) always fits, the tier changes take what is left. With
        ``new_logical`` equal to ``old_logical`` this is EPLB's tier pass.
        ``min_gain`` (> 0): a tier swap is taken only when the call cost it
        saves over the window's load repays its move that many times
        (:func:`entry_gains`); None or 0: hysteresis alone.
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
        budget = self.spare if read_budget is None else min(read_budget, self.spare)
        budget = max(budget, n_changed)
        is_fast = self.is_fast()
        fast_delta, pool_delta = (
            entry_gains(self.tier, min_gain) if min_gain else (None, None)
        )
        if not self.host_tier:
            new_fast = (
                choose_fast_set(
                    slot_load,
                    is_fast,
                    hysteresis,
                    changed=changed_mask,
                    read_budget=budget - n_changed,
                    min_delta=fast_delta,
                )
                if tier_moves
                else is_fast.copy()
            )
            return self.plan_from_set(old_logical, new_logical, new_fast)
        # Host slots: the card set and the pinned set are cut from one order
        # of the slots by load (the card the hottest, the pinned rows the
        # next). Each slot entering a pinned row, or a card row from the host,
        # is a read: the most swaps whose reads fit the budget are taken,
        # hottest first; the rest wait for the next pass.
        plan = None
        if tier_moves:

            def pool_for(fast: np.ndarray, k: int) -> np.ndarray:
                return choose_pool_set(
                    slot_load,
                    ~fast,
                    self.is_pool(),
                    self.num_pinned,
                    hysteresis,
                    max_changes=k,
                    min_delta=pool_delta,
                )

            for k in range(budget, 0, -1):
                fast_k = choose_fast_set(
                    slot_load,
                    is_fast,
                    hysteresis,
                    changed=changed_mask,
                    max_swaps=k,
                    min_delta=fast_delta,
                )
                # both sets, then the pinned set alone, then the card set alone
                for fast, pool in (
                    (fast_k, pool_for(fast_k, k)),
                    (is_fast, pool_for(is_fast, k)),
                    (fast_k, self._default_pool(fast_k)),
                ):
                    plan = self.plan_from_set(
                        old_logical, new_logical, fast, pool, check_budget=False
                    )
                    if len(plan.reads) <= budget:
                        return plan
        for fast in (is_fast.copy(),):
            plan = self.plan_from_set(
                old_logical,
                new_logical,
                fast,
                self._default_pool(fast),
                check_budget=False,
            )
            if len(plan.reads) <= self.spare:
                return plan
        assert plan is not None
        raise RowsUnavailable(
            f"the move reads {len(plan.reads)} experts and the layer has "
            f"{self.spare} spare rows"
        )

    def tier_changes_wanted(
        self,
        slot_load: np.ndarray,
        hysteresis: float,
        min_gain: float | None,
        budget: int | None = None,
    ) -> int:
        """About how many slots a tier pass would move in this layer now
        (vectorised, without planning rows): the swaps into the card set,
        and with a CPU tier the fills and swaps of the pinned set, under the
        hysteresis and the gain test, at most ``budget`` (the spare rows by
        default). For the pass-level test ``EPLBConfig.tier_min_changes``."""
        load = np.asarray(slot_load, dtype=np.float64)
        budget = self.spare if budget is None else min(budget, self.spare)
        fast_delta, pool_delta = (
            entry_gains(self.tier, min_gain) if min_gain else (None, None)
        )
        fast = self.is_fast()
        out = np.flatnonzero(~fast)
        out = out[np.argsort(-load[out], kind="stable")]
        inn = np.flatnonzero(fast)
        inn = inn[np.argsort(load[inn], kind="stable")]
        changes = _leading_swaps(load, out, inn, hysteresis, fast_delta)
        if self.host_tier:
            # the pinned set is cut from the slots off the card after the swaps
            fast = fast.copy()
            fast[out[:changes]], fast[inn[:changes]] = True, False
            pool = self.is_pool() & ~fast
            outside = np.flatnonzero(~fast & ~pool)
            outside = outside[np.argsort(-load[outside], kind="stable")]
            room = max(0, self.num_pinned - int(pool.sum()))
            eligible = (
                outside if pool_delta is None else outside[load[outside] >= pool_delta]
            )
            fills = min(room, len(eligible))
            changes += fills
            rest = outside[~np.isin(outside, eligible[:fills])]
            inside = np.flatnonzero(pool)
            inside = inside[np.argsort(load[inside], kind="stable")]
            changes += _leading_swaps(load, rest, inside, hysteresis, pool_delta)
        return min(changes, budget)

    def _default_pool(self, new_fast: np.ndarray) -> np.ndarray:
        """The pinned set when only the fast set is chosen: the slots in
        pinned rows that do not go fast stay; a slot leaving the fast tier
        takes a pinned row while fewer than ``num_pinned`` are used, else it
        goes to the host."""
        new_fast = np.asarray(new_fast, dtype=bool)
        pool = self.is_pool() & ~new_fast
        room = self.num_pinned - int(pool.sum())
        for s in np.flatnonzero(self.is_fast() & ~new_fast):
            if room <= 0:
                break
            pool[s] = True
            room -= 1
        return pool

    def plan_from_set(
        self,
        old_logical: np.ndarray,
        new_logical: np.ndarray,
        new_fast: np.ndarray,
        new_pool: np.ndarray | None = None,
        *,
        check_budget: bool = True,
    ) -> MovePlan:
        """The move that puts the experts ``new_logical`` in the slots,
        exactly the slots ``new_fast`` in fast rows and, with host slots,
        the slots ``new_pool`` in pinned rows (the rest on the host; by
        default as :meth:`_default_pool` chooses). Without host slots every
        slot that is not fast is in a pinned row."""
        old_logical = np.asarray(old_logical)
        new_logical = np.asarray(new_logical)
        new_fast = np.asarray(new_fast, dtype=bool)
        changed_mask = old_logical != new_logical
        is_fast = self.is_fast()
        changed = np.flatnonzero(changed_mask).tolist()
        demoted = np.flatnonzero(is_fast & ~new_fast & ~changed_mask).tolist()
        to_fast = np.flatnonzero(new_fast & (changed_mask | ~is_fast)).tolist()
        if not self.host_tier:
            pool = None
            reads = {s: int(new_logical[s]) for s in changed}
            reads.update({s: int(old_logical[s]) for s in demoted})
        else:
            pool = (
                self._default_pool(new_fast)
                if new_pool is None
                else np.asarray(new_pool, dtype=bool) & ~new_fast
            )
            if int(pool.sum()) > self.num_pool_rows:
                raise RowsUnavailable("more pinned slots than pool rows")
            is_pool, is_host = self.is_pool(), self.is_host()
            reads = {}
            for s in range(self.num_slots):
                lands = new_fast[s] or pool[s]
                if changed_mask[s]:
                    if lands:  # the new expert's bytes, into a row
                        reads[s] = int(new_logical[s])
                elif (pool[s] and not is_pool[s]) or (new_fast[s] and is_host[s]):
                    reads[s] = int(old_logical[s])  # into a pinned row
        if check_budget and len(reads) > self.spare:
            raise RowsUnavailable(
                f"the move reads {len(reads)} experts and the layer has "
                f"{self.spare} spare rows"
            )
        return MovePlan(
            changed, demoted, to_fast, new_fast, new_logical.copy(), reads, pool
        )

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
        rows go back and the error propagates. A failed ``fill`` whose reads
        may still be writing some rows names them in the error's
        ``rows_in_flight`` (row -> the read's future): those rows are held
        until their read is done, so a later move cannot be handed a row a
        stale read then overwrites."""
        slots = sorted(plan.reads)
        rows = self.pool.acquire(len(slots), timeout=timeout)
        pool_dest = dict(zip(slots, rows))
        try:
            fill(plan.reads, pool_dest)
            return self._assign(plan, pool_dest)
        except BaseException as exc:
            busy: Mapping[int, Any] = getattr(exc, "rows_in_flight", None) or {}
            self.pool.give_back_unused([r for r in rows if r not in busy])
            for row, read in busy.items():
                self.pool.release([row], read)
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
            elif plan.new_pool is None or plan.new_pool[s]:
                end_tier[s] = POOL
                if s in pool_dest:
                    end_row[s] = pool_dest[s]
                    if was_pool:
                        free_pool.append(int(self.row[s]))
                # unchanged pool slots keep their row
                assert end_row[s] >= 0, f"slot {s} would be pinned with no row"
            else:
                end_tier[s] = HOST
                end_row[s] = -1
                if was_pool:
                    free_pool.append(int(self.row[s]))
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
        if (
            not move.pool_dest
            and not move.fast_dest
            and np.array_equal(move.end_tier, self.tier)
            and np.array_equal(move.end_row, self.row)
        ):
            return  # nothing moves on this rank (a host slot's new expert
            # changes no row and no map)
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
