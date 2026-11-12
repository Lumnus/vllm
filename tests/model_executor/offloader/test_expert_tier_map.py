# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier placement of one layer's expert slots: row allocation with the
quarantine rule, the set-form tier choice, and the ordering of a commit.

The device is simulated: every stream is a queue of operations, an operation
runs when the test scheduler picks its stream, and an operation that waits on
an incomplete event blocks its stream. Copies and map writes land when their
operation runs, so a kernel sees exactly what a device would have in the rows
and in the maps at that point of the schedule. Host writes (a store read into a
pool row) land at once, as a host thread's would.
"""

import random
from typing import Any

import numpy as np
import pytest
import torch

from vllm.model_executor.offloader.expert_tier_map import (
    FAST,
    POOL,
    RowAllocator,
    RowsUnavailable,
    TierPlacement,
    choose_fast_set,
)

pytestmark = pytest.mark.skip_global_cleanup

ROW = 4  # bytes per row in the simulated tensors


class _Event:
    def __init__(self) -> None:
        self.done = False


class _Stream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: list = []

    def step(self) -> bool:
        if not self.ops:
            return False
        if self.ops[0]():
            self.ops.pop(0)
            return True
        return False


class SimOps:
    """:class:`StreamOps` on a compute and a side queue."""

    def __init__(self) -> None:
        self.compute = _Stream("compute")
        self.side = _Stream("side")
        self.streams = [self.compute, self.side]

    def record_compute(self) -> _Event:
        return self._record(self.compute)

    def record_side(self) -> _Event:
        return self._record(self.side)

    @staticmethod
    def _record(stream: _Stream) -> _Event:
        event = _Event()

        def complete() -> bool:
            event.done = True
            return True

        stream.ops.append(complete)
        return event

    def side_wait(self, event: _Event) -> None:
        self.side.ops.append(lambda: event.done)

    def compute_wait(self, event: _Event) -> None:
        self.compute.ops.append(lambda: event.done)

    def side_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        self.side.ops.append(lambda: dst.copy_(src) is not None)

    def compute_write(self, dst: torch.Tensor, values: np.ndarray) -> None:
        values = values.copy()
        self.compute.ops.append(lambda: dst.copy_(torch.from_numpy(values)) is not None)

    def done(self, event: _Event) -> bool:
        return event.done

    def run_some(self, rng: random.Random, n: int) -> None:
        for _ in range(n):
            rng.choice(self.streams).step()

    def drain(self) -> None:
        while any(s.ops for s in self.streams):
            if not any(s.step() for s in self.streams):
                raise AssertionError("deadlock: every stream waits on an event")


def expert_bytes(logical: int) -> torch.Tensor:
    return torch.full((ROW,), 10 + logical, dtype=torch.uint8)


# -- the allocator and the choice ----------------------------------------------


def test_quarantined_rows_are_handed_out_only_after_their_event():
    ops = SimOps()
    alloc = RowAllocator(5, [3, 4], ops.done)
    assert alloc.acquire(2) == [3, 4]
    event = ops.record_compute()
    alloc.release([3, 4], event)
    assert alloc.free_count == 0 and alloc.quarantined == 2
    with pytest.raises(RowsUnavailable):
        alloc.acquire(1)
    ops.compute.step()  # the event completes
    assert alloc.free_count == 2
    assert alloc.acquire(2) == [3, 4]
    alloc.give_back_unused([3, 4])
    assert alloc.free_rows() == [3, 4]
    with pytest.raises(RowsUnavailable):
        alloc.acquire(3)  # all or nothing
    assert alloc.free_count == 2


def test_fast_set_choice_follows_the_hysteresis_rule_without_permuting():
    load = np.array([1.0, 1.0, 1.0, 0.0, 2.0, 40.0, 50.0, 60.0])
    fast = np.array([True] * 3 + [False] * 5)
    out = choose_fast_set(load, fast, 1.1)
    assert out.tolist() == [False, False, False, False, False, True, True, True]
    # near-equal loads do not trade places
    load = np.array([10.0, 10.0, 10.0, 10.5, 10.5, 10.5])
    out = choose_fast_set(load, np.array([True] * 3 + [False] * 3), 1.1)
    assert out.tolist() == [True] * 3 + [False] * 3
    # the read budget bounds the swaps of unchanged slots
    load = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    start = np.array([True] * 3 + [False] * 3)
    assert choose_fast_set(load, start, 1.1, read_budget=2).sum() == 3
    assert (
        choose_fast_set(load, start, 1.1, read_budget=2)[:3].tolist().count(False) == 2
    )
    # a changed slot's swap costs no extra read
    changed = np.array([True, True, True, False, False, False])
    out = choose_fast_set(load, start, 1.1, changed=changed, read_budget=0)
    assert out.tolist() == [False] * 3 + [True] * 3


# -- placement and commit on the simulated device ---------------------------------


class _Layer:
    """One layer on one rank: S slots, F fast rows, R pool rows, in simulated
    device tensors, and the bookkeeping to check what kernels read."""

    def __init__(self, S=8, F=3, spare=3, num_logical=12, seed=0):
        self.S, self.F, self.spare = S, F, spare
        self.num_logical = num_logical
        self.ops = SimOps()
        self.placement = TierPlacement(S, F, spare, self.ops.done)
        self.R = self.placement.num_pool_rows
        self.logical = np.arange(S)  # expert in each slot
        self.slot_global = list(range(S))
        # simulated memory: row r holds the bytes of the expert in it
        self.fast = torch.zeros((F, ROW), dtype=torch.uint8)
        self.pool = torch.zeros((self.R, ROW), dtype=torch.uint8)
        for s in range(S):
            if s < F:
                self.fast[s] = expert_bytes(s)
            else:
                self.pool[s - F] = expert_bytes(s)
        fm, sm = self.placement.maps(self.slot_global, S)
        self.fast_map = torch.from_numpy(fm)
        self.slow_map = torch.from_numpy(sm)
        self.kernel_errors: list[str] = []
        self.kernels = 0
        self.rng = random.Random(seed)

    def fill(self, reads, rows) -> None:
        for slot, logical in reads.items():
            self.pool[rows[slot]] = expert_bytes(logical)

    def copy_rows(self, slot: int, pool_row: int, fast_row: int) -> None:
        self.ops.side_copy(self.fast[fast_row], self.pool[pool_row])

    def enqueue_kernel(self) -> None:
        """A step's expert kernel: reads every slot through the maps as they
        are when it runs and checks the bytes against the expert the slot
        held when it was enqueued."""
        expected = self.logical.copy()

        def run() -> bool:
            self.kernels += 1
            fm, sm = self.fast_map, self.slow_map
            for s, g in enumerate(self.slot_global):
                f, p = int(fm[g]), int(sm[g])
                if (f >= 0) == (p >= 0):
                    self.kernel_errors.append(f"slot {s}: maps {f}, {p}")
                    continue
                row = self.fast[f] if f >= 0 else self.pool[p]
                if not torch.equal(row, expert_bytes(int(expected[s]))):
                    self.kernel_errors.append(
                        f"slot {s} read {row.tolist()} want expert {expected[s]}"
                    )
            return True

        self.ops.compute.ops.append(run)

    def move(self, new_logical: np.ndarray, load: np.ndarray) -> Any:
        plan = self.placement.plan(self.logical, new_logical, load, 1.1)
        # the worker thread waits for quarantined rows while steps run
        for _ in range(200):
            try:
                move = self.placement.prepare(plan, self.fill, timeout=0.0)
                break
            except RowsUnavailable:
                self.ops.run_some(self.rng, 6)
        else:
            raise AssertionError("pool rows never came out of quarantine")
        self.placement.commit(
            move,
            self.ops,
            slot_global=self.slot_global,
            num_global=self.S,
            fast_map=self.fast_map,
            slow_map=self.slow_map,
            copy_rows=self.copy_rows,
        )
        self.logical = move.plan.new_logical.copy()
        return move


def _random_target(layer: _Layer, rng: random.Random) -> np.ndarray:
    new = layer.logical.copy()
    outside = [e for e in range(layer.num_logical) if e not in set(new.tolist())]
    rng.shuffle(outside)
    for slot in rng.sample(range(layer.S), rng.randint(0, layer.spare)):
        if outside:
            new[slot] = outside.pop()
    return new


@pytest.mark.parametrize("seed", range(40))
def test_no_kernel_reads_a_row_under_write_and_every_slot_is_mapped_once(seed):
    rng = random.Random(seed)
    layer = _Layer(seed=seed)
    for _ in range(30):
        for _ in range(rng.randint(0, 3)):
            layer.enqueue_kernel()
            layer.ops.run_some(rng, rng.randint(0, 5))
        load = np.array([rng.choice([0, 1, 5, 40, 60]) for _ in range(layer.S)], float)
        layer.move(_random_target(layer, rng), load)
        layer.placement.check()
        layer.ops.run_some(rng, rng.randint(0, 8))
    layer.enqueue_kernel()
    layer.ops.drain()
    assert layer.kernel_errors == []
    assert layer.kernels > 30
    # the host state and the device maps agree
    fm, sm = layer.placement.maps(layer.slot_global, layer.S)
    assert layer.fast_map.tolist() == fm.tolist()
    assert layer.slow_map.tolist() == sm.tolist()


def test_hot_experts_end_in_fast_rows_with_the_store_bytes():
    layer = _Layer(S=8, F=3, spare=3)
    load = np.array([1.0, 1.0, 1.0, 0.0, 2.0, 40.0, 50.0, 60.0])
    layer.move(layer.logical.copy(), load)
    layer.ops.drain()
    p = layer.placement
    assert set(np.flatnonzero(p.tier == FAST).tolist()) == {5, 6, 7}
    for s in range(8):
        row = layer.fast[p.row[s]] if p.tier[s] == FAST else layer.pool[p.row[s]]
        assert torch.equal(row, expert_bytes(int(layer.logical[s])))
    # nothing more to do at the same load
    move = layer.move(layer.logical.copy(), load)
    assert move.plan.reads == {} and move.fast_dest == {}


def test_a_changed_slot_in_a_fast_row_keeps_its_row():
    layer = _Layer(S=6, F=2, spare=2, num_logical=10)
    load = np.array([50.0, 50.0, 1.0, 1.0, 1.0, 1.0])
    new = layer.logical.copy()
    new[0] = 9  # slot 0 gets a new expert, stays hot
    move = layer.move(new, load)
    layer.ops.drain()
    assert move.fast_dest == {0: 0}
    assert layer.placement.tier[0] == FAST and layer.placement.row[0] == 0
    assert torch.equal(layer.fast[0], expert_bytes(9))
    layer.placement.check()


def test_spare_rows_bound_the_changes_of_one_move():
    layer = _Layer(S=6, F=2, spare=2, num_logical=12)
    new = layer.logical.copy()
    new[:3] = [9, 10, 11]
    with pytest.raises(RowsUnavailable, match="spare"):
        layer.placement.plan(layer.logical, new, np.ones(6), 1.1)


def test_a_failed_fill_returns_its_rows():
    layer = _Layer()
    new = layer.logical.copy()
    new[4] = 11
    plan = layer.placement.plan(layer.logical, new, np.ones(layer.S), 1.1)

    def broken(reads, rows):
        raise OSError("drive gone")

    free = layer.placement.pool.free_count
    with pytest.raises(OSError):
        layer.placement.prepare(plan, broken)
    assert layer.placement.pool.free_count == free


def test_demote_promote_swap_leave_every_slot_mapped_once():
    layer = _Layer(S=6, F=2, spare=2, num_logical=6)
    p = layer.placement
    fast_slots = np.flatnonzero(p.is_fast()).tolist()
    new_fast = p.is_fast().copy()
    new_fast[fast_slots[0]] = False  # demote one, no promotion: a fast row frees
    plan = p.plan_from_set(layer.logical, layer.logical, new_fast)
    move = p.prepare(plan, layer.fill)
    p.commit(
        move, layer.ops, slot_global=layer.slot_global, num_global=layer.S,
        fast_map=layer.fast_map, slow_map=layer.slow_map, copy_rows=layer.copy_rows,
    )  # fmt: skip
    assert len(p.fast_free) == 1
    # promote another slot into the free row
    target = int(np.flatnonzero(p.tier == POOL)[0])
    new_fast = p.is_fast().copy()
    new_fast[target] = True
    plan = p.plan_from_set(layer.logical, layer.logical, new_fast)
    move = p.prepare(plan, layer.fill)
    p.commit(
        move, layer.ops, slot_global=layer.slot_global, num_global=layer.S,
        fast_map=layer.fast_map, slow_map=layer.slow_map, copy_rows=layer.copy_rows,
    )  # fmt: skip
    layer.enqueue_kernel()
    layer.ops.drain()
    assert layer.kernel_errors == []
    assert p.fast_free == []
    # no free fast row: another promotion cannot be placed
    new_fast = p.is_fast().copy()
    new_fast[int(np.flatnonzero(p.tier == POOL)[0])] = True
    plan = p.plan_from_set(layer.logical, layer.logical, new_fast)
    with pytest.raises(RowsUnavailable, match="fast row"):
        p.prepare(plan, layer.fill)


def _errors_over_seeds(monkeypatch, patch) -> int:
    total = 0
    for seed in range(40):
        with monkeypatch.context() as m:
            patch(m)
            rng = random.Random(seed)
            layer = _Layer(seed=seed)
            for _ in range(30):
                for _ in range(rng.randint(0, 3)):
                    layer.enqueue_kernel()
                    layer.ops.run_some(rng, rng.randint(0, 5))
                load = np.array(
                    [rng.choice([0, 1, 5, 40, 60]) for _ in range(layer.S)], float
                )
                layer.move(_random_target(layer, rng), load)
                layer.ops.run_some(rng, rng.randint(0, 8))
            layer.ops.drain()
            total += len(layer.kernel_errors)
    return total


def test_the_ordering_rules_are_what_keeps_kernels_correct(monkeypatch):
    """The schedules above do find a violation when a rule is dropped."""
    real_release = RowAllocator.release
    assert _errors_over_seeds(monkeypatch, lambda m: None) == 0
    # R1 for pool rows: no quarantine
    assert (
        _errors_over_seeds(
            monkeypatch,
            lambda m: m.setattr(
                RowAllocator,
                "release",
                lambda self, rows, event: real_release(self, rows, None),
            ),
        )
        > 0
    )
    # R1 for fast rows: the side stream does not wait for the drain event
    assert (
        _errors_over_seeds(
            monkeypatch,
            lambda m: m.setattr(SimOps, "side_wait", lambda self, event: None),
        )
        > 0
    )
    # R2: the compute stream does not wait for the copies
    assert (
        _errors_over_seeds(
            monkeypatch,
            lambda m: m.setattr(SimOps, "compute_wait", lambda self, event: None),
        )
        > 0
    )
