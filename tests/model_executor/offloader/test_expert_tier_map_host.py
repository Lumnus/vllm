# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The placement with host slots: slots in no row the kernels read, computed
on the CPU. Without them every plan is the one of the two-tier placement."""

import numpy as np
import pytest

from vllm.model_executor.offloader.expert_tier_map import (
    FAST,
    HOST,
    POOL,
    RowsUnavailable,
    TierPlacement,
    choose_pool_set,
)


def _done(event):
    return True


def _fill(reads, rows):
    pass


class _Ops:
    """Stream operations that record what a commit enqueues."""

    def __init__(self):
        self.copies = []
        self.writes = []

    def record_compute(self):
        return object()

    record_side = record_compute

    def side_wait(self, event):
        pass

    compute_wait = side_wait

    def side_copy(self, dst, src):
        self.copies.append((dst, src))

    def compute_write(self, dst, values):
        self.writes.append(np.array(values))

    def done(self, event):
        return True


def _commit(p: TierPlacement, move, slot_global, ops=None):
    ops = ops or _Ops()
    fast_map = np.full(len(slot_global), -1)
    slow_map = np.full(len(slot_global), -1)
    p.commit(
        move,
        ops,
        slot_global=slot_global,
        num_global=len(slot_global),
        fast_map=fast_map,
        slow_map=slow_map,
        copy_rows=lambda s, src, dst: ops.side_copy(dst, src),
    )
    return ops


def test_initial_placement_and_maps():
    p = TierPlacement(8, 3, spare=2, done=_done, num_pool_slots=2, extra_rows=1)
    assert p.num_host == 3 and p.num_pool_rows == 5
    assert p.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
    assert p.row.tolist() == [0, 1, 2, 0, 1, -1, -1, -1]
    assert sorted(p.pool.free_rows()) == [2, 3, 4]
    fast_map, slow_map = p.maps(list(range(8)), 8)
    assert fast_map.tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    # a host slot is in neither map: the kernels skip it, the CPU serves it
    assert slow_map.tolist() == [-1, -1, -1, 0, 1, -1, -1, -1]
    p.check()


def test_without_host_slots_the_placement_is_the_two_tier_one():
    a = TierPlacement(6, 2, spare=2, done=_done)
    b = TierPlacement(6, 2, spare=2, done=_done, num_pool_slots=4)
    assert a.num_host == b.num_host == 0
    assert np.array_equal(a.tier, b.tier) and np.array_equal(a.row, b.row)
    rng = np.random.default_rng(0)
    old = np.arange(6)
    for _ in range(20):
        new = old.copy()
        i, j = rng.choice(6, 2, replace=False)
        new[i], new[j] = 7 + i, 7 + j
        load = rng.random(6)
        pa = a.plan(old, new, load, 1.1)
        pb = b.plan(old, new, load, 1.1)
        assert pa.reads == pb.reads and pa.new_pool is None and pb.new_pool is None
        assert np.array_equal(pa.new_fast, pb.new_fast)


def test_a_host_slot_with_a_new_expert_reads_nothing():
    p = TierPlacement(8, 3, spare=2, done=_done, num_pool_slots=2)
    old = np.arange(8)
    new = old.copy()
    new[6] = 20  # a host slot gets another expert
    load = np.array([9, 9, 9, 5, 5, 0, 0, 0], dtype=float)
    plan = p.plan(old, new, load, 1.1)
    assert plan.reads == {}
    move = p.prepare(plan, _fill)
    assert move.end_tier[6] == HOST and move.end_row[6] == -1
    ops = _commit(p, move, list(range(8)))
    assert ops.writes == []  # no row and no map changes


def test_a_hot_host_slot_takes_a_pinned_row_and_the_cold_one_goes_to_the_host():
    p = TierPlacement(8, 3, spare=2, done=_done, num_pool_slots=2)
    old = np.arange(8)
    load = np.array([9, 9, 9, 1, 5, 7, 0, 0], dtype=float)  # slot 5 hotter than 3
    plan = p.plan(old, old, load, 1.1)
    assert plan.reads == {5: 5}  # unchanged expert into a pinned row
    assert plan.new_pool.tolist() == [False] * 3 + [False, True, True, False, False]
    move = p.prepare(plan, _fill)
    ops = _commit(p, move, list(range(8)))
    assert p.tier.tolist() == [FAST] * 3 + [HOST, POOL, POOL, HOST, HOST]
    assert p.row[3] == -1 and p.row[5] >= 2  # a spare row
    _, slow_map = ops.writes
    assert slow_map[3] == -1 and slow_map[5] == p.row[5]
    p.check()


def test_a_host_slot_promoted_to_the_card_is_read_and_copied():
    p = TierPlacement(8, 3, spare=2, done=_done, num_pool_slots=2)
    old = np.arange(8)
    load = np.array([1, 9, 9, 5, 5, 20, 0, 0], dtype=float)
    plan = p.plan(old, old, load, 1.1)
    assert plan.new_fast[5] and not plan.new_fast[0]
    assert 5 in plan.reads  # staged through a spare row
    move = p.prepare(plan, _fill)
    ops = _commit(p, move, list(range(8)))
    assert p.tier[5] == FAST and p.row[5] == 0  # slot 0's fast row
    assert ops.copies  # pool row -> fast row
    p.check()


def test_the_read_budget_falls_back_to_fewer_tier_changes():
    p = TierPlacement(10, 2, spare=1, done=_done, num_pool_slots=3)
    old = np.arange(10)
    new = old.copy()
    new[2] = 30  # a pinned slot changes: one read, the whole budget
    load = np.array([5, 5, 1, 1, 1, 9, 9, 9, 0, 0], dtype=float)
    plan = p.plan(old, new, load, 1.1)
    # one read: either the changed slot's new expert into its pinned row, or
    # (the cut by load) the new, cold expert on the CPU and the hot slot 5
    # into the pinned row instead
    assert len(plan.reads) <= 1 and plan.reads in ({2: 30}, {5: 5})
    move = p.prepare(plan, _fill)
    _commit(p, move, list(range(10)))
    p.check()


def test_choose_pool_set():
    load = np.array([0, 0, 3, 1, 9, 2], dtype=float)
    cand = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
    # free capacity is filled with the hottest
    got = choose_pool_set(load, cand, np.zeros(6, bool), 2, 1.1)
    assert np.flatnonzero(got).tolist() == [2, 4]
    # a current member stays unless an outside slot beats it by the hysteresis
    cur = np.array([0, 0, 0, 1, 0, 1], dtype=bool)
    got = choose_pool_set(load, cand, cur, 2, 10.0)
    assert np.flatnonzero(got).tolist() == [3, 5]
    got = choose_pool_set(load, cand, cur, 2, 1.1)
    assert np.flatnonzero(got).tolist() == [4, 5] or np.flatnonzero(got).tolist() == [
        2,
        4,
    ]
    # over capacity: the hottest stay
    got = choose_pool_set(load, cand, cand, 2, 1.1)
    assert np.flatnonzero(got).tolist() == [2, 4]


def test_reset_to_with_host_slots_and_the_invariant():
    p = TierPlacement(6, 2, spare=1, done=_done, num_pool_slots=2)
    tier = np.array([HOST, FAST, POOL, FAST, HOST, POOL], dtype=np.int8)
    row = np.array([-1, 0, 1, 1, -1, 0], dtype=np.int32)
    p.reset_to(tier, row)
    assert sorted(p.pool.free_rows()) == [2]
    bad = row.copy()
    bad[0] = 2  # a host slot with a row
    with pytest.raises(AssertionError):
        p.reset_to(tier, bad)
    with pytest.raises(RowsUnavailable):
        p.plan_from_set(
            np.arange(6),
            np.arange(6),
            p.is_fast(),
            np.array([1, 0, 1, 0, 1, 1], dtype=bool),
        )


def test_tier_passes_sort_the_tiers_within_their_read_budget():
    """A tier pass is a plan with the experts unchanged: the card set and the
    pinned set are cut from one order by load, as many swaps as the read
    budget takes per pass (hottest first), the rest at the next pass, until
    the order holds: the card the hottest, the pinned rows the next."""
    p = TierPlacement(16, 4, spare=2, done=_done, num_pool_slots=4)
    old = np.arange(16)
    load = np.arange(16, dtype=float)  # by id: the hottest are on the CPU
    passes = 0
    for budget in [1] * 4 + [None] * 30:
        plan = p.plan(old, old, load, 1.1, read_budget=budget)
        assert len(plan.reads) <= (budget or p.spare)
        if not plan.reads and np.array_equal(plan.new_fast, p.is_fast()):
            break
        _commit(p, p.prepare(plan, _fill), list(range(16)))
        p.check()
        passes += 1
    assert np.flatnonzero(p.tier == FAST).tolist() == [12, 13, 14, 15]
    assert np.flatnonzero(p.tier == POOL).tolist() == [8, 9, 10, 11]
    assert passes >= 4  # bounded per pass: more passes, never a larger one


def test_a_pass_is_partial_where_the_whole_choice_does_not_fit():
    """Four hot CPU slots and two spare rows: one pass moves two of them
    into pinned rows (before, the whole pinned choice fell back to none)."""
    p = TierPlacement(10, 2, spare=2, done=_done, num_pool_slots=4)
    old = np.arange(10)
    load = np.array([9, 9, 1, 1, 1, 1, 8, 8, 8, 8], dtype=float)
    plan = p.plan(old, old, load, 1.1)
    assert len(plan.reads) == 2
    assert int((plan.new_pool & np.isin(np.arange(10), [6, 7, 8, 9])).sum()) == 2
