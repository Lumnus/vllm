# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The churn limiter inside EPLB's own tier pass: a swap only when its gain
repays its move (`tier_min_gain`), a pass only when its cut moves enough
(`tier_min_changes`), a read cap per window (`tier_window_moves`), and
`interval_from_end` restarting the tier count at a pass's end so passes do
not chain. Every setting unset: the passes as before.

The plan rules on a bare ``TierPlacement``; the passes on the one-rank
``EplbState`` over real tiers of tests/distributed/test_eplb_cpu_experts_control.py."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.distributed.eplb.async_worker as async_worker_mod
import vllm.distributed.eplb.runtime_control as control
from vllm.config.parallel import EPLBConfig
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import TierPlacement

from .test_eplb_cpu_experts_control import _build, _cycle, _isolated  # noqa: F401

pytestmark = pytest.mark.skip_global_cleanup


def _done(event):
    return True


def _placement():
    """4 card slots, 4 pinned, 4 on the CPU, 4 spare rows."""
    return TierPlacement(12, 4, spare=4, done=_done, num_pool_slots=4)


# -- the plan ------------------------------------------------------------------


def test_the_gain_test_leaves_near_equal_experts_where_they_are():
    p = _placement()
    old = np.arange(12)
    # card 10 each, pinned 5 each, two CPU slots a little hotter than the card
    load = np.array([10, 10, 10, 10, 5, 5, 5, 5, 12, 13, 1, 1], dtype=float)
    base = p.plan(old, old, load, 1.1)
    assert set(base.reads) >= {8, 9}  # hysteresis alone: both move
    assert p.plan(old, old, load, 1.1, min_gain=None).reads == base.reads
    assert p.plan(old, old, load, 1.1, min_gain=0).reads == base.reads
    from vllm.model_executor.offloader.expert_tier_map import entry_gains

    to_card, to_pool = entry_gains(p.tier, 4.0)
    assert to_card[8] == pytest.approx(4 * 400 / 120)
    assert to_card[4] == pytest.approx(4 * 150 / 28)
    assert to_pool == pytest.approx(4 * 250 / 92)
    held = p.plan(old, old, load, 1.1, min_gain=4.0)
    assert held.reads == {} and np.array_equal(held.new_fast, p.is_fast())
    # a CPU slot far hotter repays its move: it enters
    load[9] = 40.0
    moved = p.plan(old, old, load, 1.1, min_gain=4.0)
    assert 9 in moved.reads and 8 not in moved.reads


def test_the_change_count_follows_the_plan():
    p = _placement()
    old = np.arange(12)
    for load, gain in (
        (np.array([10, 10, 10, 10, 5, 5, 5, 5, 12, 13, 1, 1], float), None),
        (np.array([10, 10, 10, 10, 5, 5, 5, 5, 12, 13, 1, 1], float), 4.0),
        (np.array([10, 10, 10, 10, 5, 5, 5, 5, 12, 40, 1, 1], float), 4.0),
        (np.arange(12, dtype=float)[::-1].copy(), None),
    ):
        plan = p.plan(old, old, load, 1.1, min_gain=gain)
        want = p.tier_changes_wanted(load, 1.1, gain)
        assert want == len(plan.reads), (load, gain)
    # bounded like a pass's reads
    assert p.tier_changes_wanted(np.arange(12, dtype=float), 1.1, None, 1) == 1


# -- the passes ----------------------------------------------------------------


def _loads(model_state, rows):
    loads = torch.tensor(rows, dtype=torch.int32)
    window = model_state.expert_load_window
    per_slot = torch.gather(loads, 1, model_state.physical_to_logical_map.long())
    window.copy_(per_slot.to(window.dtype).expand_as(window))


HOT = [[1, 1, 1, 1, 1, 1, 60, 70], [9, 8, 7, 1, 1, 1, 1, 1]]


def _units(env):
    return sum(m.get("units", 0) for t in env.tiers for m in t.move_log)


def test_a_settled_window_skips_its_passes(monkeypatch, tmp_path):
    env, state, model_state = _build(
        monkeypatch,
        tmp_path,
        eplb_extra={"tier_interval": 2, "step_interval": 10**6},
    )
    for _ in range(40):  # converge, as before
        _loads(model_state, HOT)
        _cycle(state, model_state, state.step)
    passes = state.tier_passes
    # unset: a pass every interval although nothing moves
    for _ in range(8):
        _loads(model_state, HOT)
        _cycle(state, model_state, state.step)
    assert state.tier_passes == passes + 4 and state.tier_pass_skips == 0
    control.apply_settings(state, {"tier_min_changes": 1})
    units, passes = _units(env), state.tier_passes
    for _ in range(8):
        _loads(model_state, HOT)
        _cycle(state, model_state, state.step)
    assert state.tier_passes == passes and state.tier_pass_skips == 4
    assert state.last_tier_changes == 0 and _units(env) == units
    # a change of the load: the cut moves again, the pass runs
    _loads(model_state, [[60, 70, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 7, 8, 9]])
    for _ in range(2):
        _cycle(state, model_state, state.step)
    assert state.tier_passes == passes + 1 and state.last_tier_changes > 0
    assert control.summary(state)["tier_pass_skips"] == 4
    tstore.reset_sessions()


@pytest.mark.parametrize("gain,moves", [(None, True), (8.0, False)])
def test_the_gain_test_holds_a_noisy_window(monkeypatch, tmp_path, gain, moves):
    """Two CPU experts a third hotter than the rest, a few calls over the
    window (4 steps): the hysteresis moves them, the gain test does not."""
    noisy = [[3, 3, 3, 3, 3, 3, 4, 4], [3, 3, 3, 3, 3, 3, 4, 4]]
    env, state, model_state = _build(
        monkeypatch,
        tmp_path,
        eplb_extra={"tier_interval": 2, "step_interval": 10**6, "tier_min_gain": gain},
    )
    for _ in range(6):
        _loads(model_state, noisy)
        _cycle(state, model_state, state.step)
    assert state.tier_passes == 3
    assert (_units(env) > 0) is moves, gain
    tstore.reset_sessions()


def test_the_window_cap_bounds_the_reads(monkeypatch, tmp_path):
    env, state, model_state = _build(
        monkeypatch,
        tmp_path,
        eplb_extra={
            "tier_interval": 2,
            "step_interval": 10**6,
            "tier_max_moves": 2,
            "tier_window_moves": 1,
        },
    )
    for _ in range(16):
        _loads(model_state, HOT)
        _cycle(state, model_state, state.step)
    ledger = model_state.tier_reads
    assert ledger and state.tier_window_limited > 0
    w = state.expert_load_window_size
    for reads in ledger.values():
        for step, _ in reads:
            inside = sum(n for s, n in reads if step - w < s <= step)
            assert inside <= 1
    assert all(m.get("units", 0) <= 1 for t in env.tiers for m in t.move_log)
    tstore.reset_sessions()


def test_the_pass_args_unset_are_the_budget_alone():
    config = EPLBConfig(tier_interval=2, tier_max_moves=3)
    state = SimpleNamespace(
        parallel_config=SimpleNamespace(eplb_config=config),
        steps_total=10,
        expert_load_window_size=4,
        tier_window_limited=0,
    )
    assert async_worker_mod._tier_pass_args(state, SimpleNamespace(), 0) == {
        "read_budget": 3
    }
    config.tier_min_gain = 2.0
    config.tier_window_moves = 1
    ms = SimpleNamespace()
    async_worker_mod._tier_reads(ms, 0).append((9, 1))
    args = async_worker_mod._tier_pass_args(state, ms, 0)
    assert args == {"read_budget": 0, "min_gain": 2.0}
    assert state.tier_window_limited == 1
    state.steps_total = 13  # the read at step 9 left the window
    assert async_worker_mod._tier_pass_args(state, ms, 0)["read_budget"] == 1


def test_interval_from_end_restarts_the_tier_count_at_a_pass_end():
    """Chaining: the tier count runs while a pass is in flight, so a
    pass at least `tier_interval` steps long is followed by the next in the
    same step; with `interval_from_end` the count restarts at its end."""
    from vllm.distributed.eplb.eplb_state import EplbState

    for from_end, want in ((False, 37), (True, 0)):
        config = EPLBConfig(tier_interval=8, interval_from_end=from_end)
        state = EplbState.__new__(EplbState)
        state.parallel_config = SimpleNamespace(eplb_config=config)
        state.expert_rearrangement_step = 5
        state.tier_step = 37
        ms = SimpleNamespace(eplb_stats=SimpleNamespace(tier_only=True), move_cost={})
        state._after_async_cycle(ms, 0)
        assert state.tier_step == want
        assert state.expert_rearrangement_step == 5
    ms.eplb_stats.tier_only = False  # a full rearrangement: its own counter
    state.tier_step = 37
    state._after_async_cycle(ms, 0)
    assert state.tier_step == 37 and state.expert_rearrangement_step == 0


def test_eplb_set_takes_the_limiter_and_hashes_ignore_it(monkeypatch, tmp_path):
    env, state, model_state = _build(monkeypatch, tmp_path)
    changes = {"tier_min_gain": 4.0, "tier_min_changes": 8, "tier_window_moves": 16}
    assert control.check_settings(state, changes) is None
    assert control.apply_settings(state, changes) == {
        "tier_min_gain": [None, 4.0],
        "tier_min_changes": [0, 8],
        "tier_window_moves": [None, 16],
    }
    for bad in (
        {"tier_min_gain": -1},
        {"tier_min_changes": -1},
        {"tier_window_moves": 0},
        {"tier_min_changes": True},
    ):
        assert control.check_settings(state, bad) is not None, bad
    summary = control.summary(state)
    for key in ("tier_pass_skips", "last_tier_changes", "tier_window_limited"):
        assert key in summary
    assert EPLBConfig().compile_factors() == EPLBConfig(**changes).compile_factors()
    tstore.reset_sessions()
