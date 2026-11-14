# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What an EPLB rearrangement costs a serving engine, and the settings that
bound it: periodic rearrangements off by default, a rearrangement started
only when the window is imbalanced enough, a smallest gain per swap, the
interval counted from the end of a move, and the per-move debug line."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_mod
import vllm.distributed.eplb.runtime_control as control
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.policy import IncrementalEplbPolicy
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.v1 import runtime_debug

from .eplb_single_rank import _Event, physical_rows_hold
from .test_eplb_runtime_control import _eplb, _model_state, _run_steps
from .test_eplb_store_tiers import _build, _holds, _run_async_cycle, _window

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _debug_off():
    yield
    for part in list(runtime_debug.state()["on"]):
        runtime_debug.disarm(part, "test")


# -- the default: EPLB observes ------------------------------------------


def test_the_default_interval_rearranges_nothing_but_records_the_window(
    monkeypatch,
):
    assert EPLBConfig().step_interval == 0
    state, model = _eplb(monkeypatch, step_interval=0)
    assert state.expert_rearrangement_step_interval == 0
    assert _run_steps(state, 50) == []
    assert state._window_records_current_step()
    assert control.summary(state)["steps_until_next"] is None
    assert control.summary(state)["periodic"] is False
    # Asked for, it rearranges at the next step, and only once.
    assert control.check_trigger(state) is None
    control.trigger(state)
    assert _run_steps(state, 20) == [0]
    assert physical_rows_hold(model, state)


def test_an_explicit_interval_rearranges_as_before(monkeypatch):
    state, _ = _eplb(monkeypatch)  # step_interval 10, min_imbalance 0
    first = state.expert_rearrangement_step_interval - state.expert_rearrangement_step
    assert _run_steps(state, 25) == [first - 1, first - 1 + 10, first - 1 + 20]
    assert state.low_imbalance_skips == 0


def test_a_requested_rearrangement_can_be_taken_back(monkeypatch):
    state, _ = _eplb(monkeypatch, step_interval=0)
    old = control.trigger(state)
    control.untrigger(state, old)
    assert _run_steps(state, 5) == []


def test_the_interval_can_be_switched_on_and_off_while_serving(monkeypatch):
    state, _ = _eplb(monkeypatch, step_interval=0)
    assert control.check_settings(state, {"step_interval": 0}) is None
    assert "integer >= 0" in control.check_settings(state, {"step_interval": -1})
    assert control.apply_settings(state, {"step_interval": 5}) == {
        "step_interval": [0, 5]
    }
    state.expert_rearrangement_step = 0
    assert _run_steps(state, 11) == [4, 9]
    control.apply_settings(state, {"step_interval": 0})
    assert _run_steps(state, 20) == []


# -- min_imbalance -------------------------------------------------------


def test_min_imbalance_skips_periodic_rearrangements_not_requested_ones(
    monkeypatch,
):
    # One rank: the busiest rank is the mean, imbalance 1.0.
    state, model = _eplb(monkeypatch, min_imbalance=0.2)
    assert _run_steps(state, 25) == []
    assert state.low_imbalance_skips == 3
    assert state.last_imbalance == pytest.approx(1.0)
    control.trigger(state)
    assert _run_steps(state, 1) == [0]
    assert physical_rows_hold(model, state)


def test_the_window_imbalance_is_the_busiest_rank_over_the_mean(monkeypatch):
    state, _ = _eplb(monkeypatch)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    monkeypatch.setattr(
        eplb_state_mod, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    ms = _model_state(state)
    # 2 layers, 6 slots (3 per rank) over 4 logical experts, 2 replicated.
    ms.physical_to_logical_map.copy_(torch.tensor([[0, 1, 2, 3, 0, 1]] * 2))
    load = torch.tensor([[10.0, 10.0, 40.0, 0.0], [0.0, 0.0, 0.0, 60.0]])
    # rank 0: 5 + 5 + 40 = 50 and 0 + 0 + 0; rank 1: 0 + 5 + 5 and 60
    # totals 50 and 70, mean 60
    assert state._window_imbalance([load]) == pytest.approx(70 / 60)
    assert state._window_imbalance([torch.zeros_like(load)]) == 1.0


def test_min_imbalance_and_the_other_settings_change_and_go_back(monkeypatch):
    state, _ = _eplb(monkeypatch, policy="incremental")
    changes = control.apply_settings(
        state,
        {
            "min_imbalance": 0.1,
            "move_min_gain": 0.05,
            "interval_from_end": True,
        },
    )
    assert changes == {
        "min_imbalance": [0.0, 0.1],
        "move_min_gain": [None, 0.05],
        "interval_from_end": [False, True],
    }
    assert state.policy.min_gain == 0.05
    assert state.policy.max_moves_per_rank == 8
    back = control.apply_settings(
        state,
        {
            "min_imbalance": 0.0,
            "move_min_gain": None,
            "interval_from_end": False,
        },
    )
    assert back["move_min_gain"] == [0.05, None]
    assert state.policy.min_gain == IncrementalEplbPolicy.min_gain


@pytest.mark.parametrize(
    "changes, refusal",
    [
        ({"min_imbalance": -0.1}, "min_imbalance"),
        ({"move_min_gain": "big"}, "move_min_gain"),
        ({"interval_from_end": 1}, "interval_from_end"),
    ],
)
def test_bad_new_settings_are_refused(monkeypatch, changes, refusal):
    state, _ = _eplb(monkeypatch)
    assert refusal in control.check_settings(state, changes)


def test_the_gain_per_swap_cannot_change_mid_flight(monkeypatch):
    state, _ = _eplb(monkeypatch, use_async=True, policy="incremental")
    _model_state(state).rebalanced = True
    assert "in flight" in control.check_settings(state, {"move_min_gain": 0.1})
    assert control.check_settings(state, {"min_imbalance": 0.1}) is None


def test_the_new_settings_are_not_compile_factors():
    base = EPLBConfig(step_interval=300).compile_factors()
    tuned = EPLBConfig(
        step_interval=300,
        min_imbalance=0.1,
        move_min_gain=0.02,
        interval_from_end=True,
    ).compile_factors()
    assert base == tuned


# -- move_min_gain -------------------------------------------------------


def _near_balanced():
    # 2 ranks x 2 slots, 4 experts: ranks carry 53 and 47 (mean 50); the best
    # swap lowers the busiest rank to 51, a gain of 4 % of the mean.
    load = torch.tensor([[28.0, 25.0, 24.0, 23.0]])
    old = torch.tensor([[0, 1, 2, 3]])
    return load, old


def test_move_min_gain_leaves_a_near_balanced_layer_alone():
    load, old = _near_balanced()
    default = IncrementalEplbPolicy.configured(8)
    picky = IncrementalEplbPolicy.configured(8, 0.05)
    assert picky.min_gain == 0.05 and default.min_gain == 0.001
    kept = picky.rebalance_experts(load, 4, 1, 1, 2, old)
    assert torch.equal(kept, old)
    moved = default.rebalance_experts(load, 4, 1, 1, 2, old)
    assert not torch.equal(moved, old)


# -- interval_from_end ---------------------------------------------------


@pytest.mark.parametrize(
    "from_end, tier_only, expect",
    [(True, False, 0), (False, False, 7), (True, True, 7)],
)
def test_the_interval_counts_from_the_end_of_a_rearrangement(
    monkeypatch, from_end, tier_only, expect
):
    state, _ = _eplb(monkeypatch, use_async=True, interval_from_end=from_end)
    ms = _model_state(state)
    ms.eplb_stats = SimpleNamespace(tier_only=tier_only)
    ms.move_cost = {"steps": 3.0}
    state.expert_rearrangement_step = 7
    state._after_async_cycle(ms, 0)
    assert state.expert_rearrangement_step == expect
    assert ms.move_cost == {}


# -- the async worker's wait, and the move line ----------------------------


def test_the_cpu_gpu_event_host_wait_then_wait_is_one_wait(monkeypatch):
    monkeypatch.setattr(torch, "Event", _Event)
    event = CpuGpuEvent()
    event.record()
    event.wait_host()  # returns: recorded
    assert event._recorded.is_set()
    event.wait()  # the device part; clears
    assert not event._recorded.is_set()
    event.record()  # re-armed: a second record is allowed


def test_an_async_move_fills_the_cost_counters_and_the_debug_line(
    monkeypatch, tmp_path
):
    env, _, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    loads = [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]]
    _window(model_state, loads)
    runtime_debug.arm("eplb", 60, 100)
    _run_async_cycle(state, model_state)
    p0, p1 = env.tiers[0].placement, env.tiers[1].placement
    assert set(np.flatnonzero(p0.tier == 0).tolist()) == {5, 6, 7}
    assert set(np.flatnonzero(p1.tier == 0).tolist()) == {0, 1, 2}
    for i in range(len(env.tiers)):
        env.tiers[i].placement.check()
        for slot in range(len(env.tiers[i].slot_logical)):
            assert _holds(env, i, slot, slot), (i, slot)
    cost = dict(model_state.move_cost)
    assert cost["commit_steps"] == len(env.tiers)
    assert cost["ready_checks"] >= len(env.tiers)
    assert cost["units"] >= 3 and cost["bytes"] > 0 and cost["card_bytes"] > 0
    assert "plan_ms" in cost and "prepare_ms" in cost and "flag_ms" in cost
    state._after_async_cycle(model_state, 0)
    line = next(x for x in log.lines if x.startswith("[debug eplb] move ("))
    assert "device wait" in line and "units read" in line
    assert "ready checks" in line and "copied to card rows" in line
    assert model_state.move_cost == {}
    tstore.reset_sessions()


def test_no_move_line_when_debug_is_off(monkeypatch, tmp_path):
    env, _, state, model_state, log = _build(monkeypatch, tmp_path, spare=4)
    _window(model_state, [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]])
    _run_async_cycle(state, model_state)
    state._after_async_cycle(model_state, 0)
    assert not [x for x in log.lines if x.startswith("[debug eplb]")]
    assert model_state.move_cost == {}
    tstore.reset_sessions()


def test_min_imbalance_skips_an_async_rearrangement_before_the_worker(
    monkeypatch, tmp_path
):
    env, _, state, model_state, _ = _build(
        monkeypatch, tmp_path, spare=4, min_imbalance=0.5
    )
    _window(model_state, [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]])
    before = [t.placement.tier.copy() for t in env.tiers]
    state.rearrange(periodic=True)
    assert not model_state.rebalanced
    assert state.low_imbalance_skips == 1
    for t, b in zip(env.tiers, before):
        assert (t.placement.tier == b).all()
    tstore.reset_sessions()
