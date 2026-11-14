# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB changed while serving: pause, resume, a rearrangement on demand, the
settings, the state file on demand, injected allocation failures."""

import json
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.rebalance_execute as rebalance_mod
import vllm.distributed.eplb.runtime_control as control
from vllm.distributed.eplb import fault_injection

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)


@pytest.fixture(autouse=True)
def _no_faults():
    fault_injection.clear()
    yield
    fault_injection.clear()


def _eplb(monkeypatch, **kwargs):
    kwargs.setdefault("step_interval", 10)
    kwargs.setdefault("window_size", 4)
    state = single_rank_eplb(monkeypatch, **kwargs)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    monkeypatch.setattr(
        control, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    model = FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2)
    add_model(state, model)
    return state, model


def _model_state(state):
    return next(iter(state.model_states.values()))


def _hot(state, loads=((1, 1, 50, 1), (1, 1, 1, 70))):
    window = _model_state(state).expert_load_window
    window.zero_()
    window[:, :, :4] = torch.tensor(loads, dtype=window.dtype)


def _run_steps(state, n: int) -> list[int]:
    """Step n times; the steps at which a rearrangement started."""
    started = []
    for i in range(n):
        before = state.rearrangements
        _hot(state)
        state.step()
        if state.rearrangements != before:
            started.append(i)
    return started


def test_unused_surface_rearranges_on_the_interval(monkeypatch):
    state, model = _eplb(monkeypatch)
    # add_model starts at 3/4 of the interval: the first one comes early.
    first = state.expert_rearrangement_step_interval - state.expert_rearrangement_step
    assert _run_steps(state, 25) == [first - 1, first - 1 + 10, first - 1 + 20]
    assert physical_rows_hold(model, state)
    assert state.window_fill <= state.expert_load_window_size


def test_pause_skips_intervals_and_resume_restores_them(monkeypatch):
    state, model = _eplb(monkeypatch)
    before = _model_state(state).physical_to_logical_map.clone()
    state.rearrangements_paused = True
    assert _run_steps(state, 25) == []
    assert state.paused_intervals == 3
    assert torch.equal(_model_state(state).physical_to_logical_map, before)
    state.rearrangements_paused = False
    assert len(_run_steps(state, 10)) == 1
    assert physical_rows_hold(model, state)


def test_a_rearrangement_on_demand_starts_at_the_next_step(monkeypatch):
    state, model = _eplb(monkeypatch, step_interval=1000)
    state.expert_rearrangement_step = 0
    assert control.check_trigger(state) is None
    old = control.trigger(state)
    assert old == 0
    assert _run_steps(state, 1) == [0]
    assert physical_rows_hold(model, state)
    # Paused: refused, nothing changed.
    state.rearrangements_paused = True
    assert "paused" in control.check_trigger(state)


def test_an_async_rearrangement_in_flight_refuses_policy_changes(monkeypatch):
    state, _ = _eplb(monkeypatch, use_async=True, policy="incremental")
    _model_state(state).rebalanced = True
    assert "in flight" in control.check_trigger(state)
    assert "in flight" in control.check_settings(state, {"policy": "incremental"})
    assert "in flight" in control.check_settings(state, {"max_moves_per_rank": 2})
    # The interval and the window may change mid-flight (the worker reads a
    # snapshot of the window).
    assert control.check_settings(state, {"step_interval": 50}) is None


@pytest.mark.parametrize(
    "changes, refusal",
    [
        ({"step_interval": -1}, "integer >= 0"),
        ({"window_size": -1}, "positive integer"),
        ({"max_moves_per_rank": True}, "positive integer"),
        ({"policy": "magic"}, "unknown policy"),
        ({"speed": 3}, "unknown EPLB settings"),
    ],
)
def test_bad_settings_are_refused(monkeypatch, changes, refusal):
    state, _ = _eplb(monkeypatch)
    assert refusal in control.check_settings(state, changes)


def test_settings_change_and_go_back(monkeypatch):
    state, model = _eplb(monkeypatch)
    _hot(state)
    window = _model_state(state).expert_load_window
    state.expert_load_window_step = 1  # oldest record at index 1
    window[0, 0, 0] = 7  # the newest record
    changes = control.apply_settings(
        state, {"step_interval": 30, "window_size": 2, "policy": "incremental"}
    )
    assert changes == {
        "step_interval": [10, 30],
        "window_size": [4, 2],
        "policy": ["default", "incremental"],
    }
    new_window = _model_state(state).expert_load_window
    assert new_window.shape[0] == 2
    assert int(new_window[1, 0, 0]) == 7  # newest kept, in order
    assert state.expert_load_window_step == 0
    assert len(state._window_token_weights) == 2
    assert state.parallel_config.eplb_config.step_interval == 30
    assert state.policy.max_moves_per_rank == 8
    # The rearrangement after the change runs on the new settings.
    state.expert_rearrangement_step = 29
    assert _run_steps(state, 1) == [0]
    assert physical_rows_hold(model, state)
    back = control.apply_settings(
        state, {"step_interval": 10, "window_size": 4, "policy": "default"}
    )
    assert back["window_size"] == [2, 4]
    assert control.settings(state)["policy"] == "default"


def test_max_moves_beyond_the_transfer_buffer_is_refused(monkeypatch):
    state, _ = _eplb(monkeypatch, policy="incremental", max_moves_per_rank=2)
    rows = _model_state(state).expert_buffer[0].shape[0]
    assert rows == 2  # sized for the bounded policy
    assert "above the 2 rows" in control.check_settings(
        state, {"max_moves_per_rank": 3}
    )
    assert control.check_settings(state, {"max_moves_per_rank": 1}) is None
    assert "may move every slot" in control.check_settings(state, {"policy": "default"})


def test_save_and_load_the_state_file(monkeypatch, tmp_path):
    state, model = _eplb(monkeypatch, step_interval=1000)
    path = str(tmp_path / "eplb.json")
    loads = [[1, 2, 3, 400], [500, 1, 1, 1]]
    _model_state(state).last_expert_load.copy_(torch.tensor(loads))
    assert control.check_save(state, path, "last") is None
    result = control.save(state, path, "last")
    assert result == {"path": path, "source": "last", "layers": 2}
    with open(path) as handle:
        saved = json.load(handle)
    assert saved["model.layers.0.mlp.experts"] == loads[0]

    _model_state(state).last_expert_load.fill_(1)
    assert control.check_load(state, path) is None
    result = control.load(state, path)
    assert result["layers"] == 2
    assert state.next_uses_saved_load
    # The next step places by the file's loads, not the window.
    _model_state(state).expert_load_window.zero_()
    state.step()
    assert state.rearrangements == 1
    assert not state.next_uses_saved_load
    assert physical_rows_hold(model, state)
    p2l = _model_state(state).physical_to_logical_map
    # The redundant slots replicate the file's hottest expert of each layer
    # (the window, all zeros, would not move anything).
    assert set(p2l[0].tolist()).issuperset({0, 1, 2, 3})
    assert p2l[0].tolist().count(3) == 3
    assert p2l[1].tolist().count(0) == 3


def test_load_is_refused_without_a_usable_file(monkeypatch, tmp_path):
    state, _ = _eplb(monkeypatch)
    assert "does not exist" in control.check_load(state, str(tmp_path / "none"))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"elsewhere.layers.0": [1, 2]}))
    assert "no loads" in control.check_load(state, str(other))
    assert "no path" in control.check_load(state, None)


def test_load_can_be_undone(monkeypatch, tmp_path):
    state, _ = _eplb(monkeypatch, step_interval=1000)
    path = tmp_path / "eplb.json"
    path.write_text(json.dumps({"model.layers.0.mlp.experts": [9, 9, 9, 9]}))
    step = state.expert_rearrangement_step
    result = control.load(state, str(path))
    control.unload(state, result["undo"])
    assert state.expert_rearrangement_step == step
    assert not state.next_uses_saved_load
    assert _model_state(state).last_expert_load[0].tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("stage", ["load", "plan"])
def test_an_injected_failure_skips_one_rearrangement(monkeypatch, stage):
    state, model = _eplb(monkeypatch)
    before = _model_state(state).physical_to_logical_map.clone()
    fault_injection.arm(stage, 1)
    state.rearrange()
    assert state.skipped_rearrangements == 1
    assert torch.equal(_model_state(state).physical_to_logical_map, before)
    assert fault_injection.armed() == {}
    assert fault_injection.fired()[stage] >= 1
    _hot(state)
    state.rearrange()  # disarmed after one shot
    assert state.skipped_rearrangements == 1
    assert physical_rows_hold(model, state)


def test_an_injected_failure_does_not_touch_a_profile_run(monkeypatch):
    state, _ = _eplb(monkeypatch)
    fault_injection.arm("load", 1)
    state.rearrange(is_profile=True)
    assert fault_injection.armed() == {"load": 1}


def test_copy_rows_falls_back_to_row_copies_without_memory(monkeypatch):
    """The one allocation of the row copy (the index upload) fails: the rows
    are copied one by one, with the same result."""
    monkeypatch.setattr(rebalance_mod, "_rows_kernel_applies", lambda tensors: True)

    def no_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(rebalance_mod, "async_tensor_h2d", no_memory)
    dst = [torch.zeros(4, 3), torch.zeros(4, 2)]
    src = [torch.arange(12.0).view(4, 3), torch.arange(8.0).view(4, 2)]
    rebalance_mod.copy_rows(dst, [0, 2], src, [3, 1])
    assert torch.equal(dst[0][0], src[0][3]) and torch.equal(dst[0][2], src[0][1])
    assert torch.equal(dst[1][0], src[1][3])
    assert dst[0][1].abs().sum() == 0

    # Injected at the same point.
    monkeypatch.setattr(rebalance_mod, "async_tensor_h2d", lambda *a, **k: 1 / 0)
    fault_injection.arm("copy_rows", 1)
    dst = [torch.zeros(4, 3), torch.zeros(4, 2)]
    rebalance_mod.copy_rows(dst, [1], src, [0])
    assert torch.equal(dst[0][1], src[0][0])


def test_the_trace_turns_on_and_off_while_serving(monkeypatch, tmp_path):
    import vllm.v1.worker.runtime_control as wrc

    state, _ = _eplb(monkeypatch)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=state))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    path = str(tmp_path / "trace")
    assert (
        "needs a path"
        in wrc.handle(worker, "check", "trace.set", {"enabled": True})["refusal"]
    )
    reply = wrc.handle(
        worker, "apply", "trace.set", {"enabled": True, "path": path, "interval": 2}
    )
    assert reply["ok"], reply
    trace = state._trace
    assert trace is not None and trace.interval == 2
    assert _model_state(state).map_change_hook == trace.before_map_change
    _run_steps(state, 4)
    trace.drain()
    assert (tmp_path / "trace.rank0").stat().st_size > 0
    reply = wrc.handle(worker, "apply", "trace.set", {"rotate_mb": 1})
    assert state._trace is trace and trace.rotate_bytes == 1 << 20
    reply = wrc.handle(worker, "apply", "trace.set", {"enabled": False})
    assert reply["changes"]["enabled"] == [True, False]
    assert state._trace is None
    assert _model_state(state).map_change_hook is None
    _run_steps(state, 3)  # stepping without a trace again


def test_the_summary_reads_the_state(monkeypatch):
    state, _ = _eplb(monkeypatch)
    _run_steps(state, 4)
    out = control.summary(state)
    assert out["settings"] == {
        "step_interval": 10,
        "window_size": 4,
        "max_moves_per_rank": 8,
        "policy": "default",
        # the tier pass's settings (off by default)
        "tier_interval": 0,
        "tier_max_moves": None,
        "tier_hot": 1.5,
        "tier_warm": 0.5,
        # what a periodic move must be worth, and when the interval counts
        "min_imbalance": 0.0,
        "move_min_gain": None,
        "interval_from_end": False,
    }
    assert out["rearrangements"] == 1
    assert out["in_flight"] is None
    model = out["models"][0]
    assert model["layers"] == 2 and model["physical_experts"] == 6
    assert model["replicas_max"] == 3  # both redundant slots on the hot expert
    assert 0 < model["balancedness"] <= 1
    json.dumps(out)  # what the workers send back must serialize
