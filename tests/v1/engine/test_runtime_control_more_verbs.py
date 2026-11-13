# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The governor, drafter and scheduler verbs of the runtime control surface:
applied between steps in the engine core, or refused (restart_required with
the reason) with nothing changed."""

from types import SimpleNamespace

import pytest

import vllm.v1.engine.runtime_control as erc
import vllm.v1.worker.runtime_control as wrc

from ..core.utils import create_requests, create_scheduler
from .test_runtime_control_engine import _Engine


@pytest.fixture(autouse=True)
def _ranks(monkeypatch):
    monkeypatch.setattr(wrc, "_ep_rank", lambda worker: worker.rank)


# ---------------------------------------------------------------- governor


def test_the_governor_is_made_on_first_use_and_tuned_live():
    engine = _Engine()
    control = erc.RuntimeControl(engine)
    assert control.state()["governor"] == {"mode": "off"}
    assert control.apply("governor.state", {})["governor"] == {"mode": "off"}
    reply = control.apply(
        "governor.set", {"mode": "dry_run", "pressure_window_s": 20.0}
    )
    assert reply["changes"] == {
        "mode": ["off", "dry_run"],
        "pressure_window_s": [30.0, 20.0],
    }
    assert engine._kv_governor.settings.mode == "dry_run"
    state = control.state()["governor"]
    assert state["mode"] == "dry_run" and state["level"] == "neutral"
    with pytest.raises(erc.ControlRefused, match="hysteresis"):
        control.apply("governor.set", {"usage_low": 0.95})
    assert engine._kv_governor.settings.usage_low == 0.6
    control.apply("governor.set", {"mode": "off"})
    assert engine._kv_governor.seconds_to_next_event() is None


def test_the_governor_needs_a_resizable_kv_cache():
    engine = _Engine()

    def not_resizable():
        raise ValueError("Resizing the KV cache needs the extensible KV cache.")

    engine._check_kv_cache_resizable = not_resizable
    with pytest.raises(erc.RestartRequired, match="restart_required: .*extensible"):
        erc.RuntimeControl(engine).apply("governor.set", {"mode": "on"})
    assert getattr(engine, "_kv_governor", None) is None
    # tuning while off is allowed
    erc.RuntimeControl(engine).apply("governor.set", {"usage_high": 0.85})


# ---------------------------------------------------------------- drafter


def _engine_with(scheduler):
    engine = _Engine()
    engine.scheduler = scheduler
    engine.vllm_config = scheduler.vllm_config
    return engine


def test_the_drafter_moves_within_its_start_size():
    scheduler = create_scheduler(num_speculative_tokens=3)
    control = erc.RuntimeControl(_engine_with(scheduler))
    reply = control.apply("spec.set", {"num_speculative_tokens": 1})
    assert reply["changes"] == {"num_speculative_tokens": [3, 1]}
    assert scheduler.num_spec_tokens_now == 1 and scheduler.spec_tokens_cap == 1
    # the next step asks the runner for one draft token per request
    for request in create_requests(num_requests=2, num_tokens=8):
        scheduler.add_request(request)
    assert scheduler.schedule().num_spec_tokens_to_schedule == 1
    with pytest.raises(erc.RestartRequired, match="above 3"):
        control.apply("spec.set", {"num_speculative_tokens": 4})
    assert scheduler.num_spec_tokens_now == 1
    control.apply("spec.set", {"enabled": False})
    assert scheduler.num_spec_tokens_now == 0
    assert scheduler.schedule().num_spec_tokens_to_schedule == 0
    control.apply("spec.set", {"enabled": True})
    assert scheduler.spec_tokens_cap is None and scheduler.num_spec_tokens_now == 3
    with pytest.raises(erc.ControlRefused, match="unknown drafter"):
        control.apply("spec.set", {"method": "ngram"})


def test_a_lower_cap_pads_decode_steps_to_the_new_width():
    scheduler = create_scheduler(num_speculative_tokens=3)
    scheduler.spec_tokens_cap = 1
    assert scheduler.num_spec_tokens_now == 1
    scheduler.spec_tokens_cap = None
    assert scheduler.num_spec_tokens_now == 3


def test_acceptance_since_start_and_its_reset():
    scheduler = create_scheduler(num_speculative_tokens=3)
    scheduler.log_stats = True
    control = erc.RuntimeControl(_engine_with(scheduler))
    scheduler.make_spec_decoding_stats(None, 3, 2, None, "a")
    scheduler.make_spec_decoding_stats(None, 3, 0, None, "b")
    acc = control.state()["spec"]["acceptance"]
    assert acc["drafts"] == 2 and acc["draft_tokens"] == 6
    assert acc["accepted_tokens"] == 2
    assert acc["acceptance_rate"] == pytest.approx(2 / 6)
    assert acc["mean_acceptance_length"] == pytest.approx(2.0)
    assert acc["per_position"] == [0.5, 0.5, 0.0]
    control.apply("spec.set", {"reset_stats": True})
    assert control.state()["spec"]["acceptance"] is None


def test_without_a_drafter_the_drafter_verb_needs_a_restart():
    scheduler = create_scheduler()
    control = erc.RuntimeControl(_engine_with(scheduler))
    assert control.state()["spec"] == {"drafter": False}
    with pytest.raises(erc.RestartRequired, match="--speculative-config"):
        control.apply("spec.set", {"num_speculative_tokens": 1})


# ---------------------------------------------------------------- scheduler


def test_the_running_request_cap_moves_below_max_num_seqs():
    scheduler = create_scheduler(max_num_seqs=8)
    control = erc.RuntimeControl(_engine_with(scheduler))
    reply = control.apply("sched.set", {"max_num_active_seqs": 2})
    assert reply["changes"] == {"max_num_active_seqs": [8, 2]}
    for request in create_requests(num_requests=4, num_tokens=8):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2 and len(scheduler.waiting) == 2
    # raised again: the waiting ones are admitted at the next step
    control.apply("sched.set", {"max_num_active_seqs": 8})
    assert len(scheduler.schedule().scheduled_new_reqs) == 2
    with pytest.raises(erc.RestartRequired, match="--max-num-seqs 9"):
        control.apply("sched.set", {"max_num_active_seqs": 9})
    with pytest.raises(erc.RestartRequired, match="max_num_seqs"):
        control.apply("sched.set", {"max_num_seqs": 4})


def test_the_token_budget_per_step_moves_below_max_num_batched_tokens():
    scheduler = create_scheduler(max_num_batched_tokens=1024)
    control = erc.RuntimeControl(_engine_with(scheduler))
    control.apply("sched.set", {"max_num_scheduled_tokens": 16})
    for request in create_requests(num_requests=1, num_tokens=64):
        scheduler.add_request(request)
    assert scheduler.schedule().total_num_scheduled_tokens == 16
    with pytest.raises(erc.RestartRequired, match="max_num_batched_tokens 1024"):
        control.apply("sched.set", {"max_num_scheduled_tokens": 2048})
    with pytest.raises(erc.ControlRefused, match=">= 1"):
        control.apply("sched.set", {"max_num_scheduled_tokens": 0})
    assert scheduler.max_num_scheduled_tokens == 16
    state = control.state()["sched"]
    assert state["max_num_scheduled_tokens"] == 16
    assert state["max_num_batched_tokens"] == 1024


def test_the_new_operations_are_listed():
    state = erc.RuntimeControl(_engine_with(create_scheduler())).state()
    for op in (
        "governor.set",
        "governor.state",
        "spec.set",
        "sched.set",
        "defaults.set",
    ):
        assert op in state["operations"]


def test_restart_required_text():
    e = erc.RestartRequired("why")
    assert str(e) == "restart_required: why" and isinstance(e, erc.ControlRefused)


def test_the_engine_core_wakes_and_steps_the_governor(monkeypatch):
    """The busy loop's hook: a failing governor step is logged, a worker
    timeout fails the executor as the other between-steps hooks do."""
    from vllm.v1.engine.core import EngineCoreProc

    failed = []
    proc = SimpleNamespace(
        _kv_governor=SimpleNamespace(advance=lambda: 1 / 0),
        _fail_kv_resize_executor=lambda: failed.append(True),
    )
    EngineCoreProc._advance_kv_governor(proc)  # ZeroDivisionError logged
    assert failed == []

    def timeout():
        raise TimeoutError

    proc._kv_governor = SimpleNamespace(advance=timeout)
    with pytest.raises(TimeoutError):
        EngineCoreProc._advance_kv_governor(proc)
    assert failed == [True]
