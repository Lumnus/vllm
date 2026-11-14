# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`tiers.set` (the pinned bound) while EPLB moves experts: the engine core
holds it (`applied: waiting`), asks the ranks again between steps through
the card tier's controller and its one wait rule, applies it once EPLB is
done or fails it at its deadline; any other refusal is the answer at once.
The ranks' check uses EPLB's one in-flight predicate; a serving profile's
bound step waits the same way; the CLI says `waiting`. Fake engine and
ranks."""

import json
from types import SimpleNamespace

import pytest

import vllm.v1.engine.fast_tier as ftmod
from vllm.v1.engine.fast_tier import FastTierController, FastTierState
from vllm.v1.engine.runtime_control import ControlRefused, RuntimeControl

EPLB_BUSY = (
    "an async rearrangement of m is in flight at layer 3 of 48; ask after it completes"
)


class Ranks:
    """Two ranks' tiers.set, as RuntimeControl._worker_op answers it."""

    def __init__(self) -> None:
        self.bound = [22, 22]
        self.refusal: str | None = None
        self.asked = 0

    def worker_op(self, op, args):
        assert op == "tiers.set"
        self.asked += 1
        if self.refusal:
            raise ControlRefused(f"rank 0: {self.refusal}")
        old = self.bound[0]
        self.bound = [args["host_tier_bound"]] * 2
        return {
            "op": op,
            "applied": "all",
            "ranks": [
                {"rank": r, "ok": True, "changes": {"host_tier_bound": [old, 11]}}
                for r in range(2)
            ],
        }

    def rpc(self, phase, op="", args=None):
        if phase == "check":
            return [{"rank": r, "refusal": self.refusal} for r in range(2)]
        assert phase == "state"
        return [{"rank": r, "tiers": {"fast": None}} for r in range(2)]


@pytest.fixture
def setup():
    ranks = Ranks()
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=100)),
        _kv_resize=None,
    )
    control = SimpleNamespace(_rpc=ranks.rpc, _worker_op=ranks.worker_op)
    return engine, ranks, FastTierController(engine, control)


def test_applied_at_once_when_eplb_is_idle(setup):
    _, ranks, ctl = setup
    reply = ctl.set_tiers({"host_tier_bound": 11})
    assert reply["applied"] == "all" and ranks.bound == [11, 11]
    assert reply["tiers_set"]["state"] == "finished"


def test_while_eplb_moves_it_waits_then_applies(setup, caplog_vllm):
    _, ranks, ctl = setup
    ranks.refusal = EPLB_BUSY
    reply = ctl.set_tiers({"host_tier_bound": 11})
    assert reply["applied"] == "waiting" and ranks.bound == [22, 22]
    assert reply["tiers_set"]["state"] == "waiting"
    assert "in flight" in reply["tiers_set"]["message"]
    ctl.advance()  # a step boundary with the pass still in flight
    assert ctl.tier_set.state == FastTierState.WAITING
    ranks.refusal = None
    ctl.advance()  # the pass ended: applied between these two steps
    assert ctl.tier_set.state == FastTierState.FINISHED and ranks.bound == [11, 11]
    assert ctl.tier_set.asks == 3
    text = caplog_vllm.text
    assert "tiers.set {'host_tier_bound': 11} waits for EPLB" in text
    assert "applied after waiting" in text
    ctl.advance()  # nothing more to do
    assert ranks.asked == 3


def test_one_at_a_time_and_other_refusals_answer_at_once(setup):
    _, ranks, ctl = setup
    ranks.refusal = "host_tier_bound: must be 1..26, got 99"
    with pytest.raises(ControlRefused, match="1..26"):
        ctl.set_tiers({"host_tier_bound": 99})
    assert ctl.tier_set is None
    ranks.refusal = EPLB_BUSY
    ctl.set_tiers({"host_tier_bound": 11})
    with pytest.raises(ControlRefused, match="one at a time"):
        ctl.set_tiers({"host_tier_bound": 12})
    with pytest.raises(ControlRefused, match="deadline_s is a number"):
        FastTierController(None, None).set_tiers({"deadline_s": "x"})


def test_the_deadline_counts_busy_time_and_fails_the_change(setup, monkeypatch):
    _, ranks, ctl = setup
    clock = {"t": 1000.0}
    monkeypatch.setattr(ftmod.time, "monotonic", lambda: clock["t"])
    ranks.refusal = EPLB_BUSY
    ctl.set_tiers({"host_tier_bound": 11, "deadline_s": 2.5})
    clock["t"] += 600  # an idle engine: one gap counts one second at most
    ctl.advance()
    assert ctl.tier_set.state == FastTierState.WAITING
    for _ in range(2):
        clock["t"] += 1.0
        ctl.advance()
    record = ctl.tier_set.record()
    assert record["state"] == "failed" and "still moving experts" in record["message"]
    assert ranks.bound == [22, 22]


def test_the_surface_routes_tiers_set_and_reports_the_wait():
    ranks = Ranks()
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks=100,
                enable_extensible_kv_cache=False,
                kv_cache_release_chunk_mib=None,
                kv_cache_grow_torch_floor_mib=None,
            )
        ),
        _kv_resize=None,
        _tower_residency=None,
        _fast_tier=None,
    )
    control = RuntimeControl(engine)
    control._rpc = ranks.rpc
    control._worker_op = ranks.worker_op
    ranks.refusal = EPLB_BUSY
    reply = control.apply("tiers.set", {"host_tier_bound": 11, "deadline_s": 30})
    assert reply["applied"] == "waiting"
    assert engine._fast_tier is not None  # the busy loop advances it
    held = control.state(["tiers"])["fast_tier"]["tier_set"]
    assert held["state"] == "waiting" and held["args"] == {"host_tier_bound": 11}
    assert held["deadline_s"] == 30
    ranks.refusal = None
    engine._fast_tier.advance()
    assert control.state(["tiers"])["fast_tier"]["tier_set"]["state"] == "finished"


def test_the_http_route_answers_202_for_waiting():
    import inspect

    from vllm.entrypoints.serve.dev.control import api_router

    assert '"waiting": 202' in inspect.getsource(api_router)


def test_the_ranks_check_uses_eplbs_one_predicate(monkeypatch):
    from vllm.v1.worker import runtime_control as wrc

    model_state = SimpleNamespace(
        rebalanced=True,
        model_name="m",
        pending_result=SimpleNamespace(layer_idx=3),
        model=SimpleNamespace(num_moe_layers=48),
    )
    eplb = SimpleNamespace(model_states={"m": model_state})
    monkeypatch.setattr(wrc, "_tiers", lambda: [object()])
    monkeypatch.setattr(wrc, "_eplb", lambda worker: eplb)
    why = wrc._check_tiers_set(None, {"host_tier_bound": 11})
    assert why == EPLB_BUSY
    assert FastTierController(None, None)._eplb_busy(why)
    model_state.rebalanced = False
    monkeypatch.setitem(
        wrc._TIER_SETTINGS,
        "host_tier_bound",
        SimpleNamespace(check=lambda v: None, get=lambda: 22, apply=lambda v: None),
    )
    assert wrc._check_tiers_set(None, {"host_tier_bound": 11}) is None


def test_a_profile_step_waits_for_the_held_bound():
    from vllm.v1.engine import serving_profiles as sp

    held = SimpleNamespace(state=FastTierState.WAITING, message="")
    replies = [{"op": "tiers.set", "applied": "waiting"}]
    control = SimpleNamespace(apply=lambda op, args: replies[0])
    engine = SimpleNamespace(_fast_tier=SimpleNamespace(tier_set=held))
    profiles = sp.ServingProfiles(engine, control=control)
    step = profiles._bound_step(11)
    assert step.run() == "wait"
    assert step.done() == "wait"
    held.state = FastTierState.FINISHED
    assert step.done() is None
    held.state, held.message = FastTierState.FAILED, "EPLB still moving"
    assert step.done() == "EPLB still moving"
    replies[0] = {"op": "tiers.set", "applied": "all"}
    assert profiles._bound_step(11).run() is None


def test_a_profile_check_lets_the_bound_wait_but_not_other_refusals():
    from vllm.v1.engine import serving_profiles as sp

    refusal = {"text": EPLB_BUSY}
    control = SimpleNamespace(
        _rpc=lambda phase, op, args: [{"rank": 0, "refusal": refusal["text"]}],
        _spec_set=None,
        _sched_set=None,
    )
    profiles = sp.ServingProfiles(SimpleNamespace(), control=control)
    profiles._check({"name": "p", "tiers": {"host_tier_bound": 11}})
    refusal["text"] = "host_tier_bound: must be 1..26, got 99"
    with pytest.raises(ControlRefused, match="1..26"):
        profiles._check({"name": "p", "tiers": {"host_tier_bound": 99}})


def test_the_cli_says_waiting(monkeypatch, capsys):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[3] / "tools" / "runtime_control.py"
    spec = importlib.util.spec_from_file_location("rc_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sent = {}

    def request(url, method, path, body, timeout):
        sent.update(path=path, body=body)
        return 202, {
            "op": "tiers.set",
            "applied": "waiting",
            "tiers_set": {"state": "waiting", "message": "waiting for EPLB: x"},
        }

    monkeypatch.setattr(cli, "_request", request)
    rc = cli.main(["tiers", "set", "--host-tier-bound", "11", "--deadline", "90"])
    assert rc == 0
    assert sent == {
        "path": "/runtime_control/tiers.set",
        "body": {"host_tier_bound": 11, "deadline_s": 90.0},
    }
    out = capsys.readouterr().out
    assert "tiers.set waiting: waiting for EPLB: x" in out
    assert json.loads(out[: out.rindex("}") + 1])["applied"] == "waiting"


def test_the_schema_lists_the_deadline_and_the_new_settings():
    from vllm.v1.engine import runtime_params as rp

    assert "deadline_s" in rp.VERBS["tiers.set"].args
    for name, verb in (
        ("eplb.tier_min_gain", "eplb.set"),
        ("eplb.tier_min_changes", "eplb.set"),
        ("eplb.tier_window_moves", "eplb.set"),
        ("cpu.stage_min_tokens", "cpu.set"),
        ("cpu.stage_copy_threads", "cpu.set"),
        ("cpu.stage_card_us", "cpu.set"),
    ):
        assert rp.PARAMS[name].verb == verb, name
