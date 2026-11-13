# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug tracing switched on and off while serving (bounded by time and
lines), the surface's self-description, the live long-prefill threshold, and
the fast tier's port against the fast tier's own verb shape."""

import logging

import pytest

import vllm.v1.engine.runtime_control as erc
import vllm.v1.worker.runtime_control as wrc
from vllm.v1 import runtime_debug as rd
from vllm.v1.engine import runtime_params as rp
from vllm.v1.engine.serving_profiles import FastTierView, SurfaceFastTierPort

from ..core.utils import create_requests, create_scheduler
from .test_runtime_control_engine import _Engine


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(wrc, "_ep_rank", lambda worker: worker.rank)
    yield
    for part in list(rd._ACTIVE):
        rd.disarm(part, "test end")
    rd._LAST.clear()


def _engine_with(scheduler):
    engine = _Engine()
    engine.scheduler = scheduler
    engine.vllm_config = scheduler.vllm_config
    return engine


# ---------------------------------------------------------------- debug


def test_debug_checks():
    assert rd.check({"part": "eplb"}) is None
    assert "part must be" in rd.check({"part": "everything"})
    assert "name the logger" in rd.check({"part": "logger:<name>"})
    assert "duration_s" in rd.check({"part": "kv", "duration_s": 7200})
    assert "count" in rd.check({"part": "kv", "count": 0})


def test_a_part_ends_after_its_count():
    rd.apply({"part": "eplb", "count": 3})
    log = logging.getLogger("test.debug")
    for _ in range(5):
        if rd.on("eplb"):
            rd.emit("eplb", log, "line")
    assert not rd.on("eplb")
    assert rd.state()["last"]["eplb"]["off"] == "count"
    assert rd.state()["last"]["eplb"]["lines"] == 3


def test_a_part_ends_after_its_duration(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rd.time, "monotonic", lambda: now[0])
    rd.apply({"part": "kv", "duration_s": 10})
    assert rd.on("kv")
    now[0] += 11
    rd.sweep()
    assert not rd.armed()
    assert rd.state()["last"]["kv"]["off"] == "duration"


def test_a_loggers_debug_records_show_at_info_within_bounds():
    target = logging.getLogger("vllm.test_runtime_debug_target")
    seen: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = seen.append
    target.addHandler(handler)
    try:
        target.debug("hidden before")
        rd.apply({"part": "logger:vllm.test_runtime_debug_target", "count": 2})
        for i in range(4):
            target.debug("detail %d", i)
        assert [r.getMessage() for r in seen] == [
            "[debug logger:vllm.test_runtime_debug_target] detail 0",
            "[debug logger:vllm.test_runtime_debug_target] detail 1",
        ]
        assert all(r.levelno == logging.INFO for r in seen)
        assert target.level == logging.NOTSET  # put back
    finally:
        target.removeHandler(handler)


def test_the_placement_trace_switch(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_XPU_MOE_PLACEMENT_TRACE", raising=False)
    assert rd.check({"part": "placement"}).startswith("restart_required: ")
    prefix = tmp_path / "trace"
    monkeypatch.setenv("VLLM_XPU_MOE_PLACEMENT_TRACE", str(prefix))
    assert rd.check({"part": "placement"}) is None
    rd.apply({"part": "placement"})
    assert (tmp_path / "trace.on").exists()
    rd.apply({"part": "placement", "on": False})
    assert not (tmp_path / "trace.on").exists()


def test_debug_over_the_surface(monkeypatch):
    monkeypatch.delenv("VLLM_XPU_MOE_PLACEMENT_TRACE", raising=False)
    scheduler = create_scheduler()
    engine = _engine_with(scheduler)
    control = erc.RuntimeControl(engine)
    # worker parts reach every rank
    reply = control.apply("debug.set", {"part": "cpu", "count": 10})
    assert reply["applied"] == "all" and len(reply["ranks"]) == 2
    # (the stand-in ranks share this process, so the second finds it on)
    assert reply["ranks"][0]["changes"] == {"cpu": [False, True]}
    state = control.apply("debug.state", {})
    assert all("cpu" in r["on"] for r in state["ranks"])
    # engine-core parts stay here
    reply = control.apply("debug.set", {"part": "scheduler", "count": 3})
    assert "scheduler" in reply["debug"]["on"]
    for request in create_requests(num_requests=2, num_tokens=8):
        scheduler.add_request(request)
    scheduler.schedule()
    assert rd.state()["on"]["scheduler"]["lines"] == 2  # two admissions
    scheduler._preempt_request(scheduler.running.pop(), 0.0)
    assert "scheduler" not in rd.state()["on"]  # the third line was the last
    with pytest.raises(erc.RestartRequired, match="VLLM_XPU_MOE_PLACEMENT_TRACE"):
        control.apply("debug.set", {"part": "placement"})


# ---------------------------------------------------------------- schema


def test_the_schema_describes_verbs_and_parameters_with_values():
    scheduler = create_scheduler(max_num_seqs=8)
    control = erc.RuntimeControl(_engine_with(scheduler))
    schema = control.apply("schema", {})
    params = {p["name"]: p for p in schema["params"]}
    verbs = {v["name"]: v for v in schema["verbs"]}
    p = params["sched.max_num_active_seqs"]
    assert p["live"] is True and p["verb"] == "sched.set" and p["value"] == 8
    assert "max_num_seqs" in p["restart_reason"]
    assert params["sched.max_num_seqs"]["live"] is False
    assert params["tower.policy"]["choices"] == ["dynamic", "host", "card"]
    # the fast tier's verb is the fast tier's to register
    assert params["tiers.fast_rows"]["live"] is False
    assert "tiers.fast" in params["tiers.fast_rows"]["unavailable"]
    assert "sched.long_prefill_token_threshold" in verbs["sched.set"]["params"]
    assert {"debug.set", "profile.apply", "defaults.set"} <= set(verbs)
    assert "kv" in schema["profile_sections"]


def test_a_component_registers_its_own_verb_and_parameters():
    rp.register_param(
        rp.Param("tiers.example_rows", "int", "An example.", "tiers.example", min=0)
    )
    rp.register_verb(rp.Verb("tiers.example", "An example verb.", kind="set"))
    try:
        schema = rp.schema(None, type("C", (), {"state": lambda self: {}})())
        params = {p["name"]: p for p in schema["params"]}
        verbs = {v["name"]: v for v in schema["verbs"]}
        assert params["tiers.example_rows"]["live"] is True
        assert verbs["tiers.example"]["params"] == ["tiers.example_rows"]
    finally:
        rp.PARAMS.pop("tiers.example_rows")
        rp.VERBS.pop("tiers.example")


# ---------------------------------------------------------------- sched


def test_the_long_prefill_threshold_moves_live():
    scheduler = create_scheduler(max_num_batched_tokens=1024)
    control = erc.RuntimeControl(_engine_with(scheduler))
    reply = control.apply("sched.set", {"long_prefill_token_threshold": 16})
    assert reply["changes"] == {"long_prefill_token_threshold": [0, 16]}
    # (it caps a prefill only when another request is waiting or running)
    for request in create_requests(num_requests=2, num_tokens=64):
        scheduler.add_request(request)
    assert scheduler.schedule().total_num_scheduled_tokens == 32
    with pytest.raises(erc.ControlRefused, match="long_prefill"):
        control.apply("sched.set", {"long_prefill_token_threshold": -1})


# ---------------------------------------------------------------- fast tier port


class _FastControl:
    """The fast tier's own verb and state part, as its cell defines them."""

    def __init__(self) -> None:
        self.rows, self.state_name = 20, "idle"
        self.asked: list[dict] = []

    def state(self, only=None):
        return {
            "fast_tier": {
                "rows": self.rows,
                "range": [8, 32],
                "resizable": True,
                "record": {"state": self.state_name, "message": ""},
            }
        }

    def apply(self, op, args):
        assert op == "tiers.fast"
        self.asked.append(args)
        self.rows = args["rows"]
        self.state_name = "draining"
        return {"op": op, "applied": "all", "fast": {"state": "draining"}}


def test_the_fast_tier_port_speaks_the_fast_tiers_verb():
    control = _FastControl()
    port = SurfaceFastTierPort(control)
    assert port.read() == (FastTierView(20, 8, 32), None)
    assert port.resize(24, 60.0, False) == "wait"
    assert control.asked == [
        {"rows": 24, "from_kv": False, "deadline_s": 60.0, "preempt": False}
    ]
    assert port.settled() == "wait"
    control.state_name = "finished"
    assert port.settled() is None
    control.state_name = "failed"
    assert "ended failed" in port.settled()
    absent = type("C", (), {"state": lambda self, only=None: {}})()
    view, why = SurfaceFastTierPort(absent).read()
    assert view is None and "tiers.fast" in why


# ---------------------------------------------------------------- schema shape


def _walk_engine():
    engine = _Engine()
    engine.scheduler = create_scheduler(num_speculative_tokens=2)
    engine.vllm_config.cache_config.block_size = 16
    engine.vllm_config.cache_config.kv_cache_size_tokens = 100 * 16
    return engine


def test_zero_and_false_bounds_and_defaults_are_kept():
    schema = erc.RuntimeControl(_walk_engine()).apply("schema", {})
    params = {p["name"]: p for p in schema["params"]}
    assert params["kv.grow_floor_mib"]["min"] == 0
    assert params["spec.num_speculative_tokens"]["min"] == 0
    assert params["tower.host_available_floor_gib"]["default"] == 0.0
    assert params["tower.host_available_floor_gib"]["min"] == 0
    assert params["sched.max_num_active_seqs"]["nullable"] is False
    assert params["kv.release_chunk_mib"]["nullable"] is True


def _sample(param: dict):
    if param.get("choices"):
        return param["choices"][0]
    if param["type"] == "bool":
        return True
    low = param.get("min")
    if param["type"] == "int":
        return max(int(low) if low is not None else 1, 1)
    if param["type"] == "float":
        return float(low) + 1.0 if low is not None else 1.0
    return "x"


def test_every_parameter_listed_under_an_operation_is_accepted_by_it():
    """Posting each operation with each of its parameters, against stand-in
    workers: a refusal may say the engine cannot do it now (no tower, no
    EPLB, a range), never that the operation does not take that key."""
    from vllm.entrypoints.serve.dev.control import server_defaults

    engine = _walk_engine()
    control = erc.RuntimeControl(engine)
    schema = control.apply("schema", {})
    verbs = {v["name"]: v for v in schema["verbs"]}
    walked = 0
    for param in schema["params"]:
        if not param["live"]:
            continue
        assert param["name"] in verbs[param["verb"]]["params"], param["name"]
        if param["side"] == "server":
            assert param["key"] in [
                *server_defaults.SAMPLING_DEFAULTS,
                server_defaults.CJK_KEY,
            ]
            continue
        body = {param["key"]: _sample(param)}
        try:
            control.apply(param["verb"], body)
        except erc.ControlRefused as e:
            text = str(e)
            assert "unknown" not in text and "takes no arguments" not in text, (
                param["name"],
                text,
            )
        walked += 1
    assert walked >= 30


def test_kv_resize_takes_tokens_and_says_both():
    engine = _walk_engine()
    control = erc.RuntimeControl(engine)
    reply = control.apply("kv.resize", {"tokens": 800})
    assert reply["num_blocks"] == 50 and reply["tokens"] == 800
    assert engine.resizes[-1][0] == 50
    with pytest.raises(erc.ControlRefused, match="exactly one"):
        control.apply("kv.resize", {"tokens": 800, "num_blocks": 50})
    with pytest.raises(erc.ControlRefused, match="unknown"):
        control.apply("kv.resize", {"blocks": 50})


def test_profile_get_returns_the_body_as_defined():
    engine = _walk_engine()
    control = erc.RuntimeControl(engine)
    body = {"name": "p", "description": "d", "sched": {"max_num_active_seqs": 4}}
    control.apply("profile.define", body)
    reply = control.apply("profile.get", {"name": "p"})
    assert reply["profile"] == body and reply["active"] is False
    control.apply("profile.apply", {"name": "p"})
    assert control.apply("profile.get", {"name": "p"})["active"] is True
    with pytest.raises(erc.ControlRefused, match="no profile"):
        control.apply("profile.get", {"name": "q"})
    verbs = {v["name"] for v in control.apply("schema", {})["verbs"]}
    assert {
        "profile.define",
        "profile.list",
        "profile.apply",
        "profile.state",
        "profile.get",
    } <= verbs
