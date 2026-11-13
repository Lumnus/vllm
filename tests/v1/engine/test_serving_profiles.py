# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving profiles against the engine core's real KV cache resize state
machine and tower residency controller: checked before anything changes,
memory freed before it is taken, a drain waited for between steps, a failure
reported with what stands where; and the tower's placement policy."""

import json
import pathlib

import pytest

import vllm.v1.engine.runtime_control as erc
from vllm.v1.engine import serving_profiles as sp
from vllm.v1.engine.serving_profiles import FastTierView, ServingProfiles
from vllm.v1.engine.tower_residency import TowerPlace

from .test_kv_cache_shrink import _core
from .test_tower_residency import _image, _setup, _step

EXAMPLES = pathlib.Path(__file__).parents[3] / "examples/features/serving_profiles"


class FakeFast:
    """The fast tier's verb (`tiers.set fast_rows`), recorded in the
    executor's call list so that its order against the KV calls shows."""

    def __init__(self, calls: list, slots=20, lo=8, hi=32) -> None:
        self.calls, self.slots, self.lo, self.hi = calls, slots, lo, hi
        self.refuse: str | None = None
        self.unavailable: str | None = None

    def read(self):
        if self.unavailable:
            return None, self.unavailable
        return FastTierView(self.slots, self.lo, self.hi), None

    def resize(self, rows, deadline_s, preempt):
        if self.refuse:
            raise erc.ControlRefused(self.refuse)
        self.calls.append(f"fast:{rows}")
        self.slots = rows
        return "wait" if self.pending else None

    pending = False

    def settled(self):
        return "wait" if self.pending else None


def _profiles(core, executor, **fast):
    port = FakeFast(executor.calls, **fast)
    profiles = ServingProfiles(core, control=erc.RuntimeControl(core), fast_tier=port)
    core._serving_profiles = profiles
    return profiles, port


def _define(profiles, **profile):
    return profiles.operation("profile.define", profile)


# ---------------------------------------------------------------- shape


def test_profile_shapes():
    assert sp.check_profile({"name": "a", "kv": {"tokens": 1000}}) is None
    assert "exactly one" in sp.check_profile({"name": "a", "kv": {}})
    assert "unknown profile sections" in sp.check_profile({"name": "a", "x": {}})
    assert "fast_rows" in sp.check_profile({"name": "a", "tiers": {"fast_rows": "all"}})
    assert "name" in sp.check_profile({"kv": {"fit": True}})


def test_the_shipped_examples_are_profiles():
    names = []
    for path in sorted(EXAMPLES.glob("*.json")):
        profile = json.loads(path.read_text())
        assert sp.check_profile(profile) is None, path
        names.append(profile["name"])
    assert names == ["agentic-coding", "long-sessions"]


# ---------------------------------------------------------------- order


def test_memory_is_freed_before_it_is_taken():
    core, executor, _ = _core(100)
    profiles, port = _profiles(core, executor, slots=20)
    _define(profiles, name="experts", kv={"blocks": 60}, tiers={"fast_rows": 24})
    _define(profiles, name="kv", kv={"blocks": 100}, tiers={"fast_rows": 20})
    reply = profiles.operation("profile.apply", {"name": "experts"})
    assert reply["applied"] == "all"
    # the KV cache shrinks first, then the fast tier grows into the room
    assert executor.calls == ["plan:60", "shrink:60", "fast:24"]
    executor.calls.clear()
    reply = profiles.operation("profile.apply", {"name": "kv"})
    assert reply["applied"] == "all"
    # the fast tier shrinks first, then the KV cache grows
    assert executor.calls == ["fast:20", "grow:100"]
    state = profiles.state()
    assert state["active"] == "kv" and state["previous"] == "experts"
    assert state["converged"] is True
    assert all(t["match"] for t in state["targets"].values())


def test_a_draining_shrink_is_waited_for_between_steps():
    core, executor, scheduler = _core(100)
    profiles, _ = _profiles(core, executor, slots=20)
    _define(profiles, name="experts", kv={"blocks": 60}, tiers={"fast_rows": "max"})
    scheduler.held = 5  # a request still holds fenced blocks
    reply = profiles.operation("profile.apply", {"name": "experts"})
    assert reply["applied"] == "applying"
    assert executor.calls == ["plan:60"]  # the fast tier has not moved
    state = profiles.state()
    assert state["applying"] == "experts" and state["converged"] is False
    assert state["targets"]["kv.blocks"]["resize"] == "draining"
    assert profiles.seconds_to_next_event() == ServingProfiles.POLL_S
    profiles.advance()
    assert "fast:32" not in executor.calls
    scheduler.held = 0
    core._advance_kv_resize()  # the busy loop's own step
    profiles.advance()
    assert executor.calls == ["plan:60", "shrink:60", "fast:32"]
    assert profiles.active == "experts" and profiles.current is None


def test_a_fast_tier_grow_that_runs_on_is_waited_for():
    core, executor, _ = _core(100)
    profiles, port = _profiles(core, executor, slots=20)
    _define(profiles, name="experts", kv={"blocks": 60}, tiers={"fast_rows": 24})
    port.pending = True
    assert profiles.operation("profile.apply", {"name": "experts"})["applied"] == (
        "applying"
    )
    steps = {s.label: s.status for s in profiles.current.steps}
    assert steps["fast tier up"] == "waiting"
    port.pending = False
    profiles.advance()
    assert profiles.active == "experts"


def test_a_refused_check_changes_nothing():
    core, executor, _ = _core(100)
    profiles, port = _profiles(core, executor)
    port.unavailable = "no tier setting 'fast_rows'"
    _define(profiles, name="experts", kv={"blocks": 60}, tiers={"fast_rows": 24})
    with pytest.raises(erc.ControlRefused, match="tiers.fast_rows: no tier"):
        profiles.operation("profile.apply", {"name": "experts"})
    _define(profiles, name="tiny", kv={"blocks": 5})
    with pytest.raises(erc.ControlRefused, match="below the 10"):
        profiles.operation("profile.apply", {"name": "tiny"})
    assert executor.calls == [] and profiles.active is None


def test_a_failure_midway_says_what_stands_where():
    core, executor, _ = _core(100)
    profiles, port = _profiles(core, executor, slots=20)
    _define(profiles, name="base", kv={"blocks": 100})
    profiles.operation("profile.apply", {"name": "base"})
    _define(profiles, name="experts", kv={"blocks": 60}, tiers={"fast_rows": 24})
    port.refuse = "rank 1: an EPLB rearrangement is in flight"
    reply = profiles.operation("profile.apply", {"name": "experts"})
    assert reply["applied"] == "partial"
    steps = {s["step"]: s["status"] for s in reply["apply"]["steps"]}
    assert steps["KV cache down"] == "done" and steps["fast tier up"] == "failed"
    state = profiles.state()
    assert state["active"] is None and state["previous"] == "base"
    assert state["targets"]["kv.blocks"]["match"] is True
    assert state["targets"]["tiers.fast_rows"]["match"] is False
    # going back is one apply away
    port.refuse = None
    assert profiles.operation("profile.apply", {"name": "base"})["applied"] == "all"


def test_one_apply_at_a_time_and_unknown_names():
    core, executor, scheduler = _core(100)
    profiles, _ = _profiles(core, executor)
    _define(profiles, name="a", kv={"blocks": 60})
    scheduler.held = 5
    profiles.operation("profile.apply", {"name": "a"})
    with pytest.raises(erc.ControlRefused, match="one at a time"):
        profiles.operation("profile.apply", {"name": "a"})
    with pytest.raises(erc.ControlRefused, match="no profile 'b'"):
        profiles.operation("profile.apply", {"name": "b"})
    listed = profiles.operation("profile.list", {})["profiles"]
    assert listed == [{"name": "a", "description": "", "sections": ["kv"]}]


def test_kv_tokens_become_blocks():
    core, executor, _ = _core(100)
    core.vllm_config.cache_config.kv_cache_size_tokens = 100 * 16
    profiles, _ = _profiles(core, executor)
    assert profiles._kv_target({"tokens": 800}) == 50


def test_the_boot_profile_from_a_file(tmp_path, monkeypatch):
    core, executor, _ = _core(100)
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {"profiles": [{"name": "small", "kv": {"blocks": 80}}], "boot": "small"}
        )
    )
    monkeypatch.setenv("VLLM_SERVING_PROFILES", str(path))
    profiles = ServingProfiles.maybe_create(core)
    assert profiles.pending_boot == "small" and executor.calls == []
    profiles._control_obj = erc.RuntimeControl(core)
    profiles.advance()  # the busy loop's first step
    assert executor.calls == ["plan:80", "shrink:80"] and profiles.active == "small"
    monkeypatch.delenv("VLLM_SERVING_PROFILES")
    assert ServingProfiles.maybe_create(core) is None


def test_the_surface_dispatches_profile_operations():
    core, executor, _ = _core(100)
    control = erc.RuntimeControl(core)
    reply = control.apply("profile.define", {"name": "a", "kv": {"blocks": 90}})
    assert reply["profile"] == "a" and core._serving_profiles is not None
    core._serving_profiles._fast = FakeFast(executor.calls)
    assert control.apply("profile.apply", {"name": "a"})["applied"] == "all"
    assert control.apply("profile.state", {})["active"] == "a"


# ---------------------------------------------------------------- the tower


def _with_profiles(env):
    env.core._tower_residency = env.controller
    profiles = ServingProfiles(
        env.core,
        control=erc.RuntimeControl(env.core),
        fast_tier=FakeFast(env.executor.calls),
    )
    return profiles


def test_a_profile_takes_the_tower_off_the_card_before_memory_moves():
    env = _setup()
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.CARD
    profiles = _with_profiles(env)
    _define(profiles, name="text", tower={"policy": "host", "disk_idle_s": 900})
    assert (
        profiles.operation("profile.apply", {"name": "text"})["applied"] == "applying"
    )
    _step(env, 1002)  # the tower's own controller acts on its policy
    profiles.advance()
    assert env.tower.place == TowerPlace.HOST and profiles.active == "text"
    # images no longer bring it back
    _image(env, 1003)
    _image(env, 1004)
    _step(env, 1004)
    assert env.tower.place == TowerPlace.HOST
    # and after 900 s idle it leaves host memory too (the drive tier)
    _step(env, 1004 + 899)
    assert env.tower.place == TowerPlace.HOST
    _step(env, 1004 + 900)
    assert env.tower.place == TowerPlace.DISK


def test_a_profile_with_the_tower_on_the_card_puts_it_there_last():
    env = _setup()
    profiles = _with_profiles(env)
    _define(profiles, name="vision", kv={"blocks": 90}, tower={"policy": "card"})
    reply = profiles.operation("profile.apply", {"name": "vision"})
    assert reply["applied"] == "applying"
    # the KV step came first; the tower then takes its bytes from the KV cache
    assert env.executor.calls[:2] == ["plan:90", "shrink:90"]
    for t in range(1, 10):
        _step(env, 1000 + t)
        env.core._advance_kv_resize()
        profiles.advance()
    assert env.tower.place == TowerPlace.CARD and profiles.active == "vision"
    # no idle time demotes a tower whose policy is card
    _step(env, 1000 + 10 + 3600)
    assert env.tower.place == TowerPlace.CARD


# ---------------------------------------------------------------- tower policy


def test_tower_policy_over_the_surface():
    env = _setup(promote_after=3, promote_window_s=300.0)
    env.core._tower_residency = env.controller
    control = erc.RuntimeControl(env.core)
    reply = control.apply("tower.set", {"policy": "host", "disk_idle_s": 600})
    assert reply["changes"] == {
        "policy": ["dynamic", "host"],
        "disk_idle_s": [None, 600],
    }
    with pytest.raises(erc.ControlRefused, match="policy host"):
        control.apply("tower.promote", {})
    with pytest.raises(erc.ControlRefused, match="policy must be"):
        control.apply("tower.set", {"policy": "gpu"})
    with pytest.raises(erc.ControlRefused, match="disk_idle_s"):
        control.apply("tower.set", {"disk_idle_s": 0})
    assert env.tower.policy == "host" and env.tower.disk_idle_s == 600
    state = control._towers_state()["vision_tower"]
    assert state["policy"] == "host" and state["disk_idle_s"] == 600
    control.apply("tower.set", {"policy": "dynamic"})
    # 3 requests in 5 minutes earn the card
    for t in (1000, 1100):
        _image(env, t)
        _step(env, t)
    assert env.tower.place == TowerPlace.HOST
    _image(env, 1250)
    _step(env, 1250)
    assert env.tower.place == TowerPlace.CARD


def test_without_disk_idle_the_old_idle_location_rule_holds():
    env = _setup(idle_location="disk", demote_idle_s=100.0)
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    _step(env, 1101)
    assert env.tower.place == TowerPlace.DISK
