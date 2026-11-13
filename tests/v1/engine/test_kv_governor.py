# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV governor on simulated signal traces: what it moves, in which order,
when, and that it neither flaps nor changes anything in a dry run."""

import dataclasses
from types import SimpleNamespace

import pytest

from vllm.v1.engine import kv_governor as kg
from vllm.v1.engine.kv_governor import (
    FastTierView,
    GovernorSettings,
    KVGovernor,
    KVSignals,
    TowerView,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Towers:
    def __init__(self, idle_s: float | None = 600.0) -> None:
        self.idle_s = idle_s
        self.on_card = idle_s is not None
        self.demoted: list[str] = []
        self.holds: list[float] = []

    def read(self):
        if self.idle_s is None:
            return []
        return [TowerView("vt", self.on_card, True, self.idle_s)]

    def demote(self, name):
        self.demoted.append(name)
        self.on_card = False
        return None

    def hold(self, until):
        self.holds.append(until)


class Fast:
    def __init__(self, slots=20, lo=16, hi=24, blocks_per_slot=10.0) -> None:
        self.slots, self.lo, self.hi, self.bps = slots, lo, hi, blocks_per_slot
        self.resizes: list[int] = []
        self.refuse: str | None = None
        self.unavailable: str | None = None

    def read(self):
        if self.unavailable:
            return None, self.unavailable
        return FastTierView(self.slots, self.lo, self.hi, self.bps), None

    def resize(self, slots):
        if self.refuse:
            return self.refuse
        self.resizes.append(slots)
        self.slots = slots
        return None


class Signals:
    """A trace: `pattern(t)` gives the state at time t; the counters add up."""

    def __init__(self, clock: Clock, pattern) -> None:
        self.clock = clock
        self.pattern = pattern
        self.preempted = 0
        self.queries = 0
        self.hits = 0
        self.resize = "idle"

    def __call__(self) -> KVSignals:
        p = self.pattern(self.clock.t)
        self.preempted += p.get("preempt", 0)
        self.queries += p.get("queries", 0)
        self.hits += p.get("hits", 0)
        return KVSignals(
            num_blocks=1000,
            used=p["used"],
            cached_free=p.get("cached", 0.0),
            waiting=p.get("waiting", 0),
            running=p.get("running", 4),
            preemptions_total=self.preempted,
            prefix_queries_total=self.queries,
            prefix_hits_total=self.hits,
            resize_state=self.resize,
        )


HOT = {"used": 0.95, "waiting": 3}
COOL = {"used": 0.2, "cached": 0.1}


def _governor(pattern, mode="on", towers=None, fast=None, **settings):
    clock = Clock()
    calls: list[tuple] = []
    signals = Signals(clock, pattern)
    gov = KVGovernor(
        SimpleNamespace(),
        dataclasses.replace(GovernorSettings(mode=mode), **settings),
        clock=clock,
        signals=signals,
        towers=towers if towers is not None else Towers(),
        fast_tier=fast if fast is not None else Fast(),
        control=lambda op, args: calls.append((op, args)) or {"applied": "all"},
    )
    return gov, clock, signals, calls


def _run(gov, clock, until, start=None, step=1.0):
    t = clock.t if start is None else start
    while t <= until:
        clock.t = t
        gov.advance()
        t += step


def _moves(gov):
    return [(d.kind, d.action) for d in gov.decisions if d.kind in ("reclaim", "lend")]


def test_sustained_pressure_gives_back_the_cheapest_first():
    towers, fast = Towers(idle_s=600.0), Fast(slots=20, lo=16)
    gov, clock, _, calls = _governor(lambda t: HOT, towers=towers, fast=fast)
    _run(gov, clock, 200)
    # an idle tower first, then the fast tier two slots per layer at a time,
    # each a whole pressure window after the last (the windows were cleared)
    assert _moves(gov) == [
        ("reclaim", "tower vt card -> host"),
        ("reclaim", "fast tier 20 -> 18 slots per layer"),
        ("reclaim", "fast tier 18 -> 16 slots per layer"),
    ]
    times = [d.t for d in gov.decisions if d.kind == "reclaim"]
    assert times[0] == pytest.approx(29.0)
    assert all(b - a >= 29.0 for a, b in zip(times, times[1:]))
    assert towers.demoted == ["vt"] and fast.resizes == [18, 16]
    # after each resize: the KV cache takes the memory, EPLB re-sorts
    assert calls == [("kv.fit", {}), ("eplb.rearrange", {})] * 2
    # the towers are held off the card while the pressure lasts
    assert towers.holds and towers.holds[-1] > clock.t
    # at the floor: said once, nothing more moves
    waits = [d.action for d in gov.decisions if d.kind == "wait"]
    assert waits == [
        "nothing left to give back: the fast tier is at its floor (16 slots per layer)"
    ]


def test_a_tower_in_use_goes_after_the_fast_tier():
    towers, fast = Towers(idle_s=5.0), Fast(slots=18, lo=16)
    gov, clock, _, _ = _governor(lambda t: HOT, towers=towers, fast=fast)
    _run(gov, clock, 100)
    assert _moves(gov) == [
        ("reclaim", "fast tier 18 -> 16 slots per layer"),
        ("reclaim", "tower vt card -> host"),
    ]


def test_preemptions_are_pressure_even_below_the_usage_mark():
    pattern = lambda t: {"used": 0.7, "waiting": 0, "preempt": 1}  # noqa: E731
    gov, clock, _, _ = _governor(pattern, towers=Towers(idle_s=None))
    _run(gov, clock, 40)
    assert _moves(gov) == [("reclaim", "fast tier 20 -> 18 slots per layer")]


@pytest.mark.parametrize("period", [1, 5, 10, 20])
def test_load_that_comes_and_goes_moves_nothing(period):
    def pattern(t):
        return HOT if int(t // period) % 2 == 0 else COOL

    gov, clock, _, _ = _governor(pattern)
    _run(gov, clock, 1800)
    assert _moves(gov) == []
    assert gov.counts["samples"] == 1801


def test_low_demand_lends_back_after_the_dwell_and_a_whole_relax_window():
    towers, fast = Towers(idle_s=None), Fast(slots=20, lo=16, hi=24)

    def pattern(t):
        return HOT if t < 40 else COOL

    gov, clock, _, _ = _governor(pattern, towers=towers, fast=fast)
    _run(gov, clock, 1300)
    moves = [
        (d.kind, d.action, d.t) for d in gov.decisions if d.kind in ("reclaim", "lend")
    ]
    assert [m[:2] for m in moves] == [
        ("reclaim", "fast tier 20 -> 18 slots per layer"),
        ("lend", "fast tier 18 -> 20 slots per layer"),
        ("lend", "fast tier 20 -> 22 slots per layer"),
        ("lend", "fast tier 22 -> 24 slots per layer"),
    ]
    reclaim_t = moves[0][2]
    assert moves[1][2] - reclaim_t >= 299.0  # a whole relax window of cool
    assert all(b[2] - a[2] >= 299.0 for a, b in zip(moves[1:], moves[2:]))
    waits = [d.action for d in gov.decisions if d.kind == "wait"]
    assert "the fast tier is at its ceiling (24 slots per layer)" in waits


def test_lending_stops_where_the_pool_would_be_too_full():
    # 1000 blocks, 0.5 used: giving 2 slots x 150 blocks leaves 700 -> 0.71
    fast = Fast(slots=16, lo=16, hi=24, blocks_per_slot=150.0)
    gov, clock, _, _ = _governor(
        lambda t: {"used": 0.5}, towers=Towers(idle_s=None), fast=fast
    )
    _run(gov, clock, 400)
    assert _moves(gov) == []
    assert any("not lending" in d.action for d in gov.decisions)


def test_cached_trees_that_are_being_hit_count_as_demand():
    reused = {"used": 0.3, "cached": 0.5, "queries": 100, "hits": 60}
    stale = {"used": 0.3, "cached": 0.5, "queries": 100, "hits": 5}
    fast = Fast(slots=16, lo=16, hi=24, blocks_per_slot=1.0)
    gov, clock, _, _ = _governor(lambda t: reused, towers=Towers(None), fast=fast)
    _run(gov, clock, 400)
    assert _moves(gov) == []  # 0.3 + 0.5 > usage_low 0.6
    fast = Fast(slots=16, lo=16, hi=24, blocks_per_slot=1.0)
    gov, clock, _, _ = _governor(lambda t: stale, towers=Towers(None), fast=fast)
    _run(gov, clock, 400)
    assert _moves(gov) == [("lend", "fast tier 16 -> 18 slots per layer")]


def test_dry_run_decides_the_same_and_changes_nothing():
    def pattern(t):
        return HOT if t < 120 else COOL

    runs = {}
    for mode in ("on", "dry_run"):
        towers, fast = Towers(idle_s=600.0), Fast(slots=20, lo=16)
        gov, clock, _, calls = _governor(pattern, mode=mode, towers=towers, fast=fast)
        _run(gov, clock, 1200)
        runs[mode] = (gov, towers, fast, calls)
    on, dry = runs["on"], runs["dry_run"]
    assert _moves(dry[0]) == _moves(on[0]) and _moves(on[0])
    assert dry[1].demoted == [] and dry[1].holds == []
    assert dry[2].resizes == [] and dry[2].slots == 20
    assert dry[3] == []
    assert all(d.dry_run for d in dry[0].decisions)
    assert dry[0].state()["dry_run_position"]["towers_off"] == ["vt"]


def test_nothing_moves_while_a_kv_resize_is_in_progress():
    gov, clock, signals, _ = _governor(lambda t: HOT)
    signals.resize = "draining"
    _run(gov, clock, 100)
    assert _moves(gov) == []
    waits = [d.action for d in gov.decisions if d.kind == "wait"]
    assert waits == ["waiting: a KV cache resize is draining"]


def test_a_fast_tier_that_cannot_move_is_said_once():
    fast = Fast()
    fast.unavailable = "no tier setting 'fast_slots'"
    gov, clock, _, _ = _governor(lambda t: HOT, towers=Towers(None), fast=fast)
    _run(gov, clock, 300)
    waits = [d.action for d in gov.decisions if d.kind == "wait"]
    want = (
        "nothing left to give back: the fast tier cannot move (no tier setting "
        "'fast_slots')"
    )
    assert waits == [want]


def test_a_refused_move_is_reported_and_paced():
    fast = Fast()
    fast.refuse = "rank 1: an EPLB rearrangement is in flight"
    gov, clock, _, _ = _governor(lambda t: HOT, towers=Towers(None), fast=fast)
    _run(gov, clock, 60)
    refused = [d for d in gov.decisions if d.kind == "refused"]
    assert len(refused) == 3  # t=29, 44, 59: action_interval_s apart
    assert "in flight" in refused[0].reason
    assert gov.counts["refused"] == 3 and fast.resizes == []


def test_every_decision_carries_its_inputs(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(kg.logger, "info", lambda msg, *args: lines.append(msg % args))
    gov, clock, _, _ = _governor(lambda t: HOT, towers=Towers(None))
    _run(gov, clock, 30)
    reclaim = [d for d in gov.decisions if d.kind == "reclaim"][0]
    for key in ("used", "cached_free", "hit_rate", "waiting", "hot", "blocks"):
        assert key in reclaim.inputs
    assert any(
        line.startswith("KV governor: reclaim fast tier 20 -> 18")
        and "used 0.95" in line
        and "hot 1.0 of 30 samples in 30 s" in line
        for line in lines
    ), lines
    gov, clock, _, _ = _governor(lambda t: HOT, mode="dry_run", towers=Towers(None))
    _run(gov, clock, 30)
    assert any(
        line.startswith("KV governor (dry run, would): reclaim fast tier 20 -> 18")
        for line in lines
    )


def test_settings_checks_and_the_start_switch():
    base = GovernorSettings()
    assert kg.check_settings(base, {"usage_low": 0.95}) is not None  # hysteresis
    assert kg.check_settings(base, {"mode": "maybe"}) is not None
    assert kg.check_settings(base, {"fast_step_slots": 0}) is not None
    assert kg.check_settings(base, {"nope": 1}).startswith("unknown")
    assert kg.check_settings(base, {"mode": "on", "usage_high": 0.8}) is None
    assert kg.settings_from_env("") is None
    assert kg.settings_from_env("off") is None
    assert kg.settings_from_env("on").mode == "on"
    s = kg.settings_from_env('{"pressure_window_s": 10}')
    assert s.mode == "dry_run" and s.pressure_window_s == 10
    with pytest.raises(ValueError):
        kg.settings_from_env('{"usage_high": 2}')


def test_unset_means_no_governor(monkeypatch):
    monkeypatch.delenv("VLLM_KV_GOVERNOR", raising=False)
    assert KVGovernor.maybe_create(SimpleNamespace()) is None


def test_switching_the_mode_resets_the_evidence():
    gov, clock, _, _ = _governor(lambda t: HOT, mode="dry_run")
    _run(gov, clock, 20)
    assert len(gov.samples) == 21
    assert gov.set({"mode": "on"}) == {"mode": ["dry_run", "on"]}
    assert len(gov.samples) == 0 and gov.level == "neutral"
    gov.set({"mode": "off"})
    _run(gov, clock, 200, start=21)
    assert gov.counts["samples"] == 21 and gov.seconds_to_next_event() is None


# ---------------------------------------------------------------- the ports


def test_the_engine_tower_port_holds_and_releases_only_its_own_hold():
    from vllm.v1.engine.tower_residency import TowerPhase, TowerPlace

    tower = SimpleNamespace(
        place=TowerPlace.CARD,
        phase=TowerPhase.STEADY,
        last_use=10.0,
        hold_until=0.0,
        _clock=lambda: 100.0,
    )
    engine = SimpleNamespace(
        _tower_residency=SimpleNamespace(towers={"vt": tower}),
        runtime_control=lambda op, args: {"applied": "all"},
    )
    port = kg.EngineTowerPort(engine)
    assert port.read() == [TowerView("vt", True, True, 90.0)]
    port.hold(400.0)
    assert tower.hold_until == 400.0
    port.hold(0)
    assert tower.hold_until == 0.0
    tower.hold_until = 500.0  # a failed promotion's own hold
    port.hold(0)
    assert tower.hold_until == 500.0
    assert port.demote("vt") is None


def test_the_surface_fast_tier_port_reads_the_ranks_and_calls_tiers_set():
    states = [
        {
            "tiers": {
                "settings": {"fast_slots": 20},
                "setting_bounds": {"fast_slots": [8, 30]},
                "expert_bytes": 2 << 20,
                "layers": 40,
            },
            "kv": {"bytes_per_block": 8 << 20},
        },
        {
            "tiers": {
                "settings": {"fast_slots": 20},
                "setting_bounds": {"fast_slots": [10, 28]},
                "expert_bytes": 2 << 20,
                "layers": 40,
            },
            "kv": {"bytes_per_block": 8 << 20},
        },
    ]
    asked: list[tuple] = []
    control = SimpleNamespace(_rpc=lambda phase, op, args: states)
    engine = SimpleNamespace(
        _control=lambda: control,
        runtime_control=lambda op, args: asked.append((op, args)) or {"applied": "all"},
    )
    port = kg.SurfaceFastTierPort(engine, lambda: "fast_slots")
    view, why = port.read()
    assert why is None
    assert view == FastTierView(20, 10, 28, 10.0)
    assert port.resize(18) is None
    assert asked == [("tiers.set", {"fast_slots": 18})]
    states[0]["tiers"]["settings"] = {"host_tier_bound": 22}
    states[0]["tiers"]["resize_blocked"] = "the tier sizes are fixed at load"
    view, why = port.read()
    assert view is None and "no tier setting 'fast_slots'" in why


# ---------------------------------------------------------------- real signals


def test_the_signals_of_a_real_scheduler():
    from ..core.utils import create_requests, create_scheduler

    scheduler = create_scheduler(enable_prefix_caching=True, num_blocks=64)
    engine = SimpleNamespace(
        scheduler=scheduler, get_kv_cache_resize_state=lambda: {"state": "idle"}
    )
    first = kg.read_engine_signals(engine)
    assert first.used == 0.0 and first.cached_free == 0.0 and first.waiting == 0
    requests = create_requests(num_requests=2, num_tokens=64)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    busy = kg.read_engine_signals(engine)
    assert busy.running == 2 and busy.used > 0
    assert busy.prefix_queries_total == 128
    victim = scheduler.running.pop()
    scheduler._preempt_request(victim, 0.0)
    after = kg.read_engine_signals(engine)
    assert after.preemptions_total == 1 and after.waiting == 1
    # the preempted request's full blocks stay cached with no owner
    assert after.cached_free > 0
