# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's residency controller for encoder towers with residency
"dynamic", against the engine core's real KV cache resize state machine, a
clock the test moves, and stand-ins for the ranks' accelerator copies."""

import logging
from types import SimpleNamespace

import pytest

from tests.v1.engine.test_kv_cache_shrink import MIB, _core
from vllm.config.multimodal import HostTowerOptions
from vllm.v1.engine.tower_residency import (
    TowerPhase,
    TowerPlace,
    TowerResidencyController,
)

TOWER = "vision_tower"
TOWER_BYTES = 6 * MIB


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeHostTower:
    def __init__(self) -> None:
        self.location = SimpleNamespace(value="host")
        self.nbytes = 7 * MIB
        self.releases = 0

    def release(self, wait: bool = True):
        self.releases += 1
        self.location = SimpleNamespace(value="disk")


class FakeDelegate:
    """The scheduler delegate's residency surface."""

    def __init__(self) -> None:
        self.host = FakeHostTower()
        self.on_card = False
        self.flips: list[bool] = []
        self.stats: dict[str, float] = {}
        self.listener = None

    def set_arrival_listener(self, listener) -> None:
        self.listener = listener

    def set_tower_on_card(self, name: str, on_card: bool) -> None:
        self.on_card = on_card
        self.flips.append(on_card)

    def host_tower(self, name: str) -> FakeHostTower:
        return self.host

    def update_tower_stats(self, name: str, **values: float) -> None:
        self.stats.update(values)


class FakeRanks:
    """Each rank's accelerator copy, driven by `collective_rpc`. A failure
    set for a rank raises on that rank only; the others carry out the call,
    as worker processes would."""

    def __init__(self, executor, num_ranks: int = 2) -> None:
        self.executor = executor
        self.state = ["none"] * num_ranks
        self.polls_until_staged = 1
        self.polls = 0
        self.read_error: str | None = None
        self.fail: dict[str, int] = {}  # method -> rank that raises

    def __call__(self, method, timeout=None, args=(), kwargs=None):
        self.executor.calls.append(method.removeprefix("tower_residency_"))
        results = []
        error: Exception | None = None
        for rank in range(len(self.state)):
            try:
                results.append(self._one(method, rank))
            except Exception as e:
                error = error or e
        if error is not None:
            raise error
        return results

    def _one(self, method: str, rank: int):
        if self.fail.get(method) == rank:
            if method == "tower_residency_commit":
                self.state[rank] = "none"
            raise RuntimeError(f"rank {rank} failed in {method}")
        if method == "tower_residency_stage":
            if self.state[rank] != "card":
                self.state[rank] = "staged"
            return {"bytes": TOWER_BYTES, "on_card": self.state[rank] == "card"}
        if method == "tower_residency_poll":
            if rank == 0:
                self.polls += 1
            return {
                "staged": self.polls >= self.polls_until_staged
                and self.read_error is None,
                "error": self.read_error,
            }
        if method == "tower_residency_commit":
            assert self.state[rank] == "staged"
            self.state[rank] = "card"
            return {"bytes": TOWER_BYTES, "seconds": 0.1}
        if method == "tower_residency_release":
            was = self.state[rank] == "card"
            self.state[rank] = "none"
            return {"bytes": TOWER_BYTES if was else 0}
        if method == "tower_residency_discard":
            if self.state[rank] != "card":
                self.state[rank] = "none"
            return None
        raise AssertionError(method)


def _setup(num_blocks=100, **options):
    core, executor, scheduler = _core(num_blocks)
    ranks = FakeRanks(executor)
    executor.collective_rpc = ranks
    delegate = FakeDelegate()
    clock = Clock()
    opts = HostTowerOptions(residency="dynamic", card_reserve_mib=2, **options)
    controller = TowerResidencyController(core, delegate, {TOWER: opts}, clock=clock)
    return SimpleNamespace(
        core=core,
        executor=executor,
        scheduler=scheduler,
        ranks=ranks,
        delegate=delegate,
        clock=clock,
        controller=controller,
        tower=controller.towers[TOWER],
    )


def _image(env, at: float | None = None) -> None:
    if at is not None:
        env.clock.now = at
    env.delegate.listener(TOWER)


def _step(env, at: float | None = None) -> None:
    if at is not None:
        env.clock.now = at
    env.controller.advance()


# ----------------------------------------------------------------------
# Triggers
# ----------------------------------------------------------------------


def test_one_image_request_does_not_promote():
    env = _setup()
    _image(env, 1000)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.HOST
    assert env.executor.calls == []


def test_the_second_request_inside_the_window_promotes():
    env = _setup()
    _image(env, 1000)
    _image(env, 1599)
    _step(env, 1599.5)
    assert env.tower.place == TowerPlace.CARD
    assert env.delegate.on_card


def test_two_requests_further_apart_than_the_window_do_not():
    env = _setup()
    _image(env, 1000)
    _step(env, 1000)
    _image(env, 1601)
    _step(env, 1601)
    assert env.tower.place == TowerPlace.HOST
    assert len(env.tower.arrivals) == 1


def test_promote_after_counts_requests():
    env = _setup(promote_after=3)
    for t in (1000, 1010):
        _image(env, t)
        _step(env, t)
    assert env.tower.place == TowerPlace.HOST
    _image(env, 1020)
    _step(env, 1020)
    assert env.tower.place == TowerPlace.CARD


def test_demotion_after_the_idle_time_and_not_before():
    env = _setup()
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.CARD
    _step(env, 1001 + 1799)
    assert env.tower.place == TowerPlace.CARD
    _step(env, 1001 + 1800)
    assert env.tower.place == TowerPlace.HOST
    assert env.delegate.flips == [True, False]


def test_a_request_on_the_card_restarts_the_idle_time():
    env = _setup()
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    _image(env, 2500)
    _step(env, 1001 + 1800)
    assert env.tower.place == TowerPlace.CARD
    _step(env, 2500 + 1800)
    assert env.tower.place == TowerPlace.HOST


def test_seconds_to_the_next_event():
    env = _setup()
    assert env.controller.seconds_to_next_event() is None
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.controller.seconds_to_next_event() == pytest.approx(1800.01)
    env.clock.now = 1501
    assert env.controller.seconds_to_next_event() == pytest.approx(1300.01)


# ----------------------------------------------------------------------
# The moves and the memory exchange
# ----------------------------------------------------------------------


def test_a_promotion_stages_then_shrinks_then_commits_then_routes(caplog):
    env = _setup()
    caplog.set_level(logging.INFO, logger="vllm.v1.engine.tower_residency")
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)

    # need = 6 MiB + 2 MiB reserve: the fake's rank 1 needs 10 blocks for it.
    assert env.executor.calls == [
        "stage",
        "poll",
        "plan:90",
        "shrink:90",
        "commit",
    ]
    assert env.scheduler.events == ["fence:90", "finish:90"]
    assert env.core.vllm_config.cache_config.num_gpu_blocks == 90
    assert env.delegate.flips == [True]
    assert env.ranks.state == ["card", "card"]
    assert env.delegate.stats["location"] == 2.0
    assert env.delegate.stats["promotions"] == 1.0
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        "vision_tower: host -> card in" in line
        and "2 requests for its modalities in 600 s" in line
        and f"{TOWER_BYTES:,} bytes per rank" in line
        and "KV cache 100 -> 90 blocks" in line
        for line in lines
    ), lines


def test_a_demotion_routes_back_first_then_releases_then_grows(caplog):
    env = _setup()
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    env.executor.calls.clear()
    caplog.set_level(logging.INFO, logger="vllm.v1.engine.tower_residency")

    _step(env, 1001 + 1800)

    assert env.delegate.flips == [True, False]
    assert env.executor.calls == ["release", "grow:100"]
    assert env.ranks.state == ["none", "none"]
    assert env.core.vllm_config.cache_config.num_gpu_blocks == 100
    assert env.tower.place == TowerPlace.HOST
    assert env.delegate.stats["demotions"] == 1.0
    assert any(
        "vision_tower: card -> host in" in r.getMessage()
        and "no request for its modalities in 1800 s" in r.getMessage()
        and "KV cache 90 -> 100 blocks" in r.getMessage()
        for r in caplog.records
    )


def test_staging_waits_for_every_rank_and_the_host_serves_meanwhile():
    env = _setup()
    env.ranks.polls_until_staged = 3
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.phase == TowerPhase.STAGING
    assert env.delegate.flips == []
    # Polls are spaced.
    _step(env, 1001.1)
    assert env.executor.calls.count("poll") == 1
    _step(env, 1001.6)
    assert env.tower.phase == TowerPhase.STAGING
    # A request arriving mid-transition is counted and changes nothing.
    _image(env, 1001.7)
    _step(env, 1002.2)
    assert env.tower.place == TowerPlace.CARD
    assert env.delegate.flips == [True]
    assert env.tower.last_use == 1001.7


def test_a_drain_in_progress_keeps_the_host_path_until_it_finishes():
    env = _setup()
    env.scheduler.held = 3
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.phase == TowerPhase.SHRINKING
    assert env.delegate.flips == []
    _step(env, 1002)
    assert env.tower.phase == TowerPhase.SHRINKING
    env.scheduler.held = 0
    _step(env, 1003)
    assert env.tower.place == TowerPlace.CARD
    assert env.delegate.flips == [True]


def test_a_refused_shrink_leaves_the_tower_on_the_host_and_holds_off():
    env = _setup()
    env.scheduler.floor = 95  # the shrink to 90 is below the floor
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)

    assert env.tower.place == TowerPlace.HOST
    assert env.tower.phase == TowerPhase.STEADY
    assert env.delegate.flips == []
    assert env.ranks.state == ["none", "none"]
    assert env.executor.calls[-1] == "discard"
    assert env.delegate.stats["failed_promotions"] == 1.0
    assert env.core.vllm_config.cache_config.num_gpu_blocks == 100

    env.executor.calls.clear()
    _image(env, 1030)
    _step(env, 1030)
    assert env.executor.calls == []  # held off
    env.scheduler.floor = 10
    _step(env, 1061.5)
    assert env.tower.place == TowerPlace.CARD


def test_a_drain_that_runs_out_of_time_gives_the_blocks_back():
    env = _setup(shrink_deadline_s=5)
    env.scheduler.held = 3
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.phase == TowerPhase.SHRINKING
    # The resize's deadline runs on the real clock; make it pass.
    env.core._kv_resize.started -= 10
    env.core._advance_kv_resize()
    _step(env, 1002)
    assert env.tower.place == TowerPlace.HOST
    assert env.tower.phase == TowerPhase.STEADY
    assert env.scheduler.events[-1] == "abort"
    assert env.core.vllm_config.cache_config.num_gpu_blocks == 100
    assert env.ranks.state == ["none", "none"]
    assert env.delegate.flips == []


def test_a_failed_copy_on_one_rank_releases_every_rank_and_grows_back():
    env = _setup()
    env.ranks.fail["tower_residency_commit"] = 1
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)

    assert env.delegate.flips == []
    assert env.tower.place == TowerPlace.HOST
    assert env.ranks.state == ["none", "none"]
    assert env.executor.calls[-4:] == ["commit", "release", "discard", "grow:100"]
    assert env.core.vllm_config.cache_config.num_gpu_blocks == 100


def test_a_failed_read_stops_before_the_kv_cache_is_touched():
    env = _setup()
    env.ranks.read_error = "OSError(5)"
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.HOST
    assert env.executor.calls == ["stage", "poll", "discard"]
    assert env.scheduler.events == []


def test_a_failed_stage_on_one_rank_discards_on_every_rank():
    env = _setup()
    env.ranks.fail["tower_residency_stage"] = 0
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.HOST
    assert env.executor.calls == ["stage", "discard"]
    assert env.ranks.state == ["none", "none"]


def test_no_promotion_starts_while_another_resize_runs():
    env = _setup()
    env.scheduler.held = 2
    env.core.resize_kv_cache(80)  # an operator's shrink, draining
    _image(env, 1000)
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.phase == TowerPhase.STEADY
    assert "stage" not in env.executor.calls
    env.scheduler.held = 0
    env.core._advance_kv_resize()
    _step(env, 1002)
    assert env.tower.place == TowerPlace.CARD


def test_idle_location_disk_releases_the_host_copy_at_start_and_after_the_card():
    env = _setup(idle_location="disk")
    assert env.delegate.host.releases == 1
    assert env.tower.place == TowerPlace.DISK

    # The first image loads the host copy by itself.
    env.delegate.host.location = SimpleNamespace(value="host")
    _image(env, 1000)
    _step(env, 1000)
    assert env.tower.place == TowerPlace.HOST
    _image(env, 1001)
    _step(env, 1001)
    assert env.tower.place == TowerPlace.CARD
    _step(env, 1001 + 1800)
    assert env.tower.place == TowerPlace.DISK
    assert env.delegate.host.releases == 2


def test_the_host_copy_goes_to_disk_after_the_idle_time():
    env = _setup(idle_location="disk", promote_after=5)
    env.delegate.host.location = SimpleNamespace(value="host")
    _image(env, 1000)
    _step(env, 1000)
    assert env.tower.place == TowerPlace.HOST
    _step(env, 2799)
    assert env.tower.place == TowerPlace.HOST
    _step(env, 2800)
    assert env.tower.place == TowerPlace.DISK
    assert env.delegate.host.releases == 2


def test_a_failing_step_does_not_stop_the_engine(monkeypatch):
    env = _setup()

    def boom(now):
        raise RuntimeError("boom")

    monkeypatch.setattr(env.tower, "advance", boom)
    env.controller.advance()  # logged, not raised


# ----------------------------------------------------------------------
# Off by default
# ----------------------------------------------------------------------


def _engine_with(towers):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_host_towers=towers)
            )
        ),
        scheduler=None,  # never consulted without a dynamic tower
    )


@pytest.mark.parametrize(
    "towers",
    [
        {},
        {TOWER: HostTowerOptions(threads=16)},
        {TOWER: HostTowerOptions(residency="host")},
    ],
)
def test_no_controller_without_a_dynamic_tower(towers):
    assert TowerResidencyController.maybe_create(_engine_with(towers)) is None


def test_no_multimodal_config_no_controller():
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace()), scheduler=None
    )
    assert TowerResidencyController.maybe_create(engine) is None


def test_the_engine_core_has_no_controller_by_default():
    from vllm.v1.engine.core import EngineCore

    assert EngineCore._tower_residency is None
