# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host tower delegate's side of residency "dynamic": which copy an item
goes to before, during and after a move, what is counted, and that nothing
of it runs without a dynamic tower."""

import threading

import pytest

from tests.v1.ec_connector.unit.host_tower.test_host_tower_connector import (
    _Env,
    _request,
)
from tests.v1.ec_connector.unit.host_tower.utils import make_tower

TOWER = "vision_tower"


@pytest.fixture
def make_env(monkeypatch):
    envs: list[_Env] = []

    def make(**kwargs) -> _Env:
        env = _Env(monkeypatch, **kwargs)
        envs.append(env)
        return env

    yield make
    for env in envs:
        module = env.tower._module
        if module is not None and module.gate is not None:
            module.gate.set()
        env.sched.shutdown()


def _listen(env) -> list[str]:
    heard: list[str] = []
    env.sched.set_arrival_listener(heard.append)
    return heard


def test_without_a_dynamic_tower_nothing_is_heard_or_counted(make_env):
    env = make_env()
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is False
    env.wait("a")
    assert env.admit(request) is True
    assert env.sched._seen_requests == set()
    assert env.sched.host_tower_stats() == {}
    assert env.sched._on_card == set()


def test_each_request_is_heard_once_per_tower(make_env):
    env = make_env()
    heard = _listen(env)
    request = _request("r1", ("a", 0, 4), ("b", 10, 4))
    env.admit(request)
    env.admit(request)
    env.admit(_request("r2", ("c", 0, 4)))
    # A request without the tower's modalities is not.
    env.admit(_request("r3", ("x", 0, 4), modality="audio"))
    assert heard == [TOWER, TOWER]
    assert env.sched.host_tower_stats()[TOWER]["image_requests"] == 2

    # A finished request is forgotten.
    env.sched.request_finished(request)
    assert "r1" not in env.sched._seen_requests


def test_on_the_card_a_new_item_goes_to_the_workers(make_env):
    env = make_env()
    _listen(env)
    env.sched.set_tower_on_card(TOWER, True)
    request = _request("r1", ("a", 0, 4))

    assert env.admit(request) is True
    assert env.sched._encodes == {}
    env.sched.update_state_after_alloc(request, 0)
    assert env.sched.build_connector_meta(scheduler_output=None).loads == {}
    stats = env.sched.host_tower_stats()[TOWER]
    assert stats["items_encoded_card"] == 1
    assert stats["items_encoded_host"] == 0


def test_an_item_on_its_way_through_the_host_keeps_waiting_for_it(make_env):
    gate = threading.Event()
    env = make_env(gate=gate)
    _listen(env)
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is False  # the host encode is queued

    # The tower moves to the card while the host encode runs.
    env.sched.set_tower_on_card(TOWER, True)
    assert env.admit(request) is False
    gate.set()
    env.wait("a")
    assert env.admit(request) is True

    # It is loaded from the region: the card does not encode it again.
    cache: dict = {}
    meta, _ = env.load(request, cache)
    assert set(meta.loads) == {"a"}
    stats = env.sched.host_tower_stats()[TOWER]
    assert stats["items_encoded_host"] == 1
    assert stats["items_encoded_card"] == 0


def test_an_item_already_in_the_region_is_loaded_on_the_card_too(make_env):
    env = make_env()
    _listen(env)
    first = _request("r1", ("a", 0, 4))
    env.admit(first)
    env.wait("a")
    assert env.admit(first) is True

    env.sched.set_tower_on_card(TOWER, True)
    second = _request("r2", ("a", 0, 4))
    assert env.admit(second) is True
    env.sched.update_state_after_alloc(second, 0)
    assert set(env.sched.build_connector_meta(scheduler_output=None).loads) == {"a"}


def test_demotion_while_an_image_request_waits_sends_it_to_the_host(make_env):
    env = make_env()
    _listen(env)
    env.sched.set_tower_on_card(TOWER, True)
    request = _request("r1", ("a", 0, 4))
    assert env.admit(request) is True  # bound for the card, not yet scheduled

    # The tower leaves the card before the request's item was scheduled.
    env.sched.set_tower_on_card(TOWER, False)
    assert env.admit(request) is False
    assert set(env.sched._encodes) == {"a"}
    env.wait("a")
    assert env.admit(request) is True
    assert env.sched.host_tower_stats()[TOWER]["items_encoded_host"] == 1


def test_encode_seconds_are_counted(make_env):
    env = make_env()
    _listen(env)
    request = _request("r1", ("a", 0, 4))
    env.admit(request)
    env.wait("a")
    env.admit(request)
    assert env.sched.host_tower_stats()[TOWER]["encode_seconds_host"] > 0


def test_stats_written_by_the_controller_are_reported(make_env):
    env = make_env()
    _listen(env)
    env.sched.update_tower_stats(TOWER, location=2.0, promotions=1.0)
    stats = env.sched.host_tower_stats()[TOWER]
    assert (stats["location"], stats["promotions"]) == (2.0, 1.0)


# ----------------------------------------------------------------------
# HostTower.release
# ----------------------------------------------------------------------


def test_release_waits_for_the_encode_in_flight_then_frees_the_host():
    gate = threading.Event()
    tower = make_tower(gate)
    tower.to_host()
    from tests.v1.ec_connector.unit.host_tower.utils import make_item

    encode = tower.submit([("image", make_item("a", 3))])
    released = tower.release(wait=False)
    assert not released.done()
    assert tower.location.value == "host"
    gate.set()
    assert len(encode.result(timeout=60).embeddings) == 1
    assert released.result(timeout=60) > 0
    assert tower.location.value == "disk"
    assert tower.nbytes == 0
    # The modalities stay known; the next encode brings the tower back.
    assert tower.modalities == frozenset({"image"})
    again = tower.submit([("image", make_item("b", 3))])
    assert len(again.result(timeout=60).embeddings) == 1
    assert tower.location.value == "host"
    assert tower._weights.loads == 2
    tower.shutdown()


def test_release_of_a_tower_not_in_host_memory_gives_nothing():
    tower = make_tower()
    assert tower.release() is not None
    assert tower.release(wait=False).result(timeout=60) == 0
    tower.shutdown()
