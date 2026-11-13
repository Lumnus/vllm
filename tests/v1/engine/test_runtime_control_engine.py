# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's side of the runtime control surface: every rank or
none, a partial apply undone, engine-core operations (KV, towers)."""

from types import SimpleNamespace

import pytest

import vllm.v1.engine.runtime_control as erc
import vllm.v1.worker.runtime_control as wrc
from vllm.config.multimodal import HostTowerOptions
from vllm.v1.engine.tower_residency import TowerPhase, TowerPlace

from ..worker.test_runtime_control_worker import _worker


class _Engine:
    """collective_rpc to real worker handlers on stand-in workers."""

    def __init__(self, num_ranks: int = 2) -> None:
        self.workers = [_worker(rank=r) for r in range(num_ranks)]
        self.calls: list[tuple] = []
        self.resizes: list[tuple] = []
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_extensible_kv_cache=True,
                num_gpu_blocks=100,
                kv_cache_release_chunk_mib=None,
                kv_cache_grow_torch_floor_mib=None,
            )
        )
        self._tower_residency = None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        assert method == "runtime_control"
        self.calls.append(args[:2])
        return [wrc.handle(w, *args) for w in self.workers]

    def get_kv_cache_resize_state(self):
        return {"state": "idle", "num_blocks": 100}

    def resize_kv_cache(self, n, deadline_s=None, preempt=False):
        self.resizes.append((n, deadline_s, preempt))
        return {"state": "finished", "num_blocks": n}

    def abort_kv_cache_resize(self, reason):
        return {"state": "aborted", "message": reason}

    def _check_kv_cache_resizable(self):
        pass


@pytest.fixture(autouse=True)
def _ranks(monkeypatch):
    monkeypatch.setattr(wrc, "_ep_rank", lambda worker: worker.rank)


def test_every_rank_checks_before_any_rank_applies():
    engine = _Engine()
    engine.workers[1].model_runner.extensible_kv_cache = None  # refuses
    control = erc.RuntimeControl(engine)
    with pytest.raises(erc.ControlRefused, match="rank 1: the KV cache is not"):
        control.apply("kv.set", {"grow_floor_mib": 512})
    assert engine.calls == [("check", "kv.set")]
    assert engine.workers[0].cache_config.kv_cache_grow_torch_floor_mib is None


def test_applied_on_every_rank_and_in_the_engine_core():
    engine = _Engine()
    reply = erc.RuntimeControl(engine).apply("kv.set", {"grow_floor_mib": 512})
    assert reply["applied"] == "all"
    assert [r["changes"] for r in reply["ranks"]] == [
        {"grow_floor_mib": [None, 512]}
    ] * 2
    assert engine.vllm_config.cache_config.kv_cache_grow_torch_floor_mib == 512
    assert all(
        w.cache_config.kv_cache_grow_torch_floor_mib == 512 for w in engine.workers
    )


def test_a_rank_that_cannot_apply_is_reported_and_the_others_undone(monkeypatch):
    engine = _Engine()
    real = wrc._set_kv

    def set_kv(worker, values):
        if worker.rank == 1 and values.get("grow_floor_mib") == 512:
            raise MemoryError("no room")
        real(worker, values)

    monkeypatch.setattr(wrc, "_set_kv", set_kv)
    reply = erc.RuntimeControl(engine).apply("kv.set", {"grow_floor_mib": 512})
    assert reply["applied"] == "none"
    assert reply["rolled_back"] is True
    assert "rank 1: MemoryError: no room" in reply["error"]
    assert [c[0] for c in engine.calls] == ["check", "apply", "restore"]
    assert all(
        w.cache_config.kv_cache_grow_torch_floor_mib is None for w in engine.workers
    )
    assert engine.vllm_config.cache_config.kv_cache_grow_torch_floor_mib is None


def test_an_operation_that_cannot_be_undone_reports_partial(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(wrc, "_check_scrub", lambda worker, args: None)

    def scrub(worker, args):
        if worker.rank == 1:
            raise RuntimeError("device read failed")
        return {}, {"scrub": []}, None

    monkeypatch.setitem(wrc.OPS, "store.scrub", wrc._Op(lambda w, a: None, scrub))
    reply = erc.RuntimeControl(engine).apply("store.scrub", {})
    assert reply["applied"] == "partial" and reply["rolled_back"] is False


def test_the_state_reads_the_engine_core_and_every_rank():
    engine = _Engine()
    state = erc.RuntimeControl(engine).state()
    assert state["kv"]["resize"] == {"state": "idle", "num_blocks": 100}
    assert [r["rank"] for r in state["ranks"]] == [0, 1]
    assert state["ranks_agree"] == {"kv_blocks": True}
    assert "eplb.set" in state["operations"] and "tower.promote" in state["operations"]


def test_kv_operations_of_the_engine_core():
    engine = _Engine()
    control = erc.RuntimeControl(engine)
    assert control.apply("kv.resize", {"num_blocks": 80, "deadline_s": 5})["kv"] == {
        "state": "finished",
        "num_blocks": 80,
    }
    assert engine.resizes == [(80, 5, False)]
    with pytest.raises(erc.ControlRefused):
        control.apply("kv.resize", {"num_blocks": 0})
    engine.workers[1].model_runner.extensible_kv_cache.capacity_blocks = 300
    control.apply("kv.fit", {})
    assert engine.resizes[-1] == (300, None, False)  # the poorest rank's capacity
    assert control.apply("kv.abort", {})["kv"]["state"] == "aborted"


def test_fault_operations_need_development_mode(monkeypatch):
    monkeypatch.setattr(erc.envs, "VLLM_SERVER_DEV_MODE", False)
    with pytest.raises(erc.ControlRefused, match="development mode"):
        erc.RuntimeControl(_Engine()).apply("fault.arm", {"stage": "load"})


class _Tower:
    def __init__(self, place=TowerPlace.HOST) -> None:
        self.name = "vision_tower"
        self.options = HostTowerOptions(residency="dynamic")
        self.place = place
        self.phase = TowerPhase.STEADY
        self.last_use = 0.0
        self.hold_until = 0.0
        self.promotions = self.demotions = self.failed_promotions = 0
        self.started: list[str] = []
        self.demoted = 0
        self._delegate = SimpleNamespace(host_tower_stats=lambda: {})

    def _clock(self) -> float:
        return 100.0

    def _resize_active(self) -> bool:
        return False

    def _start_promotion(self, now, reason) -> None:
        self.started.append(reason)

    def _demote(self, now, idle) -> None:
        self.demoted += 1

    def record(self):
        return {
            "place": self.place.value,
            "phase": self.phase.value,
            "arrivals_in_window": 0,
            "promotions": 0,
            "demotions": 0,
            "failed_promotions": 0,
        }


def _with_tower(tower):
    engine = _Engine()
    engine._tower_residency = SimpleNamespace(
        towers={tower.name: tower}, _clock=lambda: 100.0
    )
    return engine


def test_tower_promote_demote_and_settings():
    tower = _Tower()
    engine = _with_tower(tower)
    control = erc.RuntimeControl(engine)
    control.apply("tower.promote", {})
    assert tower.started == ["asked over the runtime control surface"]
    with pytest.raises(erc.ControlRefused, match="not on the"):
        control.apply("tower.demote", {})
    tower.place = TowerPlace.CARD
    control.apply("tower.demote", {"name": "vision_tower"})
    assert tower.demoted == 1
    tower.phase = TowerPhase.STAGING
    with pytest.raises(erc.ControlRefused, match="settles"):
        control.apply("tower.promote", {})
    reply = control.apply(
        "tower.set",
        {"promote_after": 1, "demote_idle_s": 60.0, "idle_location": "disk"},
    )
    assert reply["changes"] == {
        "promote_after": [2, 1],
        "demote_idle_s": [1800.0, 60.0],
        "idle_location": ["host", "disk"],
    }
    assert tower.options.promote_after == 1
    with pytest.raises(erc.ControlRefused, match="invalid"):
        control.apply("tower.set", {"promote_after": 0})
    with pytest.raises(erc.ControlRefused, match="unknown tower settings"):
        control.apply("tower.set", {"card_layout": "sharded"})
    assert tower.options.promote_after == 1
    state = control.state()["towers"]["vision_tower"]
    assert state["options"]["demote_idle_s"] == 60.0


def test_without_a_dynamic_tower_tower_operations_are_refused():
    with pytest.raises(erc.ControlRefused, match="no encoder tower"):
        erc.RuntimeControl(_Engine()).apply("tower.promote", {})
