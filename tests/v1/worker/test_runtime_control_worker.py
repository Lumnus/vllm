# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The worker side of the runtime control surface: state, check, apply,
restore, on stand-ins for the subsystems."""

import json
from types import SimpleNamespace

import pytest

import vllm.v1.worker.runtime_control as wrc
from vllm.distributed.eplb import fault_injection


class _Buffer:
    granularity = 2 << 20

    def __init__(self) -> None:
        self.max_chunk_bytes = None


class _Extensible:
    """`ExtensibleTensor`'s surface: the chunk lives in its VA buffer."""

    physical_bytes = 100 << 20

    def __init__(self) -> None:
        self._buffer = _Buffer()

    @property
    def max_chunk_bytes(self):
        return self._buffer.max_chunk_bytes


class _KV:
    def __init__(self) -> None:
        self.num_committed_blocks = 100
        self.capacity_blocks = 400
        self.bytes_per_block = 1 << 20
        self.torch_memory_limit = None
        self.last_grow = None
        self.buffer = _Extensible()

    @property
    def chunk(self):
        return self.buffer._buffer.max_chunk_bytes


def _worker(kv=True, rank=0):
    cache_config = SimpleNamespace(
        kv_cache_release_chunk_mib=None, kv_cache_grow_torch_floor_mib=None
    )
    runner = SimpleNamespace(
        eplb_state=None,
        extensible_kv_cache=_KV() if kv else None,
        get_model=lambda: None,
    )
    worker = SimpleNamespace(
        rank=rank,
        model_runner=runner,
        cache_config=cache_config,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
    )
    return worker


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    fault_injection.clear()
    monkeypatch.setattr(wrc, "_ep_rank", lambda worker: worker.rank)
    # The KV buffer's chunk property reads the inner buffer.
    yield
    fault_injection.clear()


def test_state_reports_what_exists_and_serializes():
    worker = _worker()
    out = wrc.handle(worker, "state")
    assert out["rank"] == 0 and out["eplb"] is None and out["tiers"] is None
    assert out["kv"]["committed_blocks"] == 100
    assert out["store"] == [] and out["ple"] == []
    json.dumps(out)
    only = wrc.handle(worker, "state", args={"only": ["kv"]})
    assert set(only) == {"rank", "ep_rank", "kv"}


def test_kv_settings_check_apply_restore():
    worker = _worker()
    assert wrc.handle(worker, "check", "kv.set", {"release_chunk_mib": 0})[
        "refusal"
    ].startswith("release_chunk_mib must be")
    reply = wrc.handle(
        worker, "apply", "kv.set", {"release_chunk_mib": 3, "grow_floor_mib": 1024}
    )
    assert reply["ok"], reply
    assert reply["changes"] == {
        "release_chunk_mib": [None, 3],
        "grow_floor_mib": [None, 1024],
    }
    # Rounded up to the driver granularity (2 MiB here).
    assert worker.model_runner.extensible_kv_cache.chunk == 4 << 20
    assert worker.cache_config.kv_cache_grow_torch_floor_mib == 1024
    undone = wrc.handle(worker, "restore", "kv.set", {"ranks": [0]})
    assert undone["ok"]
    assert worker.model_runner.extensible_kv_cache.chunk is None
    assert worker.cache_config.kv_cache_grow_torch_floor_mib is None


def test_restore_only_on_the_named_ranks():
    worker = _worker(rank=1)
    wrc.handle(worker, "apply", "kv.set", {"grow_floor_mib": 8})
    skipped = wrc.handle(worker, "restore", "kv.set", {"ranks": [0]})
    assert skipped["skipped"]
    assert worker.cache_config.kv_cache_grow_torch_floor_mib == 8


def test_a_refused_apply_leaves_no_undo_of_an_older_change():
    worker = _worker()
    wrc.handle(worker, "apply", "kv.set", {"grow_floor_mib": 8})
    failed = wrc.handle(worker, "apply", "kv.set", {"grow_floor_mib": -1})
    assert not failed["ok"]
    assert not wrc.handle(worker, "restore", "kv.set")["ok"]
    assert worker.cache_config.kv_cache_grow_torch_floor_mib == 8


def test_without_the_subsystem_operations_are_refused():
    worker = _worker(kv=False)
    for op, args in [
        ("kv.set", {"grow_floor_mib": 1}),
        ("eplb.pause", {}),
        ("eplb.set", {"step_interval": 5}),
        ("store.set", {"check_every": 1}),
        ("tiers.set", {"host_tier_bound": 4}),
        ("trace.set", {"enabled": True}),
        ("ple.set", {"stats_interval_s": 10}),
        ("no.such", {}),
    ]:
        reply = wrc.handle(worker, "apply", op, args)
        assert not reply["ok"] and reply["error"], op


def test_fault_injection_is_development_mode_only(monkeypatch):
    worker = _worker()
    monkeypatch.setattr(wrc.envs, "VLLM_SERVER_DEV_MODE", False)
    assert (
        "development mode"
        in wrc.handle(worker, "check", "fault.arm", {"stage": "load"})["refusal"]
    )


def test_fault_arms_on_the_chosen_rank_only(monkeypatch):
    monkeypatch.setattr(wrc.envs, "VLLM_SERVER_DEV_MODE", True)
    monkeypatch.setattr(wrc, "_need_eplb", lambda worker: None)
    rank1 = _worker(rank=1)
    reply = wrc.handle(rank1, "apply", "fault.arm", {"stage": "plan", "rank": 0})
    assert reply["ok"] and not reply["result"]["armed_here"]
    assert fault_injection.armed() == {}
    rank0 = _worker(rank=0)
    reply = wrc.handle(
        rank0, "apply", "fault.arm", {"stage": "plan", "rank": 0, "count": 2}
    )
    assert reply["result"]["armed_here"]
    assert fault_injection.armed() == {"plan": 2}
    wrc.handle(rank0, "apply", "fault.clear", {})
    assert fault_injection.armed() == {}


class _Stats:
    def __init__(self) -> None:
        self.interval_s = 0.0
        self.enabled = False
        self._reset(0.0)

    def _reset(self, now: float) -> None:
        self.started = now
        self.steps = self.lookups = self.hits = self.misses = 0
        self.reads = self.from_lookahead = 0


def test_ple_stats_interval_changes_and_reads(monkeypatch):
    worker = _worker()
    table = SimpleNamespace(_server=SimpleNamespace(stats=_Stats()))
    worker.__dict__["_runtime_control_row_tables"] = [("model.ple", table)]
    reply = wrc.handle(worker, "apply", "ple.set", {"stats_interval_s": 30})
    assert reply["ok"] and reply["changes"] == {"stats_interval_s": [0.0, 30.0]}
    assert table._server.stats.enabled
    table._server.stats.lookups, table._server.stats.hits = 10, 9
    state = wrc.handle(worker, "state", args={"only": ["ple"]})["ple"][0]
    assert state["hit_rate"] == pytest.approx(0.9)
    wrc.handle(worker, "restore", "ple.set")
    assert not table._server.stats.enabled


class _Session:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            dirs=["/d0"],
            check_every=8,
            scrub_every=0,
            stats_interval_s=0.0,
            check_threads=2,
        )
        self.mode = "fill"
        self.status = "complete"
        self.pool_rows = 10
        self.pool_bytes_pinned = 1 << 30
        self.is_open = True
        self.intervals: list[float] = []
        store = SimpleNamespace(
            check_every=8,
            can_check_units=lambda: True,
            stats=lambda: {"reads": 5, "check_failures": 0},
        )

        def set_check_every(n, threads=2):
            store.check_every = n

        store.set_check_every = set_check_every
        self.store = store

    def set_stats_interval(self, s: float) -> None:
        self.intervals.append(s)
        self.config.stats_interval_s = s


def test_store_settings_apply_and_undo(monkeypatch):
    session = _Session()
    monkeypatch.setattr(wrc, "_store_sessions", lambda: [session])
    worker = _worker()
    assert (
        "integer >= 0"
        in wrc.handle(worker, "check", "store.set", {"check_every": -1})["refusal"]
    )
    reply = wrc.handle(
        worker,
        "apply",
        "store.set",
        {"check_every": 1, "scrub_every": 4, "stats_interval_s": 60},
    )
    assert reply["ok"], reply
    assert reply["changes"] == {
        "check_every": [8, 1],
        "scrub_every": [0, 4],
        "stats_interval_s": [0.0, 60],
    }
    assert session.intervals == [60.0]
    wrc.handle(worker, "restore", "store.set")
    assert session.store.check_every == 8 and session.config.scrub_every == 0
    assert session.intervals == [60.0, 0.0]


def test_a_tier_setting_registered_by_a_component(monkeypatch):
    worker = _worker()
    monkeypatch.setattr(wrc, "_tiers", lambda: [object()])
    assert (
        "fixed at load"
        in wrc.handle(worker, "check", "tiers.set", {"host_tier_bound": 4})["refusal"]
    )
    value = {"bound": None}
    monkeypatch.setitem(
        wrc._TIER_SETTINGS,
        "host_tier_bound",
        wrc.RuntimeSetting(
            get=lambda: value["bound"],
            check=lambda v: None if isinstance(v, int) and v >= 0 else "bad",
            apply=lambda v: value.update(bound=v),
        ),
    )
    monkeypatch.setattr(wrc, "_tiers_state", lambda: {"settings": dict(value)})
    reply = wrc.handle(worker, "apply", "tiers.set", {"host_tier_bound": 4})
    assert reply["ok"] and reply["changes"] == {"host_tier_bound": [None, 4]}
    wrc.handle(worker, "restore", "tiers.set")
    assert value["bound"] is None
