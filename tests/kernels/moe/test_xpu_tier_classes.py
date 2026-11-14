# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On CPU: hot / warm / cold as thresholds over EPLB's load window and their
readout against the tiers, EPLB's moves reading CPU-tier experts from the
host cache, prefill staging (a read path that moves nothing), the seed file
shapes, the trace tool's loads, the gauges, and the switches unset. The tier
passes themselves are in tests/distributed/test_eplb_cpu_experts_control.py
and tests/model_executor/offloader/test_expert_tier_map_host.py."""

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_temperature as temp_mod
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_experts import CPU_CFG, _reference_op, _units
from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def op(monkeypatch):
    real = cx.host_op_available()
    if not real:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    gc.collect()  # tiers of earlier tests leave the live set
    cx.reset_runtime()
    yield real
    cx.reset_runtime()
    tstore.reset_sessions()


def _env(monkeypatch, tmp_path, **extra):
    """3 card slots, 2 pinned, 3 on the CPU per layer (of 8)."""
    run(monkeypatch, tmp_path, store=True)
    return run(
        monkeypatch, tmp_path, store=True, pinned=2, store_extra=dict(CPU_CFG, **extra)
    )


# -- classes ------------------------------------------------------------------


def test_classes_with_hysteresis():
    t = np.array([3.0, 1.4, 0.6, 0.45, 0.1])
    c = temp_mod.classify(t, None, 1.5, 0.5, 1.25)
    H, W, C = temp_mod.HOT, temp_mod.WARM, temp_mod.COLD
    assert c.tolist() == [H, W, W, C, C]
    # a hot one at 1.3 stays hot (above 1.5 / 1.25); at 1.1 it is warm
    prev = np.array([H, H, W, W, W], np.int8)
    c = temp_mod.classify(np.array([1.3, 1.1, 0.45, 0.39, 2.0]), prev, 1.5, 0.5, 1.25)
    assert c.tolist() == [H, W, W, C, H]


def test_the_readout_counts_classes_against_rows():
    H, W, C = temp_mod.HOT, temp_mod.WARM, temp_mod.COLD
    cls = np.array([H] * 4 + [W] * 3 + [C], np.int8)
    tier = np.array([FAST, FAST, FAST, HOST, POOL, HOST, HOST, FAST], np.int8)
    r = temp_mod.class_readout(cls, tier, 4, 1)
    assert (r["hot"], r["warm"], r["cold"]) == (4, 3, 1)
    assert r["hot_on_card"] == 3 and r["cold_on_card"] == 1
    assert r["warm_or_hot_on_cpu"] == 3
    assert r["card_short"] == 0 and r["rows_short"] == 2


def test_the_temperature_is_eplbs_window(monkeypatch, tmp_path, op):
    """One signal: the load the last EPLB move or fill planned with, over the
    mean expert's. The classes cut the same order the tiers are cut from."""
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    assert tiers.temperature() is None and tiers.class_readout(1.5, 0.5, 1.1) is None
    load = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    tiers.fill_to(np.arange(8), load, 1.1)
    assert tiers.temperature() == pytest.approx(load / load.mean())
    r = tiers.class_readout(1.5, 0.5, 1.1)
    # temperatures 0 .29 .57 .86 1.1 1.4 1.7 2.0: hot 6 7, warm 2-5, cold 0 1
    assert (r["hot"], r["warm"], r["cold"]) == (2, 4, 2)
    assert r["hot_on_card"] == 2 and r["warm_or_hot_on_cpu"] == 1  # slot 2
    assert r["rows_short"] == 1  # 6 warm or hot, 3 card + 2 pinned rows
    # an EPLB move replaces the window: slot 0 hot now
    load2 = np.array([9, 1, 1, 1, 1, 9, 9, 9], dtype=float)
    tiers.commit(tiers.prepare(np.arange(8), np.arange(8), load2, 1.1))
    assert tiers.temperature()[0] == pytest.approx(9 / load2.mean())
    state = xt.tier_classes_state(1.5, 0.5, 1.1)
    assert state["layers"] == 1 and state["totals"]["hot"] >= 4


def test_eplbs_moves_read_cpu_experts_from_the_host_cache(monkeypatch, tmp_path, op):
    """A CPU slot entering a pinned row in an EPLB move is copied from the host
    cache (the cache row held while it is copied), not read from the drive."""
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    cache = tiers.cpu.cache
    deadline = time.time() + 5
    while cache._filling and time.time() < deadline:
        time.sleep(0.01)
    assert (0, 6) in cache
    reads = []
    store = tiers.store_binding.session.store
    real = store.read_into_async
    store.read_into_async = lambda k, d: (reads.append(k), real(k, d))[1]
    load = np.array([9, 9, 9, 1, 1, 0, 8, 0], dtype=float)
    tiers.commit(tiers.prepare(np.arange(8), np.arange(8), load, 1.1))
    store.read_into_async = real
    p = tiers.placement
    # expert 6 came from the host cache; the drive read that follows is the
    # host cache reading ahead the slot that went to the CPU (3)
    assert p.tier[6] == POOL and (0, 6) not in reads and reads == [(0, 3)]
    for name, t in tiers.slow.items():
        want = expert_in_store(env, 0, 6)[name].reshape(-1).view(torch.uint8)
        assert torch.equal(t[p.row[6]].reshape(-1).view(torch.uint8), want)


def test_a_row_being_copied_out_is_not_evicted():
    from .test_xpu_cpu_experts import _cache

    c = _cache(rows=1)
    c.release(c.acquire([(0, 1)]))
    dests = {n: torch.zeros_like(t) for n, t in c.dests(c.row_of((0, 1))).items()}
    entered, gate = [], []

    class SlowDst:
        """A destination whose copy waits until the test has tried an
        eviction."""

        def __init__(self, t):
            self.t = t

        def copy_(self, src):
            entered.append(1)
            while not gate:
                time.sleep(0.005)
            self.t.copy_(src)

    slow = {n: SlowDst(t) for n, t in dests.items()}
    with ThreadPoolExecutor(1) as ex:
        f = ex.submit(c.copy_unit_out, (0, 1), slow)
        while not entered:
            time.sleep(0.005)
        with pytest.raises(cx.CpuExpertError, match="raise host_cache_gib"):
            c.acquire([(0, 2)])  # the only row is held by the copy
        assert c.resize_refusal() is not None
        gate.append(1)
        assert f.result(2) is True
    assert dests["w13"][:2].tolist() == [0, 1] and not c._held
    assert c.copy_unit_out((0, 9), dests) is False
    c.release(c.acquire([(0, 2)]))


# -- prefill staging --------------------------------------------------------------


def _pending(tiers, ids, w, x):
    rt = tiers.cpu
    n = ids.shape[0]
    pending = cx.Pending(
        tiers,
        n,
        SimpleNamespace(synchronize=lambda: None),
        rt._buffers_for(n, 2, HIDDEN),
    )
    pending.buffers.ids[:n].copy_(ids)
    pending.buffers.weights[:n].copy_(w)
    pending.buffers.x[:n].copy_(x)
    return pending


def test_a_large_call_stages_its_busy_cpu_experts_to_the_card(
    monkeypatch, tmp_path, op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=3)
    tiers = env.tiers[1]
    rt = tiers.cpu
    tier_before = tiers.placement.tier.copy()
    ids = torch.tensor([[6, 0], [6, 1], [6, 7], [5, 6]], dtype=torch.int32)
    w = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    pending = _pending(tiers, ids, w, x)
    impl = rt.stage(pending, torch.device("cpu"))
    assert impl is not None
    rows = xt._STAGE_ROWS
    assert rows.num_rows == tiers.num_host == 3  # one per CPU slot by default
    # expert 6 (4 calls) is staged in row 0: its map entry and its bytes
    assert rows.map.tolist()[6] == 0 and sum(v >= 0 for v in rows.map.tolist()) == 1
    for name, unit in expert_in_store(env, 1, 6).items():
        assert torch.equal(rows.rows_u8(name)[0], unit.reshape(-1).view(torch.uint8))
    assert impl.expert_map is rows.map and impl.num_experts == rows.num_rows
    assert tiers.stage_impl(rows) is impl  # built once
    # the CPU computes the rest: expert 7 (token 2) and 5 (token 3)
    out = rt.finish(pending, torch.device("cpu"))
    want = cx.reference_int4_experts(
        x, _units(env, 1, {5, 7}), [(7, 2, 0.3), (5, 3, 0.2)], INTER, GROUP
    )
    if op:
        assert cx.bf16_steps(out, want) <= 1.0
    else:
        assert torch.equal(out, want)
    s = rt.snapshot()
    assert s["routed_staged"] == 4 and s["routed_cpu"] == 2
    assert s["stage_layers"] == 1 and s["stage_experts"] == 1
    # a read path: no slot changed tier
    assert np.array_equal(tiers.placement.tier, tier_before)


def test_staging_off_or_a_small_call_changes_nothing(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    rt = tiers.cpu
    ids = torch.tensor([[6, 7]], dtype=torch.int32)
    pending = _pending(tiers, ids, torch.ones(1, 2), torch.zeros(1, HIDDEN).bfloat16())
    assert rt.stage(pending, torch.device("cpu")) is None
    assert not pending.waited and pending.selection is None  # did not even wait
    rt.config.stage_min_calls = 2  # what cpu.set changes
    assert rt.stage(pending, torch.device("cpu")) is None  # 1 call each
    assert xt._STAGE_ROWS is None


def test_the_layer_runs_the_staged_call_between_the_card_and_the_cpu():
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    order = []

    class Impl:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def apply(self, output, **kw):
            order.append(self.name)
            output.fill_(self.value)

    class Cpu:
        def post(self, tiers, h, ids, w, weights=True):
            order.append("post")
            return "pending"

        def stage(self, pending, device):
            order.append("stage")
            return Impl("staged", 1000.0)

        def finish(self, pending, device):
            order.append("finish")
            return torch.full((2, 4), 100.0)

    experts = xpu_moe.XPUExperts.__new__(xpu_moe.XPUExperts)
    experts.tiers = SimpleNamespace(num_global=8, cpu=Cpu())
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    experts.impl_apply_kwargs = {}
    experts.placement_trace = None
    experts._tier_impls = lambda w1, w2, k: (Impl("fast", 1.0), Impl("slow", 10.0))
    out = torch.zeros(2, 4)
    experts._apply_tiered(
        out, torch.zeros(2, 4), None, None, torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int32),
    )  # fmt: skip
    assert order == ["post", "fast", "slow", "stage", "staged", "finish"]
    assert torch.equal(out, torch.full((2, 4), 1111.0))


# -- seed, trace tool, gauges --------------------------------------------------------


def test_the_seed_reads_three_shapes(tmp_path):
    from vllm.distributed.eplb.eplb_state_file import read_seed_loads

    names = [f"model.language_model.layers.{i}.mlp.experts" for i in range(3)]
    state = tmp_path / "state.json"
    state.write_text(json.dumps({names[0]: [1, 2, 3, 4], names[1]: [1, 2]}))
    assert read_seed_loads(str(state), names, 4) == {names[0]: [1, 2, 3, 4]}
    ranks = tmp_path / "rankings.json"
    ranks.write_text(json.dumps({"1": [3, 0], "2": [9, 1, 2, 0, 3], "7": [0]}))
    got = read_seed_loads(str(ranks), names, 4)
    assert got[names[1]] == [1, 0, 0, 2]
    assert got[names[2]] == [2, 4, 3, 1]  # id 9 is outside: skipped
    learned = tmp_path / "learned.json"
    learned.write_text(
        json.dumps({"periods": 20, "layers": {"0": {"2": 4.0, "1": 1.0}}})
    )
    assert read_seed_loads(str(learned), names, 4) == {
        names[0]: [0, 250000, 1000000, 0]
    }
    assert read_seed_loads(str(tmp_path / "missing.json"), names, 4) == {}


def test_the_trace_tool_writes_loads(tmp_path):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[3] / "tools" / "analyze_expert_trace.py"
    spec = importlib.util.spec_from_file_location("aet", path)
    aet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aet)
    counts = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint32)
    trace = aet.Trace(["a", "b"], 10, np.array([10, 20]), np.zeros(2), counts)
    out = tmp_path / "loads.json"
    aet.write_loads(trace, str(out))
    assert json.loads(out.read_text()) == {"a": [6, 8], "b": [10, 12]}
    aet.write_loads(trace, str(out), last=1)
    assert json.loads(out.read_text()) == {"a": [5, 6], "b": [7, 8]}


def test_the_gauges_carry_the_classes_and_staging():
    from vllm.v1.engine.runtime_control import (
        EXPERT_TIER_GAUGES,
        expert_tier_series,
    )

    counters = dict.fromkeys(
        ["routed_card", "routed_pinned", "routed_cpu", "layers_with_cpu_calls",
         "layers_without", "cpu_experts", "cache_hits", "cache_misses",
         "cache_prefetched", "cache_evicted", "promotions", "demotions",
         "routed_staged", "stage_layers", "stage_experts"], 2,
    ) | {"compute_ms": 1.0, "read_ms": 0.0, "wait_ms": 0.0}  # fmt: skip
    counters["stage_ms"] = 500.0
    reply = {
        "rank": 1,
        "cpu": {"counters": counters, "cache": {"rows": 4}, "cpu_threads": 8},
        "tiers": {
            "settings": {"host_tier_bound": 22},
            "slots_now": {"card": 1, "pinned": 1, "cpu": 1},
            "classes": {
                "totals": dict.fromkeys(temp_mod.CLASS_KEYS, 3),
                "tier_passes": 7,
            },
        },
    }
    series = expert_tier_series([reply])["1"]
    assert set(series) == set(EXPERT_TIER_GAUGES)
    assert series["stage_seconds"] == 0.5 and series["tier_passes"] == 7


def test_the_readout_over_the_control_surface(monkeypatch, tmp_path, op):
    from vllm.config.parallel import EPLBConfig
    from vllm.v1.worker import runtime_control as wrc

    env = _env(monkeypatch, tmp_path)
    env.tiers[0].fill_to(np.arange(8), np.arange(8, dtype=float), 1.1)
    eplb = SimpleNamespace(
        parallel_config=SimpleNamespace(eplb_config=EPLBConfig(tier_warm=0.4)),
        tier_passes=3,
        model_states={},
    )
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=eplb))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    state = wrc.handle(worker, "state", args={"only": ["tiers"]})["tiers"]
    classes = state["classes"]
    assert classes["thresholds"]["warm"] == 0.4 and classes["tier_passes"] == 3
    assert classes["layers"] == 1 and "per_layer" not in classes
    json.dumps(state)


# -- with every switch unset ---------------------------------------------------------


def test_unset_hashes_are_equal_and_the_removed_promotion_is_refused():
    from vllm.config.expert_store import ExpertStoreConfig
    from vllm.config.parallel import EPLBConfig

    base = ExpertStoreConfig(dirs=["/x"])
    on = ExpertStoreConfig(dirs=["/x"], stage_min_calls=8, stage_rows=64)
    assert base.compile_factors() == on.compile_factors()
    e0 = EPLBConfig()
    e1 = EPLBConfig(
        tier_interval=64, tier_max_moves=4, tier_hot=2.0, tier_warm=0.3,
        seed_path="/s.json",
    )  # fmt: skip
    assert e0.compile_factors() == e1.compile_factors()
    with pytest.raises(ValueError, match="tier_warm"):
        EPLBConfig(tier_hot=0.5, tier_warm=1.0)
    with pytest.raises(ValueError, match="EPLB's tier pass"):
        ExpertStoreConfig(dirs=["/x"], host_promote_misses=4)


def test_unset_a_full_rearrangement_reads_within_the_spare_rows(
    monkeypatch, tmp_path, op
):
    """Without a tier pass EPLB's move reads at most the spare rows and cuts
    the tiers from its window, as on the base; a hot CPU slot gets a pinned
    row."""
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    load = np.array([9, 9, 9, 5, 5, 0, 8, 0], dtype=float)
    move = tiers.prepare(np.arange(8), np.arange(8), load, 1.1)
    assert move.stats["units"] <= tiers.placement.spare
    tiers.commit(move)
    assert tiers.placement.tier[6] == POOL
