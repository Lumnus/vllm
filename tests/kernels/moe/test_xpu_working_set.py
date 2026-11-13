# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert working set (``xpu_working_set``) on CPU: temperature and
classes, the swap choice and its cost rule, batches read in the background and
committed between steps, their interplay with EPLB's moves and the pinned
bound, prefill staging of CPU-tier experts, the tier seed, the readout, the
runtime settings and the metrics. Same harness and op stand-in as
``test_xpu_cpu_experts.py``."""

import gc
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.fused_moe import xpu_working_set as wsm
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_experts import CPU_CFG, _reference_op, _units
from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup

WS_ON = {
    "working_set": True,
    "ws_halflife_steps": 4.0,
    "ws_move_us": 0.0,
    "ws_batch": 2,
    "ws_units_per_step": 8,
}


@pytest.fixture
def op(monkeypatch):
    real = cx.host_op_available()
    if not real:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    gc.collect()  # tiers of earlier tests leave the live set
    cx.reset_runtime()
    wsm.reset_working_set()
    xt.reset_stage_rows()
    yield real
    cx.reset_runtime()
    wsm.reset_working_set()
    xt.reset_stage_rows()
    tstore.reset_sessions()


def _env(monkeypatch, tmp_path, **extra):
    """3 card slots, 2 pinned, 3 on the CPU per layer (of 8)."""
    run(monkeypatch, tmp_path, store=True)
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=2, store_extra=cfg)


def _serve(tiers, ids, times=4):
    ids = torch.tensor(ids, dtype=torch.int32)
    x = torch.zeros(ids.shape[0], HIDDEN, dtype=torch.bfloat16)
    w = torch.full(ids.shape, 0.5)
    for _ in range(times):
        tiers.cpu.compute(tiers, ids, w, x, torch.empty_like(x))


def _steps(env, until, limit=200):
    """Between-steps calls of every layer (one engine step each) until
    ``until()`` holds."""
    for _ in range(limit):
        for t in env.tiers:
            t.between_steps()
        if until():
            return
        time.sleep(0.005)
    raise AssertionError("not reached")


# -- the pure decisions -------------------------------------------------------


def test_classes_with_hysteresis():
    temp = np.array([3.0, 1.4, 0.6, 0.45, 0.1])
    c = wsm.classify(temp, None, 1.5, 0.5, 1.25)
    assert c.tolist() == [wsm.HOT, wsm.WARM, wsm.WARM, wsm.COLD, wsm.COLD]
    # a hot slot that cools to 1.3 stays hot (above 1.5 / 1.25 = 1.2); at
    # 1.1 it drops to warm; a warm one at 0.45 stays warm (above 0.4)
    prev = np.array([wsm.HOT, wsm.HOT, wsm.WARM, wsm.WARM, wsm.WARM], np.int8)
    temp = np.array([1.3, 1.1, 0.45, 0.39, 2.0])
    c = wsm.classify(temp, prev, 1.5, 0.5, 1.25)
    assert c.tolist() == [wsm.HOT, wsm.WARM, wsm.WARM, wsm.COLD, wsm.HOT]


def test_swaps_reinject_warm_and_hot_experts_by_cost():
    s = wsm.WorkingSetSettings(ws_move_us=0.0, ws_halflife_steps=64)
    # slots: 0-1 card, 2 pinned, 3-5 CPU
    tier = np.array([FAST, FAST, POOL, HOST, HOST, HOST], np.int8)
    temp = np.array([0.1, 2.0, 0.2, 3.0, 0.8, 0.05])
    cls = wsm.classify(temp, None, 1.5, 0.5, 1.25)
    rate = temp * 2 / 8
    new, swaps = wsm.plan_swaps(tier, temp, cls, rate, 16, 1, s, 4)
    # the hot CPU slot takes the coldest card row, the warm one the pinned row
    # of a colder slot; the cold CPU slot never moves up
    assert new[3] == FAST and new[0] == HOST
    assert new[4] == POOL and new[2] == HOST
    assert new[5] == HOST and new[1] == FAST
    assert [(sw.lower, sw.upper) for sw in swaps] == [(HOST, FAST), (HOST, POOL)]
    # the cost rule: a move worth more than one half-life of gain is refused
    s2 = wsm.WorkingSetSettings(ws_move_us=1e9)
    _, none = wsm.plan_swaps(tier, temp, cls, rate, 16, 1, s2, 4)
    assert none == []
    # the budget bounds the swaps
    _, one = wsm.plan_swaps(tier, temp, cls, rate, 16, 1, s, 1)
    assert len(one) == 1 and one[0].entrant == 3


def test_within_a_class_a_swap_needs_the_hysteresis_ratio():
    s = wsm.WorkingSetSettings(ws_move_us=0.0, ws_hysteresis=1.25)
    tier = np.array([FAST, HOST], np.int8)
    for temp_host, moves in ((1.0, False), (1.3, True)):
        temp = np.array([1.0 if not moves else 1.0, temp_host])
        temp[0] = 1.0
        cls = np.array([wsm.WARM, wsm.WARM], np.int8)
        _, swaps = wsm.plan_swaps(tier, temp, cls, temp / 4, 16, 0, s, 2)
        assert bool(swaps) is moves


def test_a_free_pinned_row_is_taken_without_a_victim():
    s = wsm.WorkingSetSettings(ws_move_us=0.0)
    tier = np.array([FAST, POOL, HOST], np.int8)
    temp = np.array([5.0, 4.0, 1.0])
    cls = wsm.classify(temp, None, 1.5, 0.5, 1.25)
    new, swaps = wsm.plan_swaps(tier, temp, cls, temp / 4, 16, 2, s, 2)
    assert new.tolist() == [FAST, POOL, POOL]
    assert swaps[0].displaced is None


def test_the_readout_counts_classes_against_rows():
    cls = np.array([wsm.HOT] * 4 + [wsm.WARM] * 3 + [wsm.COLD], np.int8)
    tier = np.array([FAST, FAST, FAST, HOST, POOL, HOST, HOST, FAST], np.int8)
    r = wsm.class_readout(cls, tier, 4, 1)
    assert (r["hot"], r["warm"], r["cold"]) == (4, 3, 1)
    assert r["hot_on_card"] == 3 and r["cold_on_card"] == 1
    assert r["warm_or_hot_on_cpu"] == 3
    assert r["card_short"] == 0 and r["rows_short"] == 2


# -- the layer ------------------------------------------------------------------


def test_a_hot_cpu_expert_is_reinjected_to_the_card_between_steps(
    monkeypatch, tmp_path, op
):
    env = _env(monkeypatch, tmp_path, **WS_ON)
    tiers = env.tiers[0]
    p = tiers.placement
    assert p.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
    _serve(tiers, [[6, 7]] * 4)
    assert tiers.temperature()[6] > 1.5
    _steps(env, lambda: p.tier[6] == FAST and p.tier[7] == FAST)
    ws = tiers._ws
    assert ws.counters.committed >= 1 and ws.counters.units_cache >= 2
    assert ws.counters.swaps["cpu>card"] >= 2
    # the rows hold the experts, the maps say so, and the CPU serves none
    summary = tiers.scrub(np.arange(8), full=True)
    assert summary["mismatches"] == 0 and summary["maps_equal"]
    before = tiers.cpu.snapshot()
    _serve(tiers, [[6, 7]] * 4, times=1)
    after = tiers.cpu.snapshot()
    assert after["routed_cpu"] == before["routed_cpu"]
    assert after["routed_card"] - before["routed_card"] == 8
    # the readout
    r = tiers.working_set_readout()
    assert r["hot_on_card"] >= 2 and r["warm_or_hot_on_cpu"] == 0
    state = xt.working_set_state()
    assert state["totals"]["hot"] >= 2 and state["layers"] == 2


def test_nothing_moves_with_the_working_set_off(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, ws_halflife_steps=4.0)
    tiers = env.tiers[0]
    _serve(tiers, [[6, 7]] * 4)
    for _ in range(10):
        for t in env.tiers:
            t.between_steps()
    assert tiers.placement.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
    assert tiers._ws.counters.batches == 0
    # the classes are still read out (the mismatch is visible when off)
    assert tiers.working_set_readout()["warm_or_hot_on_cpu"] == 2


def test_an_eplb_prepare_drops_a_batch_being_read(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, **WS_ON)
    tiers = env.tiers[0]
    p = tiers.placement
    free = p.pool.free_count
    held = Future()  # a drive read that has not returned
    store = tiers.store_binding.session.store
    real_read = store.read_into_async
    tiers.cpu.cache.copy_out_async = lambda *a: None
    store.read_into_async = lambda k, d: held
    _serve(tiers, [[6, 7]] * 4)
    tiers.between_steps()
    assert tiers._ws_batch is not None and p.pool.free_count == free - 2
    tiers.between_steps()  # not landed: the step does not wait, nothing moves
    assert tiers._ws_batch is not None and p.tier[6] == HOST
    store.read_into_async = real_read  # the real store for EPLB's own reads
    del tiers.cpu.cache.copy_out_async
    held.set_result("direct")
    old = np.arange(8)
    new = old.copy()
    new[3], new[5] = 5, 3
    load = np.array([9, 9, 9, 1, 4, 8, 0, 0], dtype=float)
    move = tiers.prepare(old, new, load, 1.1)
    assert tiers._ws_batch is None and tiers._ws.counters.dropped == 1
    tiers.commit(move)
    # with the working set on EPLB's move keeps every slot's tier
    assert p.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
    assert tiers.slot_logical[3] == 5 and tiers.slot_logical[5] == 3
    # the slot with a new expert starts from that expert's load
    assert tiers._heat[3] == pytest.approx(8 / load.sum() * 2)
    assert tiers.scrub(new, full=True)["mismatches"] == 0


def test_a_batch_outrun_by_a_bound_change_is_dropped(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, **WS_ON)
    tiers = env.tiers[0]
    gate = Future()
    real = tiers.cpu.cache.copy_out_async

    def slow_copy(key, dests, executor):
        f = real(key, dests, executor)
        out = Future()
        f.add_done_callback(
            lambda _: gate.add_done_callback(lambda _: out.set_result(1))
        )
        return out

    monkeypatch.setattr(tiers.cpu.cache, "copy_out_async", slow_copy)
    _serve(tiers, [[6, 7]] * 4)
    tiers.between_steps()
    assert tiers._ws_batch is not None
    tiers.set_pinned_bound(1)
    assert tiers._ws_batch is None and tiers._ws.counters.dropped >= 1
    gate.set_result(None)
    assert int((tiers.placement.tier == POOL).sum()) == 1
    deadline = time.time() + 2
    while tiers.placement.pool.quarantined and time.time() < deadline:
        time.sleep(0.01)
    tiers.placement.check()


def test_the_runtime_settings(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    assert xt._ws_get("working_set") is False
    assert xt._ws_check("ws_warm", 2.0) and "above ws_hot" in xt._ws_check(
        "ws_warm", 2.0
    )
    assert xt._ws_check("ws_batch", 0) is not None
    assert xt._ws_check("working_set", 1) is not None  # a bool, not a number
    assert xt._ws_check("working_set", True) is None
    xt._ws_apply("working_set", True)
    xt._ws_apply("ws_hot", 2)
    assert tiers._ws.settings.working_set and tiers._ws.settings.ws_hot == 2.0
    from vllm.v1.worker import runtime_control as wrc

    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=None))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    out = wrc.handle(worker, "apply", "tiers.set", {"ws_warm": 0.25})
    assert out["ok"], out
    assert tiers._ws.settings.ws_warm == 0.25
    state = wrc.handle(worker, "state", args={"only": ["tiers"]})["tiers"]
    assert state["working_set"]["settings"]["ws_warm"] == 0.25
    assert "per_layer" not in state["working_set"]
    json.dumps(state)


def test_the_metrics_carry_the_working_set():
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
    totals = dict.fromkeys(
        ["hot", "warm", "cold", "hot_on_card", "warm_or_hot_on_cpu",
         "cold_on_card", "card_short", "rows_short"], 3,
    )  # fmt: skip
    reply = {
        "rank": 1,
        "cpu": {"counters": counters, "cache": {"rows": 4}, "cpu_threads": 8},
        "tiers": {
            "settings": {"host_tier_bound": 22},
            "slots_now": {"card": 1, "pinned": 1, "cpu": 1},
            "working_set": {
                "totals": totals,
                "counters": {
                    "batches": 5,
                    "committed": 4,
                    "dropped": 1,
                    "swaps": {"cpu>card": 1, "pinned>card": 2, "cpu>pinned": 3},
                },
            },
        },
    }
    series = expert_tier_series([reply])["1"]
    assert set(series) == set(EXPERT_TIER_GAUGES)
    assert series["stage_seconds"] == 0.5 and series["ws_swaps_cpu_pinned"] == 3


# -- prefill staging --------------------------------------------------------------


def test_a_large_call_stages_its_busy_cpu_experts_to_the_card(
    monkeypatch, tmp_path, op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=3)
    tiers = env.tiers[1]
    rt = tiers.cpu
    ids = torch.tensor([[6, 0], [6, 1], [6, 7], [5, 6]], dtype=torch.int32)
    w = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    pending = cx.Pending(
        tiers,
        4,
        SimpleNamespace(synchronize=lambda: None),
        rt._buffers_for(4, 2, HIDDEN),
    )
    pending.buffers.ids[:4].copy_(ids)
    pending.buffers.weights[:4].copy_(w)
    pending.buffers.x[:4].copy_(x)
    impl = rt.stage(pending, torch.device("cpu"))
    assert impl is not None
    rows = xt._STAGE_ROWS
    # expert 6 (4 calls) is staged in row 0: its map entry and its bytes
    assert rows.map.tolist()[6] == 0 and sum(v >= 0 for v in rows.map.tolist()) == 1
    for name, unit in expert_in_store(env, 1, 6).items():
        assert torch.equal(rows.rows_u8(name)[0], unit.reshape(-1).view(torch.uint8))
    assert impl.expert_map is rows.map and impl.num_experts == rows.num_rows
    assert rows.num_rows == tiers.num_host == 3  # one per CPU slot by default
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


def test_staging_off_or_a_small_call_changes_nothing(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    rt = tiers.cpu
    pending = cx.Pending(
        tiers,
        1,
        SimpleNamespace(synchronize=lambda: None),
        rt._buffers_for(1, 2, HIDDEN),
    )
    pending.buffers.ids[:1].copy_(torch.tensor([[6, 7]], dtype=torch.int32))
    assert rt.stage(pending, torch.device("cpu")) is None
    assert not pending.waited and pending.selection is None  # did not even wait
    xt._ws_apply("stage_min_calls", 2)
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
        def post(self, tiers, h, ids, w):
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


# -- the host cache's copies out ----------------------------------------------------


def test_a_row_being_copied_out_is_not_evicted():
    from .test_xpu_cpu_experts import _cache

    c = _cache(rows=1)
    c.release(c.acquire([(0, 1)]))
    gate = Future()
    ex = ThreadPoolExecutor(1)
    ex.submit(gate.result)  # the copy waits behind this
    dests = {n: torch.zeros_like(t) for n, t in c.dests(c.row_of((0, 1))).items()}
    f = c.copy_out_async((0, 1), dests, ex)
    assert f is not None and c.copy_out_async((0, 9), dests, ex) is None
    with pytest.raises(cx.CpuExpertError, match="raise host_cache_gib"):
        c.acquire([(0, 2)])  # the only row is held by the copy
    assert c.resize_refusal() is not None
    gate.set_result(None)
    f.result(2)
    deadline = time.time() + 2
    while c._held and time.time() < deadline:
        time.sleep(0.01)
    assert dests["w13"][:2].tolist() == [0, 1]
    c.release(c.acquire([(0, 2)]))
    ex.shutdown()


# -- the seed --------------------------------------------------------------------------


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


def test_the_seed_is_the_store_config_of_the_movers():
    from vllm.distributed.eplb.eplb_state import _tier_seed

    cfg = SimpleNamespace(tier_seed="/x.json")
    mover = SimpleNamespace(
        store_binding=SimpleNamespace(session=SimpleNamespace(config=cfg))
    )
    assert _tier_seed(SimpleNamespace(movers=[mover])) == "/x.json"
    assert _tier_seed(SimpleNamespace(movers=None)) is None
    cfg.tier_seed = None
    assert _tier_seed(SimpleNamespace(movers=[mover])) is None


def test_a_fill_from_seed_loads_starts_the_temperature(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    load = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    tiers.fill_to(np.arange(8), load, 1.1)
    # calls per token: load / total x top-k; temperature: x 8 experts / top-k
    assert tiers._heat == pytest.approx(load / load.sum() * 2)
    assert tiers.temperature() == pytest.approx(load / load.sum() * 8)
    assert tiers._cls[7] == wsm.HOT and tiers._cls[0] == wsm.COLD
    assert np.flatnonzero(tiers.placement.tier == FAST).tolist() == [5, 6, 7]


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


# -- with every switch unset ---------------------------------------------------------


def test_unset_eplb_moves_still_choose_tiers_and_hashes_are_equal(
    monkeypatch, tmp_path, op
):
    from vllm.config.expert_store import ExpertStoreConfig

    base = ExpertStoreConfig(dirs=["/x"])
    on = ExpertStoreConfig(
        dirs=["/x"], tier_seed="/s", working_set=True, stage_min_calls=8,
        ws_hot=3.0, stage_rows=64,
    )  # fmt: skip
    assert base.compile_factors() == on.compile_factors()
    env = _env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    calls = []
    real = tiers.placement.plan

    def spy(*a, **kw):
        calls.append(kw.get("tier_moves", True))
        return real(*a, **kw)

    monkeypatch.setattr(tiers.placement, "plan", spy)
    load = np.array([9, 9, 9, 5, 5, 0, 8, 0], dtype=float)
    tiers.commit(tiers.prepare(np.arange(8), np.arange(8), load, 1.1))
    assert calls == [True]  # EPLB chose the tiers, as on the base
    assert tiers.placement.tier[6] == POOL  # the hot CPU slot got a pinned row
