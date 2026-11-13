# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dynamic expert features together, on CPU: experts computed on the CPU
(--cpu-offload-pinned-gb) with the placement-independent sums
(VLLM_XPU_MOE_COMBINE rank/exact) and the placement trace, and the CPU tier
changed while serving through the runtime control surface (the pinned
bound, threads, promotion settings, the host cache), with its readout.

The host op is the torch reference of the card's arithmetic unless a build
of it is loaded (see test_xpu_cpu_experts.py); the Triton sums run under the
interpreter in a subprocess."""

import json
import os
import subprocess
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.runtime_control as wrc
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import HOST, POOL

from .test_xpu_cpu_experts import CPU_CFG, REAL_OP, _reference_op, _settle, _units
from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def op(monkeypatch):
    """The host op, real when loaded, else the torch reference."""
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield REAL_OP
    cx.reset_runtime()


def _host_env(monkeypatch, tmp_path, pinned=2, **extra):
    run(monkeypatch, tmp_path, store=True)
    # the layers of the start that built the store are not this engine's
    xt._LIVE_TIERS.clear()
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=pinned, store_extra=cfg)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    # The CPU tier registers its runtime setting once per process, and the
    # readout lists the layers of this process: this test's only.
    monkeypatch.setattr(xt, "_RUNTIME_SETTINGS_REGISTERED", False)
    monkeypatch.setattr(wrc, "_TIER_SETTINGS", {})
    monkeypatch.setattr(xt, "_LIVE_TIERS", weakref.WeakSet())
    yield
    tstore.reset_sessions()


# -- the combine modes with the CPU's experts ----------------------------------

_COMBINE_WITH_CPU = r"""
import torch
from types import SimpleNamespace
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe.experts import xpu_moe

bf = torch.bfloat16
T, K, H, E = 6, 4, 64, 16
g = torch.Generator().manual_seed(1)
expert_w = (torch.randn(E, H, generator=g)).to(bf)
x = (torch.randn(T, H, generator=g)).to(bf)
topk_ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).int()
topk_w = torch.softmax(torch.randn(T, K, generator=g), dim=-1)


def row(t, e):
    return (x[t].float() * expert_w[e].float()).to(bf)


class Impl:
    # a fused MoE over the experts its map holds (as in test_xpu_moe_combine)
    def __init__(self, emap):
        self.emap = emap

    def apply(self, output, hidden_states, topk_weights, topk_ids, output_hook=None):
        local = self.emap[topk_ids.long()]
        pairs = (local >= 0).nonzero().tolist()
        pairs.sort(key=lambda p: (int(local[p[0], p[1]]), p[0]))
        rows = torch.zeros(max(len(pairs), 1), H, dtype=bf)
        rmap = torch.full((T, K), -1, dtype=torch.int32)
        for r, (t, k) in enumerate(pairs):
            rows[r] = row(t, int(topk_ids[t, k]))
            rmap[t, k] = r
        if output_hook is not None:
            output_hook(rows, None, rmap)
        acc = torch.zeros(T, H)
        for k in range(K):
            m = rmap[:, k] >= 0
            v = rows[rmap[:, k].clamp(min=0).long()].float() * m[:, None]
            acc = acc + v * topk_weights[:, k : k + 1]
        output.copy_(acc.to(bf))


class Cpu:
    # the CPU tier: the experts in cpu_set; the per-token sum (split) and the
    # per-pair rows (rank, exact), in a shuffled order
    def __init__(self, cpu_set):
        self.cpu_set = cpu_set
        self.posted = 0

    def post(self, tiers, h, ids, w):
        self.posted += 1
        return (ids, w)

    def finish(self, pending, device):
        ids, w = pending
        acc = torch.zeros(T, H)
        hit = False
        for t in range(T):
            for k in range(K):
                if int(ids[t, k]) in self.cpu_set:
                    acc[t] += row(t, int(ids[t, k])).float() * float(w[t, k])
                    hit = True
        return acc.to(bf) if hit else None

    def finish_pairs(self, pending, device):
        ids, _ = pending
        pairs = [(t, k) for t in range(T) for k in range(K)
                 if int(ids[t, k]) in self.cpu_set]
        if not pairs:
            return None
        pairs = pairs[::-1]
        rows = torch.stack([row(t, int(ids[t, k])) for t, k in pairs])
        rmap = torch.full((T, K), -1, dtype=torch.int32)
        for r, (t, k) in enumerate(pairs):
            rmap[t, k] = r
        return cx.HostPairs(rows, rmap)


def run(mode, fast_set, cpu_set):
    fast_map = torch.full((E,), -1, dtype=torch.int32)
    slow_map = torch.full((E,), -1, dtype=torch.int32)
    nf = ns = 0
    for e in range(E):
        if e in fast_set:
            fast_map[e] = nf; nf += 1
        elif e not in cpu_set:
            slow_map[e] = ns; ns += 1
    cpu = Cpu(cpu_set) if cpu_set else None
    experts = object.__new__(xpu_moe.XPUExperts)
    experts.tiers = SimpleNamespace(num_global=E, cpu=cpu)
    experts.impl_apply_kwargs = {}
    experts.combine_mode = mode
    experts.placement_trace = None
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    experts._tier_impls = lambda w1, w2, topk: (Impl(fast_map), Impl(slow_map))
    out = torch.empty(T, H, dtype=bf)
    experts._apply_tiered(out, x, None, None, topk_w, topk_ids)
    assert cpu is None or cpu.posted == 1
    return out


xpu_moe.staged_impl = lambda impl, n, slot: impl
xpu_moe.fused_moe_has_output_hook = lambda: True
placements = [
    (set(range(0, 6)), set(range(10, 16))),
    (set(range(8, 14)), set(range(0, 5))),
    ({0, 3, 5, 6, 9, 12}, {1, 2, 4, 7}),
    (set(), set(range(16))),          # every expert on the CPU
    (set(range(0, 8)), set()),        # none on the CPU
]
ref = None
for mode in ("rank", "exact"):
    outs = [run(mode, f, c) for f, c in placements]
    assert all(torch.equal(o, outs[0]) for o in outs), mode
    ref = outs[0] if ref is None else ref
    assert torch.equal(outs[0], ref), "rank and exact differ on one rank"
splits = [run("split", f, c) for f, c in placements]
assert any(not torch.equal(s, splits[0]) for s in splits), "split did not move"

# the three-source sum equals the sum of the same pairs from two sources
part = torch.randint(0, 3, (T, K), generator=torch.Generator().manual_seed(5))
def src(which):
    idx = (part == which).nonzero().tolist()
    rows = torch.zeros(max(len(idx), 1), H, dtype=bf)
    rmap = torch.full((T, K), -1, dtype=torch.int32)
    for r, (t, k) in enumerate(idx):
        rows[r] = row(t, int(topk_ids[t, k]))
        rmap[t, k] = r
    return rows, rmap
three = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(three, topk_w, [src(0), src(1), src(2)])
part = torch.where(part == 2, torch.ones_like(part), part)
two = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(two, topk_w, [src(0), src(1)])
assert torch.equal(three, two)
picked = torch.empty(T * K, H, dtype=bf)
part = torch.randint(0, 3, (T, K), generator=torch.Generator().manual_seed(5))
xpu_moe.pick_rows(picked, [src(0), src(1), src(2)])
for t in range(T):
    for k in range(K):
        assert torch.equal(picked[t * K + k], row(t, int(topk_ids[t, k])))
print("CPU-COMBINE-OK")
"""


def test_rank_and_exact_include_the_cpu_rows_interpreted():
    """Rank and exact sum the CPU's pairs with the card's in top-k order: the
    same bits whichever experts are on the card, in pinned rows or on the
    CPU (given the same per-expert rows); split does not. The three-source
    kernels equal the two-source ones on the same pairs."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", _COMBINE_WITH_CPU],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "CPU-COMBINE-OK" in result.stdout, result.stderr[
        -3000:
    ]


def test_cpu_pairs_are_the_rows_of_the_per_token_sum(monkeypatch, tmp_path, op):
    """The CPU's per-pair rows (combine rank/exact) are each expert's own
    output: weighted and summed in top-k order they give the per-token sum
    the split path computes, to the bit."""
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    summed = torch.empty_like(x)
    assert rt.compute(tiers, ids, w, x, summed)
    rows = torch.empty(6, HIDDEN, dtype=torch.bfloat16)
    pair = rt.compute_pairs(tiers, ids, x, rows)
    assert pair.tolist() == [1, 2, 4, 5]  # (0,1) (1,0) (2,0) (2,1)
    want_rows = [
        cx.reference_int4_experts(
            x[t : t + 1], _units(env, 1, {e}), [(e, 0, 1.0)], INTER, GROUP
        )[0]
        for t, e in ((0, 5), (1, 6), (2, 7), (2, 5))
    ]
    for i, want in enumerate(want_rows):
        if op:
            assert cx.bf16_steps(rows[i : i + 1], want[None]) <= 1.0
        else:
            assert torch.equal(rows[i], want)
    # the per-token sum from the rows, in top-k order, fp32
    acc = torch.zeros(3, HIDDEN)
    for i, p in enumerate(pair.tolist()):
        t, k = divmod(p, 2)
        acc[t] += rows[i].float() * float(w[t, k])
    if not op:
        assert torch.equal(acc.bfloat16(), summed)
    # finish_pairs: the rows and their [tokens, top_k] map, on the device
    pending = cx.Pending(
        tiers,
        3,
        SimpleNamespace(synchronize=lambda: None),
        rt._buffers_for(3, 2, HIDDEN),
    )
    pending.buffers.ids[:3].copy_(ids)
    pending.buffers.weights[:3].copy_(w)
    pending.buffers.x[:3].copy_(x)
    host = rt.finish_pairs(pending, torch.device("cpu"))
    assert host.row_map.tolist() == [[-1, 0], [1, -1], [2, 3]]
    assert torch.equal(host.rows, rows[:4])
    # a routing with nothing on the CPU
    pending.buffers.ids[:3].copy_(ids.clamp(max=4))
    assert rt.finish_pairs(pending, torch.device("cpu")) is None


def test_the_placement_trace_names_the_cpu_slots(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    prefix = str(tmp_path / "pt")
    open(prefix + ".on", "w").close()
    trace = xpu_moe.PlacementTrace(prefix, ep_rank=0)
    trace.record(
        tiers, torch.tensor([[0, 3, 6]], dtype=torch.int32), torch.ones(1, 3) / 3
    )
    with open(prefix + ".rank0.jsonl") as f:
        entry = json.loads(f.read().splitlines()[0])
    p = tiers.placement
    assert entry["where"] == [[f"F{p.row[0]}", f"S{p.row[3]}", "C"]]
    assert entry["logical"] == [[0, 3, 6]]


# -- the pinned bound while serving (tiers.set host_tier_bound) ----------------


def _layer_load(tiers, hot):
    """EPLB's view of the slots' load, hottest first in ``hot``."""
    load = np.zeros(tiers.num_local)
    for i, s in enumerate(hot):
        load[s] = 100 - i
    return load


def test_lowering_and_raising_the_pinned_bound(monkeypatch, tmp_path, op):
    """The bound is a capacity; EPLB's tier pass (here the layer's own
    prepare with the placement unchanged, as EPLB's worker calls it) moves
    the slots to it from the window: lowering reads nothing, raising reads
    the hottest CPU slots (from the host cache where it holds them)."""
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    p = tiers.placement
    # card 5 6 7, pinned 4 3, CPU 2 1 0 (hottest first)
    load = _layer_load(tiers, [5, 6, 7, 4, 3, 2, 1, 0])
    tiers.fill_to(np.arange(8), load, 1.1)
    assert np.flatnonzero(p.tier == POOL).tolist() == [3, 4]
    assert tiers.pinned_bound_range() == (1, 5)  # 8 rows, 2 spares; 5 off the card
    stats = tiers.set_pinned_bound(1)
    assert stats["bound"] == 1 and stats["pinned_now"] == 2
    assert tiers.tier_pass_wanted and np.flatnonzero(p.tier == POOL).tolist() == [3, 4]
    # EPLB's tier pass: the colder pinned slot (3) goes to the CPU, no read
    move = tiers.prepare(np.arange(8), np.arange(8), load, 1.1)
    assert not tiers.tier_pass_wanted and move.stats["units"] == 0
    tiers.commit(move)
    assert np.flatnonzero(p.tier == POOL).tolist() == [4] and p.tier[3] == HOST
    assert p.num_pinned == 1 and p.num_host == 4
    assert tiers.slow_map[3] == -1 and tiers._tier_of_global[3] == HOST
    x = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(4)).bfloat16()
    out = torch.empty_like(x)
    assert tiers.cpu.compute(
        tiers, torch.tensor([[3, 5]], dtype=torch.int32), torch.ones(1, 2), x, out
    )
    want = cx.reference_int4_experts(
        x, _units(env, 0, {3}), [(3, 0, 1.0)], INTER, GROUP
    )
    assert cx.bf16_steps(out, want) <= (1.0 if op else 0.0)
    # raise to 4: the pass reads the three hottest CPU slots (3, 2, 1), two
    # per pass (the spare rows), so two passes
    _settle(tiers.cpu.cache)
    tiers.set_pinned_bound(4)
    for _ in range(2):
        tiers.commit(tiers.prepare(np.arange(8), np.arange(8), load, 1.1))
    assert sorted(np.flatnonzero(p.tier == POOL).tolist()) == [1, 2, 3, 4]
    assert p.tier[0] == HOST and p.num_pinned == 4
    for s in (1, 2, 3, 4):
        assert tiers.slow_map[s] == p.row[s]
        for name, t in tiers.slow.items():
            want = expert_in_store(env, 0, s)[name].reshape(-1).view(torch.uint8)
            assert torch.equal(t[p.row[s]].reshape(-1).view(torch.uint8), want)
    summary = tiers.scrub(np.arange(8), full=True)
    assert summary["mismatches"] == 0 and summary["maps_equal"]
    assert p.pool.free_count >= p.spare
    # the next rearrangement plans with the bound: slot 0 is hot now
    load = np.array([90, 1, 1, 1, 1, 100, 100, 100], dtype=float)
    tiers.commit(tiers.prepare(np.arange(8), np.arange(8), load, 1.1))
    assert int((p.tier == POOL).sum()) == 4 and p.tier[0] == POOL
    with pytest.raises(ValueError, match="1..5"):
        tiers.set_pinned_bound(6)


def test_the_bound_waits_for_a_prepared_eplb_move(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    tiers._eplb_pending = True
    assert "EPLB move is prepared" in xt.check_host_tier_bound(1)
    with pytest.raises(RuntimeError, match="EPLB move is prepared"):
        tiers.set_pinned_bound(1)
    assert tiers.placement.num_pinned == 2 and not tiers.tier_pass_wanted
    tiers._eplb_pending = False


def test_the_bound_over_the_control_surface(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=None))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 0)
    state = wrc.handle(worker, "state", args={"only": ["tiers", "cpu"]})
    tiers_state = state["tiers"]
    assert tiers_state["resizable"] and tiers_state["resize_blocked"] is None
    assert tiers_state["settings"] == {"host_tier_bound": 2}
    assert tiers_state["pinned_slots_per_layer"] == [2]
    assert tiers_state["cpu_slots_per_layer"] == [3]
    assert tiers_state["slots_now"] == {"card": 6, "pinned": 4, "cpu": 6}
    refusal = wrc.handle(worker, "check", "tiers.set", {"host_tier_bound": 9})
    assert "outside 1..5" in refusal["refusal"]
    reply = wrc.handle(worker, "apply", "tiers.set", {"host_tier_bound": 1})
    assert reply["ok"], reply
    assert reply["changes"] == {"host_tier_bound": [2, 1]}
    assert all(t.placement.num_pinned == 1 for t in env.tiers)
    # the slots move in EPLB's next tier pass, not here
    assert reply["result"]["tiers"]["slots_now"]["pinned"] == 4
    assert all(t.tier_pass_wanted for t in env.tiers)
    undone = wrc.handle(worker, "restore", "tiers.set")
    assert undone["ok"]
    assert all(t.placement.num_pinned == 2 for t in env.tiers)
    # refused while EPLB rearranges
    worker.model_runner.eplb_state = SimpleNamespace(
        model_states={"m": SimpleNamespace(rebalanced=True)}
    )
    refusal = wrc.handle(worker, "check", "tiers.set", {"host_tier_bound": 1})
    assert "in flight" in refusal["refusal"]


def test_without_cpu_experts_the_bound_is_refused_with_the_reason(
    monkeypatch, tmp_path
):
    run(monkeypatch, tmp_path, store=True)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=None))
    refusal = wrc.handle(worker, "check", "tiers.set", {"host_tier_bound": 1})
    assert "--cpu-offload-pinned-gb" in refusal["refusal"]
    assert "outside the captured graphs" in refusal["refusal"]


# -- the CPU tier's settings and counters over the control surface -------------


def test_cpu_settings_change_and_go_back(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(eplb_state=None))
    rt = env.tiers[0].cpu
    _settle(rt.cache)
    for args, refusal in [
        ({"cpu_isa": "avx2"}, "cannot change while serving"),
        ({"host_promote_rows": 2}, "sized at load"),
        ({"host_promote_misses": 2}, "EPLB's tier pass"),
        ({"cpu_threads": 0}, "integer >= 1"),
        ({"host_cache_gib": 0}, "number > 0"),
        ({"warp": 1}, "unknown CPU tier setting"),
    ]:
        assert refusal in wrc.handle(worker, "check", "cpu.set", args)["refusal"]
    held = len(rt.cache._rows)
    assert held > 0
    reply = wrc.handle(
        worker,
        "apply",
        "cpu.set",
        {
            "cpu_threads": 3,
            "stage_min_calls": 12,
            "host_cache_gib": rt.cache.layout.row_bytes * 2 / 2**30,
        },
    )
    assert reply["ok"], reply
    assert reply["changes"]["cpu_threads"] == [2, 3]
    assert reply["changes"]["stage_min_calls"] == [0, 12]
    assert rt.threads == 3 and rt.config.stage_min_calls == 12
    assert rt.cache.num_rows == 2 and len(rt.cache._rows) == 2
    assert rt.cache.evicted >= held - 2
    # the experts kept are the units of the store, in their new rows
    for key, row in rt.cache._rows.items():
        assert tstore_holds(env, key, rt.cache.dests(row))
    # the host cache still serves
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    tiers = env.tiers[0]
    host = np.flatnonzero(tiers.placement.tier == HOST).tolist()
    assert rt.compute(
        tiers,
        torch.tensor([[host[0], host[1]]], dtype=torch.int32),
        torch.ones(1, 2),
        x,
        torch.empty_like(x),
    )
    assert wrc.handle(worker, "restore", "cpu.set")["ok"]
    assert rt.threads == 2 and rt.config.stage_min_calls == 0
    assert rt.cache.num_rows == rt.cache.rows_for_gib(CPU_CFG["host_cache_gib"])
    # a cache smaller than what one layer call needed is refused
    rt.stats.max_experts = 5
    refusal = wrc.handle(
        worker,
        "check",
        "cpu.set",
        {"host_cache_gib": rt.cache.layout.row_bytes * 2 / 2**30},
    )["refusal"]
    assert "already needed 5" in refusal
    # stats: the line now, the counters back
    reply = wrc.handle(worker, "apply", "cpu.stats", {})
    assert reply["ok"] and reply["result"]["cpu"]["counters"]["cpu_calls"] >= 2


def tstore_holds(env, key, dests):
    session = env.tiers[0].store_binding.session
    return session.store.holds_unit(key, dests)


def test_the_store_stats_interval_paces_the_cpu_line(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    rt = env.tiers[0].cpu
    lines = []
    monkeypatch.setattr(rt, "log_stats", lambda: lines.append(1))
    rt._last_stats = 0.0
    rt._maybe_log()
    assert lines == []  # stats_interval_s 0: off
    rt.config.stats_interval_s = 1e-6  # what store.set changes
    rt._maybe_log()
    assert lines == [1]


def test_the_cpu_readout_and_its_metric_series(monkeypatch, tmp_path, op):
    from vllm.v1.engine.runtime_control import (
        EXPERT_TIER_GAUGES,
        expert_tier_series,
    )

    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    tiers.cpu.compute(
        tiers,
        torch.tensor([[0, 6], [7, 3]], dtype=torch.int32),
        torch.ones(2, 2),
        x,
        torch.empty_like(x),
    )
    worker = SimpleNamespace(rank=1, model_runner=SimpleNamespace(eplb_state=None))
    monkeypatch.setattr(wrc, "_ep_rank", lambda w: 1)
    reply = wrc.handle(worker, "state", args={"only": ["cpu", "tiers"]})
    json.dumps(reply)
    cpu = reply["cpu"]
    assert cpu["counters"]["routed_cpu"] == 2 and cpu["counters"]["routed_card"] == 1
    assert cpu["settings"]["cpu_threads"] == 2 and cpu["cache"]["rows"] >= 1
    series = expert_tier_series([reply, {"rank": 0, "cpu": None}])
    assert list(series) == ["1"]
    # without EPLB in this harness there are no classes by its window
    assert set(series["1"]) == {
        k for k in EXPERT_TIER_GAUGES if not k.startswith("tier_")
    }
    assert series["1"]["routed_cpu"] == 2.0 and series["1"]["pinned_bound"] == 2.0
    assert series["1"]["slots_cpu"] == 6.0
