# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experts computed on the CPU summed as the card-only layer sums them.

1. With the CPU tier on and VLLM_XPU_MOE_COMBINE unset, a token's routed sum
   is one fp32 sum over its pairs in top-k order (the "rank" sum), whatever
   tier (card, pinned rows, staging rows, CPU) holds which expert; setting
   the variable keeps that mode (split: the per-tier sums); without the CPU
   tier nothing changes.
2. The CPU computes each GEMM with the arithmetic of the card kernel that the
   card would run for the same layer call (the int4 GEMV or the grouped
   GEMM), when the host op has the ``arith`` argument.

The Triton sums run under the interpreter in a subprocess."""

import os
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx

from .test_xpu_cpu_experts import CPU_CFG, REAL_OP, _units
from .xpu_store_harness import GROUP, HIDDEN, INTER, run

pytestmark = pytest.mark.skip_global_cleanup

_DEFAULT_WITH_CPU = r"""
import numpy as np
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
    # a fused MoE over the experts its map holds; records whether it was
    # asked for its per-pair rows
    def __init__(self, emap, log):
        self.emap = emap
        self.log = log

    def apply(self, output, hidden_states, topk_weights, topk_ids, output_hook=None):
        self.log.append(output_hook is not None)
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


def emap_of(ids):
    m = torch.full((E,), -1, dtype=torch.int32)
    for i, e in enumerate(sorted(ids)):
        m[e] = i
    return m


class Cpu:
    # the CPU tier over cpu_set; stage() sends stage_set (a subset) to the
    # card through an Impl over the staging rows, as the runtime does
    def __init__(self, cpu_set, stage_set, log):
        self.cpu_set, self.stage_set, self.log = cpu_set, stage_set, log
        self.finished = []

    def post(self, tiers, h, ids, w, weights=True):
        return SimpleNamespace(ids=ids, w=w, staged=None)

    def stage(self, pending, device):
        if not self.stage_set:
            return None
        pending.staged = np.array(sorted(self.stage_set))
        return Impl(emap_of(self.stage_set), self.log)

    def held(self, pending, e):
        staged = pending.staged
        return e in self.cpu_set and (staged is None or e not in staged)

    def finish(self, pending, device):
        self.finished.append("finish")
        acc = torch.zeros(T, H)
        hit = False
        for t in range(T):
            for k in range(K):
                e = int(pending.ids[t, k])
                if self.held(pending, e):
                    acc[t] += row(t, e).float() * float(pending.w[t, k])
                    hit = True
        return acc.to(bf) if hit else None

    def finish_pairs(self, pending, device):
        self.finished.append("pairs")
        pairs = [(t, k) for t in range(T) for k in range(K)
                 if self.held(pending, int(pending.ids[t, k]))]
        if not pairs:
            return None
        pairs = pairs[::-1]
        rows = torch.stack([row(t, int(pending.ids[t, k])) for t, k in pairs])
        rmap = torch.full((T, K), -1, dtype=torch.int32)
        for r, (t, k) in enumerate(pairs):
            rmap[t, k] = r
        return cx.HostPairs(rows, rmap)


def run(mode, explicit, fast_set, cpu_set, stage_set=frozenset()):
    log = []
    slow_set = set(range(E)) - set(fast_set) - set(cpu_set)
    cpu = Cpu(set(cpu_set), set(stage_set), log) if cpu_set else None
    experts = object.__new__(xpu_moe.XPUExperts)
    experts.tiers = SimpleNamespace(num_global=E, cpu=cpu)
    experts.impl_apply_kwargs = {}
    experts.combine_mode = mode
    experts.combine_explicit = explicit
    experts.placement_trace = None
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    impls = (Impl(emap_of(fast_set), log), Impl(emap_of(slow_set), log))
    experts._tier_impls = lambda w1, w2, topk: impls
    out = torch.empty(T, H, dtype=bf)
    experts._apply_tiered(out, x, None, None, topk_w, topk_ids)
    return out, log, (cpu.finished if cpu else [])


xpu_moe.staged_impl = lambda impl, n, slot: impl
xpu_moe.fused_moe_has_output_hook = lambda: True
placements = [
    (set(range(0, 6)), set(range(10, 16)), set()),
    (set(range(8, 14)), set(range(0, 5)), {1, 3}),       # two staged
    ({0, 3, 5, 6, 9, 12}, {1, 2, 4, 7}, {2, 4, 7}),
    (set(), set(range(16)), set(range(0, 16, 2))),        # every expert off the card
    (set(range(0, 4)), set(range(4, 16)), set()),
]
# 1. unset with the CPU tier: one sum per rank, the same bits for every
#    placement, equal to the explicit "rank" sum; the per-pair rows are asked
#    for on every call (card, pinned, staging) and the CPU gives pairs.
outs = []
for f, c, s in placements:
    out, log, finished = run("split", False, f, c, s)
    assert all(log), log
    assert finished == ["pairs"], finished
    outs.append(out)
assert all(torch.equal(o, outs[0]) for o in outs), "the default moved with the tiers"
rank = [run("rank", True, f, c, s)[0] for f, c, s in placements]
assert all(torch.equal(o, outs[0]) for o in rank)
# 2. explicit split keeps the per-tier sums (they move with the placement)
splits = []
for f, c, s in placements:
    out, log, finished = run("split", True, f, c, s)
    assert not any(log) and finished == ["finish"]
    splits.append(out)
assert any(not torch.equal(o, splits[0]) for o in splits), "split did not move"
# 3. without the CPU tier the unset default is the per-tier sum, unchanged
out_unset, log, _ = run("split", False, set(range(8)), set())
out_split, _, _ = run("split", True, set(range(8)), set())
assert not any(log) and torch.equal(out_unset, out_split)
# 4. four sources: the same bits as the same pairs from two
part = torch.randint(0, 4, (T, K), generator=torch.Generator().manual_seed(5))
def src(which, part):
    idx = (part == which).nonzero().tolist()
    rows = torch.zeros(max(len(idx), 1), H, dtype=bf)
    rmap = torch.full((T, K), -1, dtype=torch.int32)
    for r, (t, k) in enumerate(idx):
        rows[r] = row(t, int(topk_ids[t, k]))
        rmap[t, k] = r
    return rows, rmap
four = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(four, topk_w, [src(i, part) for i in range(4)])
two_part = part % 2
two = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(two, topk_w, [src(i, two_part) for i in range(2)])
assert torch.equal(four, two)
picked = torch.empty(T * K, H, dtype=bf)
xpu_moe.pick_rows(picked, [src(i, part) for i in range(4)])
for t in range(T):
    for k in range(K):
        assert torch.equal(picked[t * K + k], row(t, int(topk_ids[t, k])))
# a pair held by none of four sources adds an exact zero
none = torch.where(part == 3, torch.full_like(part, 9), part)
three = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(three, topk_w, [src(i, none) for i in range(4)])
held = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(held, topk_w, [src(i, none) for i in range(3)])
assert torch.equal(three, held)
print("DEFAULT-CPU-OK")
"""


def test_default_with_cpu_experts_sums_once_per_rank_interpreted():
    """VLLM_XPU_MOE_COMBINE unset with experts on the CPU: one sum per rank
    in top-k order, the same bits for every placement over card, pinned
    rows, staging rows and CPU; explicit split keeps the per-tier sums; no
    CPU tier, no change. The sums take a fourth source (the staging rows)."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    env.pop("VLLM_XPU_MOE_COMBINE", None)
    result = subprocess.run(
        [sys.executable, "-c", _DEFAULT_WITH_CPU],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "DEFAULT-CPU-OK" in result.stdout, result.stderr[
        -3000:
    ]


def test_combine_explicit_follows_the_environment(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    experts = object.__new__(xpu_moe.XPUExperts)
    experts.impl_apply_kwargs = {}
    experts.moe_config = SimpleNamespace(dp_size=1, ep_size=2, ep_rank=0)
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    cpu = object()
    for value, explicit, want_cpu in (
        (None, False, "rank"),
        ("split", True, "split"),
        ("rank", True, "rank"),
        ("exact", True, "exact"),
    ):
        if value is None:
            monkeypatch.delenv("VLLM_XPU_MOE_COMBINE", raising=False)
        else:
            monkeypatch.setenv("VLLM_XPU_MOE_COMBINE", value)
        import vllm.envs as envs

        experts.combine_mode = envs.VLLM_XPU_MOE_COMBINE
        experts.combine_explicit = envs.is_set("VLLM_XPU_MOE_COMBINE")
        assert experts.combine_explicit is explicit
        assert experts._tiered_combine_mode(cpu) == want_cpu
        assert experts._tiered_combine_mode(None) == (value or "split")
    # unset, a kernels build without output_hook: the per-tier sums, no error
    monkeypatch.delenv("VLLM_XPU_MOE_COMBINE", raising=False)
    experts.combine_mode, experts.combine_explicit = "split", False
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: False)
    assert experts._tiered_combine_mode(cpu) == "split"


def _fake_kernels(monkeypatch, max_rows=8, env="TEST_INT4_GEMV"):
    pkg = types.ModuleType("vllm_xpu_kernels")
    fmi = types.ModuleType("vllm_xpu_kernels.fused_moe_interface")
    fmi.INT4_GEMV_MAX_ROWS = max_rows
    fmi.INT4_GEMV_ENV = env
    pkg.fused_moe_interface = fmi
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface", fmi)


def test_card_arith_is_the_kernel_the_card_takes(monkeypatch):
    _fake_kernels(monkeypatch)
    monkeypatch.delenv("TEST_INT4_GEMV", raising=False)
    # QFN: top-10, so every layer call has more than 8 rows: grouped, both
    assert cx.card_arith(10, 2560, 640, 128) == 3
    assert cx.card_arith(80, 2560, 640, 128) == 3
    # a top-8 model at one token: GEMM1 (K 2560) on the GEMV, GEMM2 (K 640,
    # not a multiple of 512) on the grouped GEMM
    assert cx.card_arith(8, 2560, 640, 128) == 2
    assert cx.card_arith(8, 2048, 1024, 128) == 0
    assert cx.card_arith(8, 2048, 1024, 16) == 3  # group not a multiple of 32
    monkeypatch.setenv("TEST_INT4_GEMV", "0")  # the GEMV switched off
    assert cx.card_arith(8, 2048, 1024, 128) == 3
    # kernels without the GEMV: always the grouped GEMM
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface", None)
    assert cx.card_arith(1, 2048, 1024, 128) == 3


def _reference_op_arith(
    out, x, units, cu, ct, cw, inter, group, offsets, threads, isa, arith=0
):
    """The torch reference standing in for an op with the arith argument."""
    names = cx.UNIT_TENSORS
    rows = {}
    for r in set(cu.tolist()):
        rows[r] = {}
        for i, name in enumerate(names):
            end = offsets[i + 1] if i + 1 < len(names) else None
            if end is None:
                end = offsets[i] + x.shape[1] * (inter // group) * 2
            rows[r][name] = units[r, offsets[i] : end]
    calls = list(zip(cu.tolist(), ct.tolist(), cw.tolist()))
    out.copy_(cx.reference_int4_experts(x, rows, calls, inter, group, arith=arith))


@pytest.fixture
def arith_op(monkeypatch):
    """A host op with the arith argument: the real one when loaded and it
    has it, else the torch reference; every op call's arith recorded."""
    seen: list[int] = []
    real = REAL_OP and cx.op_takes_arith()
    if real:
        run_real = cx._op_run

        def run_op(*args):
            seen.append(args[11] if len(args) > 11 else 0)
            run_real(*args)
    else:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "op_takes_arith", lambda: True)

        def run_op(*args):
            seen.append(args[11] if len(args) > 11 else 0)
            _reference_op_arith(*args)

    monkeypatch.setattr(cx, "_op_run", run_op)
    cx.reset_runtime()
    yield seen
    cx.reset_runtime()


def _host_env(monkeypatch, tmp_path, pinned=2, **extra):
    run(monkeypatch, tmp_path, store=True)
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=pinned, store_extra=cfg)


def _pending(rt, tiers, ids, w, x):
    pending = cx.Pending(
        tiers,
        ids.shape[0],
        SimpleNamespace(synchronize=lambda: None),
        rt._buffers_for(ids.shape[0], ids.shape[1], x.shape[1]),
    )
    pending.buffers.ids[: ids.shape[0]].copy_(ids)
    pending.buffers.weights[: ids.shape[0]].copy_(w)
    pending.buffers.x[: ids.shape[0]].copy_(x)
    return pending


def test_the_cpu_takes_the_card_kernels_arithmetic(monkeypatch, tmp_path, arith_op):
    """The self-check runs the grouped GEMM's arithmetic too; a layer call
    hands the op the arith of the card kernel for its rows, and the CPU's
    rows are that arithmetic's (to the bit with the torch reference, within
    the order of fp32 sums with the real op)."""
    _fake_kernels(monkeypatch)
    env = _host_env(monkeypatch, tmp_path)
    assert 3 in arith_op  # the self-check's grouped run
    tiers = env.tiers[1]
    rt = tiers.cpu
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    arith_op.clear()
    pending = _pending(rt, tiers, ids, w, x)
    pending.arith = cx.card_arith(6, HIDDEN, INTER, GROUP)
    assert pending.arith == 3  # this harness's K are not multiples of 512
    host = rt.finish_pairs(pending, torch.device("cpu"))
    assert arith_op == [3]
    for i, (t, e) in enumerate(((0, 5), (1, 6), (2, 7), (2, 5))):
        want = cx.reference_int4_experts(
            x[t : t + 1], _units(env, 1, {e}), [(e, 0, 1.0)], INTER, GROUP, arith=3
        )[0]
        gemv = cx.reference_int4_experts(
            x[t : t + 1], _units(env, 1, {e}), [(e, 0, 1.0)], INTER, GROUP
        )[0]
        got = host.rows[i]
        if REAL_OP:
            assert cx.bf16_steps(got[None], want[None]) <= 1.0
            assert (got != want).float().mean().item() < 0.05
        else:
            assert torch.equal(got, want)
        assert not torch.equal(want, gemv)
    # the per-token sum path (split) takes the same arith
    arith_op.clear()
    out = rt.finish(_pending(rt, tiers, ids, w, x), torch.device("cpu"))
    assert out is not None and arith_op == [0]  # a Pending made by hand: 0
    pending = _pending(rt, tiers, ids, w, x)
    pending.arith = 1
    arith_op.clear()
    rt.finish(pending, torch.device("cpu"))
    assert arith_op == [1]


def test_cpu_arith_gemv_keeps_the_gemv_arithmetic(monkeypatch, tmp_path, arith_op):
    """--expert-store-config cpu_arith gemv: every call the GEMV's arithmetic
    (the op gets no arith), the self-check runs that one only."""
    _fake_kernels(monkeypatch)
    env = _host_env(monkeypatch, tmp_path, cpu_arith="gemv")
    assert arith_op and set(arith_op) == {0}
    tiers = env.tiers[1]
    ids = torch.tensor([[0, 5], [6, 3]], dtype=torch.int32)
    x = torch.randn(2, HIDDEN).bfloat16()
    assert tiers.cpu.post(tiers, x, ids, torch.ones(2, 2)).arith == 0
    # the default, card: the grouped GEMM's for these rows
    monkeypatch.setattr(tiers.cpu.config, "cpu_arith", "card")
    assert tiers.cpu.post(tiers, x, ids, torch.ones(2, 2)).arith == 3


def test_without_the_arith_argument_the_op_call_is_unchanged(monkeypatch, tmp_path):
    """An op without arith (kernels before it) gets the eleven arguments it
    always got, and post() asks for no arith."""
    seen = []
    monkeypatch.setattr(cx, "host_op_available", lambda: True)
    monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
    monkeypatch.setattr(cx, "op_takes_arith", lambda: False)

    def run_op(*args):
        seen.append(len(args))
        _reference_op_arith(*args)

    monkeypatch.setattr(cx, "_op_run", run_op)
    cx.reset_runtime()
    try:
        env = _host_env(monkeypatch, tmp_path)
        assert seen and set(seen) == {11}
        tiers = env.tiers[1]
        ids = torch.tensor([[0, 5], [6, 3]], dtype=torch.int32)
        x = torch.randn(2, HIDDEN).bfloat16()
        pending = tiers.cpu.post(tiers, x, ids, torch.ones(2, 2))
        assert pending.arith == 0
    finally:
        cx.reset_runtime()


def test_finish_pairs_leaves_the_staged_experts_out(monkeypatch, tmp_path, arith_op):
    """Pairs of experts that stage() sent to the card are not computed on the
    CPU in the per-pair path either, and are counted as staged, not CPU."""
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    whole = rt.finish_pairs(_pending(rt, tiers, ids, w, x), torch.device("cpu"))
    assert whole.row_map.tolist() == [[-1, 0], [1, -1], [2, 3]]
    # on a CPU "device" the rows are the runtime's buffer itself: keep a copy
    whole_rows = whole.rows.clone()
    before = dict(rt.stats.per_tier)
    pending = _pending(rt, tiers, ids, w, x)
    pending.staged = np.array([5])
    part = rt.finish_pairs(pending, torch.device("cpu"))
    assert part.row_map.tolist() == [[-1, -1], [0, -1], [1, -1]]
    assert torch.equal(part.rows[0], whole_rows[1])
    assert torch.equal(part.rows[1], whole_rows[2])
    assert rt.stats.per_tier["cpu"] - before["cpu"] == 2
    pending = _pending(rt, tiers, ids, w, x)
    pending.staged = np.array([5, 6, 7])
    assert rt.finish_pairs(pending, torch.device("cpu")) is None


def test_reference_arith_zero_is_the_previous_reference():
    """reference_int4_experts without arith is the GEMV arithmetic it always
    was (the self-check and the existing tests depend on it)."""
    gen = torch.Generator().manual_seed(11)
    hidden, inter, group = 256, 128, 64
    u = {
        "w13": torch.randint(0, 256, (2 * inter * hidden // 2,), dtype=torch.uint8),
        "w2": torch.randint(0, 256, (hidden * inter // 2,), dtype=torch.uint8),
        "w13_scales": (torch.rand(2 * inter * hidden // group, generator=gen) * 0.01)
        .bfloat16()
        .view(torch.uint8),
        "w2_scales": (torch.rand(hidden * inter // group, generator=gen) * 0.01)
        .bfloat16()
        .view(torch.uint8),
    }
    x = torch.randn(2, hidden, generator=gen).bfloat16()
    calls = [(0, 0, 0.5), (0, 1, 1.0)]
    a = cx.reference_int4_experts(x, {0: u}, calls, inter, group)
    b = cx.reference_int4_experts(x, {0: u}, calls, inter, group, arith=0)
    c = cx.reference_int4_experts(x, {0: u}, calls, inter, group, arith=3)
    assert torch.equal(a, b) and not torch.equal(a, c)
