# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU part on the host thread (VLLM_CPU_EXPERTS_ASYNC=spin) together with
staging, into device staging rows filled by DMA straight from the host cache
(stage_device_rows). A layer call that stages is staged on the model's thread
after the host thread drained; the staged copies and the card's call over the
rows are enqueued before the card's wait for the host word; the host thread
computes only the pairs left to the CPU; the cache rows a DMA reads stay held
until it is known done. The layer's per-pair sum is the same bits with both
switches on, either one, or neither."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_critical_path import _torch_sum_topk
from .test_xpu_cpu_experts import _reference_op
from .test_xpu_stage_device import _direct
from .test_xpu_tier_classes import _env
from .xpu_store_harness import HIDDEN

pytestmark = pytest.mark.skip_global_cleanup

# (VLLM_CPU_EXPERTS_ASYNC, stage_device_rows); device rows = the CPU slots, so
# the threshold rule stages the same experts into either kind of rows
ARMS = {
    "neither": (None, 0),
    "host thread": ("spin", 0),
    "device rows": (None, 3),
    "both": ("spin", 3),
}


@pytest.fixture
def host_op(monkeypatch):
    if not cx.host_op_available():
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield
    cx.reset_runtime()
    tstore.reset_sessions()


class _Kernel:
    """Stand-in for the card's kernel over staging rows: each pair routed to
    a mapped expert gets a row that depends on the bytes of that expert's
    staging row, so rows that are not the expert's change the bits."""

    def __init__(self, *, w13, w2, num_experts, expert_map, **kw):
        self.w13, self.w2 = w13, w2
        self.num_experts, self.expert_map = num_experts, expert_map

    def apply(self, output, hidden_states, topk_weights, topk_ids, **kw):
        ids = topk_ids.numpy()
        slot = self.expert_map.numpy()[ids].reshape(-1)
        idx = np.flatnonzero(slot >= 0)
        rmap = torch.full(ids.shape, -1, dtype=torch.int32)
        rmap.view(-1)[torch.from_numpy(idx)] = torch.arange(len(idx), dtype=torch.int32)
        rows = torch.zeros(max(len(idx), 1), output.shape[-1])
        for r, p in enumerate(idx.tolist()):
            s = int(slot[p])
            assert s < self.num_experts
            a = int(self.w13[s].view(torch.uint8).to(torch.int64).sum()) % 13
            b = int(self.w2[s].view(torch.uint8).to(torch.int64).sum()) % 7
            rows[r] = hidden_states[p // ids.shape[1]].float() * (2 + a + 0.25 * b)
        kw["output_hook"](rows.bfloat16(), None, rmap)


class _Impl:
    """Stand-in for the card's own experts of one tier."""

    def __init__(self, tiers, tier):
        self.tiers, self.tier = tiers, tier

    def apply(self, output, hidden_states, topk_weights, topk_ids, **kw):
        ids = topk_ids.numpy()
        held = self.tiers._tier_of_global[ids] == self.tier
        rmap = torch.full(ids.shape, -1, dtype=torch.int32)
        idx = np.flatnonzero(held.reshape(-1))
        rmap.view(-1)[torch.from_numpy(idx)] = torch.arange(len(idx), dtype=torch.int32)
        rows = torch.empty(max(len(idx), 1), output.shape[-1])
        for r, p in enumerate(idx.tolist()):
            t, k = divmod(p, ids.shape[1])
            rows[r] = hidden_states[t].float() * (1 + int(ids[t, k]))
        kw["output_hook"](rows.bfloat16(), None, rmap)


def _routings(tiers, topk=2):
    rng = np.random.default_rng(5)
    out = [rng.integers(0, tiers.num_global, size=(t, topk)) for t in (4, 8, 3, 6, 8)]
    # one with no CPU-tier expert, one whose CPU-tier experts are all staged
    card = [g for g in range(tiers.num_global) if tiers._tier_of_global[g] != HOST]
    host = [g for g in range(tiers.num_global) if tiers._tier_of_global[g] == HOST]
    out.append(np.array([[card[0], card[1]], [card[2], card[0]]]))
    out.append(np.array([[host[0], card[0]], [host[0], card[1]]]))
    return [torch.from_numpy(r.astype(np.int32)) for r in out]


def _arm(monkeypatch, tmp_path, name, check_pairs=False, last_twice=False):
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    mode, device_rows = ARMS[name]
    if mode is None:
        monkeypatch.delenv(cx._ASYNC_ENV, raising=False)
    else:
        monkeypatch.setenv(cx._ASYNC_ENV, mode)
    regs = _direct(monkeypatch)
    cx.reset_runtime()
    env = _env(monkeypatch, tmp_path, stage_min_calls=2, stage_device_rows=device_rows)
    monkeypatch.setattr(xt, "_xpu_fused_moe", lambda: _Kernel)
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    monkeypatch.setattr(xpu_moe, "sum_topk", _torch_sum_topk)
    tiers = env.tiers[1]
    rt = tiers.cpu
    assert (rt._async is not None) == (mode is not None)
    staged, entered_busy, pairs = [], [], []
    stage = rt.stage

    def watched_stage(pending, device):
        # staging lists on the model's thread: the host thread is idle
        entered_busy.append(rt._async is not None and rt._async._open > 0)
        out = stage(pending, device)
        staged.append([] if pending.staged is None else sorted(pending.staged.tolist()))
        return out

    monkeypatch.setattr(rt, "stage", watched_stage)
    if check_pairs:
        # the host thread's rows: drain, then read the map (test only)
        inner = rt.finish_pairs_async

        def checked(pending, device):
            host = inner(pending, device)
            rt.drain()
            pairs.append(host.row_map.reshape(-1).clone())
            return host

        monkeypatch.setattr(rt, "finish_pairs_async", checked)
    experts = xpu_moe.XPUExperts.__new__(xpu_moe.XPUExperts)
    experts.tiers = tiers
    experts.moe_config = type("C", (), {"ep_size": 1, "ep_rank": 0})()
    experts.impl_apply_kwargs = {}
    experts.combine_explicit = False
    experts.combine_mode = "split"
    experts.placement_trace = None
    experts._tier_impls = lambda w1, w2, k: (_Impl(tiers, FAST), _Impl(tiers, POOL))
    gen = torch.Generator().manual_seed(17)
    outs, routings = [], _routings(tiers)
    if last_twice:
        # the same routing twice: on the CPU the second call's post refills
        # the pinned inputs at once (a card orders that after the wait)
        routings = routings[-1:] * 2
    for ids in routings:
        x = torch.randn(ids.shape[0], HIDDEN, generator=gen).bfloat16()
        w = torch.rand(ids.shape, generator=gen)
        out = torch.empty(ids.shape[0], HIDDEN, dtype=torch.bfloat16)
        experts._apply_tiered(out, x, None, None, w, ids)
        outs.append(out)
    rt.drain()
    outs = [o.view(torch.int16).clone() for o in outs]
    return env, rt, regs, outs, staged, entered_busy, pairs, routings


def test_both_switches_give_the_same_bits(monkeypatch, tmp_path, host_op):
    got = {}
    for name in ARMS:
        env, rt, regs, outs, staged, busy, _, _ = _arm(monkeypatch, tmp_path, name)
        got[name] = (outs, staged)
        assert not any(busy)
        s = rt.snapshot()
        mode, device_rows = ARMS[name]
        if mode is not None:
            assert rt._async.posted == len(outs)  # every call on the host thread
        if device_rows:
            assert s["stage_device"] == sum(1 for k in staged if k)
            # registered at load; the last DMA's rows held until it settles
            assert regs and regs[0][0] == "register"
            assert sorted(rt.cache._held) == sorted(rt._dma_held)
            assert len(rt._dma_held) == len(staged[-1])
            rt.settle_dma()
            assert not rt.cache._held
        else:
            assert s["stage_device"] == 0
        cx.reset_runtime()
    want_outs, want_staged = got["neither"]
    assert any(want_staged) and not all(want_staged)  # stages and computes
    for name, (outs, staged) in got.items():
        assert staged == want_staged, name
        for a, b in zip(want_outs, outs):
            assert torch.equal(a, b), name


def test_the_host_thread_computes_only_the_pairs_not_staged(
    monkeypatch, tmp_path, host_op
):
    env, rt, _, _, staged, _, pairs, routings = _arm(
        monkeypatch, tmp_path, "both", check_pairs=True
    )
    tiers = env.tiers[1]
    assert len(pairs) == len(routings) == len(staged)
    for ids, row_map, on_card in zip(routings, pairs, staged):
        flat = ids.reshape(-1).numpy()
        host = tiers._tier_of_global[flat] == HOST
        on_cpu = host & ~np.isin(flat, on_card)
        assert ((row_map >= 0).numpy() == on_cpu).all()


def test_staging_waits_for_the_host_thread(monkeypatch, tmp_path, host_op):
    """The card does not make the model's thread wait (the device wait is a
    no-op here): a layer call that stages must still wait for the host
    thread's previous layer call before it lists, copies and holds rows on
    the model's thread."""
    import threading

    from vllm.model_executor.layers.fused_moe import xpu_host_wait as hw

    seen: list[str] = []
    gate = threading.Event()
    real_stage = cx.CpuExpertRuntime.stage
    real_pairs = cx.CpuExpertRuntime._pairs_into_buffers

    def stage(self, pending, device):
        seen.append("stage")
        return real_stage(self, pending, device)

    def pairs(self, pending, fill_map=False):
        if fill_map:  # the host thread
            gate.wait(10)
            seen.append("host thread")
        return real_pairs(self, pending, fill_map)

    monkeypatch.setattr(cx.CpuExpertRuntime, "stage", stage)
    monkeypatch.setattr(cx.CpuExpertRuntime, "_pairs_into_buffers", pairs)
    monkeypatch.setattr(hw, "wait_host_flag", lambda *a, **k: None)
    threading.Timer(0.3, gate.set).start()
    _, _, _, _, staged, busy, _, _ = _arm(
        monkeypatch, tmp_path, "both", last_twice=True
    )
    assert len(staged) == 2 and all(staged)
    assert seen == ["stage", "host thread", "stage", "host thread"]
    assert not any(busy)
