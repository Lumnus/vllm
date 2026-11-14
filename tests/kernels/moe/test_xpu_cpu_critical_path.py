# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU tier off the model's thread (VLLM_CPU_EXPERTS_ASYNC=spin): a host
thread computes a layer call's CPU experts while the model's thread enqueues
on; the device waits for the result rows on a host word. The rows, the map
and the layer's sum are those of the model's-thread path, bit for bit; a
failure or a wait that gives up fails the step; unset, nothing changes.
Also the "cpustep" timeline lines."""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_host_wait as hw
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL
from vllm.v1 import runtime_debug

from .test_xpu_cpu_experts import REAL_OP, _host_env, _reference_op
from .xpu_store_harness import HIDDEN

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
    tstore.reset_sessions()


def _env(monkeypatch, tmp_path, mode=None, **extra):
    if mode is None:
        monkeypatch.delenv(cx._ASYNC_ENV, raising=False)
    else:
        monkeypatch.setenv(cx._ASYNC_ENV, mode)
    cx.reset_runtime()
    return _host_env(monkeypatch, tmp_path, **extra)


def _routings(num_global, topk):
    """Routings over every expert, then one with no host call per layer."""
    rng = np.random.default_rng(11)
    out = [rng.integers(0, num_global, size=(t, topk)) for t in (1, 3, 8, 8, 2)]
    return [torch.from_numpy(r.astype(np.int32)) for r in out]


def _no_host(tiers, topk, tokens=4):
    ids = [g for g in range(tiers.num_global) if tiers._tier_of_global[g] != HOST]
    pick = [ids[i % len(ids)] for i in range(tokens * topk)]
    return torch.tensor(pick, dtype=torch.int32).reshape(tokens, topk)


def _rows_by_pair(host, n, topk):
    """{flat pair: row bits} of a HostPairs (None: no host call)."""
    if host is None:
        return {}
    flat = host.row_map.reshape(-1).tolist()
    return {
        p: host.rows[r].view(torch.int16).clone() for p, r in enumerate(flat) if r >= 0
    }


def _calls(env, topk=2):
    """Every (tiers, ids, x) layer call of the comparison, in order."""
    gen = torch.Generator().manual_seed(7)
    out = []
    for ids_list in zip(*[_routings(t.num_global, topk) for t in env.tiers]):
        for tiers, ids in zip(env.tiers, ids_list):
            x = torch.randn(ids.shape[0], HIDDEN, generator=gen).bfloat16()
            out.append((tiers, ids, x))
    for tiers in env.tiers:
        ids = _no_host(tiers, topk)
        x = torch.randn(ids.shape[0], HIDDEN, generator=gen).bfloat16()
        out.append((tiers, ids, x))
    return out


def _run(env, calls, use_async):
    """The HostPairs of each layer call, as {pair: row bits} and the map;
    posted back to back without waiting (the async path) or one by one."""
    got = []
    hosts = []
    for tiers, ids, x in calls:
        rt = tiers.cpu
        w = torch.full(ids.shape, 0.5)
        pending = rt.post(tiers, x, ids, w, weights=False)
        if use_async:
            assert rt.takes_async(pending)
            hosts.append((rt.finish_pairs_async(pending, x.device), pending))
        else:
            hosts.append((rt.finish_pairs(pending, x.device), pending))
        # the copies on the CPU are the pinned buffers themselves: read them
        # before the next layer call refills them (on a card the copies have
        # left before that, in stream order)
        host, pending = hosts[-1]
        n, topk = ids.shape
        got.append(
            (
                _rows_by_pair(host, n, topk),
                None if host is None else host.row_map.reshape(-1).clone().tolist(),
            )
        )
    for tiers in {id(t): t for t, _, _ in calls}.values():
        tiers.cpu.drain()
    return got


def test_unset_keeps_the_model_thread(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[0].cpu
    assert rt._async is None
    ids = torch.zeros((2, 2), dtype=torch.int32)
    pending = rt.post(env.tiers[0], torch.zeros(2, HIDDEN).bfloat16(), ids, ids)
    assert not rt.takes_async(pending)
    rt.finish_pairs(pending, torch.device("cpu"))
    assert not any(t.name == "cpu-experts-host" for t in threading.enumerate())


def test_the_switch_is_read_and_checked(monkeypatch):
    monkeypatch.delenv(cx._ASYNC_ENV, raising=False)
    assert cx._async_settings() is None
    monkeypatch.setenv(cx._ASYNC_ENV, "0")
    assert cx._async_settings() is None
    monkeypatch.setenv(cx._ASYNC_ENV, "spin")
    assert cx._async_settings() == (32, 1_000_000)
    monkeypatch.setenv(cx._ASYNC_TOKENS_ENV, "8")
    monkeypatch.setenv(cx._ASYNC_SPINS_ENV, "100")
    assert cx._async_settings() == (8, 100)
    monkeypatch.setenv(cx._ASYNC_ENV, "yes")
    with pytest.raises(cx.CpuExpertError, match="'spin' or unset"):
        cx._async_settings()
    monkeypatch.delenv(hw._READ_ENV, raising=False)
    assert hw.read_mode() == "volatile"
    monkeypatch.setenv(hw._READ_ENV, "atomic")
    assert hw.read_mode() == "atomic"
    monkeypatch.setenv(hw._READ_ENV, "cached")
    with pytest.raises(ValueError):
        hw.read_mode()


def test_the_host_thread_gives_the_model_threads_rows(monkeypatch, tmp_path, op):
    """The same layer calls through both paths: every pair the CPU computes
    has the same row bits and the same place in the map; a layer call with
    no CPU call gives a map of -1 on the host thread (None on the model's
    thread), which the per-pair sum reads as nothing held."""
    env = _env(monkeypatch, tmp_path)
    calls = _calls(env)
    want = _run(env, calls, use_async=False)
    env = _env(monkeypatch, tmp_path, "spin")
    calls = _calls(env)
    got = _run(env, calls, use_async=True)
    assert len(want) == len(got)
    empty = 0
    for (w_rows, w_map), (g_rows, g_map) in zip(want, got):
        assert g_map is not None
        if w_map is None:
            empty += 1
            assert set(g_map) == {-1} and not g_rows
            continue
        assert g_map == w_map
        assert w_rows.keys() == g_rows.keys()
        for p in w_rows:
            assert torch.equal(w_rows[p], g_rows[p])
    assert empty >= len(env.tiers)  # the calls without a host call ran too


def test_a_failed_layer_call_fails_the_step(monkeypatch, tmp_path, op):
    env = _env(monkeypatch, tmp_path, "spin")
    tiers = env.tiers[0]
    rt = tiers.cpu

    def broken(keys):
        raise cx.CpuExpertError("the store could not give an expert")

    monkeypatch.setattr(rt.cache, "acquire", broken)
    ids = torch.tensor([[g, g] for g in range(tiers.num_global)], dtype=torch.int32)
    ids = ids[torch.from_numpy(tiers._tier_of_global[ids[:, 0].numpy()] == HOST)]
    x = torch.zeros(ids.shape[0], HIDDEN).bfloat16()
    pending = rt.post(tiers, x, ids, torch.ones(ids.shape), weights=False)
    started = time.monotonic()
    rt.finish_pairs_async(pending, x.device)  # the wait ends on "failed"
    assert time.monotonic() - started < 30
    assert int(rt._async.words[hw.FAILED]) == rt._async.seq
    with pytest.raises(cx.CpuExpertError, match="failed"):
        cx.drain_async()
    with pytest.raises(cx.CpuExpertError, match="failed"):
        rt.post(tiers, x, ids, torch.ones(ids.shape), weights=False)


def test_a_wait_that_gives_up_fails_the_step(monkeypatch, tmp_path, op):
    monkeypatch.setenv(cx._ASYNC_SPINS_ENV, "3")
    env = _env(monkeypatch, tmp_path, "spin")
    tiers = env.tiers[0]
    rt = tiers.cpu
    gate = threading.Event()
    inner = rt._pairs_into_buffers

    def slow(pending, fill_map=False):
        gate.wait(10)
        return inner(pending, fill_map)

    monkeypatch.setattr(rt, "_pairs_into_buffers", slow)
    ids = _no_host(tiers, 2)
    x = torch.zeros(ids.shape[0], HIDDEN).bfloat16()
    pending = rt.post(tiers, x, ids, torch.ones(ids.shape), weights=False)
    rt.finish_pairs_async(pending, x.device)
    assert int(rt._async.words[hw.TIMED_OUT]) == rt._async.seq
    gate.set()
    with pytest.raises(cx.CpuExpertError, match="stopped waiting"):
        cx.drain_async()


def test_staging_and_large_calls_stay_on_the_model_thread(monkeypatch, tmp_path, op):
    monkeypatch.setenv(cx._ASYNC_TOKENS_ENV, "4")
    env = _env(monkeypatch, tmp_path, "spin")
    tiers = env.tiers[0]
    rt = tiers.cpu
    x = torch.zeros(5, HIDDEN).bfloat16()
    ids = torch.zeros((5, 2), dtype=torch.int32)
    w = torch.ones(5, 2)
    assert not rt.takes_async(rt.post(tiers, x, ids, w, weights=False))
    small = rt.post(tiers, x[:4], ids[:4], w[:4], weights=False)
    assert rt.takes_async(small)
    rt.config.stage_min_calls = 2
    assert not rt.takes_async(small)
    rt.config.stage_min_calls = 0
    rt.config.stage_min_tokens = 4
    assert not rt.takes_async(small)
    rt.config.stage_min_tokens = 5
    assert rt.takes_async(small)


def test_the_model_thread_path_and_the_step_hook_drain(monkeypatch, tmp_path, op):
    """A layer call on the model's thread, and the model runner's step hook,
    first wait for what the host thread still has."""
    from vllm.v1.worker.gpu import eplb_utils

    env = _env(monkeypatch, tmp_path, "spin")
    tiers = env.tiers[1]
    rt = tiers.cpu
    gate = threading.Event()
    seen = []
    inner = rt._pairs_into_buffers

    def slow(pending, fill_map=False):
        if fill_map:  # the host thread
            gate.wait(10)
            seen.append("host thread")
        else:
            seen.append("model thread")
        return inner(pending, fill_map)

    monkeypatch.setattr(rt, "_pairs_into_buffers", slow)
    ids = torch.tensor([[0, 5], [6, 3]], dtype=torch.int32)
    x = torch.zeros(2, HIDDEN).bfloat16()
    w = torch.ones(2, 2)
    # host side of the card's wait: let the CPU "device" wait return at once
    monkeypatch.setattr(hw, "wait_host_flag", lambda *a, **k: None)
    rt.finish_pairs_async(rt.post(tiers, x, ids, w, weights=False), x.device)
    threading.Timer(0.2, gate.set).start()
    rt.finish_pairs(rt.post(tiers, x, ids, w, weights=False), x.device)
    assert seen == ["host thread", "model thread"]
    gate.clear()
    rt.finish_pairs_async(rt.post(tiers, x, ids, w, weights=False), x.device)
    threading.Timer(0.2, gate.set).start()

    class Runner:
        eplb = SimpleNamespace(step=lambda **kw: seen.append("eplb step"))

        @eplb_utils.step_eplb_after()
        def sample_tokens(self):
            return "out"

    assert Runner().sample_tokens() == "out"
    assert seen[-2:] == ["host thread", "eplb step"]


def _torch_sum_topk(out, w, sources):
    """The per-pair sum in torch (the order and the fp32 accumulation of
    xpu_moe._sum_topk_kernel): the first source holding a pair gives it."""
    T, K = w.shape
    acc = torch.zeros(T, out.shape[-1], dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            v = torch.zeros(out.shape[-1], dtype=torch.float32)
            for rows, rmap in sources:
                r = int(rmap.reshape(T, K)[t, k])
                if r >= 0:
                    v = rows[r].float()
                    break
            acc[t] += v * float(w[t, k])
    out.copy_(acc.to(out.dtype))


def test_the_layer_sum_is_the_same_bits(monkeypatch, tmp_path, op):
    """XPUExperts._apply_tiered with the per-pair sum: the layer's output
    with the CPU part on the host thread equals the model's-thread output
    bit for bit, over routings with and without CPU calls."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    monkeypatch.setattr(xpu_moe, "sum_topk", _torch_sum_topk)

    class Impl:
        def __init__(self, tiers, tier):
            self.tiers, self.tier = tiers, tier

        def apply(self, output, hidden_states, topk_weights, topk_ids, **kw):
            ids = topk_ids.numpy()
            held = self.tiers._tier_of_global[ids] == self.tier
            rmap = torch.full(ids.shape, -1, dtype=torch.int32)
            idx = np.flatnonzero(held.reshape(-1))
            rmap.view(-1)[torch.from_numpy(idx)] = torch.arange(
                len(idx), dtype=torch.int32
            )
            rows = torch.empty(max(len(idx), 1), output.shape[-1])
            for r, p in enumerate(idx.tolist()):
                t, k = divmod(p, ids.shape[1])
                rows[r] = hidden_states[t].float() * (1 + int(ids[t, k]))
            kw["output_hook"](rows.bfloat16(), None, rmap)

    def layer_outputs(env):
        outs = []
        for tiers, ids, x in _calls(env):
            experts = xpu_moe.XPUExperts.__new__(xpu_moe.XPUExperts)
            experts.tiers = tiers
            experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
            experts.impl_apply_kwargs = {}
            experts.combine_explicit = False
            experts.combine_mode = "split"
            experts.placement_trace = None
            experts._tier_impls = lambda w1, w2, k, t=tiers: (
                Impl(t, FAST),
                Impl(t, POOL),
            )
            out = torch.empty(x.shape[0], HIDDEN, dtype=torch.bfloat16)
            w = torch.rand(ids.shape, generator=torch.Generator().manual_seed(9))
            experts._apply_tiered(out, x, None, None, w, ids)
            outs.append(out.view(torch.int16).clone())
        for tiers in env.tiers:
            tiers.cpu.drain()
        return outs

    want = layer_outputs(_env(monkeypatch, tmp_path))
    env = _env(monkeypatch, tmp_path, "spin")
    got = layer_outputs(env)
    assert env.tiers[0].cpu._async.posted == len(got)
    assert len(want) == len(got)
    for a, b in zip(want, got):
        assert torch.equal(a, b)


def test_the_timeline_lines(monkeypatch, tmp_path, op):
    """Debug part "cpustep": one line per layer call on the model's thread,
    a handing line and a host-thread line with the switch."""
    lines = []
    monkeypatch.setattr(
        runtime_debug,
        "emit",
        lambda part, log, msg, *args: lines.append(f"[debug {part}] " + msg % args),
    )
    runtime_debug.arm("cpustep", 60.0, 100)
    try:
        for mode in (None, "spin"):
            env = _env(monkeypatch, tmp_path, mode)
            tiers = env.tiers[1]
            rt = tiers.cpu
            ids = torch.tensor([[0, 5], [6, 3]], dtype=torch.int32)
            x = torch.zeros(2, HIDDEN).bfloat16()
            pending = rt.post(tiers, x, ids, torch.ones(2, 2), weights=False)
            rt.card_enqueued(pending, x.device)
            if mode is None:
                rt.finish_pairs(pending, x.device)
            else:
                rt.finish_pairs_async(pending, x.device)
                rt.drain()
    finally:
        runtime_debug.disarm("cpustep", "test")
    text = "\n".join(lines)
    assert "[debug cpustep]" in text
    assert "list+cache" in text and "op " in text
    assert "handed to the host thread" in text
    assert "host thread:" in text


_KERNEL = r"""
import threading, time, torch
from vllm.model_executor.layers.fused_moe import xpu_host_wait as hw
for atomic in (True, False):
    flag = torch.zeros(4, dtype=torch.int32)
    w = flag.numpy()
    # done already
    w[0] = 5
    hw._wait_host_flag_kernel[(1,)](flag, 5, 1000, ATOMIC=atomic)
    assert w[2] == 0
    # failed
    w[1] = 6
    hw._wait_host_flag_kernel[(1,)](flag, 6, 1000, ATOMIC=atomic)
    assert w[2] == 0
    # nobody writes it: gives up and says so
    hw._wait_host_flag_kernel[(1,)](flag, 7, 50, ATOMIC=atomic)
    assert w[2] == 7, w
    w[2] = 0
    # written by another thread while it spins
    threading.Timer(0.3, lambda: w.__setitem__(0, 8)).start()
    hw._wait_host_flag_kernel[(1,)](flag, 8, 10_000_000, ATOMIC=atomic)
    assert w[2] == 0 and w[0] == 8
print("WAIT-KERNEL-OK")
"""


def test_the_wait_kernel_interpreted():
    """The device wait (Triton, interpreted on the CPU): ends on done and on
    failed, gives up after its reads and writes the timed-out word, and sees
    a write another thread makes while it spins."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", _KERNEL],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "WAIT-KERNEL-OK" in result.stdout, result.stderr[
        -3000:
    ]
