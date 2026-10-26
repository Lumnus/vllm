# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-V32-HC3: hot/cold cold prefetch never runs inside a graph capture (CPU only).

Measured 2026-10-26: with B70_MOE_COLD_PREFETCH=1 a 1024-token capture ran the
prefetcher, recorded its side-stream events into the graph, and the next capture waited forever.
Capture now takes the UVA path at any size; the prefetcher refuses to run while capturing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import b70_hotcold


class _Cold:
    def __init__(self):
        self.calls = []

    def apply(self, **kw):
        self.calls.append(kw["hidden_states"].shape[0])


class _Prefetch:
    def __init__(self):
        self.calls = []

    def acquire(self, plan, device):
        self.calls.append("acquire")
        raise AssertionError("prefetcher used")

    def release_and_prefetch(self, plan):
        self.calls.append("release")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("B70_MOE_COLD_PREFETCH", "1")
    monkeypatch.setenv("B70_MOE_COLD_PREFETCH_MIN_TOKENS", "512")
    fake = _Prefetch()
    monkeypatch.setattr(b70_hotcold, "_PREFETCH", fake)
    state = SimpleNamespace(capturing=False)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: state.capturing)
    return SimpleNamespace(prefetch=fake, state=state, mp=monkeypatch)


def _apply(n):
    plan = b70_hotcold.HotColdPlan.__new__(b70_hotcold.HotColdPlan)
    cold = _Cold()
    x = torch.zeros(n, 8)
    plan.apply_cold(cold, torch.zeros_like(x), x, torch.zeros(n, 8), torch.zeros(n, 8, dtype=torch.int32))
    return cold


@pytest.mark.parametrize("n", [512, 1024, 4096])
def test_capture_reads_cold_over_uva(env, n):
    env.state.capturing = True
    cold = _apply(n)
    assert cold.calls == [n] and env.prefetch.calls == []


def test_small_batches_never_prefetch(env):
    assert _apply(256).calls == [256] and env.prefetch.calls == []


def test_eager_large_batch_still_prefetches(env):
    with pytest.raises(AssertionError, match="prefetcher used"):
        _apply(1024)
    assert env.prefetch.calls == ["acquire"]


def test_prefetch_off_never_prefetches(env):
    env.mp.setenv("B70_MOE_COLD_PREFETCH", "0")
    env.state.capturing = True
    assert _apply(1024).calls == [1024]
    env.state.capturing = False
    assert _apply(1024).calls == [1024] and env.prefetch.calls == []


@pytest.mark.parametrize("call", ["_load", "acquire", "release_and_prefetch"])
def test_prefetcher_refuses_under_capture(env, call):
    env.state.capturing = True
    real = b70_hotcold._ColdPrefetcher()
    plan = SimpleNamespace(layer_idx=3, cold={})
    args = {"_load": (0, plan), "acquire": (plan, "cpu"), "release_and_prefetch": (plan,)}[call]
    with pytest.raises(RuntimeError, match="during graph capture"):
        getattr(real, call)(*args)
