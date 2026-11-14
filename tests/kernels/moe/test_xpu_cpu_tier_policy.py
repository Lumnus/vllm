# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On CPU: prefill staging by the cost model (`stage_min_tokens`). A large
layer call whose CPU-tier experts each get a few tokens, which the per-expert
call threshold never stages, stages the set that makes its host-side time
smallest; the copies into the staging rows split over threads give the same
bytes; the CPU tier's cost fit follows its own measured calls; the settings
change live and leave the compile keys alone; unset, staging is as before."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.expert_store import ExpertStoreConfig
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore

from .test_xpu_cpu_experts import _reference_op, _units
from .test_xpu_tier_classes import _env, _pending
from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def host_op(monkeypatch):
    """The host op, or the torch reference where the kernels lack it (as in
    test_xpu_tier_classes); True when the real op runs."""
    real = cx.host_op_available()
    if not real:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    gc.collect()
    cx.reset_runtime()
    yield real
    cx.reset_runtime()
    tstore.reset_sessions()


# every CPU-tier expert of the layer (physical 5, 6, 7) gets two tokens
IDS = torch.tensor(
    [[5, 0], [5, 1], [6, 0], [6, 1], [7, 0], [7, 1], [0, 1], [1, 2]],
    dtype=torch.int32,
)


def _call(tiers, seed=3):
    n = IDS.shape[0]
    w = torch.full((n, 2), 0.5)
    x = torch.randn(n, HIDDEN, generator=torch.Generator().manual_seed(seed))
    return _pending(tiers, IDS, w, x.bfloat16()), x.bfloat16()


def test_the_call_threshold_misses_what_the_cost_model_stages(
    monkeypatch, tmp_path, host_op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=12)
    tiers = env.tiers[1]
    rt = tiers.cpu
    pending, _ = _call(tiers)
    # the threshold rule (the card's row: 12 calls per expert) stages nothing
    assert rt.stage(pending, torch.device("cpu")) is None
    # by the cost model, the card's time per staged expert small: all three
    rt.config.stage_min_tokens = 8
    rt.config.stage_card_us = 1.0
    pending, x = _call(tiers)
    impl = rt.stage(pending, torch.device("cpu"))
    assert impl is not None
    rows = xt._STAGE_ROWS
    assert sorted(pending.staged.tolist()) == [5, 6, 7]
    for e in (5, 6, 7):
        row = rows.map.tolist()[e]
        for name, unit in expert_in_store(env, 1, e).items():
            assert torch.equal(
                rows.rows_u8(name)[row], unit.reshape(-1).view(torch.uint8)
            )
    # nothing is left for the CPU
    assert rt.finish(pending, torch.device("cpu")) is None
    s = rt.snapshot()
    assert s["stage_by_cost"] == 1 and s["routed_staged"] == 6
    assert s["routed_cpu"] == 0 and s["stage_copy_ms"] >= 0


def test_the_cost_model_leaves_the_cpu_its_cheap_experts(
    monkeypatch, tmp_path, host_op
):
    env = _env(monkeypatch, tmp_path, stage_min_tokens=8, stage_card_us=1e6)
    tiers = env.tiers[1]
    rt = tiers.cpu
    pending, x = _call(tiers)
    assert rt.stage(pending, torch.device("cpu")) is None  # the card is dearer
    assert pending.waited and rt.snapshot()["stage_declined"] == 1
    out = rt.finish(pending, torch.device("cpu"))
    calls = [(e, t, 0.5) for t, row in enumerate(IDS.tolist()) for e in row if e >= 5]
    want = cx.reference_int4_experts(x, _units(env, 1, {5, 6, 7}), calls, INTER, GROUP)
    if host_op:
        assert cx.bf16_steps(out, want) <= 1.0
    else:
        assert torch.equal(out, want)


def test_a_small_call_does_not_wait(monkeypatch, tmp_path, host_op):
    env = _env(monkeypatch, tmp_path, stage_min_tokens=9)
    pending, _ = _call(env.tiers[1])  # 8 tokens
    assert env.tiers[1].cpu.stage(pending, torch.device("cpu")) is None
    assert not pending.waited and pending.selection is None


def test_parallel_copies_give_the_same_rows(monkeypatch, tmp_path, host_op):
    env = _env(monkeypatch, tmp_path, stage_min_tokens=8, stage_card_us=1.0)
    tiers = env.tiers[1]
    rt = tiers.cpu
    for threads in (1, 3):
        rt.config.stage_copy_threads = threads
        if xt._STAGE_ROWS is not None:
            for name in cx.UNIT_TENSORS:
                xt._STAGE_ROWS.rows_u8(name).zero_()
        pending, _ = _call(tiers)
        assert rt.stage(pending, torch.device("cpu")) is not None
        rows = xt._STAGE_ROWS
        for e in (5, 6, 7):
            for name, unit in expert_in_store(env, 1, e).items():
                got = rows.rows_u8(name)[rows.map.tolist()[e]]
                assert torch.equal(got, unit.reshape(-1).view(torch.uint8))
    assert rt._stage_pool._max_workers == 3


class _Fixed:
    """A cost stub: CPU a + b * calls, a fixed copy cost."""

    def cpu_us(self, calls):
        return 100.0 + 30.0 * np.asarray(calls, dtype=float)

    def copy_us(self, unit_bytes, threads):
        return 50.0


def test_the_rule_minimises_the_layers_host_time():
    rt = cx.CpuExpertRuntime.__new__(cx.CpuExpertRuntime)
    rt.config = ExpertStoreConfig(dirs=["/x"], stage_card_us=100.0)
    rt.layout = SimpleNamespace(row_bytes=1)
    rt.cost = _Fixed()
    experts = np.array([10, 11, 12, 13])
    calls = np.array([1, 40, 1, 20])
    # k=0: 2260; k=1: 50 + max(100, 960); k=2: 100 + max(200, 260) = 360;
    # k=3: 150 + max(300, 130) = 450; k=4: 600
    staged, plan = rt._staged_by_cost(experts, calls, num_rows=8)
    assert staged.tolist() == [11, 13] and plan.startswith("2 staged")
    staged, _ = rt._staged_by_cost(experts, calls, num_rows=1)  # the rows bound it
    assert staged.tolist() == [11]
    rt.config.stage_card_us = 1e5
    assert rt._staged_by_cost(experts, calls, num_rows=8)[0].size == 0


def test_the_cpu_fit_follows_the_measured_calls():
    cost = cx.StageCost()
    assert cost.coeffs() == pytest.approx((cx.StageCost.A0_US, cx.StageCost.B0_US))
    rng = np.random.default_rng(0)
    for _ in range(400):
        e = int(rng.integers(5, 40))
        n = e * int(rng.integers(1, 30))
        cost.observe(e, n, 60.0 * e + 12.0 * n)
    a, b = cost.coeffs()
    assert a == pytest.approx(60.0, rel=0.05) and b == pytest.approx(12.0, rel=0.05)
    # decode only (one call per expert): a + b is what the data say
    cost = cx.StageCost()
    for _ in range(400):
        cost.observe(20, 20, 20 * 150.0)
    a, b = cost.coeffs()
    assert a + b == pytest.approx(150.0, rel=0.02)
    cost.copied(4, 400.0)
    assert cost.copy_us(1 << 20, 4) == pytest.approx(100.0)
    assert cx.StageCost().copy_us(32_000, 4) == pytest.approx(1.0)


def test_the_settings_change_live(monkeypatch, tmp_path, host_op):
    from vllm.v1.worker import runtime_control as wrc

    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[0].cpu
    monkeypatch.setattr(wrc, "_cpu_runtime", lambda: rt)
    args = {"stage_min_tokens": 256, "stage_copy_threads": 8, "stage_card_us": 80.5}
    assert wrc._check_cpu_set(None, args) is None
    changes, result, undo = wrc._apply_cpu_set(None, args)
    assert changes["stage_card_us"] == [100.0, 80.5]
    assert rt.config.stage_min_tokens == 256 and rt.config.stage_copy_threads == 8
    assert isinstance(rt.config.stage_card_us, float)
    assert result["cpu"]["stage_cost"]["cpu_us_per_expert"] > 0
    wrc._restore_cpu_set(None, undo)
    assert rt.config.stage_min_tokens == 0
    assert wrc._check_cpu_set(None, {"stage_copy_threads": 0}) is not None
    assert wrc._check_cpu_set(None, {"stage_min_tokens": -1}) is not None
    # a change of threads restarts the CPU fit (it was of the old count)
    rt.cost.observe(10, 10, 1000.0)
    rt.set_threads(rt.threads + 1)
    assert rt.cost.samples == 0


def test_unset_the_compile_keys_are_equal():
    base = ExpertStoreConfig(dirs=["/x"])
    on = ExpertStoreConfig(
        dirs=["/x"], stage_min_tokens=512, stage_copy_threads=8, stage_card_us=70.0
    )
    assert base.compile_factors() == on.compile_factors()
