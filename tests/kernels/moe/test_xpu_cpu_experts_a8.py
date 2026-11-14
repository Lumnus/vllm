# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert slots computed on the CPU for layers whose experts run W4A8 (int8
activations) on the card: the CPU tier follows the card's arithmetic class,
computes with the kernels' ``int4_moe_cpu_a8`` (real when
``VLLM_TEST_INT4_MOE_CPU_LIB`` loads it, else the torch reference stands in),
checks itself at start, and refuses what it cannot serve."""

import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore

from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup

_LIB = os.environ.get("VLLM_TEST_INT4_MOE_CPU_LIB")
if _LIB and not cx.host_op_a8_available():
    torch.ops.load_library(_LIB)
REAL_OP = cx.host_op_available() and cx.host_op_a8_available()

CPU_CFG = {"cpu_isa": "scalar", "cpu_threads": 2, "host_cache_gib": 1e-3}


def _rows(units, cu, offsets, hidden, inter, group):
    rows = {}
    for r in set(cu.tolist()):
        rows[r] = {}
        for i, name in enumerate(cx.UNIT_TENSORS):
            end = offsets[i + 1] if i + 1 < len(offsets) else None
            if end is None:
                end = offsets[i] + hidden * (inter // group) * 2
            rows[r][name] = units[r, offsets[i] : end]
    return rows


def _reference_a16(out, x, units, cu, ct, cw, inter, group, offsets, threads, isa):
    rows = _rows(units, cu, offsets, x.shape[1], inter, group)
    calls = list(zip(cu.tolist(), ct.tolist(), cw.tolist()))
    out.copy_(cx.reference_int4_experts(x, rows, calls, inter, group))


def _reference_a8(
    out, x, units, cu, ct, cw, inter, group, offsets, scheme, threads, isa
):
    assert list(scheme) == cx.W4A8_CARD_SCHEME
    rows = _rows(units, cu, offsets, x.shape[1], inter, group)
    calls = list(zip(cu.tolist(), ct.tolist(), cw.tolist()))
    out.copy_(cx.reference_w4a8_experts(x, rows, calls, inter, group))


@pytest.fixture
def op(monkeypatch):
    """The host ops, real when loaded, else the torch references; the
    harness's 64-element groups count as W4A8-eligible here."""
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "host_op_a8_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_a16)
        monkeypatch.setattr(cx, "_op_run_a8", _reference_a8)
    monkeypatch.setattr(xt, "W4A8_GROUP", GROUP)
    cx.reset_runtime()
    yield REAL_OP
    cx.reset_runtime()


def _host_env(monkeypatch, tmp_path, w4a8=True, **extra):
    run(monkeypatch, tmp_path, store=True)
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=2, store_extra=cfg, w4a8=w4a8)


def _units(env, layer, logicals):
    return {
        lg: {k: v.reshape(-1).view(torch.uint8) for k, v in u.items()}
        for lg in logicals
        for u in [expert_in_store(env, layer, lg)]
    }


def test_w4a8_layers_get_a_w4a8_cpu_tier_equal_to_the_card(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    assert rt is cx.current_runtime() and rt.w4a8
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16()
    out = torch.full_like(x, 7.0)
    assert rt.compute(tiers, ids, w, x, out)
    calls = [(5, 0, 0.3), (6, 1, 0.6), (7, 2, 0.5), (5, 2, 0.5)]
    units = _units(env, 1, {5, 6, 7})
    want = cx.reference_w4a8_experts(x, units, calls, INTER, GROUP)
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))
    # not the W4A16 arithmetic
    a16 = cx.reference_int4_experts(x, units, calls, INTER, GROUP)
    assert not torch.equal(out.view(torch.int16), a16.view(torch.int16))
    # the per-pair rows (weight 1) are the experts' own outputs
    rows = torch.empty(8, HIDDEN, dtype=torch.bfloat16)
    pair = rt.compute_pairs(tiers, ids, x, rows)
    assert pair is not None and pair.tolist() == [1, 2, 4, 5]
    one = cx.reference_w4a8_experts(x, units, [(5, 0, 1.0)], INTER, GROUP)
    assert torch.equal(rows[0].view(torch.int16), one[0].view(torch.int16))
    tstore.reset_sessions()


def test_a_layer_the_card_keeps_w4a16_keeps_the_cpu_w4a16(monkeypatch, tmp_path, op):
    monkeypatch.setattr(xt, "W4A8_GROUP", 128)  # the harness's g64 is refused
    env = _host_env(monkeypatch, tmp_path)
    assert not env.tiers[0].cpu.w4a8
    tstore.reset_sessions()


def test_w4a16_layers_are_unchanged(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path, w4a8=False)
    assert not env.tiers[0].cpu.w4a8
    tstore.reset_sessions()


def test_w4a8_without_the_a8_op_refuses(monkeypatch, tmp_path, op):
    run(monkeypatch, tmp_path, store=True)
    monkeypatch.setattr(cx, "host_op_a8_available", lambda: False)
    with pytest.raises(cx.CpuExpertError, match="int4_moe_cpu_a8"):
        run(
            monkeypatch, tmp_path, store=True, pinned=2, store_extra=CPU_CFG,
            w4a8=True,
        )  # fmt: skip
    tstore.reset_sessions()


def test_the_self_check_refuses_an_op_that_is_not_the_cards(monkeypatch, tmp_path, op):
    run(monkeypatch, tmp_path, store=True)
    real = cx._op_run_a8

    def off_by_one_bit(out, *args):
        real(out, *args)
        out.view(torch.int16)[0, 0] ^= 1

    monkeypatch.setattr(cx, "_op_run_a8", off_by_one_bit)
    with pytest.raises(cx.CpuExpertError, match="differs from the card's W4A8"):
        run(
            monkeypatch, tmp_path, store=True, pinned=2, store_extra=CPU_CFG,
            w4a8=True,
        )  # fmt: skip
    tstore.reset_sessions()


def test_one_process_does_not_mix_the_classes(op):
    layout = cx.UnitLayout(HIDDEN, INTER, GROUP, {n: 0 for n in cx.UNIT_TENSORS})

    class Cfg:
        cpu_threads, cpu_isa, host_cache_gib, read_timeout_s = 1, "scalar", 1e-4, 1

    cx._RUNTIME = object.__new__(cx.CpuExpertRuntime)
    cx._RUNTIME.layout, cx._RUNTIME.w4a8 = layout, True
    with pytest.raises(cx.CpuExpertError, match="W4A8 and W4A16"):
        cx.get_runtime(Cfg(), None, layout, w4a8=False)
    assert cx.get_runtime(Cfg(), None, layout, w4a8=True) is cx._RUNTIME


def test_int8_per_token_is_the_cards_rule():
    # amax 127 -> scale 1.0: ties go to even; the extremes clamp
    v = torch.tensor([[127.0, 2.5, 3.5, -2.5, -0.5, 0.0]])
    q, s = cx.int8_per_token(v)
    assert s.item() == 1.0
    assert q.tolist() == [[127, 2, 4, -2, 0, 0]]
    # a zero row: scale 1e-5 rounded to bf16, q 0
    q0, s0 = cx.int8_per_token(torch.zeros(1, 4))
    assert q0.abs().sum() == 0
    assert s0.item() == torch.tensor(1e-5).bfloat16().float().item()
