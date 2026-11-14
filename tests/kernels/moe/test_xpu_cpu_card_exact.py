# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU experts in the card's exact arithmetic (``cpu_arith card-exact``, the
default): the card's silu taken on the card as a table, the op's arith 7
(the card's grouped GEMM and silu exactly), the start check against the
card's model with no differing value, the fallbacks to ``card``. CPU only:
the card's table comes from a stand-in, the op is the real one when loaded
(``VLLM_TEST_INT4_MOE_CPU_LIB``) and has ``silu_table``, else the torch
model of the card."""

import sys
import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx

from .test_xpu_cpu_exact_combine import _fake_kernels, _pending
from .test_xpu_cpu_experts import CPU_CFG, REAL_OP, _units
from .xpu_store_harness import GROUP, HIDDEN, INTER, run

pytestmark = pytest.mark.skip_global_cleanup


def _table() -> torch.Tensor:
    """A stand-in for the card's silu table: the host's bf16 silu of every
    bf16 gate with every 97th finite entry one step off, so an op or a model
    that computes silu itself instead of reading the table shows."""
    g = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    t = (g.float() / (1.0 + torch.exp(-g.float()))).to(torch.bfloat16)
    bits = t.view(torch.int16).clone()
    idx = torch.arange(0, 65536, 97)
    bits[idx[(bits[idx] & 0x7F80) != 0x7F80]] += 1
    return bits.view(torch.bfloat16)


TABLE = _table()


def _reference_op_exact(
    out,
    x,
    units,
    cu,
    ct,
    cw,
    inter,
    group,
    offsets,
    threads,
    isa,
    arith=0,
    x_row=None,
    silu_table=None,
):
    """The torch model of the card standing in for an op with arith 7."""
    if arith == cx.ARITH_CARD_EXACT and silu_table is None:
        raise RuntimeError("int4_moe_cpu: arith 7 (card-exact) needs silu_table")
    names = cx.UNIT_TENSORS
    rows = {}
    for r in set(cu.tolist()):
        rows[r] = {}
        for i, name in enumerate(names):
            end = offsets[i + 1] if i + 1 < len(names) else None
            if end is None:
                end = offsets[i] + x.shape[1] * (inter // group) * 2
            rows[r][name] = units[r, offsets[i] : end]
    src = ct.tolist() if x_row is None else x_row.tolist()
    if x_row is None:
        calls = list(zip(cu.tolist(), ct.tolist(), cw.tolist()))
        out.copy_(
            cx.reference_int4_experts(
                x, rows, calls, inter, group, arith=arith, silu_table=silu_table
            )
        )
        return
    out.zero_()
    for c, (u, t, w) in enumerate(zip(cu.tolist(), ct.tolist(), cw.tolist())):
        out[t] = cx.reference_int4_experts(
            x[src[c] : src[c] + 1],
            rows,
            [(u, 0, w)],
            inter,
            group,
            arith=arith,
            silu_table=silu_table,
        )[0]


@pytest.fixture
def exact_op(monkeypatch):
    """The host op with silu_table (real when loaded and it has it, else the
    torch model of the card), the card's table from a stand-in; records the
    arith of every op call and how often the table was taken."""
    seen: list[tuple[int, bool]] = []
    taken: list[int] = []
    real = REAL_OP and cx.op_takes_silu_table()
    run_real = cx._op_run
    if not real:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "op_takes_arith", lambda: True)
        monkeypatch.setattr(cx, "op_takes_silu_table", lambda: True)
        monkeypatch.setattr(cx, "op_takes_x_row", lambda: False)

    def run_op(*args):
        arith = args[11] if len(args) > 11 else 0
        table = args[13] if len(args) > 13 else None
        seen.append((arith, table is not None))
        if real:
            run_real(*args)
        else:
            _reference_op_exact(*args)

    def capture(device=None):
        taken.append(1)
        return TABLE.clone()

    monkeypatch.setattr(cx, "_op_run", run_op)
    monkeypatch.setattr(cx, "capture_card_silu_table", capture)
    cx.reset_runtime()
    yield seen, taken, real
    cx.reset_runtime()


def _env(monkeypatch, tmp_path, **extra):
    run(monkeypatch, tmp_path, store=True)
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=2, store_extra=cfg)


def _inputs():
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(3)).bfloat16()
    return ids, w, x


def test_the_default_computes_the_cards_bits(monkeypatch, tmp_path, exact_op):
    """Unset cpu_arith: the table is taken once at start, the start check
    runs arith 7, a layer call of the grouped GEMM hands the op arith 7 with
    the table, and every CPU row equals the card's model to the bit."""
    seen, taken, _ = exact_op
    _fake_kernels(monkeypatch)
    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[1].cpu
    assert rt.config.cpu_arith == "card-exact"
    assert rt.card_exact and rt.arith_mode() == "card-exact"
    assert taken == [1]
    assert (cx.ARITH_CARD_EXACT, True) in seen  # the start check
    tiers = env.tiers[1]
    ids, w, x = _inputs()
    assert tiers.cpu.post(tiers, x, ids, w).arith == cx.ARITH_CARD_EXACT
    seen.clear()
    pending = _pending(rt, tiers, ids, w, x)
    pending.arith = cx.ARITH_CARD_EXACT
    host = rt.finish_pairs(pending, torch.device("cpu"))
    assert seen == [(cx.ARITH_CARD_EXACT, True)]
    for i, (t, e) in enumerate(((0, 5), (1, 6), (2, 7), (2, 5))):
        want = cx.reference_int4_experts(
            x[t : t + 1],
            _units(env, 1, {e}),
            [(e, 0, 1.0)],
            INTER,
            GROUP,
            arith=cx.ARITH_CARD_EXACT,
            silu_table=TABLE,
        )[0]
        assert torch.equal(host.rows[i].view(torch.int16), want.view(torch.int16))
        # the card kernel's arithmetic per call (cpu_arith card) is not it
        card = cx.reference_int4_experts(
            x[t : t + 1], _units(env, 1, {e}), [(e, 0, 1.0)], INTER, GROUP, arith=3
        )[0]
        assert not torch.equal(card, want)


def test_the_start_check_logs_no_differing_value_and_the_cost(
    monkeypatch, tmp_path, exact_op
):
    lines = []
    real_info = cx.logger.info

    def info(msg, *args, **kw):
        lines.append(msg % args if args else msg)
        real_info(msg, *args, **kw)

    monkeypatch.setattr(cx.logger, "info", info)
    _env(monkeypatch, tmp_path)
    exact = [line for line in lines if "card-exact" in line]
    assert len(exact) == 1
    assert "0 of 768 values differ from the card's model" in exact[0]
    assert "CPU cost per expert-token at decode (8 experts x 1 token" in exact[0]
    assert "us card-exact" in exact[0] and "us card (" in exact[0]


def test_an_op_that_is_not_the_cards_is_refused(monkeypatch, tmp_path, exact_op):
    """An op whose arith 7 computes anything but the card's model (here the
    grouped GEMM's per-call arithmetic) does not start."""
    seen, _, _ = exact_op
    wrapped = cx._op_run

    def off_by_arith(*args):
        if len(args) > 11 and args[11] == cx.ARITH_CARD_EXACT:
            args = list(args[:11]) + [3]
            _reference_op_exact(*args)
            return
        wrapped(*args)

    monkeypatch.setattr(cx, "_op_run", off_by_arith)
    with pytest.raises(cx.CpuExpertError, match="differs from the card's model"):
        _env(monkeypatch, tmp_path)


def test_an_op_without_the_table_falls_back_to_card(monkeypatch, tmp_path, exact_op):
    seen, taken, _ = exact_op
    warned = []
    monkeypatch.setattr(cx, "op_takes_silu_table", lambda: False)
    monkeypatch.setattr(
        cx.logger, "warning_once", lambda msg, *a, **k: warned.append(msg % a)
    )
    _fake_kernels(monkeypatch)
    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[1].cpu
    assert not rt.card_exact and rt.arith_mode() == "card"
    assert taken == []
    assert all(arith != cx.ARITH_CARD_EXACT for arith, _ in seen)
    assert len([m for m in warned if "silu_table" in m]) == 1
    ids, w, x = _inputs()
    assert rt.post(env.tiers[1], x, ids, w).arith == 3


def test_no_card_to_take_the_table_from_falls_back_to_card(
    monkeypatch, tmp_path, exact_op
):
    warned = []
    monkeypatch.setattr(cx, "capture_card_silu_table", lambda device=None: None)
    monkeypatch.setattr(
        cx.logger, "warning_once", lambda msg, *a, **k: warned.append(msg % a)
    )
    _fake_kernels(monkeypatch)
    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[1].cpu
    assert not rt.card_exact and rt.arith_mode() == "card"
    assert len([m for m in warned if "no card" in m]) == 1
    ids, w, x = _inputs()
    assert rt.post(env.tiers[1], x, ids, w).arith == 3


def test_cpu_arith_card_and_gemv_are_unchanged(monkeypatch, tmp_path, exact_op):
    seen, taken, _ = exact_op
    _fake_kernels(monkeypatch)
    for mode, want in (("card", 3), ("gemv", 0)):
        cx.reset_runtime()
        seen.clear()
        env = _env(monkeypatch, tmp_path, cpu_arith=mode)
        rt = env.tiers[1].cpu
        assert not rt.card_exact and rt.silu_table is None
        assert all(arith != cx.ARITH_CARD_EXACT for arith, _ in seen)
        ids, w, x = _inputs()
        assert rt.post(env.tiers[1], x, ids, w).arith == want
    assert taken == []


def test_a_call_on_the_cards_gemv_keeps_the_per_call_arithmetic(
    monkeypatch, tmp_path, exact_op
):
    """card-exact is the grouped GEMM's: a layer call the card runs partly
    on its int4 GEMV (few rows, K a multiple of 512) gets card_arith's
    value."""
    _fake_kernels(monkeypatch)
    env = _env(monkeypatch, tmp_path)
    rt = env.tiers[1].cpu
    monkeypatch.setattr(cx, "card_arith", lambda rows, h, i, g: 2 if rows <= 8 else 3)
    ids, w, x = _inputs()
    assert rt.post(env.tiers[1], x[:1], ids[:1], w[:1]).arith == 2
    many = torch.zeros(5, 2, dtype=torch.int32)
    xs = torch.zeros(5, HIDDEN).bfloat16()
    assert rt.post(env.tiers[1], xs, many, torch.ones(5, 2)).arith == (
        cx.ARITH_CARD_EXACT
    )


def test_the_table_is_taken_with_the_cards_activation_op(monkeypatch):
    """capture_card_silu_table runs the kernels' silu activation (the one
    the card's fused MoE runs) over every bf16 gate with up = 1, once, and
    returns it indexed by the gate's bits."""
    calls = []

    def silu_and_mul(out, inp):
        calls.append(tuple(inp.shape))
        d = inp.shape[-1] // 2
        g = inp[:, :d].float()
        out.copy_((g / (1.0 + torch.exp(-g))).to(torch.bfloat16) * inp[:, d:])

    pkg = types.ModuleType("vllm_xpu_kernels")
    mu = types.ModuleType("vllm_xpu_kernels.moe_utils")
    mu.moe_activation_op = lambda name: silu_and_mul if name == "silu" else None
    pkg.moe_utils = mu
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.moe_utils", mu)
    table = cx.capture_card_silu_table(torch.device("cpu"))
    assert calls == [(4096, 32)]
    assert table.dtype == torch.bfloat16 and table.shape == (65536,)
    g = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    want = (g.float() / (1.0 + torch.exp(-g.float()))).to(torch.bfloat16)
    assert torch.equal(table.view(torch.int16), want.view(torch.int16))
    # without a card: none
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert cx.capture_card_silu_table() is None
