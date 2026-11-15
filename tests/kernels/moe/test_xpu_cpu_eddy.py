# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The eddy (VLLM_CPU_EXPERTS_EDDY): rows whose MoE layer needs CPU-tier
experts park at that layer and rejoin there at a later step, as substitutes
in between; the card never waits for the CPU.

A small decoder drives the real ``XPUExperts._apply_tiered`` (stand-in card
experts, the real CPU tier with the reference op) through steps of several
streams, with a KV cache written through a slot mapping and a recurrent
state through state indices (both skip ``PAD`` as the engine's writers do),
once waiting for the CPU and once with the eddy. Shown: every stream's
tokens and every KV row and state are the same bits; substitutes write no
KV, advance no state and make no CPU call; a step finishes while the CPU is
held; the switch and the cpu_arith names."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_eddy
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .test_xpu_cpu_critical_path import _torch_sum_topk
from .test_xpu_cpu_experts import REAL_OP, _host_env, _reference_op
from .xpu_store_harness import HIDDEN, NAME

pytestmark = pytest.mark.skip_global_cleanup

ROWS, VOCAB, MAXLEN, TOPK, TOKENS = 4, 32, 16, 2, 6


@pytest.fixture
def op(monkeypatch):
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    monkeypatch.delenv(cx._ASYNC_ENV, raising=False)
    cx.reset_runtime()
    xpu_eddy.reset()
    yield REAL_OP
    xpu_eddy.reset()
    cx.reset_runtime()
    tstore.reset_sessions()


class _CardImpl:
    """A stand-in card expert call for one tier: each pair's row from the
    token's row and the expert id; pairs of other tiers are not held."""

    calls: list[int] = []

    def __init__(self, tiers, tier):
        self.tiers, self.tier = tiers, tier

    def apply(self, output, hidden_states, topk_weights, topk_ids, **kw):
        ids = topk_ids.numpy()
        held = self.tiers._tier_of_global[ids] == self.tier
        rmap = torch.full(ids.shape, -1, dtype=torch.int32)
        idx = np.flatnonzero(held.reshape(-1))
        rmap.view(-1)[torch.from_numpy(idx)] = torch.arange(len(idx), dtype=torch.int32)
        rows = torch.zeros(max(len(idx), 1), output.shape[-1])
        for r, p in enumerate(idx.tolist()):
            t, k = divmod(p, ids.shape[1])
            g = int(ids[t, k])
            rows[r] = torch.tanh(hidden_states[t].float() * (0.5 + 0.25 * g))
        kw["output_hook"](rows.bfloat16(), None, rmap)


def _experts(tiers):
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    e = xpu_moe.XPUExperts.__new__(xpu_moe.XPUExperts)
    e.tiers = tiers
    e.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    e.impl_apply_kwargs = {}
    e.combine_explicit = False
    e.combine_mode = "split"
    e.placement_trace = None
    e._tier_impls = lambda w1, w2, k, t=tiers: (_CardImpl(t, FAST), _CardImpl(t, POOL))
    return e


def _norm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()


class Toy:
    """Embedding, per layer a KV-writing attention and a recurrent state,
    then the MoE layer; the residual stream as a decoder carries it."""

    def __init__(self, env):
        g = torch.Generator().manual_seed(5)
        self.env = env
        self.layers = [_experts(t) for t in env.tiers]
        self.emb = (torch.randn(VOCAB, HIDDEN, generator=g) * 0.5).bfloat16()
        self.wr = [torch.randn(HIDDEN, 8, generator=g) for _ in env.tiers]
        self.wo = torch.randn(HIDDEN, VOCAB, generator=g)
        self.kv = [torch.zeros(ROWS * MAXLEN, HIDDEN) for _ in env.tiers]
        self.st = [torch.zeros(ROWS + 1, HIDDEN) for _ in env.tiers]  # 0: null
        # the step's index buffers (the engine's slot mapping, state indices)
        self.slot = torch.full((ROWS,), -1, dtype=torch.int64)
        self.sidx = torch.full((ROWS,), -1, dtype=torch.int64)

    def set_indices(self, pos):
        """The step's index buffers, as the engine builds them before a step."""
        self.slot.copy_(torch.arange(ROWS) * MAXLEN + pos)
        self.sidx.copy_(torch.arange(ROWS) + 1)

    def forward(self, tokens, pos, book=None):
        h = self.emb[tokens]
        res = None
        for i, experts in enumerate(self.layers):
            res = h if res is None else res + h
            x = _norm(res)
            # attention: write this position through the slot mapping (PAD:
            # nothing), read the sequence's positions so far
            ok = self.slot >= 0
            self.kv[i][self.slot[ok]] = x[ok].float()
            a = torch.stack(
                [
                    self.kv[i][r * MAXLEN : r * MAXLEN + int(pos[r]) + 1].mean(0)
                    for r in range(ROWS)
                ]
            )
            # recurrent state through state indices (<= 0: touch nothing)
            live = self.sidx > 0
            new = 0.5 * self.st[i][self.sidx.clamp(min=0)] + x.float()
            self.st[i][self.sidx[live]] = new[live]
            res = res + (a + 0.1 * new).bfloat16()
            xm = _norm(res)
            probs = torch.softmax(xm.float() @ self.wr[i], -1)
            w, ids = probs.topk(TOPK, -1)
            w = w / w.sum(-1, keepdim=True)
            out = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
            experts._apply_tiered(out, xm, None, None, w, ids.int())
            if book is not None:
                res = book.hold(NAME.format(i), "residual", res)
            h = out
        logits = (res + h).float() @ self.wo
        return logits.argmax(-1)


def _setup(monkeypatch, tmp_path):
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    monkeypatch.setattr(xpu_moe, "sum_topk", _torch_sum_topk)
    cx.reset_runtime()
    env = _host_env(monkeypatch, tmp_path)
    for t in env.tiers:
        assert (t._tier_of_global == HOST).sum() == 3
    return Toy(env)


def _run_waiting(toy):
    tokens = torch.arange(ROWS) * 3 % VOCAB
    pos = torch.zeros(ROWS, dtype=torch.int64)
    out = [[] for _ in range(ROWS)]
    for _ in range(TOKENS):
        toy.set_indices(pos)
        nxt = toy.forward(tokens, pos)
        for r in range(ROWS):
            out[r].append(int(nxt[r]))
        tokens, pos = nxt, pos + 1
    return out


def _run_eddy(toy, book, max_steps=200, before_step=None):
    tokens = torch.arange(ROWS) * 3 % VOCAB
    pos = torch.zeros(ROWS, dtype=torch.int64)
    out = [[] for _ in range(ROWS)]
    step = 0
    while min(len(o) for o in out) < TOKENS:
        assert step < max_steps
        if before_step is not None:
            before_step(step)
        done = np.array([len(o) >= TOKENS for o in out])
        toy.set_indices(pos)
        book.begin_step(step, ROWS, torch.device("cpu"), [toy.slot, toy.sidx], done)
        nxt = toy.forward(tokens, pos, book)
        produced = book.end_step() & ~done
        for r in np.flatnonzero(produced).tolist():
            out[r].append(int(nxt[r]))
        mask = torch.from_numpy(produced)
        tokens = torch.where(mask, nxt, tokens)
        pos = pos + mask.long()
        step += 1
    return out, step


def _state(toy):
    return (
        [kv.clone() for kv in toy.kv],
        [st.clone() for st in toy.st],
        int(toy.env.tiers[0].cpu.snapshot()["cpu_calls"]),
    )


def _assert_same(want, got, want_state, got_state):
    assert got == want
    for a, b in zip(want_state[0] + want_state[1], got_state[0] + got_state[1]):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    # every real CPU pair once, no substitute ever reached the CPU
    assert got_state[2] == want_state[2]


def test_parked_streams_give_the_waiting_runs_bits(monkeypatch, tmp_path, op):
    toy = _setup(monkeypatch, tmp_path)
    want = _run_waiting(toy)
    want_state = _state(toy)
    assert xpu_eddy.current_book() is None

    toy = _setup(monkeypatch, tmp_path)
    book = xpu_eddy.EddyBook(ROWS, depth=2)
    got, steps = _run_eddy(toy, book)
    got_state = _state(toy)
    book.stop()
    assert book.parks_seen > 0 and book.rejoins > 0
    assert steps > TOKENS  # parked rows took extra steps
    _assert_same(want, got, want_state, got_state)


def test_a_slow_cpu_keeps_rows_in_the_eddy_and_the_bits(monkeypatch, tmp_path, op):
    """CPU jobs finish late: rows wait whole steps as substitutes (and the
    ring's backpressure engages); the result is still the same bits."""
    toy = _setup(monkeypatch, tmp_path)
    want = _run_waiting(toy)
    want_state = _state(toy)

    toy = _setup(monkeypatch, tmp_path)
    real = cx._op_run
    import time

    def slow(*a):
        time.sleep(0.05)
        return real(*a)

    monkeypatch.setattr(cx, "_op_run", slow)
    book = xpu_eddy.EddyBook(ROWS, depth=2)
    got, _ = _run_eddy(toy, book)
    got_state = _state(toy)
    book.stop()
    assert book.waited_steps > 0
    _assert_same(want, got, want_state, got_state)


def test_the_step_does_not_wait_for_the_cpu(monkeypatch, tmp_path, op):
    """With the CPU held, the waiting path blocks inside the step; the eddy's
    step finishes (rows park), and the next step too (they wait as
    substitutes); released, the run gives the waiting run's bits."""
    toy = _setup(monkeypatch, tmp_path)
    want = _run_waiting(toy)
    want_state = _state(toy)

    gate = threading.Event()
    real = cx._op_run

    def held(*a):
        gate.wait()
        return real(*a)

    # the waiting path: one step does not finish while the CPU is held
    toy = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cx, "_op_run", held)
    toy.set_indices(torch.zeros(ROWS, dtype=torch.int64))
    t = threading.Thread(
        target=toy.forward,
        args=(torch.arange(ROWS) * 3 % VOCAB, torch.zeros(ROWS, dtype=torch.int64)),
        daemon=True,
    )
    t.start()
    t.join(1.0)
    assert t.is_alive()
    gate.set()
    t.join(30)
    assert not t.is_alive()

    gate.clear()
    monkeypatch.setattr(cx, "_op_run", real)  # the tier's start check runs it
    toy = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cx, "_op_run", held)
    book = xpu_eddy.EddyBook(ROWS, depth=2)
    finished = []

    def before(step):
        if step == 2:  # two steps ran with the CPU held
            finished.append(step)
            gate.set()

    got, _ = _run_eddy(toy, book, before_step=before)
    got_state = _state(toy)
    book.stop()
    assert finished == [2]
    _assert_same(want, got, want_state, got_state)


def test_substitutes_write_nothing(monkeypatch, tmp_path, op):
    """One step in which every row is a substitute (all wait): no KV row,
    no state, no CPU call, no card pair with weight."""
    toy = _setup(monkeypatch, tmp_path)
    book = xpu_eddy.EddyBook(ROWS, depth=2)
    book.wait_layer[:ROWS] = 0
    book.wait_step[:ROWS] = 10**9  # a job that never finishes
    before = _state(toy)
    tokens = torch.arange(ROWS)
    pos = torch.zeros(ROWS, dtype=torch.int64)
    toy.set_indices(pos)
    book.begin_step(0, ROWS, torch.device("cpu"), [toy.slot, toy.sidx])
    assert not bool(book.active[:ROWS].any())
    toy.forward(tokens, pos, book)
    produced = book.end_step()
    after = _state(toy)
    book.stop()
    assert not produced.any()
    for a, b in zip(before[0] + before[1], after[0] + after[1]):
        assert torch.equal(a, b)
    assert after[2] == before[2]
    # the index buffers are back as the driver set them
    assert toy.slot.tolist() == (torch.arange(ROWS) * MAXLEN).tolist()


def test_the_switch(monkeypatch):
    monkeypatch.delenv(xpu_eddy._ENV, raising=False)
    assert not xpu_eddy.eddy_enabled()
    monkeypatch.setenv(xpu_eddy._ENV, "0")
    assert not xpu_eddy.eddy_enabled()
    monkeypatch.setenv(xpu_eddy._ENV, "1")
    assert xpu_eddy.eddy_enabled()
    monkeypatch.setenv(xpu_eddy._ENV, "yes")
    with pytest.raises(ValueError):
        xpu_eddy.eddy_enabled()
    monkeypatch.setenv(xpu_eddy._DEPTH_ENV, "4")
    with pytest.raises(ValueError):
        xpu_eddy._depth()
    assert xpu_eddy.current_book() is None


def test_cpu_arith_names_and_aliases(monkeypatch):
    from vllm.config import expert_store as es

    warned = []
    monkeypatch.setattr(
        es.logger, "warning_once", lambda msg, *a, **k: warned.append(msg % a)
    )
    assert es.ExpertStoreConfig().cpu_arith == "exact"
    assert es.ExpertStoreConfig(cpu_arith="fast").cpu_arith == "fast"
    assert es.ExpertStoreConfig(cpu_arith="exact").cpu_arith == "exact"
    assert warned == []
    assert es.ExpertStoreConfig(cpu_arith="card").cpu_arith == "fast"
    assert es.ExpertStoreConfig(cpu_arith="card-exact").cpu_arith == "exact"
    assert len(warned) == 2
    assert "fast (within the floor)" in warned[0]
    assert "exact (bit-identical to the card)" in warned[1]
    assert es.cpu_arith_name("card") == "fast"
    with pytest.raises(ValueError):
        es.ExpertStoreConfig(cpu_arith="cardish")
