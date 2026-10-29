# SPDX-License-Identifier: Apache-2.0
"""B70-Q27-0006: CPU proofs for the DFlash2 shortlist draft head
(B70_DFLASH_DRAFT_VOCAB).

1. Exactness of the verification math with a shortlist draft (draft probs 0 outside the
   shortlist and outside the K candidates): the single-position output distribution,
   computed in closed form with the rejection kernel's own formulas, equals the target;
   and a Monte Carlo run of the two-position block (Markov candidate walk, standard
   rejection, residual resample, bonus, next step from the resampled context) matches
   the target joint by chi-square.
2. The shortlist head returns the full-vocab top-k restricted to the shortlist (the same
   k values bit for bit; among tied bf16 values either id), at TP1 and at TP2 (two
   gloo CPU processes: the balanced row exchange and the all-gather merge).
3. Env unset / non-Qwen3_5 target (Flash-Next Qwen4Exp) -> no head; compute_candidates
   takes the stock call. No device is opened: everything runs on CPU tensors.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from vllm.v1.worker.gpu.spec_decode.dflash2 import draft_vocab as dv

NEG_INF = float("-inf")


# ---- 1. sampling math ---
# The verification below mirrors
# vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:
#   q from draft_logits / temp with logsumexp over the vocab, -inf rows give q = 0
#   (_compute_local_logits_stats_kernel, _compute_global_logprobs_and_logsumexp);
#   accept iff log p(x) > log u + log q(x) (_rejection_kernel, standard branch);
# residual logits: ratio = exp(log q - log p); where(ratio < 1, log p + log1p(-ratio),
# -inf)
# (_resample_kernel), then a Gumbel argmax over the residual (_seeded_resample_argmax).
# The draft side mirrors dflash2/speculator.py: candidates = top-K of the (shortlist)
# head,
# a Markov walk over K with scores[prev, :] (_selector_walk_kernel), and the realized
# scores
# cached at the K candidate ids, -inf elsewhere (_cache_draft_logits_kernel).


def _logsoftmax(x):
    return torch.log_softmax(x, dim=-1)


def _residual_logits(logp, logq):
    ratio = torch.exp(logq - logp)
    return torch.where(
        ratio < 1.0, logp + torch.log1p(-ratio), torch.full_like(logp, NEG_INF)
    )


def _draft_row(vocab, cand_ids, cand_scores):
    """draft_logits row as the cache kernel leaves it: K scores, -inf elsewhere."""
    row = torch.full((vocab,), NEG_INF, dtype=torch.float64)
    row[cand_ids] = cand_scores
    return row


def _output_dist(p_logits, draft_row, temp):
    """Closed-form output distribution of one verified position (kernel formulas)."""
    logp = _logsoftmax(p_logits / temp)
    logq = _logsoftmax(draft_row / temp)
    p, q = logp.exp(), logq.exp()
    accept = torch.where(q > 0, torch.minimum(q, p), torch.zeros_like(q))
    res = _residual_logits(logp, logq)
    res_mass = torch.exp(res).sum()
    rej = 1.0 - accept.sum()
    out = accept.clone()
    if res_mass > 0:
        out += rej * torch.exp(res) / res_mass
    return out, p


def _shortlist_candidates(head_logits, shortlist, k):
    masked = torch.full_like(head_logits, NEG_INF)
    masked[shortlist] = head_logits[shortlist]
    vals, ids = torch.topk(masked, k)
    return ids, vals


@pytest.mark.parametrize("seed", range(20))
def test_single_position_exact(seed):
    g = torch.Generator().manual_seed(seed)
    vocab, k, temp = 64, 6, [0.7, 1.0, 1.3][seed % 3]
    shortlist = torch.randperm(vocab, generator=g)[:24]
    p_logits = torch.randn(vocab, generator=g, dtype=torch.float64) * 2.5
    # Edge cases: a target-impossible token inside the shortlist, heavy target mass
    # outside it.
    p_logits[shortlist[0]] = NEG_INF
    outside = [t for t in range(vocab) if t not in set(shortlist.tolist())]
    p_logits[outside[seed % len(outside)]] += 4.0
    head = torch.randn(vocab, generator=g, dtype=torch.float64) * 2.0
    head[shortlist[0]] += 10.0  # the draft proposes the token the target forbids
    ids, unary = _shortlist_candidates(head, shortlist, k)
    scores = unary + torch.randn(k, generator=g, dtype=torch.float64)
    row = _draft_row(vocab, ids, scores)
    out, p = _output_dist(p_logits, row, temp)
    assert set(ids.tolist()) <= set(shortlist.tolist())
    assert torch.all(
        row[[t for t in range(vocab) if t not in set(ids.tolist())]] == NEG_INF
    )
    torch.testing.assert_close(out, p, atol=1e-12, rtol=0)


def _gumbel_argmax(logits, g):
    u = torch.rand(logits.shape, generator=g, dtype=torch.float64).clamp_min(1e-300)
    return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)


@pytest.mark.parametrize("mode", ["shortlist", "full_vocab", "inconsistent_q"])
def test_two_position_block_chi_square(mode):
    """Joint of the first two emitted tokens == p1(y1) p2(y2|y1) under the full step
    loop.
    full_vocab is the stock candidate pool; inconsistent_q is a negative control whose
    acceptance ratio reads a q other than the one the walk sampled (unary scores without
    the
    edge term), which must fail."""
    g = torch.Generator().manual_seed(1234)
    vocab, k, temp, n = 16, 4, 0.8, 400_000
    shortlist = torch.tensor([0, 1, 2, 3, 5, 8, 9, 12, 13])
    if mode == "full_vocab":
        shortlist = torch.arange(vocab)
    p1_logits = torch.randn(vocab, generator=g, dtype=torch.float64) * 1.5
    p1_logits[14] += 1.5  # mass outside the shortlist
    p2_logits = torch.randn(vocab, vocab, generator=g, dtype=torch.float64) * 1.5
    p2_logits[:, 2] = NEG_INF  # p2 never emits 2; the draft may still propose it
    logp1 = _logsoftmax(p1_logits / temp)
    logp2 = _logsoftmax(p2_logits / temp)
    # Drafter: a noisy copy of the target (so drafts are often accepted and position 2
    # is
    # exercised), head logits per (context, block position), edge scores over K x K that
    # carry the dependence on the previous candidate, as the selector does.
    p2_fin = p2_logits.clamp_min(-3.0)

    def noise(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * 0.7

    head_ctx0 = torch.stack([p1_logits + noise(vocab), p2_fin.mean(0) + noise(vocab)])
    head_ctx0[1, 2] += 2.0  # the draft proposes a token p2 forbids
    head_ctx1 = p2_fin + noise(vocab, vocab)

    c1, u1 = _shortlist_candidates(head_ctx0[0], shortlist, k)
    c2, u2 = _shortlist_candidates(head_ctx0[1], shortlist, k)
    edges = p2_fin[c1][:, c2] - p2_fin.mean(0)[c2][None, :] + noise(k, k)
    q1_row = _draft_row(vocab, c1, u1)
    # Step 1 draft: x1 index i ~ softmax(u1/T), x2 index j ~ softmax((u2 + edges[i])/T).
    i = _gumbel_argmax((u1 / temp).expand(n, k), g)
    s2 = u2[None, :] + edges[i]
    j = _gumbel_argmax(s2 / temp, g)
    x1, x2 = c1[i], c2[j]
    lq1 = _logsoftmax(q1_row / temp)
    # Position 1.
    u = torch.rand(n, generator=g, dtype=torch.float64)
    acc1 = logp1[x1] > torch.log(u) + lq1[x1]
    y1 = torch.where(
        acc1, x1, _gumbel_argmax(_residual_logits(logp1, lq1).expand(n, vocab), g)
    )
    # Position 2 after an accept: verify x2 against p2(.|x1) with q2 = the realized row.
    q2_rows = torch.full((n, vocab), NEG_INF, dtype=torch.float64)
    q2_rows.scatter_(
        1, c2.expand(n, k), u2.expand(n, k) if mode == "inconsistent_q" else s2
    )
    lq2 = _logsoftmax(q2_rows / temp)
    lp2_y1 = logp2[y1]
    u = torch.rand(n, generator=g, dtype=torch.float64)
    acc2 = (
        lp2_y1.gather(1, x2[:, None])[:, 0]
        > torch.log(u) + lq2.gather(1, x2[:, None])[:, 0]
    )
    y2_block = torch.where(acc2, x2, _gumbel_argmax(_residual_logits(lp2_y1, lq2), g))
    # After a reject at position 1: the next step drafts from context y1 and verifies
    # its
    # first draft against p2(.|y1).
    ids_ctx1 = torch.stack(
        [_shortlist_candidates(head_ctx1[t], shortlist, k)[0] for t in range(vocab)]
    )
    un_ctx1 = torch.stack(
        [_shortlist_candidates(head_ctx1[t], shortlist, k)[1] for t in range(vocab)]
    )
    cc, uu = ids_ctx1[y1], un_ctx1[y1]
    ii = _gumbel_argmax(uu / temp, g)
    xx = cc.gather(1, ii[:, None])[:, 0]
    qq_rows = torch.full((n, vocab), NEG_INF, dtype=torch.float64)
    qq_rows.scatter_(1, cc, uu)
    lqq = _logsoftmax(qq_rows / temp)
    u = torch.rand(n, generator=g, dtype=torch.float64)
    acc3 = (
        lp2_y1.gather(1, xx[:, None])[:, 0]
        > torch.log(u) + lqq.gather(1, xx[:, None])[:, 0]
    )
    y2_next = torch.where(acc3, xx, _gumbel_argmax(_residual_logits(lp2_y1, lqq), g))
    y2 = torch.where(acc1, y2_block, y2_next)

    joint = (logp1[:, None] + logp2).exp()
    expected = (joint * n).flatten()
    observed = torch.bincount(y1 * vocab + y2, minlength=vocab * vocab).double()
    keep = expected >= 5
    other = (expected[~keep].sum(), observed[~keep].sum())
    exp_c = torch.cat([expected[keep], other[0][None]])
    obs_c = torch.cat([observed[keep], other[1][None]])
    chi2 = ((obs_c - exp_c) ** 2 / exp_c).sum()
    df = exp_c.numel() - 1
    pval = torch.special.gammaincc(torch.tensor(df / 2.0), chi2 / 2.0).item()
    print(
        f"two-position [{mode}] chi-square {chi2.item():.1f} df {df} p {pval:.3g} "
        f"accept1 {acc1.double().mean():.3f} accept2|1 {acc2[acc1].double().mean():.3f}"
    )
    if mode == "inconsistent_q":
        assert pval < 1e-6
        return
    assert observed[expected == 0].sum() == 0, "emitted a target-impossible pair"
    assert pval > 1e-3
    assert observed.sum() == n


# ---- 2. the head itself ---


class _Proc:
    def __init__(self, scale=1.0, soft_cap=None):
        self.scale, self.soft_cap = scale, soft_cap


def _ref_topk(full_w, hidden, shortlist, k, proc):
    logits = torch.nn.functional.linear(hidden, full_w)
    masked = torch.full_like(logits, NEG_INF)
    masked[:, shortlist] = logits[:, shortlist]
    v, i = torch.topk(masked, k, dim=-1)
    v = v.float() * proc.scale
    if proc.soft_cap is not None:
        v = torch.tanh(v / proc.soft_cap) * proc.soft_cap
    return i, v


def _fake_lm_head(full_w, tp_size, tp_rank):
    vocab = full_w.shape[0]
    per = vocab // tp_size
    start, end = tp_rank * per, (tp_rank + 1) * per
    return SimpleNamespace(
        weight=full_w[start:end].clone(),
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end
        ),
        tp_size=tp_size,
        org_vocab_size=vocab,
    )


def _tp_worker(rank, world, port, path, out):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    import vllm.distributed as vd
    import vllm.distributed.parallel_state as ps

    group = torch.distributed.group.WORLD
    ps.get_tp_group = lambda: SimpleNamespace(cpu_group=group)
    ps.get_tensor_model_parallel_world_size = lambda: world
    ps.get_tensor_model_parallel_rank = lambda: rank

    def all_gather(t, dim=-1):
        parts = [torch.empty_like(t) for _ in range(world)]
        torch.distributed.all_gather(parts, t.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    vd.tensor_model_parallel_all_gather = all_gather
    dv.target_is_qwen3_5 = lambda m: True
    g = torch.Generator().manual_seed(7)
    vocab, hidden, k = 512, 64, 16
    full_w = torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    x = torch.randn(10, hidden, generator=g).to(torch.bfloat16)
    os.environ[dv.ENV] = path
    draft = SimpleNamespace(lm_head=_fake_lm_head(full_w, world, rank))
    head = dv.maybe_build_draft_vocab_head(draft, object(), k)
    proc = _Proc(scale=0.5, soft_cap=30.0)
    ids, vals = head.get_top_k_tokens(proc, x, k)
    shortlist = dv.read_draft_vocab(path, vocab)
    # Rows are the exact target rows of this rank's balanced share.
    mine = dv.rank_assignment(shortlist, world, rank)
    assert head.ids.tolist() == mine
    assert torch.equal(head.weight, full_w[mine])
    ref_i, ref_v = _ref_topk(full_w, x, shortlist, k, proc)
    # bf16 logits tie often, and top-k among ties is not unique, so the check is: the
    # same
    # k values as the reference, every returned id in the shortlist and unique, and each
    # returned id's own full-head logit (after scale / soft cap) equal to its returned
    # value.
    same_vals = torch.equal(vals.sort(-1).values, ref_v.sort(-1).values)
    full = torch.nn.functional.linear(x, full_w).float() * proc.scale
    full = torch.tanh(full / proc.soft_cap) * proc.soft_cap
    own = torch.equal(full.gather(1, ids), vals)
    in_list = set(ids.flatten().tolist()) <= set(shortlist)
    unique = all(len(set(r)) == k for r in ids.tolist())
    print(
        f"rank {rank}: values equal {same_vals}, ids score their values {own}, "
        f"in shortlist {in_list}, unique {unique}, "
        f"id sets equal {torch.equal(ids.sort(-1).values, ref_i.sort(-1).values)}"
    )
    out[rank] = (same_vals and own and in_list and unique, len(mine))
    torch.distributed.destroy_process_group()


def _write_ids(ids):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(map(str, ids)) + "\n")
    return f.name


@pytest.mark.parametrize("world", [1, 2])
def test_head_matches_full_vocab_topk_on_shortlist(world):
    # Skewed list (mostly low ids, as the frequency list is) to exercise the balanced
    # split.
    g = torch.Generator().manual_seed(3)
    low = torch.randperm(256, generator=g)[:150].tolist()
    high = (256 + torch.randperm(256, generator=g)[:30]).tolist()
    path = _write_ids(low + high)
    out = mp.Manager().dict()
    port = 29500 + (os.getpid() % 1000)
    mp.spawn(_tp_worker, args=(world, port, path, out), nprocs=world, join=True)
    for r in range(world):
        ok, n = out[r]
        assert ok, f"rank {r} top-k differs from the full-vocab reference"
    sizes = [out[r][1] for r in range(world)]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == 180


# ---- 3. off paths ---


def test_env_unset_builds_nothing(monkeypatch):
    monkeypatch.delenv(dv.ENV, raising=False)
    assert dv.maybe_build_draft_vocab_head(object(), object(), 16) is None


def test_flash_next_target_is_off(monkeypatch, tmp_path):
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForCausalLMBase,
        Qwen3_5ForConditionalGeneration,
    )
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )

    for cls in (Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration):
        assert not issubclass(cls, Qwen3_5ForCausalLMBase)
    fn_lm = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    assert not dv.target_is_qwen3_5(SimpleNamespace(get_language_model=lambda: fn_lm))
    q_lm = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
    assert dv.target_is_qwen3_5(SimpleNamespace(get_language_model=lambda: q_lm))
    assert hasattr(Qwen3_5ForConditionalGeneration, "get_language_model")
    p = tmp_path / "ids.txt"
    p.write_text("1 2 3\n")
    monkeypatch.setenv(dv.ENV, str(p))
    assert (
        dv.maybe_build_draft_vocab_head(
            object(), SimpleNamespace(get_language_model=lambda: fn_lm), 16
        )
        is None
    )


def test_compute_candidates_stock_call_without_head():
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

    calls = []
    stub = SimpleNamespace(
        lm_head="LM_HEAD",
        model=SimpleNamespace(candidate_selector=SimpleNamespace(top_k=16)),
        candidate_logits_processor=SimpleNamespace(
            get_top_k_tokens=lambda *a: calls.append(a) or ("ids", "vals")
        ),
    )
    assert DFlash2Qwen3ForCausalLM.compute_candidates(stub, "H") == ("ids", "vals")
    assert calls == [("LM_HEAD", "H", 16)]
