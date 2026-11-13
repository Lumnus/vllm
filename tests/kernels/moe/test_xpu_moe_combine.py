# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How the XPU fused MoE sums a token's routed experts when they are computed
in pieces (VLLM_XPU_MOE_COMBINE), and the placement trace.

"split" (the default) rounds each piece (a rank's device tier, its host tier)
on its own and adds the pieces, so the bits depend on where each expert sits.
"rank" and "exact" must give the same bits for every placement of the same
per-expert outputs. The Triton kernels run on CPU under the interpreter, in a
subprocess so the interpreter setting stays out of this session."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import xpu_moe

_INTERPRETED_CHECK = r"""
import torch
from vllm.model_executor.layers.fused_moe.experts import xpu_moe

g = torch.Generator().manual_seed(0)
T, K, H = 9, 10, 1100  # H not a multiple of the block
bf = torch.bfloat16
ys = (torch.randn(T, K, H, generator=g) * 0.05).to(bf)  # per-pair expert outputs
w = torch.softmax(torch.randn(T, K, generator=g), dim=-1)


def reference(ys, w):
    acc = torch.zeros(T, H)
    for k in range(K):
        acc = acc + ys[:, k].float() * w[:, k : k + 1]
    return acc.to(bf)


def placement(seed, parts):
    # part of each (t, k): 0 .. parts-1
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, parts, (T, K), generator=gen)


def sources(part, which):
    # expert-grouped rows of the pairs with part == which, in a shuffled
    # order, and the [T, K] row map (-1 elsewhere)
    idx = (part == which).nonzero()
    order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(7 + which))
    idx = idx[order]
    rows = torch.zeros(max(len(idx), 1), H, dtype=bf)
    rmap = torch.full((T, K), -1, dtype=torch.int32)
    for r, (t, k) in enumerate(idx.tolist()):
        rows[r] = ys[t, k]
        rmap[t, k] = r
    return rows, rmap


def split(part):
    # the stock sum: each tier rounded on its own, then added in bf16
    out = None
    for which in (0, 1):
        m = (part == which).float()
        acc = torch.zeros(T, H)
        for k in range(K):
            acc = acc + ys[:, k].float() * w[:, k : k + 1] * m[:, k : k + 1]
        out = acc.to(bf) if out is None else (out.float() + acc.to(bf).float()).to(bf)
    return out


want = reference(ys, w)


def near(out, ref):
    # the interpreter's float -> bfloat16 cast may round differently from
    # torch's: at most one bf16 step of the reference
    step = ref.float().abs().clamp(min=2.0**-126) * 2.0**-7
    return bool(((out.float() - ref.float()).abs() <= step).all())


outs, splits = [], []
for seed in range(4):
    part = placement(seed, 2)
    out = torch.empty(T, H, dtype=bf)
    xpu_moe.sum_topk(out, w, [sources(part, 0), sources(part, 1)])
    outs.append(out)
    splits.append(split(part))
for out in outs:
    assert torch.equal(out, outs[0]), "the sum depends on the placement"
assert near(outs[0], want), (outs[0].float() - want.float()).abs().max()
assert any(not torch.equal(s, splits[0]) for s in splits[1:]), "split did not move"

# one source (a rank without tiers) and pairs nobody holds (zeros)
part = placement(11, 3)  # part 2: held by no source on this rank
out = torch.empty(T, H, dtype=bf)
xpu_moe.sum_topk(out, w, [sources(part, 0)])
held = (part == 0).float()
acc = torch.zeros(T, H)
for k in range(K):
    acc = acc + ys[:, k].float() * w[:, k : k + 1] * held[:, k : k + 1]
assert near(out, acc.to(bf))

# exact: two ranks, each with two tiers; the all-reduce adds the other
# rank's pick (one nonzero per element: exact). Rank 0 gets the full sum,
# rank 1 zeros; any placement gives the same bits.
results = []
for seed in range(3):
    part = placement(100 + seed, 4)  # (rank, tier) = divmod(part, 2)
    rank_sources = [
        [sources(part, 2 * r), sources(part, 2 * r + 1)] for r in range(2)
    ]
    for chunk in (512, 4):
        per_rank = []
        for r in range(2):
            other = rank_sources[1 - r]
            starts = iter(range(0, T, chunk))

            def all_reduce(per_k, other=other, starts=starts):
                s = next(starts)
                e = min(s + chunk, T)
                mine = torch.empty_like(per_k)
                xpu_moe.pick_rows(mine, [(rows, m[s:e]) for rows, m in other])
                return (per_k.float() + mine.float()).to(bf)

            out = torch.full((T, H), 7.0, dtype=bf)
            xpu_moe.combine_exact(out, w, rank_sources[r], r, all_reduce, chunk=chunk)
            per_rank.append(out)
        assert not per_rank[1].any()
        results.append(per_rank[0])
for out in results:
    # the same kernel and order as the "rank" sum above: the same bits
    assert torch.equal(out, outs[0]), "the exact sum depends on the placement"
print("COMBINE-OK")
"""


def test_combine_kernels_interpreted():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, "-c", _INTERPRETED_CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert run.returncode == 0 and "COMBINE-OK" in run.stdout, run.stderr[-3000:]


_TIERED_CHECK = r"""
import torch
from types import SimpleNamespace
from vllm.model_executor.layers.fused_moe.experts import xpu_moe

bf = torch.bfloat16
T, K, H, E = 6, 4, 64, 16
g = torch.Generator().manual_seed(1)
expert_w = (torch.randn(E, H, generator=g)).to(bf)
x = (torch.randn(T, H, generator=g)).to(bf)
topk_ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).int()
topk_w = torch.softmax(torch.randn(T, K, generator=g), dim=-1)


class Impl:
    # a fused MoE over the experts its map holds: row of pair (t, k) is
    # x[t] * expert_w[e] (bf16), grouped by local expert; moe_gather-like sum
    def __init__(self, emap):
        self.emap = emap

    def apply(self, output, hidden_states, topk_weights, topk_ids, output_hook=None):
        local = self.emap[topk_ids.long()]
        pairs = (local >= 0).nonzero().tolist()
        pairs.sort(key=lambda p: (int(local[p[0], p[1]]), p[0]))
        rows = torch.zeros(max(len(pairs), 1), H, dtype=bf)
        rmap = torch.full((T, K), -1, dtype=torch.int32)
        for r, (t, k) in enumerate(pairs):
            y = hidden_states[t].float() * expert_w[topk_ids[t, k]].float()
            rows[r] = y.to(bf)
            rmap[t, k] = r
        if output_hook is not None:
            output_hook(rows, None, rmap)
        acc = torch.zeros(T, H)
        for k in range(K):
            m = rmap[:, k] >= 0
            v = rows[rmap[:, k].clamp(min=0).long()].float() * m[:, None]
            acc = acc + v * topk_weights[:, k : k + 1]
        output.copy_(acc.to(bf))


def run(mode, fast_set):
    fast_map = torch.full((E,), -1, dtype=torch.int32)
    slow_map = torch.full((E,), -1, dtype=torch.int32)
    nf = ns = 0
    for e in range(E):
        if e in fast_set:
            fast_map[e] = nf; nf += 1
        else:
            slow_map[e] = ns; ns += 1
    tiers = SimpleNamespace(num_global=E)
    experts = object.__new__(xpu_moe.XPUExperts)
    experts.tiers = tiers
    experts.impl_apply_kwargs = {}
    experts.combine_mode = mode
    experts.placement_trace = None
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    experts._tier_impls = lambda w1, w2, topk: (Impl(fast_map), Impl(slow_map))
    out = torch.empty(T, H, dtype=bf)
    experts._apply_tiered(out, x, None, None, topk_w, topk_ids)
    return out


xpu_moe.staged_impl = lambda impl, n, slot: impl
xpu_moe.fused_moe_has_output_hook = lambda: True
placements = [set(range(0, 8)), set(range(8, 16)), {0, 3, 5, 6, 9, 12, 13, 15}, set()]
for mode in ("rank", "exact"):
    outs = [run(mode, p) for p in placements]
    assert all(torch.equal(o, outs[0]) for o in outs), mode
splits = [run("split", p) for p in placements]
assert any(not torch.equal(s, splits[0]) for s in splits), "split did not move"
print("TIERED-OK")
"""


def test_tiered_apply_is_placement_invariant_interpreted():
    """XPUExperts._apply_tiered over stub fused MoEs: "rank" and "exact" give
    the same bits for every split of the experts between the tiers; "split"
    (the stock sum) does not."""
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    run = subprocess.run(
        [sys.executable, "-c", _TIERED_CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert run.returncode == 0 and "TIERED-OK" in run.stdout, run.stderr[-3000:]


def test_default_mode_is_split(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_XPU_MOE_COMBINE", raising=False)
    assert envs.VLLM_XPU_MOE_COMBINE == "split"
    monkeypatch.setenv("VLLM_XPU_MOE_COMBINE", "exact")
    assert envs.VLLM_XPU_MOE_COMBINE == "exact"
    monkeypatch.setenv("VLLM_XPU_MOE_COMBINE", "fast")
    with pytest.raises(ValueError):
        _ = envs.VLLM_XPU_MOE_COMBINE
    assert "VLLM_XPU_MOE_COMBINE" not in envs.compile_factors()
    assert "VLLM_XPU_MOE_PLACEMENT_TRACE" not in envs.compile_factors()


def _experts(mode, **kwargs):
    experts = object.__new__(xpu_moe.XPUExperts)
    experts.combine_mode = mode
    experts.impl_apply_kwargs = kwargs
    experts.moe_config = SimpleNamespace(dp_size=1, ep_size=2, ep_rank=0)
    return experts


def test_combine_mode_refuses_what_cannot_run(monkeypatch):
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    assert _experts("split", w4a8=True)._combine_mode() == "split"
    assert _experts("rank")._combine_mode() == "rank"
    with pytest.raises(ValueError, match="W4A8"):
        _experts("exact", w4a8=True)._combine_mode()
    with pytest.raises(ValueError, match="quantized"):
        _experts("exact")._combine_mode(torch.ones(1))
    dp = _experts("exact")
    dp.moe_config.dp_size = 2
    with pytest.raises(ValueError, match="data parallelism"):
        dp._combine_mode()
    dp.combine_mode = "rank"  # one rank's tiers only: no exchange
    assert dp._combine_mode() == "rank"
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: False)
    with pytest.raises(ValueError, match="output_hook"):
        _experts("rank")._combine_mode()


@pytest.mark.parametrize("mode", ["split", "rank"])
def test_untiered_ep_call_unchanged_unless_exact(monkeypatch, mode):
    """Without tiers, "split" and "rank" call the fused MoE exactly as
    before: no output_hook, one call."""
    seen = []

    class Impl:
        num_experts = 4

        def apply(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr(xpu_moe, "staged_impl", lambda impl, n, slot: impl)
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    experts = _experts(mode)
    experts.tiers = None
    experts.fused_moe_impl = Impl()
    experts.ep_contract = xpu_moe.EPIdsContract.PADDING_MASK
    experts.moe_config = SimpleNamespace(ep_size=2, ep_rank=0)
    ids = torch.tensor([[0, 5]], dtype=torch.int32)
    experts.apply(
        torch.zeros(1, 8),
        torch.zeros(1, 8),
        None,
        None,
        torch.ones(1, 2),
        ids,
        None,
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert len(seen) == 1 and "output_hook" not in seen[0]


def test_untiered_ep_exact_hooks_and_combines(monkeypatch):
    calls = []

    class Impl:
        num_experts = 4

        def apply(self, **kwargs):
            hook = kwargs["output_hook"]
            hook(torch.zeros(2, 8), None, torch.tensor([[0, -1]], dtype=torch.int32))

    monkeypatch.setattr(xpu_moe, "staged_impl", lambda impl, n, slot: impl)
    monkeypatch.setattr(xpu_moe, "fused_moe_has_output_hook", lambda: True)
    monkeypatch.setattr(
        xpu_moe.XPUExperts,
        "_combine_exact",
        lambda self, out, w, sources: calls.append(len(sources)),
    )
    experts = _experts("exact")
    experts.tiers = None
    experts.fused_moe_impl = Impl()
    experts.ep_contract = xpu_moe.EPIdsContract.PADDING_MASK
    experts.moe_config = SimpleNamespace(ep_size=2, ep_rank=1)
    experts.apply(
        torch.zeros(1, 8),
        torch.zeros(1, 8),
        None,
        None,
        torch.ones(1, 2),
        torch.tensor([[0, 5]], dtype=torch.int32),
        None,
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert calls == [1]


def test_placement_trace(tmp_path):
    prefix = str(tmp_path / "trace")
    trace = xpu_moe.PlacementTrace(prefix, ep_rank=1)
    assert not trace.active()  # no <prefix>.on yet
    open(prefix + ".on", "w").close()
    assert trace.active()
    # 6 physical experts, this rank holds 3..5: slot 0 fast row 0, slots 1, 2
    # host rows 1, 0
    tiers = SimpleNamespace(
        layer_name="model.layers.7.mlp.experts",
        num_global=6,
        local_of={0: -1, 1: -1, 2: -1, 3: 0, 4: 1, 5: 2},
        fast_map=torch.tensor([-1, -1, -1, 0, -1, -1], dtype=torch.int32),
        slow_map=torch.tensor([-1, -1, -1, -1, 1, 0], dtype=torch.int32),
        slot_logical=torch.tensor([30, 40, 50]).numpy(),
    )
    ids = torch.tensor([[3, 1], [5, 4]], dtype=torch.int32)
    w = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    trace.record(tiers, ids, w)
    trace.record(tiers, ids, w)
    with open(prefix + ".rank1.jsonl") as f:
        lines = f.read().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[1])
    assert entry["layer"] == "model.layers.7.mlp.experts" and entry["call"] == 1
    assert entry["rank"] == 1 and entry["tokens"] == 2
    assert entry["physical"] == [[3, 1], [5, 4]]
    assert entry["where"] == [["F0", "-"], ["S0", "S1"]]
    assert entry["logical"] == [[30, -1], [50, 40]]
    assert entry["weight"] == [[0.75, 0.25], [0.5, 0.5]]
