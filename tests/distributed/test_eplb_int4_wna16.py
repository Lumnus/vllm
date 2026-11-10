# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB on int4 (W4A16 / W4A8 weight set) MoE experts.

After loading, the int4 GPTQ/AutoRound experts on the XPU and Triton backends
are expert-major: packed int4 weights `[E, N, K / 2]` and group scales
`[E, N, K / group]`, one row per local expert. EPLB moves them as rows. These
tests run EPLB's transfer plan (`move_to_buffer` / `move_from_buffer`) for
several EP ranks in one process, with a communicator that delivers the
point-to-point messages between the ranks, and check that every physical slot
ends up with its logical expert's bytes and that a reference int4 MoE computes
the same output before and after.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.rebalance_execute import move_from_buffer, move_to_buffer
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    wna16_backend_supports_eplb,
)
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)

N, K, G = 16, 32, 8  # intermediate, hidden, group size


@pytest.mark.parametrize(
    "backend,expected",
    [
        (WNA16MoEBackend.XPU, True),
        (WNA16MoEBackend.TRITON, True),
        (WNA16MoEBackend.MARLIN, False),
        (WNA16MoEBackend.CPU, False),
        (WNA16MoEBackend.HUMMING, False),
    ],
)
def test_supports_eplb_follows_the_backend(backend, expected):
    assert wna16_backend_supports_eplb(backend) is expected
    gptq = object.__new__(AutoGPTQMoEMethod)
    gptq.wna16_moe_backend = backend
    assert gptq.supports_eplb is expected
    ct = object.__new__(CompressedTensorsWNA16MoEMethod)
    ct.wna16_backend = backend
    assert ct.supports_eplb is expected


class _Mailbox:
    """Point-to-point messages between EP ranks simulated in one process.

    Sends snapshot their rows when posted; every rank posts before any rank
    copies out of its buffer, as with a real communicator."""

    def __init__(self) -> None:
        self.sent: dict[tuple[int, int, int], list[torch.Tensor]] = {}
        self.wanted: list[tuple[tuple[int, int, int], list[torch.Tensor]]] = []

    def rank(self, rank: int) -> "_RankComm":
        return _RankComm(self, rank)

    def deliver(self) -> None:
        for key, targets in self.wanted:
            rows = self.sent.pop(key)
            for dst, src in zip(targets, rows):
                dst.copy_(src)
        assert not self.sent, f"sent but never received: {sorted(self.sent)}"


class _RankComm:
    needs_profile_buffer_reservation = False
    needs_recv_rows = False

    def __init__(self, box: _Mailbox, rank: int) -> None:
        self.box, self.me = box, rank

    def set_transfer_context(self, *args) -> None:
        pass

    def add_send(self, tensors, dst_rank, expert_id=-1) -> None:
        key = (self.me, dst_rank, expert_id)
        assert key not in self.box.sent
        self.box.sent[key] = [t.clone() for t in tensors]

    def add_recv(self, tensors, src_rank, expert_id=-1) -> None:
        self.box.wanted.append(((src_rank, self.me, expert_id), list(tensors)))

    def execute(self) -> None:
        pass


def _expert(logical: int) -> dict[str, torch.Tensor]:
    """One logical expert's int4 weight set, as XPU holds it after loading."""
    gen = torch.Generator().manual_seed(1000 + logical)
    return {
        "w13": torch.randint(-128, 128, (2 * N, K // 2), generator=gen).to(torch.int8),
        "w2": torch.randint(-128, 128, (K, N // 2), generator=gen).to(torch.int8),
        "w13_scales": (torch.rand(2 * N, K // G, generator=gen) / 8).bfloat16(),
        "w2_scales": (torch.rand(K, N // G, generator=gen) / 8).bfloat16(),
    }


_NAMES = ("w13", "w2", "w13_scales", "w2_scales")


def _rank_tensors(phy2log: np.ndarray, rank: int, local: int) -> list[torch.Tensor]:
    """A rank's four expert tensors for the slots of `phy2log`."""
    slots = phy2log[rank * local : (rank + 1) * local]
    experts = [_expert(int(g)) for g in slots]
    return [torch.stack([e[name] for e in experts]).contiguous() for name in _NAMES]


def _eplb_views(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """What `MixtureOfExperts.set_eplb_state` registers: `[E, -1]` views, 1-byte
    dtypes as uint8."""
    views = [t.view(t.shape[0], -1) for t in tensors]
    return [v.view(torch.uint8) if v.element_size() == 1 else v for v in views]


def _rearrange(
    old: np.ndarray, new: np.ndarray, ranks: list[list[torch.Tensor]], local: int
) -> None:
    box = _Mailbox()
    weights = [_eplb_views(t) for t in ranks]
    buffers = [[torch.empty_like(w) for w in ws] for ws in weights]
    metas = [
        move_to_buffer(
            num_local_experts=local,
            old_indices=old,
            new_indices=new,
            expert_weights=weights[r],
            expert_weights_buffers=buffers[r],
            stream=None,
            ep_rank=r,
            communicator=box.rank(r),
        )
        for r in range(len(ranks))
    ]
    box.deliver()
    for r in range(len(ranks)):
        move_from_buffer(weights[r], buffers[r], metas[r], new, r)


def _random_map(rng: np.random.Generator, logical: int, physical: int) -> np.ndarray:
    """Every logical expert at least once, the rest as replicas, shuffled."""
    slots = np.concatenate(
        [np.arange(logical), rng.integers(0, logical, physical - logical)]
    )
    return rng.permutation(slots).astype(np.int64)


def _dequant(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Signed int4 pairs `[rows, cols / 2]` (low nibble first) times group scales."""
    lo = (q.to(torch.int16) << 12 >> 12).float()
    hi = (q.to(torch.int16) << 8 >> 12).float()
    w = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    return w * scales.float().repeat_interleave(G, dim=-1)


def _moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    expert_of: Callable[[int], list[torch.Tensor]],
) -> torch.Tensor:
    """Reference SiLU-gated MoE; `expert_of(logical)` returns its four tensors."""
    out = torch.zeros(x.shape[0], K)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            w13, w2, s13, s2 = expert_of(int(topk_ids[t, j]))
            gate_up = _dequant(w13, s13) @ x[t]
            act = torch.nn.functional.silu(gate_up[:N]) * gate_up[N:]
            out[t] += topk_w[t, j] * (_dequant(w2, s2) @ act)
    return out


@pytest.mark.parametrize("ep,local,logical", [(2, 4, 6), (4, 3, 9), (4, 4, 16)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_int4_rows_and_scales_follow_the_new_map(ep, local, logical, seed):
    rng = np.random.default_rng(seed)
    physical = ep * local
    old = _random_map(rng, logical, physical)
    new = _random_map(rng, logical, physical)
    ranks = [_rank_tensors(old, r, local) for r in range(ep)]
    _rearrange(old, new, ranks, local)
    for r in range(ep):
        for name, tensor in zip(_NAMES, ranks[r]):
            for s in range(local):
                want = _expert(int(new[r * local + s]))[name]
                assert torch.equal(tensor[s], want), (r, s, name)


@pytest.mark.parametrize("seed", [0, 1])
def test_reference_int4_moe_output_is_unchanged(seed):
    ep, local, logical = 2, 4, 6
    rng = np.random.default_rng(seed)
    old = _random_map(rng, logical, ep * local)
    new = _random_map(rng, logical, ep * local)
    ranks = [_rank_tensors(old, r, local) for r in range(ep)]

    def through(phy2log: np.ndarray):
        def expert_of(g: int):
            p = int(np.flatnonzero(phy2log == g)[0])  # the router's first replica
            return [t[p % local] for t in ranks[p // local]]

        return expert_of

    torch.manual_seed(seed)
    x = torch.randn(5, K)
    topk_ids = torch.randint(0, logical, (5, 2))
    topk_w = torch.rand(5, 2)
    before = _moe(x, topk_ids, topk_w, through(old))
    _rearrange(old, new, ranks, local)
    after = _moe(x, topk_ids, topk_w, through(new))
    assert torch.equal(before, after)


def test_xpu_int4_experts_register_four_expert_major_tensors():
    """The tensors EPLB registers for an XPU int4 layer, after the kernels'
    relayout: packed weights as uint8 views, bf16 group scales."""
    ws = _eplb_views(_rank_tensors(np.arange(4), 0, 4))
    assert [w.dtype for w in ws] == [torch.uint8, torch.uint8] + [torch.bfloat16] * 2
    assert all(w.is_contiguous() and w.shape[0] == 4 for w in ws)
    row_bytes = sum(w[0].numel() * w.element_size() for w in ws)
    assert row_bytes == 2 * N * K // 2 + K * N // 2 + 2 * (2 * N * K // G + K * N // G)
