# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The incremental EPLB policy and the bounded transfer buffer it allows.

The policy improves the current placement by at most `max_moves_per_rank`
changed slots per rank and MoE layer, so a rearrangement moves a small part of
the weights and the transfer buffer only needs that many rows. The transfer
tests run EPLB's plan (`move_to_buffer` / `move_from_buffer`) for several EP
ranks in one process with buffers of that many rows.
"""

import numpy as np
import pytest
import torch

from tests.distributed.eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.policy import DefaultEplbPolicy, IncrementalEplbPolicy
from vllm.distributed.eplb.rebalance_execute import move_from_buffer, move_to_buffer


def _skewed_load(layers: int, experts: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.distributions.Pareto(1.0, 1.2).sample((layers, experts)) * 100).to(
        torch.int64
    )


def _rank_balance(load: torch.Tensor, phy2log: torch.Tensor, ranks: int) -> float:
    """Mean over layers of mean/max rank load (replicas split their expert's
    load evenly)."""
    out = []
    for layer in range(load.shape[0]):
        slots = phy2log[layer]
        count = torch.bincount(slots, minlength=load.shape[1]).double()
        per_slot = load[layer].double()[slots] / count[slots]
        per_rank = per_slot.reshape(ranks, -1).sum(1)
        out.append(float(per_rank.mean() / per_rank.max()))
    return float(np.mean(out))


def _changed_per_rank(old: torch.Tensor, new: torch.Tensor, ranks: int) -> int:
    return int((old != new).reshape(old.shape[0], ranks, -1).sum(-1).max())


def _assert_valid(phy2log: torch.Tensor, num_logical: int, old: torch.Tensor):
    for layer in range(phy2log.shape[0]):
        count = torch.bincount(phy2log[layer], minlength=num_logical)
        assert bool((count >= 1).all()), "every logical expert keeps a slot"
        assert count.sum() == old.shape[1]


@pytest.mark.parametrize(
    "ranks,experts,redundant", [(4, 64, 0), (4, 64, 8), (2, 16, 2)]
)
@pytest.mark.parametrize("max_moves", [1, 4])
def test_moves_stay_within_the_bound_and_lower_the_peak(
    ranks, experts, redundant, max_moves
):
    layers = 6
    load = _skewed_load(layers, experts, seed=ranks + experts + redundant)
    physical = experts + redundant
    old = DefaultEplbPolicy.rebalance_experts(
        torch.ones(layers, experts), physical, 1, 1, ranks, None
    )
    policy = IncrementalEplbPolicy.configured(max_moves)
    new = policy.rebalance_experts(load, physical, 1, 1, ranks, old)
    _assert_valid(new, experts, old)
    assert _changed_per_rank(old, new, ranks) <= max_moves
    assert _rank_balance(load, new, ranks) > _rank_balance(load, old, ranks)


def test_moves_far_fewer_experts_than_the_default_policy():
    layers, experts, ranks = 8, 128, 4
    load = _skewed_load(layers, experts, seed=7)
    ident = torch.arange(experts).repeat(layers, 1)
    default = DefaultEplbPolicy.rebalance_experts(load, experts, 1, 1, ranks, ident)
    # Same traffic a little later: the default policy places from scratch again.
    later = (load.double() * (1 + 0.05 * torch.randn(load.shape))).clamp_min(0)
    default_again = DefaultEplbPolicy.rebalance_experts(
        later.long(), experts, 1, 1, ranks, default
    )
    policy = IncrementalEplbPolicy.configured(4)
    incremental = policy.rebalance_experts(later.long(), experts, 1, 1, ranks, default)
    moved_default = int((default_again != default).sum())
    moved_incremental = int((incremental != default).sum())
    assert moved_default > layers * experts // 2
    assert moved_incremental <= layers * ranks * 4
    assert moved_incremental < moved_default // 10


def test_converges_towards_balance_over_rearrangements():
    layers, experts, ranks = 4, 64, 4
    load = _skewed_load(layers, experts, seed=3)
    phy2log = torch.arange(experts).repeat(layers, 1)
    policy = IncrementalEplbPolicy.configured(2)
    target = _rank_balance(
        load,
        DefaultEplbPolicy.rebalance_experts(load, experts, 1, 1, ranks, phy2log),
        ranks,
    )
    balances = [_rank_balance(load, phy2log, ranks)]
    for _ in range(12):
        phy2log = policy.rebalance_experts(load, experts, 1, 1, ranks, phy2log)
        balances.append(_rank_balance(load, phy2log, ranks))
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(balances, balances[1:]))
    assert balances[-1] >= target - 0.03


def test_balanced_placement_is_left_alone():
    layers, experts, ranks = 2, 16, 4
    load = torch.full((layers, experts), 10)
    old = torch.arange(experts).repeat(layers, 1)
    new = IncrementalEplbPolicy.rebalance_experts(load, experts, 1, 1, ranks, old)
    assert torch.equal(new, old)


def test_replicas_follow_the_hot_expert():
    experts, ranks, redundant = 8, 2, 2
    old = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 6, 7]])
    load = torch.tensor([[400, 10, 10, 10, 10, 10, 10, 10]])
    new = IncrementalEplbPolicy.configured(2).rebalance_experts(
        load, experts + redundant, 1, 1, ranks, old
    )
    count = torch.bincount(new[0], minlength=experts)
    assert count[0] >= 2
    assert bool((count >= 1).all())


def test_without_a_placement_it_places_like_the_default_policy():
    load = _skewed_load(2, 16, seed=1)
    assert torch.equal(
        IncrementalEplbPolicy.rebalance_experts(load, 16, 1, 1, 4, None),
        DefaultEplbPolicy.rebalance_experts(load, 16, 1, 1, 4, None),
    )


# Transfers with a bounded buffer, several ranks in one process.


class _Mailbox:
    def __init__(self) -> None:
        self.sent: dict[tuple[int, int, int], list[torch.Tensor]] = {}
        self.wanted: list[tuple[tuple[int, int, int], list[torch.Tensor]]] = []

    def deliver(self) -> None:
        for key, targets in self.wanted:
            for dst, src in zip(targets, self.sent.pop(key)):
                dst.copy_(src)
        assert not self.sent


class _RankComm:
    needs_profile_buffer_reservation = False
    needs_recv_rows = False

    def __init__(self, box: _Mailbox, rank: int) -> None:
        self.box, self.me = box, rank

    def set_transfer_context(self, *args) -> None:
        pass

    def add_send(self, tensors, dst_rank, expert_id=-1) -> None:
        self.box.sent[(self.me, dst_rank, expert_id)] = [t.clone() for t in tensors]

    def add_recv(self, tensors, src_rank, expert_id=-1) -> None:
        self.box.wanted.append(((src_rank, self.me, expert_id), list(tensors)))

    def execute(self) -> None:
        pass


def _row(logical: int, numel: int) -> torch.Tensor:
    return torch.full((numel,), float(logical))


def _transfer(old: np.ndarray, new: np.ndarray, ranks: int, buffer_rows: int):
    local = old.shape[0] // ranks
    numels = (6, 3)
    weights = [
        [
            torch.stack([_row(int(g), n) for g in old[r * local : (r + 1) * local]])
            for n in numels
        ]
        for r in range(ranks)
    ]
    buffers = [[w.new_empty((buffer_rows, w.shape[1])) for w in ws] for ws in weights]
    box = _Mailbox()
    metas = [
        move_to_buffer(
            num_local_experts=local,
            old_indices=old,
            new_indices=new,
            expert_weights=weights[r],
            expert_weights_buffers=buffers[r],
            stream=None,
            ep_rank=r,
            communicator=_RankComm(box, r),
        )
        for r in range(ranks)
    ]
    box.deliver()
    for r in range(ranks):
        move_from_buffer(weights[r], buffers[r], metas[r], new, r)
    return weights


@pytest.mark.parametrize("redundant", [0, 4])
def test_bounded_buffer_moves_every_expert_to_its_slot(redundant):
    ranks, experts, max_moves = 4, 32, 3
    physical = experts + redundant
    load = _skewed_load(1, experts, seed=11)
    old = DefaultEplbPolicy.rebalance_experts(
        torch.ones(1, experts), physical, 1, 1, ranks, None
    )
    new = IncrementalEplbPolicy.configured(max_moves).rebalance_experts(
        load, physical, 1, 1, ranks, old
    )
    assert not torch.equal(old, new)
    old_np, new_np = old[0].numpy(), new[0].numpy()
    weights = _transfer(old_np, new_np, ranks, buffer_rows=max_moves)
    local = physical // ranks
    for r in range(ranks):
        for row in range(local):
            logical = int(new_np[r * local + row])
            for w in weights[r]:
                assert torch.equal(w[row], _row(logical, w.shape[1]))


def test_bounded_buffer_refuses_a_bigger_move():
    ranks, experts = 2, 8
    old = np.arange(experts)
    new = np.array([4, 5, 6, 7, 0, 1, 2, 3])  # every slot changes rank
    with pytest.raises(RuntimeError, match="transfer buffer holds 2"):
        _transfer(old, new, ranks, buffer_rows=2)


# EplbState: buffer size, a real rearrangement, config.


def test_eplb_state_sizes_the_buffer_by_the_bound(monkeypatch):
    state = single_rank_eplb(monkeypatch, policy="incremental", max_moves_per_rank=2)
    model = FakeMoEModel(num_layers=2, num_logical=6, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    assert [b.shape[0] for b in model_state.expert_buffer] == [2, 2]
    # One rank: no move changes a rank's load, so nothing moves.
    model_state.expert_load_window.zero_()
    model_state.expert_load_window[:, :, 0] = 500
    model_state.expert_load_window[:, :, 1:] = 5
    state.rearrange()
    assert not state.last_rearrange_changed
    assert physical_rows_hold(model, state)


def test_default_policy_keeps_a_row_per_local_expert(monkeypatch):
    state = single_rank_eplb(monkeypatch)
    model = FakeMoEModel(num_layers=1, num_logical=6, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    assert [b.shape[0] for b in model_state.expert_buffer] == [8, 8]


def test_incremental_config():
    assert EPLBConfig(policy="incremental", use_async=True).max_moves_per_rank == 8
    a = EPLBConfig(policy="incremental", max_moves_per_rank=4).compile_factors()
    b = EPLBConfig(policy="incremental", max_moves_per_rank=16).compile_factors()
    assert a == b
    with pytest.raises(ValueError):
        EPLBConfig(policy="incremental", max_moves_per_rank=0)


@pytest.mark.parametrize("policy,calls", [("incremental", 3), ("default", 1)])
def test_restore_takes_rounds_until_nothing_moves(monkeypatch, policy, calls):
    state = single_rank_eplb(
        monkeypatch, policy=policy, max_moves_per_rank=2, state_path="unused"
    )
    monkeypatch.setattr(state, "_read_saved_load", lambda model_state: None)
    model = FakeMoEModel(num_layers=1, num_logical=8)
    add_model(state, model)
    next(iter(state.model_states.values())).has_saved_load = True
    made: list[dict] = []

    def rearrange(**kwargs):
        made.append(kwargs)
        state.last_rearrange_changed = len(made) < 3

    monkeypatch.setattr(state, "rearrange", rearrange)
    state.restore_saved_placement()
    assert len(made) == calls
    assert all(k == {"use_last_expert_load": True, "blocking": True} for k in made)
