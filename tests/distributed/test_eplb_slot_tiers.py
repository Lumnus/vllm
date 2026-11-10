# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB with expert slots in tiers (`TieredExpertWeight`).

A layer may keep its first local slots in device memory and the rest in host
memory (`--cpu-offload-per-expert`). EPLB addresses both storages as one row
per slot, and its tier step keeps each rank's most-loaded experts in the fast
slots.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
from vllm.distributed.eplb.expert_weights import (
    TieredExpertWeight,
    fast_slot_count,
    new_expert_buffer,
    order_slots_by_tier,
)
from vllm.distributed.eplb.rebalance_execute import move_from_buffer, move_to_buffer

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    expert_row,
    physical_rows_hold,
    single_rank_eplb,
)
from .test_eplb_int4_wna16 import _Mailbox, _random_map


def _tiered(rows: torch.Tensor, fast: int, written: list | None = None):
    callback = None if written is None else written.extend
    return TieredExpertWeight([rows[:fast].clone(), rows[fast:].clone()], callback)


def test_tiered_weight_addresses_slots_across_parts():
    rows = torch.arange(12.0).view(6, 2)
    written: list[int] = []
    w = _tiered(rows, 4, written)
    assert w.shape == (6, 2) and len(w) == 6 and w.tier_sizes == (4, 2)
    assert [w.tier_of(s) for s in range(6)] == [0, 0, 0, 0, 1, 1]
    for s in range(6):
        assert torch.equal(w[s], rows[s])
    w[5].copy_(torch.tensor([-1.0, -2.0]))
    assert torch.equal(w.parts[1][1], torch.tensor([-1.0, -2.0]))
    w.rows_written([1, 4, 5])  # only the slow ones are reported
    assert written == [4, 5]
    with pytest.raises(IndexError):
        w[6]
    u8 = TieredExpertWeight([torch.zeros(2, 3, dtype=torch.int8)] * 2).view(torch.uint8)
    assert u8.dtype == torch.uint8 and u8.element_size() == 1
    buf = new_expert_buffer(w)
    assert isinstance(buf, torch.Tensor) and buf.shape == (6, 2)
    assert fast_slot_count([torch.zeros(6, 2), w]) == 4
    assert fast_slot_count([torch.zeros(6, 2)]) is None


def test_tier_parts_must_agree():
    with pytest.raises(AssertionError):
        TieredExpertWeight([torch.zeros(2, 3), torch.zeros(2, 4)])
    with pytest.raises(AssertionError):
        TieredExpertWeight([torch.zeros(2, 3), torch.zeros(2, 3).bfloat16()])


def test_tier_step_keeps_the_most_loaded_experts_fast():
    # 2 ranks x 4 slots, 2 fast slots per rank; one layer
    phy2log = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    load = np.array([[1.0, 2.0, 9.0, 8.0, 5.0, 5.0, 1.0, 30.0]])
    out = order_slots_by_tier(phy2log, load, [2], num_ranks=2, hysteresis=1.1)
    # each rank keeps its own experts; its two hottest take its fast slots
    assert sorted(out[0, :4]) == [0, 1, 2, 3] and sorted(out[0, 4:]) == [4, 5, 6, 7]
    assert set(out[0, :2]) == {2, 3}
    assert 7 in out[0, 4:6] and {4, 5} & set(out[0, 4:6])
    # experts already in the right tier keep their slot
    again = order_slots_by_tier(out, load, [2], num_ranks=2, hysteresis=1.1)
    assert np.array_equal(again, out)


def test_tier_step_hysteresis_and_layers_without_tiers():
    phy2log = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    load = np.array([[10.0, 10.0, 10.5, 1.0], [1.0, 1.0, 50.0, 1.0]])
    out = order_slots_by_tier(phy2log, load, [2, None], num_ranks=1, hysteresis=1.1)
    # 10.5 is within 1.1x of 10: no move; layer 1 has no tiers
    assert np.array_equal(out, phy2log)
    out = order_slots_by_tier(phy2log, load, [2, 2], num_ranks=1, hysteresis=1.1)
    assert out[1, :2].tolist().count(2) == 1


def test_tier_step_counts_replicas_and_empty_slots():
    # expert 3 has two replicas: 6 each, ahead of expert 1's 5
    phy2log = np.array([[0, 1, 3, 3, -1, 2]])
    load = np.array([[4.0, 5.0, 1.0, 12.0]])
    out = order_slots_by_tier(phy2log, load, [2], num_ranks=1, hysteresis=1.0)
    assert out[0, :2].tolist() == [3, 3]
    assert sorted(out[0].tolist()) == sorted(phy2log[0].tolist())
    # an empty fast slot takes any loaded expert
    phy2log = np.array([[-1, 0, 1, 2]])
    out = order_slots_by_tier(phy2log, np.array([[1.0, 1.0, 7.0]]), [1], 1, 1.1)
    assert out[0, 0] == 2


def _tiered_model(num_layers, num_logical, fast, written):
    initial = list(range(num_logical))
    weights = [
        [
            _tiered(
                torch.stack([expert_row(layer, g, n) for g in initial]), fast, written
            )
            for n in (8, 4)
        ]
        for layer in range(num_layers)
    ]
    return FakeMoEModel(num_layers, num_logical, weights=weights)


def _window(state, logical_loads):
    """Record `logical_loads` in the window, which counts per physical slot."""
    model_state = next(iter(state.model_states.values()))
    phy2log = model_state.physical_to_logical_map
    loads = torch.tensor(logical_loads, dtype=torch.int32)
    model_state.expert_load_window.zero_()
    model_state.expert_load_window[0] = torch.gather(loads, 1, phy2log.long())
    return model_state


def test_one_rank_rearrangement_moves_hot_experts_into_fast_slots(monkeypatch):
    written: list[int] = []
    state = single_rank_eplb(monkeypatch)
    model = _tiered_model(num_layers=2, num_logical=8, fast=3, written=written)
    add_model(state, model)
    model_state = _window(
        state, [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]]
    )
    assert model_state.fast_slots == [3, 3]
    state.rearrange()
    phy2log = model_state.physical_to_logical_map.clone()
    assert set(phy2log[0, :3].tolist()) == {5, 6, 7}
    assert phy2log[1, :3].tolist() == [0, 1, 2]  # already the hottest
    assert physical_rows_hold(model, state)
    assert written and all(s >= 3 for s in written)  # slow slots were written
    # the same load again: nothing moves, nothing is written
    written.clear()
    _window(state, [[1, 1, 1, 0, 2, 40, 50, 60], [9, 8, 7, 1, 1, 1, 1, 1]])
    state.rearrange()
    assert model_state.physical_to_logical_map.tolist() == phy2log.tolist()
    assert written == []


def test_restart_with_saved_loads_puts_the_saved_hot_set_in_fast_slots(
    monkeypatch, tmp_path
):
    from vllm.distributed.eplb.eplb_state_file import write_expert_loads

    path = str(tmp_path / "eplb.json")
    write_expert_loads(path, {"model.layers.0.mlp.experts": [0, 0, 0, 0, 0, 7, 9, 8]})
    state = single_rank_eplb(monkeypatch, state_path=path)
    model = _tiered_model(num_layers=1, num_logical=8, fast=3, written=[])
    add_model(state, model)
    state.restore_saved_placement()
    phy2log = next(iter(state.model_states.values())).physical_to_logical_map
    assert set(phy2log[0, :3].tolist()) == {5, 6, 7}
    assert physical_rows_hold(model, state)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tiered_rows_follow_the_new_map_across_ranks(seed):
    """EPLB's transfer plan for 2 ranks whose slots are in two tiers each."""
    ep, local, fast, logical = 2, 5, 2, 7
    rng = np.random.default_rng(seed)
    old = _random_map(rng, logical, ep * local)
    new = _random_map(rng, logical, ep * local)

    def rank_weights(rank):
        slots = old[rank * local : (rank + 1) * local]
        return [
            _tiered(torch.stack([expert_row(0, int(g), n) for g in slots]), fast)
            for n in (6, 3)
        ]

    weights = [rank_weights(r) for r in range(ep)]
    buffers = [[new_expert_buffer(w) for w in ws] for ws in weights]
    box = _Mailbox()
    metas = [
        move_to_buffer(local, old, new, weights[r], buffers[r], None, r, box.rank(r))
        for r in range(ep)
    ]
    box.deliver()
    for r in range(ep):
        move_from_buffer(weights[r], buffers[r], metas[r], new, r)
        for s in range(local):
            g = int(new[r * local + s])
            for w in weights[r]:
                assert torch.equal(w[s], expert_row(0, g, w.shape[1])), (r, s)


@pytest.mark.parametrize("backend", ["nixl", "symm_mem"])
def test_registering_communicators_refuse_tiered_weights(backend):
    w = TieredExpertWeight([torch.zeros(2, 3), torch.zeros(1, 3)])
    group = SimpleNamespace(cpu_group=None, device_group=None)
    with pytest.raises(ValueError, match="tiered"):
        create_eplb_communicator(group, backend, [[w]], [torch.zeros(3, 3)])


def test_single_rank_eplb_needs_tiers():
    from vllm.config.vllm import VllmConfig

    def check(per_expert: bool, tp: int) -> None:
        fake = SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_eplb=True,
                tensor_parallel_size=tp,
                prefill_context_parallel_size=1,
                data_parallel_size=1,
            ),
            offload_config=SimpleNamespace(
                uva=SimpleNamespace(cpu_offload_per_expert=per_expert)
            ),
        )
        VllmConfig._validate_eplb_world(fake)  # type: ignore[arg-type]

    check(per_expert=True, tp=1)
    check(per_expert=False, tp=2)
    with pytest.raises(ValueError, match="cpu-offload-per-expert"):
        check(per_expert=False, tp=1)
