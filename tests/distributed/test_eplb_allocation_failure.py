# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A rearrangement that cannot allocate is skipped on every rank, and the
engine keeps serving with the placement it had."""

import threading
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_mod

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)

_real_zeros = torch.zeros


def _fill_window(state, loads: list[list[int]]) -> None:
    """Loads of the first physical slots (the routed experts at the initial
    placement); the redundant slots carry none."""
    model_state = next(iter(state.model_states.values()))
    window = model_state.expert_load_window
    window.zero_()
    window[0, :, : len(loads[0])] = torch.tensor(loads, dtype=window.dtype)


def _oom_on_the_load_window(monkeypatch, state, ranks=None) -> list[int]:
    """`torch.zeros` fails for the logical load window (the rearrangement's
    first allocation, the one that failed on the card), on the threads named
    in ranks (all when None)."""
    hits: list[int] = []
    model_state = next(iter(state.model_states.values()))
    shape = (
        state.expert_load_window_size,
        model_state.model.num_moe_layers,
        model_state.model.num_logical_experts + 1,
    )

    def zeros(*size, **kwargs):
        dims = (
            tuple(size[0]) if len(size) == 1 and not isinstance(size[0], int) else size
        )
        name = threading.current_thread().name
        if dims == shape and (ranks is None or name in ranks):
            hits.append(1)
            raise torch.OutOfMemoryError(
                "XPU out of memory. Tried to allocate 58.00 MiB"
            )
        return _real_zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    return hits


@pytest.mark.parametrize("use_async", [False, True])
def test_an_allocation_failure_skips_the_rearrangement(monkeypatch, use_async):
    state = single_rank_eplb(monkeypatch, use_async=use_async)
    model = FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    _fill_window(state, [[1, 1, 50, 1], [1, 1, 1, 70]])
    before = model_state.physical_to_logical_map.clone()
    hits = _oom_on_the_load_window(monkeypatch, state)

    assert state.rearrange() is None  # no exception: skipped

    assert hits == [1]
    assert state.skipped_rearrangements == 1
    assert torch.equal(model_state.physical_to_logical_map, before)
    assert physical_rows_hold(model, state)
    assert not model_state.rebalanced
    assert not state.rearrange_event._recorded.is_set()

    # Memory back: the next rearrangement runs as before.
    monkeypatch.setattr(torch, "zeros", _real_zeros)
    state.rearrange()
    if use_async:
        assert model_state.rebalanced
        assert state.rearrange_event._recorded.is_set()
    else:
        assert not torch.equal(model_state.physical_to_logical_map, before)
        assert physical_rows_hold(model, state)
    assert state.skipped_rearrangements == 1


def test_an_allocation_failure_while_planning_skips_the_rearrangement(monkeypatch):
    state = single_rank_eplb(monkeypatch)
    model = FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    _fill_window(state, [[1, 1, 50, 1], [1, 1, 1, 70]])
    before = model_state.physical_to_logical_map.clone()

    def rebalance_experts(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(state.policy, "rebalance_experts", rebalance_experts)
    state.rearrange()
    assert state.skipped_rearrangements == 1
    assert torch.equal(model_state.physical_to_logical_map, before)
    assert physical_rows_hold(model, state)


def test_a_profile_run_still_fails_on_an_allocation_failure(monkeypatch):
    state = single_rank_eplb(monkeypatch)
    add_model(state, FakeMoEModel(num_layers=2, num_logical=4))
    _oom_on_the_load_window(monkeypatch, state)
    with pytest.raises(torch.OutOfMemoryError):
        state.rearrange(is_profile=True)


class _TwoRanks:
    """Two EP ranks in two threads: the host all-reduce sums over both; the
    device collective only records who entered it."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=10)
        self.slots: list[torch.Tensor | None] = [None, None]
        self.device_collectives: list[str] = []

    def group(self) -> SimpleNamespace:
        rank = int(threading.current_thread().name[-1])
        return SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: rank, size=lambda: 2),
            cpu_all_reduce_=lambda t: self._reduce(rank, t),
        )

    def _reduce(self, rank: int, tensor: torch.Tensor) -> None:
        self.slots[rank] = tensor.clone()
        self.barrier.wait()
        total = self.slots[0] + self.slots[1]  # type: ignore[operator]
        self.barrier.wait()
        tensor.copy_(total)

    def all_reduce(self, tensor, group=None) -> None:
        self.device_collectives.append(threading.current_thread().name)


def test_one_rank_failing_holds_every_rank(monkeypatch):
    """EP rank 0 cannot allocate, rank 1 can: neither enters the device
    all-reduce (rank 1 alone would wait in it for ever) and both keep their
    placement."""
    ranks = _TwoRanks()
    states = []
    for _ in range(2):
        state = single_rank_eplb(monkeypatch)
        add_model(state, FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2))
        _fill_window(state, [[1, 1, 50, 1], [1, 1, 1, 70]])
        states.append(state)
    monkeypatch.setattr(eplb_state_mod, "get_ep_group", ranks.group)
    monkeypatch.setattr(eplb_state_mod, "all_reduce", ranks.all_reduce)
    _oom_on_the_load_window(monkeypatch, states[0], ranks={"rank0"})
    before = [
        next(iter(s.model_states.values())).physical_to_logical_map.clone()
        for s in states
    ]
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            states[rank].rearrange()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=run, args=(r,), name=f"rank{r}") for r in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)

    assert errors == []
    assert ranks.device_collectives == []
    assert [s.skipped_rearrangements for s in states] == [1, 1]
    for state, old in zip(states, before):
        model_state = next(iter(state.model_states.values()))
        assert torch.equal(model_state.physical_to_logical_map, old)
