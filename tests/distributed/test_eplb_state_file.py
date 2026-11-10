# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`EPLBConfig.state_path`: the expert loads EPLB keeps across restarts."""

import json

import pytest
import torch

from vllm.distributed.eplb.eplb_state_file import read_expert_loads, write_expert_loads

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)

L0, L1 = "model.layers.0.mlp.experts", "model.layers.1.mlp.experts"


def test_round_trip_and_writers_merge_their_layers(tmp_path):
    path = str(tmp_path / "eplb.json")
    write_expert_loads(path, {L0: [3, 1, 2]})
    write_expert_loads(path, {L1: [0, 5, 0]})  # another pipeline stage
    write_expert_loads(path, {L0: [1, 1, 9]})  # a later rearrangement
    assert read_expert_loads(path) == {L0: [1, 1, 9], L1: [0, 5, 0]}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"a": 3}', ""])
def test_unreadable_state_reads_as_none(tmp_path, raw):
    path = tmp_path / "eplb.json"
    path.write_text(raw)
    assert read_expert_loads(str(path)) == {}
    assert read_expert_loads(str(tmp_path / "missing.json")) == {}


def _fill_window(state, loads: list[list[int]]) -> None:
    model_state = next(iter(state.model_states.values()))
    window = model_state.expert_load_window
    window.zero_()
    window[0] = torch.tensor(loads, dtype=window.dtype)


def test_rearrangement_writes_the_loads_it_used(monkeypatch, tmp_path):
    path = str(tmp_path / "eplb.json")
    state = single_rank_eplb(monkeypatch, state_path=path)
    model = FakeMoEModel(num_layers=2, num_logical=4)
    add_model(state, model)
    _fill_window(state, [[5, 1, 0, 2], [0, 0, 0, 0]])
    state.rearrange()
    # the idle layer keeps no entry: no traffic must not erase a saved state
    assert read_expert_loads(path) == {L0: [5, 1, 0, 2]}
    assert physical_rows_hold(model, state)


def test_a_restart_places_experts_by_the_saved_loads(monkeypatch, tmp_path):
    path = str(tmp_path / "eplb.json")
    # expert 2 of layer 0 and expert 3 of layer 1 were the hot ones
    write_expert_loads(path, {L0: [1, 1, 50, 1], L1: [1, 1, 1, 70]})
    state = single_rank_eplb(monkeypatch, state_path=path)
    model = FakeMoEModel(num_layers=2, num_logical=4, num_redundant=2)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    assert model_state.has_saved_load
    identity = model_state.physical_to_logical_map.clone()
    state.restore_saved_placement()
    phy2log = model_state.physical_to_logical_map
    assert not torch.equal(phy2log, identity)
    # the redundant slots went to the hot experts, and they hold their bytes
    assert phy2log[0].tolist().count(2) == 3
    assert phy2log[1].tolist().count(3) == 3
    assert physical_rows_hold(model, state)
    assert not model_state.has_saved_load
    # the file is only read: a restore writes nothing back
    assert json.loads((tmp_path / "eplb.json").read_text()) == {
        L0: [1, 1, 50, 1],
        L1: [1, 1, 1, 70],
    }


def test_no_saved_loads_no_restore(monkeypatch, tmp_path):
    state = single_rank_eplb(monkeypatch, state_path=str(tmp_path / "none.json"))
    model = FakeMoEModel(num_layers=1, num_logical=4, num_redundant=2)
    add_model(state, model)
    calls = []
    monkeypatch.setattr(state, "rearrange", lambda **kw: calls.append(kw))
    state.restore_saved_placement()
    assert calls == []


def test_mismatched_saved_layer_is_not_used(monkeypatch, tmp_path):
    path = str(tmp_path / "eplb.json")
    write_expert_loads(path, {L0: [1, 2, 3]})  # 3 loads for 4 experts
    state = single_rank_eplb(monkeypatch, state_path=path)
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    assert not next(iter(state.model_states.values())).has_saved_load


def test_restore_follows_ep_rank_0(monkeypatch, tmp_path):
    """Every rank reads the file, but only rank 0's reading decides, so the
    ranks cannot disagree on whether to enter the collective rearrangement."""
    path = str(tmp_path / "eplb.json")
    write_expert_loads(path, {L0: [1, 1, 50, 1]})
    state = single_rank_eplb(monkeypatch, rank=1, state_path=path)
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4, num_redundant=2))
    calls = []
    monkeypatch.setattr(state, "rearrange", lambda **kw: calls.append(kw))
    state.restore_saved_placement()  # all_reduce is a no-op: rank 0 said 0
    assert calls == []


def test_state_path_is_not_a_compile_factor(tmp_path):
    from vllm.config.parallel import EPLBConfig

    a = EPLBConfig(state_path=str(tmp_path / "a.json")).compile_factors()
    b = EPLBConfig().compile_factors()
    assert a == b
