# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`EPLBConfig.decode_token_weight` / `prefill_token_weight`: whose routing
the expert placement serves."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.parallel import EPLBConfig
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from .eplb_single_rank import (
    FakeMoEModel,
    add_model,
    physical_rows_hold,
    single_rank_eplb,
)

CONFIG = SimpleNamespace(model="fake")


def test_weights_validate_and_stay_out_of_the_compile_hash():
    assert not EPLBConfig().weights_tokens
    assert EPLBConfig(decode_token_weight=4).weights_tokens
    assert EPLBConfig(prefill_token_weight=0).weights_tokens
    with pytest.raises(ValueError, match="both be 0"):
        EPLBConfig(decode_token_weight=0, prefill_token_weight=0)
    with pytest.raises(ValueError):
        EPLBConfig(decode_token_weight=-1)
    assert (
        EPLBConfig(decode_token_weight=3, prefill_token_weight=0.5).compile_factors()
        == EPLBConfig().compile_factors()
    )


@pytest.mark.parametrize(
    "decode,prefill,tokens,prefill_tokens,want",
    [
        (1, 1, 100, 60, 1.0),  # default: every step alike
        (4, 1, 8, 0, 4.0),  # a pure decode step
        (4, 1, 100, 100, 1.0),  # a pure prefill step
        (4, 1, 100, 50, 2.5),  # a mixed step weighs by its token mix
        (0, 1, 100, 25, 0.25),  # prefill only
        (4, 1, 0, 0, 0.0),  # an empty step counts nothing
        (4, 1, 100, None, 1.0),  # mix unknown: unweighted
    ],
)
def test_step_weight_is_the_token_weighted_mean(
    monkeypatch, decode, prefill, tokens, prefill_tokens, want
):
    state = single_rank_eplb(
        monkeypatch, decode_token_weight=decode, prefill_token_weight=prefill
    )
    assert state.step_token_weight(tokens, prefill_tokens) == pytest.approx(want)


def _serve(state, model_state, steps):
    """Run `steps` of (tokens, prefill tokens, load per physical slot)."""
    for tokens, prefill_tokens, load in steps:
        state.prepare_forward(
            CONFIG,
            tokens,
            num_prefill_tokens=prefill_tokens,  # type: ignore[arg-type]
        )
        row = model_state.expert_load_pass[0]
        row.zero_()
        row[: len(load)] = torch.tensor(load, dtype=torch.int32)
        state.step()


# A decode step routes 8 tokens to expert 3, a prefill step 1000 to expert 1.
STEPS = [(8, 0, [0, 0, 0, 8]), (1000, 1000, [0, 1000, 0, 0])]


@pytest.mark.parametrize(
    "decode,prefill,replicated",
    [
        (1, 1, 1),  # every token alike: the prefill-heavy expert gets a replica
        (200, 1, 3),  # decode priority: the decode expert does
        (1, 0, 3),  # decode only
        (0, 1, 1),  # prefill only
    ],
)
def test_the_weighting_decides_which_expert_is_replicated(
    monkeypatch, decode, prefill, replicated
):
    state = single_rank_eplb(
        monkeypatch, decode_token_weight=decode, prefill_token_weight=prefill
    )
    model = FakeMoEModel(num_layers=1, num_logical=4, num_redundant=1)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    _serve(state, model_state, STEPS)
    state.rearrange()
    phy2log = model_state.physical_to_logical_map[0].tolist()
    assert phy2log.count(replicated) == 2, phy2log
    assert physical_rows_hold(model, state)


def test_the_policy_and_the_state_file_see_the_weighted_load(monkeypatch, tmp_path):
    from vllm.distributed.eplb.eplb_state_file import read_expert_loads

    path = str(tmp_path / "eplb.json")
    state = single_rank_eplb(
        monkeypatch, decode_token_weight=10, prefill_token_weight=1, state_path=path
    )
    add_model(state, FakeMoEModel(num_layers=1, num_logical=4))
    model_state = next(iter(state.model_states.values()))
    seen = []
    real = state.policy.rebalance_experts

    def spy(weight, *args):
        seen.append(weight.clone())
        return real(weight, *args)

    monkeypatch.setattr(state.policy, "rebalance_experts", spy)
    _serve(state, model_state, STEPS)
    state.rearrange()
    assert seen[0][0].tolist() == pytest.approx([0, 1000, 0, 80])
    assert read_expert_loads(path) == {"model.layers.0.mlp.experts": [0, 1000, 0, 80]}


def test_the_runner_reports_the_prefill_tokens_of_the_batch(monkeypatch):
    controller = EPLBController(
        SimpleNamespace(
            enable_eplb=True, eplb_config=EPLBConfig(decode_token_weight=2)
        ),
        torch.device("cpu"),
    )
    got = []
    controller.state = SimpleNamespace(  # type: ignore[assignment]
        prepare_forward=lambda *args: got.append(args)
    )
    batch = SimpleNamespace(
        num_reqs=3,
        is_prefilling_np=np.array([False, True, True, False]),
        num_scheduled_tokens=np.array([1, 500, 200, 9]),
    )
    controller.prepare_forward(CONFIG, 701, None, input_batch=batch)  # type: ignore[arg-type]
    assert got[0][3] == 700
    # unweighted: the batch is not read
    controller.parallel_config.eplb_config = EPLBConfig()
    controller.prepare_forward(CONFIG, 701, None, input_batch=None)  # type: ignore[arg-type]
    assert got[1][3] is None
