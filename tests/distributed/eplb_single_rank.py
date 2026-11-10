# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A one-rank `EplbState` on CPU, for tests of EPLB's own logic.

The EP group, its all-reduce and the communicator are stand-ins for one rank,
so `EplbState.add_model`, `rearrange` and the real policy and transfer code
run in the test process. Every expert move is local.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_mod
from vllm.config import ParallelConfig
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.model_executor.models.interfaces import MixtureOfExperts


class LocalOnlyCommunicator:
    needs_profile_buffer_reservation = False
    needs_recv_rows = False
    owned_expert_buffer = None

    def set_transfer_context(self, *args) -> None:
        pass

    def add_send(self, *args, **kwargs) -> None:
        raise AssertionError("one rank: no remote sends")

    def add_recv(self, *args, **kwargs) -> None:
        raise AssertionError("one rank: no remote receives")

    def execute(self) -> None:
        pass


class _Event:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, *args) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, other) -> float:
        return 0.0


def expert_row(layer: int, logical: int, row_numel: int) -> torch.Tensor:
    """The bytes logical expert `logical` of `layer` holds in every tensor."""
    return torch.full((row_numel,), float(layer * 1000 + logical))


class FakeMoEModel:
    """The `MixtureOfExperts` surface `EplbState` reads, one EP rank."""

    set_eplb_state = MixtureOfExperts.set_eplb_state

    def __init__(
        self,
        num_layers: int,
        num_logical: int,
        num_redundant: int = 0,
        row_numels: tuple[int, ...] = (8, 4),
        weights: list[list] | None = None,
    ):
        self.num_moe_layers = num_layers
        self.num_logical_experts = num_logical
        self.num_routed_experts = num_logical
        self.num_redundant_experts = num_redundant
        self.num_physical_experts = num_logical + num_redundant
        self.num_local_physical_experts = self.num_physical_experts
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        if weights is None:
            initial = EplbState.build_initial_global_physical_to_logical_map(
                num_logical, num_redundant
            )
            weights = [
                [
                    torch.stack([expert_row(layer, g, n) for g in initial])
                    for n in row_numels
                ]
                for layer in range(num_layers)
            ]
        self.weights = weights
        self.moe_layers = [
            SimpleNamespace(
                layer_name=f"model.layers.{layer}.mlp.experts",
                get_expert_weights=(lambda ws=ws: ws),
                set_eplb_state=lambda **_: None,
            )
            for layer, ws in enumerate(weights)
        ]


def single_rank_eplb(
    monkeypatch: pytest.MonkeyPatch, rank: int = 0, **eplb_kwargs
) -> EplbState:
    group = SimpleNamespace(rank=lambda: rank, size=lambda: 1)
    monkeypatch.setattr(
        eplb_state_mod, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    monkeypatch.setattr(eplb_state_mod, "get_eplb_group", lambda: None)
    monkeypatch.setattr(eplb_state_mod, "get_node_count", lambda: 1)
    monkeypatch.setattr(eplb_state_mod, "all_reduce", lambda tensor, group=None: None)
    monkeypatch.setattr(
        eplb_state_mod,
        "create_eplb_communicator",
        lambda **kwargs: LocalOnlyCommunicator(),
    )
    monkeypatch.setattr(eplb_state_mod, "hash_config_cached", lambda config: "model")
    monkeypatch.setattr(torch, "Event", _Event)
    eplb_kwargs.setdefault("use_async", False)
    eplb_kwargs.setdefault("communicator", "torch_gloo")
    parallel_config = ParallelConfig(eplb_config=EPLBConfig(**eplb_kwargs))
    return EplbState(parallel_config, torch.device("cpu"))


def add_model(state: EplbState, model: FakeMoEModel) -> None:
    state.add_model(model, SimpleNamespace(model="fake"))  # type: ignore[arg-type]


def physical_rows_hold(model: FakeMoEModel, state: EplbState) -> bool:
    """Every physical slot holds the bytes of the logical expert the map
    says it holds."""
    phy2log = next(iter(state.model_states.values())).physical_to_logical_map
    for layer, tensors in enumerate(model.weights):
        for slot, logical in enumerate(phy2log[layer].tolist()):
            for t in tensors:
                if not torch.equal(t[slot], expert_row(layer, logical, t.shape[1])):
                    return False
    return True
