# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker's accelerator copy of a tower with residency "dynamic": staged
without a copy of the weights, built and streamed in at commit, put in place
of the model's placeholder, taken out, and what is left after a failure at
each step. CPU stands in for the accelerator."""

import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal.card_tower import (
    CardTower,
    CheckpointTowerTensors,
    module_bytes,
    tower_paths,
)

pytestmark = pytest.mark.cpu_test

TOWER = "vision_tower"


class Tower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 6)
        # Built at construction, not in the checkpoint (like a rotary table).
        self.register_buffer("table", torch.arange(6.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.table


class Model(nn.Module):
    """A worker's model built without its tower: a placeholder in its place."""

    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            meta = Tower()
        self.visual = StageMissingLayer(TOWER, meta)
        self.language_model = nn.Linear(6, 6)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class Twin(nn.Module):
    """What the twin build gives: the tower under the model's own path."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = Tower()


def _weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "visual.proj.weight": torch.randn(6, 4, generator=generator),
        "visual.proj.bias": torch.randn(6, generator=generator),
    }


def _build(device: torch.device) -> nn.Module:
    with torch.device(device):
        return Twin()


def _stream(gate=None):
    def stream():
        for name, tensor in _weights().items():
            if gate is not None:
                assert gate.wait(timeout=60)
            yield name, tensor

    return stream


def _load(module: nn.Module, tensors) -> None:
    state = dict(tensors)
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(state[name])


def _sizes(module: nn.Module) -> dict[str, int]:
    return {n: t.numel() * t.element_size() for n, t in _weights().items()}


def _card_tower(model, *, build=_build, stream=None, load=_load, gate=None):
    return CardTower(
        TOWER,
        model=model,
        device=torch.device("cpu"),
        build=build,
        tensors=_sizes,
        stream=stream or _stream(gate),
        load=load,
    )


def _wait_staged(tower: CardTower) -> dict:
    for _ in range(600):
        state = tower.poll()
        if state["staged"] or state["error"]:
            return state
        threading.Event().wait(0.01)
    raise AssertionError("never staged")


def _expected(x):
    w = _weights()
    return torch.nn.functional.linear(
        x, w["visual.proj.weight"], w["visual.proj.bias"]
    ) + torch.arange(6.0)


def test_the_placeholder_is_found():
    assert tower_paths(Model(), TOWER) == ["visual"]
    assert tower_paths(Model(), "audio_tower") == []


def test_stage_holds_no_weights_and_commit_builds_on_the_device():
    model = Model()
    x = torch.randn(3, 4)
    tower = _card_tower(model)

    staged = tower.stage()
    assert staged == {
        "bytes": module_bytes(Twin()),
        "host_bytes": 6 * 4 * 4,  # the largest tensor, loaded one at a time
        "on_card": False,
    }
    assert tower._module is None  # nothing built in host memory
    assert _wait_staged(tower) == {"staged": True, "error": None}
    assert isinstance(model.visual, StageMissingLayer)

    committed = tower.commit()
    assert committed["bytes"] == staged["bytes"]
    assert tower.on_card
    assert torch.equal(model.embed(x), _expected(x))
    assert tower.stage()["on_card"] is True
    assert tower.commit()["seconds"] == 0.0

    assert tower.release() == {"bytes": staged["bytes"]}
    assert isinstance(model.visual, StageMissingLayer)
    assert tower.release() == {"bytes": 0}
    tower.stage()
    _wait_staged(tower)
    tower.commit()
    assert torch.equal(model.embed(x), _expected(x))


def test_only_the_tower_goes_in_not_the_rest_of_the_twin():
    """A tensor without data elsewhere in the twin (the language model's
    place) does not stop the copy: only the tower's subtree is used."""

    def build(device):
        twin = _build(device)
        twin.register_buffer("elsewhere", torch.empty(3, device="meta"))
        return twin

    model = Model()
    tower = _card_tower(model, build=build)
    tower.stage()
    _wait_staged(tower)
    tower.commit()
    assert not isinstance(model.visual, StageMissingLayer)


def test_a_tower_tensor_without_data_is_refused_before_it_goes_in():
    def build(device):
        twin = _build(device)
        twin.visual.table = torch.empty(6, device="meta")
        return twin

    model = Model()
    tower = _card_tower(model, build=build)
    tower.stage()
    _wait_staged(tower)
    with pytest.raises(ValueError, match="hold no data.*buffer table"):
        tower.commit()
    assert isinstance(model.visual, StageMissingLayer)
    assert tower.release() == {"bytes": 0}  # never raises after a failure
    tower.discard()


def test_the_warm_up_runs_beside_the_caller():
    gate = threading.Event()
    tower = _card_tower(Model(), gate=gate)
    tower.stage()
    assert tower.poll() == {"staged": False, "error": None}
    gate.set()
    assert _wait_staged(tower)["staged"]


def test_a_failed_read_is_reported_and_the_commit_refuses():
    model = Model()

    def stream():
        raise OSError(5, "read error")
        yield

    tower = _card_tower(model, stream=stream)
    tower.stage()
    state = _wait_staged(tower)
    assert state["staged"] is False and "read error" in state["error"]
    with pytest.raises(RuntimeError, match="reading its weights failed"):
        tower.commit()
    assert isinstance(model.visual, StageMissingLayer)
    assert not tower.staged


def test_a_failed_load_leaves_the_model_as_it_was_and_release_is_quiet():
    model = Model()

    def load(module, tensors):
        raise RuntimeError("bad tensor")

    tower = _card_tower(model, load=load)
    tower.stage()
    _wait_staged(tower)
    with pytest.raises(RuntimeError, match="bad tensor"):
        tower.commit()
    assert isinstance(model.visual, StageMissingLayer)
    assert not tower.on_card and not tower.staged
    assert tower._module is None
    assert tower.release() == {"bytes": 0}
    tower.discard()


def test_a_slow_warm_up_does_not_stop_the_commit():
    gate = threading.Event()
    model = Model()
    calls = []

    def stream():
        calls.append(1)
        if len(calls) == 1:  # the warm up waits; the commit's read does not
            assert gate.wait(timeout=60)
        yield from _weights().items()

    tower = _card_tower(model, stream=stream)
    tower.stage()
    tower.commit(wait_s=0.05)
    assert tower.on_card
    gate.set()


def test_discard_forgets_a_staging():
    tower = _card_tower(Model())
    tower.stage()
    _wait_staged(tower)
    tower.discard()
    assert not tower.staged
    with pytest.raises(RuntimeError, match="nothing is staged"):
        tower.commit()


def test_a_model_without_a_placeholder_is_refused():
    model = Model()
    model.visual = Tower()
    with pytest.raises(ValueError, match="no place"):
        _card_tower(model).stage()


def test_the_checkpoint_tensors_are_only_the_towers(tmp_path, monkeypatch):
    weights = _weights()
    other = {"model.language_model.weight": torch.ones(6, 6)}
    save_file(
        {f"model.{k}": v for k, v in weights.items()} | other,
        str(tmp_path / "model-00001.safetensors"),
    )

    class Mapper:
        def map_name(self, name):
            return name.removeprefix("model.")

    class MappedTwin(Twin):
        hf_to_vllm_mapper = Mapper()

    class Loader:
        def _prepare_weights(self, *args, **kwargs):
            files = [str(tmp_path / "model-00001.safetensors")]
            return str(tmp_path), files, True, None

    import vllm.model_executor.model_loader as model_loader
    import vllm.model_executor.model_loader.default_loader as default_loader

    monkeypatch.setattr(default_loader, "DefaultModelLoader", Loader)
    monkeypatch.setattr(model_loader, "get_model_loader", lambda _: Loader())

    twin_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), revision=None),
        load_config=None,
    )
    source = CheckpointTowerTensors(twin_config, ["visual"])
    sizes = source.index(MappedTwin())
    assert sizes == {"model.visual.proj.weight": 96, "model.visual.proj.bias": 24}
    streamed = dict(source.stream())
    assert set(streamed) == set(sizes)
    assert torch.equal(
        streamed["model.visual.proj.weight"], weights["visual.proj.weight"]
    )
