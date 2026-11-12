# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker's accelerator copy of a tower with residency "dynamic": staged,
put in place of the model's placeholder, taken out, and what is left after a
failure at each step. CPU stands in for the accelerator."""

import threading

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal.card_tower import (
    CardTower,
    checkpoint_tower_reader,
    module_bytes,
    tower_paths,
)

pytestmark = pytest.mark.cpu_test

TOWER = "vision_tower"


class Tower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


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


class TwinOfModel(nn.Module):
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


def _load(module: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(tensors, strict=True)


def _card_tower(model, *, read=None, load=_load, gate=None):
    def default_read():
        if gate is not None:
            assert gate.wait(timeout=60)
        return _weights()

    return CardTower(
        TOWER,
        model=model,
        device=torch.device("cpu"),
        build=TwinOfModel,
        read=read or default_read,
        load=load,
    )


def _wait_staged(tower: CardTower) -> dict:
    for _ in range(600):
        state = tower.poll()
        if state["staged"] or state["error"]:
            return state
        threading.Event().wait(0.01)
    raise AssertionError("never staged")


def test_the_placeholder_is_found():
    assert tower_paths(Model(), TOWER) == ["visual"]
    assert tower_paths(Model(), "audio_tower") == []


def test_stage_commit_release():
    model = Model()
    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="should not be called"):
        model.embed(x)

    tower = _card_tower(model)
    staged = tower.stage()
    assert staged == {"bytes": module_bytes(TwinOfModel()), "on_card": False}
    assert _wait_staged(tower) == {"staged": True, "error": None}
    # Still the placeholder until the commit.
    assert isinstance(model.visual, StageMissingLayer)

    committed = tower.commit()
    assert committed["bytes"] == staged["bytes"]
    assert tower.on_card
    weights = _weights()
    expected = torch.nn.functional.linear(
        x, weights["visual.proj.weight"], weights["visual.proj.bias"]
    )
    assert torch.equal(model.embed(x), expected)
    # Repeating the steps changes nothing.
    assert tower.stage()["on_card"] is True
    assert tower.commit()["seconds"] == 0.0

    assert tower.release() == {"bytes": staged["bytes"]}
    assert isinstance(model.visual, StageMissingLayer)
    assert not tower.on_card
    assert tower.release() == {"bytes": 0}
    # And it can come back.
    tower.stage()
    _wait_staged(tower)
    tower.commit()
    assert torch.equal(model.embed(x), expected)


def test_the_read_runs_beside_the_caller():
    gate = threading.Event()
    tower = _card_tower(Model(), gate=gate)
    tower.stage()  # returns while the read waits
    assert tower.poll() == {"staged": False, "error": None}
    gate.set()
    assert _wait_staged(tower)["staged"]


def test_a_failed_read_is_reported_and_the_commit_refuses():
    model = Model()

    def read():
        raise OSError(5, "read error")

    tower = _card_tower(model, read=read)
    tower.stage()
    state = _wait_staged(tower)
    assert state["staged"] is False and "read error" in state["error"]
    with pytest.raises(RuntimeError, match="reading its weights failed"):
        tower.commit()
    assert isinstance(model.visual, StageMissingLayer)
    assert not tower.staged


def test_a_failed_load_leaves_the_model_as_it_was():
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


def test_a_read_that_takes_too_long_fails_the_commit():
    gate = threading.Event()
    model = Model()
    tower = _card_tower(model, gate=gate)
    tower.stage()
    with pytest.raises(TimeoutError):
        tower.commit(wait_s=0.05)
    assert isinstance(model.visual, StageMissingLayer)
    gate.set()


def test_discard_drops_a_staged_copy():
    tower = _card_tower(Model())
    tower.stage()
    _wait_staged(tower)
    tower.discard()
    assert not tower.staged
    with pytest.raises(RuntimeError, match="nothing is staged"):
        tower.commit()


def test_a_model_without_a_placeholder_is_refused():
    model = Model()
    model.visual = Tower()  # built with the tower
    with pytest.raises(ValueError, match="no place"):
        _card_tower(model).stage()


def test_the_checkpoint_reader_reads_only_the_tower(tmp_path, monkeypatch):
    weights = _weights()
    other = {"model.language_model.weight": torch.ones(6, 6)}
    save_file(
        {f"model.{k}": v for k, v in weights.items()} | other,
        str(tmp_path / "model-00001.safetensors"),
    )

    class Mapper:
        def map_name(self, name):
            return name.removeprefix("model.")

    class Twin(TwinOfModel):
        hf_to_vllm_mapper = Mapper()

    class Loader:
        def _prepare_weights(self, *args, **kwargs):
            return (
                str(tmp_path),
                [str(tmp_path / "model-00001.safetensors")],
                True,
                None,
            )

    import vllm.model_executor.model_loader as model_loader
    import vllm.model_executor.model_loader.default_loader as default_loader

    monkeypatch.setattr(default_loader, "DefaultModelLoader", Loader)
    monkeypatch.setattr(model_loader, "get_model_loader", lambda _: Loader())

    from types import SimpleNamespace

    twin_config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), revision=None),
        load_config=None,
    )
    read = checkpoint_tower_reader(twin_config, Twin(), ["visual"])
    tensors = read()
    assert set(tensors) == {"model.visual.proj.weight", "model.visual.proj.bias"}
    assert torch.equal(
        tensors["model.visual.proj.weight"], weights["visual.proj.weight"]
    )
