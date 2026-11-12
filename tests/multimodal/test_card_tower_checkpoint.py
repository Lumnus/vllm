# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accelerator copy of a real checkpoint's vision tower, made the way a
worker makes it (the tower's tensors streamed alone from the checkpoint into
a tower built on the device), equals the tower the standard loader builds;
also in a process that already built the tower on the meta device, as a
worker whose model left the tower out has. CPU stands in for the
accelerator; the whole-replica layout."""

import threading

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY, card_tower
from vllm.multimodal.card_tower import make_card_tower
from vllm.multimodal.host_tower import (
    CheckpointTowerWeights,
    build_host_tower_module,
    make_host_tower_config,
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.utils.torch_utils import (
    set_default_torch_dtype,
    set_default_torch_num_threads,
)

pytestmark = pytest.mark.cpu_test

MODEL = "Qwen/Qwen3.5-0.8B"
TOWER = "vision_tower"
IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def engine_config() -> VllmConfig:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} is not in the local cache")
    return VllmConfig(
        model_config=ModelConfig(
            MODEL,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_encoder_host_towers={"vision_tower": {"threads": 4}},
        )
    )


@pytest.fixture
def rope_cache():
    """The process's rotary embedding cache, restored after the test."""
    saved = dict(rotary_embedding._ROPE_DICT)
    yield rotary_embedding._ROPE_DICT
    rotary_embedding._ROPE_DICT.clear()
    rotary_embedding._ROPE_DICT.update(saved)


class WorkerModel(nn.Module):
    """A worker's model as built with the tower kept off it."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = StageMissingLayer(TOWER)


def _image_kwargs(engine_config):
    processor = MULTIMODAL_REGISTRY.create_processor(engine_config.model_config)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (224, 320, 3), dtype=np.uint8))
    inputs = processor(
        IMAGE_PROMPT,
        mm_items=processor.info.parse_mm_data({"image": [image]}),
        hf_processor_mm_kwargs={},
    )
    item = inputs["mm_kwargs"]["image"][0]
    ((_, _, kwargs),) = list(group_and_batch_mm_kwargs([("image", item)]))
    return kwargs


def _commit(engine_config, model):
    card = make_card_tower(engine_config, TOWER, model, torch.device("cpu"))
    staged = card.stage()
    for _ in range(6000):
        state = card.poll()
        if state["staged"] or state["error"]:
            break
        threading.Event().wait(0.05)
    assert state == {"staged": True, "error": None}
    committed = card.commit()
    assert committed["bytes"] == staged["bytes"]
    return card


def _reference(engine_config):
    mm_config = engine_config.model_config.get_multimodal_config()
    twin_config = make_host_tower_config(
        engine_config,
        TOWER,
        mm_config.mm_encoder_host_towers[TOWER],
        for_card=True,
    )
    reference = build_host_tower_module(twin_config, host_forward=False)
    CheckpointTowerWeights(twin_config).load_into(reference)
    return reference


def test_the_streamed_copy_equals_the_loaders_tower(engine_config, rope_cache):
    model = WorkerModel()
    card = _commit(engine_config, model)
    assert not isinstance(model.visual, StageMissingLayer)
    reference = _reference(engine_config)

    ours = dict(model.visual.named_parameters())
    theirs = dict(reference.visual.named_parameters())
    assert ours.keys() == theirs.keys()
    for name, value in ours.items():
        assert value.dtype == engine_config.model_config.dtype, name
        assert torch.equal(value, theirs[name]), name

    kwargs = _image_kwargs(engine_config)
    with set_default_torch_num_threads(4), torch.inference_mode():
        got = card._module.embed_multimodal(**kwargs)
        want = reference.embed_multimodal(**kwargs)
    assert len(got) == len(want) == 1
    assert torch.equal(got[0], want[0])

    card.release()
    assert isinstance(model.visual, StageMissingLayer)


def test_a_process_that_built_the_tower_on_meta_still_gets_a_real_copy(
    engine_config, rope_cache
):
    """What a worker has done before the copy is made: its model's tower was
    constructed on the meta device, and `get_rope` cached that tower's rotary
    embedding (qwen3_vl.py, `Qwen3_VisionTransformer.__init__`), whose
    `cos_sin_cache` has no data."""
    vision = engine_config.model_config.hf_config.vision_config
    head_dim = vision.hidden_size // vision.num_heads
    with (
        set_current_vllm_config(engine_config),
        torch.device("meta"),
        set_default_torch_dtype(engine_config.model_config.dtype),
    ):
        cached = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
    assert cached.cos_sin_cache.is_meta

    model = WorkerModel()
    card = _commit(engine_config, model)

    assert card_tower.meta_tensors(model.visual) == []
    assert model.visual.rotary_pos_emb is not cached
    reference = _reference(engine_config)
    kwargs = _image_kwargs(engine_config)
    with set_default_torch_num_threads(4), torch.inference_mode():
        got = card._module.embed_multimodal(**kwargs)
        want = reference.embed_multimodal(**kwargs)
    assert torch.equal(got[0], want[0])
    card.release()
