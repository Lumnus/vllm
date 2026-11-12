# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A host tower built from a real checkpoint: the model's own vision tower,
loaded alone in host memory, hands the workers exactly what its module
computes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler as sched_mod
import vllm.distributed.ec_transfer.ec_connector.host_tower.worker as worker_mod
from tests.v1.ec_connector.unit.host_tower.utils import FakeRegion
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.host_tower.scheduler import (
    ECHostTowerScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.host_tower.worker import (
    ECHostTowerWorker,
)
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.host_tower import TowerLocation, build_host_towers
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.utils.torch_utils import set_default_torch_num_threads

pytestmark = pytest.mark.cpu_test

MODEL = "Qwen/Qwen3.5-0.8B"
THREADS = 4
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
            limit_mm_per_prompt={"image": 2, "video": 0},
            mm_encoder_host_towers={"vision_tower": {"threads": THREADS}},
        )
    )


@pytest.fixture(scope="module")
def tower(engine_config):
    towers = build_host_towers(engine_config)
    yield towers["vision_tower"]
    for tower in towers.values():
        tower.shutdown()


@pytest.fixture(scope="module")
def features(engine_config) -> list[MultiModalFeatureSpec]:
    """Two images of different shapes, processed the way the frontend does."""
    processor = MULTIMODAL_REGISTRY.create_processor(engine_config.model_config)
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, size, dtype=np.uint8))
        for size in [(224, 320, 3), (448, 256, 3)]
    ]
    inputs = processor(
        IMAGE_PROMPT * len(images),
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )
    return [
        MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier=f"image-{i}",
            mm_position=position,
        )
        for i, (item, position) in enumerate(
            zip(inputs["mm_kwargs"]["image"], inputs["mm_placeholders"]["image"])
        )
    ]


def _direct(tower, feature) -> torch.Tensor:
    """The tower's module called directly, with the tower's thread count."""
    (_, _, mm_kwargs) = next(
        group_and_batch_mm_kwargs([(feature.modality, feature.data)])
    )
    with set_default_torch_num_threads(tower.threads), torch.inference_mode():
        (embedding,) = tower._module.embed_multimodal(**mm_kwargs)
    return embedding.to(tower.output_dtype)


def test_tower_is_on_the_host_alone_with_exact_weights(tower, engine_config):
    module = tower._module

    assert tower.location is TowerLocation.HOST
    assert tower.modalities == {"image", "video"}
    assert tower.threads == THREADS
    assert tower.output_dtype == engine_config.model_config.dtype
    assert isinstance(module.language_model, StageMissingLayer)

    names = [name for name, _ in module.named_parameters()]
    assert names and all(name.startswith("visual.") for name in names)
    assert all(p.device.type == "cpu" for p in module.parameters())
    assert all(p.dtype == torch.float32 for p in module.parameters())
    # float32 holds every bfloat16 value of the checkpoint exactly.
    weight = module.visual.patch_embed.proj.weight
    assert torch.equal(weight.to(torch.bfloat16).to(torch.float32), weight)


def test_submit_is_bit_equal_to_the_module(tower, features):
    for feature in features:
        result = tower.submit([(feature.modality, feature.data)]).result(timeout=600)
        (embedding,) = result.embeddings

        assert embedding.dtype == tower.output_dtype
        assert embedding.shape[0] == feature.mm_position.get_num_embeds()
        assert torch.isfinite(embedding.float()).all()
        assert torch.equal(embedding, _direct(tower, feature))


def test_workers_receive_the_towers_exact_output(
    tower, features, engine_config, monkeypatch
):
    model_config = engine_config.model_config
    block_bytes = model_config.get_inputs_embeds_size() * model_config.dtype.itemsize
    region = FakeRegion(4096, block_bytes)
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda _: region)
    monkeypatch.setattr(worker_mod, "create_ec_shared_region", lambda _: region)
    monkeypatch.setattr(tower, "shutdown", lambda: None)

    sched = ECHostTowerScheduler(engine_config, {tower.name: tower})
    worker = ECHostTowerWorker(engine_config)
    request = SimpleNamespace(request_id="r1", mm_features=features)

    assert sched.ensure_cache_available(request, 0) is False
    for feature in features:
        sched._encodes[feature.identifier].future.result(timeout=600)
    assert sched.ensure_cache_available(request, 0) is True

    for index in range(len(features)):
        sched.update_state_after_alloc(request, index)
    cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(
        cache, connector_metadata=sched.build_connector_meta(scheduler_output=None)
    )

    for feature in features:
        assert torch.equal(cache[feature.identifier], _direct(tower, feature))
    sched.shutdown()
