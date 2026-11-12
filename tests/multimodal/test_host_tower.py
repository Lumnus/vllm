# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host towers: the residency contract of `HostTower`, its own thread, and how
a model is built with a tower on the host and without it on the workers."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.model_loader.utils as loader_utils
from tests.v1.ec_connector.unit.host_tower.utils import (
    IN_DIM,
    OUT_DIM,
    NoWeights,
    ToyTowerModule,
    direct_embedding,
    make_item,
    make_tower,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.config.multimodal import HostTowerOptions, MultiModalConfig
from vllm.distributed.parallel_state import single_rank_model_parallel
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    get_host_tower_modalities,
)
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.multimodal.host_tower import (
    HostTower,
    TowerLocation,
    _use_host_forward,
    build_host_tower_module,
    make_host_tower_config,
    resolve_host_tower_threads,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.cpu_test

MODEL = "llava-hf/llava-1.5-7b-hf"


# ----------------------------------------------------------------------
# The residency contract
# ----------------------------------------------------------------------


def test_location_follows_the_tower():
    tower = make_tower()
    assert tower.location is TowerLocation.DISK
    assert tower.nbytes == 0
    assert tower.modalities == frozenset()

    tower.to_host()
    assert tower.location is TowerLocation.HOST
    assert tower.nbytes == (OUT_DIM * IN_DIM + OUT_DIM) * 4
    assert tower.modalities == {"image"}
    tower.shutdown()


def test_to_host_is_idempotent_from_any_thread():
    builds = []
    weights = NoWeights()

    def build() -> nn.Module:
        builds.append(threading.current_thread().name)
        return ToyTowerModule()

    tower = HostTower(
        "vision_tower",
        build=build,
        weights=weights,
        threads=1,
        output_dtype=torch.float16,
    )
    threads = [threading.Thread(target=tower.to_host) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    tower.to_host()

    assert len(builds) == 1
    assert weights.loads == 1
    assert tower.location is TowerLocation.HOST
    tower.shutdown()


def test_submit_brings_the_tower_to_the_host_by_itself():
    tower = make_tower()
    item = make_item("a", 4)

    result = tower.submit([("image", item)]).result(timeout=60)

    assert tower.location is TowerLocation.HOST
    assert torch.equal(result.embeddings[0], direct_embedding(tower, item))
    tower.shutdown()


@pytest.mark.parametrize("move", ["to_card"])
def test_a_move_that_is_not_built_leaves_the_tower_serving(move):
    # The accelerator copy is the workers' (vllm/multimodal/card_tower.py);
    # the host copy does not move there itself.
    tower = make_tower()
    tower.to_host()
    module = tower._module

    with pytest.raises(NotImplementedError):
        getattr(tower, move)()

    assert tower.location is TowerLocation.HOST
    assert tower._module is module
    item = make_item("a", 4)
    result = tower.submit([("image", item)]).result(timeout=60)
    assert torch.equal(result.embeddings[0], direct_embedding(tower, item))
    tower.shutdown()


def test_release_drops_the_host_copy_and_the_next_encode_brings_it_back():
    tower = make_tower()
    tower.to_host()
    nbytes = tower.nbytes
    assert tower.release(wait=False).result(timeout=60) == nbytes
    assert tower.location is TowerLocation.DISK
    assert tower._module is None
    item = make_item("a", 4)
    result = tower.submit([("image", item)]).result(timeout=60)
    assert torch.equal(result.embeddings[0], direct_embedding(tower, item))
    assert tower.location is TowerLocation.HOST
    tower.shutdown()


def test_a_weights_source_need_not_release():
    with pytest.raises(NotImplementedError):
        NoWeights().release(ToyTowerModule())


# ----------------------------------------------------------------------
# The tower's own flow
# ----------------------------------------------------------------------


def test_output_is_what_the_module_computes_in_the_output_dtype():
    tower = make_tower(output_dtype=torch.bfloat16)
    items = [make_item("a", 3), make_item("b", 7)]

    result = tower.submit([("image", item) for item in items]).result(timeout=60)

    assert [e.shape for e in result.embeddings] == [(3, OUT_DIM), (7, OUT_DIM)]
    for embedding, item in zip(result.embeddings, items):
        assert embedding.dtype == torch.bfloat16
        assert embedding.is_contiguous()
        assert torch.equal(embedding, direct_embedding(tower, item, torch.bfloat16))
    assert result.queue_seconds >= 0
    assert result.encode_seconds >= 0
    tower.shutdown()


def test_encodes_run_with_the_tower_threads_and_leave_the_caller_alone():
    before = torch.get_num_threads()
    tower = make_tower(threads=3)

    tower.submit([("image", make_item("a", 4))]).result(timeout=60)

    assert tower._module.threads_seen == [3]
    assert torch.get_num_threads() == before
    tower.shutdown()


def test_encodes_run_one_at_a_time_in_submission_order():
    gate = threading.Event()
    tower = make_tower(gate)
    tower.to_host()

    first = tower.submit([("image", make_item("a", 4))])
    second = tower.submit([("image", make_item("b", 4))])
    third = tower.submit([("image", make_item("c", 4))])
    assert tower._module.started.wait(timeout=60)
    assert not first.done()

    # Waiting behind the first: can still be called off.
    assert second.cancel()
    gate.set()
    first.result(timeout=60)
    result = third.result(timeout=60)

    assert result.queue_seconds > 0
    assert tower.num_encodes == 2
    tower.shutdown()


def test_default_thread_count_is_half_of_the_usable_cpus():
    assert resolve_host_tower_threads(HostTowerOptions(threads=5)) == 5
    assert resolve_host_tower_threads(HostTowerOptions()) >= 1


# ----------------------------------------------------------------------
# Building a model around a host tower
# ----------------------------------------------------------------------


class ToyVLM(nn.Module, SupportsMultiModal):
    """A model with two towers and a language model."""

    supports_host_tower = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        mm_config = vllm_config.model_config.get_multimodal_config()
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = nn.Linear(IN_DIM, OUT_DIM)
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio = nn.Linear(IN_DIM, OUT_DIM)

        if mm_config.mm_encoder_host_twin is not None:
            self.language_model = self._omit_language_model("language_model")
            return
        with self._mark_language_model(vllm_config):
            self.language_model = nn.Linear(4, 4)


class InheritingVLM(ToyVLM):
    """Inherits the opt-in flag, which says nothing about its constructor."""


def _toy_config(**mm_kwargs):
    mm_config = MultiModalConfig(**mm_kwargs)
    return SimpleNamespace(
        model_config=SimpleNamespace(get_multimodal_config=lambda: mm_config)
    )


def _build_toy(**mm_kwargs) -> ToyVLM:
    with single_rank_model_parallel():
        return ToyVLM(vllm_config=_toy_config(**mm_kwargs))


def test_model_is_built_whole_without_host_towers():
    model = _build_toy()

    assert isinstance(model.visual, nn.Linear)
    assert isinstance(model.audio, nn.Linear)
    assert isinstance(model.language_model, nn.Linear)
    assert get_host_tower_modalities(model, {}) == frozenset()


def test_workers_leave_out_the_tower_that_is_kept_on_the_host():
    model = _build_toy(mm_encoder_host_towers={"vision_tower": 2})

    assert isinstance(model.visual, StageMissingLayer)
    assert isinstance(model.audio, nn.Linear)
    assert isinstance(model.language_model, nn.Linear)
    assert {name for name, _ in model.named_parameters()} == {
        "audio.weight",
        "audio.bias",
        "language_model.weight",
        "language_model.bias",
    }
    assert model._tower_modalities == {
        "vision_tower": {"image", "video"},
        "audio_tower": {"audio"},
    }
    assert get_host_tower_modalities(model, {"vision_tower"}) == {"image", "video"}


def test_host_copy_holds_its_tower_and_nothing_else():
    model = _build_toy(
        mm_encoder_host_towers={"audio_tower": 2}, mm_encoder_host_twin="audio_tower"
    )

    assert isinstance(model.visual, StageMissingLayer)
    assert isinstance(model.audio, nn.Linear)
    assert isinstance(model.language_model, StageMissingLayer)
    assert model._language_model_names == ["language_model"]
    assert {name for name, _ in model.named_parameters()} == {
        "audio.weight",
        "audio.bias",
    }


def test_host_copy_of_a_disabled_tower_is_empty():
    model = _build_toy(
        mm_encoder_host_twin="audio_tower", limit_per_prompt={"audio": 0}
    )
    assert isinstance(model.audio, StageMissingLayer)


def test_parallel_groups_are_restored_after_a_single_rank_build():
    import vllm.distributed.parallel_state as parallel_state

    def groups():
        return parallel_state._TP, parallel_state._PP

    before = groups()
    with single_rank_model_parallel():
        assert parallel_state.get_tensor_model_parallel_world_size() == 1
        assert parallel_state.get_tensor_model_parallel_rank() == 0
        assert parallel_state.get_pp_group().is_first_rank
    assert groups() == before

    with pytest.raises(RuntimeError), single_rank_model_parallel():
        raise RuntimeError
    assert groups() == before


@pytest.fixture(scope="module")
def engine_config() -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            MODEL,
            mm_encoder_host_towers={"vision_tower": {"threads": 3}},
        )
    )


def test_host_tower_config_is_a_private_copy(engine_config):
    mm_config = engine_config.model_config.get_multimodal_config()
    options = mm_config.mm_encoder_host_towers["vision_tower"]

    twin = make_host_tower_config(engine_config, "vision_tower", options)
    twin_mm = twin.model_config.get_multimodal_config()

    assert twin_mm.mm_encoder_host_twin == "vision_tower"
    assert twin_mm.mm_encoder_only is True
    assert twin.model_config.dtype == torch.float32
    assert twin.device_config.device_type == "cpu"
    assert twin.parallel_config.tensor_parallel_size == 1
    assert twin.ec_transfer_config is None

    # The engine's own config is as it was.
    assert mm_config.mm_encoder_host_twin is None
    assert mm_config.mm_encoder_only is False
    assert engine_config.model_config.dtype != torch.float32
    assert engine_config.ec_transfer_config is not None
    assert twin.compilation_config is not engine_config.compilation_config


def test_host_tower_config_can_keep_the_model_dtype(engine_config):
    twin = make_host_tower_config(
        engine_config, "vision_tower", HostTowerOptions(dtype="model")
    )
    assert twin.model_config.dtype == engine_config.model_config.dtype


def _twin_config(engine_config, tower_name="vision_tower"):
    return make_host_tower_config(engine_config, tower_name, HostTowerOptions())


def test_host_tower_module_is_the_models_own_class(engine_config, monkeypatch):
    monkeypatch.setattr(
        loader_utils, "get_model_architecture", lambda _: (ToyVLM, "ToyVLM")
    )

    module = build_host_tower_module(_twin_config(engine_config))

    assert isinstance(module, ToyVLM)
    assert isinstance(module.visual, nn.Linear)
    assert module.visual.weight.dtype == torch.float32
    assert module.visual.weight.device.type == "cpu"
    assert isinstance(module.audio, StageMissingLayer)
    assert isinstance(module.language_model, StageMissingLayer)


def test_model_without_the_opt_in_is_refused(engine_config, monkeypatch):
    monkeypatch.setattr(
        loader_utils,
        "get_model_architecture",
        lambda _: (InheritingVLM, "InheritingVLM"),
    )
    with pytest.raises(ValueError, match="does not support"):
        build_host_tower_module(_twin_config(engine_config))


def test_unknown_tower_name_is_refused(engine_config, monkeypatch):
    monkeypatch.setattr(
        loader_utils, "get_model_architecture", lambda _: (ToyVLM, "ToyVLM")
    )
    with pytest.raises(ValueError, match="no tower named 'speech_tower'"):
        build_host_tower_module(_twin_config(engine_config, "speech_tower"))


def test_custom_ops_of_a_host_tower_take_the_cpu_path(default_vllm_config):
    from vllm.model_executor.layers.activation import SiluAndMul

    op = SiluAndMul()
    op._forward_method = op.forward_cuda
    module = nn.Sequential(nn.Linear(4, 4), op)

    _use_host_forward(module)

    assert op._forward_method == op.forward_cpu
    x = torch.randn(2, 8)
    assert torch.equal(module[1](x), op.forward_native(x))


# ----------------------------------------------------------------------
# The workers' memory profile
# ----------------------------------------------------------------------


class _Budget:
    mm_max_toks_per_item = {"image": 6}

    def get_encoder_budget(self) -> int:
        return 64


def _encoder_runner(host_modalities=()) -> tuple[EncoderRunner, list]:
    calls: list = []

    class _Model:
        def embed_multimodal(self, **kwargs):
            calls.append(kwargs)
            return [torch.ones(6, OUT_DIM, dtype=torch.float16)]

    runner = EncoderRunner(
        model=_Model(),  # type: ignore[arg-type]
        max_num_tokens=16,
        hidden_size=OUT_DIM,
        encoder_cache=EncoderCache(),
        dtype=torch.float16,
        device=torch.device("cpu"),
        host_modalities=host_modalities,
    )
    return runner, calls


def test_profile_reserves_the_encoder_cache_without_running_a_host_tower():
    runner, calls = _encoder_runner(host_modalities={"image"})
    dummy = [("image", make_item("a", 6))] * 3

    runner.profile_encoder_cache(dummy, _Budget())  # type: ignore[arg-type]

    assert calls == []
    outputs = runner.encoder_cache.encoder_outputs
    assert sorted(outputs) == ["tmp_0", "tmp_1", "tmp_2"]
    for output in outputs.values():
        assert output.shape == (6, OUT_DIM)
        assert output.dtype == torch.float16


def test_profile_runs_the_tower_that_is_on_the_device():
    runner, calls = _encoder_runner()
    dummy = [("image", make_item("a", 6))]

    runner.profile_encoder_cache(dummy, _Budget())  # type: ignore[arg-type]

    assert len(calls) == 1
    assert sorted(runner.encoder_cache.encoder_outputs) == ["tmp_0"]
