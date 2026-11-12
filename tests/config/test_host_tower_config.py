# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--mm-encoder-host-towers`: its formats, what it derives, what it refuses."""

import pytest

from vllm.config import CompilationConfig, ModelConfig, VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.multimodal import HostTowerOptions, MultiModalConfig
from vllm.config.utils import hash_config
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.cpu_test

MODEL = "llava-hf/llava-1.5-7b-hf"


def test_off_by_default():
    config = MultiModalConfig()
    assert config.mm_encoder_host_towers == {}
    assert config.mm_encoder_host_twin is None


def test_short_format_is_a_thread_count():
    config = MultiModalConfig(mm_encoder_host_towers={"vision_tower": 16})
    assert config.mm_encoder_host_towers == {
        "vision_tower": HostTowerOptions(threads=16)
    }


def test_full_format():
    config = MultiModalConfig(
        mm_encoder_host_towers={
            "vision_tower": {"threads": 8, "dtype": "model", "cache_gb": 0.5},
            "audio_tower": {},
        }
    )
    vision = config.mm_encoder_host_towers["vision_tower"]
    assert (vision.threads, vision.dtype, vision.cache_gb) == (8, "model", 0.5)
    audio = config.mm_encoder_host_towers["audio_tower"]
    assert (audio.threads, audio.dtype, audio.cache_gb) == (0, "float32", 1.0)


@pytest.mark.parametrize(
    "options",
    [
        {"thread": 4},
        {"threads": -1},
        {"dtype": "float64"},
        {"cache_gb": 0},
        "all",
        True,
    ],
)
def test_invalid_options_are_refused(options):
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_host_towers={"vision_tower": options})


def test_host_towers_do_not_change_the_compile_hash():
    assert hash_config(MultiModalConfig()) == hash_config(
        MultiModalConfig(mm_encoder_host_towers={"vision_tower": 4})
    )


def test_cli_flag_reaches_the_engine_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    args = parser.parse_args([])
    assert EngineArgs.from_cli_args(args=args).mm_encoder_host_towers == {}

    args = parser.parse_args(
        ["--mm-encoder-host-towers", '{"vision_tower": {"threads": 16}}']
    )
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.mm_encoder_host_towers == {"vision_tower": {"threads": 16}}


def _vllm_config(host_towers=None, **kwargs) -> VllmConfig:
    model_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key.startswith("mm_") or key == "language_model_only"
    }
    return VllmConfig(
        model_config=ModelConfig(
            MODEL, mm_encoder_host_towers=host_towers, **model_kwargs
        ),
        **kwargs,
    )


def test_no_connector_without_host_towers():
    config = _vllm_config()
    assert config.ec_transfer_config is None


def test_host_towers_derive_the_connector():
    config = _vllm_config(
        {"vision_tower": {"cache_gb": 0.5}, "audio_tower": {"cache_gb": 0.25}}
    )

    ec_config = config.ec_transfer_config
    assert ec_config is not None
    assert ec_config.ec_connector == "ECHostTowerConnector"
    assert ec_config.is_ec_consumer and not ec_config.is_ec_producer
    assert ec_config.ec_connector_extra_config == {"ec_cpu_bytes": 3 * (1 << 28)}

    # What a consumer of another engine's encoder is granted stays off: the
    # frontend still requires the media of every request.
    mm_config = config.model_config.get_multimodal_config()
    assert mm_config.allow_missing_mm_embeddings is False
    assert mm_config.enable_mm_embeds is False
    assert mm_config.mm_encoder_only is False


def test_language_model_only_has_no_tower_to_host():
    config = _vllm_config({"vision_tower": 4}, language_model_only=True)
    assert config.ec_transfer_config is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "ec_transfer_config": ECTransferConfig(
                    ec_connector="ECExampleConnector", ec_role="ec_consumer"
                )
            },
            "ec-transfer-config",
        ),
        ({"mm_processor_cache_type": "shm"}, "lru"),
        ({"mm_encoder_only": True}, "mm-encoder-only"),
        (
            {"compilation_config": CompilationConfig(cudagraph_mm_encoder=True)},
            "cudagraph_mm_encoder",
        ),
    ],
)
def test_deployments_that_cannot_host_a_tower_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _vllm_config({"vision_tower": 4}, **kwargs)
