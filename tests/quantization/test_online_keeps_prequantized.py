# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online quantization over a pre-quantized checkpoint keeps calibrated layers.

A layer the checkpoint already holds in an integer format is left as it is when
the online spec asks for the same or a coarser integer format; a different
format is re-quantized (and recorded); `requantize_prequantized` restores the
re-rounding. Resolution only: no weights, no kernels.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import quant_config_args
from vllm.config.quantization import user_activation_override
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import resolve_quant_method
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8PerChannelOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

_NOT_MTP = r"^(?!(.*\.)?mtp\.)(.*\.)?"

# The form of the QFN W4A8 checkpoint: one int-quantized W8A8 group (dense
# layers and the head) and one pack-quantized W4A16 g128 group (experts).
CT_W4A8_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "quantization_status": "compressed",
    "kv_cache_scheme": None,
    "config_groups": {
        "group_0": {
            "targets": [
                f"re:{_NOT_MTP}layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)$",
                f"re:{_NOT_MTP}layers\\.\\d+\\.linear_attn\\.(in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj)$",
                "re:^(.*\\.)?lm_head$",
            ],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "channel",
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
            "format": "int-quantized",
        },
        "group_1": {
            "targets": [
                f"re:{_NOT_MTP}layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$",
            ],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 128,
                "strategy": "group",
                "dynamic": False,
            },
            "input_activations": None,
            "format": "pack-quantized",
        },
    },
    "ignore": [
        f"re:{_NOT_MTP}layers\\.\\d+\\.linear_attn\\.(in_proj_a|in_proj_b|in_proj_ba)$",
        "re:^(.*\\.)?embed_tokens$",
    ],
}

QKV = "language_model.model.layers.3.self_attn.qkv_proj"
BA = "language_model.model.layers.0.linear_attn.in_proj_ba"
EXPERTS = "language_model.model.layers.0.mlp.experts"
HEAD = "language_model.lm_head"


@pytest.fixture
def vllm_config(default_vllm_config):
    # Online methods read the model dtype; no model is loaded here.
    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, quantization_config=None, convert_type=None
    )
    return default_vllm_config


class _TestLinear(ColumnParallelLinear):
    def __init__(self):
        torch.nn.Module.__init__(self)


class _TestHead(ParallelLMHead):
    def __init__(self):
        torch.nn.Module.__init__(self)


class _TestRoutedExperts(RoutedExperts):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.ckpt_gate_proj_name = "gate_proj"
        self.ckpt_up_proj_name = "up_proj"
        self.ckpt_down_proj_name = "down_proj"


def _ct_config(**online) -> CompressedTensorsConfig:
    config = CompressedTensorsConfig.from_config(copy.deepcopy(CT_W4A8_CONFIG))
    config.online_quantization_config = OnlineQuantizationConfig(
        quant_config_args(**online)
    )
    return config


def _online(config) -> OnlineQuantizationConfig:
    online = config.online_quantization_config
    assert isinstance(online, OnlineQuantizationConfig)
    return online


@pytest.mark.parametrize(
    ("quant_key", "expected"),
    [
        (kInt8StaticChannelSym, ("int", 8)),
        (kInt8DynamicTokenSym, ("int", 8)),
        (kInt4Static, ("int", 4)),
        (kFp8StaticChannelSym, ("float", 8)),
        (kMxfp4Static, None),  # packed uint8 storage, not int8
        (None, None),
    ],
)
def test_quant_key_weight_format(quant_key, expected) -> None:
    from vllm.model_executor.layers.quantization.base_config import (
        quant_key_weight_format,
    )

    assert quant_key_weight_format(quant_key) == expected


@pytest.mark.parametrize(
    ("checkpoint", "requested", "keep"),
    [
        (("int", 8), ("int", 8), True),  # int8 on int8
        (("int", 4), ("int", 8), True),  # int8 on int4 experts
        (("int", 4), ("int", 4), True),  # int4 on int4
        (("int", 8), ("int", 4), False),  # a finer request is a different format
        (("int", 8), ("float", 8), False),  # fp8 on int8
        (("float", 8), ("float", 8), False),  # float formats keep today's path
        (None, ("int", 8), False),  # unquantized layer
    ],
)
def test_keeps_checkpoint_weights_rule(checkpoint, requested, keep) -> None:
    from vllm.model_executor.layers.quantization.base_config import (
        keeps_checkpoint_weights,
    )

    assert keeps_checkpoint_weights(checkpoint, requested) is keep


def test_ct_checkpoint_weight_format() -> None:
    config = CompressedTensorsConfig.from_config(copy.deepcopy(CT_W4A8_CONFIG))
    assert config.get_checkpoint_weight_format(_TestLinear(), QKV) == ("int", 8)
    assert config.get_checkpoint_weight_format(_TestHead(), HEAD) == ("int", 8)
    assert config.get_checkpoint_weight_format(_TestRoutedExperts(), EXPERTS) == (
        "int",
        4,
    )
    assert config.get_checkpoint_weight_format(_TestLinear(), BA) is None


@pytest.mark.parametrize(
    "online",
    [
        {"linear": "int8_per_channel"},
        {"targets": {"re:.*self_attn\\.qkv_proj$": "int8_per_channel"}},
    ],
)
def test_int8_online_keeps_checkpoint_int8_linear(vllm_config, online):
    """A blanket or targeted int8 spec does not re-round a checkpoint int8 layer."""
    config = _ct_config(**online)
    method = resolve_quant_method(config, _TestLinear(), QKV)
    assert isinstance(method, CompressedTensorsLinearMethod)
    assert QKV in _online(config).kept_checkpoint_layers
    assert QKV not in _online(config).quantized_layers
    assert not _online(config).requantized_checkpoint_layers
    summaries = _online(config).checkpoint_layer_summaries(
        _online(config).kept_checkpoint_layers
    )
    assert len(summaries) == 1
    assert summaries[0].startswith("self_attn.qkv_proj: 1 (int8 -> int8_per_channel")


def test_requantize_prequantized_opt_in(vllm_config) -> None:
    """The explicit opt-in restores today's re-rounding of a calibrated layer."""
    config = _ct_config(linear="int8_per_channel", requantize_prequantized=True)
    method = resolve_quant_method(config, _TestLinear(), QKV)
    assert isinstance(method, Int8PerChannelOnlineLinearMethod)
    assert isinstance(method.requantization_source, CompressedTensorsLinearMethod)
    assert QKV in _online(config).requantized_checkpoint_layers
    assert not _online(config).kept_checkpoint_layers


def test_different_format_requantizes_and_is_recorded(vllm_config) -> None:
    """fp8 on an int8 layer is a different format: re-quantized, and recorded."""
    config = _ct_config(linear="fp8_per_channel")
    method = resolve_quant_method(config, _TestLinear(), QKV)
    assert isinstance(method, Fp8PtpcOnlineLinearMethod)
    assert isinstance(method.requantization_source, CompressedTensorsLinearMethod)
    assert _online(config).requantized_checkpoint_layers[QKV] == (
        "CompressedTensorsLinearMethod",
        "fp8_per_channel_static",
    )


def test_unquantized_layer_is_still_quantized_online(vllm_config) -> None:
    config = _ct_config(linear="int8_per_channel")
    method = resolve_quant_method(config, _TestLinear(), BA)
    assert isinstance(method, Int8PerChannelOnlineLinearMethod)
    assert method.requantization_source is None
    assert not _online(config).kept_checkpoint_layers


def _record_moe_activation(config, monkeypatch) -> list:
    seen = []

    def fake_get_quant_method(layer, prefix):
        seen.append(user_activation_override("moe", None)[0])
        return UnquantizedLinearMethod()  # stands in for the checkpoint method

    monkeypatch.setattr(config, "get_quant_method", fake_get_quant_method)
    return seen


@pytest.mark.parametrize(
    "online",
    [
        {"moe": "int8_per_channel"},
        {"targets": {"re:.*\\.mlp\\.experts$": "int8_per_channel"}},
    ],
)
def test_int8_online_keeps_int4_experts_with_int8_activations(
    vllm_config, monkeypatch, online
) -> None:
    """int8 on checkpoint int4 experts keeps the int4 weights; the spec's int8
    per-token activations reach the checkpoint method (the W4A8 path)."""
    config = _ct_config(**online)
    seen = _record_moe_activation(config, monkeypatch)
    resolve_quant_method(config, _TestRoutedExperts(), EXPERTS)
    assert seen == [kInt8DynamicTokenSym]
    assert _online(config).kept_checkpoint_layers[EXPERTS][0] == "int4"
    assert EXPERTS not in _online(config).quantized_layers


def test_activation_only_target_still_overrides(vllm_config, monkeypatch):
    config = _ct_config(
        targets={"re:.*\\.mlp\\.experts$": {"activation": "int8_per_token"}}
    )
    seen = _record_moe_activation(config, monkeypatch)
    resolve_quant_method(config, _TestRoutedExperts(), EXPERTS)
    assert seen == [kInt8DynamicTokenSym]
    assert not _online(config).kept_checkpoint_layers


def test_int8_head_target_on_checkpoint_int8_head_is_kept(vllm_config):
    """Before: NotImplementedError (online re-quantization of the lm_head)."""
    config = _ct_config(targets={"re:.*lm_head$": "int8_per_channel"})
    method = resolve_quant_method(config, _TestHead(), HEAD)
    assert isinstance(method, CompressedTensorsLinearMethod)
    assert HEAD in _online(config).kept_checkpoint_layers


def test_fp8_head_target_on_checkpoint_int8_head_still_raises(vllm_config):
    config = _ct_config(targets={"re:.*lm_head$": "fp8_per_channel"})
    with pytest.raises(NotImplementedError, match="already quantized"):
        resolve_quant_method(config, _TestHead(), HEAD)
