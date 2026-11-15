# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA indexer shares the q/k/v GEMM only under the same scheme.

The fused layer is built under q/k/v's prefix, so it takes q/k/v's scheme; an
indexer quantized with another scheme of the same method class used to fuse
and load under q/k/v's scheme silently. Resolution only: no weights."""

import copy

import pytest
import torch

from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization import resolve_quant_method
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.models.qwen4_exp.nvidia import model as _nvidia_model  # noqa: F401
from vllm.models.qwen4_exp.nvidia.qsa import indexer_projection_mismatch

_W8 = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "channel",
    "dynamic": False,
}
_A8_TOKEN = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "token",
    "dynamic": True,
}
_A8_TENSOR_STATIC = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "tensor",
    "dynamic": False,
}
QKV_RE = "re:^(.*\\.)?layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|qkv_proj)$"
IDX_RE = "re:^(.*\\.)?layers\\.\\d+\\.self_attn\\.indexer\\.index_qk_proj$"


def _group(targets, input_activations):
    return {
        "targets": targets,
        "weights": copy.deepcopy(_W8),
        "input_activations": copy.deepcopy(input_activations),
        "format": "int-quantized",
    }


def _config(groups: dict, ignore=()) -> CompressedTensorsConfig:
    config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": groups,
            "ignore": list(ignore),
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    return config


class _QKV(QKVParallelLinear):
    def __init__(self, prefix):
        torch.nn.Module.__init__(self)
        self.prefix = prefix


class _Indexer(ReplicatedLinear):
    def __init__(self, prefix):
        torch.nn.Module.__init__(self)
        self.prefix = prefix


@pytest.fixture(params=["language_model.model.layers.3", "mtp.layers.0"])
def layers(request):
    base = f"{request.param}.self_attn"
    return _QKV(f"{base}.qkv_proj"), _Indexer(f"{base}.indexer.index_qk_proj")


def _resolve(config, qkv, indexer):
    qkv.quant_method = resolve_quant_method(config, qkv, qkv.prefix)
    indexer.quant_method = resolve_quant_method(config, indexer, indexer.prefix)
    return indexer_projection_mismatch(config, qkv, indexer)


def test_int8_qkv_and_int8_indexer_in_one_group_fuse(default_vllm_config, layers):
    config = _config({"group_0": _group([QKV_RE, IDX_RE], _A8_TOKEN)})
    assert _resolve(config, *layers) is None


def test_int8_qkv_and_bf16_indexer_run_unfused(default_vllm_config, layers):
    config = _config({"group_0": _group([QKV_RE], _A8_TOKEN)}, ignore=[IDX_RE])
    reason = _resolve(config, *layers)
    assert reason is not None and "quant methods differ" in reason


def test_same_method_class_different_scheme_runs_unfused(default_vllm_config, layers):
    """Before: same method class, so the indexer fused under q/k/v's scheme."""
    config = _config(
        {
            "group_0": _group([QKV_RE], _A8_TOKEN),
            "group_2": _group([IDX_RE], _A8_TENSOR_STATIC),
        }
    )
    qkv, indexer = layers
    reason = _resolve(config, qkv, indexer)
    # The old rule compared only the method classes, which are equal here.
    assert type(qkv.quant_method) is type(indexer.quant_method)
    assert reason == "checkpoint schemes differ"


def test_same_scheme_in_another_group_fuses(default_vllm_config, layers):
    config = _config(
        {
            "group_0": _group([QKV_RE], _A8_TOKEN),
            "group_2": _group([IDX_RE], _A8_TOKEN),
        }
    )
    assert _resolve(config, *layers) is None
