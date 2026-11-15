# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A compressed-tensors checkpoint whose int4 experts declare int8 per-token
input activations (W4A8) selects, on XPU, the WNA16 method with int8
activations: the XPU W4A8 experts the serve-time override selects on a W4A16
checkpoint. Resolution only (the backend selectors are recorded, not run)."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import quant_config_args
from vllm.config.quantization import user_activation_override
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as ct_moe_pkg,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe as ct_moe,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a8_int8 as ct_w4a8,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_wna16 as ct_wna16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)

_INT4_G128 = {
    "num_bits": 4,
    "type": "int",
    "symmetric": True,
    "group_size": 128,
    "strategy": "group",
    "dynamic": False,
}
_INT8_TOKEN = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "token",
    "dynamic": True,
}


def _config(input_activations) -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_1": {
                    "targets": [
                        "re:.*mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
                    ],
                    "weights": copy.deepcopy(_INT4_G128),
                    "input_activations": copy.deepcopy(input_activations),
                    "format": "pack-quantized",
                }
            },
            "ignore": [],
        }
    )


class _TestRoutedExperts(RoutedExperts):
    def __init__(self):
        from tests.kernels.moe.utils import make_dummy_moe_config

        torch.nn.Module.__init__(self)
        self.ckpt_gate_proj_name = "gate_proj"
        self.ckpt_up_proj_name = "up_proj"
        self.ckpt_down_proj_name = "down_proj"
        self.moe_config = make_dummy_moe_config(
            num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=256
        )


@pytest.fixture
def selections(default_vllm_config, monkeypatch):
    """Record which backend selector runs and the activation it sees."""
    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, quantization_config=None
    )
    seen: list[tuple[str, object]] = []

    def wna16(*args, **kwargs):
        seen.append(("wna16", user_activation_override("moe", default_vllm_config)[0]))
        return WNA16MoEBackend.XPU, object

    def w4a8_cpu(*args, **kwargs):
        seen.append(("w4a8_cpu", None))
        return None, object

    monkeypatch.setattr(ct_wna16, "select_wna16_moe_backend", wna16)
    monkeypatch.setattr(ct_w4a8, "select_w4a8_int8_moe_backend", w4a8_cpu)
    return seen


def _platform(monkeypatch, xpu: bool) -> None:
    monkeypatch.setattr(ct_moe.current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr(ct_moe.current_platform, "is_rocm", lambda: False)


PREFIX = "model.layers.0.mlp.experts"


def test_w4a8_checkpoint_selects_xpu_w4a8_experts(selections, monkeypatch):
    _platform(monkeypatch, xpu=True)
    method = ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        _config(_INT8_TOKEN), _TestRoutedExperts(), PREFIX
    )
    assert isinstance(method, ct_wna16.CompressedTensorsWNA16MoEMethod)
    assert selections == [("wna16", kInt8DynamicTokenSym)]


def test_w4a8_checkpoint_user_override_wins(
    selections, monkeypatch, default_vllm_config
):
    """`moe.activation: null` runs the declared W4A8 experts as W4A16."""
    _platform(monkeypatch, xpu=True)
    default_vllm_config.model_config.quantization_config = quant_config_args(
        moe={"activation": None}
    )
    ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        _config(_INT8_TOKEN), _TestRoutedExperts(), PREFIX
    )
    assert selections == [("wna16", None)]


def test_w4a16_checkpoint_unchanged_on_xpu(selections, monkeypatch):
    _platform(monkeypatch, xpu=True)
    method = ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        _config(None), _TestRoutedExperts(), PREFIX
    )
    assert isinstance(method, ct_wna16.CompressedTensorsWNA16MoEMethod)
    assert selections == [("wna16", None)]


def test_w4a16_checkpoint_serve_time_override_still_works(
    selections, monkeypatch, default_vllm_config
):
    _platform(monkeypatch, xpu=True)
    default_vllm_config.model_config.quantization_config = quant_config_args(
        moe={"activation": "int8_per_token"}
    )
    ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        _config(None), _TestRoutedExperts(), PREFIX
    )
    assert selections == [("wna16", kInt8DynamicTokenSym)]


def test_w4a8_checkpoint_off_xpu_keeps_cpu_method(selections, monkeypatch):
    _platform(monkeypatch, xpu=False)
    method = ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        _config(_INT8_TOKEN), _TestRoutedExperts(), PREFIX
    )
    assert isinstance(method, ct_w4a8.CompressedTensorsW4A8Int8MoEMethod)
    assert selections == [("w4a8_cpu", None)]
