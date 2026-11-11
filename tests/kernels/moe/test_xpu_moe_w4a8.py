# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 MoE on XPU, chosen by `quantization_config.moe.activation`: backend
selection and the call into vllm-xpu-kernels, on CPU with stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
    XPUExpertsW4A8Int,
    XPUExpertsWNA16,
)
from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt4Static,
    kInt8DynamicTokenSym,
)


def test_quant_schemes():
    assert XPUExpertsWNA16._supports_quant_scheme(kInt4Static, None)
    assert not XPUExpertsWNA16._supports_quant_scheme(kInt4Static, kInt8DynamicTokenSym)
    assert XPUExpertsW4A8Int._supports_quant_scheme(kInt4Static, kInt8DynamicTokenSym)
    assert not XPUExpertsW4A8Int._supports_quant_scheme(kInt4Static, None)
    assert XPUExpertsW4A8Int.impl_apply_kwargs == {"w4a8": True}
    assert XPUExpertsWNA16.impl_apply_kwargs == {}


@pytest.fixture
def on_xpu(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(xpu_moe, "xpu_w4a8_moe_available", lambda: True)
    monkeypatch.setattr(XPUExpertsWNA16, "_supports_current_device", staticmethod(
        lambda: True))


def _select(moe_args):
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    config.model_config = SimpleNamespace(quantization_config=moe_args)
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=256
    )
    with set_current_vllm_config(config):
        return int_wna16.select_wna16_moe_backend(
            moe, kInt4Static, quant_config=None, may_have_zp=False,
            may_have_bias=False,
        )


def test_default_selects_w4a16(on_xpu):
    backend, cls = _select(None)
    assert backend == int_wna16.WNA16MoEBackend.XPU and cls is XPUExpertsWNA16


def test_moe_activation_override_selects_w4a8(on_xpu):
    args = QuantizationConfigArgs(moe=QuantSpec(activation="int8_per_token"))
    backend, cls = _select(args)
    assert backend == int_wna16.WNA16MoEBackend.XPU and cls is XPUExpertsW4A8Int


def test_override_without_the_kernel_fails_loudly(on_xpu, monkeypatch):
    monkeypatch.setattr(xpu_moe, "xpu_w4a8_moe_available", lambda: False)
    args = QuantizationConfigArgs(moe=QuantSpec(activation="int8_per_token"))
    with pytest.raises(NotImplementedError, match="activation="):
        _select(args)


class _FakeImpl:
    def __init__(self):
        self.calls = []

    def apply(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize(("cls", "w4a8"), [(XPUExpertsWNA16, False),
                                           (XPUExpertsW4A8Int, True)])
def test_apply_passes_w4a8(cls, w4a8):
    experts = object.__new__(cls)
    experts.tiers = None
    experts.fused_moe_impl = _FakeImpl()
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    x = torch.zeros(3, 8)
    experts.apply(
        output=torch.empty_like(x), hidden_states=x, w1=None, w2=None,
        topk_weights=torch.ones(3, 2), topk_ids=torch.zeros(3, 2, dtype=torch.int32),
        activation=None, global_num_experts=8, expert_map=None, a1q_scale=None,
        a2_scale=None, workspace13=None, workspace2=None, expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    (call,) = experts.fused_moe_impl.calls
    assert call.get("w4a8", False) is w4a8
    # The W4A8 experts report the kernels' decision once, after the first call.
    assert getattr(experts, "_w4a8_reported", False) is w4a8
