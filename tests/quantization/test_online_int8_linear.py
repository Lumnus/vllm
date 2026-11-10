# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online W8A8 INT8 linear quantization (`int8_per_channel`), on CPU: config
resolution, the weight quantizer, and a layer run through a reference INT8
kernel."""

from types import SimpleNamespace

import pytest
import torch

from tests.quantization.utils import quant_config_args
from vllm.config.quantization import (
    QUANT_KEY_NAMES,
    QuantizationConfigArgs,
    QuantSpec,
    resolve_quantization_config,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.online import int8 as online_int8
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
    Int8PerChannelOnlineLinearMethod,
    quantize_int8_per_channel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)


class _TestLinear(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def test_names_and_shorthand():
    assert QUANT_KEY_NAMES["int8_per_token"] == kInt8DynamicTokenSym
    assert "int8_per_channel" in QUANTIZATION_METHODS
    args = resolve_quantization_config("int8_per_channel", None)
    assert args.linear.weight == kInt8StaticChannelSym
    assert args.moe.weight == kInt8StaticChannelSym


def test_shorthand_selects_the_int8_linear_method(default_vllm_config):
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    config = OnlineQuantizationConfig(quant_config_args(linear="int8_per_channel"))
    resolved = config.resolve_quant_method_cls(_TestLinear, "model.layers.0.o_proj")
    assert resolved is not None
    assert resolved[-1] is Int8PerChannelOnlineLinearMethod
    # An omitted activation resolves to dynamic per-token int8.
    assert resolved[3].activation == kInt8DynamicTokenSym
    method = config.get_quant_method(_TestLinear(), "model.layers.0.o_proj")
    assert isinstance(method, Int8PerChannelOnlineLinearMethod)
    assert method.activation_quant_key == kInt8DynamicTokenSym


def test_moe_shorthand_keeps_the_int8_moe_method():
    config = OnlineQuantizationConfig(
        resolve_quantization_config("int8_per_channel", None)
    )
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    resolved = config.resolve_quant_method_cls(RoutedExperts, "mlp.experts")
    assert resolved is not None and resolved[-1] is Int8OnlineMoEMethod


@pytest.mark.parametrize(
    ("activation", "ok"),
    [("int8_per_token", True), (None, False), ("fp8_per_token", False)],
)
def test_activation_override(activation, ok):
    spec = QuantSpec(weight="int8_per_channel_static", activation=activation)
    config = OnlineQuantizationConfig(QuantizationConfigArgs(linear=spec))
    if ok:
        resolved = config.resolve_quant_method_cls(_TestLinear, "x")
        assert resolved[3].activation == kInt8DynamicTokenSym
    else:
        with pytest.raises(NotImplementedError, match="supports only"):
            config.resolve_quant_method_cls(_TestLinear, "x")
    assert kFp8DynamicTokenSym != kInt8DynamicTokenSym


def test_ignore_and_composition_with_a_checkpoint_method(default_vllm_config):
    """On a pre-quantized checkpoint, layers its method leaves unquantized are
    quantized online; ignored ones stay as the checkpoint has them."""
    from vllm.model_executor.layers.quantization import resolve_quant_method

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    online = OnlineQuantizationConfig(
        quant_config_args(linear="int8_per_channel", ignore=["re:.*in_proj_ba$"])
    )
    checkpoint = SimpleNamespace(
        online_quantization_config=online,
        packed_modules_mapping={},
        get_quant_method=lambda *_: UnquantizedLinearMethod(),
    )
    method = resolve_quant_method(
        checkpoint, _TestLinear(), "model.layers.0.linear_attn.out_proj"
    )
    assert isinstance(method, Int8PerChannelOnlineLinearMethod)
    method = resolve_quant_method(
        checkpoint, _TestLinear(), "model.layers.0.linear_attn.in_proj_ba"
    )
    assert isinstance(method, UnquantizedLinearMethod)


def test_quantize_int8_per_channel_matches_the_rule():
    torch.manual_seed(0)
    w = torch.randn(48, 96, dtype=torch.bfloat16)
    w[3] = 0  # an all-zero row keeps a finite scale and zero codes
    amax = w.float().abs().amax(dim=-1, keepdim=True)
    q, s = quantize_int8_per_channel(w, amax)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (48, 1)
    assert torch.isfinite(s).all() and not q[3].any()
    ref_s = (amax / 127).clamp_min(torch.finfo(torch.float32).tiny)
    ref_q = torch.round(w.float() / ref_s).clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(s, ref_s)
    torch.testing.assert_close(q, ref_q)
    assert q.abs().max() == 127


class _RefInt8Kernel:
    """INT8 scaled mm in fp32 with the platform kernels' layout: `[K, N]` weight
    view, per-token symmetric activation quant."""

    def process_weights_after_loading(self, layer):
        layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)

    def apply_weights(self, layer, x, bias=None):
        xf = x.reshape(-1, x.shape[-1]).float()
        xs = (xf.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-10)
        xq = torch.round(xf / xs).clamp(-128, 127)
        out = (xq @ layer.weight.float()) * xs * layer.weight_scale.float().t()
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype).reshape(*x.shape[:-1], -1)


def test_layer_quantizes_and_runs(default_vllm_config, monkeypatch):
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    chosen = {}

    def fake_init(**kwargs):
        chosen.update(kwargs)
        return _RefInt8Kernel()

    import vllm.model_executor.layers.quantization.online.fp8 as online_fp8
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(online_int8, "init_int8_linear_kernel", fake_init)
    monkeypatch.setattr(online_fp8, "initialize_online_processing", lambda layer: None)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)

    torch.manual_seed(0)
    k, n = 128, 64
    w = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    layer = torch.nn.Module()
    method = Int8PerChannelOnlineLinearMethod()
    method.create_weights(layer, k, [n // 2, n // 2], k, n, torch.bfloat16,
                          weight_loader=None)
    assert chosen == dict(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
        module_name="Int8PerChannelOnlineLinearMethod",
        is_fused_module=True,
    )
    assert layer.weight.is_meta
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.tp_size = 1
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int8 and layer.weight.shape == (k, n)
    assert layer.input_scale is None and layer.azp_adj is None

    x = torch.randn(5, k, dtype=torch.bfloat16)
    out = method.apply(layer, x)
    ref = torch.nn.functional.linear(x.float(), w.float())
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.02, rel
    # A second call is a no-op.
    method.process_weights_after_loading(layer)
    assert layer.weight.shape == (k, n)
