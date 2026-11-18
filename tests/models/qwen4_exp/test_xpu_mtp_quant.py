# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU MTP fc projections take the quantization config, so an online config
can quantize the whole draft (CPU, stand-in submodules)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.xpu import mtp as mtp_module


class _Rec(nn.Module):
    built: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        _Rec.built.append(kwargs)


def test_fc_projections_take_the_quant_config(monkeypatch):
    _Rec.built = []
    for name in (
        "VocabParallelEmbedding",
        "HyperConnection",
        "_WideRMSNorm",
        "HyperConnectionRMSNorm",
        "Qwen4ExpDecoderLayer",
    ):
        monkeypatch.setattr(mtp_module, name, lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp_module, "ReplicatedLinear", _Rec)
    monkeypatch.setattr(mtp_module, "_uses_qsa", lambda config: False)
    quant_config = object()
    config = SimpleNamespace(
        vocab_size=16,
        hc_count=4,
        hidden_size=8,
        rms_norm_eps=1e-6,
        mtp_num_hidden_layers=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    mtp_module.Qwen4ExpMultiTokenPredictor(vllm_config=vllm_config, prefix="mtp")
    fc = {k["prefix"]: k["quant_config"] for k in _Rec.built}
    assert fc == {"mtp.fc_embedding": quant_config, "mtp.fc_hidden": quant_config}
    assert torch is not None


# The 8-bit MTP drafter over a W4A8 checkpoint whose MTP layer is BF16: the MTP
# fc and attention projections W8A8 online, the MTP routed experts online FP8
# (no XPU INT8 MoE backend, no int4 weights for the MTP), the target's experts
# keep their checkpoint int4 weights with int8 activations (W4A8), and the head
# is INT8. The target-experts pattern excludes ``mtp.`` so that each experts
# layer matches one target.
_TARGET_EXPERTS = r"re:^(?!(.*\.)?mtp\.)(.*\.)?layers\.\d+\.mlp\.experts$"
_MTP_LINEARS = (
    r"re:.*mtp\.(fc_embedding|fc_hidden|"
    r"layers\.\d+\.self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj))$"
)
_MTP_EXPERTS = r"re:.*mtp\.layers\.\d+\.mlp\.experts$"
_DRAFTER_8BIT = {
    "targets": {
        "re:.*lm_head$": "int8_per_channel",
        _TARGET_EXPERTS: {"activation": "int8_per_token"},
        _MTP_LINEARS: "int8_per_channel",
        _MTP_EXPERTS: "fp8_per_tensor",
    }
}


def _online_config(targets):
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    config = OnlineQuantizationConfig(QuantizationConfigArgs(**targets))
    config.packed_modules_mapping = mtp_module.Qwen4ExpMTP.packed_modules_mapping
    return config


def _resolve(config, layer_cls, prefix):
    resolved = config.resolve_quant_method_cls(layer_cls, prefix)
    if resolved is not None:
        return resolved[-1].__name__, resolved[3].activation
    target = config._resolve_targets_quant_method_metadata(prefix, layer_cls)
    if target is not None:
        return "checkpoint", target[3].activation
    return None


def test_8bit_drafter_targets_resolve():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    config = _online_config(_DRAFTER_8BIT)
    int8 = ("Int8PerChannelOnlineLinearMethod", kInt8DynamicTokenSym)
    target = "language_model.model.layers.7."
    for prefix in (
        "mtp.fc_embedding",
        "mtp.fc_hidden",
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.self_attn.o_proj",
        "model.mtp.layers.0.self_attn.qkv_proj",
    ):
        assert _resolve(config, LinearBase, prefix) == int8, prefix
    moe = _resolve(config, RoutedExperts, "mtp.layers.0.mlp.experts")
    assert moe is not None and moe[0] == "Fp8PerTensorOnlineMoEMethod"
    assert _resolve(config, RoutedExperts, target + "mlp.experts") == (
        "checkpoint",
        kInt8DynamicTokenSym,
    )
    assert _resolve(config, ParallelLMHead, "lm_head") == int8
    # Left as the checkpoint has them (calibrated INT8 there, BF16 in the MTP).
    for prefix in (
        target + "self_attn.qkv_proj",
        target + "mlp.shared_expert.gate_up_proj",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj",
        "mtp.layers.0.mlp.gate",
    ):
        assert _resolve(config, LinearBase, prefix) is None, prefix


def test_broad_experts_target_collides_with_the_mtp_experts_target():
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    targets = dict(_DRAFTER_8BIT["targets"])
    targets[r"re:.*\.mlp\.experts$"] = targets.pop(_TARGET_EXPERTS)
    config = _online_config({"targets": targets})
    with pytest.raises(ValueError, match="multiple"):
        _resolve(config, RoutedExperts, "mtp.layers.0.mlp.experts")
