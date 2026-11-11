# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU MTP fc projections take the quantization config, so an online config
can quantize the whole draft (CPU, stand-in submodules)."""

from types import SimpleNamespace

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
