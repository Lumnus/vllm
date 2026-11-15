# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU MTP drafter's own LM head takes the checkpoint's form.

The drafter loads every tensor whose name holds ``lm_head`` before the target's
head replaces its own. Built without the quant config, a checkpoint whose head
is int8 (``weight`` + ``weight_scale``) failed to load: no ``weight_scale``
parameter (CPU, stand-in submodules)."""

from types import SimpleNamespace

from torch import nn

from vllm.models.qwen4_exp.xpu import mtp as mtp_module


class _Rec(nn.Module):
    built: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        _Rec.built.append(kwargs)


class _Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.make_empty_intermediate_tensors = None


def test_drafter_lm_head_takes_the_quant_config(monkeypatch):
    _Rec.built = []
    monkeypatch.setattr(mtp_module, "Qwen4ExpMultiTokenPredictor", _Predictor)
    monkeypatch.setattr(mtp_module, "ParallelLMHead", _Rec)
    monkeypatch.setattr(mtp_module, "LogitsProcessor", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp_module.Qwen4ExpMTP, "set_moe_parameters", lambda self: None)
    quant_config = object()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(vocab_size=16, hidden_size=8)
        ),
        quant_config=quant_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    mtp_module.Qwen4ExpMTP(vllm_config=vllm_config, prefix="")
    assert [k.get("quant_config") for k in _Rec.built] == [quant_config]
    assert [k["prefix"] for k in _Rec.built] == ["lm_head"]
