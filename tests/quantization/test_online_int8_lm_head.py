# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An lm_head named in `quantization_config.targets` runs online INT8 W8A8
(`int8_per_channel`), on CPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8PerChannelOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)


class _TestLMHead(ParallelLMHead):
    def __init__(self):
        torch.nn.Module.__init__(self)


@pytest.mark.parametrize(
    ("args", "selected"),
    [
        (QuantizationConfigArgs(targets={"re:.*lm_head$": "int8_per_channel"}), True),
        (
            QuantizationConfigArgs(
                targets={"lm_head": QuantSpec(weight="int8_per_channel_static")}
            ),
            True,
        ),
        # Broad patterns keep meaning linear and MoE layers.
        (QuantizationConfigArgs(targets={"re:.*": "int8_per_channel"}), False),
    ],
)
def test_int8_lm_head_needs_explicit_target(args, selected):
    config = OnlineQuantizationConfig(args)
    resolved = config.resolve_quant_method_cls(ParallelLMHead, "lm_head")
    if not selected:
        assert resolved is None
        return
    assert resolved is not None
    assert resolved[-1] is Int8PerChannelOnlineLinearMethod
    assert resolved[3].activation == kInt8DynamicTokenSym


def test_int8_lm_head_on_an_unquantized_head(default_vllm_config):
    from vllm.model_executor.layers.quantization import resolve_quant_method

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    quant_config = SimpleNamespace(
        online_quantization_config=OnlineQuantizationConfig(
            QuantizationConfigArgs(
                targets={"re:.*lm_head$": "int8_per_channel"},
            )
        ),
        packed_modules_mapping={},
        get_quant_method=lambda *_: UnquantizedEmbeddingMethod(),
    )
    method = resolve_quant_method(quant_config, _TestLMHead(), "language_model.lm_head")
    assert isinstance(method, Int8PerChannelOnlineLinearMethod)
    assert method.activation_quant_key == kInt8DynamicTokenSym
