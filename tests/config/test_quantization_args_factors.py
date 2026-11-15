# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.config.quantization import QuantizationConfigArgs
from vllm.config.utils import normalize_value

SPEC = {"targets": {"re:.*\\.mlp\\.experts$": {"activation": "int8_per_token"}}}


def _names(args: QuantizationConfigArgs) -> set[str]:
    _, items = normalize_value(args)
    return {name for name, _ in items}


def test_requantize_prequantized_unset_is_not_hashed():
    assert "requantize_prequantized" not in _names(QuantizationConfigArgs(**SPEC))
    assert "targets" in _names(QuantizationConfigArgs(**SPEC))


def test_requantize_prequantized_set_is_hashed():
    unset = json.dumps(normalize_value(QuantizationConfigArgs(**SPEC)))
    on = QuantizationConfigArgs(**SPEC, requantize_prequantized=True)
    assert "requantize_prequantized" in _names(on)
    assert json.dumps(normalize_value(on)) != unset
