# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU HC down projection in INT8 when an online INT8 config targets it, on
CPU: torch stand-ins for the HC kernels and a reference for the vllm-xpu-kernels
INT8 GEMM."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.models.qwen4_exp.test_xpu_hc import (  # noqa: F401
    CONFIG,
    HC,
    LOWRANK,
    N_EMBD,
    TOKENS,
    _hc_with_weights,
    _no_tp,
    _torch_ops,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.int8 import (
    quantize_int8_per_channel,
)
from vllm.models.qwen4_exp.xpu import hyperconnection as hc_module
from vllm.models.qwen4_exp.xpu.hyperconnection import HyperConnection

_SCHEMA = (
    "int8_dense_w8a8_fused_gemm(Tensor x, Tensor w, Tensor w_scale, Tensor? bias) "
    "-> Tensor"
)


def _ref_w8a8(x, w, w_scale, bias):
    xf = x.float()
    xs = (xf.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-10)
    xq = torch.round(xf / xs).clamp(-127, 127)
    out = (xq @ w.float().t()) * xs * w_scale.float().reshape(1, -1)
    return out.to(x.dtype)


@pytest.fixture
def int8_gemm():
    calls = []
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def impl(x, w, w_scale, bias):
            calls.append((tuple(x.shape), tuple(w.shape), w.dtype, w.is_contiguous()))
            return _ref_w8a8(x, w, w_scale, bias)

        lib.impl("int8_dense_w8a8_fused_gemm", impl, "CPU")
        yield calls


def _as_online_int8(hc: HyperConnection) -> torch.Tensor:
    """Give the merged down GEMM the state the online INT8 method leaves: int8
    rows as a [K, N] view, fp32 row scales. Returns the BF16 weight it replaced."""
    layer = hc.input_mix_weight_down_block_inject
    weight = layer.weight.data.clone()
    q, s = quantize_int8_per_channel(weight, weight.abs().amax(-1, keepdim=True))
    layer.quant_method = SimpleNamespace(name="online int8")
    layer.weight = torch.nn.Parameter(q.t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
    return weight


def test_constructor_takes_quant_config():
    hc = HyperConnection(CONFIG, prefix="hc", quant_config=None)
    layer = hc.input_mix_weight_down_block_inject
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    weight, scale = hc._down_operands()
    assert weight is layer.weight and scale is None


def test_int8_down_runs_the_int8_gemm(monkeypatch, int8_gemm):
    _torch_ops(monkeypatch)
    torch.manual_seed(0)
    hc = _hc_with_weights(0)
    x = torch.randn(TOKENS, HC, N_EMBD)
    mixed_ref, inject_ref = hc.mix(x)
    bf16_down = _as_online_int8(hc)

    mixed, inject = hc.mix(x)
    assert int8_gemm and int8_gemm[0][1:] == (tuple(bf16_down.shape), torch.int8, True)
    assert mixed.shape == mixed_ref.shape and inject.shape == (TOKENS, HC)
    rel = (mixed - mixed_ref).norm() / mixed_ref.norm()
    assert rel < 0.05, rel
    torch.testing.assert_close(inject, inject_ref, atol=0.05, rtol=0.05)

    # combine_and_mix takes the same operands.
    residual = torch.randn(TOKENS, HC, N_EMBD)
    block = torch.randn(TOKENS, N_EMBD)
    combined, mixed2, inject2 = hc.combine_and_mix(residual, block, inject)
    assert combined.shape == residual.shape and inject2.shape == (TOKENS, HC)
    assert len(int8_gemm) == 2


def test_int8_down_is_row_batch_invariant(monkeypatch, int8_gemm):
    _torch_ops(monkeypatch)
    hc = _hc_with_weights(1)
    _as_online_int8(hc)
    x = torch.randn(8, HC, N_EMBD)
    full, full_inject = hc.mix(x)
    # The down GEMM (whose columns are the injection logits) is per row; the
    # torch stand-in for the up GEMM is not bitwise across batch sizes on CPU.
    for rows in (1, 3):
        part, part_inject = hc.mix(x[:rows])
        assert torch.equal(part_inject, full_inject[:rows])
        torch.testing.assert_close(part, full[:rows])


def test_int8_down_without_the_kernel_refuses(monkeypatch):
    if hasattr(torch.ops._xpu_C, "int8_dense_w8a8_fused_gemm"):
        pytest.skip("a build with the op is installed")
    hc = _hc_with_weights(0)
    _as_online_int8(hc)
    with pytest.raises(NotImplementedError, match="int8_dense_w8a8_fused_gemm"):
        hc._down_operands()


def test_other_quant_methods_refuse():
    hc = _hc_with_weights(0)
    layer = hc.input_mix_weight_down_block_inject
    layer.quant_method = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="BF16 or online INT8"):
        hc._down_operands()


def test_fakes_take_the_scale():
    w = torch.zeros(LOWRANK + HC + 4, HC * N_EMBD, dtype=torch.int8)
    x = torch.zeros(TOKENS, HC * N_EMBD, dtype=torch.bfloat16)
    up = torch.zeros(HC * N_EMBD, LOWRANK, dtype=torch.bfloat16)
    norm = torch.zeros(HC * N_EMBD)
    mixed, inject = hc_module._hc_mix_fake(
        x, norm, w, up, 1e-6, HC, LOWRANK, False, torch.ones(w.shape[0], 1)
    )
    assert mixed.shape == (TOKENS, N_EMBD) and inject.shape == (TOKENS, HC)
    assert F is not None
