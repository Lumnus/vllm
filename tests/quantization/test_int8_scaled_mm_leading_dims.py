# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The INT8 scaled-mm linear kernels take activations with leading dims.

The Qwen4-Exp MTP fc stage hands ``fc_hidden`` a ``[T, hc, K]`` activation. Under
an online ``int8_per_channel`` target the XPU kernel passed it unflattened to
``triton_scaled_mm``, which unpacks ``M, K = input.shape`` and raised
``ValueError: too many values to unpack``. CPU: a reference per-token quant and
a reference GEMM with the Triton kernel's 2-D contract stand in for the ops."""

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm import triton as triton_mod
from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mod


def _ref_quant(x, scale=None, azp=None, symmetric=True):
    # ops.scaled_int8_quant: int8 output of x's shape, one scale per row.
    rows = x.reshape(-1, x.shape[-1]).float()
    s = rows.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (rows / s).round().clamp(-128, 127).to(torch.int8)
    return q.view(x.shape), s, None


def _ref_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None, **_):
    M, K = input.shape  # the Triton kernel's contract: 2-D rows only
    out = (input.float() @ weight.float()) * scale_a.reshape(-1, 1)
    out = out * scale_b.reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _weights(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    w_s = w.abs().amax(dim=-1, keepdim=True) / 127.0
    w_q = (w / w_s).round().clamp(-128, 127).to(torch.int8)
    # The kernels keep the [N, K] weight as a [K, N] view.
    return w_q.t(), w_s.reshape(-1)


def _kernel(cls, w_q, w_s):
    kernel = object.__new__(cls)
    kernel._get_layer_params = lambda layer: (w_q, w_s, None, None, None)
    return kernel


@pytest.fixture
def ref_ops(monkeypatch):
    monkeypatch.setattr(xpu_mod.ops, "scaled_int8_quant", _ref_quant)
    monkeypatch.setattr(xpu_mod, "triton_scaled_mm", _ref_scaled_mm)
    monkeypatch.setattr(xpu_mod.entry_rows, "ACTIVE", False)


@pytest.mark.parametrize("shape", [(3, 4, 64), (1, 4, 64), (2, 3, 4, 64)])
def test_xpu_int8_kernel_takes_leading_dims(ref_ops, shape):
    k, n = shape[-1], 32
    w_q, w_s = _weights(n, k)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    kernel = _kernel(xpu_mod.XPUInt8ScaledMMLinearKernel, w_q, w_s)

    out = kernel.apply_weights(None, x)

    assert out.shape == (*shape[:-1], n)
    assert out.dtype == x.dtype
    flat = kernel.apply_weights(None, x.reshape(-1, k))
    torch.testing.assert_close(out.reshape(-1, n), flat, rtol=0, atol=0)


def test_xpu_int8_kernel_two_dim_unchanged(ref_ops):
    w_q, w_s = _weights(32, 64)
    x = torch.randn(12, 64, dtype=torch.bfloat16)
    kernel = _kernel(xpu_mod.XPUInt8ScaledMMLinearKernel, w_q, w_s)
    x_q, x_s, _ = _ref_quant(x)
    ref = _ref_scaled_mm(x_q, w_q, x_s, w_s, x.dtype)
    torch.testing.assert_close(kernel.apply_weights(None, x), ref, rtol=0, atol=0)


def test_triton_int8_kernel_takes_leading_dims(monkeypatch):
    monkeypatch.setattr(triton_mod.ops, "scaled_int8_quant", _ref_quant)
    monkeypatch.setattr(triton_mod.entry_rows, "ACTIVE", False)
    seen = []

    def _ref_func(qx, weight, x_scale, weight_scale, output_dtype, bias=None):
        seen.append(qx.dim())
        return _ref_scaled_mm(qx, weight, x_scale, weight_scale, output_dtype, bias)

    monkeypatch.setattr(torch.ops.vllm, "int8_scaled_mm_func", _ref_func, raising=False)
    w_q, w_s = _weights(32, 64)
    x = torch.randn(3, 4, 64, dtype=torch.bfloat16)
    kernel = _kernel(triton_mod.TritonInt8ScaledMMLinearKernel, w_q, w_s)

    out = kernel.apply_weights(None, x)

    assert out.shape == (3, 4, 32)
    assert seen == [2]
