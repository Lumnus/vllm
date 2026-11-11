# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU W8A8 INT8 linear: oneDNN above 64 rows, Triton at or below, and only
when the kernels build has `int8_gemm_w8a8` with an `out_dtype` argument (CPU,
a stand-in op with the kernels' schema)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mm
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonInt8ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum

_SCHEMA = (
    "int8_gemm_w8a8(Tensor A_, Tensor A_scale, Tensor? A_zp, Tensor B, "
    "Tensor B_scale, Tensor? B_zp, int group_size, Tensor? bias, "
    "ScalarType? out_dtype=None) -> Tensor"
)
_OLD_SCHEMA = (
    "int8_gemm_w8a8(Tensor A_, Tensor A_scale, Tensor? A_zp, Tensor B, "
    "Tensor B_scale, Tensor? B_zp, int group_size, Tensor? bias) -> Tensor"
)


def _reference(qx, weight, x_scale, weight_scale, out_dtype, bias):
    out = (
        (qx.float() @ weight.float())
        * x_scale.float()
        * weight_scale.float().view(1, -1)
    )
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def _config(symmetric: bool = True) -> Int8ScaledMMLinearLayerConfig:
    return Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False,
        is_channelwise=True,
        input_symmetric=symmetric,
    )


@pytest.fixture
def onednn_op():
    """Register the kernels' op under _xpu_C for the test only."""
    calls = []
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def impl(A, A_scale, A_zp, B, B_scale, B_zp, group_size, bias, out_dtype):
            assert A_zp is None and B_zp is None and group_size == -1
            calls.append((tuple(A.shape), tuple(A_scale.shape), out_dtype))
            return _reference(A, B, A_scale, B_scale, out_dtype, bias)

        lib.impl("int8_gemm_w8a8", impl, "CPU")
        yield calls


@pytest.fixture
def xpu_platform(monkeypatch):
    monkeypatch.setattr(xpu_mm.current_platform, "is_xpu", lambda: True)


@pytest.fixture
def triton_calls(monkeypatch):
    calls = []

    def fake_triton(qx, weight, scale_a, scale_b, out_dtype, bias=None):
        calls.append(tuple(qx.shape))
        return _reference(qx, weight, scale_a, scale_b, out_dtype, bias)

    monkeypatch.setattr(xpu_mm, "triton_scaled_mm", fake_triton)
    return calls


def test_listed_before_triton_on_xpu():
    kernels = _POSSIBLE_INT8_KERNELS[PlatformEnum.XPU]
    assert kernels == [
        xpu_mm.XPUInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    ]


def test_probe_without_op():
    if hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
        pytest.skip("a build with the op is installed")
    assert not xpu_mm.xpu_int8_onednn_gemm_available()


def test_probe_follows_the_schema(onednn_op):
    assert xpu_mm.xpu_int8_onednn_gemm_available()


def test_older_build_without_out_dtype_is_refused(xpu_platform):
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_OLD_SCHEMA)
        assert not xpu_mm.xpu_int8_onednn_gemm_available()
        supported, reason = xpu_mm.XPUInt8ScaledMMLinearKernel.is_supported()
    assert not supported
    assert "out_dtype" in reason


def test_supported_needs_xpu_and_op(onednn_op, monkeypatch):
    monkeypatch.setattr(xpu_mm.current_platform, "is_xpu", lambda: False)
    assert not xpu_mm.XPUInt8ScaledMMLinearKernel.is_supported()[0]
    monkeypatch.setattr(xpu_mm.current_platform, "is_xpu", lambda: True)
    assert xpu_mm.XPUInt8ScaledMMLinearKernel.is_supported() == (True, None)


def test_asymmetric_activations_stay_on_triton():
    cls = xpu_mm.XPUInt8ScaledMMLinearKernel
    assert cls.can_implement(_config(True))[0]
    assert not cls.can_implement(_config(False))[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 64, 65, 300])
def test_row_split(onednn_op, triton_calls, monkeypatch, rows, dtype):
    k, n = 32, 16
    torch.manual_seed(0)
    w_q = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    layer = SimpleNamespace(
        weight=w_q.t(),
        weight_scale=torch.rand(n, 1) * 0.01 + 0.001,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    kernel = xpu_mm.XPUInt8ScaledMMLinearKernel.__new__(
        xpu_mm.XPUInt8ScaledMMLinearKernel
    )
    kernel.config = _config()
    kernel.layer_param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]

    def quant(x, scale, azp, symmetric=True):
        s = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
        return torch.round(x.float() / s).to(torch.int8), s, None

    monkeypatch.setattr(xpu_mm.ops, "scaled_int8_quant", quant)

    x = torch.randn(rows, k).to(dtype)
    bias = torch.randn(n).to(dtype)
    out = kernel.apply_weights(layer, x, bias)

    assert out.shape == (rows, n) and out.dtype == dtype
    if rows <= xpu_mm.XPU_INT8_ONEDNN_MIN_ROWS:
        assert triton_calls == [(rows, k)]
        assert onednn_op == []
    else:
        assert triton_calls == []
        assert onednn_op == [((rows, k), (rows, 1), dtype)]
    q, s, _ = quant(x, None, None)
    torch.testing.assert_close(
        out, _reference(q, layer.weight, s, layer.weight_scale, dtype, bias)
    )


def test_float32_activations_stay_on_triton(onednn_op, triton_calls):
    qx = torch.randint(-127, 128, (100, 8), dtype=torch.int8)
    w = torch.randint(-127, 128, (8, 4), dtype=torch.int8)
    out = torch.ops.vllm.xpu_int8_scaled_mm_func(
        qx, w, torch.ones(100, 1), torch.ones(4, 1), torch.float32, None
    )
    assert out.dtype == torch.float32
    assert triton_calls == [(100, 8)]
    assert onednn_op == []
