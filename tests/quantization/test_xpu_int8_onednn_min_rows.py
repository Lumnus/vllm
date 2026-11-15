# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_XPU_INT8_ONEDNN_MIN_ROWS: where the XPU W8A8 INT8 linear switches from
the Triton kernel to oneDNN (CPU, a stand-in op with the kernels' schema)."""

import os
import subprocess
import sys

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mm

_SCHEMA = (
    "int8_gemm_w8a8(Tensor A_, Tensor A_scale, Tensor? A_zp, Tensor B, "
    "Tensor B_scale, Tensor? B_zp, int group_size, Tensor? bias, "
    "ScalarType? out_dtype=None) -> Tensor"
)
_ENV = "VLLM_XPU_INT8_ONEDNN_MIN_ROWS"


def _reference(qx, weight, x_scale, weight_scale, out_dtype, bias):
    out = (
        (qx.float() @ weight.float())
        * x_scale.float()
        * weight_scale.float().view(1, -1)
    )
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@pytest.fixture
def onednn_op():
    calls = []
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def impl(A, A_scale, A_zp, B, B_scale, B_zp, group_size, bias, out_dtype):
            calls.append(A.shape[0])
            return _reference(A, B, A_scale, B_scale, out_dtype, bias)

        lib.impl("int8_gemm_w8a8", impl, "CPU")
        yield calls


@pytest.fixture
def triton_calls(monkeypatch):
    calls = []

    def fake_triton(qx, weight, scale_a, scale_b, out_dtype, bias=None):
        calls.append(qx.shape[0])
        return _reference(qx, weight, scale_a, scale_b, out_dtype, bias)

    monkeypatch.setattr(xpu_mm, "triton_scaled_mm", fake_triton)
    return calls


def _run(rows):
    k, n = 32, 16
    qx = torch.randint(-127, 128, (rows, k), dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8).t()
    xs = torch.rand(rows, 1) * 0.01 + 0.001
    ws = torch.rand(n, 1) * 0.01 + 0.001
    out = torch.ops.vllm.xpu_int8_scaled_mm_func(qx, w, xs, ws, torch.bfloat16, None)
    torch.testing.assert_close(out, _reference(qx, w, xs, ws, torch.bfloat16, None))


def test_unset_is_the_base_split(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert envs.VLLM_XPU_INT8_ONEDNN_MIN_ROWS == 64
    assert xpu_mm.XPU_INT8_ONEDNN_MIN_ROWS == 64


def test_env_is_read(monkeypatch):
    monkeypatch.setenv(_ENV, "0")
    assert envs.VLLM_XPU_INT8_ONEDNN_MIN_ROWS == 0
    monkeypatch.setenv(_ENV, "100000")
    assert envs.VLLM_XPU_INT8_ONEDNN_MIN_ROWS == 100000


def test_negative_is_refused_at_import():
    # A fresh interpreter: the module registers a custom op at import, so it
    # cannot be imported twice in one process.
    env = dict(os.environ, **{_ENV: "-1"})
    proc = subprocess.run(
        [sys.executable, "-c", f"import {xpu_mm.__name__}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr and _ENV in proc.stderr


@pytest.mark.parametrize(
    "split,rows,expect",
    [
        (64, 64, "triton"),
        (64, 65, "onednn"),
        (0, 1, "onednn"),
        (0, 2048, "onednn"),
        (100000, 65, "triton"),
        (100000, 2048, "triton"),
        (128, 128, "triton"),
        (128, 129, "onednn"),
    ],
)
def test_split_follows_the_setting(
    onednn_op, triton_calls, monkeypatch, split, rows, expect
):
    monkeypatch.setattr(xpu_mm, "XPU_INT8_ONEDNN_MIN_ROWS", split)
    _run(rows)
    if expect == "triton":
        assert triton_calls == [rows] and onednn_op == []
    else:
        assert onednn_op == [rows] and triton_calls == []


def test_not_a_compile_factor(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    unset = envs.compile_factors()
    assert _ENV not in unset
    monkeypatch.setenv(_ENV, "0")
    assert envs.compile_factors() == unset
