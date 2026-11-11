# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU int8 activation quant: the vllm-xpu-kernels op when it is registered,
the torch.compile reference otherwise, with the same outputs (CPU, a stand-in
op with the kernels' schema)."""

import importlib.util
import sys
import types

import pytest
import torch

if importlib.util.find_spec("vllm_xpu_kernels") is None:
    # CPU run: the two names vllm._xpu_ops imports at module level.
    _pkg = types.ModuleType("vllm_xpu_kernels")
    _pkg.__path__ = []
    for _name, _attr in (
        ("flash_attn_interface", "flash_attn_varlen_func"),
        ("rotary", "apply_rotary_emb"),
    ):
        _mod = types.ModuleType(f"vllm_xpu_kernels.{_name}")
        setattr(_mod, _attr, None)
        setattr(_pkg, _name, _mod)
        sys.modules[_mod.__name__] = _mod
    sys.modules["vllm_xpu_kernels"] = _pkg

    from vllm import _xpu_ops  # noqa: E402

    # Keep the stand-in out of every other test's view.
    for _key in [k for k in sys.modules if k.startswith("vllm_xpu_kernels")]:
        del sys.modules[_key]

from vllm import _custom_ops as ops  # noqa: E402
from vllm import _xpu_ops  # noqa: E402
from vllm._xpu_ops import xpu_ops  # noqa: E402

_SCHEMA = (
    "dynamic_per_token_scaled_int8_quant(Tensor! result, Tensor input, "
    "Tensor! scale) -> ()"
)


def _reference_rule(x: torch.Tensor):
    xf = x.reshape(-1, x.shape[-1]).float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-5).to(x.dtype).float()
    q = torch.round(xf / scale).clamp(-128, 127).to(torch.int8)
    return q.view(x.shape), scale.view(x.shape[:-1] + (1,))


@pytest.fixture
def native_op():
    """Register the kernels' op under _C for the test only."""
    calls = []
    with torch.library._scoped_library("_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def impl(result, input, scale):
            calls.append(tuple(input.shape))
            q, s = _reference_rule(input)
            result.copy_(q)
            scale.copy_(s.to(scale.dtype))

        lib.impl("dynamic_per_token_scaled_int8_quant", impl, "CPU")
        yield calls


def test_probe_follows_registration(native_op):
    assert _xpu_ops.native_int8_quant_available()


def test_probe_without_op(monkeypatch):
    if hasattr(torch.ops._C, "dynamic_per_token_scaled_int8_quant"):
        pytest.skip("a build with the op is installed")
    assert not _xpu_ops.native_int8_quant_available()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dynamic_per_token_int8_quant_uses_native_op(native_op, dtype):
    torch.manual_seed(0)
    x = torch.randn(5, 64, dtype=dtype) * 3
    q, s, zp = xpu_ops.dynamic_per_token_int8_quant(x, True, 8)
    assert native_op == [(5, 64)]
    assert q.dtype == torch.int8 and s.dtype == dtype and zp.dtype == torch.int32
    assert s.shape == (5, 1) and zp.shape == (5, 1) and not zp.any()
    q_ref, s_ref = _reference_rule(x)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(s.float(), s_ref)


def test_asymmetric_or_other_bits_keep_reference(native_op, monkeypatch):
    called = []

    def ref(input, use_sym_quant, bits):
        called.append((use_sym_quant, bits))
        return None, None, None

    monkeypatch.setattr(xpu_ops, "dynamic_per_token_int8_quant_ref", ref)
    xpu_ops.dynamic_per_token_int8_quant(torch.zeros(2, 8), False, 8)
    xpu_ops.dynamic_per_token_int8_quant(torch.zeros(2, 8), True, 4)
    assert called == [(False, 8), (True, 4)] and native_op == []


def test_scaled_int8_quant_on_xpu_uses_native_op(native_op, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    torch.manual_seed(1)
    x = torch.randn(3, 4, 32, dtype=torch.bfloat16)
    q, s, azp = ops.scaled_int8_quant(x)
    assert native_op == [(3, 4, 32)]
    assert azp is None and s.dtype == torch.float32 and s.shape == (12, 1)
    q_ref, s_ref = _reference_rule(x)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(s, s_ref.reshape(-1, 1))


def test_scaled_int8_quant_on_xpu_without_op(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(_xpu_ops, "native_int8_quant_available", lambda: False)
    seen = []

    def ref(input, use_sym_quant, bits):
        seen.append(tuple(input.shape))
        return _reference_rule(input) + (None,)

    monkeypatch.setattr(xpu_ops, "dynamic_per_token_int8_quant_ref", ref)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    q, s, _ = ops.scaled_int8_quant(x)
    assert seen == [(2, 16)] and s.dtype == torch.float32 and s.shape == (2, 1)
