# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_XPU_ATTN_META_ONCE on CPU: the XPU FA wrapper hands vllm-xpu-kernels
one metadata cache per forward context (the same dict to every layer of a
forward, a new one for the next forward), and nothing when the switch is off,
outside a forward or with a kernels build without the cache. The values the
kernels build from it are checked in vllm-xpu-kernels
(tests/flash_attn/test_spec_decode_meta_cpu.py)."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm import forward_context


@pytest.fixture(scope="module")
def _loaded():
    """vllm._xpu_ops imported once over stand-ins of the kernels' modules (it
    registers custom ops at import, so it cannot import twice)."""
    calls = []
    fai = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")

    def flash_attn_varlen_func(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    fai.flash_attn_varlen_func = flash_attn_varlen_func
    fai.SPEC_DECODE_META_CACHE = True
    rotary = types.ModuleType("vllm_xpu_kernels.rotary")
    rotary.apply_rotary_emb = lambda *a, **k: None
    pkg = types.ModuleType("vllm_xpu_kernels")
    pkg.flash_attn_interface = fai
    pkg.rotary = rotary
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "vllm_xpu_kernels", pkg)
        mp.setitem(sys.modules, "vllm_xpu_kernels.flash_attn_interface", fai)
        mp.setitem(sys.modules, "vllm_xpu_kernels.rotary", rotary)
        module = sys.modules.get("vllm._xpu_ops") or importlib.import_module(
            "vllm._xpu_ops"
        )
        yield module, fai, calls


@pytest.fixture
def xpu_ops_module(_loaded):
    _loaded[2].clear()
    return _loaded


def _call(module):
    q = torch.zeros(4, 2, 8)
    kv = torch.zeros(3, 16, 2, 8)
    return module.xpu_ops.flash_attn_varlen_func(
        q,
        kv,
        kv,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=20,
        causal=True,
        block_table=torch.zeros(1, 3, dtype=torch.int32),
        seqused_k=torch.tensor([20], dtype=torch.int32),
    )


def test_one_cache_per_forward(monkeypatch, xpu_ops_module):
    module, fai, calls = xpu_ops_module
    monkeypatch.setenv("VLLM_XPU_ATTN_META_ONCE", "1")
    first, second = SimpleNamespace(), SimpleNamespace()
    with forward_context.override_forward_context(first):
        for _ in range(12):
            _call(module)
    with forward_context.override_forward_context(second):
        _call(module)
    caches = [c["spec_decode_meta_cache"] for c in calls]
    assert all(c is caches[0] for c in caches[:12])
    assert caches[12] is not caches[0]
    assert first._xpu_fa_meta_cache is caches[0]
    assert second._xpu_fa_meta_cache is caches[12]


def test_off_outside_or_unsupported(monkeypatch, xpu_ops_module):
    module, fai, calls = xpu_ops_module
    monkeypatch.setenv("VLLM_XPU_ATTN_META_ONCE", "0")
    with forward_context.override_forward_context(SimpleNamespace()):
        _call(module)
    monkeypatch.setenv("VLLM_XPU_ATTN_META_ONCE", "1")
    _call(module)  # no forward context
    monkeypatch.setattr(fai, "SPEC_DECODE_META_CACHE", False)
    with forward_context.override_forward_context(SimpleNamespace()):
        _call(module)
    assert len(calls) == 3
    assert all("spec_decode_meta_cache" not in c for c in calls)
