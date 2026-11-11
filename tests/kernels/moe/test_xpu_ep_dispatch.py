# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating of the XPU expert-parallel MoE dispatch (host-side logic only)."""

import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import xpu_ep_dispatch as ep


def _body(x):
    # comment that the fingerprint ignores
    return x + 1


def _body_moved(x):
    return x + 1


def _body_changed(x):
    return x + 2


def test_fingerprint_tracks_the_definition_not_its_comments():
    assert ep.fingerprint(_body) == ep.fingerprint(_body)
    assert ep.fingerprint(_body) != ep.fingerprint(_body_changed)
    # The name is part of the definition.
    assert ep.fingerprint(_body) != ep.fingerprint(_body_moved)


class _Impl:
    _use_ref = False

    def _apply_kernel(self):
        return None


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(ep, "_armed", True)


def _t(dtype):
    return torch.empty((2, 4), dtype=dtype)


def test_eligible_on_the_kernel_path(armed):
    assert ep.eligible(_Impl(), _t(torch.bfloat16), _t(torch.bfloat16), {})


@pytest.mark.parametrize(
    "impl,hidden,out,kwargs",
    [
        (types.SimpleNamespace(_use_ref=True), torch.bfloat16, torch.bfloat16, {}),
        (_Impl(), torch.bfloat16, torch.bfloat16, {"w4a8": True}),
        (_Impl(), torch.float32, torch.float32, {}),
        (_Impl(), torch.bfloat16, torch.float16, {}),
    ],
)
def test_not_eligible(armed, impl, hidden, out, kwargs):
    assert not ep.eligible(impl, _t(hidden), _t(out), kwargs)


def test_unknown_wheel_keeps_its_own_path(monkeypatch):
    monkeypatch.setattr(ep, "_armed", None)
    assert not ep.eligible(_Impl(), _t(torch.bfloat16), _t(torch.bfloat16), {})
    assert ep._armed is False
