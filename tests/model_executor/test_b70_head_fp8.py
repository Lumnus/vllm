"""B70-0053: the target's lm_head as FP8 W8A16, one FP8 copy shared with the draft (0048), BF16 shard released."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import mtp as mtp_mod
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP, _B70Fp8Head, b70_fp8_head_of


class _Head(torch.nn.Module):
    def __init__(self, rows=1000, k=256):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(rows, k) * 0.02).to(torch.bfloat16), requires_grad=False)
        self.quant_method = object()
        self.tp_size, self.shard_indices = 2, None


@pytest.fixture
def xpu(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)


def _dequant(head8):
    return (head8.w8.float() * head8.scale).t()


def test_builder_caches_and_is_accurate():
    head = _Head()
    h1 = b70_fp8_head_of(head)
    assert isinstance(h1, _B70Fp8Head) and b70_fp8_head_of(head) is h1
    assert h1.w8.shape == (256, 1000) and h1.scale.shape == (1, 1000)
    w = head.weight.float()
    assert ((_dequant(h1) - w).norm() / w.norm()).item() < 0.04


def test_builder_none_without_bf16_weight():
    head = _Head()
    head.weight.data = torch.empty((0, 256), dtype=torch.bfloat16)
    assert b70_fp8_head_of(head) is None


def _target(head):
    t = SimpleNamespace(lm_head=head)
    t._b70_prepare_head_fp8 = lambda: Qwen4ExpForCausalLM._b70_prepare_head_fp8(t)
    return t


def test_target_off_by_default(monkeypatch, xpu):
    monkeypatch.delenv("B70_HEAD_FP8", raising=False)
    head = _Head()
    qm = head.quant_method
    _target(head)._b70_prepare_head_fp8()
    assert head.quant_method is qm and head.weight.shape == (1000, 256)


def test_target_on_releases_bf16_and_shares_with_draft(monkeypatch, xpu):
    monkeypatch.setenv("B70_HEAD_FP8", "1")
    monkeypatch.setenv("B70_DRAFT_HEAD_FP8", "1")
    head = _Head()
    w = head.weight.detach().float().clone()
    _target(head)._b70_prepare_head_fp8()
    head8 = head._b70_fp8_head
    assert head.quant_method is head8
    assert head.weight.numel() == 0 and head.weight.shape[1] == 256
    assert ((_dequant(head8) - w).norm() / w.norm()).item() < 0.04
    draft = SimpleNamespace(lm_head=head)
    Qwen4ExpMTP._b70_prepare_draft_head(draft)
    assert draft._b70_head8 is head8


def test_draft_first_then_target_reuses(monkeypatch, xpu):
    monkeypatch.setenv("B70_HEAD_FP8", "1")
    monkeypatch.setenv("B70_DRAFT_HEAD_FP8", "1")
    head = _Head()
    draft = SimpleNamespace(lm_head=head)
    Qwen4ExpMTP._b70_prepare_draft_head(draft)
    _target(head)._b70_prepare_head_fp8()
    assert head.quant_method is draft._b70_head8 and head.weight.numel() == 0


def test_draft_alone_unchanged(monkeypatch, xpu):
    """0048 without 0053: the draft gets its FP8 copy, the BF16 head stays for the target."""
    monkeypatch.delenv("B70_HEAD_FP8", raising=False)
    monkeypatch.setenv("B70_DRAFT_HEAD_FP8", "1")
    head = _Head()
    qm = head.quant_method
    draft = SimpleNamespace(lm_head=head)
    Qwen4ExpMTP._b70_prepare_draft_head(draft)
    assert isinstance(draft._b70_head8, _B70Fp8Head)
    assert head.quant_method is qm and head.weight.shape == (1000, 256)
    assert mtp_mod.b70_fp8_head_of(head) is draft._b70_head8


def test_once_runs_prepare_a_single_time(monkeypatch, xpu):
    monkeypatch.setenv("B70_HEAD_FP8", "1")
    calls = []
    t = SimpleNamespace()
    t._b70_prepare_head_fp8 = lambda: calls.append(1)
    Qwen4ExpForCausalLM._b70_head_fp8_once(t)
    Qwen4ExpForCausalLM._b70_head_fp8_once(t)
    assert calls == [1]
