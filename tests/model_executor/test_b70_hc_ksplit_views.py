"""B70-0052: the HC down GEMM K-halves are views of the one weight (no second copy) unless B70_HC_KSPLIT_VIEWS=0."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual


def _fake(use_combine: bool, n: int = 336, k2: int = 10240):
    weight = torch.nn.Parameter(torch.randn(n, k2).to(torch.bfloat16), requires_grad=False)
    linear = SimpleNamespace(weight=weight)
    return SimpleNamespace(
        _kdown_lo=None,
        _kdown_hi=None,
        _kdown_k=0,
        use_combine=use_combine,
        input_mix_weight_down_block_inject=linear,
        input_mix_weight_down=linear,
    )


def _prepare(fake):
    GatedResidual._prepare_kdown_ksplit(fake)
    return fake


@pytest.mark.parametrize("use_combine,n", [(True, 336), (False, 320)])
def test_views_by_default(monkeypatch, use_combine, n):
    monkeypatch.setenv("B70_HC_KSPLIT_VIEWS", "1")
    monkeypatch.delenv("B70_HC_KSPLIT_VIEWS")
    f = _prepare(_fake(use_combine, n))
    w = f.input_mix_weight_down.weight.data
    k = w.shape[1] // 2
    assert f._kdown_k == k
    assert f._kdown_lo.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    assert f._kdown_hi.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    assert f._kdown_lo.data_ptr() == w.data_ptr()
    assert f._kdown_hi.data_ptr() == w[:, k:].data_ptr()
    assert f._kdown_lo.stride() == (2 * k, 1) and f._kdown_hi.stride() == (2 * k, 1)
    assert torch.equal(f._kdown_lo, w[:, :k]) and torch.equal(f._kdown_hi, w[:, k:])


def test_copies_when_off(monkeypatch):
    monkeypatch.setenv("B70_HC_KSPLIT_VIEWS", "0")
    f = _prepare(_fake(True))
    w = f.input_mix_weight_down.weight.data
    k = w.shape[1] // 2
    assert f._kdown_lo.is_contiguous() and f._kdown_hi.is_contiguous()
    assert f._kdown_lo.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()
    assert torch.equal(f._kdown_lo, w[:, :k]) and torch.equal(f._kdown_hi, w[:, k:])


def test_split_matches_full_gemm_shape(monkeypatch):
    """The two half GEMMs over the views add up to the full GEMM (fp32 on CPU, tolerance for the summation order)."""
    monkeypatch.delenv("B70_HC_KSPLIT_VIEWS", raising=False)
    f = _prepare(_fake(True, 336, 1024))
    w = f.input_mix_weight_down.weight.data.float()
    x = torch.randn(80, 1024)
    k = f._kdown_k
    split = torch.nn.functional.linear(x[:, :k], f._kdown_lo.float()) + torch.nn.functional.linear(
        x[:, k:], f._kdown_hi.float()
    )
    torch.testing.assert_close(split, torch.nn.functional.linear(x, w), rtol=1e-4, atol=1e-3)


def test_prepare_is_idempotent(monkeypatch):
    monkeypatch.delenv("B70_HC_KSPLIT_VIEWS", raising=False)
    f = _prepare(_fake(True))
    lo = f._kdown_lo
    _prepare(f)
    assert f._kdown_lo is lo
