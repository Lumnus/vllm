# SPDX-License-Identifier: Apache-2.0
"""B70-0057: Qwen4Exp DFlash2 target-feature contract (CPU, no device)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import model as m


def _stub(hc=4, hidden=8, layers=48, start=0, end=48):
    obj = m.Qwen4ExpModel.__new__(m.Qwen4ExpModel)
    torch.nn.Module.__init__(obj)
    obj.config = SimpleNamespace(hc_count=hc, hidden_size=hidden, num_hidden_layers=layers)
    obj.start_layer, obj.end_layer = start, end
    return obj


def test_boundaries_are_completed_layers_plus_one():
    assert m._B70_DFLASH2_AUX_BOUNDARIES == (6, 16, 26, 36, 46)


def test_set_accepts_contract_and_empty(monkeypatch):
    obj = _stub()
    monkeypatch.setattr(obj, "_cache_aux_pp_layout", lambda: None)
    obj._set_aux_hidden_state_layers((46, 6, 16, 26, 36))
    assert obj.aux_hidden_state_layers == (6, 16, 26, 36, 46)
    obj._set_aux_hidden_state_layers(())
    assert obj.aux_hidden_state_layers == ()


@pytest.mark.parametrize("layers", [(2, 24, 45), (6, 16, 26, 36), (5, 15, 25, 35, 45)])
def test_set_rejects_other_layers(layers):
    with pytest.raises(ValueError):
        _stub()._set_aux_hidden_state_layers(layers)


@pytest.mark.parametrize("kw", [dict(hc=2), dict(layers=40, end=40), dict(start=24)])
def test_set_rejects_other_shapes(kw):
    with pytest.raises(ValueError):
        _stub(**kw)._set_aux_hidden_state_layers(m._B70_DFLASH2_AUX_BOUNDARIES)


def test_capture_materializes_delayed_combine_and_means_streams():
    obj = _stub(hc=4, hidden=8)
    hs = torch.randn(3, 32)
    bo, inj = torch.randn(3, 8), torch.randn(3, 4)
    combined = torch.randn(3, 32)
    calls = []

    def combine(h, b, i):
        calls.append((h, b, i))
        return combined

    layer = SimpleNamespace(mlp_hyper_connection=SimpleNamespace(combine=combine))
    out: list[torch.Tensor] = []
    obj._b70_capture_completed_layer(out, layer, hs, bo, inj)
    assert len(calls) == 1 and calls[0][0] is hs
    torch.testing.assert_close(out[0], combined.view(3, 4, 8).mean(dim=1))
    # the stream state handed to the next layer is untouched
    assert out[0].data_ptr() != combined.data_ptr()


def test_capture_without_pending_combine_uses_state():
    obj = _stub(hc=4, hidden=8)
    hs = torch.randn(2, 32)
    layer = SimpleNamespace(mlp_hyper_connection=SimpleNamespace(combine=None))
    out: list[torch.Tensor] = []
    obj._b70_capture_completed_layer(out, layer, hs, None, None)
    torch.testing.assert_close(out[0], hs.view(2, 4, 8).mean(dim=1))


def test_top_level_classes_expose_eagle3():
    from vllm.model_executor.models.interfaces import supports_eagle3

    assert supports_eagle3(m.Qwen4ExpForCausalLM)
    assert supports_eagle3(m.Qwen4ExpForConditionalGeneration)
    assert issubclass(m.Qwen4ExpModel, m.EagleModelMixin)
