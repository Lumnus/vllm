# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-C1-R1: C1 target-graph capture snapshots in host RAM, moved in chunks (CPU only).

The snapshot must be exact (save -> mutate -> restore gives the original bytes), unions of
indices over shared views must round-trip, and the equality check must see a one-element
difference in any chunk."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import b70_target_graph as tg


@pytest.fixture(params=[1 << 30, 3 * 4 * 5 * 4])  # one chunk / a few rows per chunk
def chunk(request, monkeypatch):
    monkeypatch.setattr(tg, "SNAPSHOT_CHUNK_BYTES", request.param)
    return request.param


def _routes():
    torch.manual_seed(0)
    ssm = torch.randn(40, 3, 4, 5)                      # fp32 state, 240 B per row
    conv = torch.randn(40, 6, 2).to(torch.bfloat16)
    shared = torch.randn(40, 7)                         # a view several layers share
    idx = torch.tensor([3, 9, 10, 11, 17, 25, 26, 39])
    return [("ssm", ssm, idx), ("conv", conv, idx.flip(0)),
            ("shared", shared, torch.cat([idx, idx + 0, torch.tensor([0, 1])]).unique())]


def test_snapshot_is_host_and_exact(chunk):
    routes = _routes()
    before = [t.clone() for _, t, _ in routes]
    snap = tg.save(routes)
    assert all(v.device.type == "cpu" for v in snap)
    for (_, t, i), v in zip(routes, snap):
        assert torch.equal(v, t[i])
    for _, t, i in routes:
        t[i] = torch.randn_like(t[i].float()).to(t.dtype)
    assert not tg.state_equal(routes, snap)
    tg.restore(routes, snap)
    assert all(torch.equal(t, b) for (_, t, _), b in zip(routes, before))
    assert tg.state_equal(routes, snap)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_one_element_difference_is_seen(chunk, which):
    routes = _routes()
    snap = tg.save(routes)
    _, t, i = routes[which]
    flat = t[i[-1]].view(-1)
    flat[-1] = flat[-1] + 1
    t[i[-1]] = flat.view(t[i[-1]].shape)
    assert not tg.state_equal(routes, snap)


def _runner(method="dflash", arch="DFlash2DraftModel", dev="xpu", mt="qwen4_exp_text"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method,
                                           draft_model_config=SimpleNamespace(architectures=[arch])),
        device=SimpleNamespace(type=dev),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=mt)))


def test_reserve_only_on_the_c1_path(monkeypatch):
    monkeypatch.delenv("VLLM_XPU_C1_TARGET_GRAPH", raising=False)
    assert tg.capture_reserve_bytes(_runner()) == 2 * tg.SNAPSHOT_CHUNK_BYTES
    assert tg.capture_reserve_bytes(_runner(method="mtp")) == 0
    assert tg.capture_reserve_bytes(_runner(arch="DFlashDraftModel")) == 0
    assert tg.capture_reserve_bytes(_runner(dev="cuda")) == 0
    assert tg.capture_reserve_bytes(_runner(mt="qwen3")) == 0
    assert tg.capture_reserve_bytes(SimpleNamespace(speculative_config=None)) == 0
    monkeypatch.setenv("VLLM_XPU_C1_TARGET_GRAPH", "0")
    assert tg.capture_reserve_bytes(_runner()) == 0
