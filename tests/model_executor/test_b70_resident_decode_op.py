# SPDX-License-Identifier: Apache-2.0
"""B70-0054 on the CPU: expert maps for the all-resident decode op, the switch, and the per-batch decision.
The op itself is checked on a card by kernels/b70-moe-decode/test_moe_decode_resident.py (engine stopped)."""
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import b70_hotcold
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsWNA16, b70_resident_maps

E, TOPK = 512, 10


def ep_map(rank, size):
    """What _moe_C.init_expert_map builds (linear placement): rank r owns [r*E/size, (r+1)*E/size)."""
    n = E // size
    m = torch.full((E,), -1, dtype=torch.int32)
    m[rank * n:(rank + 1) * n] = torch.arange(n, dtype=torch.int32)
    return m


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_ep4_maps(rank):
    hot, cold = b70_resident_maps(128, ep_map(rank, 4), None, "cpu")
    assert hot.dtype == torch.int32 and torch.equal(hot, ep_map(rank, 4))
    assert cold.shape == (E,) and bool((cold == -1).all())


def test_no_ep_map_is_identity():
    hot, cold = b70_resident_maps(E, None, None, "cpu")
    assert torch.equal(hot, torch.arange(E, dtype=torch.int32)) and bool((cold == -1).all())


def test_vllm_map_must_match():
    assert isinstance(b70_resident_maps(128, ep_map(1, 4), ep_map(2, 4), "cpu"), str)
    hot, _ = b70_resident_maps(128, ep_map(1, 4), ep_map(1, 4).long(), "cpu")
    assert torch.equal(hot, ep_map(1, 4))


def test_map_must_cover_local_slots():
    bad = ep_map(0, 4)
    bad[5] = -1
    assert isinstance(b70_resident_maps(128, bad, None, "cpu"), str)
    dup = ep_map(0, 4)
    dup[200] = 3
    assert isinstance(b70_resident_maps(128, dup, None, "cpu"), str)


def _experts():
    ex = object.__new__(XPUExpertsWNA16)  # no __init__: it asks the device for its architecture
    ex.gemm1_clamp_limit = None
    return ex


def test_switch_off_means_stock(monkeypatch):
    monkeypatch.delenv("B70_MOE_DECODE_OP_RESIDENT", raising=False)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    ex = _experts()
    ex.fused_moe_impl = SimpleNamespace()
    x = torch.zeros(4, 2560, dtype=torch.bfloat16)
    assert ex._b70_resident_decode_op(x, x, None, None, MoEActivation.SILU, None) is False
    assert ex._b70_decode_args is False


def test_refuses_without_library(monkeypatch):
    monkeypatch.setenv("B70_MOE_DECODE_OP_RESIDENT", "1")
    monkeypatch.setattr(b70_hotcold, "decode_op_max_rows", lambda: 0)
    ex = _experts()
    ex.fused_moe_impl = SimpleNamespace(is_int4=True)
    assert ex._b70_resident_setup(MoEActivation.SILU, None) is False


def test_refuses_cpu_weights_and_non_silu(monkeypatch):
    monkeypatch.setenv("B70_MOE_DECODE_OP_RESIDENT", "1")
    monkeypatch.setattr(b70_hotcold, "decode_op_max_rows", lambda: 16)
    w = torch.zeros(2, 1280, 1280, dtype=torch.int8)
    s = torch.zeros(2, 1280, 20, dtype=torch.bfloat16)
    impl = SimpleNamespace(is_int4=True, w13_bias=None, w2_bias=None, w13=w, gemm1_wei_scales=s, w2=w,
                           gemm2_wei_scales=s, num_experts=2, expert_map=None)
    ex = _experts()
    ex.fused_moe_impl = impl
    assert ex._b70_resident_setup(MoEActivation.GELU, None) is False
    assert ex._b70_resident_setup(MoEActivation.SILU, None) is False  # CPU tensors: not on the device


def test_rows_decide_op_or_stock(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.ops, "b70moe", SimpleNamespace(moe_decode=lambda *a: calls.append(a)), raising=False)
    monkeypatch.setattr(b70_hotcold, "decode_op_max_rows", lambda: 16)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True)
    ex = _experts()
    ex._b70_decode_args = ("hm", "cm", 1, 2, 3, 4, 1, 2, 3, 4)
    for m, taken in [(1, True), (16, True), (17, False), (0, False), (64, False)]:
        x = torch.zeros(m, 2560, dtype=torch.bfloat16)
        ids = torch.zeros(m, TOPK, dtype=torch.int64)
        tw = torch.zeros(m, TOPK, dtype=torch.float32)
        assert ex._b70_resident_decode_op(torch.empty_like(x), x, tw, ids, MoEActivation.SILU, None) is taken
    assert len(calls) == 2 and all(c[3].dtype == torch.int32 for c in calls)
    assert calls[0][4:] == ("hm", "cm", 1, 2, 3, 4, 1, 2, 3, 4)


def test_undecided_inside_capture_stays_stock(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True)
    ex = _experts()
    x = torch.zeros(4, 2560, dtype=torch.bfloat16)
    assert ex._b70_resident_decode_op(x, x, None, None, MoEActivation.SILU, None) is False
    assert ex._b70_decode_args is None  # decided later, eagerly


def test_refuses_group_size_other_than_128(monkeypatch):
    monkeypatch.setenv("B70_MOE_DECODE_OP_RESIDENT", "1")
    monkeypatch.setattr(b70_hotcold, "decode_op_max_rows", lambda: 16)
    calls = {}

    class Fake(torch.Tensor):
        pass

    w13 = torch.zeros(2, 1280, 1280, dtype=torch.int8)
    s13 = torch.zeros(2, 1280, 80, dtype=torch.bfloat16)  # g32
    w2 = torch.zeros(2, 2560, 320, dtype=torch.int8)
    s2 = torch.zeros(2, 2560, 20, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("xpu", 0)))
    impl = SimpleNamespace(is_int4=True, w13_bias=None, w2_bias=None, w13=w13, gemm1_wei_scales=s13, w2=w2,
                           gemm2_wei_scales=s2, num_experts=2, expert_map=None)
    ex = _experts()
    ex.fused_moe_impl = impl
    assert ex._b70_resident_setup(MoEActivation.SILU, None) is False
