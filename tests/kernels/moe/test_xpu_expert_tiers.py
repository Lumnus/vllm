# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert slots in device and host tiers on XPU (--cpu-offload-per-expert):
the plan and the EPLB surface on CPU, the split layer on a card."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig, UVAOffloadConfig
from vllm.distributed.eplb.expert_weights import TieredExpertWeight
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod

E, EP = 16, 2
NAME = "model.language_model.layers.3.mlp.experts"
HIDDEN, INTER, GROUP = 256, 128, 64


class _Linear:
    """ExpertMapManager stand-in: linear placement, E // EP slots per rank."""

    def __init__(self, rank: int):
        self.rank = rank

    def map_global_to_local(self, g: int) -> int:
        per = E // EP
        return (
            g - self.rank * per if self.rank * per <= g < (self.rank + 1) * per else -1
        )


class _GPTQ(AutoGPTQMoEMethod):
    """The real method's create_weights without its backend selection."""

    def __init__(self, bits: int = 4, group_size: int = GROUP):
        self.quant_config = SimpleNamespace(
            weight_bits=bits, pack_factor=32 // bits, group_size=group_size
        )
        self.moe = SimpleNamespace(w13_num_shards=2)


class _GPTQ8(_GPTQ):
    def __init__(self):
        super().__init__(bits=8)


class OtherMoEMethod:
    pass


def _params() -> dict:
    return {
        "hidden_size": HIDDEN,
        "intermediate_size_per_partition": INTER,
        "params_dtype": torch.bfloat16,
    }


def _config(gb=1.0, per_expert=True, **kwargs) -> OffloadConfig:
    uva = UVAOffloadConfig(cpu_offload_gb=gb, cpu_offload_per_expert=per_expert)
    return OffloadConfig(uva=uva, **kwargs)


@pytest.fixture
def xpu(monkeypatch):
    monkeypatch.setattr(xt, "current_platform", SimpleNamespace(is_xpu=lambda: True))
    monkeypatch.setattr(
        "vllm.utils.platform_utils.is_uva_available", lambda: True, raising=True
    )
    monkeypatch.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", raising=False)
    return monkeypatch


def _tier(monkeypatch, config, rank=0, name=NAME, fast=3, method=_GPTQ, eplb=True):
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            offload_config=config,
            parallel_config=SimpleNamespace(enable_eplb=eplb),
        ),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=48)
            )
        ),
    )
    # A budget that buys E // EP - fast slow slots per layer.
    quant_method = method()
    if isinstance(quant_method, _GPTQ):
        bytes_each = xt.expert_bytes(quant_method, **_params())
        slow = E // EP - fast
        config.uva.cpu_offload_gb = (48 * slow * bytes_each + bytes_each // 2) / 2**30
    return xt.maybe_tier(
        name, SimpleNamespace(num_experts=E), _Linear(rank), quant_method, _params()
    )


def test_selected_only_with_the_flag_and_a_budget_on_xpu(xpu):
    assert xt.selects_expert_tiers(_config())
    assert not xt.selects_expert_tiers(_config(gb=0))
    assert not xt.selects_expert_tiers(_config(per_expert=False))
    assert not xt.selects_expert_tiers(
        _config(prefetch=PrefetchOffloadConfig(offload_group_size=4))
    )
    xpu.setattr(xt, "current_platform", SimpleNamespace(is_xpu=lambda: False))
    assert not xt.selects_expert_tiers(_config())


def test_refused_when_uva_is_disabled_or_unavailable(xpu):
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    xpu.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    with pytest.raises(ValueError, match="VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"):
        create_offloader(_config())
    # Without the flag the regular offloader takes it, with its copy fallback.
    offloader = create_offloader(_config(per_expert=False))
    assert isinstance(offloader, UVAOffloader) and not offloader.uva_offloading
    xpu.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA")
    xpu.setattr("vllm.utils.platform_utils.is_uva_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        xt.selects_expert_tiers(_config())


def test_the_offloader_leaves_the_budget_to_the_slots(xpu):
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    offloader = create_offloader(_config(gb=8))
    assert isinstance(offloader, UVAOffloader)
    assert offloader.cpu_offload_max_bytes == 0


def test_needs_eplb(xpu):
    with pytest.raises(ValueError, match="--enable-eplb"):
        _tier(xpu, _config(), eplb=False)


def test_expert_bytes_match_create_weights():
    layer = torch.nn.Module()
    with torch.device("meta"):
        AutoGPTQMoEMethod.create_weights(
            _GPTQ(), layer=layer, num_experts=1, **_params()
        )
    built = sum(
        p.numel() * p.element_size()
        for name, p in layer.named_parameters()
        if name in ("w13_qweight", "w13_scales", "w2_qweight", "w2_scales")
    )
    assert xt.expert_bytes(_GPTQ(), **_params()) == built


def test_budget_keeps_one_fast_slot():
    assert xt.budget_slow_slots(1, 48, 2**20, 256) == 21
    assert xt.budget_slow_slots(1000, 48, 2**20, 256) == 255
    assert xt.budget_slow_slots(0.0001, 48, 2**20, 256) == 0


def test_slots_split_by_index_and_maps_cover_each_slot_once(xpu):
    for rank in range(EP):
        tiers = _tier(xpu, _config(), rank=rank)
        assert (tiers.num_local, tiers.num_fast, tiers.num_slow) == (8, 3, 5)
        fast_map, slow_map = xt._slot_maps(tiers.local_of, E, tiers.num_fast)
        own = range(rank * 8, rank * 8 + 8)
        assert [int(fast_map[g]) for g in own] == [0, 1, 2, -1, -1, -1, -1, -1]
        assert [int(slow_map[g]) for g in own] == [-1, -1, -1, 0, 1, 2, 3, 4]
        others = [g for g in range(E) if g not in own]
        assert all(int(fast_map[g]) == int(slow_map[g]) == -1 for g in others)


def test_no_tiers_outside_decoder_layers_or_for_other_methods(xpu):
    config = _config()
    assert _tier(xpu, config, name="mtp.layers.0.mlp.experts") is None
    assert _tier(xpu, config, method=OtherMoEMethod) is None
    assert _tier(xpu, config, method=_GPTQ8) is None


def test_loader_routes_slow_slots_to_twins(xpu):
    tiers = _tier(xpu, _config())
    layer = torch.nn.Module()
    w = torch.nn.Parameter(
        torch.zeros(tiers.num_fast, 4, 6, dtype=torch.int32), requires_grad=False
    )
    setattr(w, "is_transposed", True)  # noqa: B010 (a weight attr, not a field)
    layer.register_parameter("w13_qweight", w)
    tiers.attach_twins(layer)
    twin = tiers.twins["w13_qweight"]
    assert twin.shape == (tiers.num_slow, 4, 6) and twin.device.type == "cpu"
    assert twin.is_transposed is True
    assert tiers.route(w, 2) == (w, 2)
    param, row = tiers.route(w, 4)  # slot 4 is the second slow slot
    assert param is twin and row == 1


def test_arena_packs_blocks_into_chunks(monkeypatch):
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: torch.zeros(*a, **k)
    )
    arena = xt._PinnedArena()
    arena.CHUNK = 1 << 16
    a = arena.empty((3, 1000), torch.uint8)
    b = arena.empty((5000,), torch.uint8)
    big = arena.empty((1 << 17,), torch.uint8)
    assert len(arena.chunks) == 2 and arena.total == (1 << 16) + (1 << 17)
    assert b.data_ptr() - a.data_ptr() == 4096
    assert big.numel() == 1 << 17


def test_remap_unrouted_uses_an_expert_this_rank_does_not_own():
    ids = torch.tensor([[-1, 3], [7, -1]])
    assert xt.remap_unrouted(ids, 16, 0).tolist() == [[15, 3], [7, 15]]
    assert xt.remap_unrouted(ids, 16, 1).tolist() == [[0, 3], [7, 0]]


# -- the experts object: built at weight processing, EPLB sees both tiers --------


class _FakeXpuFusedMoe:
    """Rebinds `.data` like `XpuFusedMoe.__init__` for int4; `apply` adds, per
    routed (token, expert) that its `expert_map` maps, the expert's first w2
    byte, so the two tiers' outputs can be told apart."""

    def __init__(self, w13, w13_scales, w13_bias, w2, w2_scales, w2_bias, **kw):
        if not hasattr(w13, "xpu_fused_moe"):
            for w in (w13, w2):
                w.data = (w.data ^ 0x88).view(torch.int8).clone()
            w13.xpu_fused_moe = True
        self.w13, self.w2 = w13, w2
        self.gemm1_wei_scales, self.gemm2_wei_scales = w13_scales, w2_scales
        self.expert_map = kw["expert_map"]

    def apply(self, output, hidden_states, topk_weights, topk_ids, **kw):
        output.zero_()
        rows = self.expert_map[topk_ids.long()]
        for t in range(topk_ids.shape[0]):
            for row in rows[t].tolist():
                if row >= 0:
                    output[t] += float(self.w2[row].reshape(-1)[0])


def _built_layer(monkeypatch, fast=2, local=5):
    """An XPU int4 layer with `fast` of its `local` slots on the device, after
    the method's post-load step and ExpertTiers.finalize (host memory faked)."""
    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", _FakeXpuFusedMoe, raising=False)
    monkeypatch.setattr(xt, "_xpu_fused_moe", lambda: _FakeXpuFusedMoe)
    n, k = 4, 8

    def slot_bytes(slot):  # distinct per slot, as u4 before the s4 rebuild
        return torch.full((n, k // 2), 0x88 ^ (slot + 1), dtype=torch.uint8)

    layer = torch.nn.Module()
    layer.w13_qweight = torch.nn.Parameter(
        torch.stack([slot_bytes(s).repeat(2, 1) for s in range(fast)]),
        requires_grad=False,
    )
    layer.w2_qweight = torch.nn.Parameter(
        torch.stack([slot_bytes(s) for s in range(fast)]), requires_grad=False
    )
    layer.w13_scales = torch.nn.Parameter(
        torch.ones(fast, 2 * n, k // GROUP or 1).bfloat16(), requires_grad=False
    )
    layer.w2_scales = torch.nn.Parameter(
        torch.ones(fast, n, 1).bfloat16(), requires_grad=False
    )
    layer.w13_weight, layer.w2_weight = layer.w13_qweight, layer.w2_qweight
    layer.expert_map = None

    tiers = xt.ExpertTiers(NAME, {g: g for g in range(local)}, local, local, fast)
    slow = range(fast, local)
    tiers.slow = {
        "w13": torch.stack([(slot_bytes(s) ^ 0x88).repeat(2, 1) for s in slow]),
        "w2": torch.stack([slot_bytes(s) ^ 0x88 for s in slow]),
        "w13_scales": torch.ones(local - fast, 2 * n, 1).bfloat16(),
        "w2_scales": torch.ones(local - fast, n, 1).bfloat16(),
    }
    tiers.slow_views = dict(tiers.slow)  # CPU stands in for the UVA views
    fast_map, slow_map = xt._slot_maps(tiers.local_of, local, fast)
    tiers.fast_map, tiers.slow_map = fast_map, slow_map

    experts = object.__new__(xpu_moe.XPUExpertsWNA16)
    experts.moe_config = SimpleNamespace(
        experts_per_token=2,
        activation=MoEActivation.SILU,
        num_local_experts=local,
        ep_rank=0,
        ep_size=1,
    )
    experts.quant_config = SimpleNamespace(
        w1_scale=layer.w13_scales,
        w2_scale=layer.w2_scales,
        w1_bias=None,
        w2_bias=None,
        weight_quant_dtype="int4",
    )
    experts.gemm1_clamp_limit = None
    experts.fused_moe_impl = None
    experts.tiers = tiers
    return layer, experts, tiers


def test_eplb_sees_both_tiers_as_the_kernel_tensors(monkeypatch):
    layer, experts, tiers = _built_layer(monkeypatch)
    experts.finalize_weight_storage(layer)  # what get_expert_weights runs first
    weights = experts.eplb_expert_weights(layer)
    assert weights is not None and len(weights) == 4
    assert all(isinstance(w, TieredExpertWeight) for w in weights)
    assert [w.tier_sizes for w in weights] == [(2, 3)] * 4
    fast, slow = tiers._impls
    w13 = weights[0]
    assert w13.parts[0].data_ptr() == fast.w13.data_ptr()  # after the s4 rebuild
    assert w13.parts[1].data_ptr() == slow.w13.data_ptr()
    assert w13.dtype == torch.int8
    # slot s holds signed bytes (s + 1) in both tiers, as one rows space
    w2 = weights[1]
    assert [int(w2[s][0]) for s in range(5)] == [1, 2, 3, 4, 5]


def test_eplb_moves_rows_across_the_tiers_the_kernels_read(monkeypatch):
    """A move between a fast and a slow slot, through EPLB's transfer code,
    lands in the tensors both impls read."""
    import numpy as np

    from vllm.distributed.eplb.rebalance_execute import (
        move_from_buffer,
        move_to_buffer,
    )

    class _LocalOnly:
        needs_recv_rows = False

        def set_transfer_context(self, *args) -> None:
            pass

        def execute(self) -> None:
            pass

    layer, experts, tiers = _built_layer(monkeypatch)
    experts.finalize_weight_storage(layer)
    weights = [w.view(torch.uint8) for w in experts.eplb_expert_weights(layer)]
    buffers = [torch.empty(w.shape, dtype=w.dtype) for w in weights]
    old = np.arange(5)
    new = np.array([4, 1, 2, 3, 0])  # swap slot 0 (fast) and slot 4 (slow)
    meta = move_to_buffer(5, old, new, weights, buffers, None, 0, _LocalOnly())
    move_from_buffer(weights, buffers, meta, new, 0)
    fast, slow = tiers._impls
    assert int(fast.w2[0].reshape(-1)[0]) == 5
    assert int(slow.w2[2].reshape(-1)[0]) == 1


def test_apply_sums_the_fast_and_the_slow_tier(monkeypatch):
    layer, experts, _ = _built_layer(monkeypatch)
    experts.finalize_weight_storage(layer)
    topk_ids = torch.tensor([[0, 4], [1, 3]], dtype=torch.int32)
    out = torch.empty(2, 3)
    experts.apply(
        output=out,
        hidden_states=torch.zeros(2, 3),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=torch.ones(2, 2),
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=5,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    # token 0: slot 0 (fast, 1) + slot 4 (slow, 5); token 1: 2 + 4
    assert out[:, 0].tolist() == [6.0, 6.0]


def _xpu_ready() -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        from vllm_xpu_kernels.fused_moe_interface import FUSEDMOE_AVAILABLE
    except ImportError:
        return False
    return FUSEDMOE_AVAILABLE


@pytest.mark.skipif(not _xpu_ready(), reason="needs an XPU and its MoE kernels")
def test_split_layer_matches_the_resident_layer():
    """Fast slots in VRAM plus slow slots on pinned host views, summed, equal
    the layer with every expert in VRAM up to the order of the sum."""
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe, implement_zp

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dev = torch.device("xpu:0")
    e, k, inter, group, topk, tokens, fast = 32, 2560, 640, 128, 8, 24, 11
    gen = torch.Generator().manual_seed(5)
    w13 = torch.randint(
        0, 256, (e, 2 * inter, k // 2), dtype=torch.uint8, generator=gen
    )
    w2 = torch.randint(0, 256, (e, k, inter // 2), dtype=torch.uint8, generator=gen)
    s13 = (torch.rand(e, 2 * inter, k // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )
    s2 = (torch.rand(e, k, inter // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )
    ids = torch.stack([torch.randperm(e, generator=gen)[:topk] for _ in range(tokens)])
    weights = torch.softmax(torch.randn(tokens, topk, generator=gen), dim=-1)
    x = (torch.randn(tokens, k, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    topk_w, topk_ids = weights.to(dev), ids.to(dev, torch.int32)

    full = XpuFusedMoe(
        w13=torch.nn.Parameter(w13.to(dev), requires_grad=False),
        w13_scales=s13.to(dev),
        w13_bias=None,
        w2=torch.nn.Parameter(w2.to(dev), requires_grad=False),
        w2_scales=s2.to(dev),
        w2_bias=None,
        n_experts_per_token=topk,
        activation="silu",
        num_experts=e,
    )
    ref = torch.empty_like(x)
    full.apply(ref, x, topk_w, topk_ids)

    tiers = xt.ExpertTiers(NAME, {g: g for g in range(e)}, e, e, fast)
    host = {
        "w13": torch.stack([implement_zp(w) for w in w13[fast:]]).pin_memory(),
        "w2": torch.stack([implement_zp(w) for w in w2[fast:]]).pin_memory(),
        "w13_scales": s13[fast:].contiguous().pin_memory(),
        "w2_scales": s2[fast:].contiguous().pin_memory(),
    }
    tiers.slow_views = {
        key: get_accelerator_view_from_cpu_tensor(t) for key, t in host.items()
    }
    fast_map, slow_map = xt._slot_maps(tiers.local_of, e, fast)
    tiers.fast_map, tiers.slow_map = fast_map.to(dev), slow_map.to(dev)
    hot, cold = tiers.impls(
        torch.nn.Parameter(w13[:fast].to(dev), requires_grad=False),
        s13[:fast].to(dev),
        torch.nn.Parameter(w2[:fast].to(dev), requires_grad=False),
        s2[:fast].to(dev),
        topk,
        "silu",
        None,
    )
    out, cold_out = torch.empty_like(x), torch.empty_like(x)
    hot.apply(out, x, topk_w, topk_ids)
    cold.apply(cold_out, x, topk_w, topk_ids)
    out.add_(cold_out)
    torch.xpu.synchronize()
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    assert err < 1e-2, err
    tiered = tiers.eplb_weights()
    assert tiered[0].parts[1].data_ptr() == host["w13"].data_ptr()


def test_eplb_writes_into_the_slow_tier_drop_staged_copies(monkeypatch):
    """A row EPLB writes into a slow slot invalidates the offloader's staged
    copies; writes into fast slots do not."""
    import vllm.model_executor.offloader.base as offloader_base

    calls: list[int] = []
    monkeypatch.setattr(
        offloader_base,
        "get_offloader",
        lambda: SimpleNamespace(invalidate_staged=lambda: calls.append(1)),
    )
    layer, experts, _ = _built_layer(monkeypatch)
    experts.finalize_weight_storage(layer)
    w2 = experts.eplb_expert_weights(layer)[1]
    w2.rows_written([0, 1])
    assert calls == []
    w2.rows_written([1, 3])
    assert calls == [1]
