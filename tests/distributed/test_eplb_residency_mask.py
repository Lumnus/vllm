# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EPLB residency mask: selection, the router's use of it, the step hook."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm._custom_ops as ops
import vllm.distributed.eplb.eplb_state as eplb_state_mod
from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.distributed.eplb.residency_mask import (
    DISPLACED_CALLS,
    DISPLACED_WEIGHT,
    ROUTED_CALLS,
    ResidencyMask,
    coldest_share_mask,
    logical_mask_from_residency,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)

from .eplb_single_rank import FakeMoEModel, add_model, single_rank_eplb

# ---------------------------------------------------------------- selection


def test_coldest_share_masks_the_least_routed_experts_up_to_the_share():
    load = np.array([[50, 1, 0, 30, 4, 15]], dtype=float)  # total 100
    # ascending: e2 (0), e1 (1), e4 (4), e5 (15), e3 (30), e0 (50)
    assert coldest_share_mask(load, 0.05, keep=2).nonzero()[1].tolist() == [1, 2, 4]
    assert coldest_share_mask(load, 0.049, keep=2).nonzero()[1].tolist() == [1, 2]
    assert coldest_share_mask(load, 0.20, keep=2).nonzero()[1].tolist() == [1, 2, 4, 5]
    # at least `keep` experts stay unmasked
    assert coldest_share_mask(load, 0.99, keep=2).sum() == 4


def test_coldest_share_ties_and_empty_layers_are_deterministic():
    load = np.array([[0, 0, 0, 0, 10, 10], [0, 0, 0, 0, 0, 0]], dtype=float)
    mask = coldest_share_mask(load, 0.01, keep=2)
    assert mask[0].tolist() == [True, True, True, True, False, False]
    assert not mask[1].any()  # no traffic: nothing masked
    tied = np.array([[5, 5, 5, 5]], dtype=float)
    assert coldest_share_mask(tied, 0.5, keep=1)[0].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert not coldest_share_mask(load, 0.0, keep=1).any()


def test_ranks_choose_the_same_mask_from_the_same_all_reduced_load():
    rng = np.random.default_rng(3)
    total = rng.integers(0, 50, size=(4, 64)).astype(float)
    part = rng.integers(0, 50, size=(4, 64)).astype(float)
    part = np.minimum(part, total)
    rank0, rank1 = part, total - part  # two ranks' counts, one sum
    a = coldest_share_mask(rank0 + rank1, 0.05, keep=8)
    b = coldest_share_mask(rank1 + rank0, 0.05, keep=8)
    assert np.array_equal(a, b)
    assert a.any()


def test_logical_mask_needs_every_replica_nonresident():
    # 4 logical experts, 6 physical slots: logical 0 and 1 have two replicas
    phys2log = np.array([[0, 1, 2, 3, 0, 1]])
    nonresident = np.array([[1, 1, 0, 1, 0, 0]])
    mask = logical_mask_from_residency(nonresident, phys2log, 4, keep=1)
    # 0: one replica resident; 1: one resident; 2: resident; 3: none
    assert mask[0].tolist() == [False, False, False, True]
    with pytest.raises(RuntimeError, match="fewer than top_k"):
        logical_mask_from_residency(nonresident, phys2log, 4, keep=4)


# ---------------------------------------------------------------- router


def _ref_topk_softmax(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    is_padding=None,
):
    probs = gating_output.float().softmax(dim=-1)
    w, i = probs.topk(topk_weights.shape[-1], dim=-1)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    topk_weights.copy_(w)
    topk_indices.copy_(i.to(topk_indices.dtype))


def _layer_state(num_experts: int, num_real: int = 1 << 20) -> EplbLayerState:
    """An EPLB layer state whose mapping is the identity (CPU: the EPLB map
    kernel is not used)."""
    state = EplbLayerState()
    state.expert_load_view = torch.zeros(num_experts, dtype=torch.int32)
    state.logical_to_physical_map = torch.arange(num_experts).unsqueeze(1)
    state.logical_replica_count = torch.ones(num_experts, dtype=torch.long)
    state.should_record_tensor = torch.ones((), dtype=torch.bool)
    state.num_unpadded_tokens_tensors = [torch.tensor(num_real, dtype=torch.int32)]
    return state


@pytest.fixture
def ref_softmax(monkeypatch):
    monkeypatch.setattr(ops, "topk_softmax", _ref_topk_softmax)


def _router(state: EplbLayerState, top_k: int = 4, num_experts: int = 16):
    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=num_experts,
        renormalize=True,
        eplb_state=state,
    )


def _route(router, logits):
    return router.select_experts(
        hidden_states=torch.zeros(logits.shape[0], 8),
        router_logits=logits,
        topk_indices_dtype=torch.int32,
    )


def test_mask_off_routes_bit_identically(ref_softmax):
    torch.manual_seed(0)
    logits = torch.randn(32, 16)
    plain_w, plain_ids = _route(_router(_layer_state(16)), logits)
    state = _layer_state(16)
    assert state.residency_mask is None
    w, ids = _route(_router(state), logits)
    assert torch.equal(ids, plain_ids)
    assert torch.equal(w, plain_w)
    # an all-false mask is a no-op too
    state.residency_mask = torch.zeros(16, dtype=torch.bool)
    w, ids = _route(_router(state), logits)
    assert torch.equal(ids, plain_ids)
    assert torch.equal(w, plain_w)


def test_masked_experts_never_reach_top_k_and_weights_renormalise(ref_softmax):
    torch.manual_seed(1)
    logits = torch.randn(64, 16)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[[0, 3, 5, 6, 11]] = True
    state = _layer_state(16)
    state.residency_mask = mask
    w, ids = _route(_router(state), logits)
    assert not mask[ids.long()].any()
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(64))
    # the same as the router restricted to the kept experts
    kept = (~mask).nonzero().squeeze(1)
    probs = logits[:, kept].softmax(dim=-1)
    ref_w, ref_local = probs.topk(4, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
    assert torch.equal(ids.long(), kept[ref_local])
    torch.testing.assert_close(w, ref_w)
    # the caller's logits are left alone
    assert torch.isfinite(logits).all()


def test_displaced_calls_are_counted_from_the_unmasked_logits(ref_softmax):
    torch.manual_seed(2)
    num_tokens, num_real, top_k = 24, 20, 4
    logits = torch.randn(num_tokens, 16)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[[1, 2, 9]] = True
    state = _layer_state(16, num_real=num_real)
    state.residency_mask = mask
    state.residency_stats = torch.zeros(3)
    state.residency_shadow_load = torch.zeros(16)
    _route(_router(state, top_k=top_k), logits)
    _route(_router(state, top_k=top_k), logits)  # counters add up
    vals, ids = logits[:num_real].topk(top_k, dim=-1)
    displaced = mask[ids]
    weights = vals.softmax(dim=-1)
    stats = state.residency_stats
    assert stats[DISPLACED_CALLS].item() == 2 * displaced.sum().item()
    assert stats[ROUTED_CALLS].item() == 2 * num_real * top_k
    torch.testing.assert_close(
        stats[DISPLACED_WEIGHT], 2 * (weights * displaced).sum(), rtol=1e-5, atol=1e-5
    )
    expected = torch.bincount(ids.reshape(-1), minlength=16).float() * 2
    assert torch.equal(state.residency_shadow_load, expected)  # padding excluded


def test_a_router_with_a_score_bias_refuses_the_mask():
    state = _layer_state(16)
    state.residency_mask = torch.zeros(16, dtype=torch.bool)
    router = FusedTopKBiasRouter(
        top_k=4,
        global_num_experts=16,
        eplb_state=state,
        e_score_correction_bias=torch.zeros(16),
        renormalize=True,
        scoring_func="sigmoid",
    )
    assert router.residency_mask_active
    with pytest.raises(NotImplementedError, match="residency mask"):
        router._apply_residency_mask(torch.randn(3, 16))


# ---------------------------------------------------------------- step hook


def _masked_model(num_layers: int, num_experts: int, top_k: int) -> FakeMoEModel:
    model = FakeMoEModel(num_layers, num_experts)
    for layer in model.moe_layers:
        layer.eplb_state = EplbLayerState()
        layer.router = SimpleNamespace(top_k=top_k)
    return model


def _state(monkeypatch, **kwargs):
    kwargs.setdefault("window_size", 40)
    kwargs.setdefault("step_interval", 10**9)
    return single_rank_eplb(monkeypatch, **kwargs)


def test_mask_is_off_by_default(monkeypatch):
    assert EPLBConfig().residency_mask == "off"
    state = _state(monkeypatch)
    model = _masked_model(2, 8, 2)
    add_model(state, model)
    assert next(iter(state.model_states.values())).residency is None
    assert all(layer.eplb_state.residency_mask is None for layer in model.moe_layers)
    for _ in range(25):
        state.step()


def test_coldest_mode_masks_from_the_unmasked_routing_of_the_window(monkeypatch):
    num_layers, num_experts, top_k, interval = 2, 8, 2, 5
    state = _state(
        monkeypatch,
        residency_mask="coldest",
        residency_mask_share=0.1,
        residency_mask_interval=interval,
        window_size=10,  # two intervals
    )
    model = _masked_model(num_layers, num_experts, top_k)
    add_model(state, model)
    rm = next(iter(state.model_states.values())).residency
    assert isinstance(rm, ResidencyMask)
    layer0 = model.moe_layers[0].eplb_state
    # the layer's rows alias the model's tensors (one write reaches the router)
    assert layer0.residency_mask.data_ptr() == rm.mask[0].data_ptr()
    mask_ptr = rm.mask.data_ptr()

    def route(load_per_step: np.ndarray, steps: int):
        for _ in range(steps):
            rm.shadow_load.add_(torch.from_numpy(load_per_step).float())
            state.step()

    first = np.array([[40, 30, 20, 5, 3, 1, 1, 0]] * num_layers, dtype=float)
    route(first, interval - 1)
    assert not rm.mask.any()  # not due yet
    route(first, 1)
    expected = coldest_share_mask(first * interval, 0.1, top_k)
    assert np.array_equal(rm.mask.numpy(), expected)
    assert rm.mask.data_ptr() == mask_ptr  # written in place
    assert rm.shadow_load.sum() == 0  # counting starts again

    # Expert 7 becomes the busiest. Its unmasked routing is what is counted,
    # so it is unmasked once the window holds it, though it was masked.
    assert rm.mask[0, 7]
    second = np.array([[1, 1, 1, 5, 3, 1, 1, 90]] * num_layers, dtype=float)
    route(second, interval)
    window = (first + second) * interval
    assert np.array_equal(rm.mask.numpy(), coldest_share_mask(window, 0.1, top_k))
    assert not rm.mask[0, 7]
    route(second, interval)  # the first interval leaves the window
    assert np.array_equal(
        rm.mask.numpy(), coldest_share_mask(second * 2 * interval, 0.1, top_k)
    )


def test_coldest_mode_logs_what_it_displaced(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(
        eplb_state_mod.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    state = _state(
        monkeypatch,
        residency_mask="coldest",
        residency_mask_share=0.1,
        residency_mask_interval=2,
    )
    model = _masked_model(1, 8, 2)
    add_model(state, model)
    rm = next(iter(state.model_states.values())).residency
    rm.stats[0, ROUTED_CALLS] = 200
    rm.stats[0, DISPLACED_CALLS] = 3
    rm.stats[0, DISPLACED_WEIGHT] = 0.5
    state.step()
    assert not any("went to masked experts" in line for line in lines)
    state.step()
    assert any("3 (1.500 %) went to masked experts" in line for line in lines), lines
    assert any("carrying 0.500 % of the routed weight" in line for line in lines)
    assert rm.stats.sum() == 0


def test_nonresident_mode_follows_the_layers_and_settled_commits(monkeypatch):
    num_layers, num_experts, top_k = 2, 8, 2
    state = _state(
        monkeypatch,
        residency_mask="nonresident",
        residency_mask_interval=1000,
        use_async=True,
    )
    model = _masked_model(num_layers, num_experts, top_k)
    add_model(state, model)
    model_state = next(iter(state.model_states.values()))
    held_out: dict[int, list[int]] = {0: [], 1: []}
    model_state.movers = [
        SimpleNamespace(
            nonresident_physical=lambda layer=layer: held_out[layer], scrub_every=0
        )
        for layer in range(num_layers)
    ]
    rm = model_state.residency
    state.step()  # first step: refreshed, every expert held
    assert rm.refreshes == 1 and not rm.mask.any()

    # A commit at this step makes slot 5 of layer 1 non-resident: the mask
    # follows at the same step hook, before the next forward.
    held_out[1] = [5]
    model_state.rebalanced = True
    model_state.pending_result = object()
    monkeypatch.setattr(state, "_all_ranks_result_ready", lambda s: True)

    def commit(model_state, ep_rank):
        model_state.pending_result = None

    monkeypatch.setattr(eplb_state_mod, "_move_to_workspace", commit)
    state.step()
    assert rm.refreshes == 2
    assert rm.mask[1].nonzero().squeeze(1).tolist() == [5]
    assert not rm.mask[0].any()

    model_state.rebalanced = False
    state.step()  # nothing settled, not due: no refresh
    assert rm.refreshes == 2


def test_expert_tiers_report_slots_held_by_no_tier():
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import ExpertTiers
    from vllm.model_executor.offloader.expert_tier_map import TierPlacement

    placement = TierPlacement(4, 2, 1, lambda event: True)
    tiers = SimpleNamespace(
        placement=placement, local_of={0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: -1}
    )
    report = ExpertTiers.nonresident_physical
    assert report(tiers) == []  # the invariant of the tiers as they are
    placement.tier[2] = 7  # a slot in neither tier
    assert report(tiers) == [3]
    assert report(SimpleNamespace(placement=None, local_of={})) == []


def test_config_validation_and_compile_factors():
    with pytest.raises(ValueError, match="residency_mask_stats"):
        EPLBConfig(residency_mask="coldest", residency_mask_stats=False)
    with pytest.raises(ValueError, match="coldest"):
        EPLBConfig(residency_mask="nonresident", residency_mask_share=0.1)
    with pytest.raises(ValueError):
        EPLBConfig(residency_mask="coldest", residency_mask_share=1.0)
    a = EPLBConfig(residency_mask="coldest", residency_mask_share=0.01)
    b = EPLBConfig(residency_mask="coldest", residency_mask_share=0.05)
    off = EPLBConfig()
    assert a.compile_factors() == b.compile_factors()
    assert a.compile_factors() != off.compile_factors()
