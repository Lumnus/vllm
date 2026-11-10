# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB must move the bytes the XPU MoE kernels read.

`XpuFusedMoe` (vllm-xpu-kernels) rewrites the loaded expert weights into its
kernel layout when it is built, rebinding the parameters' storage: int4
weights become signed int4 in a new int8 buffer, wider weights are
transposed. EPLB keeps views of the expert tensors from model registration on.
If the kernels object is built after that (lazily, on the first forward), the
views point at the orphaned pre-relayout storage, and a rearrangement moves
bytes no kernel reads while the maps say the experts moved.

These tests run on CPU with a stand-in for `XpuFusedMoe` that rebinds storage
the same way.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.models.interfaces import MixtureOfExperts

E, N, K, G = 4, 8, 16, 8  # experts, intermediate, hidden, group size


class _FakeXpuFusedMoe:
    """Rebinds `.data` like `XpuFusedMoe.__init__` in vllm-xpu-kernels."""

    builds = 0

    def __init__(self, w13, w13_scales, w13_bias, w2, w2_scales, w2_bias, **kw):
        type(self).builds += 1
        if not hasattr(w13, "xpu_fused_moe"):
            if w13.dtype == torch.uint8:
                # u4 -> s4 into a new int8 buffer
                for w in (w13, w2):
                    w.data = (w.data ^ 0x88).view(torch.int8).clone()
            else:
                for w in (w13, w2):
                    w.data = w.data.transpose(-1, -2).contiguous()
            w13.xpu_fused_moe = True
        self.w13, self.w2 = w13, w2
        self.gemm1_wei_scales, self.gemm2_wei_scales = w13_scales, w2_scales

    def apply(self, **kwargs) -> None:
        pass


class _Layer(torch.nn.Module):
    """The parts of RoutedExperts that `get_expert_weights` reads."""

    get_expert_weights = RoutedExperts.get_expert_weights

    def __init__(self, int4: bool):
        super().__init__()
        self.local_num_experts = E
        self.expert_map = None
        if int4:
            # The AutoGPTQ layout after its post-load step on XPU: packed int4
            # as uint8 [E, N, K / 2], bf16 group scales [E, N, K / G].
            self.w13_qweight = _param(torch.randint(0, 256, (E, 2 * N, K // 2)))
            self.w2_qweight = _param(torch.randint(0, 256, (E, K, N // 2)))
            self.w13_scales = _param(torch.randn(E, 2 * N, K // G).bfloat16())
            self.w2_scales = _param(torch.randn(E, K, N // G).bfloat16())
            self.w13_weight = self.w13_qweight
            self.w2_weight = self.w2_qweight
            scales = (self.w13_scales, self.w2_scales)
        else:
            self.w13_weight = _param(torch.randn(E, 2 * N, K).bfloat16())
            self.w2_weight = _param(torch.randn(E, K, N).bfloat16())
            scales = (None, None)
        self.experts = _experts(*scales)
        self.quant_method = SimpleNamespace(
            moe_kernel=SimpleNamespace(fused_experts=self.experts)
        )

    def forward_once(self) -> None:
        """What the first forward does to the experts object."""
        hidden = torch.zeros(2, K)
        topk_ids = torch.zeros(2, 2, dtype=torch.int32)
        self.experts.apply(
            output=torch.empty(2, K),
            hidden_states=hidden,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=torch.ones(2, 2),
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=E,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.empty(0),
            workspace2=torch.empty(0),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def _param(t: torch.Tensor) -> torch.nn.Parameter:
    if t.dtype == torch.int64:
        t = t.to(torch.uint8)
    return torch.nn.Parameter(t.contiguous(), requires_grad=False)


def _experts(w1_scale, w2_scale) -> xpu_moe.XPUExpertsWNA16:
    # __init__ queries the device architecture; set the fields apply() reads.
    experts = object.__new__(xpu_moe.XPUExpertsWNA16)
    experts.moe_config = SimpleNamespace(
        experts_per_token=2,
        activation=MoEActivation.SILU,
        num_local_experts=E,
        ep_rank=0,
        ep_size=1,
    )
    experts.quant_config = SimpleNamespace(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        weight_quant_dtype="int4" if w1_scale is not None else None,
    )
    experts.gemm1_clamp_limit = None
    experts.fused_moe_impl = None
    return experts


class _LocalOnly:
    """EPLB communicator for one rank: every move is local."""

    needs_profile_buffer_reservation = False
    needs_recv_rows = False

    def set_transfer_context(self, *args) -> None:
        pass

    def add_send(self, *args, **kwargs) -> None:
        raise AssertionError("no remote sends with one rank")

    def add_recv(self, *args, **kwargs) -> None:
        raise AssertionError("no remote receives with one rank")

    def execute(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_kernels(monkeypatch):
    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", _FakeXpuFusedMoe, raising=False)
    _FakeXpuFusedMoe.builds = 0


def _register(layer: _Layer) -> list[torch.Tensor]:
    """`MixtureOfExperts.set_eplb_state`, as `load_model` runs it."""
    model = SimpleNamespace(
        moe_layers=[
            SimpleNamespace(
                get_expert_weights=layer.get_expert_weights,
                set_eplb_state=lambda **_: None,
            )
        ]
    )
    empty = torch.empty(0)
    MixtureOfExperts.set_eplb_state(model, empty, empty, empty)
    return model.expert_weights


def _kernel_tensors(layer: _Layer) -> list[torch.Tensor]:
    impl = layer.experts.fused_moe_impl
    tensors = [impl.w13, impl.w2, impl.gemm1_wei_scales, impl.gemm2_wei_scales]
    return [t for t in tensors if t is not None]


def _swap_experts_0_1(expert_weights: list[list[torch.Tensor]]) -> None:
    old = torch.arange(E).unsqueeze(0)
    new = old.clone()
    new[0, :2] = torch.tensor([1, 0])
    buffer = [torch.empty_like(w) for w in expert_weights[0]]
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    rearrange_expert_weights_inplace(
        old, new, expert_weights, buffer, group, _LocalOnly()
    )


@pytest.mark.parametrize("int4", [True, False], ids=["int4", "bf16"])
def test_eplb_views_are_the_kernel_tensors(int4):
    layer = _Layer(int4)
    registered = _register(layer)[0]
    layer.forward_once()
    assert _FakeXpuFusedMoe.builds == 1
    ptrs = {w.data_ptr() for w in registered}
    for tensor in _kernel_tensors(layer):
        assert tensor.data_ptr() in ptrs


@pytest.mark.parametrize("int4", [True, False], ids=["int4", "bf16"])
def test_rearrangement_moves_the_bytes_the_kernels_read(int4):
    layer = _Layer(int4)
    expert_weights = _register(layer)
    layer.forward_once()
    before = [t.clone() for t in _kernel_tensors(layer)]
    _swap_experts_0_1(expert_weights)
    for old, new in zip(before, _kernel_tensors(layer)):
        assert torch.equal(new[0], old[1])
        assert torch.equal(new[1], old[0])
        assert torch.equal(new[2:], old[2:])


@pytest.mark.parametrize("int4", [True, False], ids=["int4", "bf16"])
def test_control_lazy_build_leaves_eplb_on_stale_storage(int4, monkeypatch):
    """Without the build at weight processing, the kernels object is built at
    the first forward and the rearrangement misses the kernels' bytes."""
    monkeypatch.setattr(
        xpu_moe.XPUExperts,
        "finalize_weight_storage",
        lambda self, layer: None,
        raising=False,
    )
    layer = _Layer(int4)
    expert_weights = _register(layer)
    layer.forward_once()
    w13 = layer.experts.fused_moe_impl.w13
    before = w13.clone()
    assert not torch.equal(before[0], before[1])
    _swap_experts_0_1(expert_weights)
    # The maps would now send expert 0's tokens to row 1, which still holds
    # expert 1's weights.
    assert torch.equal(w13, before)


def test_finalize_is_idempotent_and_apply_reuses_it():
    layer = _Layer(int4=True)
    layer.experts.finalize_weight_storage(layer)
    impl = layer.experts.fused_moe_impl
    layer.experts.finalize_weight_storage(layer)
    layer.forward_once()
    assert layer.experts.fused_moe_impl is impl
    assert _FakeXpuFusedMoe.builds == 1
