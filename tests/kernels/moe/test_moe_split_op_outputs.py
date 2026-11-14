# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE ops split out of the piecewise graphs (the CPU experts' split): the
piece after the op is replayed with the input address it was captured with,
so the op's outputs must keep that address. On CPU: the op functions with a
stand-in layer, a captured piece emulated as "reads the storage it was
captured with", the registration from the real config steps."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

pytestmark = pytest.mark.skip_global_cleanup

HIDDEN = 64


class _Layer:
    """A MoE layer whose _forward_impl returns fresh tensors after a
    routing-dependent allocation (as the CPU tier's host rows and staging
    allocate per step)."""

    def __init__(self, shared: bool, split: bool = True):
        holder = getattr(mr, "SplitOpOutputs", None)
        self.split_op_outputs = holder() if split and holder is not None else None
        self.shared = shared
        self.keep: list[torch.Tensor] = []

    def _forward_impl(self, hidden_states, router_logits, shared_input, input_ids):
        rows = int(router_logits.abs().sum().item()) % 5 + 1
        self.keep.append(torch.empty(rows * 97, HIDDEN))
        fused = hidden_states * 3 + router_logits[:, :1]
        if self.shared:
            return hidden_states * 2, fused
        return fused


class _Piece:
    """A captured piece as a replay sees it: it reads the storage its input
    had at capture (a replay copies nothing in), and checks the address as
    the graph wrapper does in debug mode."""

    def __init__(self):
        self.captured: tuple[torch.Tensor, ...] | None = None

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.captured is None:
            self.captured = inputs
        return tuple(c.clone() for c in self.captured)

    def addresses_kept(self, *inputs: torch.Tensor) -> bool:
        assert self.captured is not None
        return [c.data_ptr() for c in self.captured] == [x.data_ptr() for x in inputs]


@pytest.fixture
def piecewise(monkeypatch):
    layers: dict[str, _Layer] = {}
    mode = {"piecewise": True}
    monkeypatch.setattr(mr, "get_layer_from_name", lambda name: layers[name])
    monkeypatch.setattr(
        mr, "in_piecewise_cudagraph", lambda: mode["piecewise"], raising=False
    )
    return layers, mode


def _op(layer: _Layer, x: torch.Tensor, logits: torch.Tensor):
    if layer.shared:
        return mr._moe_forward_shared(x, logits, None, None, "l0", 0)
    return (mr._moe_forward(x, logits, None, None, "l0", 0),)


def _expected(layer: _Layer, x: torch.Tensor, logits: torch.Tensor):
    fused = x * 3 + logits[:, :1]
    return (x * 2, fused) if layer.shared else (fused,)


@pytest.mark.parametrize("shared", [False, True])
def test_the_piece_after_a_split_moe_op_reads_this_steps_output(piecewise, shared):
    layers, _ = piecewise
    layer = layers["l0"] = _Layer(shared)
    g = torch.Generator().manual_seed(0)
    piece = _Piece()
    n = 8
    # capture: the inputs hold whatever the uncomputed graph memory holds
    x, logits = torch.randn(n, HIDDEN, generator=g), torch.randn(n, 4, generator=g)
    piece(*_op(layer, x, logits))
    for _ in range(4):  # replays, each step allocating differently
        x = torch.randn(n, HIDDEN, generator=g)
        logits = torch.randn(n, 4, generator=g) * 7
        outs = _op(layer, x, logits)
        # what the replayed piece computes on: this step's rows, bit for bit
        for got, want in zip(piece(*outs), _expected(layer, x, logits)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert piece.addresses_kept(*outs)


def test_smaller_graphs_keep_their_address_after_the_largest(piecewise):
    layers, _ = piecewise
    layer = layers["l0"] = _Layer(shared=True)
    sizes = [16, 8, 4, 1]  # captured largest first, then replayed in any order
    pieces = {n: _Piece() for n in sizes}
    for n in sizes:
        pieces[n](*_op(layer, torch.randn(n, HIDDEN), torch.randn(n, 4)))
    for n in [1, 16, 4, 8, 16, 1]:
        x, logits = torch.randn(n, HIDDEN), torch.randn(n, 4)
        outs = _op(layer, x, logits)
        assert pieces[n].addresses_kept(*outs)
        for got, want in zip(pieces[n](*outs), _expected(layer, x, logits)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    # a count larger than every buffer takes a new one; the others stay
    big = _op(layer, torch.randn(32, HIDDEN), torch.randn(32, 4))
    assert not pieces[16].addresses_kept(*big)
    outs = _op(layer, torch.randn(16, HIDDEN), torch.randn(16, 4))
    assert pieces[16].addresses_kept(*outs)


@pytest.mark.parametrize("shared", [False, True])
def test_outside_piecewise_or_unsplit_the_op_returns_its_own_tensors(piecewise, shared):
    layers, mode = piecewise
    x, logits = torch.randn(4, HIDDEN), torch.randn(4, 4)
    for split, piecewise_mode in [(True, False), (False, True), (False, False)]:
        layer = layers["l0"] = _Recording(shared, split=split)
        mode["piecewise"] = piecewise_mode
        outs = _op(layer, x, logits)
        made = layer.last if shared else (layer.last,)
        assert len(outs) == len(made)
        assert all(o is m for o, m in zip(outs, made))


class _Recording(_Layer):
    def _forward_impl(self, *args):
        self.last = super()._forward_impl(*args)
        return self.last


def _register(splitting_ops):
    cc = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=splitting_ops,
    )
    layer = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
    mr.register_layer_for_moe_forward_op(SimpleNamespace(compilation_config=cc), layer)
    return layer


def test_registration_follows_the_cpu_experts_split(monkeypatch):
    from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
    from vllm.config.offload import OffloadConfig, UVAOffloadConfig

    def resolved(pinned):
        monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
        uva = UVAOffloadConfig(
            cpu_offload_gb=8, cpu_offload_per_expert=True, cpu_offload_pinned_gb=pinned
        )
        cc = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_breakable_cudagraph=None,
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["a::attn"],
        )
        c = SimpleNamespace(
            offload_config=OffloadConfig(uva=uva),
            compilation_config=cc,
            _uses_breakable_cudagraph_by_default=lambda: True,
            _uses_breakable_cudagraph_for_batch_invariance=lambda: False,
        )
        VllmConfig._maybe_enable_breakable_cudagraph(c)
        VllmConfig._split_graphs_for_cpu_experts(c)
        return cc.splitting_ops

    # the CPU tier on: split at the MoE ops, the layers hold fixed outputs
    assert isinstance(_register(resolved(2)).split_op_outputs, mr.SplitOpOutputs)
    # without the CPU tier (unset, or not below the host budget): nothing
    for pinned in (None, 8):
        assert _register(resolved(pinned)).split_op_outputs is None
    assert _register(None).split_op_outputs is None
    assert _register(["vllm::moe_forward_shared"]).split_op_outputs is not None
