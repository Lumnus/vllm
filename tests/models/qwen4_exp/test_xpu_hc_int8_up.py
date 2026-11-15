# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 hyper-connection projections beyond the merged down GEMM, on CPU: the up
projection, the final mixer's down projection, and a merged down projection
without the fused kernel run through their layers' INT8 method; checkpoint
tensors in the compressed-tensors W8A8 naming load into them; an online
``targets`` entry rounds a BF16 up projection at load. Torch stand-ins for the
HC kernels and a reference for the INT8 linear kernel."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.models.qwen4_exp.test_xpu_hc import (  # noqa: F401
    EPS,
    HC,
    HC_DIM,
    N_EMBD,
    TOKENS,
    _no_tp,
    _torch_ops,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_module,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as ct_w8a8,
)
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online import int8 as online_int8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.model_loader.weight_utils import finalize_model_weights
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.xpu import hyperconnection as hc_module
from vllm.models.qwen4_exp.xpu.hyperconnection import (
    HyperConnection,
    _WeightContainer,
    log_hc_paths,
)
from vllm.models.qwen4_exp.xpu.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
)

# A low rank whose merged down GEMM has padding rows (10 + 4 -> 16).
LOWRANK = 10
PAD = -(LOWRANK + HC) % 16
CONFIG = SimpleNamespace(
    hc_count=HC, hidden_size=N_EMBD, hc_lowrank=LOWRANK, rms_norm_eps=EPS
)


def _weights(dtype: torch.dtype, seed: int = 0) -> SimpleNamespace:
    torch.manual_seed(seed)
    scale = HC_DIM**-0.5
    return SimpleNamespace(
        norm=(0.1 * torch.randn(HC_DIM)).to(dtype),
        down=(scale * torch.randn(LOWRANK, HC_DIM)).to(dtype),
        up=(scale * torch.randn(HC_DIM, LOWRANK)).to(dtype),
        inject=(scale * torch.randn(HC, HC_DIM)).to(dtype),
    )


def _merged_down(w: SimpleNamespace) -> torch.Tensor:
    return torch.cat((w.down, w.inject, torch.zeros(PAD, HC_DIM, dtype=w.down.dtype)))


_W8A8 = {
    "weights": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
    "format": "int-quantized",
}

_UP = "re:.*_hyper_connection\\.input_mix_weight_up$"
_MIXER_UP = "re:.*hyper_connection_mixer\\.input_mix_weight_up$"
_MIXER_DOWN = "re:.*hyper_connection_mixer\\.input_mix_weight_down$"
# The checkpoint's names: the fused layer resolves only through the packed mapping.
_DOWN_SHARDS = "re:.*_hyper_connection\\.(input_mix_weight_down|block_inject_weight)$"


def _ct_config(targets: list[str], packed: dict | None = None):
    config = ct_module.CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {**_W8A8, "targets": targets}},
            "ignore": [],
        }
    )
    config.packed_modules_mapping = (
        Qwen4ExpForCausalLM.packed_modules_mapping if packed is None else packed
    )
    return config


def _act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 with an fp32 scale (the native XPU rule)."""
    xf = x.float()
    xs = (xf.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-10)
    return torch.round(xf / xs).clamp(-128, 127), xs


def _ref_w8a8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``x @ (q * s).T`` with int8 activations: exact integer products, then
    the token and row scales (``q`` ``[N, K]`` int8, ``s`` ``[N, 1]``)."""
    xq, xs = _act_quant(x)
    return ((xq.double() @ q.double().t()) * xs.double() * s.double().t()).to(x.dtype)


class _RefKernel:
    """The XPU/Triton INT8 linear kernels' contract: the ``[N, K]`` weight kept
    as a ``[K, N]`` view, per-row fp32 scales, per-token int8 activations."""

    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process_weights_after_loading(self, layer):
        replace_parameter(layer, "weight", layer.weight.data.t())
        layer.input_scale = None
        layer.azp_adj = None

    def apply_weights(self, layer, x, bias=None):
        _RefKernel.calls.append((layer.prefix, tuple(x.shape)))
        return _ref_w8a8(x, layer.weight.t(), layer.weight_scale)


@pytest.fixture
def ref_kernel(monkeypatch):
    _RefKernel.calls = []
    make = lambda **kwargs: _RefKernel(**kwargs)  # noqa: E731
    monkeypatch.setattr(ct_w8a8, "init_int8_linear_kernel", make)
    monkeypatch.setattr(online_int8, "init_int8_linear_kernel", make)
    return _RefKernel.calls


def _rtn8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The checkpoint form: int8 rows, F16 scales ``amax/127``."""
    s = (w.float().abs().amax(-1, keepdim=True) / 127).to(torch.float16)
    q = torch.round(w.float() / s.float()).clamp(-127, 127).to(torch.int8)
    return q, s


class _Target(nn.Module):
    def __init__(self, quant_config) -> None:
        super().__init__()
        self.attn_hyper_connection = HyperConnection(
            CONFIG, prefix="attn_hyper_connection", quant_config=quant_config
        )
        self.hyper_connection_mixer = HyperConnection(
            CONFIG,
            has_inject=False,
            prefix="hyper_connection_mixer",
            quant_config=quant_config,
        )


def _checkpoint(w, int8_up: bool, int8_mixer_down: bool, int8_down: bool):
    """Tensors under the checkpoint's names, int8 where the variant has them."""
    tensors = {}

    def put(name, weight, as_int8):
        if as_int8:
            q, s = _rtn8(weight)
            tensors[f"{name}.weight"] = q
            tensors[f"{name}.weight_scale"] = s
        else:
            tensors[f"{name}.weight"] = weight

    for hc in ("attn_hyper_connection", "hyper_connection_mixer"):
        tensors[f"{hc}.hc_norm.weight"] = w.norm
        put(f"{hc}.input_mix_weight_up", w.up, int8_up)
    put("attn_hyper_connection.input_mix_weight_down", w.down, int8_down)
    put("attn_hyper_connection.block_inject_weight", w.inject, int8_down)
    put("hyper_connection_mixer.input_mix_weight_down", w.down, int8_mixer_down)
    return tensors


def _load(target: nn.Module, tensors: dict) -> set[str]:
    loaded = AutoWeightsLoader(target).load_weights(
        tensors.items(), mapper=Qwen4ExpModel.hf_to_vllm_mapper
    )
    _post_load(target)
    return loaded


def _post_load(target: nn.Module) -> None:
    """The loader's post-load pass: quant methods first, then module hooks. The
    BF16 layers are left as loaded: on a CPU host their method would repack the
    weight for the CPU GEMM, which the XPU model never does."""
    for module in target.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and not isinstance(method, UnquantizedLinearMethod):
            method.process_weights_after_loading(module)
    finalize_model_weights(target)


def _reference_mix(x, norm, down, up_fn, has_inject: bool):
    """The HC mix with the stand-in kernels' math and a given up GEMM."""
    grouped = x.reshape(x.shape[0], HC, N_EMBD).float()
    var = grouped.square().mean(-1, keepdim=True)
    xn = ((grouped * torch.rsqrt(var + EPS)).flatten(-2)) * (1 + norm.float())
    xn = xn.to(x.dtype)
    d = down(xn)
    lora = F.silu(d[:, :LOWRANK].float() / HC).to(x.dtype)
    gate = up_fn(lora)
    mixed = (torch.sigmoid(gate.float()) * xn.float()).unflatten(-1, (HC, -1))
    inject = d[:, LOWRANK : LOWRANK + HC] if has_inject else d[:, LOWRANK:LOWRANK]
    return mixed.mean(-2).to(x.dtype), inject


def test_bf16_keeps_the_raw_holders():
    for quant_config in (None, _ct_config(["re:.*self_attn\\.o_proj$"])):
        target = _Target(quant_config)
        for hc in (target.attn_hyper_connection, target.hyper_connection_mixer):
            assert isinstance(hc.input_mix_weight_up, _WeightContainer)
            assert hc.up_path == "bf16" and not hc._split
        mixer = target.hyper_connection_mixer
        assert isinstance(mixer.input_mix_weight_down, _WeightContainer)
        merged = target.attn_hyper_connection.input_mix_weight_down_block_inject
        assert isinstance(merged.quant_method, UnquantizedLinearMethod)


def test_packed_mapping_has_the_hc_merge():
    entry = ["input_mix_weight_down", "block_inject_weight"]
    assert (
        Qwen4ExpForCausalLM.packed_modules_mapping["input_mix_weight_down_block_inject"]
        == entry
    )
    assert (
        Qwen4ExpForConditionalGeneration.packed_modules_mapping[
            "input_mix_weight_down_block_inject"
        ]
        == entry
    )
    from vllm.models.qwen4_exp.xpu.mtp import Qwen4ExpMTP

    assert (
        Qwen4ExpMTP.packed_modules_mapping is Qwen4ExpForCausalLM.packed_modules_mapping
    )


def test_shard_names_resolve_the_merged_down_only_with_the_mapping(ref_kernel):
    without = HyperConnection(
        CONFIG,
        prefix="attn_hyper_connection",
        quant_config=_ct_config([_DOWN_SHARDS], {}),
    )
    assert isinstance(
        without.input_mix_weight_down_block_inject.quant_method, UnquantizedLinearMethod
    )
    with_mapping = HyperConnection(
        CONFIG, prefix="attn_hyper_connection", quant_config=_ct_config([_DOWN_SHARDS])
    )
    layer = with_mapping.input_mix_weight_down_block_inject
    assert type(layer.scheme).__name__ == "CompressedTensorsW8A8Int8"
    assert layer.weight.dtype == torch.int8


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_int8_up_and_mixer_load_and_run(monkeypatch, ref_kernel, dtype):
    _torch_ops(monkeypatch)
    torch.set_default_dtype(dtype)
    try:
        target = _Target(_ct_config([_UP, _MIXER_UP, _MIXER_DOWN]))
    finally:
        torch.set_default_dtype(torch.float32)
    hc, mixer = target.attn_hyper_connection, target.hyper_connection_mixer
    for module in (hc, mixer):
        assert isinstance(module.input_mix_weight_up, ReplicatedLinear)
    assert isinstance(mixer.input_mix_weight_down, ReplicatedLinear)

    w = _weights(dtype, seed=3)
    tensors = _checkpoint(w, int8_up=True, int8_mixer_down=True, int8_down=False)
    loaded = _load(target, tensors)
    assert {name for name, _ in target.named_parameters()} <= loaded
    assert hc.up_path == "layer" and hc.down_path == "bf16" and hc._split
    assert mixer.up_path == "layer" and mixer.down_path == "layer"

    torch.manual_seed(4)
    x = torch.randn(TOKENS, HC, N_EMBD).to(dtype)
    q_up, s_up = (
        tensors["attn_hyper_connection.input_mix_weight_up.weight"],
        tensors["attn_hyper_connection.input_mix_weight_up.weight_scale"],
    )
    mixed, inject = hc.mix(x)
    assert (
        "attn_hyper_connection.input_mix_weight_up",
        (TOKENS, LOWRANK),
    ) in ref_kernel

    # The explicit INT8 math, from the checkpoint tensors.
    ref_mixed, ref_inject = _reference_mix(
        x,
        w.norm,
        lambda xn: F.linear(xn, _merged_down(w)),
        lambda lora: _ref_w8a8(lora, q_up, s_up),
        True,
    )
    torch.testing.assert_close(mixed, ref_mixed, atol=0, rtol=0)
    torch.testing.assert_close(inject, ref_inject, atol=0, rtol=0)

    # Against BF16/fp32 math on the dequantized weight: the activation rounding.
    deq_up = (q_up.float() * s_up.float()).to(dtype)
    deq_mixed, _ = _reference_mix(
        x,
        w.norm,
        lambda xn: F.linear(xn, _merged_down(w)),
        lambda lora: F.linear(lora, deq_up),
        True,
    )
    rel = (mixed.float() - deq_mixed.float()).norm() / deq_mixed.float().norm()
    assert rel < 0.02, rel

    # combine_and_mix takes the same path; the mixer's down runs through its layer.
    residual = torch.randn(TOKENS, HC, N_EMBD).to(dtype)
    block = torch.randn(TOKENS, N_EMBD).to(dtype)
    wide, final, empty = mixer.combine_and_mix(residual, block, inject)
    assert empty.shape == (TOKENS, 0) and final.shape == (TOKENS, N_EMBD)
    q_md = tensors["hyper_connection_mixer.input_mix_weight_down.weight"]
    s_md = tensors["hyper_connection_mixer.input_mix_weight_down.weight_scale"]
    q_mu = tensors["hyper_connection_mixer.input_mix_weight_up.weight"]
    s_mu = tensors["hyper_connection_mixer.input_mix_weight_up.weight_scale"]
    ref_final, _ = _reference_mix(
        wide.reshape(TOKENS, HC_DIM),
        w.norm,
        lambda xn: _ref_w8a8(xn, q_md, s_md),
        lambda lora: _ref_w8a8(lora, q_mu, s_mu),
        False,
    )
    torch.testing.assert_close(final, ref_final, atol=0, rtol=0)


def test_int8_merged_down_without_the_fused_kernel_runs_through_its_layer(
    monkeypatch, ref_kernel
):
    if hasattr(torch.ops._xpu_C, "int8_dense_w8a8_fused_gemm"):
        pytest.skip("a build with the fused op is installed")
    _torch_ops(monkeypatch)
    target = _Target(_ct_config([_UP, _MIXER_UP, _DOWN_SHARDS]))
    hc = target.attn_hyper_connection
    w = _weights(torch.float32, seed=5)
    tensors = _checkpoint(w, int8_up=True, int8_mixer_down=False, int8_down=True)
    loaded = _load(target, tensors)
    assert {name for name, _ in target.named_parameters()} <= loaded
    assert hc.down_path == "layer" and hc.up_path == "layer"

    merged = hc.input_mix_weight_down_block_inject
    # The per-row scales of both shards landed (the stacked map takes
    # `weight_scale` with `weight`), and the padding rows hold zeros.
    valid = LOWRANK + HC
    rows = merged.weight.t()
    q_down = tensors["attn_hyper_connection.input_mix_weight_down.weight"]
    q_inj = tensors["attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(rows[:LOWRANK], q_down) and torch.equal(
        rows[LOWRANK:valid], q_inj
    )
    s_down = tensors["attn_hyper_connection.input_mix_weight_down.weight_scale"]
    s_inj = tensors["attn_hyper_connection.block_inject_weight.weight_scale"]
    torch.testing.assert_close(
        merged.weight_scale[:valid], torch.cat((s_down, s_inj)).float(), atol=0, rtol=0
    )
    assert rows.shape[0] == valid + PAD > valid
    assert not rows[valid:].any() and (merged.weight_scale[valid:] == 1).all()

    torch.manual_seed(6)
    x = torch.randn(TOKENS, HC, N_EMBD)
    mixed, inject = hc.mix(x)
    q_full = torch.cat((q_down, q_inj, torch.zeros_like(rows[valid:])))
    s_full = torch.cat((s_down.float(), s_inj.float(), torch.ones(PAD, 1)))
    q_up = tensors["attn_hyper_connection.input_mix_weight_up.weight"]
    s_up = tensors["attn_hyper_connection.input_mix_weight_up.weight_scale"]
    ref_mixed, ref_inject = _reference_mix(
        x,
        w.norm,
        lambda xn: _ref_w8a8(xn, q_full, s_full),
        lambda lora: _ref_w8a8(lora, q_up, s_up),
        True,
    )
    torch.testing.assert_close(mixed, ref_mixed, atol=0, rtol=0)
    torch.testing.assert_close(inject, ref_inject, atol=0, rtol=0)
    with pytest.raises(NotImplementedError, match="runs through its layer"):
        hc._down_operands()


def test_int8_merged_down_with_the_fused_kernel_keeps_it(monkeypatch, ref_kernel):
    """The checkpoint INT8 down (CT W8A8) takes the fused GEMM when it exists,
    with the INT8 up through its layer."""
    from tests.models.qwen4_exp.test_xpu_hc_int8 import _SCHEMA
    from tests.models.qwen4_exp.test_xpu_hc_int8 import _ref_w8a8 as fused_ref

    if hasattr(torch.ops._xpu_C, "int8_dense_w8a8_fused_gemm"):
        pytest.skip("a build with the fused op is installed")
    _torch_ops(monkeypatch)
    calls = []
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def impl(x, w, w_scale, bias):
            calls.append(tuple(w.shape))
            return fused_ref(x, w, w_scale, bias)

        lib.impl("int8_dense_w8a8_fused_gemm", impl, "CPU")
        target = _Target(_ct_config([_UP, _MIXER_UP, _DOWN_SHARDS]))
        _load(
            target,
            _checkpoint(
                _weights(torch.float32, 7),
                int8_up=True,
                int8_mixer_down=False,
                int8_down=True,
            ),
        )
        hc = target.attn_hyper_connection
        assert hc.down_path == "fused_int8"
        mixed, inject = hc.mix(torch.randn(TOKENS, HC, N_EMBD))
    assert calls == [(16, HC_DIM)]
    assert mixed.shape == (TOKENS, N_EMBD) and inject.shape == (TOKENS, HC)


def test_online_target_rounds_bf16_up_at_load(
    monkeypatch, ref_kernel, default_vllm_config
):
    """The serve-time route on a BF16 checkpoint: a `targets` entry naming the
    up projection (and the final mixer's down) rounds it at load, like the
    trunk's dense INT8 lever did."""
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    _torch_ops(monkeypatch)
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.float32)
    monkeypatch.setattr(online_fp8, "initialize_online_processing", lambda layer: None)
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={
                "re:.*input_mix_weight_up$": "int8_per_channel",
                _MIXER_DOWN: "int8_per_channel",
            }
        )
    )
    config.packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    target = _Target(config)
    hc, mixer = target.attn_hyper_connection, target.hyper_connection_mixer
    assert isinstance(
        hc.input_mix_weight_up.quant_method,
        online_int8.Int8PerChannelOnlineLinearMethod,
    )
    assert hc.input_mix_weight_up.weight.is_meta
    # The merged down is not targeted: BF16, today's path.
    assert isinstance(
        hc.input_mix_weight_down_block_inject.quant_method, UnquantizedLinearMethod
    )
    assert sorted(config.quantized_layers) == [
        "attn_hyper_connection.input_mix_weight_up",
        "hyper_connection_mixer.input_mix_weight_down",
        "hyper_connection_mixer.input_mix_weight_up",
    ]

    w = _weights(torch.float32, seed=8)
    # What the online loader leaves before processing: the BF16 weight.
    for linear, value in (
        (hc.input_mix_weight_up, w.up),
        (mixer.input_mix_weight_up, w.up),
        (mixer.input_mix_weight_down, w.down),
    ):
        linear.weight = nn.Parameter(value.clone(), requires_grad=False)
        linear.tp_size = 1
    hc.hc_norm.weight.data.copy_(w.norm)
    mixer.hc_norm.weight.data.copy_(w.norm)
    hc.input_mix_weight_down_block_inject.weight.data.copy_(_merged_down(w))
    _post_load(target)
    assert hc.input_mix_weight_up.weight.dtype == torch.int8
    assert hc.up_path == "layer" and mixer.down_path == "layer"

    x = torch.randn(TOKENS, HC, N_EMBD)
    mixed, _ = hc.mix(x)
    bf16_mixed, _ = _reference_mix(
        x,
        w.norm,
        lambda xn: F.linear(xn, _merged_down(w)),
        lambda lora: F.linear(lora, w.up),
        True,
    )
    rel = (mixed - bf16_mixed).norm() / bf16_mixed.norm()
    assert 0 < rel < 0.02, rel


def test_split_fakes_match_the_real_shapes():
    with FakeTensorMode() as mode:
        xn = mode.from_tensor(torch.empty(TOKENS, HC_DIM, dtype=torch.bfloat16))
        down_w = mode.from_tensor(torch.empty(16, HC_DIM, dtype=torch.int8))
        lora, inject = hc_module._hc_down_lora_fake(
            xn, down_w, HC, LOWRANK, False, mode.from_tensor(torch.ones(16, 1))
        )
        assert lora.shape == (TOKENS, LOWRANK) and inject.shape == (TOKENS, HC)
        assert inject.stride() == (16, 1)
        down = mode.from_tensor(torch.empty(TOKENS, LOWRANK, dtype=torch.bfloat16))
        assert hc_module._hc_lowrank_silu_fake(down, HC, LOWRANK).shape == (
            TOKENS,
            LOWRANK,
        )
        gate = mode.from_tensor(torch.empty(TOKENS, HC_DIM, dtype=torch.bfloat16))
        assert hc_module._hc_gate_mix_fake(xn, gate, HC).shape == (TOKENS, N_EMBD)
        up = mode.from_tensor(torch.empty(HC_DIM, LOWRANK, dtype=torch.bfloat16))
        assert hc_module._hc_up_mix_fake(lora, up, xn, HC).shape == (TOKENS, N_EMBD)
        norm = mode.from_tensor(torch.empty(HC_DIM))
        assert hc_module._hc_norm_fake(xn, norm, 1e-6, HC).shape == xn.shape
        block = mode.from_tensor(torch.empty(TOKENS, N_EMBD, dtype=torch.bfloat16))
        inj = mode.from_tensor(torch.empty(TOKENS, HC, dtype=torch.bfloat16))
        combined, normed = hc_module._hc_combine_then_norm_fake(
            xn, block, inj, norm, 1e-6, HC
        )
        assert combined.shape == normed.shape == xn.shape


def test_split_mix_compiles_without_graph_breaks(monkeypatch, ref_kernel):
    """Under torch.compile the path is fixed: the HC kernels are opaque ops and
    the up GEMM is the layer's own call, with no per-call Python choice."""
    _torch_ops(monkeypatch)
    target = _Target(_ct_config([_UP, _MIXER_UP, _MIXER_DOWN]))
    _load(
        target,
        _checkpoint(
            _weights(torch.float32, 9),
            int8_up=True,
            int8_mixer_down=True,
            int8_down=False,
        ),
    )
    hc = target.attn_hyper_connection
    x = torch.randn(TOKENS, HC, N_EMBD)
    eager = hc.mix(x)
    torch._dynamo.reset()
    compiled = torch.compile(hc.mix, backend="eager", fullgraph=True)
    out = compiled(x)
    torch.testing.assert_close(out[0], eager[0], atol=0, rtol=0)
    torch.testing.assert_close(out[1], eager[1], atol=0, rtol=0)


def test_one_load_line_counts_the_paths(monkeypatch, ref_kernel):
    target = _Target(_ct_config([_UP, _MIXER_UP, _MIXER_DOWN]))
    _load(
        target,
        _checkpoint(
            _weights(torch.float32, 10),
            int8_up=True,
            int8_mixer_down=True,
            int8_down=False,
        ),
    )
    lines = []
    monkeypatch.setattr(
        hc_module.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    log_hc_paths(target, "model")
    assert len(lines) == 1
    line = lines[0]
    assert "(model): 2 modules" in line and "INT8 dense GEMM): 2" in line
    assert "quantized layer: 1 (final mixer among them: 1)" in line
    assert "CompressedTensorsLinearMethod" in line
    lines.clear()
    log_hc_paths(_Target(None), "model")
    assert lines == []


def test_mtp_mixer_takes_the_quant_config(monkeypatch):
    from vllm.models.qwen4_exp.xpu import mtp as mtp_module

    built = []

    def record(*args, **kwargs):
        built.append(kwargs)
        return nn.Identity()

    for name in (
        "VocabParallelEmbedding",
        "_WideRMSNorm",
        "HyperConnectionRMSNorm",
        "Qwen4ExpDecoderLayer",
        "ReplicatedLinear",
    ):
        monkeypatch.setattr(mtp_module, name, lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mtp_module, "HyperConnection", record)
    monkeypatch.setattr(mtp_module, "_uses_qsa", lambda config: False)
    quant_config = object()
    config = SimpleNamespace(
        vocab_size=16,
        hc_count=4,
        hidden_size=8,
        rms_norm_eps=1e-6,
        mtp_num_hidden_layers=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    mtp_module.Qwen4ExpMultiTokenPredictor(vllm_config=vllm_config, prefix="mtp")
    assert built == [
        dict(
            has_inject=False,
            prefix="mtp.hyper_connection_mixer",
            quant_config=quant_config,
        )
    ]
    assert LinearBase is not None and MergedColumnParallelLinear is not None
