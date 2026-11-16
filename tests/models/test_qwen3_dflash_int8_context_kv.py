# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash context K/V projection with compressed-tensors W8A8 INT8 QKV weights.

The K/V-only projection must give the K/V columns of each layer's own QKV
projection, bit for bit: per-channel weight scales and dynamic per-token
activation scales make every output element independent of the other rows.

On CPU the XPU INT8 kernel class runs its own weight processing and row split
(Triton at or below `VLLM_XPU_INT8_ONEDNN_MIN_ROWS`, oneDNN above); the two
device GEMMs and the activation quantization are exact integer stand-ins."""

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.linear.scaled_mm import xpu as xpu_mm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as ct_int8,
)
from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS = 256, 64, 4, 2, 3
Q_SIZE, KV_SIZE = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM

_SCHEMA = (
    "int8_gemm_w8a8(Tensor A_, Tensor A_scale, Tensor? A_zp, Tensor B, "
    "Tensor B_scale, Tensor? B_zp, int group_size, Tensor? bias, "
    "ScalarType? out_dtype=None) -> Tensor"
)


def _quant_config(
    weight_strategy: str = "channel",
    dynamic: bool = True,
    ignore: list[str] | None = None,
) -> ct.CompressedTensorsConfig:
    """The drafter checkpoint's config: W8A8 INT8, per-channel weights, dynamic
    symmetric per-token activations."""
    config = ct.CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "dynamic": False,
                        "strategy": weight_strategy,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "dynamic": dynamic,
                        "strategy": "token" if dynamic else "tensor",
                    },
                }
            },
            "ignore": ignore or [],
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    return config


def _exact_gemm(qx, weight, x_scale, weight_scale, out_dtype, bias):
    """int32-exact dot products, then the scale epilogue, element by element."""
    acc = (qx.to(torch.int64) @ weight.to(torch.int64)).to(torch.float32)
    out = acc * x_scale.float().reshape(-1, 1) * weight_scale.float().reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@pytest.fixture
def xpu_int8_kernel(monkeypatch):
    """Select the XPU INT8 kernel class with CPU stand-ins for its device ops."""
    calls = {"triton": 0, "onednn": 0}
    monkeypatch.setattr(xpu_mm.current_platform, "is_xpu", lambda: True)

    def quant(x, scale, azp, symmetric):
        assert scale is None and azp is None and symmetric
        xf = x.float()
        x_scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        x_q = torch.round(xf / x_scale).clamp(-128, 127).to(torch.int8)
        return x_q, x_scale, None

    def triton(qx, weight, scale_a, scale_b, out_dtype, bias=None):
        calls["triton"] += 1
        return _exact_gemm(qx, weight, scale_a, scale_b, out_dtype, bias)

    monkeypatch.setattr(xpu_mm.ops, "scaled_int8_quant", quant)
    monkeypatch.setattr(xpu_mm, "triton_scaled_mm", triton)

    def make_kernel(**kwargs):
        config = xpu_mm.Int8ScaledMMLinearLayerConfig(
            is_channelwise=kwargs["is_channelwise"],
            is_static_input_scheme=kwargs["is_static_input_scheme"],
            input_symmetric=kwargs["input_symmetric"],
            is_fused_module=kwargs.get("is_fused_module", False),
        )
        return xpu_mm.XPUInt8ScaledMMLinearKernel(
            config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
        )

    monkeypatch.setattr(ct_int8, "init_int8_linear_kernel", make_kernel)
    with torch.library._scoped_library("_xpu_C", "FRAGMENT") as lib:
        lib.define(_SCHEMA)

        def onednn(A, A_scale, A_zp, B, B_scale, B_zp, group_size, bias, out_dtype):
            assert A_zp is None and B_zp is None and group_size == -1
            calls["onednn"] += 1
            return _exact_gemm(A, B, A_scale, B_scale, out_dtype, bias)

        lib.impl("int8_gemm_w8a8", onednn, "CompositeExplicitAutograd")
        yield calls


@pytest.fixture
def bf16_default(default_vllm_config, dist_init):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(old_dtype)


def _checkpoint_tensors(
    proj: QKVParallelLinear, has_bias: bool
) -> dict[str, torch.Tensor]:
    """One layer's q/k/v tensors as its checkpoint holds them: int8 weights
    with BF16 per-channel scales [rows, 1] (or one per-tensor scale), or BF16
    weights for a layer the quantization config ignores."""
    quantized = hasattr(proj, "weight_scale")
    per_tensor = quantized and proj.weight_scale.dim() == 1
    tensors = {}
    for name, rows in (("q", Q_SIZE), ("k", KV_SIZE), ("v", KV_SIZE)):
        if not quantized:
            tensors[f"{name}_proj.weight"] = torch.randn(rows, HIDDEN) / 16
        else:
            tensors[f"{name}_proj.weight"] = torch.randint(
                -127, 128, (rows, HIDDEN), dtype=torch.int8
            )
            scale_rows = 1 if per_tensor else rows
            scale = torch.rand(scale_rows, 1, dtype=torch.float32) / 64 + 1e-3
            tensors[f"{name}_proj.weight_scale"] = (
                scale.reshape(1) if per_tensor else scale
            ).to(torch.bfloat16)
        if has_bias:
            tensors[f"{name}_proj.bias"] = torch.randn(rows) / 8
    return tensors


def _build_model(quant_config, has_bias: bool) -> tuple[DFlashQwen3Model, list]:
    """The parts of DFlashQwen3Model the context K/V path reads, built and
    loaded the way __init__ and load_weights do."""
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = RMSNorm(HIDDEN)
    model._rms_norm_eps = 1e-6
    attentions, checkpoint = [], []
    for i in range(NUM_LAYERS):
        attn = nn.Module()
        attn.q_size, attn.kv_size = Q_SIZE, KV_SIZE
        attn.k_norm = RMSNorm(HEAD_DIM)
        attn.qkv_proj = QKVParallelLinear(
            HIDDEN,
            HEAD_DIM,
            NUM_HEADS,
            NUM_KV_HEADS,
            bias=has_bias,
            quant_config=quant_config,
            prefix=f"layers.{i}.self_attn.qkv_proj",
            disable_tp=True,
        )
        attentions.append(attn)
        checkpoint.append(_checkpoint_tensors(attn.qkv_proj, has_bias))
    model.layers = nn.ModuleList()
    for attn in attentions:
        layer = nn.Module()
        layer.self_attn = attn
        model.layers.append(layer)
    model._context_kv_proj = None
    model._context_kv_kind = qwen3_dflash._context_kv_proj_kind(attentions)
    if model._context_kv_kind is not None:
        model._context_kv_proj = model._make_context_kv_proj(attentions)

    # Each layer's QKV projection loads its shards with its own loaders; the
    # checkpoint stream is teed into the context projection on the way.
    stream = [
        (f"layers.{i}.self_attn.{name}", tensor)
        for i, tensors in enumerate(checkpoint)
        for name, tensor in tensors.items()
    ]
    if model._context_kv_proj is not None:
        stream = list(model._tee_context_kv(stream))
    for name, tensor in stream:
        layer_idx, _, rest = name.removeprefix("layers.").partition(".self_attn.")
        shard, _, param_name = rest.partition("_proj.")
        proj = attentions[int(layer_idx)].qkv_proj
        param = getattr(proj, param_name)
        param.weight_loader(param, tensor, shard)
    return model, attentions


def _process(model: DFlashQwen3Model, attentions: list, has_bias: bool) -> None:
    # Load-time order: every module's post-load step (the context projection's
    # included), then the model-level step that builds the context buffers.
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(module)
    model._build_context_kv_buffers(attentions, has_bias=has_bias)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("rows", [7, 80])  # Triton and oneDNN side of the split
@torch.inference_mode()
def test_int8_context_kv_equals_qkv_projection(
    bf16_default, xpu_int8_kernel, monkeypatch, has_bias, rows
):
    monkeypatch.setattr(qwen3_dflash.ops, "rms_norm", _rms_norm_reference)
    torch.manual_seed(0)
    model, attentions = _build_model(_quant_config(), has_bias)
    assert model._context_kv_kind == "int8"
    proj = model._context_kv_proj
    assert proj.output_size == NUM_LAYERS * 2 * KV_SIZE
    assert isinstance(proj.scheme, ct_int8.CompressedTensorsW8A8Int8)
    _process(model, attentions, has_bias)
    assert model._fused_kv_weight is None
    assert model._context_kv_proj is proj
    # Kept when the other draft weights are freed (prefill_context_kv_only).
    model._attn_layers, model.use_aux_hidden_state = [], False
    kept = model.context_kv_modules()
    assert proj in kept and not any(a.qkv_proj in kept for a in attentions)

    x = torch.randn(rows, HIDDEN)
    normed = _rms_norm_reference(torch.empty_like(x), x, model.hidden_norm.weight, 1e-6)
    expected = torch.cat(
        [a.qkv_proj(normed)[0][:, Q_SIZE:] for a in attentions], dim=-1
    )
    calls_before = dict(xpu_int8_kernel)
    k, v = model._project_context_kv(x, rows, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    actual = torch.stack([k, v]).permute(2, 1, 0, 3, 4).reshape(rows, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # One GEMM for every layer's K/V, on the side of the row split it belongs to.
    side = "triton" if rows <= xpu_mm.XPU_INT8_ONEDNN_MIN_ROWS else "onednn"
    assert xpu_int8_kernel[side] - calls_before[side] == 1


def _rms_norm_reference(output, hidden_states, weight, eps):
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return output.copy_((x * weight.float()).to(output.dtype))


@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("int8", {}),
        ("one layer ignored", {"ignore": ["layers.1.self_attn.qkv_proj"]}),
        ("per-tensor weights", {"weight_strategy": "tensor"}),
        ("static activations", {"dynamic": False}),
    ],
)
def test_int8_context_kv_eligibility(bf16_default, xpu_int8_kernel, case, kwargs):
    """Only an all-layer, per-channel, dynamic symmetric W8A8 INT8 attention
    gets the K/V-only projection; everything else keeps the layer path."""
    model, _ = _build_model(_quant_config(**kwargs), has_bias=False)
    assert (model._context_kv_kind == "int8") == (case == "int8")
    assert (model._context_kv_proj is not None) == (case == "int8")


def test_int8_context_kv_switch_off(bf16_default, xpu_int8_kernel, monkeypatch):
    monkeypatch.setenv("VLLM_XPU_DFLASH_CONTEXT_KV_INT8", "0")
    model, _ = _build_model(_quant_config(), has_bias=False)
    assert model._context_kv_kind is None and model._context_kv_proj is None


def test_unquantized_layers_keep_no_context_projection(bf16_default):
    model, _ = _build_model(None, has_bias=False)
    assert model._context_kv_kind is None and model._context_kv_proj is None
