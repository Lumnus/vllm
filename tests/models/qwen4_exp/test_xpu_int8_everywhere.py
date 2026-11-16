# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 for the linears that had no INT8 path: the MoE router gate, the
shared-expert gate, the PLE key/value projection and the DFlash2 candidate
selector's projection. Each takes the compressed-tensors W8A8 method only when
the checkpoint names it as INT8, and stays BF16 otherwise (never rounded at
load). On CPU, with a reference for the INT8 linear kernel."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from tests.models.qwen4_exp.test_xpu_hc import _no_tp  # noqa: F401
from tests.models.qwen4_exp.test_xpu_hc_int8_up import (
    _W8A8,
    _post_load,
    _ref_w8a8,
    _RefKernel,
    _rtn8,
)
from vllm.model_executor.layers.fused_moe.router import int8_shadow
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    can_fuse_gate_weights,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_module,
)
from vllm.model_executor.layers.quantization.utils import checkpoint_int8
from vllm.model_executor.models.qwen3_dflash2 import build_selector_projection
from vllm.model_executor.models.qwen3_next import (
    build_router_gate,
    build_shared_expert_gate,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.xpu.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpModel,
    build_ple_kv_proj,
)

HIDDEN, EXPERTS, TOP_K = 64, 16, 4
PLE_EMBED, PLE_KEY, PLE_VALUE = 32, 48, 16
RANK = 8
TOKENS = 7
CONFIG = SimpleNamespace(
    hidden_size=HIDDEN, num_experts=EXPERTS, num_experts_per_tok=TOP_K
)

# The assembler's regexes (any non-MTP prefix), as a calibrated checkpoint
# carries them.
LM = r"^(?!(.*\.)?mtp\.)(.*\.)?"
ROUTER = rf"re:{LM}layers\.\d+\.mlp\.gate$"
SHARED_GATE = rf"re:{LM}layers\.\d+\.mlp\.shared_expert_gate$"
PLE_KV = rf"re:{LM}layers\.\d+\.ple\.(key_proj|value_proj|kv_proj)$"
SELECTOR = r"re:^(.*\.)?candidate_selector\.hidden_projection$"
ALL = [ROUTER, SHARED_GATE, PLE_KV, SELECTOR]
ATTN = rf"re:{LM}layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)$"


def _ct(targets, ignore=(), group=None):
    config = ct_module.CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {**(group or _W8A8), "targets": targets}},
            "ignore": list(ignore),
        }
    )
    config.packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    return config


class _Layer(nn.Module):
    def __init__(self, quant_config, index: int) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = build_router_gate(
            CONFIG, quant_config, f"layers.{index}.mlp.gate"
        )
        self.mlp.shared_expert_gate = build_shared_expert_gate(
            CONFIG, quant_config, f"layers.{index}.mlp.shared_expert_gate"
        )
        self.ple = nn.Module()
        self.ple.kv_proj = build_ple_kv_proj(
            PLE_EMBED, [PLE_KEY, PLE_VALUE], quant_config, f"layers.{index}.ple.kv_proj"
        )


class _Target(nn.Module):
    def __init__(self, quant_config) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(quant_config, 0)])
        self.candidate_selector = nn.Module()
        self.candidate_selector.hidden_projection = build_selector_projection(
            HIDDEN,
            RANK,
            torch.float32,
            quant_config,
            "candidate_selector.hidden_projection",
        )


def _weights(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "layers.0.mlp.gate": torch.randn(EXPERTS, HIDDEN) * HIDDEN**-0.5,
        "layers.0.mlp.shared_expert_gate": torch.randn(1, HIDDEN) * HIDDEN**-0.5,
        "layers.0.ple.key_proj": torch.randn(PLE_KEY, PLE_EMBED) * PLE_EMBED**-0.5,
        "layers.0.ple.value_proj": torch.randn(PLE_VALUE, PLE_EMBED) * PLE_EMBED**-0.5,
        "candidate_selector.hidden_projection": torch.randn(RANK, HIDDEN)
        * HIDDEN**-0.5,
    }


def _checkpoint(w: dict, int8: set[str]) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, weight in w.items():
        if name in int8:
            q, s = _rtn8(weight)
            tensors[f"{name}.weight"] = q
            tensors[f"{name}.weight_scale"] = s
        else:
            tensors[f"{name}.weight"] = weight.clone()
    return tensors


def _load(target: nn.Module, tensors: dict) -> set[str]:
    loaded = AutoWeightsLoader(target).load_weights(
        tensors.items(), mapper=Qwen4ExpModel.hf_to_vllm_mapper
    )
    _post_load(target)
    return loaded


def _layers(target: _Target) -> dict[str, nn.Module]:
    layer = target.layers[0]
    return {
        "router": layer.mlp.gate,
        "shared": layer.mlp.shared_expert_gate,
        "ple": layer.ple.kv_proj,
        "selector": target.candidate_selector.hidden_projection,
    }


def _outputs(target: _Target, x: torch.Tensor, emb: torch.Tensor) -> dict:
    m = _layers(target)
    return {
        "router": m["router"](x)[0],
        "shared": m["shared"](x)[0],
        "ple": m["ple"](emb),
        "selector": m["selector"](x),
    }


def _inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(TOKENS, HIDDEN), torch.randn(TOKENS, PLE_EMBED)


@pytest.fixture
def ref_kernel(monkeypatch):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_int8 as ct_w8a8,
    )

    _RefKernel.calls = []
    monkeypatch.setattr(
        ct_w8a8, "init_int8_linear_kernel", lambda **kwargs: _RefKernel(**kwargs)
    )
    return _RefKernel.calls


@pytest.fixture(autouse=True)
def _no_shadows():
    int8_shadow.clear()
    yield
    int8_shadow.clear()


def _unquantized(layer: nn.Module) -> bool:
    return isinstance(layer.quant_method, UnquantizedLinearMethod)


# --- BF16 stays BF16 -----------------------------------------------------


@pytest.mark.parametrize(
    "quant_config",
    [
        None,
        # A calibrated checkpoint that keeps the four parts BF16 (ignored).
        _ct([ATTN], ignore=ALL),
        # Not named at all.
        _ct([ATTN]),
        # A class-name catch-all names none of them: e.g. the drafter's
        # own config ("Linear", ignore "candidate_selector.hidden_projection"
        # under a name the vLLM prefix does not equal).
        _ct(["Linear"], ignore=["candidate_selector.hidden_projection"]),
    ],
    ids=["none", "ignored", "unnamed", "class-catch-all"],
)
def test_bf16_checkpoint_keeps_every_part_unquantized(ref_kernel, quant_config):
    target = _Target(quant_config)
    for name, layer in _layers(target).items():
        assert _unquantized(layer), name
        assert getattr(layer, checkpoint_int8.INT8_ATTR) is False
    assert target.layers[0].mlp.gate.is_unquantized
    assert target.layers[0].mlp.gate.routing_shadow is None

    w = _weights(1)
    loaded = _load(target, _checkpoint(w, set()))
    assert {n for n, _ in target.named_parameters()} <= loaded
    # The BF16 weights land as they are (the forward is the unchanged
    # unquantized linear; on a CPU host it needs the CPU repack, not run here).
    m = _layers(target)
    for key, name in (
        ("router", "layers.0.mlp.gate"),
        ("shared", "layers.0.mlp.shared_expert_gate"),
        ("selector", "candidate_selector.hidden_projection"),
    ):
        assert torch.equal(m[key].weight, w[name]), key
    kv = torch.cat((w["layers.0.ple.key_proj"], w["layers.0.ple.value_proj"]))
    assert torch.equal(m["ple"].weight, kv)
    assert not ref_kernel


def test_online_targets_never_round_these_parts():
    """An online spec (no checkpoint INT8) is not taken for these parts: the
    helper answers None, so the layer is built unquantized."""
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={
                "re:.*mlp\\.gate$": "int8_per_channel",
                "re:.*shared_expert_gate$": "int8_per_channel",
                "re:.*ple\\.kv_proj$": "int8_per_channel",
            }
        )
    )
    for prefix in (
        "layers.0.mlp.gate",
        "layers.0.mlp.shared_expert_gate",
        "layers.0.ple.kv_proj",
    ):
        assert checkpoint_int8.checkpoint_int8_quant_config(config, prefix, "x") is None
    gate = build_router_gate(CONFIG, config, "layers.0.mlp.gate")
    assert _unquantized(gate)


def test_int8_weight_only_is_refused():
    group = {**_W8A8, "input_activations": None, "format": "pack-quantized"}
    with pytest.raises(ValueError, match="not W8A8"):
        build_router_gate(CONFIG, _ct([ROUTER], group=group), "layers.0.mlp.gate")


# --- INT8 from the checkpoint ---------------------------------------------


def test_int8_parts_load_and_match_the_int8_math(ref_kernel):
    target = _Target(_ct([ATTN, *ALL]))
    for name, layer in _layers(target).items():
        assert type(layer.scheme).__name__ == "CompressedTensorsW8A8Int8", name
        assert layer.weight.dtype == torch.int8, name
        assert getattr(layer, checkpoint_int8.INT8_ATTR) is True
    gate = target.layers[0].mlp.gate
    assert isinstance(gate, GateLinear) and not gate.is_unquantized

    w = _weights(2)
    tensors = _checkpoint(w, set(w))
    loaded = _load(target, tensors)
    assert {n for n, _ in target.named_parameters()} <= loaded

    # The merged PLE projection holds both shards' rows and scales.
    ple = target.layers[0].ple.kv_proj
    q_kv = torch.cat(
        (
            tensors["layers.0.ple.key_proj.weight"],
            tensors["layers.0.ple.value_proj.weight"],
        )
    )
    s_kv = torch.cat(
        (
            tensors["layers.0.ple.key_proj.weight_scale"],
            tensors["layers.0.ple.value_proj.weight_scale"],
        )
    )
    assert torch.equal(ple.weight.t(), q_kv)
    torch.testing.assert_close(ple.weight_scale, s_kv.float(), atol=0, rtol=0)

    x, emb = _inputs()
    out = _outputs(target, x, emb)
    for key, name, inp in (
        ("router", "layers.0.mlp.gate", x),
        ("shared", "layers.0.mlp.shared_expert_gate", x),
        ("selector", "candidate_selector.hidden_projection", x),
    ):
        ref = _ref_w8a8(inp, tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"])
        torch.testing.assert_close(out[key], ref, atol=0, rtol=0)
    torch.testing.assert_close(out["ple"], _ref_w8a8(emb, q_kv, s_kv), atol=0, rtol=0)
    assert len(ref_kernel) == 4

    # The router's logits stay comparable to BF16: its top-k agrees on most
    # slots (activation + weight rounding only).
    bf16 = F.linear(x, w["layers.0.mlp.gate"])
    got = torch.topk(out["router"], TOP_K).indices
    want = torch.topk(bf16, TOP_K).indices
    agree = (got.unsqueeze(-1) == want.unsqueeze(-2)).any(-1).float().mean()
    assert agree > 0.8, agree


def test_one_part_alone(ref_kernel):
    """Each part is switched by its own target; the others stay BF16."""
    for only in ALL:
        target = _Target(_ct([only], ignore=[p for p in ALL if p != only]))
        int8 = {n for n, m in _layers(target).items() if not _unquantized(m)}
        assert len(int8) == 1, (only, int8)


def test_mixed_ple_shards_fail_loudly(ref_kernel):
    """The merged PLE layer needs key and value in one scheme. A config that
    names only one keeps the layer BF16 and the int8 tensor is refused at load
    (no silent int8-as-BF16 copy)."""
    key_only = rf"re:{LM}layers\.\d+\.ple\.key_proj$"
    value_ignored = rf"re:{LM}layers\.\d+\.ple\.value_proj$"
    target = _Target(_ct([key_only], ignore=[value_ignored]))
    assert _unquantized(target.layers[0].ple.kv_proj)
    w = _weights(3)
    with pytest.raises(Exception):  # noqa: B017 - the loader's own error
        _load(target, _checkpoint(w, {"layers.0.ple.key_proj"}))


def test_load_line_counts_the_parts(ref_kernel, monkeypatch):
    target = _Target(_ct([ROUTER, PLE_KV]))
    lines = []
    monkeypatch.setattr(
        checkpoint_int8.logger, "info", lambda msg, *a: lines.append(msg % a)
    )
    checkpoint_int8.log_checkpoint_int8_parts(target, "model")
    assert len(lines) == 1
    line = lines[0]
    assert "router gate 1 of 1 INT8 (CompressedTensorsW8A8Int8)" in line
    assert "PLE kv_proj 1 of 1 INT8 (CompressedTensorsW8A8Int8)" in line
    assert "shared-expert gate 0 of 1 INT8" in line


def test_gate_weight_fusion_only_for_unquantized_gates(ref_kernel):
    bf16 = _Target(None).layers[0].mlp
    assert can_fuse_gate_weights(bf16.gate, bf16.shared_expert_gate)
    int8 = _Target(_ct([ROUTER])).layers[0].mlp
    assert not can_fuse_gate_weights(int8.gate, int8.shared_expert_gate)
    assert not can_fuse_gate_weights(None, bf16.shared_expert_gate)
    assert can_fuse_gate_weights(nn.Linear(4, 2), nn.Linear(4, 1))


# --- routing agreement against a BF16 shadow ---------------------------------


def _slots_differ(got_logits, want_logits) -> int:
    got = torch.topk(got_logits.float(), TOP_K).indices
    want = torch.topk(want_logits.float(), TOP_K).indices
    same = (got.unsqueeze(-1) == want.unsqueeze(-2)).any(-1).sum()
    return int(got.numel() - same)


def _write_bf16_checkpoint(directory, weight: torch.Tensor) -> None:
    name = "model.language_model.layers.0.mlp.gate.weight"
    mtp = "mtp.layers.0.mlp.gate.weight"
    save_file(
        {name: weight.contiguous(), mtp: torch.zeros_like(weight)},
        os.path.join(directory, "a.safetensors"),
    )
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {name: "a.safetensors", mtp: "a.safetensors"}}, f)


@pytest.mark.parametrize("source", ["dequant", "checkpoint"])
def test_router_shadow_counts_differing_slots(
    ref_kernel, monkeypatch, tmp_path, source
):
    w = _weights(4)
    torch.manual_seed(5)
    # The original router the INT8 one was calibrated from: here a perturbed
    # copy, so that some slots move.
    original = w["layers.0.mlp.gate"] + 0.05 * torch.randn(EXPERTS, HIDDEN)
    if source == "checkpoint":
        _write_bf16_checkpoint(str(tmp_path), original)
        monkeypatch.setenv("VLLM_ROUTER_INT8_SHADOW", str(tmp_path))
    else:
        monkeypatch.setenv("VLLM_ROUTER_INT8_SHADOW", "dequant")
    target = _Target(_ct([ROUTER]))
    gate = target.layers[0].mlp.gate
    assert gate.routing_shadow is not None
    # The shared gate and BF16 gates never get one.
    assert not hasattr(target.layers[0].mlp.shared_expert_gate, "routing_shadow")
    tensors = _checkpoint(w, {"layers.0.mlp.gate"})
    _load(target, tensors)

    lines = []
    monkeypatch.setattr(
        int8_shadow.logger, "info", lambda msg, *a: lines.append(msg % a)
    )
    expected = 0
    rows = 0
    for seed in (6, 7):
        x, _ = _inputs(seed)
        logits, _ = gate(x)
        q, s = (
            tensors["layers.0.mlp.gate.weight"],
            tensors["layers.0.mlp.gate.weight_scale"],
        )
        if source == "checkpoint":
            ref = F.linear(x, original)
        else:
            ref = F.linear(x, (q.float() * s.float()))
        expected += _slots_differ(logits, ref)
        rows += x.shape[0]
    # The second call reported the first forward; report the rest.
    assert len(lines) == 1
    int8_shadow.report()
    assert len(lines) == 2
    first, second = lines
    for line in lines:
        assert "routed slots differ" in line and "in 1 layers" in line
    total = sum(int(line.split(" of ")[0].split(": ")[-1]) for line in lines)
    assert total == expected, (lines, expected)
    assert f"of {TOKENS * TOP_K} routed slots" in first
    if source == "checkpoint":
        assert expected > 0
    # Counts cleared after a report.
    int8_shadow.report()
    assert len(lines) == 2


def test_shadow_switch_rejects_a_bad_path(monkeypatch, tmp_path, ref_kernel):
    monkeypatch.setenv("VLLM_ROUTER_INT8_SHADOW", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="VLLM_ROUTER_INT8_SHADOW"):
        _Target(_ct([ROUTER]))


def test_shadow_dequant_orientation():
    layer = SimpleNamespace(
        prefix="p",
        output_size=3,
        input_size=5,
        weight=torch.arange(15, dtype=torch.int8).reshape(3, 5),
        weight_scale=torch.tensor([[1.0], [2.0], [0.5]]),
    )
    want = layer.weight.float() * layer.weight_scale
    torch.testing.assert_close(
        int8_shadow.dequantized_weight(layer, torch.float32), want
    )
    layer.weight = layer.weight.t()  # the INT8 kernels' [K, N] view
    torch.testing.assert_close(
        int8_shadow.dequantized_weight(layer, torch.float32), want
    )
    assert isinstance(ReplicatedLinear, type)
