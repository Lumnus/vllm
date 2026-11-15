# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A small GPTQ / AutoRound int4 MoE checkpoint in safetensors, in the name
scheme and dtypes of the real thing: int32 ``qweight`` ``[in/8, out]``, int32
``qzeros`` ``[groups, out/8]``, float16 ``scales`` ``[groups, out]`` per routed
expert and projection, in several shard files with an index, next to tensors
that are not store material (a drafter's bf16 experts, a shared expert,
attention) and a file the index does not name."""

import json
import pathlib
from dataclasses import dataclass, field

import torch
from safetensors.torch import save_file

SYM_QZEROS = 0x77777777


@dataclass
class Checkpoint:
    path: pathlib.Path
    layers: int
    experts: int
    hidden: int
    inter: int
    group: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    prefix: str = "model.language_model"
    projs: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj")
    kinds: tuple[str, str, str] = ("qweight", "qzeros", "scales")

    def name(self, layer: int, expert: int, proj: str, kind: str) -> str:
        return f"{self.prefix}.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"

    def tensor(self, layer: int, expert: int, proj: str, kind: str) -> torch.Tensor:
        return self.tensors[self.name(layer, expert, proj, kind)]


def make_checkpoint(
    path: pathlib.Path,
    *,
    layers: int = 2,
    experts: int = 8,
    hidden: int = 256,
    inter: int = 128,
    group: int = 64,
    seed: int = 0,
    shards: int = 2,
    index: bool = True,
    decoys: bool = True,
    prefix: str = "model.language_model",
    projs: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    experts_infix: str = "mlp.experts",
    scales_dtype: torch.dtype = torch.float16,
    config: bool = True,
    quantization_config: dict | None = None,
    qzeros_value: int = SYM_QZEROS,
) -> Checkpoint:
    path.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    ck = Checkpoint(
        path, layers, experts, hidden, inter, group, prefix=prefix, projs=projs
    )
    groups13 = 1 if group == -1 else hidden // group
    groups2 = 1 if group == -1 else inter // group

    def words(*shape: int) -> torch.Tensor:
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int32)

    def scales(*shape: int) -> torch.Tensor:
        return (torch.rand(*shape, generator=g) * 0.1 + 0.01).to(scales_dtype)

    def zeros(*shape: int) -> torch.Tensor:
        return torch.full(shape, qzeros_value, dtype=torch.int32)

    gate, up, down = projs
    per_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in range(layers):
        tensors: dict[str, torch.Tensor] = {}
        for expert in range(experts):
            base = f"{prefix}.layers.{layer}.{experts_infix}.{expert}"
            for proj in (gate, up):
                tensors[f"{base}.{proj}.qweight"] = words(hidden // 8, inter)
                tensors[f"{base}.{proj}.qzeros"] = zeros(groups13, inter // 8)
                tensors[f"{base}.{proj}.scales"] = scales(groups13, inter)
            tensors[f"{base}.{down}.qweight"] = words(inter // 8, hidden)
            tensors[f"{base}.{down}.qzeros"] = zeros(groups2, hidden // 8)
            tensors[f"{base}.{down}.scales"] = scales(groups2, hidden)
        if decoys:
            # not store material: the shared expert, attention, a norm
            tensors[f"{prefix}.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = (
                torch.randn(inter, hidden, generator=g).bfloat16()
            )
            tensors[f"{prefix}.layers.{layer}.self_attn.q_proj.weight"] = torch.randn(
                8, hidden, generator=g
            ).bfloat16()
            tensors[f"{prefix}.layers.{layer}.input_layernorm.weight"] = torch.ones(
                hidden
            ).bfloat16()
        per_layer[layer] = tensors
        ck.tensors.update(tensors)
    extra: dict[str, torch.Tensor] = {}
    if decoys:
        # the drafter's experts: bf16, same shape of name under another prefix
        for expert in range(experts):
            for proj in (gate, up, down):
                extra[f"mtp.layers.0.{experts_infix}.{expert}.{proj}.weight"] = (
                    torch.randn(inter, hidden, generator=g).bfloat16()
                )
        extra["lm_head.weight"] = torch.randn(16, hidden, generator=g).bfloat16()
    # shard files: layers dealt round-robin, the drafter and head in the last
    shard_tensors: list[dict[str, torch.Tensor]] = [{} for _ in range(shards)]
    for layer, tensors in per_layer.items():
        shard_tensors[layer % shards].update(tensors)
    shard_tensors[-1].update(extra)
    weight_map = {}
    for i, tensors in enumerate(shard_tensors):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / name))
        weight_map.update({k: name for k in tensors})
    if index:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
    if decoys and index:
        # a file the index does not name (the loader ignores it): it holds a
        # complete extra expert under the real pattern that must not be read
        stray = {
            f"{prefix}.layers.{layers}.{experts_infix}.0.{gate}.qweight": words(
                hidden // 8, inter
            )
        }
        save_file(stray, str(path / "model_extra_tensors.safetensors"))
    if config:
        qcfg = quantization_config or {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": group,
            "sym": True,
            "data_type": "int",
            "autoround_version": "0.15.0",
        }
        (path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Synthetic"],
                    "quantization_config": qcfg,
                    "text_config": {
                        "hidden_size": hidden,
                        "moe_intermediate_size": inter,
                        "num_experts": experts,
                        "num_hidden_layers": layers,
                    },
                }
            )
        )
    return ck


# The routed experts' scheme of a compressed-tensors checkpoint as our
# calibration writes it (int4 symmetric group 128 there; the group here is the
# fixture's), the other modules left out.
def ct_quantization_config(group: int) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_1": {
                "targets": [
                    (
                        "re:^(?!(.*\\.)?mtp\\.)(.*\\.)?layers\\.\\d+\\.mlp\\.experts"
                        "\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
                    )
                ],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": group,
                    "strategy": "group",
                    "block_structure": None,
                    "dynamic": False,
                    "actorder": None,
                    "observer": "minmax",
                    "observer_kwargs": {},
                },
                "input_activations": None,
                "output_activations": None,
                "format": "pack-quantized",
            }
        },
        "ignore": [],
    }


def make_ct_checkpoint(gptq: Checkpoint, path: pathlib.Path) -> Checkpoint:
    """The same experts as ``gptq`` in compressed-tensors ``pack-quantized``
    form: ``weight_packed`` = qweight transposed ([out, in/8]; a GPTQ word and
    a compressed-tensors word both hold 8 consecutive ``in`` values, value + 8,
    low nibble first), ``weight_scale`` = scales transposed, ``weight_shape``
    = [out, in] as int32, no zero points; one shard, an index, config.json."""
    path.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(
        path, gptq.layers, gptq.experts, gptq.hidden, gptq.inter, gptq.group,
        prefix=gptq.prefix, projs=gptq.projs,
        kinds=("weight_packed", "weight_shape", "weight_scale"),
    )  # fmt: skip
    gate, up, down = gptq.projs
    for layer in range(gptq.layers):
        for expert in range(gptq.experts):
            for proj in (gate, up, down):
                q = gptq.tensor(layer, expert, proj, "qweight")
                s = gptq.tensor(layer, expert, proj, "scales")
                out, k = q.shape[1], q.shape[0] * 8
                ck.tensors[ck.name(layer, expert, proj, "weight_packed")] = (
                    q.t().contiguous()
                )
                ck.tensors[ck.name(layer, expert, proj, "weight_scale")] = (
                    s.t().contiguous()
                )
                ck.tensors[ck.name(layer, expert, proj, "weight_shape")] = torch.tensor(
                    [out, k], dtype=torch.int32
                )
    name = "model-00001-of-00001.safetensors"
    save_file(dict(ck.tensors), str(path / name))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": dict.fromkeys(ck.tensors, name),
            }
        )
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Synthetic"],
                "quantization_config": ct_quantization_config(gptq.group),
                "text_config": {
                    "hidden_size": gptq.hidden,
                    "moe_intermediate_size": gptq.inter,
                    "num_experts": gptq.experts,
                    "num_hidden_layers": gptq.layers,
                },
            }
        )
    )
    return ck
