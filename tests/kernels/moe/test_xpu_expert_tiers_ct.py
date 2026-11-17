# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert tiers for compressed-tensors ``pack-quantized`` int4 experts, on CPU.

A compressed-tensors int4 expert is the GPTQ expert transposed: ``weight_packed``
``[out, in/8]`` int32 against GPTQ's ``qweight`` ``[in/8, out]``, both 8
consecutive ``in`` values per word, value + 8, low nibble first; scales
``[out, groups]`` against ``[groups, out]``; no zero points (symmetric). After
the method's post-load step both are the bytes the kernels take. The tests
hold the tiers to that: the gate, the loader's routing, the pinned rows, the
store (engine-built and tool-built), a host-computed slot and an EPLB move,
each with compressed-tensors experts and each against the GPTQ path of the
same experts.
"""

import json
import os
import pathlib

import numpy as np
import pytest
import torch

from tests.quantization import test_ct_w4a8_moe_xpu as ct_w4a8_tests
from tests.quantization.test_ct_w4a8_moe_xpu import (
    _INT8_TOKEN,
    _platform,
    _TestRoutedExperts,
)
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _process_weights_xpu,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as ct_moe_pkg,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.offloader import expert_store_build as ebuild
from vllm.model_executor.offloader import expert_store_ckpt as eckpt
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from . import test_xpu_cpu_experts as cpu_tests
from . import test_xpu_expert_store_ckpt as ckpt_tests
from .ckpt_fixture import ct_quantization_config, make_checkpoint, make_ct_checkpoint
from .test_xpu_cpu_experts import CPU_CFG, _units
from .test_xpu_expert_store_ckpt import (
    CkptEnv,
    _loader_stub,
    engine_run,
    same_bytes,
    source_for,
)
from .xpu_store_harness import (
    _GPTQ,
    GROUP,
    HIDDEN,
    INTER,
    LAYERS,
    NAME,
    E,
    Env,
    _Arena,
    _env_process_ct,
    _Linear,
    ct_method,
    expert_in_store,
    pool_rows,
    run,
)

pytestmark = pytest.mark.skip_global_cleanup

# The fixtures of the suites this one extends: the host op (real or its torch
# reference), the kernels' source and their own implement_zp, and the
# recorded backend selection of the compressed-tensors MoE methods.
op = cpu_tests.op
kernels_src = ckpt_tests.kernels_src
real_transform = ckpt_tests.real_transform
selections = ct_w4a8_tests.selections

BF16 = torch.bfloat16
PARAMS = {
    "hidden_size": HIDDEN,
    "intermediate_size_per_partition": INTER,
    "params_dtype": BF16,
}
# config.json, the safetensors headers and the index of a real checkpoint (no
# tensor bytes): the header test below mirrors them as sparse files.
HEADERS_ENV = "VLLM_TEST_CT_CKPT_HEADERS"


def _ct_layer(tiers: xt.ExpertTiers) -> torch.nn.Module:
    """The method's own create_weights for the card rows, then the twins."""
    layer = torch.nn.Module()
    CompressedTensorsWNA16MoEMethod.create_weights(
        ct_method(), layer=layer, num_experts=tiers.num_fast, **PARAMS
    )
    tiers.attach_twins(layer)
    return layer


# -- the gate -----------------------------------------------------------------


def test_the_gate_takes_ct_int4_experts_and_says_so(monkeypatch, tmp_path):
    said: list[str] = []
    monkeypatch.setattr(
        xt.logger, "info_once", lambda msg, *a, **k: said.append(msg % a)
    )
    env = Env(monkeypatch, tmp_path, store=False, fmt="ct")
    env.construct()
    assert all(t.expert_format is xt.CT_PACKED_INT4 for t in env.tiers)
    assert any("compressed-tensors pack-quantized int4 experts" in s for s in said)
    assert xt.tier_expert_format(ct_method()) == (xt.CT_PACKED_INT4, "")
    assert xt.tier_expert_format(_GPTQ()) == (xt.GPTQ_INT4, "")


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"bits": 8}, "8-bit"),
        ({"symmetric": False}, "asymmetric"),
        ({"strategy": "channel", "group_size": -1}, "strategy"),
        ({"backend": WNA16MoEBackend.TRITON}, "backend"),
        ({"has_bias": True}, "bias"),
    ],
)
def test_the_gate_keeps_other_ct_experts_in_vram(monkeypatch, tmp_path, kw, why):
    warned: list[str] = []
    monkeypatch.setattr(
        xt.logger, "warning_once", lambda msg, *a, **k: warned.append(msg % a)
    )
    Env(monkeypatch, tmp_path, store=False, fmt="ct")  # the settings, faked
    fmt, reason = xt.tier_expert_format(ct_method(**kw))
    assert fmt is None and why in reason
    moe_config = type("M", (), {"num_experts": E})()
    assert (
        xt.maybe_tier(
            NAME.format(0), moe_config, _Linear(0, 1), ct_method(**kw), PARAMS
        )
        is None
    )
    assert why in warned[-1] and "stay in VRAM" in warned[-1]


def test_expert_bytes_match_the_ct_create_weights():
    layer = torch.nn.Module()
    with torch.device("meta"):
        CompressedTensorsWNA16MoEMethod.create_weights(
            ct_method(), layer=layer, num_experts=1, **PARAMS
        )
    names = {n for n, _ in layer.named_parameters()}
    # every per-expert tensor the method makes is twinned for the slow slots
    assert names <= set(xt._EXPERT_PARAMS)
    built = sum(
        p.numel() * p.element_size()
        for n, p in layer.named_parameters()
        if n in (*xt.CT_PACKED_INT4.weights, *xt.CT_PACKED_INT4.scales)
    )
    assert xt.expert_bytes(ct_method(), **PARAMS) == built
    assert xt.expert_bytes(_GPTQ(), **PARAMS) == built
    assert xt.slot_tensor_bytes(ct_method(), **PARAMS) == xt.slot_tensor_bytes(
        _GPTQ(), **PARAMS
    )


def _qfn_experts_config(input_activations=None) -> CompressedTensorsConfig:
    """The routed experts' scheme of our compressed-tensors checkpoint (the
    capability checkpoint's config group, verbatim but for this test's group
    size), or the real config.json's when the header directory is given."""
    qc = ct_quantization_config(128)
    real = os.environ.get(HEADERS_ENV)
    if real and os.path.isfile(os.path.join(real, "config.json")):
        with open(os.path.join(real, "config.json")) as handle:
            qc = json.load(handle)["quantization_config"]
    if input_activations is not None:
        for group in qc["config_groups"].values():
            if "experts" in " ".join(group["targets"]):
                group["input_activations"] = dict(input_activations)
    return CompressedTensorsConfig.from_config(qc)


@pytest.mark.parametrize("declared_w4a8", [False, True])
def test_the_qfn_expert_scheme_resolves_to_a_method_the_tiers_take(
    selections, monkeypatch, declared_w4a8
):
    """The quantization config of our checkpoint, resolved the way the model's
    MoE layers resolve it (CompressedTensorsMoEMethod.get_moe_method, the
    backend selector stubbed to XPU on this CPU host): W4A16 as written, and a
    checkpoint that declares int8 per-token activations (W4A8), both give the
    WNA16 method, which the tiers take."""
    _platform(monkeypatch, xpu=True)
    config = _qfn_experts_config(_INT8_TOKEN if declared_w4a8 else None)
    method = ct_moe_pkg.CompressedTensorsMoEMethod.get_moe_method(
        config, _TestRoutedExperts(), "model.language_model.layers.0.mlp.experts"
    )
    assert isinstance(method, CompressedTensorsWNA16MoEMethod)
    assert method.group_size == 128 and method.symmetric and method.num_bits == 4
    assert xt.tier_expert_format(method) == (xt.CT_PACKED_INT4, "")


# -- the loader ------------------------------------------------------------------


def test_the_loader_routes_ct_tensors_of_slow_slots_to_twins(monkeypatch, tmp_path):
    from safetensors import safe_open

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    gptq = make_checkpoint(
        tmp_path / "g", layers=1, experts=E, hidden=HIDDEN, inter=INTER, group=GROUP
    )
    ck = make_ct_checkpoint(gptq, tmp_path / "c")
    env = Env(monkeypatch, tmp_path / "env", store=False, fmt="ct")
    env.construct()
    tiers = env.tiers[0]
    layer = _ct_layer(tiers)
    assert set(tiers.twins) == {
        "w13_weight_packed", "w2_weight_packed", "w13_weight_scale",
        "w2_weight_scale", "w13_weight_shape", "w2_weight_shape",
    }  # fmt: skip
    stub = _loader_stub(tiers)
    stub.quant_method = ct_method()
    path = str(ck.path / "model-00001-of-00001.safetensors")
    with safe_open(path, framework="pt") as handle:
        for expert in range(E):
            for proj, shard in (
                ("gate_proj", "w1"),
                ("down_proj", "w2"),
                ("up_proj", "w3"),
            ):
                for kind in ("weight_packed", "weight_scale", "weight_shape"):
                    pname = ("w2" if shard == "w2" else "w13") + "_" + kind
                    RoutedExperts.weight_loader(
                        stub,
                        getattr(layer, pname),
                        handle.get_tensor(ck.name(0, expert, proj, kind)),
                        f"layers.0.mlp.experts.{expert}.{proj}.{kind}",
                        shard,
                        expert,
                    )
    for slot in range(E):
        if slot < tiers.num_fast:
            rows = {n: getattr(layer, n)[slot] for n in tiers.expert_format.weights}
            scales = {n: getattr(layer, n)[slot] for n in tiers.expert_format.scales}
            shape = layer.w13_weight_shape[slot]
        else:
            row = slot - tiers.num_fast
            rows = {n: tiers.twins[n][row] for n in tiers.expert_format.weights}
            scales = {n: tiers.twins[n][row] for n in tiers.expert_format.scales}
            shape = tiers.twins["w13_weight_shape"][row]
        gate_up = torch.cat(
            [ck.tensor(0, slot, p, "weight_packed") for p in ("gate_proj", "up_proj")]
        )
        assert torch.equal(rows["w13_weight_packed"], gate_up)
        assert torch.equal(
            rows["w2_weight_packed"], ck.tensor(0, slot, "down_proj", "weight_packed")
        )
        want_s = torch.cat(
            [ck.tensor(0, slot, p, "weight_scale") for p in ("gate_proj", "up_proj")]
        ).to(BF16)
        assert torch.equal(scales["w13_weight_scale"], want_s)
        assert shape.tolist() == [INTER, HIDDEN]  # the last shard written (up)


# -- the bytes: the kernels' own transform and compressed-tensors' own packer --


def _pack_gptq(values: torch.Tensor) -> torch.Tensor:
    """GPTQ words ``[E, in/8, out]`` of int4 ``values`` ``[E, out, in]``:
    word (k8, n) holds in values 8*k8 .. 8*k8+7 of row n, value + 8, value j
    at bits 4j."""
    u = (values.to(torch.int64) + 8).permute(0, 2, 1)  # [E, in, out]
    e, k, n = u.shape
    u = u.reshape(e, k // 8, 8, n)
    word = torch.zeros(e, k // 8, n, dtype=torch.int64)
    for j in range(8):
        word |= u[:, :, j, :] << (4 * j)
    return (word - ((word >> 31) & 1) * (1 << 32)).to(torch.int32)


def _pack_ct(values: torch.Tensor) -> tuple[torch.Tensor, str]:
    """compressed-tensors words ``[E, out, in/8]``: the library's own
    ``pack_to_int32`` when it is installed, else its statement (the GPTQ
    words transposed)."""
    try:
        from compressed_tensors.compressors import pack_to_int32
    except ImportError:
        try:
            from compressed_tensors.compressors.pack_quantized.helpers import (
                pack_to_int32,
            )
        except ImportError:
            return _pack_gptq(values).transpose(1, 2).contiguous(), "statement"
    return pack_to_int32(values, 4), "compressed_tensors"


def _decode_s4(rows: torch.Tensor) -> torch.Tensor:
    """Kernel bytes ``[E, out, in/2]`` (two's-complement nibbles, low nibble
    first) -> int values ``[E, out, in]``."""
    lo = (rows & 0x0F).to(torch.int16)
    hi = (rows >> 4).to(torch.int16)
    both = torch.stack([lo, hi], dim=-1).reshape(*rows.shape[:-1], -1)
    return torch.where(both >= 8, both - 16, both)


def test_ct_words_are_gptq_words_transposed_and_give_the_kernel_bytes(
    monkeypatch, real_transform
):
    g = torch.Generator().manual_seed(11)
    values = torch.randint(-8, 8, (3, 32, 256), generator=g, dtype=torch.int8)
    ct_words, source = _pack_ct(values)
    gptq_words = _pack_gptq(values)
    assert ct_words.dtype == torch.int32 and ct_words.shape == (3, 32, 32)
    assert torch.equal(ct_words.transpose(1, 2), gptq_words), source
    monkeypatch.setattr(xt, "_ARENA", _Arena())
    cpu = torch.device("cpu")
    pinned_ct = xt._repack_int4_pinned(ct_words, cpu, rows=5, k_first=False)
    pinned_gptq = xt._repack_int4_pinned(gptq_words, cpu, rows=5, k_first=True)
    assert torch.equal(pinned_ct, pinned_gptq)
    assert not pinned_ct[3:].any()  # the rows beyond the experts
    # the card rows: the oracle's XPU post-load step, then the kernels'
    # implement_zp, as XpuFusedMoe.__init__ applies it
    w13, _, _, _, _, _ = _process_weights_xpu(
        None, None, ct_words, ct_words, torch.ones(1), torch.ones(1)
    )
    card = real_transform.implement_zp(w13.contiguous())
    assert torch.equal(card, pinned_ct[:3])
    # and the bytes say the values: signed nibbles, low first, along in
    assert torch.equal(_decode_s4(pinned_ct[:3]), values.to(torch.int16))
    # the store's unit (the build tool's function) gives the same bytes
    scales = torch.rand(3, 32, 2, generator=g).to(torch.float16)
    unit_ct = xt.kernel_layout_unit(
        ct_words[0], scales[0], ct_words[1], scales[1], BF16, k_first=False
    )
    unit_g = xt.kernel_layout_unit(
        gptq_words[0], scales[0].t(), gptq_words[1], scales[1].t(), BF16
    )
    for key in unit_ct:
        assert torch.equal(unit_ct[key], unit_g[key]), key
    assert torch.equal(unit_ct["w13"], pinned_ct[0])
    assert torch.equal(unit_ct["w13_scales"], scales[0].to(BF16))


# -- the engine's path from safetensors, both formats ---------------------------


class CtCkptEnv(CkptEnv):
    """The harness with a compressed-tensors checkpoint read from safetensors
    through the method's own create_weights, the engine's ``weight_loader``
    and the oracle's own XPU post-load step; the kernels' own implement_zp."""

    def __init__(self, monkeypatch, tmp_path, ck, kernels_src, **kw):
        super().__init__(monkeypatch, tmp_path, ck, kernels_src, fmt="ct", **kw)

    def load(self) -> None:
        from safetensors import safe_open

        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        index = json.loads((self.ck.path / "model.safetensors.index.json").read_text())
        shard_of = index["weight_map"]
        for i, tiers in enumerate(self.tiers):
            layer = _ct_layer(tiers)
            stub = _loader_stub(tiers)
            for expert in range(self.local):
                for proj, shard in (
                    ("gate_proj", "w1"),
                    ("down_proj", "w2"),
                    ("up_proj", "w3"),
                ):
                    for kind in ("weight_packed", "weight_scale", "weight_shape"):
                        name = self.ck.name(i, expert, proj, kind)
                        path = str(self.ck.path / shard_of[name])
                        with safe_open(path, framework="pt") as handle:
                            value = handle.get_tensor(name)
                        pname = ("w2" if shard == "w2" else "w13") + "_" + kind
                        RoutedExperts.weight_loader(
                            stub,
                            getattr(layer, pname),
                            value,
                            f"layers.{i}.mlp.experts.{expert}.{proj}.{kind}",
                            shard,
                            expert,
                        )
            self.layers.append(layer)

    def process(self) -> None:
        _env_process_ct(self)


def ct_engine_run(monkeypatch, tmp_path, ck, kernels_src, build="auto") -> CtCkptEnv:
    env = CtCkptEnv(monkeypatch, tmp_path, ck, kernels_src, store=True, build=build)
    env.construct()
    env.load()
    env.process()
    return env


def _all_units(env) -> dict:
    return {
        (layer, expert): expert_in_store(env, layer, expert)
        for layer in range(LAYERS)
        for expert in range(E)
    }


def _checkpoints(tmp_path):
    gptq = make_checkpoint(
        tmp_path / "gptq", layers=LAYERS, experts=E, hidden=HIDDEN, inter=INTER,
        group=GROUP,
    )  # fmt: skip
    return gptq, make_ct_checkpoint(gptq, tmp_path / "ct")


def test_ct_and_gptq_checkpoints_of_the_same_experts_give_the_same_rows_and_units(
    monkeypatch, tmp_path, kernels_src
):
    gptq, ct = _checkpoints(tmp_path)
    ge = engine_run(monkeypatch, tmp_path / "ge", gptq, kernels_src)
    g_rows = [pool_rows(t) for t in ge.tiers]
    g_fast = [
        {f: getattr(t._impls[0], f).clone() for f, _ in xt._TIER_FIELDS}
        for t in ge.tiers
    ]
    g_units = _all_units(ge)
    g_schema = ge.tiers[0].store_binding.session.schema
    tstore.reset_sessions()

    ce = ct_engine_run(monkeypatch, tmp_path / "ce", ct, kernels_src)
    session = ce.tiers[0].store_binding.session
    assert session.mode == "build"
    assert session.schema == g_schema
    for i, tiers in enumerate(ce.tiers):
        assert tiers.expert_format is xt.CT_PACKED_INT4
        for key, rows in pool_rows(tiers).items():
            assert same_bytes(rows, g_rows[i][key]), (i, key)
        for field, _ in xt._TIER_FIELDS:
            assert same_bytes(getattr(tiers._impls[0], field), g_fast[i][field])
    for key, unit in _all_units(ce).items():
        for name, tensor in unit.items():
            assert same_bytes(tensor, g_units[key][name]), (key, name)
    tstore.reset_sessions()


def test_the_build_tool_reads_a_ct_checkpoint_into_the_engines_units(
    monkeypatch, tmp_path, kernels_src
):
    gptq, ct = _checkpoints(tmp_path)
    ce = ct_engine_run(monkeypatch, tmp_path / "ce", ct, kernels_src)
    session = ce.tiers[0].store_binding.session
    source = source_for(ct, kernels_src, engine_quantization="compressed-tensors")
    g_source = source_for(gptq, kernels_src)
    assert source.layout == eckpt.CT_PACKED and g_source.layout == eckpt.GPTQ
    assert (source.layers, source.experts, source.group_size) == (LAYERS, E, GROUP)
    assert source.schema == session.schema == g_source.schema
    assert source.identity == session.identity  # the engine would fill from it
    for (layer, expert), want in _all_units(ce).items():
        got = source.read((layer, expert))
        other = g_source.read((layer, expert))
        assert list(got) == list(want) == ["w13", "w2", "w13_scales", "w2_scales"]
        for key in want:
            assert same_bytes(got[key], want[key]), (layer, expert, key)
            assert same_bytes(got[key], other[key]), (layer, expert, key)
    info = source.info()
    assert info["layout"] == "compressed-tensors"
    assert info["qzeros"]["checked_units"] == LAYERS * E
    assert info["qzeros"]["not_symmetric_units"] == 0  # every weight_shape right
    source.close()
    g_source.close()
    tstore.reset_sessions()


def test_a_wrong_weight_shape_is_counted_and_refused_when_strict(tmp_path, kernels_src):
    from safetensors.torch import save_file

    _, ct = _checkpoints(tmp_path)
    name = ct.name(1, 2, "down_proj", "weight_shape")
    ct.tensors[name] = torch.tensor([INTER, HIDDEN], dtype=torch.int32)
    save_file(dict(ct.tensors), str(ct.path / "model-00001-of-00001.safetensors"))
    source = source_for(ct, kernels_src, engine_quantization="compressed-tensors")
    source.read((1, 2))
    assert source.info()["qzeros"]["first_not_symmetric"] == [(1, 2)]
    strict = source_for(
        ct, kernels_src, engine_quantization="compressed-tensors", strict_qzeros=True
    )
    with pytest.raises(eckpt.CheckpointError, match="weight_shape"):
        strict.read((1, 2))


def test_a_store_the_tool_built_from_a_ct_checkpoint_fills_the_engine(
    monkeypatch, tmp_path, kernels_src
):
    _, ct = _checkpoints(tmp_path)
    first = ct_engine_run(monkeypatch, tmp_path / "first", ct, kernels_src)
    ref_rows = [pool_rows(t) for t in first.tiers]
    tstore.reset_sessions()
    dirs = [str(tmp_path / "second" / "d0"), str(tmp_path / "second" / "d1")]
    for d in dirs:
        pathlib.Path(d).mkdir(parents=True)
    source = source_for(ct, kernels_src, engine_quantization="compressed-tensors")
    for part in range(2):
        ebuild.build_source(dirs, source, part=(part, 2), workers=2)
    ebuild.finalize_store(dirs)
    source.close()
    second = ct_engine_run(monkeypatch, tmp_path / "second", ct, kernels_src)
    session = second.tiers[0].store_binding.session
    assert session.status == "complete" and session.mode == "fill"
    assert all(t.fills_from_store for t in second.tiers)
    for built, want in zip(second.tiers, ref_rows):
        for key, rows in pool_rows(built).items():
            assert torch.equal(rows, want[key]), key
    tstore.reset_sessions()


# -- host slots and an EPLB move ------------------------------------------------


def _ct_host_env(monkeypatch, tmp_path, pinned=2):
    run(monkeypatch, tmp_path, store=True, fmt="ct")
    return run(
        monkeypatch, tmp_path, store=True, pinned=pinned, store_extra=CPU_CFG, fmt="ct"
    )


def test_ct_host_slots_are_computed_as_the_card_computes_them(
    monkeypatch, tmp_path, op
):
    env = _ct_host_env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        assert tiers.expert_format is xt.CT_PACKED_INT4
        assert (tiers.num_fast, tiers.num_pinned, tiers.num_host) == (3, 2, 3)
        assert tiers.placement.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
    tiers = env.tiers[1]
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1)).to(BF16)
    out = torch.full_like(x, 7.0)
    assert tiers.cpu.compute(tiers, ids, w, x, out)
    calls = [(5, 0, 0.3), (6, 1, 0.6), (7, 2, 0.5), (5, 2, 0.5)]
    want = cx.reference_int4_experts(x, _units(env, 1, {5, 6, 7}), calls, INTER, GROUP)
    if op:
        assert cx.bf16_steps(out, want) <= 1.0
    else:
        assert torch.equal(out, want)
    tstore.reset_sessions()


def test_ct_host_units_are_the_gptq_host_units(monkeypatch, tmp_path, op):
    """The units the CPU tier computes from: the same bytes for the same
    experts whichever format they came in (the harness gives both formats the
    same int4 experts)."""
    ct_env = _ct_host_env(monkeypatch, tmp_path / "ct")
    ct_units = _units(ct_env, 0, set(range(E)))
    cx.reset_runtime()
    tstore.reset_sessions()
    run(monkeypatch, tmp_path / "g", store=True)
    g_env = run(monkeypatch, tmp_path / "g", store=True, pinned=2, store_extra=CPU_CFG)
    g_units = _units(g_env, 0, set(range(E)))
    for lg in range(E):
        for key in g_units[lg]:
            assert torch.equal(ct_units[lg][key], g_units[lg][key]), (lg, key)
    tstore.reset_sessions()


def test_an_eplb_move_with_ct_experts(monkeypatch, tmp_path, op):
    """A pinned slot and a host slot trade experts; the pinned rows get the
    store's unit and the host slot computes the other expert."""
    env = _ct_host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    old = np.arange(8)
    new = old.copy()
    new[3], new[5] = 5, 3
    load = np.array([9, 9, 9, 1, 4, 8, 0, 0], dtype=float)
    tiers.commit(tiers.prepare(old, new, load, 1.1))
    p = tiers.placement
    assert p.tier[3] == POOL and p.tier[5] == HOST
    assert tiers.slot_logical[3] == 5 and tiers.slot_logical[5] == 3
    for name, t in tiers.slow.items():
        want = expert_in_store(env, 0, 5)[name].reshape(-1).view(torch.uint8)
        assert torch.equal(t[p.row[3]].reshape(-1).view(torch.uint8), want)
    x = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(2)).to(BF16)
    out = torch.empty_like(x)
    assert tiers.cpu.compute(
        tiers, torch.tensor([[5, 0]], dtype=torch.int32), torch.ones(1, 2), x, out
    )
    want = cx.reference_int4_experts(
        x, _units(env, 0, {3}), [(3, 0, 1.0)], INTER, GROUP
    )
    assert cx.bf16_steps(out, want) <= (1.0 if op else 0.0)
    summary = tiers.scrub(old[[0, 1, 2, 5, 4, 3, 6, 7]], full=True)
    assert summary["mismatches"] == 0 and summary["maps_equal"]
    tstore.reset_sessions()


def test_eplb_sees_the_ct_layer_as_store_backed_rows(monkeypatch, tmp_path):
    from vllm.distributed.eplb.expert_weights import StoreBackedWeight

    ct = run(monkeypatch, tmp_path / "ct", store=True, fmt="ct")
    ct_shapes = [
        [(type(w), len(w), w.tier_sizes, tuple(w.shape)) for w in t.eplb_weights()]
        for t in ct.tiers
    ]
    g = run(monkeypatch, tmp_path / "g", store=True)
    for tiers, shapes in zip(g.tiers, ct_shapes):
        want = [
            (type(w), len(w), w.tier_sizes, tuple(w.shape))
            for w in tiers.eplb_weights()
        ]
        assert shapes == want  # EPLB sees the same layer whatever the format
        assert all(s[0] is StoreBackedWeight and s[1] == 8 for s in shapes)
        assert len(shapes) == 4
    tstore.reset_sessions()


# -- a real checkpoint's names, from its headers ----------------------------------


def _sparse_mirror(src: str, dst: pathlib.Path) -> None:
    """Each ``<file>.header.json`` as ``<file>``: the 8-byte length, the
    header, and a hole as long as the tensors (no bytes stored)."""
    dst.mkdir(parents=True)
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if name.endswith(".safetensors.header.json"):
            with open(path, "rb") as handle:
                header = handle.read()
            end = max(
                (e["data_offsets"][1] for k, e in json.loads(header).items()
                 if k != "__metadata__"),
                default=0,
            )  # fmt: skip
            out = dst / name[: -len(".header.json")]
            with open(out, "wb") as handle:
                handle.write(len(header).to_bytes(8, "little"))
                handle.write(header)
                handle.truncate(8 + len(header) + end)
        elif name in ("config.json", "model.safetensors.index.json"):
            (dst / name).write_bytes(pathlib.Path(path).read_bytes())


@pytest.mark.skipif(
    not os.environ.get(HEADERS_ENV), reason=f"{HEADERS_ENV} names no header copy"
)
def test_the_capability_checkpoint_headers(tmp_path, kernels_src):
    """The capability checkpoint (int8 dense parts, int4 experts) by its own
    names and shapes: the build tool finds the routed experts as
    compressed-tensors, the unit is what the engine's tiers lay out for the
    method's parameters at these sizes, and every routed-expert tensor maps to
    a parameter the tiers twin."""
    mirror = tmp_path / "ckpt"
    _sparse_mirror(os.environ[HEADERS_ENV], mirror)
    source = eckpt.CheckpointExperts(
        str(mirror),
        engine_quantization="compressed-tensors",
        kernels_src=kernels_src,
        check_qzeros=False,
    )
    cfg = json.loads((mirror / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    hidden, inter = text["hidden_size"], text["moe_intermediate_size"]
    assert source.layout == eckpt.CT_PACKED
    assert (source.layers, source.experts) == (
        text["num_hidden_layers"],
        text["num_experts"],
    )
    assert (source.hidden, source.inter, source.group_size) == (hidden, inter, 128)
    method = ct_method(group_size=128)
    params = {
        "hidden_size": hidden,
        "intermediate_size_per_partition": inter,
        "params_dtype": BF16,
    }
    assert source.unit_tensor_bytes == xt.slot_tensor_bytes(method, **params)
    assert source.unit_nbytes == xt.expert_bytes(method, **params)
    unit = source.read((0, 0))  # zeros (a hole), in the unit's shape
    assert {k: v.numel() for k, v in unit.items()} == source.unit_tensor_bytes
    # the engine's names: experts.<e>.<proj>.<kind> -> w13_/w2_<kind>
    twinned = set(xt._EXPERT_PARAMS)
    kinds = {
        name.rsplit(".", 1)[1]
        for name in source.files.refs
        if source._rx.fullmatch(name)
    }
    assert kinds == {"weight_packed", "weight_scale", "weight_shape"}
    assert {f"w13_{k}" for k in kinds} | {f"w2_{k}" for k in kinds} <= twinned
    skipped = {s["pattern"]: s["reason"] for s in source.info()["skipped"]}
    assert any(p.startswith("mtp.") for p in skipped)  # the drafter stays resident
    source.close()
