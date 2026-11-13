# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The checkpoint reader of the expert store's build tool, on CPU.

The reference is the engine's own path: the loader's ``weight_loader`` fuses
gate and up from a safetensors checkpoint into the layer's parameters, the
post-load step and ``finalize`` and the experts' impls lay them out, and the
tiers write the units (``xpu_store_harness``, with the accelerator and the
kernels' native code faked, the transform being the kernels' own
``implement_zp`` source). The reader must give those units byte for byte, and a
store built from it must be accepted by the engine's fill.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import card_guard
from vllm.model_executor.offloader import expert_store_build as ebuild
from vllm.model_executor.offloader import expert_store_ckpt as eckpt
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_store import ExpertStore

from .ckpt_fixture import Checkpoint, make_checkpoint
from .xpu_store_harness import (
    GROUP,
    HIDDEN,
    INTER,
    LAYERS,
    N13,
    E,
    Env,
    FakeXpuFusedMoe,
    ImmediateOps,
    expert_in_store,
    pool_rows,
)

pytestmark = pytest.mark.skip_global_cleanup

MODULE = "vllm.model_executor.offloader.expert_store_build"
KERNELS_ENV = "VLLM_XPU_KERNELS_SRC"
# The kernels library's function, copied verbatim from its source file
# (fused_moe_interface.py, ``implement_zp``) for hosts without the library.
IMPLEMENT_ZP = """
def implement_zp(qweight):
    # change u4 to s4 to avoid zero point in gemm kernel
    # only support default zero point now
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result
"""
FAKE_KERNELS_MODULE = (
    "import torch\n"
    "from . import _C  # a native extension: not importable here, never imported\n"
    "_ACTIVATION = torch.ops._C.silu_and_mul  # evaluated at import by the real one\n"
    + IMPLEMENT_ZP
)


@pytest.fixture(scope="module")
def kernels_src(tmp_path_factory) -> str:
    """A directory holding ``vllm_xpu_kernels``: the real source tree when
    ``VLLM_XPU_KERNELS_SRC`` names one, else a package of the verbatim function
    that cannot be imported (it needs a native module), as the real one cannot
    without its build."""
    real = os.environ.get(KERNELS_ENV)
    if real and os.path.isfile(
        os.path.join(real, "vllm_xpu_kernels", "fused_moe_interface.py")
    ):
        return real
    root = tmp_path_factory.mktemp("kernels")
    pkg = root / "vllm_xpu_kernels"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise ImportError('native build needed')\n")
    (pkg / "fused_moe_interface.py").write_text(FAKE_KERNELS_MODULE)
    return str(root)


def small(path, **kw) -> Checkpoint:
    base = dict(hidden=128, inter=64, group=32)
    return make_checkpoint(path, **(base | kw))


def source_for(ck: Checkpoint, kernels_src: str, **kw) -> eckpt.CheckpointExperts:
    kw.setdefault("engine_quantization", "gptq")
    kw.setdefault("engine_model", str(ck.path))
    return eckpt.CheckpointExperts(str(ck.path), kernels_src=kernels_src, **kw)


@pytest.fixture
def real_transform(monkeypatch, kernels_src):
    """The kernels' own ``implement_zp`` under the real module's name, loaded
    the way the build tool loads it (no import of the package)."""
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface", False)
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels", False)
    module = eckpt.load_kernels_transform(kernels_src)
    return module


# --------------------------------------------------------------------------
# The engine's path from a safetensors checkpoint
# --------------------------------------------------------------------------


class RealZpFusedMoe(FakeXpuFusedMoe):
    """``XpuFusedMoe.__init__`` for int4: ``implement_zp`` (the kernels' own)
    over the fast tier's weights, in place, then viewed as int8."""

    def __init__(self, w13, w13_scales, w13_bias, w2, w2_scales, w2_bias, **kw):
        zp = sys.modules["vllm_xpu_kernels.fused_moe_interface"].implement_zp
        if not hasattr(w13, "xpu_fused_moe"):
            for w in (w13, w2):
                w.data = zp(w.data.contiguous()).view(torch.int8).clone()
            w13.xpu_fused_moe = True
        self.w13, self.w2 = w13, w2
        self.gemm1_wei_scales, self.gemm2_wei_scales = w13_scales, w2_scales
        self.expert_map = kw["expert_map"]
        self.num_experts = kw["num_experts"]


def _loader_stub(tiers: xt.ExpertTiers) -> SimpleNamespace:
    """``self`` for ``RoutedExperts.weight_loader`` with the class's own
    loading helpers bound to it."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    stub = SimpleNamespace(
        quant_config=None,
        quant_method=SimpleNamespace(),
        moe_config=SimpleNamespace(
            num_experts=E,
            tp_shard_with_padding=False,
            is_act_and_mul=True,
            tp_rank=0,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        ),
        expert_map_manager=SimpleNamespace(num_fused_shared_experts=0),
        shared_expert_online_loader=None,
        expert_tiers=tiers,
        _loaded_expert_biases=set(),
        layer_name=tiers.layer_name,
    )
    stub._map_global_expert_id_to_local_expert_id = lambda e: e
    for name in (
        "_load_model_weight_or_group_weight_scale",
        "_load_w13",
        "_load_w2",
        "_load_per_channel_weight_scale",
        "_load_per_tensor_weight_scale",
        "_load_single_value",
    ):
        setattr(stub, name, types.MethodType(getattr(RoutedExperts, name), stub))
    for name in ("_get_hidden_dim", "_narrow_expert_data_for_padding", "_to_scalar"):
        setattr(stub, name, getattr(RoutedExperts, name))
    return stub


class CkptEnv(Env):
    """The harness, with the checkpoint read from safetensors files through the
    engine's own ``weight_loader`` and the post-load step's own helper."""

    def __init__(self, monkeypatch, tmp_path, ck: Checkpoint, kernels_src, **kw):
        super().__init__(monkeypatch, tmp_path, **kw)
        import vllm.config as vconfig

        # the engine names the checkpoint by this string, and lists its files
        vconfig.get_current_vllm_config().model_config.model = str(ck.path)
        monkeypatch.delitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface")
        eckpt.load_kernels_transform(kernels_src)
        monkeypatch.setattr(xt, "_xpu_fused_moe", lambda: RealZpFusedMoe)
        self.ck = ck

    def load(self) -> None:
        from safetensors import safe_open

        from vllm.model_executor.layers.fused_moe.routed_experts import (
            FusedMoeWeightScaleSupported,
            RoutedExperts,
        )

        f, ck = self.fast, self.ck
        shapes = {
            "w13_qweight": ((f, HIDDEN // 8, N13), torch.int32),
            "w2_qweight": ((f, INTER // 8, HIDDEN), torch.int32),
            "w13_scales": ((f, HIDDEN // GROUP, N13), torch.bfloat16),
            "w2_scales": ((f, INTER // GROUP, HIDDEN), torch.bfloat16),
            "w13_qzeros": ((f, HIDDEN // GROUP, N13 // 8), torch.int32),
            "w2_qzeros": ((f, INTER // GROUP, HIDDEN // 8), torch.int32),
        }
        index = json.loads((ck.path / "model.safetensors.index.json").read_text())
        shard_of = index["weight_map"]
        for i, tiers in enumerate(self.tiers):
            layer = torch.nn.Module()
            for name, (shape, dtype) in shapes.items():
                param = torch.nn.Parameter(
                    torch.zeros(shape, dtype=dtype), requires_grad=False
                )
                param.is_transposed = True  # type: ignore[attr-defined]
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value  # type: ignore[attr-defined]
                layer.register_parameter(name, param)
            tiers.attach_twins(layer)
            stub = _loader_stub(tiers)
            for expert in range(self.local):
                for proj, shard in (
                    ("gate_proj", "w1"),
                    ("down_proj", "w2"),
                    ("up_proj", "w3"),
                ):
                    for kind in ("qweight", "qzeros", "scales"):
                        name = ck.name(i, expert, proj, kind)
                        with safe_open(
                            str(ck.path / shard_of[name]), framework="pt"
                        ) as handle:
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
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            _process_weights_xpu,
        )

        for i, tiers in enumerate(self.tiers):
            layer = self.layers[i]
            # auto_gptq.process_weights_after_loading: K-first -> N-first
            w13 = layer.w13_qweight.data.transpose(1, 2).contiguous()
            w2 = layer.w2_qweight.data.transpose(1, 2).contiguous()
            s13 = layer.w13_scales.data.transpose(1, 2).contiguous()
            s2 = layer.w2_scales.data.transpose(1, 2).contiguous()
            w13, w2, s13, s2, _, _ = _process_weights_xpu(layer, None, w13, w2, s13, s2)
            layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
            layer.w13_scales = torch.nn.Parameter(s13, requires_grad=False)
            layer.w2_scales = torch.nn.Parameter(s2, requires_grad=False)
            experts = SimpleNamespace(tiers=None)
            tiers.finalize(layer, SimpleNamespace(fused_experts=experts))
        for i, tiers in enumerate(self.tiers):
            layer = self.layers[i]
            tiers.impls(
                layer.w13_weight, layer.w13_scales, layer.w2_weight, layer.w2_scales,
                2, "silu", None,
            )  # fmt: skip
            if tiers.ops is not None:
                tiers.ops = ImmediateOps()


def engine_run(
    monkeypatch, tmp_path, ck, kernels_src, build="auto", construct_device=None
) -> CkptEnv:
    """``construct_device``: build the layers under that default-device
    context, as the model loader builds the model under the accelerator's
    (loading and the post-load step run outside it, as there)."""
    env = CkptEnv(monkeypatch, tmp_path, ck, kernels_src, store=True, build=build)
    if construct_device is None:
        env.construct()
    else:
        with torch.device(construct_device):
            env.construct()
    env.load()
    env.process()
    return env


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def test_reader_gives_the_units_the_engine_builds_from_the_same_checkpoint(
    monkeypatch, tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=LAYERS, experts=E, hidden=HIDDEN, inter=INTER,
        group=GROUP,
    )  # fmt: skip
    env = engine_run(monkeypatch, tmp_path / "engine", ck, kernels_src)
    session = env.tiers[0].store_binding.session
    assert session.mode == "build"
    source = source_for(ck, kernels_src)
    assert (source.layers, source.experts) == (LAYERS, E)
    assert list(source.schema.groups[0].tensors) == list(
        session.schema.groups[0].tensors
    )
    # the identity the engine computed at fill time is the reader's
    assert source.identity == session.identity
    for layer in range(LAYERS):
        for expert in range(E):
            want = expert_in_store(env, layer, expert)
            got = source.read((layer, expert))
            assert list(got) == list(want) == ["w13", "w2", "w13_scales", "w2_scales"]
            for key in want:
                assert got[key].dtype == torch.uint8 and got[key].dim() == 1
                assert same_bytes(got[key], want[key]), (layer, expert, key)
    source.close()


def test_a_store_built_from_the_checkpoint_is_filled_by_the_engine_without_rebuilding(
    monkeypatch, tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=LAYERS, experts=E, hidden=HIDDEN, inter=INTER,
        group=GROUP,
    )  # fmt: skip
    # reference: the engine builds the store itself at its first load
    first = engine_run(monkeypatch, tmp_path / "first", ck, kernels_src)
    ref_rows = [pool_rows(t) for t in first.tiers]
    tstore.reset_sessions()

    # the tool builds a store (two parts, one finalize) where a second engine
    # will look for it
    dirs = [str(tmp_path / "second" / "d0"), str(tmp_path / "second" / "d1")]
    for d in dirs:
        pathlib.Path(d).mkdir(parents=True)
    source = source_for(ck, kernels_src)
    for part in range(2):
        ebuild.build_source(dirs, source, part=(part, 2), workers=2)
    ebuild.finalize_store(dirs)
    source.close()

    second = engine_run(monkeypatch, tmp_path / "second", ck, kernels_src)
    session = second.tiers[0].store_binding.session
    assert session.status == "complete"
    assert session.mode == "fill"  # accepted: not rebuilt
    assert all(t.fills_from_store for t in second.tiers)
    # the host rows it filled from the tool's store are the rows the
    # checkpoint path gives, and the fast rows matched three units per layer
    for built, want in zip(second.tiers, ref_rows):
        for key, rows in pool_rows(built).items():
            assert torch.equal(rows, want[key]), key
    # every unit the engine's first load wrote equals the tool's, unit by unit
    first_dirs = [str(tmp_path / "first" / "d0"), str(tmp_path / "first" / "d1")]
    a = ExpertStore.open(first_dirs, session.identity, 1 << 24, 2, False, direct=False)
    b = ExpertStore.open(dirs, session.identity, 1 << 24, 2, False, direct=False)
    try:
        for layer in range(LAYERS):
            for expert in range(E):
                with a.get((layer, expert)) as ea, b.get((layer, expert)) as eb:
                    for key, tensor in ea.tensors().items():
                        assert same_bytes(tensor, eb.tensors()[key]), (
                            layer,
                            expert,
                            key,
                        )
    finally:
        a.close()
        b.close()
    tstore.reset_sessions()


def test_a_store_of_another_quantization_name_is_another_identity(
    tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=1, experts=2, hidden=128, inter=64, group=32
    )
    a = source_for(ck, kernels_src)
    b = source_for(ck, kernels_src, engine_quantization="inc")
    c = source_for(ck, kernels_src, engine_dtype="float16")
    d = source_for(ck, kernels_src, engine_model="/elsewhere/model")
    assert len({a.identity, b.identity, c.identity, d.identity}) == 4
    with pytest.raises(eckpt.CheckpointError, match="engine_quantization"):
        eckpt.CheckpointExperts(str(ck.path), kernels_src=kernels_src)
    # and it moves with the checkpoint's files (name, size, mtime)
    before = a.identity
    victim = next(ck.path.glob("model-*.safetensors"))
    os.utime(victim, ns=(1, 1))
    assert source_for(ck, kernels_src).identity != before


def test_independent_reference_of_the_layout(tmp_path, kernels_src):
    """The nibble transform and the layout without any engine function: the
    checkpoint's bytes with the zero point removed are the bytes xor 0x88, gate
    rows then up rows, K-contiguous; scales transposed and cast."""
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=1, experts=3, hidden=128, inter=64, group=32
    )
    source = source_for(ck, kernels_src)
    for expert in range(3):

        def t(proj, kind, expert=expert):
            return ck.tensor(0, expert, proj, kind)

        w13_q = np.concatenate(
            [t("gate_proj", "qweight").numpy(), t("up_proj", "qweight").numpy()], axis=1
        )
        want_w13 = np.ascontiguousarray(w13_q.T).view(np.uint8) ^ 0x88
        want_w2 = (
            np.ascontiguousarray(t("down_proj", "qweight").numpy().T).view(np.uint8)
            ^ 0x88
        )
        s13 = torch.cat([t("gate_proj", "scales"), t("up_proj", "scales")], 1)
        want_s13 = s13.T.contiguous().bfloat16()
        got = source.read((0, expert))
        assert got["w13"].numpy().tolist() == want_w13.reshape(-1).tolist()
        assert got["w2"].numpy().tolist() == want_w2.reshape(-1).tolist()
        assert same_bytes(got["w13_scales"], want_s13)
        assert same_bytes(
            got["w2_scales"], t("down_proj", "scales").T.contiguous().bfloat16()
        )
        # each projection's output rows are where the loader puts them
        assert (
            got["w13"].numpy().reshape(128, 64)[:64].tolist()
            == (
                np.ascontiguousarray(t("gate_proj", "qweight").numpy().T).view(np.uint8)
                ^ 0x88
            ).tolist()
        )
    source.close()


def test_the_recorded_example_of_the_kernels_transform(real_transform):
    """``implement_zp`` is, for every byte, an xor with 0x88 (the u4 -> s4
    change in each nibble), and its probe hash is the identity's transform
    part. A recorded value, from the kernels source."""
    import hashlib

    probe = real_transform.implement_zp(torch.arange(256, dtype=torch.uint8))
    assert probe.tolist() == [b ^ 0x88 for b in range(256)]
    assert (
        hashlib.sha256(probe.numpy().tobytes()).hexdigest()[:16] == "8d38fff6efe34092"
    )
    assert xt._layout_fingerprint() == "8d38fff6efe34092"


def test_the_fingerprint_is_taken_on_the_host_under_a_default_device(
    real_transform, tmp_path, kernels_src
):
    """The engine takes the fingerprint while the model is built under the
    accelerator's default-device context; a probe made on that device could
    not be read on the host (``TypeError: can't convert ... device type tensor
    to numpy``). The meta device stands in for the accelerator here: a tensor
    created there cannot be read on the host either, through the same
    default-device mode. The value equals the build tool's, taken without a
    context."""
    ck = small(tmp_path / "ckpt", layers=1, experts=2)
    source = source_for(ck, kernels_src)
    tool_side = source.transform_fingerprint
    source.close()
    with torch.device("meta"):
        assert torch.empty(1).device.type == "meta"  # the context is on
        engine_side = xt._layout_fingerprint()
    assert engine_side == tool_side == "8d38fff6efe34092"


def test_a_tool_built_store_binds_when_the_model_is_built_under_a_default_device(
    monkeypatch, tmp_path, kernels_src
):
    """The boot's order: the layers are constructed under a default-device
    context (the store is bound there), loaded and post-processed outside it.
    A store the tool built is accepted (same identity) and filled from."""
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=LAYERS, experts=E, hidden=HIDDEN, inter=INTER,
        group=GROUP,
    )  # fmt: skip
    dirs = [str(tmp_path / "eng" / "d0"), str(tmp_path / "eng" / "d1")]
    for d in dirs:
        pathlib.Path(d).mkdir(parents=True)
    source = source_for(ck, kernels_src)
    ebuild.build_source(dirs, source, workers=2)
    ebuild.finalize_store(dirs)
    tool_identity = source.identity
    source.close()
    env = engine_run(
        monkeypatch, tmp_path / "eng", ck, kernels_src, construct_device="meta"
    )
    session = env.tiers[0].store_binding.session
    assert session.identity == tool_identity
    assert session.status == "complete"
    assert session.mode == "fill"
    assert all(t.fills_from_store for t in env.tiers)
    tstore.reset_sessions()


# --------------------------------------------------------------------------
# Reading, discovering, refusing
# --------------------------------------------------------------------------


def test_only_the_bytes_of_the_units_asked_for_are_read(
    monkeypatch, tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=2, experts=8, hidden=256, inter=128, group=64
    )
    total = sum(f.stat().st_size for f in ck.path.glob("*.safetensors"))
    calls: list[int] = []
    real = os.preadv

    def counting(fd, buffers, offset):
        n = real(fd, buffers, offset)
        calls.append(n)
        return n

    monkeypatch.setattr(os, "preadv", counting)
    source = source_for(ck, kernels_src)
    assert calls == []  # headers are read with plain reads; no tensor bytes yet
    source.read((1, 5))
    raw = sum(
        ck.tensor(1, 5, proj, kind).numel() * ck.tensor(1, 5, proj, kind).element_size()
        for proj in ck.projs
        for kind in ck.kinds
    )
    assert sum(calls) == raw and len(calls) == 9
    assert sum(calls) < total / 8
    source.close()


def test_what_is_not_store_material_is_skipped_and_said(tmp_path, kernels_src):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=2, experts=4, hidden=128, inter=64, group=32
    )
    source = source_for(ck, kernels_src)
    info = source.info()
    assert info["tensors_used"] == 2 * 4 * 9
    assert info["files_ignored_not_in_index"] == ["model_extra_tensors.safetensors"]
    by_pattern = {s["pattern"]: s for s in info["skipped"]}
    mtp = by_pattern["mtp.layers.N.mlp.experts.N.gate_proj.weight"]
    assert mtp["tensors"] == 4 and "MTP" in mtp["reason"]  # one per expert
    shared = by_pattern[
        "model.language_model.layers.N.mlp.shared_expert.gate_proj.weight"
    ]
    assert shared["tensors"] == 2 and "shared expert" in shared["reason"]
    other = by_pattern["(tensors that are not experts)"]
    assert other["tensors"] >= 2 * 2 + 1  # attention and norm per layer, lm_head
    # the stray layer-2 expert in the unnamed file did not become a layer
    assert info["layers"] == 2 and info["experts_per_layer"] == 4
    assert info["unit_stored_bytes"] >= info["unit_payload_bytes"]
    assert info["store_bytes"] == info["unit_stored_bytes"] * 8
    source.close()


@pytest.mark.parametrize(
    "dims",
    [
        dict(layers=3, experts=5, hidden=128, inter=64, group=32, shards=3),
        dict(layers=1, experts=2, hidden=512, inter=192, group=64, shards=1),
        dict(
            layers=2, experts=3, hidden=128, inter=64, group=-1, shards=2
        ),  # per channel
    ],
)
def test_shape_numbers_come_from_the_checkpoint(tmp_path, kernels_src, dims):
    ck = make_checkpoint(tmp_path / "ckpt", index=False, prefix="model", **dims)
    source = source_for(ck, kernels_src)
    assert (source.layers, source.experts) == (dims["layers"], dims["experts"])
    assert (source.hidden, source.inter) == (dims["hidden"], dims["inter"])
    assert source.group_size == dims["group"]
    assert source.name_format.startswith("model.layers.")  # the second default
    unit = source.read((dims["layers"] - 1, dims["experts"] - 1))
    n13 = 2 * dims["inter"]
    groups13 = 1 if dims["group"] == -1 else dims["hidden"] // dims["group"]
    assert unit["w13"].numel() == n13 * dims["hidden"] // 2
    assert unit["w13_scales"].numel() == n13 * groups13 * 2  # bf16
    source.close()


def test_name_pattern_and_projection_names_are_overridable(tmp_path, kernels_src):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=2, experts=3, hidden=128, inter=64, group=32,
        prefix="model", projs=("w1", "w3", "w2"),
        experts_infix="block_sparse_moe.experts", config=False,
    )  # fmt: skip
    with pytest.raises(eckpt.CheckpointError, match="no routed-expert tensors"):
        source_for(ck, kernels_src)
    source = source_for(
        ck,
        kernels_src,
        name_format="model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.{kind}",
        proj_names=["w1", "w3", "w2"],
    )
    assert (source.layers, source.experts) == (2, 3)
    # gate is w1, up is w3: the same bytes as the default names give
    ref = make_checkpoint(
        tmp_path / "ref", layers=2, experts=3, hidden=128, inter=64, group=32,
        prefix="model", config=False,
    )  # fmt: skip
    reference = source_for(ref, kernels_src)
    for key in source.unit_keys():
        want, got = reference.read(key), source.read(key)
        assert all(same_bytes(got[k], want[k]) for k in want)
    source.close()
    reference.close()


def test_a_checkpoint_the_store_cannot_hold_is_refused_with_the_reason(
    tmp_path, kernels_src
):
    base = dict(layers=2, experts=3, hidden=128, inter=64, group=32)
    # a tensor missing
    ck = make_checkpoint(tmp_path / "a", **base)
    from safetensors.torch import load_file, save_file

    victim = next(ck.path.glob("model-00001-*.safetensors"))
    tensors = load_file(str(victim))
    gone = ck.name(0, 1, "up_proj", "scales")
    del tensors[gone]
    save_file(tensors, str(victim))
    with pytest.raises(eckpt.CheckpointError, match=r"lacks some of its 9 tensors"):
        source_for(ck, kernels_src)
    # config.json and the tensors disagree
    ck = make_checkpoint(tmp_path / "b", **base)
    cfg = json.loads((ck.path / "config.json").read_text())
    cfg["text_config"]["moe_intermediate_size"] = 96
    (ck.path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(eckpt.CheckpointError, match="moe_intermediate_size 96"):
        source_for(ck, kernels_src)
    # eight-bit weights
    ck = make_checkpoint(
        tmp_path / "c", quantization_config={"bits": 8, "group_size": 32}, **base
    )
    with pytest.raises(eckpt.CheckpointError, match="int4 only"):
        source_for(ck, kernels_src)
    # layers that do not start at 0 (a dense first layer)
    ck = make_checkpoint(tmp_path / "d", index=False, config=False, **base)
    victim = next(ck.path.glob("model-00001-*.safetensors"))
    tensors = load_file(str(victim))
    renamed = {re.sub(r"layers\.0\.", "layers.7.", k): v for k, v in tensors.items()}
    save_file(renamed, str(victim))
    with pytest.raises(eckpt.CheckpointError, match="every decoder layer from 0"):
        source_for(ck, kernels_src)
    # a shard cut short
    ck = make_checkpoint(tmp_path / "e", **base)
    victim = next(ck.path.glob("model-00002-*.safetensors"))
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])
    with pytest.raises(eckpt.CheckpointError, match="truncated"):
        source_for(ck, kernels_src)
    # an index that names a file that is not there
    ck = make_checkpoint(tmp_path / "f", **base)
    next(ck.path.glob("model-00002-*.safetensors")).unlink()
    with pytest.raises(eckpt.CheckpointError, match="not here"):
        source_for(ck, kernels_src)


def test_zero_points_that_are_not_symmetric_are_counted_or_refused(
    tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=1, experts=4, hidden=128, inter=64, group=32,
        qzeros_value=0x55555555,
    )  # fmt: skip
    source = source_for(ck, kernels_src)
    for key in source.unit_keys():
        source.read(key)
    qz = source.info()["qzeros"]
    assert qz["checked_units"] == 4 and qz["not_symmetric_units"] == 4
    strict = source_for(ck, kernels_src, strict_qzeros=True)
    with pytest.raises(eckpt.CheckpointError, match="zero points"):
        strict.read((0, 0))
    # symmetric, in either convention (7 stored as zp-1, or 8 stored)
    for value in (0x77777777, -0x77777778):
        ok = make_checkpoint(
            tmp_path / f"ok{value & 0xFF}", layers=1, experts=2, hidden=128, inter=64,
            group=32, qzeros_value=value,
        )  # fmt: skip
        s = source_for(ok, kernels_src, strict_qzeros=True)
        s.read((0, 1))
        assert s.info()["qzeros"]["not_symmetric_units"] == 0


def test_the_kernels_function_is_taken_from_its_source_without_importing_the_package(
    monkeypatch, tmp_path
):
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface", False)
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels", False)
    root = tmp_path / "k"
    pkg = root / "vllm_xpu_kernels"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('the package was imported')\n")
    (pkg / "fused_moe_interface.py").write_text(FAKE_KERNELS_MODULE)
    module = eckpt.load_kernels_transform(str(root))
    assert module.__file__.endswith("fused_moe_interface.py")
    assert module.implement_zp(
        torch.tensor([0x00, 0x1F, 0xF0], dtype=torch.uint8)
    ).tolist() == [0x88, 0x97, 0x78]
    # the engine imports it by name and gets the same function
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    assert implement_zp is module.implement_zp
    # a library without the function is an error, not a silent fallback
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface")
    monkeypatch.delitem(sys.modules, "vllm_xpu_kernels")
    (pkg / "fused_moe_interface.py").write_text("import torch\n")
    with pytest.raises(ImportError, match="no function implement_zp"):
        eckpt.load_kernels_transform(str(root))


# --------------------------------------------------------------------------
# The command line, and the card guard
# --------------------------------------------------------------------------


def run_tool(
    *args: str, launcher: bool = False, env=None
) -> subprocess.CompletedProcess:
    base = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    base.pop("VLLM_TARGET_DEVICE", None)  # the tool must set it itself
    base.update(env or {})
    cmd = (
        [sys.executable, card_guard.__file__]
        if launcher
        else [sys.executable, "-m", MODULE]
    )
    return subprocess.run(
        [*cmd, *args], capture_output=True, text=True, env=base, timeout=600
    )


def assert_guard_clean(stderr: str) -> None:
    """The tool's own last word on the accelerator: not initialised, nothing
    mapped, no refused call. (A probe such as transformers' ``is_available`` at
    import is answered 'no device' and listed, not an error.)"""
    line = next(x for x in stderr.splitlines() if x.startswith("card guard:"))
    assert "xpu initialised=False" in line and "device mappings=0" in line, line
    assert "(refused)" not in stderr


def test_command_line_build_in_parts_then_finalize_verify_and_info(
    tmp_path, kernels_src
):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=3, experts=8, hidden=256, inter=128, group=64
    )
    dirs = [str(tmp_path / "s0"), str(tmp_path / "s1")]
    flags = [
        "--source", "checkpoint", "--model", str(ck.path),
        "--engine-quantization", "gptq", "--kernels-src", kernels_src,
    ]  # fmt: skip
    info = run_tool("info", *flags, "--json")
    assert info.returncode == 0, info.stderr
    described = json.loads(info.stdout)
    assert described["source"]["units"] == 24
    assert_guard_clean(info.stderr)

    for part in ("0/2", "1/2"):
        out = run_tool(
            "build", "--dirs", *dirs, *flags, "--part", part, "--workers", "2",
            "--flush-every", "4", launcher=part == "1/2",
        )  # fmt: skip
        assert out.returncode == 0, out.stderr
        assert re.search(r"\d+/12 units, [\d.]+ units/s, [\d.]+ MiB/s", out.stderr), (
            out.stderr
        )
        assert_guard_clean(out.stderr)
    again = run_tool("build", "--dirs", *dirs, *flags, "--part", "0/2")
    assert (
        "12 already done" in again.stderr
        or "0 written, 12 already done" in again.stderr
    )
    fin = run_tool("finalize", "--dirs", *dirs)
    assert fin.returncode == 0, fin.stderr
    ver = run_tool("verify", "--dirs", *dirs, *flags, "--sample", "24")
    assert ver.returncode == 0, ver.stdout + ver.stderr
    report = json.loads(ver.stdout)
    assert report["ok"] and report["source_sampled"] == 24
    both = run_tool("info", "--dirs", *dirs, *flags)
    assert "MATCHES" in both.stdout, both.stdout

    # a store of another identity is refused, not replaced
    other = run_tool(
        "build", "--dirs", *dirs, *flags, "--engine-dtype", "float16", "--part", "0/2"
    )
    assert other.returncode == 2 and "error:" in other.stderr
    # and an identity the engine reported can be demanded up front
    demanded = run_tool(
        "build", "--dirs", str(tmp_path / "x0"), *flags, "--expect-identity", "sha256:0"
    )
    assert demanded.returncode == 2 and "expected sha256:0" in demanded.stderr
    assert not (tmp_path / "x0").exists()


def test_verify_against_the_checkpoint_catches_a_wrong_unit(tmp_path, kernels_src):
    ck = make_checkpoint(
        tmp_path / "ckpt", layers=2, experts=4, hidden=128, inter=64, group=32
    )
    dirs = [str(tmp_path / "s0")]
    flags = [
        "--source", "checkpoint", "--model", str(ck.path),
        "--engine-quantization", "gptq", "--kernels-src", kernels_src,
    ]  # fmt: skip
    assert run_tool("build", "--dirs", *dirs, *flags, "--finalize").returncode == 0
    # the checkpoint changes under the store: the units no longer match it
    victim = next(ck.path.glob("model-00001-*.safetensors"))
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(victim))
    name = ck.name(0, 0, "down_proj", "qweight")
    tensors[name] = tensors[name] ^ 1
    save_file(tensors, str(victim))
    ver = run_tool("verify", "--dirs", *dirs, *flags, "--sample", "8")
    assert ver.returncode in (1, 2), ver.stdout + ver.stderr


def test_card_guard_hides_enumeration_and_refuses_initialisation(monkeypatch):
    xpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        init=lambda: None,
        _lazy_init=lambda: None,
        current_device=lambda: 0,
        is_initialized=lambda: False,
    )
    monkeypatch.setattr(torch, "xpu", xpu, raising=False)
    monkeypatch.setattr(card_guard, "_installed", False)
    monkeypatch.setattr(card_guard, "_calls", [])
    assert card_guard.install()
    assert torch.xpu.is_available() is False and torch.xpu.device_count() == 0
    assert card_guard.check()["xpu_initialized"] is False  # enumeration is harmless
    with pytest.raises(card_guard.CardTouchedError, match="must not initialise"):
        torch.xpu.init()
    assert any("init (refused)" in c for c in card_guard.calls())
    with pytest.raises(card_guard.CardTouchedError):
        card_guard.check()


def test_card_guard_sees_device_mappings_and_descriptors(monkeypatch):
    monkeypatch.setattr(card_guard, "_calls", [])
    monkeypatch.setattr(
        torch, "xpu", types.SimpleNamespace(is_initialized=lambda: False), raising=False
    )
    assert card_guard.check()["device_mappings"] == []
    monkeypatch.setattr(card_guard, "_device_mappings", lambda: ["/dev/dri/renderD128"])
    with pytest.raises(card_guard.CardTouchedError, match="mapped"):
        card_guard.check()
    monkeypatch.undo()
    # an open descriptor is reported; it fails only when asked to
    monkeypatch.setattr(card_guard, "DEVICE_PREFIXES", ("/dev/null",))
    held = len(card_guard.report()["device_fds"])
    fd = os.open("/dev/null", os.O_RDONLY)
    try:
        assert len(card_guard.report()["device_fds"]) == held + 1
        card_guard.check()
        with pytest.raises(card_guard.CardTouchedError, match="open"):
            card_guard.check(strict_fds=True)
    finally:
        os.close(fd)
