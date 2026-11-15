# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A CPU harness for expert tiers backed by the expert store: a synthetic int4
checkpoint, the loader's routing, the post-load step, ``finalize`` and the
experts' impls, with the kernel library and the accelerator faked."""

import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.expert_store import ExpertStoreConfig
from vllm.config.offload import OffloadConfig, UVAOffloadConfig
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
from vllm.model_executor.offloader import expert_tier_map as tmap
from vllm.model_executor.offloader import expert_tier_store as tstore

LAYERS, E, FAST = 2, 8, 3
HIDDEN, INTER, GROUP = 256, 128, 64
N13 = 2 * INTER  # w13 output rows
NAME = "model.language_model.layers.{}.mlp.experts"


class _GPTQ(AutoGPTQMoEMethod):
    def __init__(self):
        self.quant_config = SimpleNamespace(
            weight_bits=4, pack_factor=8, group_size=GROUP
        )
        self.moe = SimpleNamespace(w13_num_shards=2)


def ct_method(
    bits: int = 4,
    symmetric: bool = True,
    strategy: str = "group",
    group_size: int = GROUP,
    backend=None,
    has_bias: bool = False,
):
    """The compressed-tensors WNA16 MoE method's create_weights and the
    fields the tiers read, without its backend selection (XPU unless
    ``backend``)."""
    from fractions import Fraction

    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    class _CT(CompressedTensorsWNA16MoEMethod):
        def __init__(self):
            self.weight_quant = SimpleNamespace(num_bits=bits, symmetric=symmetric)
            self.num_bits, self.symmetric = bits, symmetric
            self.strategy, self.group_size = strategy, group_size
            self.packed_factor = Fraction(32, bits)
            self.wna16_backend = WNA16MoEBackend.XPU if backend is None else backend
            self.moe = SimpleNamespace(
                w13_num_shards=2, has_bias=has_bias, is_act_and_mul=True
            )

    return _CT()


class _Linear:
    """ExpertMapManager stand-in: linear placement, E // ep_size slots per rank."""

    def __init__(self, rank: int, ep_size: int):
        self.rank, self.per = rank, E // ep_size

    def map_global_to_local(self, g: int) -> int:
        lo = self.rank * self.per
        return g - lo if lo <= g < lo + self.per else -1


class _Arena:
    """Host memory for the pinned arena, without pinning."""

    total = 0
    gib = 0.0

    def empty(self, shape, dtype):
        return torch.zeros(tuple(shape), dtype=dtype)


class _ImmediateEvent:
    done = True


class ImmediateOps:
    """Stream operations that complete where they are issued."""

    def __init__(self, device=None):
        self.copies = 0

    @classmethod
    def shared(cls, device):
        return cls(device)

    def record_compute(self):
        return _ImmediateEvent()

    record_side = record_compute

    def side_wait(self, event):
        pass

    compute_wait = side_wait

    def side_copy(self, dst, src):
        self.copies += 1
        dst.copy_(src)

    def compute_write(self, dst, values):
        dst.copy_(torch.from_numpy(np.ascontiguousarray(values, dtype=np.int32)))

    def done(self, event):
        return True


class FakeXpuFusedMoe:
    """Rebinds the fast weights like ``XpuFusedMoe.__init__`` does for int4
    (``implement_zp`` is an xor here) and keeps the tensors it reads."""

    def __init__(self, w13, w13_scales, w13_bias, w2, w2_scales, w2_bias, **kw):
        if not hasattr(w13, "xpu_fused_moe"):
            for w in (w13, w2):
                w.data = (w.data ^ 0x88).view(torch.int8).clone()
            w13.xpu_fused_moe = True
        self.w13, self.w2 = w13, w2
        self.gemm1_wei_scales, self.gemm2_wei_scales = w13_scales, w2_scales
        self.expert_map = kw["expert_map"]
        self.num_experts = kw["num_experts"]


def implement_zp(packed: torch.Tensor) -> torch.Tensor:
    return packed ^ 0x88


def checkpoint(layer: int, slot: int) -> dict[str, torch.Tensor]:
    """The kernel-layout bytes of one expert as the checkpoint gives them
    before ``implement_zp``: w13/w2 as uint8 [N, K/2], scales as bf16."""
    g = torch.Generator().manual_seed(layer * 1009 + slot * 13 + 5)
    return {
        "w13": torch.randint(
            0, 256, (N13, HIDDEN // 2), generator=g, dtype=torch.uint8
        ),
        "w2": torch.randint(
            0, 256, (HIDDEN, INTER // 2), generator=g, dtype=torch.uint8
        ),
        "w13_scales": torch.randn((N13, HIDDEN // GROUP), generator=g).bfloat16(),
        "w2_scales": torch.randn((HIDDEN, INTER // GROUP), generator=g).bfloat16(),
    }


def raw_gptq(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The checkpoint tensors in the GPTQ form the loader writes into a twin:
    int32 [in/8, out] qweight, [groups, out] scales."""
    return {
        "w13_qweight": t["w13"].contiguous().view(torch.int32).transpose(0, 1),
        "w2_qweight": t["w2"].contiguous().view(torch.int32).transpose(0, 1),
        "w13_scales": t["w13_scales"].transpose(0, 1),
        "w2_scales": t["w2_scales"].transpose(0, 1),
    }


def raw_ct(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same expert as compressed-tensors ``pack-quantized`` holds it:
    int32 [out, in/8] ``weight_packed`` (8 values along ``in`` per word, low
    nibble first), [out, groups] ``weight_scale``, and ``weight_shape``."""
    return {
        "w13_weight_packed": t["w13"].contiguous().view(torch.int32),
        "w2_weight_packed": t["w2"].contiguous().view(torch.int32),
        "w13_weight_scale": t["w13_scales"],
        "w2_weight_scale": t["w2_scales"],
        "w13_weight_shape": torch.tensor([N13, HIDDEN], dtype=torch.float32),
        "w2_weight_shape": torch.tensor([HIDDEN, INTER], dtype=torch.float32),
    }


# Per format: the method, the fast params the loader writes (shape, dtype), the
# checkpoint form, and the scales' parameter name after the post-load step.
def _fmt_params(fmt: str, f: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    bf = torch.bfloat16
    if fmt == "ct":
        return {
            "w13_weight_packed": ((f, N13, HIDDEN // 8), torch.int32),
            "w2_weight_packed": ((f, HIDDEN, INTER // 8), torch.int32),
            "w13_weight_scale": ((f, N13, HIDDEN // GROUP), bf),
            "w2_weight_scale": ((f, HIDDEN, INTER // GROUP), bf),
            "w13_weight_shape": ((f, 2), torch.float32),
            "w2_weight_shape": ((f, 2), torch.float32),
        }
    return {
        "w13_qweight": ((f, HIDDEN // 8, N13), torch.int32),
        "w2_qweight": ((f, INTER // 8, HIDDEN), torch.int32),
        "w13_scales": ((f, HIDDEN // GROUP, N13), bf),
        "w2_scales": ((f, INTER // GROUP, HIDDEN), bf),
    }


def scales_names(fmt: str) -> tuple[str, str]:
    if fmt == "ct":
        return "w13_weight_scale", "w2_weight_scale"
    return "w13_scales", "w2_scales"


class Env:
    def __init__(
        self,
        monkeypatch,
        tmp_path,
        store: bool,
        build="auto",
        fast=FAST,
        spare=2,
        rank=0,
        ep_size=1,
        reset=True,
        pinned=None,
        store_extra=None,
        w4a8=False,
        fmt="gptq",
    ):
        self.mp = monkeypatch
        # the checkpoint format of the experts: "gptq" or "ct"
        # (compressed-tensors pack-quantized)
        self.fmt = fmt
        # the experts ask for W4A8 (int8 activations), as XPUExpertsW4A8Int
        self.w4a8 = w4a8
        self.fast = fast
        self.rank, self.ep_size = rank, ep_size
        self.local = E // ep_size
        self.spare = spare
        dirs = [str(tmp_path / "d0"), str(tmp_path / "d1")]
        for d in dirs:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        self.store_cfg = ExpertStoreConfig(
            dirs=dirs if store else [], pool_gib=0.01, io_threads=4, build=build,
            direct=False, spare_rows=spare, **(store_extra or {}),
        )  # fmt: skip
        bytes_each = xt.expert_bytes(
            _GPTQ(),
            HIDDEN,
            INTER,
            torch.bfloat16,
        )
        slow = self.local - fast
        offload = OffloadConfig(
            uva=UVAOffloadConfig(
                cpu_offload_gb=(LAYERS * slow * bytes_each + bytes_each // 2) / 2**30,
                cpu_offload_per_expert=True,
                # pinned slots per layer below the slots off the card: the
                # rest are computed on the CPU
                cpu_offload_pinned_gb=None
                if pinned is None
                else (LAYERS * pinned * bytes_each + bytes_each // 2) / 2**30,
            )
        )
        parallel = SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(max_moves_per_rank=min(2, spare)),
        )
        vllm_config = SimpleNamespace(
            offload_config=offload,
            parallel_config=parallel,
            expert_store_config=self.store_cfg if store else None,
            model_config=SimpleNamespace(
                model="synthetic-ckpt",
                revision=None,
                quantization="compressed-tensors" if fmt == "ct" else "gptq",
                dtype=torch.bfloat16,
                hf_text_config=SimpleNamespace(num_hidden_layers=LAYERS),
            ),
        )
        monkeypatch.setattr(
            xt, "current_platform", SimpleNamespace(is_xpu=lambda: True)
        )
        monkeypatch.setattr(
            "vllm.utils.platform_utils.is_uva_available", lambda: True, raising=True
        )
        monkeypatch.delenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", raising=False)
        monkeypatch.setattr(
            "vllm.config.get_current_vllm_config_or_none", lambda: vllm_config
        )
        monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: vllm_config)
        monkeypatch.setattr(xt, "_ARENA", _Arena())
        monkeypatch.setattr(xt, "_xpu_fused_moe", lambda: FakeXpuFusedMoe)
        monkeypatch.setattr(
            "vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor",
            lambda t: t,
        )
        monkeypatch.setattr(tmap, "TorchStreamOps", ImmediateOps)
        fake = types.ModuleType("vllm_xpu_kernels.fused_moe_interface")
        fake.implement_zp = implement_zp  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", types.ModuleType("x"))
        monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.fused_moe_interface", fake)
        monkeypatch.setattr(
            "vllm.utils.platform_utils.is_pin_memory_available", lambda: False
        )
        if reset:
            tstore.reset_sessions()
        self.tiers: list[xt.ExpertTiers] = []
        self.layers: list[torch.nn.Module] = []

    def construct(self) -> None:
        moe_config = SimpleNamespace(
            num_experts=E,
            num_logical_experts=E,
            ep_rank=self.rank,
            ep_size=self.ep_size,
        )
        params = {
            "hidden_size": HIDDEN,
            "intermediate_size_per_partition": INTER,
            "params_dtype": torch.bfloat16,
        }
        for i in range(LAYERS):
            tiers = xt.maybe_tier(
                NAME.format(i),
                moe_config,
                _Linear(self.rank, self.ep_size),
                ct_method() if self.fmt == "ct" else _GPTQ(),
                params,
            )
            assert tiers is not None and tiers.num_fast == self.fast
            self.tiers.append(tiers)

    def load(self) -> None:
        """Create the fast params and twins, and run the loader's routing for
        every slot of every layer, as ``weight_loader`` does."""
        f = self.fast
        raw_form = raw_ct if self.fmt == "ct" else raw_gptq
        for i, tiers in enumerate(self.tiers):
            layer = torch.nn.Module()
            shapes = _fmt_params(self.fmt, f)
            for name, (shape, dtype) in shapes.items():
                layer.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(shape, dtype=dtype), requires_grad=False
                    ),
                )
            tiers.attach_twins(layer)
            for slot in range(self.local):
                raw = raw_form(checkpoint(i, self.rank * self.local + slot))
                for name, value in raw.items():
                    param = getattr(layer, name)
                    target, row = tiers.route(param, slot)
                    if target is None:
                        continue
                    target.data[row] = value
            self.layers.append(layer)

    def process(self) -> None:
        """The method's post-load step, ``finalize`` and the experts' impls."""
        if self.fmt == "ct":
            _env_process_ct(self)
            return
        for i, tiers in enumerate(self.tiers):
            layer = self.layers[i]
            # fast params in kernel layout (N-first uint8 / transposed scales)
            layer.w13_weight = torch.nn.Parameter(
                layer.w13_qweight.data.transpose(1, 2).contiguous().view(torch.uint8),
                requires_grad=False,
            )
            layer.w2_weight = torch.nn.Parameter(
                layer.w2_qweight.data.transpose(1, 2).contiguous().view(torch.uint8),
                requires_grad=False,
            )
            for key in ("w13", "w2"):
                setattr(
                    layer,
                    f"{key}_scales",
                    torch.nn.Parameter(
                        getattr(layer, f"{key}_scales")
                        .data.transpose(1, 2)
                        .contiguous(),
                        requires_grad=False,
                    ),
                )
            experts = SimpleNamespace(
                tiers=None, impl_apply_kwargs={"w4a8": True} if self.w4a8 else {}
            )
            tiers.finalize(layer, SimpleNamespace(fused_experts=experts))
        for i, tiers in enumerate(self.tiers):
            layer = self.layers[i]
            tiers.impls(
                layer.w13_weight, layer.w13_scales, layer.w2_weight, layer.w2_scales,
                2, "silu", None,
            )  # fmt: skip
            if tiers.ops is not None:
                tiers.ops = ImmediateOps()


def _process_ct_layer(layer: torch.nn.Module) -> None:
    """The compressed-tensors method's post-load step on XPU: the oracle's own
    ``_process_weights_xpu`` (N-first words viewed as bytes, scales as they
    are), the results under the method's names and the kernel aliases."""
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        _process_weights_xpu,
    )

    w13, w2, s13, s2, _, _ = _process_weights_xpu(
        layer,
        None,
        layer.w13_weight_packed.data,
        layer.w2_weight_packed.data,
        layer.w13_weight_scale.data,
        layer.w2_weight_scale.data,
    )
    for name, value in (
        ("w13_weight_packed", w13),
        ("w2_weight_packed", w2),
        ("w13_weight_scale", s13),
        ("w2_weight_scale", s2),
    ):
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
    layer.w13_weight = layer.w13_weight_packed
    layer.w2_weight = layer.w2_weight_packed


def _env_process_ct(self: "Env") -> None:
    """``Env.process`` for compressed-tensors experts."""
    for i, tiers in enumerate(self.tiers):
        layer = self.layers[i]
        _process_ct_layer(layer)
        experts = SimpleNamespace(
            tiers=None, impl_apply_kwargs={"w4a8": True} if self.w4a8 else {}
        )
        tiers.finalize(layer, SimpleNamespace(fused_experts=experts))
    for i, tiers in enumerate(self.tiers):
        layer = self.layers[i]
        tiers.impls(
            layer.w13_weight, layer.w13_weight_scale, layer.w2_weight,
            layer.w2_weight_scale, 2, "silu", None,
        )  # fmt: skip
        if tiers.ops is not None:
            tiers.ops = ImmediateOps()


def run(monkeypatch, tmp_path, store=True, build="auto", fast=FAST, **kw) -> Env:
    env = Env(monkeypatch, tmp_path, store, build, fast, **kw)
    env.construct()
    env.load()
    env.process()
    return env


def pool_rows(tiers: xt.ExpertTiers) -> dict[str, torch.Tensor]:
    return {k: v[: tiers.num_slow].clone() for k, v in tiers.slow.items()}


def expert_in_store(env: Env, layer: int, logical: int) -> dict[str, torch.Tensor]:
    session = env.tiers[layer].store_binding.session  # type: ignore[union-attr]
    with session.store.get((layer, logical)) as entry:
        return {k: v.clone() for k, v in entry.tensors().items()}
