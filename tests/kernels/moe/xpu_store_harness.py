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
    ):
        self.mp = monkeypatch
        self.fast = fast
        self.rank, self.ep_size = rank, ep_size
        self.local = E // ep_size
        self.spare = spare
        dirs = [str(tmp_path / "d0"), str(tmp_path / "d1")]
        for d in dirs:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        self.store_cfg = ExpertStoreConfig(
            dirs=dirs if store else [], pool_gib=0.01, io_threads=4, build=build,
            direct=False, spare_rows=spare,
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
                quantization="gptq",
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
                _GPTQ(),
                params,
            )
            assert tiers is not None and tiers.num_fast == self.fast
            self.tiers.append(tiers)

    def load(self) -> None:
        """Create the fast params and twins, and run the loader's routing for
        every slot of every layer, as ``weight_loader`` does."""
        f = self.fast
        for i, tiers in enumerate(self.tiers):
            layer = torch.nn.Module()
            bf = torch.bfloat16
            shapes = {
                "w13_qweight": ((f, HIDDEN // 8, N13), torch.int32),
                "w2_qweight": ((f, INTER // 8, HIDDEN), torch.int32),
                "w13_scales": ((f, HIDDEN // GROUP, N13), bf),
                "w2_scales": ((f, INTER // GROUP, HIDDEN), bf),
            }
            for name, (shape, dtype) in shapes.items():
                layer.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(shape, dtype=dtype), requires_grad=False
                    ),
                )
            tiers.attach_twins(layer)
            for slot in range(self.local):
                raw = raw_gptq(checkpoint(i, self.rank * self.local + slot))
                for name, value in raw.items():
                    param = getattr(layer, name)
                    target, row = tiers.route(param, slot)
                    if target is None:
                        continue
                    target.data[row] = value
            self.layers.append(layer)

    def process(self) -> None:
        """The method's post-load step, ``finalize`` and the experts' impls."""
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
