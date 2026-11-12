# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert tiers backed by the expert store, on CPU: the store is built from a
synthetic int4 checkpoint at the first load and fills the host tier at the
next, the host rows equal the ones the checkpoint path produces, and tier moves
read the store."""

import pathlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.expert_store import ExpertStoreConfig
from vllm.config.offload import OffloadConfig, UVAOffloadConfig
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
from vllm.model_executor.offloader import expert_tier_map as tmap
from vllm.model_executor.offloader import expert_tier_store as tstore

pytestmark = pytest.mark.skip_global_cleanup

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


class _One:
    """ExpertMapManager stand-in: one rank holds every expert in order."""

    def map_global_to_local(self, g: int) -> int:
        return g


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
    def __init__(self, monkeypatch, tmp_path, store: bool, build="auto", fast=FAST):
        self.mp = monkeypatch
        self.fast = fast
        dirs = [str(tmp_path / "d0"), str(tmp_path / "d1")]
        for d in dirs:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        self.store_cfg = ExpertStoreConfig(
            dirs=dirs if store else [], pool_gib=0.01, io_threads=4, build=build,
            direct=False, spare_rows=2,
        )  # fmt: skip
        bytes_each = xt.expert_bytes(
            _GPTQ(),
            HIDDEN,
            INTER,
            torch.bfloat16,
        )
        slow = E - fast
        offload = OffloadConfig(
            uva=UVAOffloadConfig(
                cpu_offload_gb=(LAYERS * slow * bytes_each + bytes_each // 2) / 2**30,
                cpu_offload_per_expert=True,
            )
        )
        parallel = SimpleNamespace(
            enable_eplb=True, eplb_config=SimpleNamespace(max_moves_per_rank=2)
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
        tstore.reset_sessions()
        self.tiers: list[xt.ExpertTiers] = []
        self.layers: list[torch.nn.Module] = []

    def construct(self) -> None:
        moe_config = SimpleNamespace(
            num_experts=E, num_logical_experts=E, ep_rank=0, ep_size=1
        )
        params = {
            "hidden_size": HIDDEN,
            "intermediate_size_per_partition": INTER,
            "params_dtype": torch.bfloat16,
        }
        for i in range(LAYERS):
            tiers = xt.maybe_tier(NAME.format(i), moe_config, _One(), _GPTQ(), params)
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
            for slot in range(E):
                raw = raw_gptq(checkpoint(i, slot))
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


def run(monkeypatch, tmp_path, store=True, build="auto", fast=FAST) -> Env:
    env = Env(monkeypatch, tmp_path, store, build, fast)
    env.construct()
    env.load()
    env.process()
    return env


def pool_rows(tiers: xt.ExpertTiers) -> dict[str, torch.Tensor]:
    return {k: v[: tiers.num_slow].clone() for k, v in tiers.slow.items()}


def expert_in_store(env: Env, layer: int, logical: int) -> dict[str, torch.Tensor]:
    session = env.tiers[layer].store_binding.session
    with session.store.get((layer, logical)) as entry:
        return {k: v.clone() for k, v in entry.tensors().items()}


def test_build_then_fill_give_the_host_rows_the_checkpoint_path_gives(
    monkeypatch, tmp_path
):
    # reference: no store, today's path
    ref = run(monkeypatch, tmp_path / "ref", store=False)
    ref_rows = [pool_rows(t) for t in ref.tiers]
    assert all(t.store_binding is None and t.placement is None for t in ref.tiers)
    assert all(t.slow["w13"].shape[0] == t.num_slow for t in ref.tiers)

    # first load: the store is built, the host rows come from the checkpoint
    first = run(monkeypatch, tmp_path, store=True)
    session = first.tiers[0].store_binding.session
    assert session.mode == "build" and session.is_open
    assert all(t.slow["w13"].shape[0] == t.num_slow + 2 for t in first.tiers)
    for built, want in zip(first.tiers, ref_rows):
        for key, rows in pool_rows(built).items():
            assert torch.equal(rows, want[key]), key
    # the spare rows are zero
    assert not first.tiers[0].slow["w13"][first.tiers[0].num_slow :].any()
    # every expert is in the store in kernel layout: fast ones as the kernel
    # reads them after its relayout, host ones as the repack gives them
    for i, tiers in enumerate(first.tiers):
        for slot in range(E):
            got = expert_in_store(first, i, slot)
            if slot < FAST:
                fast = tiers._impls[0]
                want = {
                    "w13": fast.w13[slot].view(torch.uint8),
                    "w2": fast.w2[slot].view(torch.uint8),
                    "w13_scales": fast.gemm1_wei_scales[slot],
                    "w2_scales": fast.gemm2_wei_scales[slot],
                }
            else:
                row = slot - FAST
                want = {k: v[row] for k, v in tiers.slow.items()}
            for key, tensor in want.items():
                assert torch.equal(
                    got[key].reshape(-1).view(torch.uint8),
                    tensor.reshape(-1).view(torch.uint8),
                ), (i, slot, key)
    first_rows = [pool_rows(t) for t in first.tiers]
    tstore.reset_sessions()

    # second load: the loader skips the host slots, the store fills them
    second = run(monkeypatch, tmp_path, store=True)
    session = second.tiers[0].store_binding.session
    assert session.mode == "fill"
    assert all(t.fills_from_store for t in second.tiers)
    assert all(t.twins == {} for t in second.tiers)  # no pageable twins
    for filled, want in zip(second.tiers, ref_rows):
        for key, rows in pool_rows(filled).items():
            assert torch.equal(rows, want[key]), key
    for filled, built in zip([pool_rows(t) for t in second.tiers], first_rows):
        for key in filled:
            assert torch.equal(filled[key], built[key])
    stats = session.store.stats()
    assert stats["reads"] >= LAYERS * (E - FAST)
    tstore.reset_sessions()


def test_a_store_of_another_checkpoint_is_rebuilt_or_refused(monkeypatch, tmp_path):
    run(monkeypatch, tmp_path, store=True)
    tstore.reset_sessions()
    # same dirs, other checkpoint identity: auto rebuilds
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "another-checkpoint"
    )
    env = run(monkeypatch, tmp_path, store=True)
    assert env.tiers[0].store_binding.session.mode == "build"
    assert env.tiers[0].store_binding.session.status == "mismatch"
    tstore.reset_sessions()
    # build=never refuses a store that does not match
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "third-checkpoint"
    )
    env = Env(monkeypatch, tmp_path, True, build="never")
    with pytest.raises(ValueError, match="build=never"):
        env.construct()
    tstore.reset_sessions()
    # build=rebuild builds even when the store matches
    monkeypatch.setattr(
        tstore, "checkpoint_identity", lambda model_config: "another-checkpoint"
    )
    env = run(monkeypatch, tmp_path, store=True, build="rebuild")
    assert env.tiers[0].store_binding.session.mode == "build"
    tstore.reset_sessions()


def test_tier_moves_read_the_store_and_keep_every_slot_mapped_once(
    monkeypatch, tmp_path
):
    env = run(monkeypatch, tmp_path, store=True)
    tiers = env.tiers[1]
    placement = tiers.placement
    assert placement is not None and placement.num_pool_rows == E - FAST + 2
    fast, slow = tiers._impls
    assert slow.num_experts == placement.num_pool_rows  # the capacity, not the slots

    def bytes_of_slot(slot):
        if placement.tier[slot] == 0:
            r = int(placement.row[slot])
            return {
                "w13": fast.w13[r].view(torch.uint8).clone(),
                "w2": fast.w2[r].view(torch.uint8).clone(),
            }
        r = int(placement.row[slot])
        return {"w13": tiers.slow["w13"][r].clone(), "w2": tiers.slow["w2"][r].clone()}

    def mapped_once():
        fm, sm = tiers.fast_map, tiers.slow_map
        for slot, g in enumerate(tiers.slot_global):
            f, p = int(fm[g]), int(sm[g])
            assert (f >= 0) != (p >= 0), slot
            tier, row = (0, f) if f >= 0 else (1, p)
            assert (int(placement.tier[slot]), int(placement.row[slot])) == (tier, row)

    mapped_once()
    expect = {
        s: {"w13": e["w13"].reshape(-1), "w2": e["w2"].reshape(-1)}
        for s in range(E)
        for e in [expert_in_store(env, 1, s)]
    }
    # swap a hot pool slot with a cold fast slot
    tiers.swap(promote=6, demote=1)
    mapped_once()
    assert placement.tier[6] == 0 and placement.tier[1] == 1
    for slot in (6, 1):
        got = bytes_of_slot(slot)
        for key in got:
            assert torch.equal(
                got[key].reshape(-1).view(torch.uint8),
                expect[slot][key].view(torch.uint8),
            ), (slot, key)
    # demote alone frees a fast row, promote uses it
    tiers.demote(0)
    assert placement.fast_free == [0]
    tiers.promote(4)
    assert placement.fast_free == [] and placement.tier[4] == 0
    with pytest.raises(RuntimeError, match="no free fast row"):
        tiers.promote(5)
    mapped_once()
    for slot in range(E):
        got = bytes_of_slot(slot)
        for key in got:
            assert torch.equal(
                got[key].reshape(-1).view(torch.uint8),
                expect[slot][key].view(torch.uint8),
            ), (slot, key)

    # a placement change: new experts in two slots, read from the store
    new = tiers.slot_logical.copy()
    new[2], new[5] = 3, 7  # experts other slots hold: replicas, fine for a test
    load = np.ones(E)
    load[5] = 100.0
    move = tiers.prepare_move(new, load, 1.1)
    assert move.stats["units"] >= 2
    tiers.commit_move(move)
    mapped_once()
    assert tiers.slot_logical.tolist() == new.tolist()
    for slot in (2, 5):
        got = bytes_of_slot(slot)
        want = expert_in_store(env, 1, int(new[slot]))
        assert torch.equal(
            got["w13"].reshape(-1).view(torch.uint8),
            want["w13"].reshape(-1).view(torch.uint8),
        )
    assert placement.tier[5] == 0  # the hot new expert is on the card
    tstore.reset_sessions()
