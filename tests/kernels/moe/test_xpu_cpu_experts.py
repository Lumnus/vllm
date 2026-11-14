# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert slots computed on the CPU (``--cpu-offload-pinned-gb``), on CPU: the
host slots of a store-backed layer, their host cache, the calls a routing
gives them, the result against the card's arithmetic, moves and promotions,
and the refusals.

The kernels' host op (``torch.ops._xpu_C.int4_moe_cpu``) is used when it is
loaded; ``VLLM_TEST_INT4_MOE_CPU_LIB`` names a library that registers it (a
build of the kernels' op outside the kernels' wheel). Without it the op is
replaced by the torch reference of the card's arithmetic, which tests
everything around the op but not the op."""

import os
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.offloader import expert_tier_store as tstore
from vllm.model_executor.offloader.expert_tier_map import FAST, HOST, POOL

from .xpu_store_harness import GROUP, HIDDEN, INTER, expert_in_store, run

pytestmark = pytest.mark.skip_global_cleanup

_LIB = os.environ.get("VLLM_TEST_INT4_MOE_CPU_LIB")
if _LIB and not cx.host_op_available():
    torch.ops.load_library(_LIB)
REAL_OP = cx.host_op_available()

CPU_CFG = {
    "cpu_isa": "scalar",
    "cpu_threads": 2,
    "host_cache_gib": 1e-3,  # 20 experts at this size
}


def _settle(cache, timeout=5.0):
    """Wait for the reads ahead of use to land."""
    deadline = time.time() + timeout
    while cache._filling and time.time() < deadline:
        time.sleep(0.01)
    assert not cache._filling


def _reference_op(out, x, units, cu, ct, cw, inter, group, offsets, threads, isa):
    """The torch reference standing in for the host op."""
    names = cx.UNIT_TENSORS
    rows = {}
    for r in set(cu.tolist()):
        rows[r] = {}
        for i, name in enumerate(names):
            end = offsets[i + 1] if i + 1 < len(names) else None
            if end is None:
                n = x.shape[1] * (inter // group) * 2
                end = offsets[i] + n
            rows[r][name] = units[r, offsets[i] : end]
    calls = list(zip(cu.tolist(), ct.tolist(), cw.tolist()))
    out.copy_(cx.reference_int4_experts(x, rows, calls, inter, group))


@pytest.fixture
def op(monkeypatch):
    """The host op, real when loaded, else the torch reference."""
    if not REAL_OP:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    cx.reset_runtime()
    yield REAL_OP
    cx.reset_runtime()


def _host_env(monkeypatch, tmp_path, pinned=2, **extra):
    """A store built at a first start, then a start that fills from it with
    ``pinned`` pinned slots per layer (of 5 off the card): the rest are host
    slots."""
    run(monkeypatch, tmp_path, store=True)
    cfg = dict(CPU_CFG, **extra)
    return run(monkeypatch, tmp_path, store=True, pinned=pinned, store_extra=cfg)


def _units(env, layer, logicals):
    return {
        lg: {k: v.reshape(-1).view(torch.uint8) for k, v in u.items()}
        for lg in logicals
        for u in [expert_in_store(env, layer, lg)]
    }


def test_the_placement_has_host_slots_and_the_maps_skip_them(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    for tiers in env.tiers:
        assert (tiers.num_fast, tiers.num_pinned, tiers.num_host) == (3, 2, 3)
        p = tiers.placement
        assert p.tier.tolist() == [FAST] * 3 + [POOL] * 2 + [HOST] * 3
        assert tiers.fast_map[5:].tolist() == [-1, -1, -1]
        assert tiers.slow_map[5:].tolist() == [-1, -1, -1]
        # the pool rows hold the pinned slots only; spares and promotion rows
        assert tiers.slow["w13"].shape[0] == 2 + 2 + 4
        assert tiers.cpu is cx.current_runtime()
        # EPLB still sees one slot per expert: the host slots have no tensor
        weights = tiers.eplb_weights()
        assert all(len(w) == 8 and w.tier_sizes == (3, 2, 3) for w in weights)
        assert tiers.nonresident_physical() == []  # served, not missing
    tstore.reset_sessions()


def test_host_calls_are_computed_as_the_card_computes_them(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    rt = tiers.cpu
    ids = torch.tensor([[0, 5], [6, 3], [7, 5]], dtype=torch.int32)
    w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
    x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16()
    out = torch.full_like(x, 7.0)
    assert rt.compute(tiers, ids, w, x, out)
    calls = [(5, 0, 0.3), (6, 1, 0.6), (7, 2, 0.5), (5, 2, 0.5)]
    want = cx.reference_int4_experts(x, _units(env, 1, {5, 6, 7}), calls, INTER, GROUP)
    if op:
        assert cx.bf16_steps(out, want) <= 1.0
    else:
        assert torch.equal(out, want)
    s = rt.snapshot()
    assert (s["routed_card"], s["routed_pinned"], s["routed_cpu"]) == (1, 1, 4)
    assert s["cpu_calls"] == 4 and s["cache_misses"] + s["cache_hits"] >= 3
    # a routing with no host call leaves the output alone
    out2 = torch.full_like(x, 7.0)
    assert not rt.compute(tiers, ids.clamp(max=4), w, x, out2)
    assert torch.equal(out2, torch.full_like(x, 7.0))
    tstore.reset_sessions()


def test_a_routed_id_outside_the_experts_fails_the_step(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(cx.CpuExpertError, match="outside the 8 physical"):
        tiers.cpu.compute(
            tiers,
            torch.tensor([[1, 8]], dtype=torch.int32),
            torch.ones(1, 2),
            x,
            torch.empty_like(x),
        )
    tstore.reset_sessions()


def test_an_expert_the_store_cannot_give_fails_the_step(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    cache = tiers.cpu.cache
    _settle(cache)

    def broken(key, dests):
        f = Future()
        f.set_exception(OSError("EIO"))
        return f

    for key in list(cache._rows):  # empty the cache
        cache._free.append(cache._rows.pop(key))
    monkeypatch.setattr(cache.store, "read_into_async", broken)
    x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(cx.CpuExpertError, match="could not be read"):
        tiers.cpu.compute(
            tiers,
            torch.tensor([[6, 0]], dtype=torch.int32),
            torch.ones(1, 2),
            x,
            torch.empty_like(x),
        )
    assert not cache._busy  # nothing stays held
    tstore.reset_sessions()


def test_an_eplb_move_with_host_slots(monkeypatch, tmp_path, op):
    """A pinned slot and a host slot trade experts; the hot one enters the
    pinned rows. The maps, the CPU's experts and the scrub agree after."""
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[0]
    old = np.arange(8)
    new = old.copy()
    new[3], new[5] = 5, 3
    load = np.array([9, 9, 9, 1, 4, 8, 0, 0], dtype=float)  # logical 5 hot
    move = tiers.prepare(old, new, load, 1.1)
    assert tiers._eplb_pending
    tiers.commit(move)
    assert not tiers._eplb_pending
    p = tiers.placement
    assert p.tier[3] == POOL and p.tier[5] == HOST
    assert tiers.slot_logical[3] == 5 and tiers.slot_logical[5] == 3
    assert tiers.slow_map[3] == p.row[3] and tiers.slow_map[5] == -1
    for name, t in tiers.slow.items():
        want = expert_in_store(env, 0, 5)[name].reshape(-1).view(torch.uint8)
        assert torch.equal(t[p.row[3]].reshape(-1).view(torch.uint8), want)
    # the host slot now computes logical 3
    x = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(2)).bfloat16()
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
    assert summary["placement_equal"]
    tstore.reset_sessions()


def test_fill_to_puts_the_next_hottest_in_pinned_rows(monkeypatch, tmp_path, op):
    env = _host_env(monkeypatch, tmp_path)
    tiers = env.tiers[1]
    load = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    tiers.fill_to(np.arange(8), load, 1.1)
    p = tiers.placement
    assert np.flatnonzero(p.tier == FAST).tolist() == [5, 6, 7]
    assert np.flatnonzero(p.tier == POOL).tolist() == [3, 4]
    assert np.flatnonzero(p.tier == HOST).tolist() == [0, 1, 2]
    fast_np, slow_np = p.maps(tiers.slot_global, tiers.num_global)
    assert tiers.fast_map.tolist() == fast_np.tolist()
    assert tiers.slow_map.tolist() == slow_np.tolist()
    assert tiers.scrub(np.arange(8))["mismatches"] == 0
    tstore.reset_sessions()


def test_refusals(monkeypatch, tmp_path, op):
    # the store must exist: a start that would build it refuses
    with pytest.raises(ValueError, match="from an existing expert store"):
        run(monkeypatch, tmp_path, store=True, pinned=2, store_extra=CPU_CFG)
    tstore.reset_sessions()
    # without a store there is nothing to compute host slots from
    with pytest.raises(ValueError, match="add --expert-store-config"):
        run(monkeypatch, tmp_path / "n", store=False, pinned=2)
    # without the host op the CPU tier does not start
    run(monkeypatch, tmp_path / "s", store=True)
    cx.reset_runtime()
    monkeypatch.setattr(cx, "host_op_available", lambda: False)
    with pytest.raises(cx.CpuExpertError, match="int4_moe_cpu"):
        run(monkeypatch, tmp_path / "s", store=True, pinned=2, store_extra=CPU_CFG)
    tstore.reset_sessions()


def test_auto_refuses_a_host_without_avx2(monkeypatch, tmp_path, op):
    run(monkeypatch, tmp_path, store=True)
    monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
    cfg = dict(CPU_CFG, cpu_isa="auto")
    with pytest.raises(cx.CpuExpertError, match="no AVX2"):
        run(monkeypatch, tmp_path, store=True, pinned=2, store_extra=cfg)
    tstore.reset_sessions()


# -- the host cache on its own ------------------------------------------------


class _Store:
    """Reads that write the key into the unit's first bytes."""

    def __init__(self, delay=0.0):
        self.reads = []
        self.delay = delay

    def read_into_async(self, key, dests):
        f = Future()

        def go():
            if self.delay:
                time.sleep(self.delay)
            dests["w13"][:2] = torch.tensor(key, dtype=torch.uint8)
            self.reads.append(key)
            f.set_result("direct")

        threading.Thread(target=go).start()
        return f


def _cache(rows=3, delay=0.0, timeout=5.0):
    layout = cx.UnitLayout(
        256, 128, 64, {"w13": 64, "w2": 32, "w13_scales": 16, "w2_scales": 16}
    )
    return cx.HostExpertCache(_Store(delay), layout, rows, timeout)


def test_cache_hits_misses_and_lru_eviction():
    c = _cache(rows=2)
    assert c.slab.data_ptr() % 4096 == 0  # direct reads can land in it
    r = c.acquire([(0, 1), (0, 2)])
    assert c.misses == 2 and c.slab[r[0], :2].tolist() == [0, 1]
    c.release(r)
    c.acquire([(0, 1)])  # a hit; (0, 2) is now the oldest
    c.release(c.acquire([(0, 1)]))
    c.release([c.row_of((0, 1))])
    r = c.acquire([(0, 3)])  # evicts (0, 2)
    c.release(r)
    assert (0, 2) not in c and (0, 1) in c and c.evicted == 1


def test_cache_rows_of_the_running_call_are_never_evicted():
    c = _cache(rows=2)
    held = c.acquire([(0, 1), (0, 2)])
    with pytest.raises(cx.CpuExpertError, match="raise host_cache_gib"):
        c.acquire([(0, 3)])
    c.release(held)
    c.release(c.acquire([(0, 3)]))
    # repeated keys in one call share a row
    rows = c.acquire([(0, 3), (0, 3), (0, 1)])
    assert rows[0] == rows[1]
    c.release(rows)


def test_cache_prefetch_and_timeout():
    c = _cache(rows=3, delay=0.05)
    assert c.prefetch([(1, 1), (1, 2)], evict=False) == 2
    rows = c.acquire([(1, 1)])  # waits for the read in flight
    assert c.slab[rows[0], :2].tolist() == [1, 1]
    c.release(rows)
    deadline = time.time() + 2
    while (1, 2) not in c and time.time() < deadline:
        time.sleep(0.01)
    assert (1, 2) in c and c.prefetched >= 1
    slow = _cache(rows=1, delay=0.5, timeout=0.05)
    with pytest.raises(cx.CpuExpertError, match="within"):
        slow.acquire([(2, 2)])


# -- the config and the step's order -------------------------------------------


def test_the_graphs_are_split_at_the_moe_ops():
    from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
    from vllm.config.offload import OffloadConfig, UVAOffloadConfig

    def cfg(pinned, mode, breakable=False, compile_mode=None, ops=("a::attn",)):
        uva = UVAOffloadConfig(
            cpu_offload_gb=8, cpu_offload_per_expert=True,
            cpu_offload_pinned_gb=pinned,
        )  # fmt: skip
        cc = SimpleNamespace(
            cudagraph_mode=mode,
            use_breakable_cudagraph=breakable,
            mode=CompilationMode.VLLM_COMPILE if compile_mode is None else compile_mode,
            splitting_ops=list(ops),
        )
        return SimpleNamespace(
            offload_config=OffloadConfig(uva=uva), compilation_config=cc
        )

    c = cfg(2, CUDAGraphMode.FULL_AND_PIECEWISE)
    VllmConfig._split_graphs_for_cpu_experts(c)
    cc = c.compilation_config
    assert cc.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert cc.splitting_ops == [
        "a::attn",
        "vllm::moe_forward",
        "vllm::moe_forward_shared",
    ]
    # unset (or not below the host budget): nothing changes
    for pinned in (None, 8):
        c = cfg(pinned, CUDAGraphMode.FULL_AND_PIECEWISE)
        VllmConfig._split_graphs_for_cpu_experts(c)
        assert c.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        assert c.compilation_config.splitting_ops == ["a::attn"]
    c = cfg(2, CUDAGraphMode.FULL_DECODE_ONLY)
    VllmConfig._split_graphs_for_cpu_experts(c)
    assert c.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    c = cfg(2, CUDAGraphMode.FULL_AND_PIECEWISE, breakable=True)
    VllmConfig._split_graphs_for_cpu_experts(c)
    assert c.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    c = cfg(2, CUDAGraphMode.PIECEWISE, compile_mode=CompilationMode.NONE)
    VllmConfig._split_graphs_for_cpu_experts(c)
    assert c.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_the_cpu_tier_needs_the_store_and_per_expert_offload():
    from vllm.config import VllmConfig
    from vllm.config.expert_store import ExpertStoreConfig
    from vllm.config.offload import OffloadConfig, UVAOffloadConfig

    def cfg(per_expert=True, store=None):
        uva = UVAOffloadConfig(
            cpu_offload_gb=8, cpu_offload_per_expert=per_expert,
            cpu_offload_pinned_gb=2,
        )  # fmt: skip
        return SimpleNamespace(
            offload_config=OffloadConfig(uva=uva), expert_store_config=store
        )

    with pytest.raises(ValueError, match="add --expert-store-config"):
        VllmConfig._verify_cpu_experts(cfg())
    with pytest.raises(ValueError, match="--cpu-offload-per-expert"):
        VllmConfig._verify_cpu_experts(cfg(per_expert=False))
    VllmConfig._verify_cpu_experts(cfg(store=ExpertStoreConfig(dirs=["/x"])))


def test_the_flag_and_the_compile_hash():
    from vllm.config.offload import UVAOffloadConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--cpu-offload-pinned-gb", "4"])
    assert EngineArgs.from_cli_args(args).cpu_offload_pinned_gb == 4.0
    base = UVAOffloadConfig(cpu_offload_gb=8, cpu_offload_per_expert=True)
    bounded = UVAOffloadConfig(
        cpu_offload_gb=8, cpu_offload_per_expert=True, cpu_offload_pinned_gb=4
    )
    assert base.compile_factors() == bounded.compile_factors()
    assert not base.host_experts_on() and bounded.host_experts_on()


def test_the_layer_posts_before_the_card_and_adds_after():
    """XPUExperts' tiered path with host slots: the inputs go to the host
    before the card's experts are enqueued, the host result is added after
    the pinned tier's."""
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    order = []

    class Impl:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def apply(self, output, **kw):
            order.append(self.name)
            if self.name == "fast":
                output.fill_(self.value)
            else:
                output.fill_(self.value)

    class Cpu:
        def post(self, tiers, h, ids, w):
            order.append("post")
            return "pending"

        def finish(self, pending, device):
            order.append("finish")
            return torch.full((2, 4), 100.0)

    experts = xpu_moe.XPUExperts.__new__(xpu_moe.XPUExperts)
    experts.tiers = SimpleNamespace(num_global=8, cpu=Cpu())
    experts.moe_config = SimpleNamespace(ep_size=1, ep_rank=0)
    experts.impl_apply_kwargs = {}
    experts._tier_impls = lambda w1, w2, k: (Impl("fast", 1.0), Impl("slow", 10.0))
    out = torch.zeros(2, 4)
    experts._apply_tiered(
        out, torch.zeros(2, 4), None, None, torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int32),
    )  # fmt: skip
    assert order == ["post", "fast", "slow", "finish"]
    assert torch.equal(out, torch.full((2, 4), 111.0))


def test_cpu_cores_for_rank(monkeypatch):
    # unset: not pinned
    assert cx.cpu_cores_for_rank("", 0, 2, 8) == []
    # one list for every rank; a list per local rank
    assert cx.cpu_cores_for_rank("0-3,8", 1, 2, 8) == [0, 1, 2, 3, 8]
    assert cx.cpu_cores_for_rank("0-11;12-23", 1, 2, 12) == list(range(12, 24))
    with pytest.raises(cx.CpuExpertError, match="2 core sets"):
        cx.cpu_cores_for_rank("0-1;2-3", 2, 3, 2)
    # auto: physical cores of the mask (SMT siblings 24-47 of 0-23), split
    # in order between the ranks, the first cpu_threads of this rank's part
    monkeypatch.setattr(cx.os, "sched_getaffinity", lambda pid: set(range(48)))
    monkeypatch.setattr(
        cx, "_physical_cores", lambda allowed: sorted(c for c in allowed if c < 24)
    )
    assert cx.cpu_cores_for_rank("auto", 0, 2, 12) == list(range(12))
    assert cx.cpu_cores_for_rank("auto", 1, 2, 8) == list(range(12, 20))
    with pytest.raises(cx.CpuExpertError, match="physical cores"):
        cx.cpu_cores_for_rank("auto", 0, 30, 1)


def test_the_op_threads_are_pinned_when_asked(monkeypatch, tmp_path, op):
    calls = []
    monkeypatch.setattr(cx, "_op_configure", lambda c, s: calls.append((c, s)) or True)
    monkeypatch.setattr(cx, "_local_rank", lambda: (1, 2))
    # unset: the op is left as it is
    monkeypatch.delenv(cx._CORES_ENV, raising=False)
    monkeypatch.delenv(cx._SPIN_ENV, raising=False)
    env = _host_env(monkeypatch, tmp_path)
    rt = env.tiers[0].cpu
    assert calls == [] and rt.cores == [] and rt.threads == CPU_CFG["cpu_threads"]
    # a set per rank: this rank's set, threads capped at one per core
    monkeypatch.setenv(cx._CORES_ENV, "0;5")
    monkeypatch.setenv(cx._SPIN_ENV, "4000")
    rt._configure_threads()
    assert calls[-1] == ([5], 4000) and rt.cores == [5] and rt.threads == 1
    rt.set_threads(4)
    assert rt.threads == 1 and calls[-1] == ([5], 4000)
    # an op without the setting refuses rather than running unpinned
    monkeypatch.setattr(cx, "_op_configure", lambda c, s: False)
    with pytest.raises(cx.CpuExpertError, match="int4_moe_cpu_configure"):
        rt._configure_threads()
    tstore.reset_sessions()


def test_the_op_is_found_in_a_host_only_library(monkeypatch):
    class _Ns:
        def __init__(self, has):
            if has:
                self.int4_moe_cpu = object()

    fake = SimpleNamespace(_xpu_C=_Ns(False), _cpu_moe_C=_Ns(True))
    monkeypatch.setattr(cx.torch, "ops", fake)
    assert cx._ops() is fake._cpu_moe_C and cx.host_op_available()
    fake._cpu_moe_C = _Ns(False)
    assert cx._ops() is None and not cx.host_op_available()
