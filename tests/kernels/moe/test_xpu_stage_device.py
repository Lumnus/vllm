# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device staging rows (`stage_device_rows`). A layer call that stages CPU-tier
experts copies them into device memory reserved at load, one DMA per batch of
rows on a side stream, and the card's call over them reads device memory with
the same kernel, map and bytes as the call over the pinned rows read through
UVA. On CPU: the reservation at load, the bytes and map the call reads, the
order of the DMAs and events, the runtime switch, the cost rule; on a card
(xpu only): the kernel's outputs over device rows equal those over UVA rows
bit for bit."""

import gc

import numpy as np
import pytest
import torch

from vllm.config.expert_store import ExpertStoreConfig
from vllm.model_executor.layers.fused_moe import xpu_cpu_experts as cx
from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
from vllm.model_executor.offloader import expert_tier_store as tstore

from .test_xpu_cpu_experts import _reference_op
from .test_xpu_tier_classes import _env, _pending
from .xpu_store_harness import HIDDEN, expert_in_store

pytestmark = pytest.mark.skip_global_cleanup

CPU = torch.device("cpu")


@pytest.fixture
def host_op(monkeypatch):
    real = cx.host_op_available()
    if not real:
        monkeypatch.setattr(cx, "host_op_available", lambda: True)
        monkeypatch.setattr(cx, "_op_isa", lambda want: 1)
        monkeypatch.setattr(cx, "_op_run", _reference_op)
    gc.collect()
    cx.reset_runtime()
    yield real
    cx.reset_runtime()
    tstore.reset_sessions()


# expert 6 gets 4 tokens, 5 and 7 one each (5, 6, 7 are the CPU tier's)
IDS = torch.tensor([[6, 0], [6, 1], [6, 7], [5, 6]], dtype=torch.int32)
W = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])


def _call(tiers, seed=3):
    x = torch.randn(4, HIDDEN, generator=torch.Generator().manual_seed(seed))
    return _pending(tiers, IDS, W, x.bfloat16())


class _Recorder:
    """A stand-in for the device module and its side stream: records the
    order of the waits, copies and events."""

    def __init__(self, log):
        self.log = log

    # module
    def stream(self, s):
        log = self.log

        class Ctx:
            def __enter__(self):
                log.append("enter side")

            def __exit__(self, *a):
                log.append("exit side")

        return Ctx()

    def Event(self):  # noqa: N802
        log = self.log

        class Ev:
            def record(self, stream):
                log.append(f"record on {stream}")

        return Ev()

    def current_stream(self):
        log = self.log

        class Compute:
            def __str__(self):
                return "compute"

            def wait_event(self, ev):
                log.append("compute waits")

        return Compute()

    # stream
    def wait_event(self, ev):
        self.log.append("side waits free")

    def __str__(self):
        return "side"


def test_the_rows_are_reserved_at_load_and_hold_the_staged_bytes(
    monkeypatch, tmp_path, host_op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=3, stage_device_rows=2)
    tiers = env.tiers[1]
    rt = tiers.cpu
    rows = xt._STAGE_ROWS
    # reserved at load (before the KV cache is sized), not at the first use
    assert rows is not None and rows.device_rows == 2 and rows.num_rows == 3
    assert set(rows.dev) == set(rows.host)
    for key, dev in rows.dev.items():
        assert dev.shape == (2, *rows.host[key].shape[1:])
        assert dev.dtype == rows.host[key].dtype
    pending = _call(tiers)
    call = rt.stage(pending, CPU)
    assert isinstance(call, xt.DeviceStagedCall)
    assert tiers.stage_device_impl(rows) is call  # built once
    impl = call.impl
    # the same map as the UVA call, over the device rows
    assert impl.expert_map is rows.map and impl.num_experts == rows.device_rows
    assert rows.map.tolist()[6] == 0 and sum(v >= 0 for v in rows.map.tolist()) == 1
    assert impl.w13.data_ptr() == rows.dev["w13"].data_ptr()
    assert impl.w2.data_ptr() == rows.dev["w2"].data_ptr()
    assert impl.gemm1_wei_scales is rows.dev["w13_scales"]
    assert impl.gemm2_wei_scales is rows.dev["w2_scales"]
    for name, unit in expert_in_store(env, 1, 6).items():
        want = unit.reshape(-1).view(torch.uint8)
        assert torch.equal(xt._u8(rows.dev[name])[0], want)
        assert torch.equal(rows.rows_u8(name)[0], want)  # the DMA's source
    s = rt.snapshot()
    assert s["stage_device"] == 1 and s["routed_staged"] == 4


def test_the_inputs_equal_the_uva_calls(monkeypatch, tmp_path, host_op):
    """Same kernel, same map, same bytes per mapped row: the outputs over the
    device rows are the UVA call's (the card test below runs both)."""
    env = _env(monkeypatch, tmp_path, stage_min_calls=1, stage_device_rows=3)
    tiers = env.tiers[1]
    rt = tiers.cpu
    rows = xt._STAGE_ROWS
    dev = rt.stage(_call(tiers), CPU).impl
    dev_map = rows.map.clone()
    dev_bytes = {k: xt._u8(t).clone() for k, t in rows.dev.items()}
    rt.config.stage_device_rows = 0  # cpu.set: back to the UVA read
    uva = rt.stage(_call(tiers), CPU)
    assert uva is tiers.stage_impl(rows)
    assert torch.equal(rows.map, dev_map)
    staged = int((dev_map >= 0).sum())
    assert staged == 3
    for k in dev_bytes:
        assert torch.equal(dev_bytes[k][:staged], rows.rows_u8(k)[:staged])
    assert type(dev) is type(uva) and dev.expert_map is uva.expert_map


def test_the_dmas_wait_for_the_last_read_and_the_call_waits_for_them(
    monkeypatch, tmp_path, host_op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=1, stage_device_rows=3)
    tiers = env.tiers[1]
    rt = tiers.cpu
    rows = xt._STAGE_ROWS
    log: list[str] = []
    rec = _Recorder(log)
    rows._mod, rows._stream = rec, rec
    monkeypatch.setattr(rows, "DMA_BATCH", 2)
    real_range = rt._copy_range

    def host_copy(part, lo, hi):
        log.append(f"host {lo}-{hi}")
        real_range(part, lo, hi)

    monkeypatch.setattr(rt, "_copy_range", host_copy)
    real_dma = rows.to_device

    def dma(lo, hi):
        log.append(f"dma {lo}-{hi}")
        real_dma(lo, hi)

    monkeypatch.setattr(rows, "to_device", dma)

    class Impl:
        def apply(self, **kw):
            log.append("card call")

    for _layer in range(2):
        call = rt.stage(_call(tiers), CPU)
        call.impl = Impl()  # the kernel stand-in
        call.apply(output=None)
    first = [
        "host 0-2", "dma 0-2", "enter side", "record on side", "exit side",
        "host 2-3", "dma 2-3", "enter side", "record on side", "exit side",
        "compute waits", "card call", "record on compute",
    ]  # fmt: skip
    # the first layer call has no read to wait for; the second's side stream
    # waits for the first's card call before any of its DMAs
    assert log[: len(first)] == first
    assert log[len(first) :] == ["side waits free"] + first


def test_the_switch_and_its_bound(monkeypatch, tmp_path, host_op):
    from vllm.v1.worker import runtime_control as wrc

    env = _env(monkeypatch, tmp_path, stage_min_calls=1, stage_device_rows=2)
    tiers = env.tiers[1]
    rt = tiers.cpu
    monkeypatch.setattr(wrc, "_cpu_runtime", lambda: rt)
    assert wrc._check_cpu_set(None, {"stage_device_rows": 3}) is not None  # > 2
    assert wrc._check_cpu_set(None, {"stage_device_rows": 1}) is None
    assert wrc._check_cpu_set(None, {"stage_dma_us": 0}) is not None
    changes, result, undo = wrc._apply_cpu_set(
        None, {"stage_device_rows": 1, "stage_dma_us": 120.0}
    )
    assert changes["stage_device_rows"] == [2, 1]
    assert result["cpu"]["stage_device_rows_reserved"] == 2
    # one row in use: the threshold rule stages the busiest one only
    pending = _call(tiers)
    call = rt.stage(pending, CPU)
    assert isinstance(call, xt.DeviceStagedCall) and pending.staged.tolist() == [6]
    rt.config.stage_device_rows = 0
    assert rt.stage(_call(tiers), CPU) is tiers.stage_impl(xt._STAGE_ROWS)
    wrc._restore_cpu_set(None, undo)
    assert rt.config.stage_device_rows == 2


def test_unset_nothing_is_reserved_and_the_call_is_the_uva_one(
    monkeypatch, tmp_path, host_op
):
    env = _env(monkeypatch, tmp_path, stage_min_calls=3)
    tiers = env.tiers[1]
    assert xt._STAGE_ROWS is None  # allocated at the first use, as before
    impl = tiers.cpu.stage(_call(tiers), CPU)
    rows = xt._STAGE_ROWS
    assert rows.device_rows == 0 and rows.dev == {}
    assert impl is tiers.stage_impl(rows) and tiers.cpu.snapshot()["stage_device"] == 0
    # the switch cannot reach rows that were not reserved
    from vllm.v1.worker import runtime_control as wrc

    monkeypatch.setattr(wrc, "_cpu_runtime", lambda: tiers.cpu)
    assert "reserved at load" in wrc._check_cpu_set(None, {"stage_device_rows": 1})


class _Fixed:
    copy = 50.0

    def cpu_us(self, calls):
        return 100.0 + 30.0 * np.asarray(calls, dtype=float)

    def copy_us(self, unit_bytes, threads):
        return self.copy


def _simulated(calls, k, copy_us, dma_us, batch):
    """The layer's host-side time with device rows, by events: the host copies
    batch after batch; a batch's DMA starts when it is copied and the previous
    DMA is done; the CPU starts on the rest after the last copy."""
    order = np.argsort(-calls, kind="stable")
    cpu = 100.0 + 30.0 * calls[order].astype(float)
    host = dma = 0.0
    for lo in range(0, k, batch):
        n = min(batch, k - lo)
        host += n * copy_us
        dma = max(dma, host) + n * dma_us
    return max(host + cpu[k:].sum(), dma)


def test_the_rule_with_device_rows():
    rt = cx.CpuExpertRuntime.__new__(cx.CpuExpertRuntime)
    rt.config = ExpertStoreConfig(dirs=["/x"], stage_card_us=1e5, stage_dma_us=140.0)
    rt.layout = type("L", (), {"row_bytes": 1})()
    rt.cost = _Fixed()
    rng = np.random.default_rng(1)
    for copy, dma in ((50.0, 140.0), (200.0, 140.0), (140.0, 140.0)):
        rt.cost.copy = copy
        rt.config.stage_dma_us = dma
        for _ in range(20):
            calls = rng.integers(1, 60, size=int(rng.integers(1, 60)))
            experts = np.arange(calls.size) + 100
            for batch in (1, 4, 16):
                staged, plan = rt._staged_by_cost(experts, calls, 64, dma_batch=batch)
                times = [
                    _simulated(calls, k, copy, dma, batch)
                    for k in range(min(calls.size, 64) + 1)
                ]
                assert staged.size == int(np.argmin(times))
                want = f"host {min(times) / 1e3:.2f} ms"
                assert plan.startswith(f"{staged.size} staged: {want}")
    rt.cost.copy, rt.config.stage_dma_us = 50.0, 140.0
    # the UVA rule at a high card cost per expert stages nothing; device rows,
    # whose read costs the card what its own experts do, stage the busy ones
    experts = np.array([10, 11, 12, 13])
    calls = np.array([1, 40, 1, 20])
    assert rt._staged_by_cost(experts, calls, 8)[0].size == 0
    staged, _ = rt._staged_by_cost(experts, calls, 8, dma_batch=16)
    assert staged.tolist() == [11, 13]
    # bounded by the rows in use
    assert rt._staged_by_cost(experts, calls, 1, dma_batch=16)[0].tolist() == [11]


def test_unset_the_compile_keys_are_equal():
    base = ExpertStoreConfig(dirs=["/x"])
    on = ExpertStoreConfig(dirs=["/x"], stage_device_rows=136, stage_dma_us=120.0)
    assert base.compile_factors() == on.compile_factors()


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="needs a card"
)
def test_on_a_card_the_outputs_equal_the_uva_calls():
    """The kernel over device rows against the same kernel over pinned rows
    read through UVA: equal outputs, bit for bit, at prefill shapes."""
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dev = torch.device("xpu")
    hidden, inter, group, rows, experts, topk = 2560, 640, 128, 24, 512, 10
    g = torch.Generator().manual_seed(0)
    host = {
        "w13": torch.randint(0, 256, (rows, 2 * inter, hidden // 2), generator=g),
        "w2": torch.randint(0, 256, (rows, hidden, inter // 2), generator=g),
        "w13_scales": torch.rand(rows, hidden // group, 2 * inter, generator=g),
        "w2_scales": torch.rand(rows, inter // group, hidden, generator=g),
    }
    host = {
        k: (v.to(torch.uint8) if "scales" not in k else (v * 0.02).bfloat16())
        for k, v in host.items()
    }
    host = {k: v.pin_memory() for k, v in host.items()}
    expert_map = torch.full((experts,), -1, dtype=torch.int32)
    staged = torch.randperm(experts, generator=g)[:rows]
    expert_map[staged] = torch.arange(rows, dtype=torch.int32)
    expert_map = expert_map.to(dev)
    common = dict(n_experts_per_token=topk, activation="silu", ep_rank=0, ep_size=2)

    def build(tensors):
        w13, w2 = tensors["w13"], tensors["w2"]
        w13.xpu_fused_moe = True
        impl = XpuFusedMoe(
            w13=w13,
            w13_scales=tensors["w13_scales"],
            w13_bias=None,
            w2=w2,
            w2_scales=tensors["w2_scales"],
            w2_bias=None,
            num_experts=rows,
            expert_map=expert_map,
            **common,
        )
        impl.w13, impl.w2 = w13.view(torch.int8), w2.view(torch.int8)
        return impl

    uva = build({k: get_accelerator_view_from_cpu_tensor(v) for k, v in host.items()})
    on_dev = build({k: v.to(dev) for k, v in host.items()})
    for n in (1, 64, 2048):
        x = (torch.randn(n, hidden, generator=g) * 0.5).bfloat16().to(dev)
        ids = torch.randint(0, experts, (n, topk), generator=g, dtype=torch.int32)
        ids[:, 0] = staged[torch.randint(0, rows, (n,), generator=g)].int()
        ids, w = ids.to(dev), torch.rand(n, topk, generator=g).to(dev)
        outs = []
        for impl in (uva, on_dev):
            out = torch.zeros(n, hidden, dtype=torch.bfloat16, device=dev)
            impl.apply(output=out, hidden_states=x, topk_weights=w, topk_ids=ids)
            outs.append(out)
        torch.xpu.synchronize()
        assert torch.equal(outs[0], outs[1]), n
