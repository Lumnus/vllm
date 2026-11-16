# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU mailbox all-reduce wiring (no device, no kernels).

1. The summation contract (reference_all_reduce): every rank gets the same
   bits for every partner order and partial mode; the fp32 mode equals the
   correctly rounded sum on (almost) every element; its error is no larger
   than a per-hop-rounding ring's.
2. Setup over a 4-process gloo group with stand-in kernel ops: each stage
   pairs the right ranks and swaps their fds over the abstract sockets; a
   failure on any rank (at region creation or at peer opening) turns the
   mailbox off on every rank without a hang.
3. Failure path: the watchdog reads the host words and every later call
   raises with the line naming rank, stage and partner.
4. Routing in XpuCommunicator: eligible tensors go to the mailbox, the rest
   (fp32, too large, non-contiguous, mailbox off) to XCCL.
"""

import os
import socket
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.distributed.device_communicators.xpu_mailbox_all_reduce as mar
from vllm.distributed.device_communicators.xpu_mailbox_all_reduce import (
    MailboxConfig,
    XpuMailboxAllReduce,
    mailbox_partners,
    parse_order,
    reference_all_reduce,
)

ORDERS = [(3, 1), (1, 2), (2, 1), (1, 3), (3, 2), (2, 3)]


def _inputs(world, n, seed, dtype=torch.bfloat16, cancel=False):
    g = torch.Generator().manual_seed(seed)
    xs = []
    for _ in range(world):
        t = torch.randn(n, generator=g) * torch.exp2(
            torch.randint(-10, 10, (n,), generator=g).float()
        )
        xs.append(t)
    if cancel:  # rank 1 cancels rank 0 on a third of the elements
        m = torch.arange(n) % 3 == 0
        xs[1][m] = -xs[0][m] + torch.randn(int(m.sum()), generator=g) * 1e-3
    return [x.to(dtype) for x in xs]


def test_parse_order_and_partners():
    assert parse_order("3,1") == (3, 1)
    assert parse_order(" 1, 2 ") == (1, 2)
    for bad in ("1,1", "0,1", "4,1", "1", "1,2,3"):
        with pytest.raises(ValueError):
            parse_order(bad)
    assert mailbox_partners(0, 2, (3, 1)) == [1]
    assert [mailbox_partners(r, 4, (3, 1)) for r in range(4)] == [
        [3, 1],
        [2, 0],
        [1, 3],
        [0, 2],
    ]
    with pytest.raises(ValueError):
        mailbox_partners(0, 8, (3, 1))
    # every rank meets all others through the two stages
    for order in ORDERS:
        for r in range(4):
            a, b = mailbox_partners(r, 4, order)
            assert {r, a, b, a ^ b ^ r} == {0, 1, 2, 3}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32p", [True, False])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("cancel", [False, True])
def test_contract_same_bits_on_every_rank(order, fp32p, dtype, cancel):
    for world in (2, 4):
        xs = _inputs(world, 20480, 11, dtype, cancel)
        outs = reference_all_reduce(xs, order, fp32p)
        for r in range(1, world):
            assert torch.equal(outs[r].view(torch.int16), outs[0].view(torch.int16))


def _ring_ll(xs):
    """XCCL's ring as the record describes it: the partial is rounded to the
    tensor dtype after every hop (rank order 0, 1, 2, 3 for rank 3's chunk)."""
    acc = xs[0]
    for x in xs[1:]:
        acc = (acc.float() + x.float()).to(xs[0].dtype)
    return acc


def test_contract_accuracy_against_exact_sum():
    xs = _inputs(4, 20480, 5, cancel=True)
    exact = sum(x.double() for x in xs)
    rn = exact.to(torch.bfloat16)  # correctly rounded result
    fp32 = reference_all_reduce(xs, (3, 1), True)[0]
    inp = reference_all_reduce(xs, (3, 1), False)[0]
    ring = _ring_ll(xs)
    eq = lambda a: float((a.view(torch.int16) == rn.view(torch.int16)).float().mean())  # noqa: E731
    err = lambda a: float((a.double() - exact).abs().sum())  # noqa: E731
    # fp32 tree of four bf16 values is exact unless their exponents span more
    # than 24 bits, so the one rounding is the correct one almost everywhere
    assert eq(fp32) > 0.999
    assert err(fp32) <= err(ring)
    assert err(fp32) <= err(inp)
    print(
        f"correctly rounded share: fp32 {eq(fp32):.5f} input {eq(inp):.5f} "
        f"ring {eq(ring):.5f}; abs error sum fp32 {err(fp32):.4g} "
        f"input {err(inp):.4g} ring {err(ring):.4g}"
    )


# --------------------------------------------------------------------------
# setup over a gloo group with stand-in kernel ops


class _FakeOps:
    """Stands in for torch.ops._xpu_C: a region is a memfd holding
    '<rank>:<stage>'; mono_ipc_open reads the fd it is handed and records
    whose region it is."""

    def __init__(self, rank, fail_at=None):
        self.rank, self.fail_at = rank, fail_at
        self.regions = []
        self.opened = []
        self.host_words = [0] * 8
        self.calls = []

    def mailbox_ar_slot_bytes(self, numel, fp32p):
        return ((numel + 5) // 6) * 16 * (2 if fp32p else 1)

    def mono_ipc_create(self, nbytes):
        if self.fail_at == "create":
            raise RuntimeError("zeMemAllocDevice failed (injected)")
        fd = os.memfd_create(f"mbox{len(self.regions)}")
        os.write(fd, f"{self.rank}:{len(self.regions)}".encode())
        self.regions.append((fd, nbytes))
        return len(self.regions) - 1

    def mono_ipc_export(self, h):
        return [os.getpid(), self.regions[h][0], 1, 2] + [0] * 16

    def mono_ipc_open(self, h, info, fd):
        if self.fail_at == "open":
            raise RuntimeError("zeMemOpenIpcHandle failed (injected)")
        os.lseek(fd, 0, os.SEEK_SET)
        self.opened.append((h, os.read(fd, 64).decode(), info[0]))
        return 2

    def mono_ipc_ptr(self, h, peer):
        return 4096 * (h + 1) + (1 << 20 if peer else 0)

    def mailbox_ar_host_alloc(self):
        return 1

    def mailbox_ar_host_free(self, host):
        pass

    def mailbox_ar_host_read(self, host):
        return list(self.host_words)

    def mailbox_all_reduce(self, t, inbox, outbox, state, host, slot, fp32p, tmo):
        self.calls.append((t.numel(), tuple(inbox), tuple(outbox), slot, fp32p))


def _worker(rank, world, init_file, fail_rank, fail_at, q):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    ops = _FakeOps(rank, fail_at if rank == fail_rank else None)
    mar._ops = lambda: ops
    mar.kernel_ops_present = lambda: True
    cfg = MailboxConfig(max_bytes=40960, order=(3, 1), timeout_s=5.0, on_fail="log")
    t0 = time.monotonic()
    obj = XpuMailboxAllReduce.create(
        dist.group.WORLD, torch.device("cpu"), rank, world, cfg
    )
    took = time.monotonic() - t0
    res = {"rank": rank, "on": obj is not None, "opened": ops.opened, "took": took}
    if obj is not None:
        x = torch.zeros(8, 2560, dtype=torch.bfloat16)
        res["use"] = [
            obj.should_use(x),
            obj.should_use(x.float()),
            obj.should_use(torch.zeros(9, 2560, dtype=torch.bfloat16)),
            obj.should_use(x.t()),
        ]
        obj.all_reduce_inplace(x)
        res["calls"] = ops.calls
        res["partners"] = obj.partners
        obj.close()
    q.send(res)
    dist.barrier()
    dist.destroy_process_group()


def _run(world, fail_rank=None, fail_at=None):
    # pipes, not a Queue: no POSIX semaphore (/dev/shm) needed
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "init")
        pipes = [ctx.Pipe(duplex=False) for _ in range(world)]
        procs = [
            ctx.Process(
                target=_worker, args=(r, world, f, fail_rank, fail_at, pipes[r][1])
            )
            for r in range(world)
        ]
        for p in procs:
            p.start()
        out = []
        for rd, _ in pipes:
            assert rd.poll(240), "worker did not report"
            out.append(rd.recv())
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0
    return sorted(out, key=lambda r: r["rank"])


@pytest.mark.skipif(not hasattr(socket, "send_fds"), reason="needs send_fds")
@pytest.mark.parametrize("world", [2, 4])
def test_setup_pairs_and_fd_swap(world):
    res = _run(world)
    for r in res:
        assert r["on"], r
        parts = mailbox_partners(r["rank"], world, (3, 1))
        assert r["partners"] == parts
        # stage s maps the stage-s partner's stage-s region (symmetric pairs)
        assert [o[1] for o in r["opened"]] == [f"{p}:{s}" for s, p in enumerate(parts)]
        assert r["use"] == [True, False, False, False]
        (numel, inbox, outbox, slot, fp32p) = r["calls"][0]
        assert numel == 8 * 2560 and len(inbox) == len(parts) and fp32p
        assert slot == ((20480 + 5) // 6 * 32 + 63) // 64 * 64


@pytest.mark.skipif(not hasattr(socket, "send_fds"), reason="needs send_fds")
@pytest.mark.parametrize("fail_at", ["create", "open"])
def test_setup_failure_on_one_rank_turns_it_off_everywhere(fail_at):
    res = _run(4, fail_rank=2, fail_at=fail_at)
    assert [r["on"] for r in res] == [False] * 4
    # nobody waited for a socket timeout at "create" (the first collective
    # already knew); an "open" failure on rank 2 lets its partners finish
    # their own opens, so nothing waits there either
    assert max(r["took"] for r in res) < 60


def test_failure_path_raises_named_line():
    ops = _FakeOps(0)
    mar_ops = mar._ops
    mar._ops = lambda: ops
    try:
        cfg = MailboxConfig(timeout_s=1.0, on_fail="log")
        obj = XpuMailboxAllReduce(None, torch.device("cpu"), 0, 4, cfg)
        obj.host = 1
        obj.state = torch.zeros(4, dtype=torch.int32)
        obj._start_watchdog()
        x = torch.zeros(16, dtype=torch.bfloat16)
        obj.all_reduce_inplace(x)  # healthy
        ops.host_words = [1, 1, 77, 5, 76, 0, 0, 0]  # stage 1 gave up, tag 77
        deadline = time.monotonic() + 5
        while not obj.failed and time.monotonic() < deadline:
            time.sleep(0.05)
        assert obj.failed
        assert not obj.should_use(x)
        with pytest.raises(RuntimeError) as e:
            obj.all_reduce_inplace(x)
        line = str(e.value)
        assert "TP rank 0" in line and "stage 1" in line and "TP rank 1" in line
        assert "tag 77" in line
        obj.close()
    finally:
        mar._ops = mar_ops


# --------------------------------------------------------------------------
# routing in XpuCommunicator


class _Rec:
    def __init__(self, take=True):
        self.take = take
        self.got = []

    def should_use(self, t):
        return self.take and t.dtype == torch.bfloat16 and t.is_contiguous()

    def all_reduce_inplace(self, t):
        self.got.append(t)
        return t


def test_communicator_routing(monkeypatch):
    from vllm.distributed.device_communicators import xpu_communicator as xc

    xla = []
    monkeypatch.setattr(xc.dist, "all_reduce", lambda t, group=None: xla.append(t))
    monkeypatch.setattr(xc.envs, "VLLM_BATCH_INVARIANT", False, raising=False)
    comm = object.__new__(xc.XpuCommunicator)
    comm.device_group = None
    comm.mailbox_ar = _Rec()
    a = torch.ones(4, 8, dtype=torch.bfloat16)
    assert comm.all_reduce_inplace(a) is a and comm.mailbox_ar.got == [a]
    b = torch.ones(4, 8, dtype=torch.float32)
    comm.all_reduce_inplace(b)
    assert xla == [b]
    c = comm.all_reduce(a)  # out of place: a clone goes to the mailbox
    assert c is not a and comm.mailbox_ar.got[-1] is c
    comm.mailbox_ar = None
    comm.all_reduce_inplace(a)
    assert xla[-1] is a
