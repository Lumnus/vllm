# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async EPLB on XPU must not run device P2P on its worker stream.

The async worker moves expert weights on its own stream while the forward runs
its collectives on the main stream over another communicator. Device
communicators (NCCL, oneCCL/XCCL) do not order operations of two communicators
on one device: when ranks enqueue the forward collective and the EPLB P2P in a
different order, each rank's device waits in one of them for peers that wait
in the other. On 4 x Arc Pro B70 this hung the first async rearrangement: three
ranks waited in EPLB's per-step CPU all-reduce while the fourth waited for its
device.

The fake multi-rank test below runs the EP ranks as threads, each with a
simulated device that executes the work of all its streams in submission order
and holds a collective until every rank's device reaches it, and drives the
real EPLB communicators chosen by the real `ParallelConfig` selection.
"""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.eplb.eplb_communicator as comm_mod
from vllm.config import ParallelConfig
from vllm.config.parallel import EPLBConfig
from vllm.platforms import current_platform

RANKS = 4
STALL = 3.0  # seconds without progress that count as a hang


def _auto_communicator(monkeypatch, *, use_async: bool, xpu: bool) -> str:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr("vllm.distributed.nixl_utils.is_nixl_available", lambda: False)
    config = ParallelConfig(
        tensor_parallel_size=RANKS,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(use_async=use_async),
    )
    return config.eplb_config.communicator


@pytest.mark.parametrize(
    "use_async,xpu,expected",
    [
        (False, True, "torch_xccl"),
        (True, True, "torch_gloo"),
    ],
)
def test_auto_selection(monkeypatch, use_async, xpu, expected):
    assert _auto_communicator(monkeypatch, use_async=use_async, xpu=xpu) == expected


@pytest.mark.parametrize("communicator", ["torch_xccl", "torch_nccl", "pynccl"])
def test_device_p2p_communicators_are_refused_for_async(communicator):
    with pytest.raises(ValueError, match="multi-stream"):
        EPLBConfig(use_async=True, communicator=communicator)
    EPLBConfig(use_async=False, communicator=communicator)


# Fake multi-rank: threads as ranks, a simulated in-order device per rank.


class _Rendezvous:
    """Collectives of all ranks' devices, matched by key."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.arrived: dict[object, set[int]] = {}

    def wait(self, key: object, rank: int) -> bool:
        with self.cond:
            self.arrived.setdefault(key, set()).add(rank)
            self.cond.notify_all()
            return self.cond.wait_for(
                lambda: len(self.arrived[key]) == RANKS, timeout=STALL
            )


class _Device:
    """One rank's device: every stream's work runs on one engine in the order
    it was enqueued; a collective holds the engine until all ranks reach it."""

    def __init__(self, rank: int, rendezvous: _Rendezvous) -> None:
        self.rank = rank
        self.rendezvous = rendezvous
        self.cond = threading.Condition()
        self.queue: list[tuple[str, object]] = []
        self.done = 0
        self.hung = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, stream: str, key: object = None) -> int:
        with self.cond:
            self.queue.append((stream, key))
            self.cond.notify_all()
            return len(self.queue)

    def _run(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.done < len(self.queue))
                _, key = self.queue[self.done]
            if key is not None and not self.rendezvous.wait(key, self.rank):
                self.hung = True
                return
            with self.cond:
                self.done += 1
                self.cond.notify_all()

    def synchronize(self, stream: str | None = None) -> bool:
        """Wait for the device (or one stream); False on a hang."""
        with self.cond:
            if stream is None:
                upto = len(self.queue)
            else:
                ids = [i for i, (s, _) in enumerate(self.queue) if s == stream]
                upto = ids[-1] + 1 if ids else 0
            return self.cond.wait_for(lambda: self.done >= upto, timeout=STALL)


class _Stream:
    def __init__(self, device: _Device, name: str) -> None:
        self.device, self.name = device, name

    def synchronize(self) -> None:
        assert self.device.synchronize(self.name), "stream sync hung"


_local = threading.local()


@contextmanager
def _device_stream(stream):
    prev = getattr(_local, "stream", None)
    _local.stream = stream if stream is not None else prev
    try:
        yield
    finally:
        _local.stream = prev


class _P2POp:
    def __init__(self, op, tensor, peer=None, group=None, group_peer=None):
        self.op, self.tensor = op, tensor
        self.group = group
        self.peer = group_peer if group_peer is not None else peer


class _HostMailbox:
    """gloo P2P on the host: a send is delivered to the matching receive."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.box: dict[tuple[int, int, int], torch.Tensor] = {}

    def send(self, src: int, dst: int, seq: int, t: torch.Tensor) -> None:
        with self.cond:
            self.box[(src, dst, seq)] = t.clone()
            self.cond.notify_all()

    def recv(self, src: int, dst: int, seq: int, t: torch.Tensor) -> None:
        with self.cond:
            assert self.cond.wait_for(lambda: (src, dst, seq) in self.box, STALL)
            t.copy_(self.box.pop((src, dst, seq)))


def _fake_batch_isend_irecv(mailbox: _HostMailbox, devices: list[_Device]):
    def run(ops):
        rank = _local.rank
        if any(op.group.kind == "device" for op in ops):
            # Device P2P: one collective-like operation on the current stream.
            stream = _local.stream or _Stream(devices[rank], "main")
            devices[rank].enqueue(stream.name, key=("eplb-p2p", _local.layer))
            _local.posted.set()
        else:
            for i, op in enumerate(ops):
                if op.op is torch.distributed.isend:
                    mailbox.send(rank, op.peer, i, op.tensor)
            _local.posted.set()
            for i, op in enumerate(ops):
                if op.op is torch.distributed.irecv:
                    # The peer posted its ops in the same order as this rank.
                    mailbox.recv(op.peer, rank, i, op.tensor)
        return [SimpleNamespace(wait=lambda: None) for _ in ops]

    return run


def _run_ranks(monkeypatch, communicator: str) -> dict[int, str]:
    rendezvous = _Rendezvous()
    devices = [_Device(r, rendezvous) for r in range(RANKS)]
    mailbox = _HostMailbox()
    monkeypatch.setattr(comm_mod, "P2POp", _P2POp)
    monkeypatch.setattr(comm_mod, "device_stream", _device_stream)
    monkeypatch.setattr(
        comm_mod, "batch_isend_irecv", _fake_batch_isend_irecv(mailbox, devices)
    )
    monkeypatch.setattr(comm_mod, "PIN_MEMORY", False, raising=False)
    monkeypatch.setattr(comm_mod, "is_local_first_rank", lambda: False)
    host_step = threading.Barrier(RANKS, timeout=STALL)  # EPLB's CPU all-reduce
    where: dict[int, str] = {}
    worker_may_go = [threading.Event() for _ in range(RANKS)]
    worker_posted = [threading.Event() for _ in range(RANKS)]
    weights = [torch.full((2, 4), float(r)) for r in range(RANKS)]
    # The classes `create_eplb_communicator` builds for these names on XPU.
    make = {
        "torch_xccl": lambda: comm_mod.TorchDistXCCLStagedEplbCommunicator(
            ep_group=SimpleNamespace(kind="device")
        ),
        "torch_gloo": lambda: comm_mod.TorchDistGlooStagedEplbCommunicator(
            cpu_group=SimpleNamespace(kind="cpu")
        ),
    }[communicator]

    def worker(rank: int) -> None:
        _local.rank, _local.layer = rank, 0
        _local.posted = worker_posted[rank]
        stream = _Stream(devices[rank], "eplb")
        comm = make()
        comm.set_stream(stream)
        worker_may_go[rank].wait()
        # Rank r sends its row 0 to r+1 and receives r-1's.
        comm.add_send([weights[rank][0]], (rank + 1) % RANKS, expert_id=0)
        comm.add_recv([weights[rank][1]], (rank - 1) % RANKS, expert_id=0)
        comm.execute()

    def main(rank: int) -> None:
        _local.rank = rank
        dev = devices[rank]
        threading.Thread(target=worker, args=(rank,), daemon=True).start()
        try:
            for step in range(4):
                # Rank 2's worker posts its transfer before the forward of
                # step 2; the others after it.
                if step == 2 and rank == 2:
                    worker_may_go[rank].set()
                    worker_posted[rank].wait(timeout=STALL)
                where[rank] = f"forward {step}"
                dev.enqueue("main", key=("forward", step))
                if step == 2 and rank != 2:
                    worker_may_go[rank].set()
                where[rank] = f"EPLB step {step}"
                host_step.wait()
                # The next step reads this step's tokens on the host.
                where[rank] = f"host read after {step}"
                if not dev.synchronize("main"):
                    return
            where[rank] = "done"
        except threading.BrokenBarrierError:
            return

    threads = [threading.Thread(target=main, args=(r,)) for r in range(RANKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=8 * STALL)
    return where


def test_async_transfer_does_not_deadlock_against_the_forward(monkeypatch):
    communicator = _auto_communicator(monkeypatch, use_async=True, xpu=True)
    where = _run_ranks(monkeypatch, communicator)
    assert where == dict.fromkeys(range(RANKS), "done"), (
        f"ranks stuck with communicator {communicator}: {where}"
    )


def test_device_p2p_on_the_worker_stream_deadlocks(monkeypatch):
    """The control: device P2P on the worker stream hangs in this schedule."""
    where = _run_ranks(monkeypatch, "torch_xccl")
    assert where != dict.fromkeys(range(RANKS), "done")
