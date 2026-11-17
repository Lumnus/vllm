# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mailbox all-reduce for the XPU tensor-parallel group (2 or 4 cards).

``VLLM_XPU_MAILBOX_ALLREDUCE=1`` routes the TP group's all-reduces of
contiguous bf16/fp16 tensors up to ``VLLM_XPU_MAILBOX_ALLREDUCE_MAX_BYTES``
through ``torch.ops._xpu_C.mailbox_all_reduce`` instead of XCCL: one kernel
per call, recursive doubling over PCIe peer-to-peer stores into mailboxes in
the partners' memory (Level Zero IPC regions, ``mono_ipc_*``). Everything
else stays on XCCL. Unset, nothing here is constructed.

Summation contract (the kernel's): every rank computes the same bits.
``VLLM_XPU_MAILBOX_ALLREDUCE_PARTIALS=fp32`` (default) sums the tree
``(x_r + x_{r^a}) + (x_{r^b} + x_{r^a^b})`` in fp32 and rounds once to the
input type; ``input`` rounds the stage-1 sums to the input type first (half
the stage-2 bytes, two roundings). ``a, b`` come from
``VLLM_XPU_MAILBOX_ALLREDUCE_ORDER`` (default ``3,1``: stage 1 with rank
``r ^ 3``, stage 2 with ``r ^ 1``). The bits differ from XCCL's ring (other
order and roundings); they are equal on every rank and between eager and
graph replay. ``reference_all_reduce`` below computes them on the CPU.

Failure: a wait in the kernel gives up after
``VLLM_XPU_MAILBOX_ALLREDUCE_TIMEOUT_S`` (default 20 s) and the call writes
NaN; a watchdog thread reads the kernel's host words and stops the process
with one CRITICAL line naming the rank, stage, partner and tag
(``VLLM_XPU_MAILBOX_ALLREDUCE_ON_FAIL=exit``, default; ``log`` only logs and
makes every later call raise, for tests). If the mailboxes cannot be set up
on any rank, every rank keeps XCCL and logs why.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

# sub-group clock of the B70 (~2.8 GHz), for the timeout in cycles
_CLOCK_HZ = 2.8e9
# host words written by the kernel (mailbox_all_reduce.cpp HostWord)
_H_FAILED, _H_STAGE, _H_TAG, _H_ITEM, _H_SEEN = 0, 1, 2, 3, 4
_KERNEL_OPS = (
    "mailbox_all_reduce",
    "mailbox_ar_slot_bytes",
    "mailbox_ar_host_alloc",
    "mailbox_ar_host_read",
    "mono_ipc_create",
    "mono_ipc_export",
    "mono_ipc_open",
    "mono_ipc_ptr",
)


@dataclass(frozen=True)
class MailboxConfig:
    max_bytes: int = 65536
    fp32_partials: bool = True
    order: tuple[int, int] = (3, 1)
    timeout_s: float = 20.0
    on_fail: str = "exit"

    @staticmethod
    def from_env() -> MailboxConfig:
        import vllm.envs as envs

        order = parse_order(envs.VLLM_XPU_MAILBOX_ALLREDUCE_ORDER)
        partials = envs.VLLM_XPU_MAILBOX_ALLREDUCE_PARTIALS
        return MailboxConfig(
            max_bytes=int(envs.VLLM_XPU_MAILBOX_ALLREDUCE_MAX_BYTES),
            fp32_partials=partials == "fp32",
            order=order,
            timeout_s=float(envs.VLLM_XPU_MAILBOX_ALLREDUCE_TIMEOUT_S),
            on_fail=envs.VLLM_XPU_MAILBOX_ALLREDUCE_ON_FAIL,
        )


def parse_order(text: str) -> tuple[int, int]:
    parts = [int(p) for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 2 or parts[0] == parts[1] or not set(parts) <= {1, 2, 3}:
        raise ValueError(
            f"VLLM_XPU_MAILBOX_ALLREDUCE_ORDER must be two different masks "
            f"out of 1, 2, 3 (got {text!r})"
        )
    return parts[0], parts[1]


def mailbox_partners(rank: int, world_size: int, order: tuple[int, int]) -> list[int]:
    """The partner of ``rank`` in each stage (recursive doubling)."""
    if world_size == 2:
        return [rank ^ 1]
    if world_size == 4:
        return [rank ^ order[0], rank ^ order[1]]
    raise ValueError(f"mailbox all-reduce needs 2 or 4 ranks, got {world_size}")


def reference_all_reduce(
    inputs: list[torch.Tensor],
    order: tuple[int, int] = (3, 1),
    fp32_partials: bool = True,
) -> list[torch.Tensor]:
    """What each rank's kernel computes, on the CPU (the summation contract).

    ``inputs[r]`` is rank r's bf16/fp16 tensor; returns rank r's result.
    Every stage adds two fp32 values with one IEEE addition (the kernel's
    ``v + peer``; fp32 addition is commutative, so partners agree)."""
    world = len(inputs)
    dtype = inputs[0].dtype
    vals = [x.float() for x in inputs]
    for s, _ in enumerate(mailbox_partners(0, world, order)):
        if s > 0 and not fp32_partials:
            vals = [v.to(dtype).float() for v in vals]
        nxt = []
        for r in range(world):
            p = mailbox_partners(r, world, order)[s]
            nxt.append(vals[r] + vals[p])
        vals = nxt
    return [v.to(dtype) for v in vals]


def _ops():
    """The kernels' op namespace (a module-level hook, so CPU tests can
    stand in for the kernels)."""
    return torch.ops._xpu_C


def kernel_ops_present() -> bool:
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    ops = _ops()
    return all(hasattr(ops, name) for name in _KERNEL_OPS)


def _send_fd(name: str, server: bool, fd: int, timeout_s: float = 60.0) -> int:
    """Swaps one fd with the stage partner over an abstract unix socket
    (SCM_RIGHTS): no ptrace permission and no file on disk needed."""
    addr = "\0" + name
    if server:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.settimeout(timeout_s)
        srv.bind(addr)
        srv.listen(1)
        try:
            conn, _ = srv.accept()
        finally:
            srv.close()
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                conn.connect(addr)
                break
            except OSError:
                if time.monotonic() > deadline:
                    conn.close()
                    raise
                time.sleep(0.02)
    with conn:
        conn.settimeout(timeout_s)
        socket.send_fds(conn, [b"f"], [fd])
        _, fds, _, _ = socket.recv_fds(conn, 1, 1)
    if not fds:
        raise RuntimeError(f"mailbox all-reduce: no fd received on {name}")
    return fds[0]


class XpuMailboxAllReduce:
    """Per-process state of the mailbox all-reduce for one TP group."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        rank: int,
        world_size: int,
        config: MailboxConfig,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.partners = mailbox_partners(rank, world_size, config.order)
        self.max_numel = config.max_bytes // 2
        self.failed = False
        self.calls = 0
        self._ops = _ops()
        self._handles: list[int] = []
        self.inbox: list[int] = []
        self.outbox: list[int] = []
        self.host = 0
        self.state: torch.Tensor | None = None
        self.slot_bytes = 0
        self.timeout_cycles = (
            int(config.timeout_s * _CLOCK_HZ) if config.timeout_s > 0 else 0
        )
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    # ----------------------------------------------------------------- setup
    @classmethod
    def create(
        cls,
        cpu_group: ProcessGroup,
        device: torch.device,
        rank: int,
        world_size: int,
        config: MailboxConfig | None = None,
    ) -> XpuMailboxAllReduce | None:
        """Collective over ``cpu_group``. Returns None on every rank if any
        rank could not set up its mailboxes (that rank logs why)."""
        try:
            config = config or MailboxConfig.from_env()
        except ValueError as e:
            logger.warning("Mailbox all-reduce off: %s", e)
            return None
        if world_size not in (2, 4):
            logger.info(
                "Mailbox all-reduce off: TP size %d (needs 2 or 4).", world_size
            )
            return None
        # Every rank runs the same two collectives whatever fails locally, so
        # a failure on one rank cannot leave another waiting in a collective.
        obj: XpuMailboxAllReduce | None = None
        infos: list[list[int]] = []
        err = ""
        if not kernel_ops_present():
            err = "the kernels wheel has no mailbox_all_reduce / mono_ipc ops"
        else:
            try:
                obj = cls(cpu_group, device, rank, world_size, config)
                infos = obj._create_regions()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        token = uuid.uuid4().hex if rank == 0 else ""
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (token, err, infos), group=cpu_group)
        bad = [(r, g[1]) for r, g in enumerate(gathered) if g[1]]
        if not bad:
            assert obj is not None
            try:
                obj._open_peers(gathered[0][0], [g[2] for g in gathered])
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            status: list = [None] * world_size
            dist.all_gather_object(status, err, group=cpu_group)
            bad = [(r, e) for r, e in enumerate(status) if e]
        if bad:
            logger.warning("Mailbox all-reduce off on every rank (XCCL stays): %s", bad)
            if obj is not None:
                obj.close()
            return None
        assert obj is not None
        obj._start_watchdog()
        logger.info(
            "Mailbox all-reduce on: TP rank %d of %d, partners %s, <= %d bytes "
            "bf16/fp16, %s partials, timeout %.0f s, slot %d bytes",
            rank,
            world_size,
            obj.partners,
            config.max_bytes,
            "fp32" if config.fp32_partials else "input-type",
            config.timeout_s,
            obj.slot_bytes,
        )
        return obj

    def _create_regions(self) -> list[list[int]]:
        """One IPC region per stage (this rank's inbox for that stage, two
        slots); returns their export records."""
        ops = self._ops
        if self.device.type == "xpu":
            torch.xpu.set_device(self.device)
        slot = int(ops.mailbox_ar_slot_bytes(self.max_numel, self.config.fp32_partials))
        self.slot_bytes = (slot + 63) // 64 * 64
        infos = []
        for _ in self.partners:
            h = int(ops.mono_ipc_create(2 * self.slot_bytes))
            self._handles.append(h)
            infos.append([int(v) for v in ops.mono_ipc_export(h)])
        return infos

    def _open_peers(self, token: str, all_infos: list[list[list[int]]]) -> None:
        """Maps each stage partner's inbox for that stage (the partner's
        region of the same stage: the pairs are symmetric)."""
        ops = self._ops
        # every fd swap first: a failing open below then leaves no partner
        # waiting on a socket
        peer_fds = []
        for s, p in enumerate(self.partners):
            lo, hi = min(self.rank, p), max(self.rank, p)
            name = f"vllm-xpu-mbox-{token}-s{s}-{lo}-{hi}"
            my_fd = all_infos[self.rank][s][1]
            peer_fds.append(_send_fd(name, server=self.rank == lo, fd=my_fd))
        for s, p in enumerate(self.partners):
            ops.mono_ipc_open(self._handles[s], all_infos[p][s], peer_fds[s])
            self.inbox.append(int(ops.mono_ipc_ptr(self._handles[s], False)))
            self.outbox.append(int(ops.mono_ipc_ptr(self._handles[s], True)))
        self.host = int(ops.mailbox_ar_host_alloc())
        self.state = torch.zeros(4, dtype=torch.int32, device=self.device)
        if self.device.type == "xpu":
            torch.xpu.synchronize(self.device)

    # ------------------------------------------------------------- the call
    def should_use(self, t: torch.Tensor) -> bool:
        return (
            not self.failed
            and t.dtype in (torch.bfloat16, torch.float16)
            and t.is_contiguous()
            and 0 < t.numel() <= self.max_numel
        )

    def all_reduce_inplace(self, t: torch.Tensor) -> torch.Tensor:
        if self.failed:
            raise RuntimeError(self._fail_line())
        assert self.state is not None
        self._ops.mailbox_all_reduce(
            t,
            self.inbox,
            self.outbox,
            self.state.data_ptr(),
            self.host,
            self.slot_bytes,
            self.config.fp32_partials,
            self.timeout_cycles,
        )
        if self.calls == 0:
            logger.info(
                "Mailbox all-reduce: first call, %d elements %s (one-time line).",
                t.numel(),
                t.dtype,
            )
        self.calls += 1
        return t

    # ---------------------------------------------------------- failure path
    def health(self) -> list[int]:
        if not self.host:
            return []
        return [int(v) for v in self._ops.mailbox_ar_host_read(self.host)]

    def _fail_line(self) -> str:
        h = self.health()
        stage = h[_H_STAGE] if h else -1
        partner = self.partners[stage] if 0 <= stage < len(self.partners) else -1
        return (
            f"Mailbox all-reduce FAILED on TP rank {self.rank}: stage {stage} "
            f"waited more than {self.config.timeout_s:.0f} s for TP rank "
            f"{partner} (tag {h[_H_TAG] if h else -1}, line "
            f"{h[_H_ITEM] if h else -1}, last tag seen "
            f"{h[_H_SEEN] if h else -1}); a peer stalled or died. The engine "
            f"stops; unset VLLM_XPU_MAILBOX_ALLREDUCE to run on XCCL."
        )

    def _start_watchdog(self) -> None:
        def run() -> None:
            while not self._stop.wait(0.2):
                h = self.health()
                if h and h[_H_FAILED]:
                    self.failed = True
                    logger.critical(self._fail_line())
                    if self.config.on_fail == "exit":
                        os._exit(70)
                    return

        self._watchdog = threading.Thread(
            target=run, name="xpu-mailbox-ar-watchdog", daemon=True
        )
        self._watchdog.start()

    def close(self) -> None:
        self._stop.set()
        if self.host:
            with contextlib.suppress(Exception):
                self._ops.mailbox_ar_host_free(self.host)
            self.host = 0
