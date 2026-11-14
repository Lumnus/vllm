# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts computed on the host CPU (the third place of an expert slot).

With ``--cpu-offload-pinned-gb`` below ``--cpu-offload-gb`` a layer's expert
slots that are off the card no longer all fit in pinned rows the kernels read
through UVA. The ones that do not are *host slots*: both tier maps on the
device say -1 for them, so the kernels skip them, and this module computes
them on the CPU and adds their contribution to the layer's output. Nothing is
masked or dropped: a routed expert of this rank is computed on the card or
here, and a step that cannot compute one fails.

Where the bytes are: a bounded cache of expert units in ordinary (pageable,
page-aligned) host memory, keyed by (layer, logical expert) and filled from
the expert store; a unit not in it is read from the drive when the step needs
it (the store's own ``read_into``). The card never reads this memory.

How a step runs (one MoE layer, inside the MoE op, which runs outside the
captured graphs when this is on; see ``VllmConfig``): the routed ids, their
weights and the hidden states are copied to pinned host buffers and an event
is recorded; the card then runs the layer's fast and pinned experts as
always; the host waits for the event only (not for those kernels), lists the
calls routed to its host slots, makes their units present, computes them with
the kernels' host op (``torch.ops._xpu_C.int4_moe_cpu``, AVX2), and copies the
result to the card, where it is added to the output. The CPU's work overlaps
the card's work on the same layer.

What the CPU computes is the card's W4A16 arithmetic at the card's rounding
points (bf16 after each GEMM and after the activation, fp32 accumulation).
Inside a dot product the card has two kernels: the int4 GEMV (a call of at
most 8 rows: the scale applied to each 32-element chunk's partial sum) and
the grouped GEMM (every larger call, so every call of a top-10 model: each
weight dequantised to bf16(q * s), then bf16 x bf16 into fp32). They differ
in most bf16 outputs, so the CPU takes, per GEMM, the one the card takes for
the same layer call (``card_arith``), when the installed op has the
``arith`` argument; then only the order of fp32 additions differs, and an
expert's output differs from the card's in the last bit of a small share of
its values. An op without it computes the GEMV's arithmetic for every call.

How the CPU's part is added (``XPUExperts._apply_tiered``): with
VLLM_XPU_MOE_COMBINE unset each pair the CPU computes is its own row, summed
with the card's rows of the token in top-k order, fp32, rounded once, as the
card-only layer does (``finish_pairs``); ``=split`` adds the CPU's own
bf16-rounded per-token sum to the card's (``finish``).

When the card runs the layer's experts W4A8 (int8 per-token activations,
``XPUExpertsW4A8Int``), the CPU does too, with the host op
``int4_moe_cpu_a8``: the same expert rows, the activations quantized on the
host by the device's own quantization source, integer dot products per
128-K group and the card's fp32 group sum, so an expert's output equals the
card's bit for bit except where silu's exp (a hardware approximation on the
card) moves a value across a rounding boundary. The W4A8 card path runs no
output hook, so its layers keep the per-tier sums (``finish``).
"""

import contextlib
import functools
import mmap
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1 import runtime_debug

if TYPE_CHECKING:
    from vllm.config.expert_store import ExpertStoreConfig
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
        ExpertTiers,
        HostSelection,
    )

logger = init_logger(__name__)

# The tensors of an expert unit, in the order the op's offsets name them.
UNIT_TENSORS = ("w13", "w2", "w13_scales", "w2_scales")
# The kernels' dispatch levels (int4_moe_cpu_isa): auto picks the widest of
# 1..7 the host has; every level but 8 computes the scalar code's result bit
# for bit; 8 adds the card-class AMX-BF16 prefill (not bit-identical, asked
# for explicitly). Kernels before the levels know 0..2 only.
_ISA = {
    "auto": 0,
    "scalar": 1,
    "avx2": 2,
    "avx-vnni": 3,
    "avx512": 4,
    "avx512-vnni": 5,
    "avx512-bf16": 6,
    "amx": 7,
    "amx-card-class": 8,
}
_ISA_NAME = {
    1: "scalar",
    2: "AVX2+FMA",
    3: "AVX2+FMA+AVX-VNNI",
    4: "AVX-512",
    5: "AVX-512+VNNI",
    6: "AVX-512+VNNI+BF16",
    7: "AMX+AVX-512",
    8: "AMX+AVX-512, card-class AMX-BF16 prefill",
}
# the level whose prefill is not the scalar code's result bit for bit
_ISA_CARD_CLASS = 8
PAGE = 4096
# Up to this many calls in a layer call, its experts are listed with a dict
# (cheaper than numpy's sort at decode sizes); above, with np.unique.
_FEW_CALLS = 256

Key = tuple[int, int]  # (store layer, logical expert)


class CpuExpertError(RuntimeError):
    """The CPU tier cannot serve an expert it was routed: the step fails."""


# The op's library namespaces, in the order they are looked up: the XPU
# kernels' extension carries it today; a host-only build of the same source
# (no device code, loadable beside any accelerator's build) would register
# it under the second name.
_OP_NAMESPACES = ("_xpu_C", "_cpu_moe_C")


def _ops() -> Any:
    for name in _OP_NAMESPACES:
        try:
            ns = getattr(torch.ops, name)
            if hasattr(ns, "int4_moe_cpu"):
                return ns
        except (AttributeError, RuntimeError):
            continue
    return None


def host_op_available() -> bool:
    return _ops() is not None


def _op_isa(want: int) -> int:
    """What the host op resolves ``want`` (0 auto, 1 scalar, 2 AVX2) to."""
    return int(_ops().int4_moe_cpu_isa(want))


def _op_run(*args: Any) -> None:
    _ops().int4_moe_cpu(*args)


def _op_configure(cores: list[int], spin_us: int) -> bool:
    """Pin the op's pool workers to ``cores`` and set their spin time; False
    when the loaded op is older than this setting."""
    ns = _ops()
    if ns is None or not hasattr(ns, "int4_moe_cpu_configure"):
        return False
    ns.int4_moe_cpu_configure(cores, spin_us)
    return True


# How a layer call's host calls are listed (environment, read when the tier
# starts): VLLM_CPU_EXPERTS_SELECT unset or "op": by the kernels' select op
# when the loaded kernels have it (one pass in C++), else with numpy;
# "numpy": with numpy always. Both give the same lists.
_SELECT_ENV = "VLLM_CPU_EXPERTS_SELECT"
_NO_MAP = torch.empty(0, dtype=torch.int32)


# Where a layer call's CPU part runs (environment, read when the tier
# starts): VLLM_CPU_EXPERTS_ASYNC unset or "0": on the thread that runs the
# model, which waits for the routing, computes and then enqueues the rest
# (finish_pairs); "spin": on a host thread of its own (_AsyncHost), while the
# model's thread goes on enqueuing; the card waits for the result rows with
# a kernel that reads a pinned host word (xpu_host_wait). Only for the sums
# that take each pair's row (VLLM_XPU_MOE_COMBINE unset with this tier, rank,
# exact), layer calls of at most VLLM_CPU_EXPERTS_ASYNC_MAX_TOKENS tokens and
# no prefill staging; every other layer call runs as with it unset. The CPU
# computes the same rows from the same inputs either way.
_ASYNC_ENV = "VLLM_CPU_EXPERTS_ASYNC"
_ASYNC_TOKENS_ENV = "VLLM_CPU_EXPERTS_ASYNC_MAX_TOKENS"
_ASYNC_SPINS_ENV = "VLLM_CPU_EXPERTS_ASYNC_SPINS"
_ASYNC_DEFAULT_TOKENS = 32
# Reads of the host word before the card gives up (each a read over PCIe,
# about 1 us: about a second; below the driver's hang check).
_ASYNC_DEFAULT_SPINS = 1_000_000
# How long the model's thread waits for the host thread to finish what was
# posted before it fails the step (it waits between steps and before a
# layer call that runs on the model's thread).
_ASYNC_DRAIN_S = 60.0


def _async_settings() -> tuple[int, int] | None:
    """(max tokens, max spins) with VLLM_CPU_EXPERTS_ASYNC=spin, else None."""
    mode = os.environ.get(_ASYNC_ENV, "").strip().lower()
    if mode in ("", "0", "off"):
        return None
    if mode != "spin":
        raise CpuExpertError(f"{_ASYNC_ENV} is 'spin' or unset, not {mode!r}")
    tokens = int(os.environ.get(_ASYNC_TOKENS_ENV, "") or _ASYNC_DEFAULT_TOKENS)
    spins = int(os.environ.get(_ASYNC_SPINS_ENV, "") or _ASYNC_DEFAULT_SPINS)
    if tokens < 1 or spins < 1:
        raise CpuExpertError(f"{_ASYNC_TOKENS_ENV} and {_ASYNC_SPINS_ENV} are >= 1")
    return tokens, spins


class _AsyncHost:
    """A layer call's CPU part on a host thread of its own
    (VLLM_CPU_EXPERTS_ASYNC=spin).

    The model's thread posts the layer call (its inputs are on their way to
    pinned buffers, :meth:`CpuExpertRuntime.post`), hands it here with a
    sequence number, enqueues the card's wait for that number and the copies
    of the result rows and the pair map, and goes on with the step: it no
    longer waits for the routing, so the device queue does not drain at every
    MoE layer. This thread waits for the routing, lists the calls, computes
    them into the pinned rows, writes the map, and then the number into the
    ``done`` word; the card's wait ends and its copies read the rows.

    One set of pinned buffers is enough: the next layer call's inputs reach
    them by a copy the card runs after this call's wait ended (stream order),
    so after this thread finished with them, and its results are written
    after those inputs arrived, so after this call's rows were copied out.

    A failure (an expert the store cannot give) writes the number into the
    ``failed`` word, so the card stops waiting, and fails the step on the
    model's thread: at the next layer call's post, or when the model's thread
    drains this thread between steps (:meth:`drain`, from the model runner's
    step hook). A card that stopped waiting (timed out) is failed the same
    way."""

    def __init__(self, runtime: "CpuExpertRuntime", max_tokens: int, spins: int):
        from vllm.model_executor.layers.fused_moe import xpu_host_wait

        self.runtime = runtime
        self.max_tokens = max_tokens
        self.spins = spins
        self._wait_mod = xpu_host_wait
        self.atomic = xpu_host_wait.read_mode() == "atomic"
        self.flag = xpu_host_wait.flag_words()
        self.words = self.flag.numpy()
        self._views: dict[str, torch.Tensor] = {}
        self.seq = 0
        self.posted = 0  # layer calls handed here
        self._jobs: list[tuple[Pending, int] | None] = []
        self._cond = threading.Condition()
        self._open = 0  # handed here and not finished
        self.error: BaseException | None = None
        self.core = runtime.cores[0] if runtime.cores else None
        self._thread = threading.Thread(
            target=self._loop, name="cpu-experts-host", daemon=True
        )
        self._thread.start()

    def view(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        v = self._views.get(key)
        if v is None:
            v = self._views[key] = self._wait_mod.device_view(self.flag, device)
        return v

    def check(self) -> None:
        """Fail the step if this thread failed a layer call or the card
        stopped waiting for one."""
        if self.error is not None:
            raise CpuExpertError(
                "CPU experts: a layer call computed on the host thread failed "
                f"({self.error}); the step's output is not valid"
            ) from self.error
        stuck = int(self.words[self._wait_mod.TIMED_OUT])
        if stuck:
            raise CpuExpertError(
                f"CPU experts: the card stopped waiting for layer call {stuck} "
                f"after {self.spins} reads of the host word (the host thread "
                "did not finish it in time); the step's output is not valid"
            )

    def drain(self, timeout_s: float = _ASYNC_DRAIN_S) -> None:
        """Wait until every layer call handed here is finished, then
        :meth:`check`."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._open == 0, timeout_s)
        if not ok:
            raise CpuExpertError(
                f"CPU experts: the host thread did not finish {self._open} "
                f"layer calls in {timeout_s:.0f} s"
            )
        self.check()

    def submit(self, pending: "Pending", device: torch.device) -> "HostPairs":
        """Hand the layer call to this thread and enqueue the card's side:
        the wait for it, then the copies of the result rows (every pair's row
        slot: how many the CPU computes is not known yet) and the map."""
        self.check()
        n, b = pending.num_tokens, pending.buffers
        topk = b.ids.shape[1]
        self.runtime._pair_buffers(b, n * topk)
        assert b.pair_rows is not None and b.row_map is not None
        self.seq = self.seq % 0x7FFFFFFF + 1
        seq = self.seq
        if pending.times is not None:
            pending.times["handed"] = time.perf_counter()
        with self._cond:
            self._jobs.append((pending, seq))
            self._open += 1
            self.posted += 1
            self._cond.notify()
        self._wait_mod.wait_host_flag(
            self.flag, self.view(device), seq, self.spins, atomic=self.atomic
        )
        return HostPairs(
            b.pair_rows[: n * topk].to(device, non_blocking=True),
            b.row_map[:n].to(device, non_blocking=True),
        )

    def stop(self) -> None:
        with self._cond:
            self._jobs.append(None)
            self._cond.notify()

    def _loop(self) -> None:
        if self.core is not None and hasattr(os, "sched_setaffinity"):
            # cores[0] of the op's set is meant for the op's caller
            with contextlib.suppress(OSError):
                os.sched_setaffinity(0, {self.core})
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._jobs))
                job = self._jobs.pop(0)
            if job is None:
                return
            pending, seq = job
            word = self._wait_mod.DONE
            try:
                if self.error is None:
                    pair = self.runtime._pairs_into_buffers(pending, fill_map=True)
                    if pending.times is not None:
                        self.runtime._emit_host_thread(pending, pair)
                else:
                    word = self._wait_mod.FAILED
            except BaseException as e:  # noqa: BLE001 - failed on the model's thread
                self.error = e
                word = self._wait_mod.FAILED
            finally:
                self.words[word] = seq
                with self._cond:
                    self._open -= 1
                    self._cond.notify_all()


def drain_async() -> None:
    """Between steps (the model runner's step hook): wait for the host thread
    of VLLM_CPU_EXPERTS_ASYNC to finish what the step handed it, and fail the
    step if a layer call failed. Nothing without that thread."""
    rt = _RUNTIME
    if rt is not None and rt._async is not None:
        rt._async.drain()


def _select_op() -> Any:
    """The kernels' int4_moe_cpu_select, or None (not loaded, or
    VLLM_CPU_EXPERTS_SELECT=numpy)."""
    want = os.environ.get(_SELECT_ENV, "").strip().lower() or "op"
    if want not in ("op", "numpy"):
        raise CpuExpertError(f"{_SELECT_ENV} is 'op' or 'numpy', not {want!r}")
    ns = _ops()
    if want == "numpy" or ns is None or not hasattr(ns, "int4_moe_cpu_select"):
        return None
    return ns.int4_moe_cpu_select


def _select_tables(tiers: Any) -> tuple[torch.Tensor, ...] | None:
    """The layer's tier, slot and logical-expert tables as tensors sharing
    the arrays' memory (the op reads them as they are now), made again when
    an array is replaced; None when they are not the op's types."""
    arrays = tuple(
        getattr(tiers, name, None)
        for name in ("_tier_of_global", "_slot_of_global", "slot_logical")
    )
    kept = getattr(tiers, "_cpu_select_tables", None)
    if kept is not None and all(a is b for a, b in zip(kept[0], arrays)):
        return kept[1]
    if not (
        isinstance(arrays[0], np.ndarray)
        and arrays[0].dtype == np.int8
        and all(isinstance(a, np.ndarray) and a.dtype == np.int64 for a in arrays[1:])
        and all(a.flags.c_contiguous for a in arrays)
    ):
        return None
    tensors = tuple(torch.from_numpy(a) for a in arrays)
    tiers._cpu_select_tables = (arrays, tensors)
    return tensors


# Where the op's threads run (environment, read once per process):
#   VLLM_CPU_EXPERTS_CORES  unset or "": not pinned (the threads go where the
#       kernel puts them). "auto": the physical cores of the process's CPU
#       mask (one hardware thread each), split in order into as many equal
#       parts as there are ranks on the host; this rank takes the first
#       cpu_threads cores of its part (on EPYC the order is CCX after CCX, so
#       the parts are whole CCXs when the counts allow). A list such as
#       "0-11;12-23" gives each local rank its own set (rank r takes the r-th,
#       one set for all ranks when only one is given).
#   VLLM_CPU_EXPERTS_SPIN_US  how long a worker spins after a phase before
#       it sleeps (unset: the op's default, 100). Spinning through the gap
#       between two MoE layers (a few ms) saves the wake-ups of every layer
#       call; worth it on cores given to the op alone.
_CORES_ENV = "VLLM_CPU_EXPERTS_CORES"
_SPIN_ENV = "VLLM_CPU_EXPERTS_SPIN_US"


def _parse_cpu_list(text: str) -> list[int]:
    cores: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return cores


def _physical_cores(allowed: set[int]) -> list[int]:
    """One hardware thread per physical core of ``allowed``, in order."""
    seen: set[str] = set()
    out = []
    for cpu in sorted(allowed):
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(path) as f:
                key = f.read().strip()
        except OSError:
            key = str(cpu)
        if key in seen:
            continue
        seen.add(key)
        out.append(cpu)
    return out


def cpu_cores_for_rank(
    setting: str, local_rank: int, local_ranks: int, threads: int
) -> list[int]:
    """The cores this rank's op threads take (empty: not pinned)."""
    setting = setting.strip()
    if not setting:
        return []
    if setting == "auto":
        cores = _physical_cores(set(os.sched_getaffinity(0)))
        per = len(cores) // max(1, local_ranks)
        if per < 1:
            raise CpuExpertError(
                f"{_CORES_ENV}=auto: {len(cores)} physical cores for "
                f"{local_ranks} ranks"
            )
        part = cores[local_rank * per : (local_rank + 1) * per]
        return part[:threads]
    sets = [s for s in setting.split(";") if s.strip()]
    if len(sets) > 1 and local_rank >= len(sets):
        raise CpuExpertError(
            f"{_CORES_ENV} names {len(sets)} core sets and this is local rank "
            f"{local_rank}"
        )
    return _parse_cpu_list(sets[local_rank] if len(sets) > 1 else sets[0])


def _local_rank() -> tuple[int, int]:
    """(local rank, ranks on this host); (0, 1) outside a distributed run.
    The host's rank count is the world group's size (one host)."""
    try:
        from vllm.distributed.parallel_state import get_world_group

        world = get_world_group()
        return int(world.local_rank), int(world.world_size)
    except (AssertionError, AttributeError, ImportError, RuntimeError):
        return 0, 1


@functools.cache
def op_takes_arith() -> bool:
    """Whether the installed host op has the ``arith`` argument (the card
    kernel's arithmetic per GEMM)."""
    ns = _ops()
    if ns is None:
        return False
    try:
        schema = ns.int4_moe_cpu.default._schema
    except (AttributeError, RuntimeError):
        return False
    return any(a.name == "arith" for a in schema.arguments)


_FMI = "vllm_xpu_kernels.fused_moe_interface"
_GEMV_LIMITS: tuple[Any, tuple[int, str]] | None = None


def _gemv_limits() -> tuple[int, str] | None:
    """The kernels' int4 GEMV row limit and its switch's variable name (None:
    kernels without the GEMV). card_arith runs per layer call: the module's
    constants are read again only when the module object changes."""
    global _GEMV_LIMITS
    cached = _GEMV_LIMITS
    if cached is not None and sys.modules.get(_FMI) is cached[0]:
        return cached[1]
    try:
        from vllm_xpu_kernels import fused_moe_interface as fmi

        limits = int(fmi.INT4_GEMV_MAX_ROWS), str(fmi.INT4_GEMV_ENV)
    except (ImportError, AttributeError):
        return None
    _GEMV_LIMITS = (fmi, limits)
    return limits


def op_takes_x_row() -> bool:
    """Whether the installed host op has ``call_x_row`` (one output row per
    call reading its input row of x in place)."""
    ns = _ops()
    if ns is None:
        return False
    try:
        schema = ns.int4_moe_cpu.default._schema
    except (AttributeError, RuntimeError):
        return False
    return any(a.name == "call_x_row" for a in schema.arguments)


def card_arith(rows: int, hidden: int, inter: int, group: int) -> int:
    """The op's ``arith`` for a layer call the card runs with ``rows``
    (tokens x top_k) rows: per GEMM (bit 0 GEMM1 with K = hidden, bit 1
    GEMM2 with K = inter) set when the card's XpuFusedMoe runs it on the
    grouped GEMM (weights bf16(q * s)), clear when on its int4 GEMV. The
    GEMV takes at most ``INT4_GEMV_MAX_ROWS`` rows, K a multiple of 512 and
    a group a multiple of 32, unless switched off (``INT4_GEMV_ENV``); kernels
    without it run the grouped GEMM always."""
    limits = _gemv_limits()
    if limits is None:
        return 3
    max_rows, env = limits
    off = os.environ.get(env, "1").strip().upper() in ("0", "OFF", "FALSE", "NO", "N")

    def grouped(k: int) -> bool:
        return off or rows > max_rows or k % 512 != 0 or group % 32 != 0

    return int(grouped(hidden)) | (int(grouped(inter)) << 1)


def host_op_a8_available() -> bool:
    """Whether the kernels have the W4A8 host op (int8 activations), in the
    library that carries the W4A16 op."""
    ns = _ops()
    return ns is not None and hasattr(ns, "int4_moe_cpu_a8")


def _op_run_a8(*args: Any) -> None:
    _ops().int4_moe_cpu_a8(*args)


# The card's W4A8 path (vllm_xpu_kernels.moe_w4a8.apply, fused_act off) in
# the host op's scheme list: per-row int8 of both GEMM inputs with the
# bf16-rounded scale, fused multiply-adds over the groups.
W4A8_CARD_SCHEME: list[int] = []


# --------------------------------------------------------------------------
# The card's arithmetic, in torch (reference for the self-check and tests)
# --------------------------------------------------------------------------


def _int4_rows_dot(
    w: torch.Tensor, s: torch.Tensor, a: torch.Tensor, group: int
) -> torch.Tensor:
    """``a`` [T, K] fp32 times the int4 rows ``w`` [N, K/2] (two's complement
    nibbles, element 2j low) with bf16 scales ``s`` [N, K/G]: fp32 partial
    sums of 32-element chunks, each times its group's scale, summed."""
    b = w.to(torch.int16)
    n = w.shape[0]
    k = w.shape[1] * 2
    q = torch.empty(n, k, dtype=torch.float32)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        q[:, half::2] = torch.where(nib >= 8, nib - 16, nib).float()
    chunks = torch.einsum(
        "nck,tck->tnc", q.view(n, k // 32, 32), a.view(-1, k // 32, 32)
    )
    scale = s.float().repeat_interleave(group // 32, dim=1)
    return (chunks * scale[None]).sum(-1)


def _int4_rows_dot_bf16w(
    w: torch.Tensor, s: torch.Tensor, a: torch.Tensor, group: int
) -> torch.Tensor:
    """The grouped GEMM's dot product: each weight dequantised to
    bf16(q * s), times ``a`` [T, K], accumulated in float64 (the order of the
    card's fp32 sums is the one thing not reproduced), returned in fp32."""
    b = w.to(torch.int16)
    q = torch.empty(w.shape[0], w.shape[1] * 2, dtype=torch.float32)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        q[:, half::2] = torch.where(nib >= 8, nib - 16, nib).float()
    wq = (q * s.float().repeat_interleave(group, dim=1)).to(torch.bfloat16)
    return (a.double() @ wq.double().T).float()


def reference_int4_experts(
    x: torch.Tensor,
    units: Mapping[Any, Mapping[str, torch.Tensor]],
    calls: Sequence[tuple[Any, int, float]],
    inter: int,
    group: int,
    arith: int = 0,
) -> torch.Tensor:
    """What the card computes for ``calls`` (unit key, token, weight) in
    token order: per call ``bf16(bf16(silu(g)) * u)`` between two bf16-rounded
    int4 GEMMs, summed per token in call order in fp32, rounded to bf16.
    ``units[key]`` maps the four unit tensor names to their bytes (uint8).
    ``arith`` as the op's (``card_arith``): bit 0 / bit 1 set, GEMM1 / GEMM2
    as the grouped GEMM computes it, clear as the int4 GEMV does."""
    dot1 = _int4_rows_dot_bf16w if arith & 1 else _int4_rows_dot
    dot2 = _int4_rows_dot_bf16w if arith & 2 else _int4_rows_dot
    hidden = x.shape[1]
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)

    def bf16(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.bfloat16).float()

    for key, token, weight in calls:
        u = units[key]
        w13 = u["w13"].view(torch.uint8).view(2 * inter, hidden // 2)
        w2 = u["w2"].view(torch.uint8).view(hidden, inter // 2)
        s13 = u["w13_scales"].view(torch.bfloat16).view(2 * inter, -1)
        s2 = u["w2_scales"].view(torch.bfloat16).view(hidden, -1)
        h = bf16(dot1(w13, s13, x[token : token + 1].float(), group))
        gate, up = h[:, :inter], h[:, inter:]
        act = bf16(bf16(gate / (1.0 + torch.exp(-gate))) * up)
        y = bf16(dot2(w2, s2, act, group))
        out[token] += y[0] * float(weight)
    return out.to(torch.bfloat16)


def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fused multiply-add: the products here (an int32 dot of at most
    2^17 times a bf16 scale; a bf16 value times an fp32 weight) are exact in
    float64, and one rounding of the float64 sum to fp32 is fma's."""
    return (a.double() * b.double() + c.double()).float()


def int8_per_token(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The card's ``dynamic_per_token_scaled_int8_quant`` on bf16 values
    ``v`` [T, K]: scale ``bf16(max(amax / 127, 1e-5))``, ``q`` half to even
    against the exact boundary, in [-128, 127]."""
    amax = v.abs().amax(-1, keepdim=True)
    s = torch.clamp(amax / 127.0, min=1e-5).to(torch.bfloat16).float()
    f = torch.floor(v / s)
    boundary = (f + 0.5) * s
    even = torch.floor(f * 0.5) == f * 0.5
    q = torch.where(
        v > boundary, f + 1, torch.where(v < boundary, f, torch.where(even, f, f + 1))
    )
    return q.clamp(-128, 127).to(torch.int64), s


def _w4a8_rows(
    w: torch.Tensor, s: torch.Tensor, q: torch.Tensor, qs: torch.Tensor, group: int
) -> torch.Tensor:
    """int8 rows ``q`` [T, K] (scale ``qs`` [T, 1]) times the int4 rows
    ``w`` [N, K/2] with bf16 scales ``s`` [N, K/G], the card's W4A8 GEMM:
    an int32 dot per group, the fp32 sum over groups in order with fused
    multiply-adds, times the token scale."""
    b = w.to(torch.int16)
    n, k = w.shape[0], w.shape[1] * 2
    wq = torch.empty(n, k, dtype=torch.int64)
    for half, nib in ((0, b & 0xF), (1, (b >> 4) & 0xF)):
        wq[:, half::2] = torch.where(nib >= 8, nib - 16, nib).to(torch.int64)
    F = torch.zeros(q.shape[0], n, dtype=torch.float32)
    sf = s.float()
    for g in range(k // group):
        d = q[:, g * group : (g + 1) * group] @ wq[:, g * group : (g + 1) * group].T
        F = _fma32(d.float(), sf[:, g][None, :].expand_as(F), F)
    return F * qs


def reference_w4a8_experts(
    x: torch.Tensor,
    units: Mapping[Any, Mapping[str, torch.Tensor]],
    calls: Sequence[tuple[Any, int, float]],
    inter: int,
    group: int,
) -> torch.Tensor:
    """What the card's W4A8 path computes for ``calls`` (unit key, token,
    weight) in token order, with silu's exp taken as fp32 of a float64 exp
    (the host op's; the card's hardware exp can differ in the last bits):
    int8 per token, W4A8 GEMM, bf16, ``bf16(bf16(silu(g)) * u)``, int8 per
    row, W4A8 GEMM, bf16, summed per token in call order with fused
    multiply-adds, bf16."""
    hidden = x.shape[1]
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)

    def bf16(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.bfloat16).float()

    for key, token, weight in calls:
        u = units[key]
        w13 = u["w13"].view(torch.uint8).view(2 * inter, hidden // 2)
        w2 = u["w2"].view(torch.uint8).view(hidden, inter // 2)
        s13 = u["w13_scales"].view(torch.bfloat16).view(2 * inter, -1)
        s2 = u["w2_scales"].view(torch.bfloat16).view(hidden, -1)
        q1, a1 = int8_per_token(x[token : token + 1].float())
        h = bf16(_w4a8_rows(w13, s13, q1, a1, group))
        gate, up = h[:, :inter], h[:, inter:]
        silu = gate / (1.0 + torch.exp(-gate.double()).float())
        q2, a2 = int8_per_token(bf16(bf16(silu) * up))
        y = bf16(_w4a8_rows(w2, s2, q2, a2, group))
        out[token] = _fma32(y[0], torch.tensor(float(weight)), out[token])
    return out.to(torch.bfloat16)


def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest difference of two bf16 results in units of the bf16
    spacing at ``b``'s largest magnitude."""
    ref = float(b.float().abs().max())
    if ref == 0.0:
        return 0.0 if float(a.float().abs().max()) == 0.0 else float("inf")
    spacing = 2.0 ** (np.floor(np.log2(ref)) - 7)
    return float((a.float() - b.float()).abs().max()) / spacing


# --------------------------------------------------------------------------
# The host cache (ordinary memory in front of the store)
# --------------------------------------------------------------------------


@dataclass
class UnitLayout:
    """Where the four tensors of an expert sit in one cache row."""

    hidden: int
    inter: int
    group: int
    nbytes: dict[str, int]  # tensor name -> bytes, in UNIT_TENSORS order

    @property
    def offsets(self) -> list[int]:
        out, at = [], 0
        for name in UNIT_TENSORS:
            out.append(at)
            at += self.nbytes[name]
        return out

    @property
    def payload(self) -> int:
        return sum(self.nbytes.values())

    @property
    def row_bytes(self) -> int:
        return -(-self.payload // PAGE) * PAGE


class HostExpertCache:
    """Expert units in ordinary host memory, a bounded LRU keyed by (store
    layer, logical expert), filled from the expert store.

    The rows are one page-aligned anonymous mapping, so the store can read a
    unit straight into a row with O_DIRECT (memory the accelerator pinned
    cannot take such reads; this memory can). The card never reads it.

    Thread use: :meth:`acquire` / :meth:`release` on the thread that runs the
    forward; :meth:`prefetch` from any thread. A row in use by the running
    call or being filled is never evicted.
    """

    def __init__(
        self,
        store: Any,
        layout: UnitLayout,
        num_rows: int,
        read_timeout_s: float,
    ):
        if num_rows < 1:
            raise ValueError("the host expert cache needs at least one row")
        self.store = store
        self.layout = layout
        self.num_rows = num_rows
        self.read_timeout_s = read_timeout_s
        self._map = mmap.mmap(-1, num_rows * layout.row_bytes)
        self.slab = torch.frombuffer(self._map, dtype=torch.uint8).view(
            num_rows, layout.row_bytes
        )
        self._rows: OrderedDict[Key, int] = OrderedDict()  # LRU order
        self._free: list[int] = list(range(num_rows))
        self._filling: dict[Key, tuple[int, Future]] = {}
        self._busy: set[int] = set()
        # Rows a background copy reads (out of the cache into a pinned row):
        # a count per row, never evicted while above zero.
        self._held: dict[int, int] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0  # read from the drive while a step waited
        self.prefetched = 0
        self.evicted = 0
        self.read_ms = 0.0

    @property
    def gib(self) -> float:
        return self.num_rows * self.layout.row_bytes / 2**30

    def dests(self, row: int) -> dict[str, torch.Tensor]:
        """The four tensor slices of a row, as the store's read wants them."""
        out = {}
        for name, offset in zip(UNIT_TENSORS, self.layout.offsets):
            out[name] = self.slab[row, offset : offset + self.layout.nbytes[name]]
        return out

    def __contains__(self, key: Key) -> bool:
        with self._lock:
            return key in self._rows

    def _take_row_locked(self, evict: bool) -> int | None:
        if self._free:
            return self._free.pop()
        if not evict:
            return None
        for key, row in self._rows.items():  # oldest first
            if row not in self._busy and row not in self._held:
                del self._rows[key]
                self.evicted += 1
                return row
        return None

    def acquire(self, keys: Sequence[Key]) -> list[int]:
        """The rows holding ``keys`` (in order; repeated keys allowed), read
        from the store first where absent; they stay in place until
        :meth:`release`. Raises :class:`CpuExpertError` when a unit cannot be
        made present (no row free of the running call, or the read failed or
        did not return within the store's read timeout)."""
        wanted = list(dict.fromkeys(keys))
        rows: dict[Key, int] = {}
        reads: list[tuple[Key, int, Future]] = []
        waits: list[tuple[Key, int, Future]] = []
        with self._lock:
            # The leading run of hits (every key, in a warm decode step) in one
            # sweep; the rest key by key, as before.
            lookup, touch = self._rows.get, self._rows.move_to_end
            hit = 0
            for key in wanted:
                row = lookup(key)
                if row is None:
                    break
                touch(key)
                rows[key] = row
                hit += 1
            self._busy.update(rows.values())
            self.hits += hit
            for key in wanted[hit:]:
                row = self._rows.get(key)
                if row is not None:
                    self._rows.move_to_end(key)
                    self._busy.add(row)
                    rows[key] = row
                    self.hits += 1
                elif key in self._filling:
                    row, future = self._filling[key]
                    self._busy.add(row)
                    waits.append((key, row, future))
                else:
                    row = self._take_row_locked(evict=True)
                    if row is None:
                        self._busy.difference_update(rows.values())
                        raise CpuExpertError(
                            f"the host expert cache has {self.num_rows} rows and "
                            f"one step needs {len(wanted)} experts on the CPU: "
                            "raise host_cache_gib"
                        )
                    self._busy.add(row)
                    self.misses += 1
                    future = self.store.read_into_async(key, self.dests(row))
                    self._filling[key] = (row, future)
                    reads.append((key, row, future))
        start = time.perf_counter()
        deadline = time.monotonic() + self.read_timeout_s
        failure: BaseException | None = None
        for key, row, future in reads + waits:
            try:
                future.result(max(0.0, deadline - time.monotonic()))
                rows[key] = row
            except FutureTimeout:
                failure = failure or CpuExpertError(
                    f"expert {key} was not read from the store within "
                    f"{self.read_timeout_s:g} s"
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failure = failure or CpuExpertError(
                    f"expert {key} could not be read from the store: {exc}"
                )
        self.read_ms += (time.perf_counter() - start) * 1e3
        with self._lock:
            for key, row, future in reads + waits:
                if self._filling.get(key, (None, None))[1] is future:
                    del self._filling[key]
                    if future.done() and future.exception() is None:
                        self._rows[key] = row
                    elif future.done():
                        self._free.append(row)
                    # a read still running keeps its row out of the cache
            if failure is not None:
                self._busy.difference_update(rows.values())
                self._busy.difference_update(r for _, r, _ in reads + waits)
        if failure is not None:
            raise failure
        return [rows[key] for key in keys]

    def release(self, rows: Sequence[int]) -> None:
        with self._lock:
            self._busy.difference_update(rows)

    def prefetch(self, keys: Sequence[Key], *, evict: bool = True) -> int:
        """Start reading ``keys`` that are absent into rows (in the
        background); returns how many reads started. Without ``evict`` only
        free rows are used."""
        started: list[tuple[Key, int, Future]] = []
        with self._lock:
            for key in dict.fromkeys(keys):
                if key in self._rows or key in self._filling:
                    continue
                row = self._take_row_locked(evict)
                if row is None:
                    break
                future = self.store.read_into_async(key, self.dests(row))
                self._filling[key] = (row, future)
                started.append((key, row, future))
        # Outside the lock: a read already done runs its callback here.
        for key, row, future in started:
            future.add_done_callback(
                lambda f, key=key, row=row: self._prefetched(key, row, f)
            )
        return len(started)

    def _prefetched(self, key: Key, row: int, future: Future) -> None:
        with self._lock:
            if self._filling.get(key, (None, None))[1] is not future:
                return  # a step took it over and installed it
            del self._filling[key]
            if future.exception() is None:
                self._rows[key] = row
                self.prefetched += 1
            else:
                self._free.append(row)
                logger.warning(
                    "CPU experts: reading expert %s into the host cache ahead "
                    "of use failed (%s); a step that needs it reads it again.",
                    key,
                    future.exception(),
                )

    def row_of(self, key: Key) -> int | None:
        with self._lock:
            return self._rows.get(key)

    def copy_unit_out(self, key: Key, dests: Mapping[str, torch.Tensor]) -> bool:
        """Copy the unit of ``key`` from its row into ``dests`` (the four
        tensors of a pinned row), from any thread: the row is held, so not
        evicted, while it is copied. False when the cache does not hold it."""
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                return False
            self._rows.move_to_end(key)
            self._held[row] = self._held.get(row, 0) + 1
        try:
            src = self.dests(row)
            for name, dst in dests.items():
                dst.copy_(src[name])
        finally:
            with self._lock:
                left = self._held.get(row, 1) - 1
                if left > 0:
                    self._held[row] = left
                else:
                    self._held.pop(row, None)
        return True

    def rows_for_gib(self, gib: float) -> int:
        return max(1, int(gib * 2**30) // self.layout.row_bytes)

    def resize_refusal(self) -> str | None:
        """Why the cache cannot be resized now (None: it can)."""
        with self._lock:
            if self._filling:
                return (
                    f"{len(self._filling)} reads into the host cache are in "
                    "flight; ask again when they have landed"
                )
            if self._busy:
                return "a layer call holds rows of the host cache"
            if self._held:
                return (
                    "rows of the host cache are being copied into pinned rows; "
                    "ask again"
                )
        return None

    def resize(self, num_rows: int) -> None:
        """A new mapping of ``num_rows`` rows, between two steps (no read in
        flight, no row held): the most recently used experts are copied over
        as far as they fit, the rest count as evicted. The old mapping goes
        when its last view does; both exist while the copy runs."""
        if num_rows < 1:
            raise ValueError("the host expert cache needs at least one row")
        with self._lock:
            if self._filling or self._busy or self._held:
                raise RuntimeError(self.resize_refusal_locked())
            keep = list(self._rows.items())[-num_rows:]  # LRU order: newest last
            new_map = mmap.mmap(-1, num_rows * self.layout.row_bytes)
            slab = torch.frombuffer(new_map, dtype=torch.uint8).view(
                num_rows, self.layout.row_bytes
            )
            rows: OrderedDict[Key, int] = OrderedDict()
            for i, (key, row) in enumerate(keep):
                slab[i].copy_(self.slab[row])
                rows[key] = i
            self.evicted += len(self._rows) - len(keep)
            self._map, self.slab = new_map, slab
            self._rows = rows
            self._free = list(range(len(keep), num_rows))
            self.num_rows = num_rows

    def resize_refusal_locked(self) -> str:
        if self._filling:
            return f"{len(self._filling)} reads into the host cache are in flight"
        if self._held:
            return "rows of the host cache are being copied into pinned rows"
        return "a layer call holds rows of the host cache"


# --------------------------------------------------------------------------
# The runtime: one per process (one rank per process)
# --------------------------------------------------------------------------


@dataclass
class _Buffers:
    """Pinned host copies of one layer's inputs and the host result."""

    ids: torch.Tensor
    weights: torch.Tensor
    x: torch.Tensor
    out: torch.Tensor
    # One output row per (token, top-k) pair the CPU computes, for a sum
    # that takes each pair's own row (VLLM_XPU_MOE_COMBINE rank/exact);
    # allocated on first use.
    pair_rows: torch.Tensor | None = None
    # With it: the pair map [tokens, top_k] that goes to the device with
    # the rows.
    row_map: torch.Tensor | None = None
    # ``x``'s bytes as int16 in numpy (the per-pair gather of its rows)
    x_i16: np.ndarray | None = None


@dataclass
class Pending:
    """A layer's inputs on their way to the host."""

    tiers: "ExpertTiers"
    num_tokens: int
    event: Any
    buffers: _Buffers
    waited: bool = False
    # whether the router weights were copied (the per-pair sum does not
    # read them: post(weights=False))
    weights: bool = True
    # The op's arith for this layer call (card_arith): the card kernel whose
    # arithmetic each GEMM takes.
    arith: int = 0
    # The layer's host calls, once listed (by `stage`), and the physical ids
    # staged to the card, which the CPU leaves out.
    selection: Any = None
    staged: np.ndarray | None = None
    # The host timeline of this layer call (debug part "cpustep"): host
    # times (perf_counter) by point, whether the card had reached the
    # routing when the host waited for it, and an event after the card's
    # own experts of the layer.
    times: dict[str, float] | None = None
    card_ahead: bool | None = None
    card_event: Any = None


@dataclass
class HostPairs:
    """The CPU's part of a layer for a per-pair sum: ``rows`` [n, hidden] on
    the device, one per pair the CPU computed, and ``row_map`` [tokens,
    top_k] (int32, -1 where the CPU computed nothing) on the device."""

    rows: torch.Tensor
    row_map: torch.Tensor


@dataclass
class CpuExpertStats:
    layers: int = 0  # MoE layer calls that had host calls
    idle_layers: int = 0  # MoE layer calls with none
    calls: int = 0  # (token, expert) pairs computed on the CPU
    experts: int = 0  # distinct experts per layer call, summed
    wait_ms: float = 0.0  # host waiting for the routed ids
    compute_ms: float = 0.0
    read_ms: float = 0.0  # host waiting for units from the drive
    promotions: int = 0
    demotions: int = 0
    max_calls: int = 0
    max_experts: int = 0  # most distinct experts one layer call needed
    per_tier: dict[str, int] = field(
        default_factory=lambda: {"card": 0, "pinned": 0, "cpu": 0, "staged": 0}
    )
    # prefill staging (stage_min_calls): layer calls that staged, experts
    # staged, calls they carried, host ms copying them into the rows
    stage_layers: int = 0
    stage_experts: int = 0
    stage_ms: float = 0.0
    # of which copying units into the rows; layer calls that staged by the
    # cost model (stage_min_tokens), and that it left on the CPU
    stage_copy_ms: float = 0.0
    stage_by_cost: int = 0
    stage_declined: int = 0


class StageCost:
    """What the staging rule weighs, from this tier's own measurements.

    CPU: one layer call's compute time ``y`` (µs) against its distinct
    experts ``E`` and calls ``N``, ``y = a * E + b * N``, a ridge fit (decayed
    sums, so it follows a change of threads or of the kernel) pulled towards
    the prior ``A0_US``, ``B0_US`` (an AVX2 host at 8 threads: about 100 µs
    per expert at one token, 34 µs per expert-token beyond) while the
    data cannot separate the two (every call one token per expert, as in
    decode). Copy: the measured µs per unit copied into the staging rows,
    a moving mean; before the first copy, the unit's bytes at 8 GB/s per
    copy thread up to 32 GB/s."""

    A0_US = 100.0
    B0_US = 34.0
    DECAY = 0.99
    PRIOR = 4.0  # weight of the prior, in layer calls of 10 experts

    def __init__(self) -> None:
        self.reset()
        self._copy_us: float | None = None

    def reset(self) -> None:
        # decayed sums of E*E, E*N, N*N, E*y, N*y
        self._s = [0.0] * 5
        self.samples = 0

    def observe(self, experts: int, calls: int, us: float) -> None:
        e, n, d = float(experts), float(calls), self.DECAY
        s = self._s
        s[0] = d * s[0] + e * e
        s[1] = d * s[1] + e * n
        s[2] = d * s[2] + n * n
        s[3] = d * s[3] + e * us
        s[4] = d * s[4] + n * us
        self.samples += 1

    def coeffs(self) -> tuple[float, float]:
        """(a, b) in µs: per expert and per call."""
        lam = self.PRIOR * 100.0  # (10 experts)^2 per prior call
        s = self._s
        xx = np.array([[s[0] + lam, s[1]], [s[1], s[2] + lam]])
        xy = np.array([s[3] + lam * self.A0_US, s[4] + lam * self.B0_US])
        a, b = np.linalg.solve(xx, xy)
        return max(float(a), 0.0), max(float(b), 0.0)

    def cpu_us(self, calls: np.ndarray) -> np.ndarray:
        a, b = self.coeffs()
        return a + b * np.asarray(calls, dtype=np.float64)

    def copied(self, units: int, us: float) -> None:
        if units <= 0:
            return
        per = us / units
        self._copy_us = (
            per if self._copy_us is None else 0.8 * self._copy_us + 0.2 * per
        )

    def copy_us(self, unit_bytes: int, threads: int) -> float:
        if self._copy_us is not None:
            return self._copy_us
        return unit_bytes / (min(8.0 * max(threads, 1), 32.0) * 1e3)

    def readout(self) -> dict[str, float]:
        a, b = self.coeffs()
        return {
            "cpu_us_per_expert": round(a, 1),
            "cpu_us_per_call": round(b, 2),
            "cpu_fit_samples": self.samples,
            "copy_us_per_unit": None
            if self._copy_us is None
            else round(self._copy_us, 1),
        }


class CpuExpertRuntime:
    """The CPU tier of one process: the host cache, the host op, the
    pinned staging buffers and the counters."""

    # VLLM_CPU_EXPERTS_ASYNC's host thread (set in __init__; None also for a
    # runtime made without it, as tests' stand-ins are)
    _async: "_AsyncHost | None" = None

    def __init__(
        self,
        config: "ExpertStoreConfig",
        store: Any,
        layout: UnitLayout,
        *,
        w4a8: bool = False,
    ):
        if not host_op_available():
            raise CpuExpertError(
                "--cpu-offload-pinned-gb computes experts on the CPU with the "
                "vllm-xpu-kernels host op int4_moe_cpu, which the installed "
                "kernels do not have: install kernels built with it, or drop "
                "--cpu-offload-pinned-gb"
            )
        if w4a8 and not host_op_a8_available():
            raise CpuExpertError(
                "the experts run W4A8 (int8 activations) on the card; computing "
                "them on the CPU needs the vllm-xpu-kernels host op "
                "int4_moe_cpu_a8, which the installed kernels do not have: "
                "install kernels built with it, or drop --cpu-offload-pinned-gb"
            )
        # The card's arithmetic class this tier follows: W4A8 (int8
        # activations) or W4A16.
        self.w4a8 = bool(w4a8)
        self.config = config
        self.layout = layout
        self.threads = int(config.cpu_threads)
        want = _ISA[config.cpu_isa]
        self.isa = _op_isa(want)
        if self.isa < 2 and config.cpu_isa == "auto":
            raise CpuExpertError(
                "this host has no AVX2 with FMA: the CPU expert tier would run "
                "its scalar code, hundreds of times slower. Set "
                "--expert-store-config cpu_isa to 'scalar' to accept that"
            )
        self.cores: list[int] = []
        self._configure_threads()
        rows = max(1, int(config.host_cache_gib * 2**30) // layout.row_bytes)
        self.cache = HostExpertCache(store, layout, rows, config.read_timeout_s)
        self.stats = CpuExpertStats()
        # The arith of the layer call being computed (card_arith); 0 when the
        # op has no such argument (it computes the GEMV's arithmetic).
        self.arith = 0
        # the staging rule's measurements (stage_min_tokens)
        self.cost = StageCost()
        self._buffers: _Buffers | None = None
        # the kernels' select op (VLLM_CPU_EXPERTS_SELECT), its kept outputs
        self._select_op = _select_op()
        self._select_out: tuple[Any, ...] | None = None
        # the op reads each pair's input row of x in place (no gathered copy)
        # (an op with it has arith too)
        self._x_row = not self.w4a8 and op_takes_arith() and op_takes_x_row()
        self._last_stats = time.monotonic()
        self._first_logged = False
        # the end of the last layer call's host part (debug part "cpustep")
        self._last_exit: float | None = None
        # (routing arrived, units present, op done) of the last call with CPU
        # calls (perf_counter)
        self._last_call: tuple[float, float, float] | None = None
        self._self_check()
        # VLLM_CPU_EXPERTS_ASYNC: the CPU part on a host thread of its own
        self._async: _AsyncHost | None = None
        settings = _async_settings()
        if settings is not None:
            self._async = _AsyncHost(self, *settings)
            logger.info(
                "CPU experts: a layer call's CPU part runs on a host thread of "
                "its own%s while the model's thread enqueues on; the card waits "
                "for the result rows by reading a host word (%s reads, at most "
                "%d; %s=spin, layer calls of at most %d tokens without prefill "
                "staging; the others as without it).",
                ""
                if self._async.core is None
                else f" (pinned to core {self._async.core})",
                "acquire" if self._async.atomic else "volatile",
                self._async.spins,
                _ASYNC_ENV,
                self._async.max_tokens,
            )
        logger.info(
            "CPU experts: %s kernel (%s), %d threads, host cache of %d experts "
            "(%.2f GiB of ordinary memory, %d B per expert); experts not in it "
            "are read from the expert store when a step routes to them.",
            _ISA_NAME[self.isa],
            "W4A8, int8 activations as the card's" if self.w4a8 else "W4A16",
            self.threads,
            rows,
            self.cache.gib,
            layout.row_bytes,
        )

    # -- settings a running engine may change (runtime control: cpu.set) -------

    def set_threads(self, threads: int) -> None:
        """Threads of the host op from the next call on (the op keeps one
        pool per thread count; pinned, at most one per core of the set)."""
        if threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        self.drain()
        old = self.threads
        self.config.cpu_threads = int(threads)
        self.threads = int(threads)
        if self.cores:
            self._configure_threads()
        if self.threads != old:
            self.cost.reset()  # the fit was of the old thread count

    def _configure_threads(self) -> None:
        """Pin the op's workers (``VLLM_CPU_EXPERTS_CORES``) and set their
        spin time (``VLLM_CPU_EXPERTS_SPIN_US``); neither set: as the op
        does by itself."""
        setting = os.environ.get(_CORES_ENV, "")
        spin_text = os.environ.get(_SPIN_ENV, "").strip()
        if not setting.strip() and not spin_text:
            return
        spin = int(spin_text) if spin_text else -1
        rank, ranks = _local_rank()
        want = int(self.config.cpu_threads)
        cores = cpu_cores_for_rank(setting, rank, ranks, want)
        if cores and want > len(cores):
            want = len(cores)  # one thread per core
        if not _op_configure(cores, spin):
            raise CpuExpertError(
                f"{_CORES_ENV} / {_SPIN_ENV} need the host op's "
                "int4_moe_cpu_configure, which the installed kernels do not "
                "have: install kernels built with it, or unset them"
            )
        self.cores = cores
        self.threads = want
        logger.info_once(
            "CPU experts: local rank %d of %d, %d op threads%s, workers spin "
            "%s after a phase.",
            rank,
            ranks,
            want,
            f" pinned to cores {cores}" if cores else " (not pinned)",
            f"{spin} us" if spin >= 0 else "the op's default",
        )

    def resize_cache(self, gib: float) -> None:
        """The host cache to ``gib`` (rows of whole experts), between two
        steps: the most recently used experts stay as far as they fit."""
        self.drain()
        rows = self.cache.rows_for_gib(gib)
        old = self.cache.num_rows
        self.cache.resize(rows)
        self.config.host_cache_gib = float(gib)
        logger.info(
            "CPU experts: host cache %d -> %d experts (%.2f GiB of ordinary memory).",
            old,
            rows,
            self.cache.gib,
        )

    # -- the op ---------------------------------------------------------------

    def run_op(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
        units: torch.Tensor,
        call_unit: torch.Tensor,
        call_token: torch.Tensor,
        call_weight: torch.Tensor,
        x_row: torch.Tensor | None = None,
    ) -> None:
        """The host op over the calls; ``x_row`` (an op with ``call_x_row``,
        W4A16): call c reads x's row x_row[c] and writes out's row
        call_token[c]."""
        if self.w4a8:
            # One arithmetic for every layer call (the card's W4A8 GEMM has
            # no GEMV); ``arith`` is the W4A16 op's.
            _op_run_a8(
                out,
                x,
                units,
                call_unit,
                call_token,
                call_weight,
                self.layout.inter,
                self.layout.group,
                self.layout.offsets,
                W4A8_CARD_SCHEME,
                self.threads,
                self.isa,
            )
            return
        args = [
            out,
            x,
            units,
            call_unit,
            call_token,
            call_weight,
            self.layout.inter,
            self.layout.group,
            self.layout.offsets,
            self.threads,
            self.isa,
        ]
        if x_row is not None:  # positional, after arith (op_takes_x_row)
            args += [self.arith, x_row]
        elif self.arith:
            args.append(self.arith)
        _op_run(*args)

    def _self_check(self) -> None:
        """The host op against the card's arithmetic in torch, at this
        model's shapes: two random experts, three tokens. Refuses to start
        beyond one bf16 step at the output's scale."""
        L = self.layout
        gen = torch.Generator().manual_seed(20261113)
        units = torch.randint(
            0, 256, (2, L.row_bytes), dtype=torch.uint8, generator=gen
        )
        for name in ("w13_scales", "w2_scales"):
            offset = L.offsets[UNIT_TENSORS.index(name)]
            count = L.nbytes[name] // 2
            scales = (torch.randn(2, count, generator=gen) * 0.01).to(torch.bfloat16)
            units[:, offset : offset + count * 2] = scales.view(torch.uint8)
        x = torch.randn(3, L.hidden, generator=gen).to(torch.bfloat16)
        calls = [(0, 0, 0.6), (1, 0, 0.4), (1, 1, 1.0), (0, 2, 0.25), (1, 2, 0.75)]
        out = torch.empty_like(x)
        self.run_op(
            out,
            x,
            units,
            torch.tensor([c[0] for c in calls], dtype=torch.int32),
            torch.tensor([c[1] for c in calls], dtype=torch.int32),
            torch.tensor([c[2] for c in calls], dtype=torch.float32),
        )
        tensors = {
            u: {
                name: units[u, off : off + L.nbytes[name]]
                for name, off in zip(UNIT_TENSORS, L.offsets)
            }
            for u in (0, 1)
        }
        reference = reference_w4a8_experts if self.w4a8 else reference_int4_experts
        want = reference(x, tensors, calls, L.inter, L.group)
        steps = bf16_steps(out, want)
        differ = int((out.view(torch.int16) != want.view(torch.int16)).sum())
        grouped_steps = None
        if not self.w4a8 and op_takes_arith() and self.config.cpu_arith == "card":
            # The grouped GEMM's arithmetic (every call of a top-10 model).
            grouped = torch.empty_like(x)
            self.arith = 3
            try:
                self.run_op(
                    grouped,
                    x,
                    units,
                    torch.tensor([c[0] for c in calls], dtype=torch.int32),
                    torch.tensor([c[1] for c in calls], dtype=torch.int32),
                    torch.tensor([c[2] for c in calls], dtype=torch.float32),
                )
            finally:
                self.arith = 0
            grouped_want = reference_int4_experts(
                x, tensors, calls, L.inter, L.group, arith=3
            )
            grouped_steps = bf16_steps(grouped, grouped_want)
            steps = max(steps, grouped_steps)
        if 2 <= self.isa != _ISA_CARD_CLASS:
            # The vector code must take the scalar code's steps in its order
            # (the build keeps fp contraction and reassociation off): the two
            # agree bit for bit, or the build is not the one tested. (Level 8
            # differs from the scalar code at prefill on purpose; its check
            # is the one against the card's arithmetic below.)
            scalar = torch.empty_like(x)
            isa, self.isa = self.isa, 1
            try:
                self.run_op(
                    scalar,
                    x,
                    units,
                    torch.tensor([c[0] for c in calls], dtype=torch.int32),
                    torch.tensor([c[1] for c in calls], dtype=torch.int32),
                    torch.tensor([c[2] for c in calls], dtype=torch.float32),
                )
            finally:
                self.isa = isa
            if not torch.equal(scalar.view(torch.int16), out.view(torch.int16)):
                raise CpuExpertError(
                    f"the CPU expert kernel's {_ISA_NAME[self.isa]} code and "
                    "its scalar code disagree: the kernels were built with "
                    "floating-point contraction or reassociation on; refusing "
                    "to compute experts with it"
                )
        if not steps <= 1.0:
            raise CpuExpertError(
                f"the CPU expert kernel ({_ISA_NAME[self.isa]}) differs from the "
                f"card's arithmetic by {steps:.2f} bf16 steps on random experts: "
                "refusing to compute experts with it"
            )
        if self.w4a8 and differ:
            # The W4A8 op reproduces the reference bit for bit (integer dots,
            # the same fp32 chain, the same exp): any difference is a build
            # or a scheme that is not the card's.
            raise CpuExpertError(
                f"the CPU expert kernel's W4A8 path ({_ISA_NAME[self.isa]}) "
                f"differs from the card's W4A8 arithmetic in {differ} of "
                f"{out.numel()} values on random experts: refusing to compute "
                "experts with it"
            )
        logger.info(
            "CPU experts: self-check against the card's %s arithmetic passed "
            "(%s, %d of %d values differ, largest difference %.3f bf16 steps "
            "at the output's scale%s%s).",
            "W4A8" if self.w4a8 else "W4A16",
            _ISA_NAME[self.isa],
            differ,
            out.numel(),
            steps,
            f"; {_ISA_NAME[self.isa]} and scalar code bit-identical"
            if 2 <= self.isa != _ISA_CARD_CLASS
            else "",
            "; the card's W4A8 GEMM for every call"
            if self.w4a8
            else "; the card's grouped GEMM and GEMV, chosen per layer call"
            if grouped_steps is not None
            else "; the GEMV's arithmetic for every call (cpu_arith gemv)"
            if op_takes_arith()
            else "; the op has no arith argument: the card's GEMV arithmetic for "
            "every call, which differs from its grouped GEMM (every call of a "
            "model with top-k above 8) in most values",
        )

    # -- a layer ----------------------------------------------------------------

    def _buffers_for(self, tokens: int, topk: int, hidden: int) -> _Buffers:
        from vllm.utils.platform_utils import is_pin_memory_available

        b = self._buffers
        if (
            b is None
            or b.x.shape[0] < tokens
            or b.ids.shape[1] != topk
            or b.x.shape[1] != hidden
        ):
            n = max(tokens, 1 if b is None else b.x.shape[0])
            pin = is_pin_memory_available()
            b = _Buffers(
                ids=torch.empty((n, topk), dtype=torch.int32, pin_memory=pin),
                weights=torch.empty((n, topk), dtype=torch.float32, pin_memory=pin),
                x=torch.empty((n, hidden), dtype=torch.bfloat16, pin_memory=pin),
                out=torch.empty((n, hidden), dtype=torch.bfloat16, pin_memory=pin),
            )
            b.x_i16 = b.x.view(torch.int16).numpy()
            self._buffers = b
        return b

    def post(
        self,
        tiers: "ExpertTiers",
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: bool = True,
    ) -> Pending:
        """Enqueue the copies of a layer's routing and inputs to the host,
        before the card's own experts of the layer. ``weights`` False: the
        router weights stay on the device (:meth:`finish_pairs` and
        :meth:`stage` do not read them; :meth:`finish` does)."""
        if (
            hidden_states.device.type != "cpu"
            and torch.accelerator.current_stream().is_capturing()
        ):
            raise CpuExpertError(
                f"{tiers.layer_name}: the MoE layer with CPU experts is being "
                "captured into a graph; it must run outside the captured "
                "graphs (the config splits the graphs at the MoE op when this "
                "is on, and a capture path that ignores that cannot run it)"
            )
        if self._async is not None:
            self._async.check()
        times = None
        if runtime_debug.on("cpustep"):
            times = {"post": time.perf_counter()}
        tokens, topk = topk_ids.shape
        b = self._buffers_for(tokens, topk, hidden_states.shape[1])
        b.ids[:tokens].copy_(topk_ids, non_blocking=True)
        if weights:
            b.weights[:tokens].copy_(topk_weights, non_blocking=True)
        b.x[:tokens].copy_(hidden_states, non_blocking=True)
        mod = torch.get_device_module(hidden_states.device)
        event = mod.Event()
        event.record(mod.current_stream())
        L = self.layout
        arith = (
            card_arith(tokens * topk, L.hidden, L.inter, L.group)
            if not self.w4a8 and op_takes_arith() and self.config.cpu_arith == "card"
            else 0
        )
        pending = Pending(tiers, tokens, event, b, arith=arith, weights=weights)
        if times is not None:
            times["posted"] = time.perf_counter()
            pending.times = times
        return pending

    # -- where the CPU part runs (VLLM_CPU_EXPERTS_ASYNC) ----------------------

    def takes_async(self, pending: Pending) -> bool:
        """Whether this layer call's CPU part runs on the host thread
        (:meth:`finish_pairs_async`): VLLM_CPU_EXPERTS_ASYNC=spin, at most its
        token bound, and no prefill staging for it (staging lists the calls
        on the model's thread)."""
        a = self._async
        if a is None or pending.num_tokens > a.max_tokens:
            return False
        if int(self.config.stage_min_calls) > 0:
            return False
        min_tokens = int(getattr(self.config, "stage_min_tokens", 0))
        return not 0 < min_tokens <= pending.num_tokens

    def finish_pairs_async(self, pending: Pending, device: torch.device) -> HostPairs:
        """:meth:`finish_pairs` without waiting: the host thread computes the
        rows; the card waits for them (stream order) before its copies of the
        rows and the map. Every pair's row slot is copied (the map says which
        hold a CPU result: -1 where none, the whole map when the CPU computes
        nothing), so the per-pair sum reads what it reads on the model's
        thread."""
        assert self._async is not None
        host = self._async.submit(pending, device)
        if pending.times is not None:
            pending.times["enqueued"] = time.perf_counter()
            self._emit_step(pending)
        return host

    def drain(self) -> None:
        """Wait until the host thread finished every layer call handed to it
        (nothing without it); raises if one failed."""
        if self._async is not None:
            self._async.drain()

    def card_enqueued(self, pending: Pending, device: torch.device) -> None:
        """The card's own experts of the layer are enqueued (debug part
        "cpustep": the time, and an event after them on ``device``)."""
        if pending.times is None:
            return
        pending.times["card"] = time.perf_counter()
        if device.type != "cpu":
            mod = torch.get_device_module(device)
            ev = mod.Event()
            ev.record(mod.current_stream())
            pending.card_event = ev

    def _emit_step(self, pending: Pending) -> None:
        """One "cpustep" line: where the host's time of this layer call went
        (ms), and what the card was doing at the two points the host can
        see without waiting."""
        t = pending.times
        assert t is not None
        last = self._last_exit
        end = t.get("end", t.get("enqueued", t["posted"]))
        self._last_exit = end

        def span(a: str, b: str) -> float:
            return (t[b] - t[a]) * 1e3 if a in t and b in t else float("nan")

        runtime_debug.emit(
            "cpustep",
            logger,
            "%s: %d tokens, %s; host ms: since the last layer call's host part "
            "%.3f, post %.3f, card's experts enqueued %.3f, %s; card at the "
            "routing when the host waited: %s; card's experts done when the "
            "CPU's were: %s",
            pending.tiers.layer_name,
            pending.num_tokens,
            "handed to the host thread"
            if "enqueued" in t
            else f"{t.get('calls', 0)} CPU calls",
            float("nan") if last is None else (t["post"] - last) * 1e3,
            span("post", "posted"),
            span("posted", "card"),
            f"handing over and the card's wait {span('card', 'enqueued'):.3f}"
            if "enqueued" in t
            else (
                f"wait {span('wait', 'waited'):.3f}, list+cache "
                f"{span('waited', 'listed'):.3f}, op {span('listed', 'computed'):.3f}, "
                f"results {span('computed', 'end'):.3f}"
            ),
            "n/a" if pending.card_ahead is None else pending.card_ahead,
            "n/a" if "card_done" not in t else bool(t["card_done"]),
        )

    def _emit_host_thread(self, pending: Pending, pair: np.ndarray | None) -> None:
        """The host thread's "cpustep" line of a layer call it computed."""
        t = pending.times
        assert t is not None
        runtime_debug.emit(
            "cpustep",
            logger,
            "%s: host thread: %d CPU calls; ms: from handed to the routing's "
            "arrival %.3f, list+cache %.3f, op %.3f; card at the routing when "
            "the host thread began: %s; card's experts done when the CPU's "
            "were: %s",
            pending.tiers.layer_name,
            0 if pair is None else len(pair),
            (t["waited"] - t.get("handed", t["wait"])) * 1e3,
            (t["listed"] - t["waited"]) * 1e3,
            (t["computed"] - t["listed"]) * 1e3,
            "n/a" if pending.card_ahead is None else pending.card_ahead,
            "n/a" if "card_done" not in t else bool(t["card_done"]),
        )

    def finish(self, pending: Pending, device: torch.device) -> torch.Tensor | None:
        """Wait for the routing only, compute the layer's host calls and
        return their summed contribution on the device (None: no host call).
        Raises :class:`CpuExpertError` when an expert cannot be served."""
        self.drain()
        n, b = pending.num_tokens, pending.buffers
        if not pending.weights:
            raise CpuExpertError(
                f"{pending.tiers.layer_name}: the per-token sum needs the router "
                "weights, which this layer call did not copy (post(weights=False))"
            )
        self._wait(pending)
        self.arith = pending.arith
        if not self.compute(
            pending.tiers, b.ids[:n], b.weights[:n], b.x[:n], b.out[:n],
            selection=pending.selection, staged=pending.staged,
        ):  # fmt: skip
            return None
        return b.out[:n].to(device, non_blocking=True)

    def stage(self, pending: Pending, device: torch.device) -> Any:
        """Prefill staging: wait for the routing, and copy CPU-tier experts
        this layer call routes to into the staging rows, with their map.
        Returns the layer's kernel call over the rows for the caller to run on
        the card, or None (nothing staged); the CPU then computes only the
        rest (:meth:`finish`).

        Which experts: in a layer call of at least ``stage_min_tokens`` tokens
        (> 0), the set the cost model gives (:meth:`_staged_by_cost`);
        otherwise, with ``stage_min_calls`` > 0, the experts routed at least
        that many tokens (most calls first, as far as the rows go). Both 0:
        nothing, without waiting."""
        threshold = int(self.config.stage_min_calls)
        min_tokens = int(getattr(self.config, "stage_min_tokens", 0))
        n, b = pending.num_tokens, pending.buffers
        by_cost = 0 < min_tokens <= n
        if threshold <= 0 and not by_cost:
            return None
        self._wait(pending)
        tiers = pending.tiers
        ids = b.ids[:n].numpy()
        selection, _ = self._selection(tiers, b.ids[:n], ids, None, None)
        pending.selection = selection
        n_calls = int(selection.pair.size)
        if not n_calls or (not by_cost and n_calls < threshold):
            return None
        start = time.perf_counter()
        phys = ids.reshape(-1)[selection.pair]
        uniq, counts = np.unique(phys, return_counts=True)
        rows = None
        if by_cost:
            rows = tiers.stage_rows(device)
            heavy, plan = self._staged_by_cost(uniq, counts, rows.num_rows)
            rule = f"by the cost model ({plan})"
        else:
            heavy = uniq[counts >= threshold]
            rule = f"with at least {threshold} calls each"
        if heavy.size == 0:
            if by_cost:
                self.stats.stage_declined += 1
            return None
        if rows is None:
            rows = tiers.stage_rows(device)
        if not by_cost and heavy.size > rows.num_rows:  # the busiest ones
            by_calls = counts[counts >= threshold]
            heavy = heavy[np.argsort(-by_calls, kind="stable")[: rows.num_rows]]
        assert tiers.slot_logical is not None and tiers.store_binding is not None
        layer = tiers.store_binding.store_layer
        heavy_keys = [
            (layer, int(tiers.slot_logical[tiers._slot_of_global[p]])) for p in heavy
        ]
        cache_rows = self.cache.acquire(heavy_keys)
        copied = time.perf_counter()
        try:
            self._copy_to_stage(cache_rows, rows)
        finally:
            self.cache.release(cache_rows)
        copy_s = time.perf_counter() - copied
        map_host = rows.map_host.numpy()
        map_host.fill(-1)
        map_host[heavy] = np.arange(heavy.size, dtype=np.int32)
        rows.map.copy_(rows.map_host, non_blocking=True)
        pending.staged = heavy
        st = self.stats
        staged_calls = int(np.isin(phys, heavy).sum())
        st.stage_layers += 1
        st.stage_experts += int(heavy.size)
        st.per_tier["staged"] += staged_calls
        st.stage_ms += (time.perf_counter() - start) * 1e3
        st.stage_copy_ms += copy_s * 1e3
        self.cost.copied(int(heavy.size), copy_s * 1e6)
        if by_cost:
            st.stage_by_cost += 1
            logger.info_once(
                "CPU experts: a layer call of %d tokens (stage_min_tokens %d) "
                "staged CPU-tier experts to the card by the cost model "
                "(%s: %d of %d experts, %d calls; the CPU computes the other %d; "
                "copies %.2f ms on %d threads).",
                n,
                min_tokens,
                tiers.layer_name,
                heavy.size,
                uniq.size,
                staged_calls,
                n_calls - staged_calls,
                copy_s * 1e3,
                self._copy_threads(),
            )
        else:
            logger.info_once(
                "CPU experts: a large step's CPU-tier experts were staged to the "
                "card (%s: %d experts with at least %d calls each, %d calls; the "
                "CPU computes the other %d).",
                tiers.layer_name,
                heavy.size,
                threshold,
                staged_calls,
                n_calls - staged_calls,
            )
        if runtime_debug.on("cpu"):
            runtime_debug.emit(
                "cpu",
                logger,
                "%s: %d tokens, staged %d of %d CPU-tier experts %s, %d of %d "
                "calls; copies %.2f ms",
                tiers.layer_name,
                n,
                heavy.size,
                uniq.size,
                rule,
                staged_calls,
                n_calls,
                copy_s * 1e3,
            )
        return tiers.stage_impl(rows)

    def _copy_threads(self) -> int:
        return max(1, int(getattr(self.config, "stage_copy_threads", 1)))

    def _copy_to_stage(self, cache_rows: Sequence[int], rows: Any) -> None:
        """The units in host cache rows ``cache_rows`` into staging rows
        ``0..len-1``, tensor by tensor, split over ``stage_copy_threads``
        threads (each a contiguous range of the rows; torch's copies leave
        the GIL, so the threads copy at once)."""
        L = self.layout
        n = len(cache_rows)
        index = torch.tensor(list(cache_rows), dtype=torch.int64)

        def part(lo: int, hi: int) -> None:
            idx = index[lo:hi]
            for name, offset in zip(UNIT_TENSORS, L.offsets):
                src = self.cache.slab[:, offset : offset + L.nbytes[name]]
                torch.index_select(src, 0, idx, out=rows.rows_u8(name)[lo:hi])

        threads = min(self._copy_threads(), n)
        if threads <= 1:
            part(0, n)
            return
        bounds = np.linspace(0, n, threads + 1).astype(int)
        pool = self._copy_pool(threads)
        futures = [
            pool.submit(part, int(lo), int(hi))
            for lo, hi in zip(bounds[:-1], bounds[1:])
            if hi > lo
        ]
        for f in futures:
            f.result()

    def _copy_pool(self, threads: int) -> Any:
        from concurrent.futures import ThreadPoolExecutor

        pool = getattr(self, "_stage_pool", None)
        if pool is None or pool._max_workers != threads:
            if pool is not None:
                pool.shutdown(wait=True)
            pool = ThreadPoolExecutor(threads, thread_name_prefix="cpu-stage")
            self._stage_pool = pool
        return pool

    def _staged_by_cost(
        self, experts: np.ndarray, calls: np.ndarray, num_rows: int
    ) -> tuple[np.ndarray, str]:
        """Which of a layer call's CPU-tier ``experts`` (with their ``calls``)
        go to the card through the staging rows: the busiest ``k`` for the
        ``k`` that minimises the layer's host-side time

            S(k) + max(G(k), C(k))

        where the host first copies the k units into the rows (S, measured
        per expert, the copies spread over ``stage_copy_threads``), then
        computes the rest on the CPU (C = sum of a + b * calls over the
        experts left, a and b fitted to this tier's own measured calls) while
        the card reads the k from the rows through UVA (G = k *
        ``stage_card_us``). S overlaps the card's own experts of the layer
        and G the CPU's work, which is why the two are not simply added; S is
        charged in full (the card's own time is not seen from the host), so
        the rule errs towards the CPU."""
        order = np.argsort(-calls, kind="stable")
        cpu_us = self.cost.cpu_us(calls[order])  # per expert, busiest first
        total = float(cpu_us.sum())
        limit = min(int(num_rows), len(order))
        k = np.arange(limit + 1)
        left = total - np.concatenate(([0.0], np.cumsum(cpu_us[:limit])))
        copy_us = self.cost.copy_us(self.layout.row_bytes, self._copy_threads())
        card_us = float(getattr(self.config, "stage_card_us", 100.0))
        objective = k * copy_us + np.maximum(k * card_us, left)
        best = int(np.argmin(objective))  # the first minimum: fewest staged
        plan = (
            f"{best} staged: host {objective[best] / 1e3:.2f} ms against "
            f"{total / 1e3:.2f} ms all on the CPU"
        )
        return experts[order[:best]], plan

    def finish_pairs(self, pending: Pending, device: torch.device) -> HostPairs | None:
        """:meth:`finish` for a sum that takes each pair's own row (the
        combine modes rank and exact, and the default with this tier): one
        output row per (token, top-k) pair the CPU computes, with its map;
        pairs of experts :meth:`stage` sent to the card are left out. None:
        no host call."""
        self.drain()
        n, b = pending.num_tokens, pending.buffers
        topk = b.ids.shape[1]
        self._pair_buffers(b, n * topk)
        assert b.pair_rows is not None and b.row_map is not None
        pair = self._pairs_into_buffers(pending)
        if pair is None:
            out = None
        else:
            out = HostPairs(
                b.pair_rows[: len(pair)].to(device, non_blocking=True),
                b.row_map[:n].to(device, non_blocking=True),
            )
        if pending.times is not None:
            pending.times["end"] = time.perf_counter()
            pending.times["calls"] = 0 if pair is None else len(pair)
            self._emit_step(pending)
        return out

    def _pairs_into_buffers(
        self, pending: Pending, fill_map: bool = False
    ) -> np.ndarray | None:
        """Wait for the layer call's routing, compute its CPU calls into the
        pinned pair rows (call ``i`` in row ``i``) and write the pair map
        (``fill_map``: also when there is no call: all -1). Returns the flat
        pair index of each call, or None (no call). On the model's thread
        (:meth:`finish_pairs`) or the host thread (VLLM_CPU_EXPERTS_ASYNC)."""
        n, b = pending.num_tokens, pending.buffers
        t = pending.times
        if t is not None:
            t["wait"] = time.perf_counter()
            event_query = getattr(pending.event, "query", None)
            if event_query is not None:
                pending.card_ahead = bool(event_query())
        self._wait(pending)
        self.arith = pending.arith
        assert b.pair_rows is not None and b.row_map is not None
        # The map goes to the reused pinned buffer: written after this layer's
        # routing arrived, so after the previous layer's copy of it left.
        row_map = b.row_map[:n]
        pair, mapped = self._compute_pairs(
            pending.tiers, b.ids[:n], b.x[:n], b.pair_rows,
            pending.staged, pending.selection, row_map,
            None if b.x_i16 is None else b.x_i16[:n],
        )  # fmt: skip
        if t is not None:
            t["waited"], t["listed"], t["computed"] = self._call_times(t)
            if pending.card_event is not None:
                t["card_done"] = float(pending.card_event.query())
        if pair is None:
            if fill_map:
                row_map.numpy().fill(-1)
            return None
        if not mapped:
            flat = row_map.numpy().reshape(-1)
            flat.fill(-1)
            flat[pair] = self._iota(len(pair)).numpy()
        return pair

    def _call_times(self, t: dict[str, float]) -> tuple[float, float, float]:
        """(routing arrived, units present, op done) of the last layer call
        (perf_counter; the wait's end when the call had no CPU calls)."""
        last = self._last_call
        now = time.perf_counter()
        if last is None or last[0] < t["wait"]:
            return now, now, now
        return last

    def _pair_buffers(self, b: _Buffers, pairs: int) -> None:
        """The per-pair host buffers of ``b`` for at least ``pairs`` pairs
        (pinned where the platform pins; grown, never shrunk)."""
        from vllm.utils.platform_utils import is_pin_memory_available

        if b.pair_rows is not None and b.pair_rows.shape[0] >= pairs:
            return
        topk, hidden = b.ids.shape[1], b.x.shape[1]
        tokens = max(-(-pairs // topk), b.x.shape[0])
        pin = is_pin_memory_available()
        b.pair_rows = torch.empty(
            (tokens * topk, hidden), dtype=torch.bfloat16, pin_memory=pin
        )
        b.row_map = torch.empty((tokens, topk), dtype=torch.int32, pin_memory=pin)

    def _pair_x(self, x: torch.Tensor, n: int) -> tuple[torch.Tensor, Any]:
        """A kept host buffer for ``n`` input rows like ``x``'s (the op's
        input of a per-pair call), grown by a quarter when too small; with
        its bytes as int16 in numpy when ``x`` is bf16 (else None)."""
        buf = getattr(self, "_pair_x_t", None)
        if (
            buf is None
            or buf.shape[0] < n
            or buf.shape[1:] != x.shape[1:]
            or buf.dtype != x.dtype
        ):
            old = 0 if buf is None or buf.dtype != x.dtype else buf.shape[0]
            rows = max(n, old + old // 4, 64)
            buf = torch.empty((rows, *x.shape[1:]), dtype=x.dtype)
            self._pair_x_t = buf
            self._pair_x_np = (
                buf.view(torch.int16).numpy() if x.dtype == torch.bfloat16 else None
            )
        view = self._pair_x_np
        return buf[:n], None if view is None else view[:n]

    def _row_table(self, n: int) -> np.ndarray:
        """A kept int32 table of at least ``n`` entries, logical expert ->
        host cache row of the running call (only its entries are read)."""
        table = getattr(self, "_row_table_t", None)
        if table is None or table.shape[0] < n:
            table = np.zeros(max(n, 256), dtype=np.int32)
            self._row_table_t = table
        return table

    def _iota(self, n: int) -> torch.Tensor:
        """The int32 values 0..n-1 (a slice of one kept tensor)."""
        iota = getattr(self, "_iota_t", None)
        if iota is None or iota.shape[0] < n:
            iota = torch.arange(max(n, 1024), dtype=torch.int32)
            self._iota_t = iota
        return iota[:n]

    def _ones(self, n: int) -> torch.Tensor:
        """Ones as fp32, ``n`` of them (a slice of one kept tensor; the op
        only reads it)."""
        ones = getattr(self, "_ones_t", None)
        if ones is None or ones.shape[0] < n:
            ones = torch.ones(max(n, 1024), dtype=torch.float32)
            self._ones_t = ones
        return ones[:n]

    def _wait(self, pending: Pending) -> None:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        if pending.waited:
            return
        start = time.perf_counter()
        with gpu_sync_allowed():
            pending.event.synchronize()
        pending.waited = True
        self.stats.wait_ms += (time.perf_counter() - start) * 1e3

    def compute(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        weights: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        selection: "HostSelection | None" = None,
        staged: np.ndarray | None = None,
    ) -> bool:
        """On the host: the calls of ``ids`` [T, K] (with ``weights``) that
        this rank computes on the CPU, applied to ``x`` [T, H] and summed per
        token into ``out`` (overwritten). False when there is none (``out``
        untouched). ``selection``: the layer's host calls if already listed;
        ``staged``: physical ids the card computes from staging rows, left
        out here."""
        waited = time.perf_counter()
        ids_np = ids.numpy()
        sel, _ = self._selection(tiers, ids, ids_np, selection, staged)
        topk = ids_np.shape[1]
        tokens = (sel.pair // topk).astype(np.int32)
        call_w = weights.numpy().reshape(-1)[sel.pair].astype(np.float32, copy=False)
        return self._run_calls(
            tiers, sel, x, out, torch.from_numpy(tokens),
            torch.from_numpy(call_w), ids.shape[0], waited,
        )  # fmt: skip

    def compute_pairs(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        x: torch.Tensor,
        out_rows: torch.Tensor,
        staged: np.ndarray | None = None,
        selection: "HostSelection | None" = None,
    ) -> np.ndarray | None:
        """On the host: each call of ``ids`` [T, K] this rank computes on the
        CPU, as its own row of ``out_rows`` (call ``i`` in row ``i``, the
        expert's bf16 output without the router weight: a call alone in its
        row with weight 1 rounds to exactly that). Returns the flat pair
        index ``token * K + k`` of each call, or None when there is none.
        ``staged``: physical ids the card computes from staging rows, left
        out here (counted as staged); ``selection``: the layer's host calls
        if already listed."""
        pair, _ = self._compute_pairs(tiers, ids, x, out_rows, staged, selection)
        return None if pair is None else pair.copy()

    def _compute_pairs(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        x: torch.Tensor,
        out_rows: torch.Tensor,
        staged: np.ndarray | None,
        selection: "HostSelection | None",
        row_map: torch.Tensor | None = None,
        x_i16: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, bool]:
        """:meth:`compute_pairs`, and whether ``row_map`` (given: [T, K]
        int32) was written as the pair map (the select op writes it as it
        lists the calls). The pair array is valid until the next listing.
        ``x_i16``: ``x``'s bytes as int16 in numpy (bf16 ``x``), for the
        gather of the calls' input rows in numpy."""
        waited = time.perf_counter()
        ids_np = ids.numpy()
        sel, mapped = self._selection(
            tiers, ids, ids_np, selection, staged, row_map if staged is None else None
        )
        pair = sel.pair
        n = int(pair.size)
        tokens = pair // ids_np.shape[1]
        if n and self._x_row and x.dtype == torch.bfloat16:
            # the op reads each call's row of x in place
            ran = self._run_calls(
                tiers, sel, x, out_rows[:n], self._iota(n), self._ones(n),
                ids.shape[0], waited, torch.from_numpy(tokens.astype(np.int32)),
            )  # fmt: skip
            return (pair, mapped) if ran else (None, False)
        if not n:
            xs = x[:0]
        elif x_i16 is not None and x.dtype == torch.bfloat16:
            xs, xs_i16 = self._pair_x(x, n)
            # tokens < T by construction: no bounds check needed (clip)
            np.take(x_i16, tokens, axis=0, out=xs_i16, mode="clip")
        else:
            xs = torch.index_select(
                x, 0, torch.from_numpy(tokens), out=self._pair_x(x, n)[0]
            )
        ran = self._run_calls(
            tiers, sel, xs, out_rows[:n], self._iota(n), self._ones(n),
            ids.shape[0], waited,
        )  # fmt: skip
        return (pair, mapped) if ran else (None, False)

    def _selection(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        ids_np: np.ndarray,
        selection: "HostSelection | None",
        staged: np.ndarray | None,
        row_map: torch.Tensor | None = None,
    ) -> tuple["HostSelection", bool]:
        """The layer's host calls (listed now unless given), without the
        calls of ``staged`` physical ids, which are counted as staged; and
        whether ``row_map`` was written (only by the select op, only when
        listing now)."""
        mapped = False
        if selection is not None:
            sel = selection
        else:
            sel = self._select(tiers, ids, ids_np, row_map)
            if sel is None:
                sel = tiers.host_selection(ids_np)
            else:
                mapped = row_map is not None
        if staged is not None and sel.pair.size:
            keep = ~np.isin(ids_np.reshape(-1)[sel.pair], staged)
            dropped = int(sel.pair.size - np.count_nonzero(keep))
            if dropped:
                counts = dict(sel.counts, cpu=sel.counts["cpu"] - dropped)
                sel = type(sel)(sel.pair[keep], sel.logical[keep], sel.layer, counts)
        return sel, mapped

    def _select(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        ids_np: np.ndarray,
        row_map: torch.Tensor | None,
    ) -> "HostSelection | None":
        """The layer's host calls listed by the kernels' select op in one
        pass (None: the op is not loaded, switched off, or the layer's tables
        are not in the op's types; the caller lists them with numpy, the same
        lists)."""
        op = self._select_op
        if op is None:
            return None
        tables = _select_tables(tiers)
        if tables is None:
            return None
        n = ids_np.size
        out = self._select_out
        if out is None or out[0].shape[0] < n:
            size = max(n, 1024)
            ts = [torch.empty(size, dtype=torch.int64) for _ in range(3)]
            out = self._select_out = (*ts, *(t.numpy() for t in ts))
        if not ids.is_contiguous():
            ids = ids.contiguous()
        card, pinned, host, calls, distinct, bad = op(
            ids, *tables, out[0], out[1], out[2],
            _NO_MAP if row_map is None else row_map,
        )  # fmt: skip
        if bad >= 0:
            raise CpuExpertError(
                f"{tiers.layer_name}: a routed expert id is outside the "
                f"{tiers.num_global} physical experts (min {int(ids_np.min())}, "
                f"max {int(ids_np.max())})"
            )
        from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
            HostSelection,
        )

        binding = tiers.store_binding
        return HostSelection(
            out[3][:calls],
            out[4][:calls],
            0 if binding is None else binding.store_layer,
            {"card": card, "pinned": pinned, "cpu": host},
            distinct=out[5][:distinct].tolist(),
        )

    def _run_calls(
        self,
        tiers: "ExpertTiers",
        sel: "HostSelection",
        x: torch.Tensor,
        out: torch.Tensor,
        tokens: torch.Tensor,
        call_w: torch.Tensor,
        n: int,
        waited: float,
        x_row: torch.Tensor | None = None,
    ) -> bool:
        st = self.stats
        per_tier = st.per_tier
        for tier, count in sel.counts.items():
            per_tier[tier] += count
        calls = int(sel.logical.size)
        if not calls:
            st.idle_layers += 1
            self._maybe_log()
            return False
        # Each expert of the call once, in order of first use (the order the
        # cache saw the per-call list in): one batched acquire; each call's
        # row then by a gather through a table indexed by logical expert.
        if sel.distinct is not None:
            uniq = sel.distinct
        elif calls <= _FEW_CALLS:
            uniq = list(dict.fromkeys(sel.logical.tolist()))
        else:
            u, first = np.unique(sel.logical, return_index=True)
            uniq = u[np.argsort(first, kind="stable")].tolist()
        distinct = len(uniq)
        layer = sel.layer
        cache_rows = self.cache.acquire([(layer, e) for e in uniq])
        try:
            read_done = time.perf_counter()
            table = self._row_table(max(uniq) + 1)
            table[uniq] = cache_rows
            call_rows = table[sel.logical]
            self.run_op(
                out,
                x,
                self.cache.slab,
                torch.from_numpy(call_rows),
                tokens,
                call_w,
                x_row,
            )
        finally:
            self.cache.release(cache_rows)
        done = time.perf_counter()
        self._last_call = (waited, read_done, done)
        st.layers += 1
        st.calls += calls
        if calls > st.max_calls:
            st.max_calls = calls
        st.experts += distinct
        if distinct > st.max_experts:
            st.max_experts = distinct
        st.read_ms += (read_done - waited) * 1e3
        st.compute_ms += (done - read_done) * 1e3
        if getattr(self.config, "stage_min_tokens", 0) > 0:  # the staging fit
            self.cost.observe(distinct, calls, (done - read_done) * 1e6)
        if not self._first_logged:
            self._first_logged = True
            logger.info(
                "CPU experts: the first routed experts were computed on the CPU "
                "(%s: %d calls of %d experts over %d tokens, %.2f ms; host "
                "cache hits %d, read from the drive %d).",
                tiers.layer_name,
                calls,
                distinct,
                n,
                (done - waited) * 1e3,
                self.cache.hits,
                self.cache.misses,
            )
        if runtime_debug.on("cpu"):
            runtime_debug.emit(
                "cpu",
                logger,
                "%s: %d calls of %d experts over %d tokens; read %.2f ms, "
                "compute %.2f ms",
                tiers.layer_name,
                calls,
                distinct,
                n,
                (read_done - waited) * 1e3,
                (done - read_done) * 1e3,
            )
        self._maybe_log()
        return True

    def _maybe_log(self) -> None:
        # Read live: the runtime control surface's store.set changes it.
        interval = float(self.config.stats_interval_s)
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats < interval:
            return
        self._last_stats = now
        self.log_stats()

    def snapshot(self) -> dict[str, float]:
        st, c = self.stats, self.cache
        return {
            "layers_with_cpu_calls": st.layers,
            "layers_without": st.idle_layers,
            "cpu_calls": st.calls,
            "cpu_experts": st.experts,
            "routed_card": st.per_tier["card"],
            "routed_pinned": st.per_tier["pinned"],
            "routed_cpu": st.per_tier["cpu"],
            "cache_hits": c.hits,
            "cache_misses": c.misses,
            "cache_prefetched": c.prefetched,
            "cache_evicted": c.evicted,
            "wait_ms": st.wait_ms,
            "read_ms": st.read_ms,
            "compute_ms": st.compute_ms,
            "promotions": st.promotions,
            "demotions": st.demotions,
            "routed_staged": st.per_tier["staged"],
            "stage_layers": st.stage_layers,
            "stage_experts": st.stage_experts,
            "stage_ms": st.stage_ms,
            "stage_copy_ms": st.stage_copy_ms,
            "stage_by_cost": st.stage_by_cost,
            "stage_declined": st.stage_declined,
        }

    def log_stats(self) -> None:
        s = self.snapshot()
        self._log_shares_since_last(s)
        routed = max(1, s["routed_card"] + s["routed_pinned"] + s["routed_cpu"])
        layers = max(1, s["layers_with_cpu_calls"])
        logger.info(
            "CPU experts: routed calls of this rank on the card %d (%.1f %%), "
            "pinned rows %d (%.1f %%), CPU %d (%.1f %%); %d layer calls computed "
            "on the CPU (%d without), %.2f ms each (compute %.2f, drive %.2f, "
            "wait for routing %.3f per layer call); host cache hits %d, read "
            "from the drive %d, read ahead %d, evicted %d; promotions %d, "
            "demotions %d.",
            s["routed_card"],
            100.0 * s["routed_card"] / routed,
            s["routed_pinned"],
            100.0 * s["routed_pinned"] / routed,
            s["routed_cpu"],
            100.0 * s["routed_cpu"] / routed,
            s["layers_with_cpu_calls"],
            s["layers_without"],
            (s["compute_ms"] + s["read_ms"]) / layers,
            s["compute_ms"] / layers,
            s["read_ms"] / layers,
            s["wait_ms"] / max(1, s["layers_with_cpu_calls"] + s["layers_without"]),
            s["cache_hits"],
            s["cache_misses"],
            s["cache_prefetched"],
            s["cache_evicted"],
            s["promotions"],
            s["demotions"],
        )

    def _log_shares_since_last(self, s: dict[str, float]) -> None:
        """The share of routed calls per tier since the last line (the main
        line's shares are since start): the CPU share over time."""
        keys = ("routed_card", "routed_pinned", "routed_cpu", "routed_staged")
        last = getattr(self, "_last_snapshot", None)
        self._last_snapshot = {k: s[k] for k in keys}
        if last is None:
            return
        delta = {k: s[k] - last[k] for k in keys}
        total = sum(delta.values())
        if total <= 0:
            return
        logger.info(
            "CPU experts: since the last line the card served %.1f %% of this "
            "rank's routed calls, pinned rows %.1f %%, the CPU %.1f %%, staging "
            "rows %.1f %% (%d calls).",
            100.0 * delta["routed_card"] / total,
            100.0 * delta["routed_pinned"] / total,
            100.0 * delta["routed_cpu"] / total,
            100.0 * delta["routed_staged"] / total,
            total,
        )


_RUNTIME: CpuExpertRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_runtime(
    config: "ExpertStoreConfig",
    store: Any,
    layout: UnitLayout,
    *,
    w4a8: bool = False,
) -> CpuExpertRuntime:
    """The process's CPU tier, created by the first layer that has host
    slots (every layer of a model has the same expert shapes and the same
    arithmetic class: W4A8 when the card runs the experts with int8
    activations, else W4A16)."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = CpuExpertRuntime(config, store, layout, w4a8=w4a8)
        elif _RUNTIME.layout != layout:
            raise CpuExpertError(
                "CPU experts: MoE layers with different expert shapes in one "
                "process are not supported"
            )
        elif _RUNTIME.w4a8 != bool(w4a8):
            raise CpuExpertError(
                "CPU experts: MoE layers with W4A8 and W4A16 experts in one "
                "process are not supported"
            )
        return _RUNTIME


def current_runtime() -> CpuExpertRuntime | None:
    return _RUNTIME


def reset_runtime() -> None:
    """Forget the process's CPU tier (tests; a new model load), with the
    staging rows that belong to it."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and _RUNTIME._async is not None:
            _RUNTIME._async.stop()
        _RUNTIME = None
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
        reset_stage_rows,
    )

    reset_stage_rows()
