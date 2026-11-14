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

import functools
import mmap
import os
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
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import ExpertTiers

logger = init_logger(__name__)

# The tensors of an expert unit, in the order the op's offsets name them.
UNIT_TENSORS = ("w13", "w2", "w13_scales", "w2_scales")
_ISA = {"auto": 0, "scalar": 1, "avx2": 2}
_ISA_NAME = {1: "scalar", 2: "AVX2+FMA"}
PAGE = 4096

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


def card_arith(rows: int, hidden: int, inter: int, group: int) -> int:
    """The op's ``arith`` for a layer call the card runs with ``rows``
    (tokens x top_k) rows: per GEMM (bit 0 GEMM1 with K = hidden, bit 1
    GEMM2 with K = inter) set when the card's XpuFusedMoe runs it on the
    grouped GEMM (weights bf16(q * s)), clear when on its int4 GEMV. The
    GEMV takes at most ``INT4_GEMV_MAX_ROWS`` rows, K a multiple of 512 and
    a group a multiple of 32, unless switched off (``INT4_GEMV_ENV``); kernels
    without it run the grouped GEMM always."""
    try:
        from vllm_xpu_kernels import fused_moe_interface as fmi

        max_rows = int(fmi.INT4_GEMV_MAX_ROWS)
        env = str(fmi.INT4_GEMV_ENV)
    except (ImportError, AttributeError):
        return 3
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
            for key in wanted:
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


@dataclass
class Pending:
    """A layer's inputs on their way to the host."""

    tiers: "ExpertTiers"
    num_tokens: int
    event: Any
    buffers: _Buffers
    waited: bool = False
    # The op's arith for this layer call (card_arith): the card kernel whose
    # arithmetic each GEMM takes.
    arith: int = 0
    # The layer's host calls, once listed (by `stage`), and the physical ids
    # staged to the card, which the CPU leaves out.
    selection: Any = None
    staged: np.ndarray | None = None


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
        if self.isa != 2 and config.cpu_isa == "auto":
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
        self._last_stats = time.monotonic()
        self._first_logged = False
        self._self_check()
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
    ) -> None:
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
        if self.arith:
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
        if self.isa == 2:
            # The vector code must take the scalar code's steps in its order
            # (the build keeps fp contraction and reassociation off): the two
            # agree bit for bit, or the build is not the one tested.
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
                    "the CPU expert kernel's AVX2 code and its scalar code "
                    "disagree: the kernels were built with floating-point "
                    "contraction or reassociation on; refusing to compute "
                    "experts with it"
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
            "; AVX2 and scalar code bit-identical" if self.isa == 2 else "",
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
            self._buffers = b
        return b

    def post(
        self,
        tiers: "ExpertTiers",
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Pending:
        """Enqueue the copies of a layer's routing and inputs to the host,
        before the card's own experts of the layer."""
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
        tokens, topk = topk_ids.shape
        b = self._buffers_for(tokens, topk, hidden_states.shape[1])
        b.ids[:tokens].copy_(topk_ids, non_blocking=True)
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
        return Pending(tiers, tokens, event, b, arith=arith)

    def finish(self, pending: Pending, device: torch.device) -> torch.Tensor | None:
        """Wait for the routing only, compute the layer's host calls and
        return their summed contribution on the device (None: no host call).
        Raises :class:`CpuExpertError` when an expert cannot be served."""
        n, b = pending.num_tokens, pending.buffers
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
        selection = tiers._host_selection(ids)
        pending.selection = selection
        keys, tok, k, _ = selection
        if not keys or (not by_cost and len(keys) < threshold):
            return None
        start = time.perf_counter()
        phys = ids[tok, k]
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
        rows.map_host.fill_(-1)
        rows.map_host[torch.from_numpy(heavy.astype(np.int64))] = torch.arange(
            heavy.size, dtype=torch.int32
        )
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
                len(keys) - staged_calls,
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
                len(keys) - staged_calls,
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
                len(keys),
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
        from vllm.utils.platform_utils import is_pin_memory_available

        n, b = pending.num_tokens, pending.buffers
        self._wait(pending)
        self.arith = pending.arith
        topk, hidden = b.ids.shape[1], b.x.shape[1]
        if b.pair_rows is None or b.pair_rows.shape[0] < n * topk:
            b.pair_rows = torch.empty(
                (max(n, b.x.shape[0]) * topk, hidden),
                dtype=torch.bfloat16,
                pin_memory=is_pin_memory_available(),
            )
        pair = self.compute_pairs(
            pending.tiers, b.ids[:n], b.x[:n], b.pair_rows, staged=pending.staged
        )
        if pair is None:
            return None
        row_map = torch.full((n, topk), -1, dtype=torch.int32)
        row_map.view(-1)[torch.from_numpy(pair)] = torch.arange(
            len(pair), dtype=torch.int32
        )
        return HostPairs(
            b.pair_rows[: len(pair)].to(device, non_blocking=True),
            row_map.to(device, non_blocking=True),
        )

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
        selection: Any = None,
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
        if selection is None:
            selection = tiers._host_selection(ids_np)
        keys, tok, k, by_tier = selection
        if staged is not None and len(keys):
            keep = ~np.isin(ids_np[tok, k], staged)
            by_tier = dict(by_tier, cpu=by_tier["cpu"] - int((~keep).sum()))
            keys = [key for key, kept in zip(keys, keep) if kept]
            tok, k = tok[keep], k[keep]
        tokens = tok.astype(np.int32)
        call_w = np.ascontiguousarray(weights.numpy()[tok, k], dtype=np.float32)
        return self._run_calls(
            tiers, keys, by_tier, x, out, torch.from_numpy(tokens),
            torch.from_numpy(call_w), ids.shape[0], waited,
        )  # fmt: skip

    def compute_pairs(
        self,
        tiers: "ExpertTiers",
        ids: torch.Tensor,
        x: torch.Tensor,
        out_rows: torch.Tensor,
        staged: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """On the host: each call of ``ids`` [T, K] this rank computes on the
        CPU, as its own row of ``out_rows`` (call ``i`` in row ``i``, the
        expert's bf16 output without the router weight: a call alone in its
        row with weight 1 rounds to exactly that). Returns the flat pair
        index ``token * K + k`` of each call, or None when there is none.
        ``staged``: physical ids the card computes from staging rows, left
        out here (counted as staged)."""
        waited = time.perf_counter()
        ids_np = ids.numpy()
        keys, pair, by_tier = tiers.host_pairs(ids_np)
        if staged is not None and len(keys):
            keep = ~np.isin(ids_np.reshape(-1)[pair], staged)
            by_tier = dict(by_tier, cpu=by_tier["cpu"] - int((~keep).sum()))
            keys = [key for key, kept in zip(keys, keep) if kept]
            pair = pair[keep]
        n = len(keys)
        tokens = torch.from_numpy(pair // ids.shape[1])
        xs = x.index_select(0, tokens) if n else x[:0]
        ran = self._run_calls(
            tiers, keys, by_tier, xs, out_rows[:n],
            torch.arange(n, dtype=torch.int32), torch.ones(n, dtype=torch.float32),
            ids.shape[0], waited,
        )  # fmt: skip
        return pair if ran else None

    def _run_calls(
        self,
        tiers: "ExpertTiers",
        keys: list[Key],
        by_tier: dict[str, int],
        x: torch.Tensor,
        out: torch.Tensor,
        tokens: torch.Tensor,
        call_w: torch.Tensor,
        n: int,
        waited: float,
    ) -> bool:
        st = self.stats
        for tier, count in by_tier.items():
            st.per_tier[tier] += count
        if not keys:
            st.idle_layers += 1
            self._maybe_log()
            return False
        rows = self.cache.acquire(keys)
        try:
            read_done = time.perf_counter()
            self.run_op(
                out,
                x,
                self.cache.slab,
                torch.tensor(rows, dtype=torch.int32),
                tokens,
                call_w,
            )
        finally:
            self.cache.release(rows)
        done = time.perf_counter()
        st.layers += 1
        st.calls += len(keys)
        st.max_calls = max(st.max_calls, len(keys))
        distinct = len(set(keys))
        st.experts += distinct
        st.max_experts = max(st.max_experts, distinct)
        st.read_ms += (read_done - waited) * 1e3
        st.compute_ms += (done - read_done) * 1e3
        if getattr(self.config, "stage_min_tokens", 0) > 0:  # the staging fit
            self.cost.observe(distinct, len(keys), (done - read_done) * 1e6)
        if not self._first_logged:
            self._first_logged = True
            logger.info(
                "CPU experts: the first routed experts were computed on the CPU "
                "(%s: %d calls of %d experts over %d tokens, %.2f ms; host "
                "cache hits %d, read from the drive %d).",
                tiers.layer_name,
                len(keys),
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
                len(keys),
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
        _RUNTIME = None
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
        reset_stage_rows,
    )

    reset_stage_rows()
