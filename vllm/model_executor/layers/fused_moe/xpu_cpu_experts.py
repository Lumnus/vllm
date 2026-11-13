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

What the CPU computes is the card's W4A16 decode arithmetic at the card's
rounding points (bf16 after each GEMM and after the activation, fp32
accumulation, scales applied per 32-element chunk partial sum); only the
order of fp32 additions differs (see the op's source), so an expert's output
can differ from the card's in the last bit of some bf16 values, like the card's
own large-batch kernel does from its decode kernel.
"""

import mmap
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


def host_op_available() -> bool:
    try:
        return hasattr(torch.ops._xpu_C, "int4_moe_cpu")
    except (AttributeError, RuntimeError):
        return False


def _op_isa(want: int) -> int:
    """What the host op resolves ``want`` (0 auto, 1 scalar, 2 AVX2) to."""
    return int(torch.ops._xpu_C.int4_moe_cpu_isa(want))


def _op_run(*args: Any) -> None:
    torch.ops._xpu_C.int4_moe_cpu(*args)


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


def reference_int4_experts(
    x: torch.Tensor,
    units: Mapping[Any, Mapping[str, torch.Tensor]],
    calls: Sequence[tuple[Any, int, float]],
    inter: int,
    group: int,
) -> torch.Tensor:
    """What the card computes for ``calls`` (unit key, token, weight) in
    token order: per call ``bf16(bf16(silu(g)) * u)`` between two bf16-rounded
    int4 GEMMs, summed per token in call order in fp32, rounded to bf16.
    ``units[key]`` maps the four unit tensor names to their bytes (uint8)."""
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
        h = bf16(_int4_rows_dot(w13, s13, x[token : token + 1].float(), group))
        gate, up = h[:, :inter], h[:, inter:]
        act = bf16(bf16(gate / (1.0 + torch.exp(-gate))) * up)
        y = bf16(_int4_rows_dot(w2, s2, act, group))
        out[token] += y[0] * float(weight)
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


class CpuExpertRuntime:
    """The CPU tier of one process: the host cache, the host op, the
    pinned staging buffers and the counters."""

    def __init__(
        self,
        config: "ExpertStoreConfig",
        store: Any,
        layout: UnitLayout,
    ):
        if not host_op_available():
            raise CpuExpertError(
                "--cpu-offload-pinned-gb computes experts on the CPU with the "
                "vllm-xpu-kernels host op int4_moe_cpu, which the installed "
                "kernels do not have: install kernels built with it, or drop "
                "--cpu-offload-pinned-gb"
            )
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
        rows = max(1, int(config.host_cache_gib * 2**30) // layout.row_bytes)
        self.cache = HostExpertCache(store, layout, rows, config.read_timeout_s)
        self.stats = CpuExpertStats()
        self._buffers: _Buffers | None = None
        self._last_stats = time.monotonic()
        self._first_logged = False
        self._self_check()
        logger.info(
            "CPU experts: %s kernel, %d threads, host cache of %d experts "
            "(%.2f GiB of ordinary memory, %d B per expert); experts not in it "
            "are read from the expert store when a step routes to them.",
            _ISA_NAME[self.isa],
            self.threads,
            rows,
            self.cache.gib,
            layout.row_bytes,
        )

    # -- settings a running engine may change (runtime control: cpu.set) -------

    def set_threads(self, threads: int) -> None:
        """Threads of the host op from the next call on (the op keeps one
        pool per thread count)."""
        if threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        self.threads = int(threads)
        self.config.cpu_threads = int(threads)

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
        _op_run(
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
        )

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
        want = reference_int4_experts(x, tensors, calls, L.inter, L.group)
        steps = bf16_steps(out, want)
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
        logger.info(
            "CPU experts: self-check against the card's arithmetic passed "
            "(%s, largest difference %.3f bf16 steps at the output's scale%s).",
            _ISA_NAME[self.isa],
            steps,
            "; AVX2 and scalar code bit-identical" if self.isa == 2 else "",
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
        return Pending(tiers, tokens, event, b)

    def finish(self, pending: Pending, device: torch.device) -> torch.Tensor | None:
        """Wait for the routing only, compute the layer's host calls and
        return their summed contribution on the device (None: no host call).
        Raises :class:`CpuExpertError` when an expert cannot be served."""
        n, b = pending.num_tokens, pending.buffers
        self._wait(pending)
        if not self.compute(
            pending.tiers, b.ids[:n], b.weights[:n], b.x[:n], b.out[:n],
            selection=pending.selection, staged=pending.staged,
        ):  # fmt: skip
            return None
        return b.out[:n].to(device, non_blocking=True)

    def stage(self, pending: Pending, device: torch.device) -> Any:
        """Prefill staging (``stage_min_calls``): wait for the routing, and
        copy the CPU-tier experts this layer call routes at least that many
        tokens to into the staging rows (most calls first, as far as the rows
        go), with their map. Returns the layer's kernel call over the rows for
        the caller to run on the card, or None (nothing staged); the CPU then
        computes only the rest (:meth:`finish`)."""
        threshold = int(self.config.stage_min_calls)
        if threshold <= 0:
            return None
        n, b = pending.num_tokens, pending.buffers
        self._wait(pending)
        tiers = pending.tiers
        ids = b.ids[:n].numpy()
        selection = tiers._host_selection(ids)
        pending.selection = selection
        keys, tok, k, _ = selection
        if not keys or len(keys) < threshold:
            return None
        start = time.perf_counter()
        phys = ids[tok, k]
        uniq, counts = np.unique(phys, return_counts=True)
        heavy = uniq[counts >= threshold]
        if heavy.size == 0:
            return None
        rows = tiers.stage_rows(device)
        if heavy.size > rows.num_rows:  # the busiest ones
            by_calls = counts[counts >= threshold]
            heavy = heavy[np.argsort(-by_calls, kind="stable")[: rows.num_rows]]
        assert tiers.slot_logical is not None and tiers.store_binding is not None
        layer = tiers.store_binding.store_layer
        heavy_keys = [
            (layer, int(tiers.slot_logical[tiers._slot_of_global[p]])) for p in heavy
        ]
        cache_rows = self.cache.acquire(heavy_keys)
        try:
            for i, row in enumerate(cache_rows):
                src = self.cache.dests(row)
                for name in UNIT_TENSORS:
                    rows.rows_u8(name)[i].copy_(src[name])
        finally:
            self.cache.release(cache_rows)
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
        logger.info_once(
            "CPU experts: a large step's CPU-tier experts were staged to the card "
            "(%s: %d experts with at least %d calls each, %d calls; the CPU "
            "computes the other %d).",
            tiers.layer_name,
            heavy.size,
            threshold,
            staged_calls,
            len(keys) - staged_calls,
        )
        return tiers.stage_impl(rows)

    def finish_pairs(self, pending: Pending, device: torch.device) -> HostPairs | None:
        """:meth:`finish` for a sum that takes each pair's own row (the
        combine modes rank and exact): one output row per (token, top-k)
        pair the CPU computes, with its map. None: no host call."""
        from vllm.utils.platform_utils import is_pin_memory_available

        n, b = pending.num_tokens, pending.buffers
        self._wait(pending)
        topk, hidden = b.ids.shape[1], b.x.shape[1]
        if b.pair_rows is None or b.pair_rows.shape[0] < n * topk:
            b.pair_rows = torch.empty(
                (max(n, b.x.shape[0]) * topk, hidden),
                dtype=torch.bfloat16,
                pin_memory=is_pin_memory_available(),
            )
        pair = self.compute_pairs(pending.tiers, b.ids[:n], b.x[:n], b.pair_rows)
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
    ) -> np.ndarray | None:
        """On the host: each call of ``ids`` [T, K] this rank computes on the
        CPU, as its own row of ``out_rows`` (call ``i`` in row ``i``, the
        expert's bf16 output without the router weight: a call alone in its
        row with weight 1 rounds to exactly that). Returns the flat pair
        index ``token * K + k`` of each call, or None when there is none."""
        waited = time.perf_counter()
        keys, pair, by_tier = tiers.host_pairs(ids.numpy())
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
    config: "ExpertStoreConfig", store: Any, layout: UnitLayout
) -> CpuExpertRuntime:
    """The process's CPU tier, created by the first layer that has host
    slots (every layer of a model has the same expert shapes)."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = CpuExpertRuntime(config, store, layout)
        elif _RUNTIME.layout != layout:
            raise CpuExpertError(
                "CPU experts: MoE layers with different expert shapes in one "
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
