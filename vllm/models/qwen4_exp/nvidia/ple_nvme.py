# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0013: serve the INT8 PLE n-gram table from NVMe (host side).

Everything here is host-only (numpy / os / threads; numba when present) and
imports nothing from vLLM, so the tests load it standalone. The model-side
wiring lives in ngram_embedding.py (``Qwen4ExpNGramEmbedding.b70_nvme_*``),
model_state.py (``b70_pre_forward``) and the V2 model runner (the call just
before "Run model", real batches only).

Per TP rank, per step:
  1. ``host_ngram_ids``: the n-gram ids of the real tokens, bit-identical to
     ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` (same integer ops in numpy;
     the products never exceed 2^63 for in-vocab tokens, and numpy/torch wrap
     and take remainders identically even when they would).
  2. ``PleNvmeServer.resolve``: own-range rows -> slots in a pinned row cache
     (CLOCK, this step and the previous step protected); the misses of the
     step are deduplicated and read from the table file with O_DIRECT.
  3. The caller H2D-copies the int64 slot ids and launches the existing
     pinned-lookup Triton kernel over the cache's UVA view (slot -1 = row not
     owned by this rank, or a padding token: nothing stored, stays zero).

Env contract (read in ngram_embedding.py, documented here):
  B70_PLE_INT8_NVME=1               enable (requires B70_PLE_INT8=1)
  B70_PLE_INT8_NVME_PATH            table file (default B70_PLE_INT8_PATH)
  B70_PLE_INT8_NVME_CACHE_GIB       row cache, TOTAL over the TP ranks (default 8)
  B70_PLE_INT8_NVME_IO_THREADS      reader threads per rank (default 16)
  B70_PLE_INT8_NVME_SYNC_ONLY=1     hook + host hash + cache bookkeeping, rows
                                    served from the in-RAM pinned table
  B70_PLE_INT8_NVME_STATS=1         periodic per-rank stats line
  B70_PLE_INT8_NVME_STATS_S         stats interval, seconds (default 60)
  B70_PLE_INT8_NVME_BOOT_SAMPLE     own rows read at boot for the scale check
                                    (default 65536)
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger("vllm.models.qwen4_exp.nvidia.ple_nvme")

PAGE = 4096
# One row read is at most two pages (a 164-byte row crosses at most one
# page boundary); every bounce slot is this large.
READ_SLOT = 2 * PAGE
# Retry backoff for a failed or short read, seconds (then raise).
RETRY_BACKOFF_S = (0.001, 0.010, 0.100)

try:  # numba makes the CLOCK sweep cheap; the fallback is a Python loop.
    import numba as _numba

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is present in the image
    _numba = None
    _HAVE_NUMBA = False


# --------------------------------------------------------------------------
# Host n-gram hash
# --------------------------------------------------------------------------


def host_ngram_ids(
    tokens: np.ndarray,
    query_start_loc: np.ndarray,
    ngram_context: np.ndarray,
    *,
    multipliers: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    eos_token_id: int,
    heads_per_ngram: int,
    heads: np.ndarray | None = None,
) -> np.ndarray:
    """N-gram ids [T, H] of the real tokens (host, bit-exact with the device).

    ``tokens`` [T] are the real (unpadded) tokens of the batch, laid out by
    ``query_start_loc`` [R + 1] (qsl[0] == 0, qsl[R] == T); ``ngram_context``
    [>= R, ngram_size - 1] the tokens before each request's first token (EOS
    where there are none). Mirrors ``compute_ngram_ids``' pure-torch path: a
    shifted token is replaced by EOS when it lies before the start of the
    token's EOS segment. ``heads`` (optional) selects which of the
    (ngram_size - 1) * heads_per_ngram heads to return, in that order.
    """
    tokens = np.asarray(tokens).reshape(-1).astype(np.int64, copy=False)
    qsl = np.asarray(query_start_loc).astype(np.int64, copy=False)
    mult = np.asarray(multipliers, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    ngram_size = int(mult.shape[0])
    ctx_len = ngram_size - 1
    num_heads = ctx_len * heads_per_ngram
    if heads is None:
        heads = np.arange(num_heads)
    heads = np.asarray(heads, dtype=np.int64)
    num_tokens = tokens.shape[0]
    num_reqs = qsl.shape[0] - 1
    if num_tokens == 0 or num_reqs <= 0:
        return np.empty((num_tokens, heads.shape[0]), dtype=np.int64)
    if qsl[0] != 0 or qsl[-1] != num_tokens:
        raise ValueError(
            f"query_start_loc [{qsl[0]}..{qsl[-1]}] does not cover {num_tokens} tokens"
        )
    lens = np.diff(qsl)
    if (lens < 0).any():
        raise ValueError("query_start_loc must be non-decreasing")
    ctx = np.asarray(ngram_context)[:num_reqs].astype(np.int64, copy=False)
    # Flat layout: each request's ctx_len context tokens, then its tokens.
    req_start = qsl[:-1] + ctx_len * np.arange(num_reqs, dtype=np.int64)
    total = num_tokens + ctx_len * num_reqs
    seq = np.empty(total, dtype=np.int64)
    req_of_tok = np.repeat(np.arange(num_reqs, dtype=np.int64), lens)
    tok_flat = np.arange(num_tokens, dtype=np.int64) + ctx_len * (req_of_tok + 1)
    seq[tok_flat] = tokens
    seq[(req_start[:, None] + np.arange(ctx_len, dtype=np.int64)).reshape(-1)] = (
        ctx.reshape(-1)
    )
    # Previous EOS strictly before each position, clamped to "just before the
    # request's context" (the device path's -1 at the start of the row).
    index = np.arange(total, dtype=np.int64)
    eos_at = np.where(seq == eos_token_id, index, -1)
    inclusive = np.maximum.accumulate(eos_at)
    previous = np.empty(total, dtype=np.int64)
    previous[0] = -1
    previous[1:] = inclusive[:-1]
    previous = np.maximum(previous, np.repeat(req_start - 1, lens + ctx_len))
    position_in_segment = (index - previous - 1)[tok_flat]
    terms = []
    for shift in range(ngram_size):
        if shift == 0:
            shifted = seq[tok_flat]
        else:
            source = np.maximum(tok_flat - shift, 0)
            shifted = np.where(
                position_in_segment >= shift, seq[source], np.int64(eos_token_id)
            )
        terms.append(shifted * mult[shift])
    out = np.empty((num_tokens, heads.shape[0]), dtype=np.int64)
    mixed_by_order: dict[int, np.ndarray] = {}
    for column, head in enumerate(heads.tolist()):
        order = head // heads_per_ngram + 2
        mixed = mixed_by_order.get(order)
        if mixed is None:
            mixed = terms[0]
            for index_ in range(1, order):
                mixed = np.bitwise_xor(mixed, terms[index_])
            mixed_by_order[order] = mixed
        out[:, column] = np.remainder(mixed, sizes[head]) + offsets[head]
    return out


def owned_heads(sizes: np.ndarray, offsets: np.ndarray, start: int, end: int) -> np.ndarray:
    """Heads whose global rows intersect [start, end)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    hit = (offsets < end) & (offsets + sizes > start)
    return np.nonzero(hit)[0].astype(np.int64)


# --------------------------------------------------------------------------
# O_DIRECT row reader
# --------------------------------------------------------------------------


def safetensors_tensor_location(path: str, name: str) -> tuple[int, list[int], str]:
    """(absolute byte offset, shape, dtype) of one tensor of a .safetensors file."""
    import json

    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    info = header[name]
    begin = int(info["data_offsets"][0])
    return 8 + header_len + begin, [int(v) for v in info["shape"]], str(info["dtype"])


class PleNvmeRowStore:
    """Read 164-byte table rows from the file with O_DIRECT (one fd per rank).

    Row g is at ``data_start + row_bytes * g``. Each row is one 4 KiB read,
    or 8 KiB when it crosses a page boundary. Reads go into page-aligned
    anonymous-mmap bounce slots (8 KiB each) and are split into contiguous
    batches over ``io_threads`` workers, each looping ``os.preadv`` (which
    releases the GIL). A failed or short read is retried 3 times (1/10/100
    ms), then raises; nothing is ever zero-filled.
    """

    def __init__(
        self,
        path: str,
        data_start: int,
        row_bytes: int,
        num_rows: int,
        *,
        io_threads: int = 16,
        max_batch_rows: int = 8192,
        direct: bool = True,
    ) -> None:
        self.path = path
        self.data_start = int(data_start)
        self.row_bytes = int(row_bytes)
        self.num_rows = int(num_rows)
        self.io_threads = max(1, int(io_threads))
        self.max_batch_rows = max(1, int(max_batch_rows))
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        self.fd = os.open(path, flags)
        self.direct = direct
        self.file_size = os.fstat(self.fd).st_size
        end = self.data_start + self.row_bytes * self.num_rows
        if end > self.file_size:
            os.close(self.fd)
            raise ValueError(
                f"{path}: {self.num_rows} rows x {self.row_bytes} B from byte "
                f"{self.data_start} end at {end}, past the file ({self.file_size} B)"
            )
        self._bounce = mmap.mmap(-1, self.max_batch_rows * READ_SLOT)
        self._bounce_view = memoryview(self._bounce)
        self._bounce_np = np.frombuffer(self._bounce, dtype=np.uint8).reshape(
            self.max_batch_rows, READ_SLOT
        )
        self._pool = (
            ThreadPoolExecutor(self.io_threads, thread_name_prefix="ple-nvme")
            if self.io_threads > 1
            else None
        )
        self._preadv = os.preadv  # tests swap this for fault injection
        self._lat = np.zeros(self.max_batch_rows, dtype=np.float64)
        self._row_offsets = np.arange(self.row_bytes, dtype=np.int64)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def _read_one(self, slot: int, row: int, offset: int, length: int, need: int) -> None:
        view = self._bounce_view[slot * READ_SLOT : slot * READ_SLOT + length]
        last_error: str = ""
        for attempt in range(len(RETRY_BACKOFF_S) + 1):
            try:
                got = self._preadv(self.fd, [view], offset)
                if got >= need:
                    return
                last_error = f"short read {got} < {need} B"
            except OSError as exc:
                last_error = f"errno {exc.errno} ({exc.strerror})"
            if attempt < len(RETRY_BACKOFF_S):
                time.sleep(RETRY_BACKOFF_S[attempt])
        raise RuntimeError(
            f"PLE NVMe read failed after {len(RETRY_BACKOFF_S)} retries: row {row}, "
            f"file offset {offset}, length {length}: {last_error} ({self.path})"
        )

    def _read_range(
        self, lo: int, hi: int, rows: np.ndarray, offsets: np.ndarray,
        lengths: np.ndarray, needs: np.ndarray,
    ) -> None:
        lat = self._lat
        for i in range(lo, hi):
            t0 = time.perf_counter()
            self._read_one(i, int(rows[i]), int(offsets[i]), int(lengths[i]), int(needs[i]))
            lat[i] = time.perf_counter() - t0

    def read_rows(self, rows: np.ndarray, out: np.ndarray | None = None,
                  latencies: list | None = None) -> np.ndarray:
        """Bytes of ``rows`` (global row ids) as uint8 [n, row_bytes]."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n = rows.shape[0]
        if out is None:
            out = np.empty((n, self.row_bytes), dtype=np.uint8)
        if n == 0:
            return out
        if rows.min() < 0 or rows.max() >= self.num_rows:
            raise IndexError(
                f"PLE NVMe row out of range [0, {self.num_rows}): "
                f"{int(rows.min())}..{int(rows.max())}"
            )
        for lo in range(0, n, self.max_batch_rows):
            hi = min(n, lo + self.max_batch_rows)
            part = rows[lo:hi]
            count = hi - lo
            byte = self.data_start + part * self.row_bytes
            page = byte & ~np.int64(PAGE - 1)
            inner = byte - page
            lengths = np.where(inner + self.row_bytes > PAGE, READ_SLOT, PAGE).astype(np.int64)
            needs = inner + self.row_bytes
            workers = min(self.io_threads, count)
            if workers <= 1 or self._pool is None:
                self._read_range(0, count, part, page, lengths, needs)
            else:
                bounds = np.linspace(0, count, workers + 1).astype(np.int64)
                futures = [
                    self._pool.submit(
                        self._read_range, int(bounds[w]), int(bounds[w + 1]),
                        part, page, lengths, needs,
                    )
                    for w in range(workers)
                    if bounds[w + 1] > bounds[w]
                ]
                for future in futures:
                    future.result()
            if latencies is not None:
                latencies.append(self._lat[:count].copy())
            columns = inner[:, None] + self._row_offsets[None, :]
            out[lo:hi] = np.take_along_axis(self._bounce_np[:count], columns, axis=1)
        return out


# --------------------------------------------------------------------------
# Row cache
# --------------------------------------------------------------------------

_EPOCH_FREE = np.iinfo(np.int64).min // 2


def _clock_alloc_py(n, hand, ref, epoch, min_epoch, step, out):
    capacity = ref.shape[0]
    got = 0
    scanned = 0
    limit = 2 * capacity + n
    while got < n:
        if scanned >= limit:
            return got, hand
        slot = hand
        hand += 1
        if hand == capacity:
            hand = 0
        scanned += 1
        if epoch[slot] >= min_epoch:
            continue
        if ref[slot]:
            ref[slot] = 0
            continue
        out[got] = slot
        epoch[slot] = step
        ref[slot] = 1
        got += 1
    return got, hand


_clock_alloc = (
    _numba.njit(cache=False, nogil=True)(_clock_alloc_py) if _HAVE_NUMBA else _clock_alloc_py
)


class PleRowCache:
    """Per-rank row cache: pinned uint8 [C, row_bytes] slots + host maps.

    ``row2slot`` is a direct int32 map over the rank's own rows (-1 = absent).
    Eviction is CLOCK (second chance) with epoch protection: a slot touched in
    this step or the previous one is never evicted. A step that needs more
    slots than CLOCK can free raises RuntimeError before any map changes
    (the boot check makes that unreachable: capacity >= 2 x the largest step).
    ``slab`` is any uint8 [C, row_bytes] numpy array (the pinned tensor's
    ``.numpy()`` view in the engine); None = bookkeeping only (SYNC_ONLY).
    Slot ids are int64 everywhere (slot * row_bytes exceeds 2^31 once
    C > 13,094,412 rows).
    """

    def __init__(self, num_rows: int, capacity: int, row_bytes: int,
                 slab: np.ndarray | None = None) -> None:
        if capacity <= 0:
            raise ValueError("PLE NVMe cache capacity must be > 0")
        if capacity > np.iinfo(np.int32).max:
            raise ValueError("PLE NVMe cache capacity exceeds the int32 row map")
        if slab is not None and (slab.shape[0] < capacity or slab.shape[1] != row_bytes
                                 or slab.dtype != np.uint8):
            raise ValueError(f"PLE NVMe cache slab {slab.shape} {slab.dtype} does not fit")
        self.num_rows = int(num_rows)
        self.capacity = int(capacity)
        self.row_bytes = int(row_bytes)
        self.slab = slab
        self.row2slot = np.full(self.num_rows, -1, dtype=np.int32)
        self.slot2row = np.full(self.capacity, -1, dtype=np.int64)
        self.ref = np.zeros(self.capacity, dtype=np.uint8)
        self.epoch = np.full(self.capacity, _EPOCH_FREE, dtype=np.int64)
        self.hand = 0
        self.step = 0
        self._alloc_out = np.empty(0, dtype=np.int64)

    def meta_bytes(self) -> int:
        return (self.row2slot.nbytes + self.slot2row.nbytes + self.ref.nbytes
                + self.epoch.nbytes)

    def begin_step(self) -> int:
        self.step += 1
        return self.step

    def lookup(self, local_rows: np.ndarray) -> np.ndarray:
        """int64 slots of ``local_rows`` (-1 = miss); marks the hits used."""
        slots = self.row2slot[local_rows].astype(np.int64)
        hit = slots >= 0
        if hit.any():
            used = slots[hit]
            self.ref[used] = 1
            self.epoch[used] = self.step
        return slots

    def allocate(self, count: int) -> np.ndarray:
        if count > self._alloc_out.shape[0]:
            self._alloc_out = np.empty(max(count, 1024), dtype=np.int64)
        out = self._alloc_out
        got, hand = _clock_alloc(
            count, self.hand, self.ref, self.epoch, self.step - 1, self.step, out
        )
        self.hand = int(hand)
        if got < count:
            # Undo the partial pick's marks so the cache state is unchanged
            # except for cleared reference bits (a normal CLOCK side effect).
            # (They were evictable before, so "old" is their honest epoch.)
            picked = out[:got]
            self.epoch[picked] = _EPOCH_FREE
            self.ref[picked] = 0
            raise RuntimeError(
                f"PLE NVMe cache: step needs {count} new slots but only {got} are "
                f"evictable (capacity {self.capacity}; slots touched this or the "
                "previous step are protected) — raise B70_PLE_INT8_NVME_CACHE_GIB"
            )
        return out[:count].copy()

    def install(self, local_rows: np.ndarray, data: np.ndarray | None) -> np.ndarray:
        """Place unique missing ``local_rows`` into freshly allocated slots."""
        slots = self.allocate(local_rows.shape[0])
        old = self.slot2row[slots]
        evicted = old[old >= 0]
        if evicted.size:
            self.row2slot[evicted] = -1
        if self.slab is not None and data is not None:
            self.slab[slots] = data
        self.slot2row[slots] = local_rows
        self.row2slot[local_rows] = slots.astype(np.int32)
        return slots


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


class PleNvmeStats:
    """Counters over an interval; one log line per interval (per rank)."""

    def __init__(self, rank: int, interval_s: float, enabled: bool) -> None:
        self.rank = rank
        self.interval_s = float(interval_s)
        self.enabled = enabled
        self._reset(time.monotonic())
        self.total_steps = 0
        self.total_reads = 0

    def _reset(self, now: float) -> None:
        self.t0 = now
        self.steps = 0
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.reads = 0
        self.hook_ms: list[float] = []
        self.io_ms: list[float] = []
        self.read_lat: list[np.ndarray] = []

    def record(self, lookups: int, hits: int, misses: int, reads: int,
               hook_ms: float, io_ms: float) -> None:
        self.total_steps += 1
        self.total_reads += reads
        if not self.enabled:
            return
        self.steps += 1
        self.lookups += lookups
        self.hits += hits
        self.misses += misses
        self.reads += reads
        self.hook_ms.append(hook_ms)
        self.io_ms.append(io_ms)
        if io_ms > 5.0:
            logger.info("PLE NVMe rank %d: slow step, %d reads took %.1f ms",
                        self.rank, reads, io_ms)
        now = time.monotonic()
        if now - self.t0 >= self.interval_s:
            logger.info("%s", self.line(now))
            self._reset(now)

    def line(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        lat = (np.concatenate(self.read_lat) * 1e3) if self.read_lat else np.zeros(0)
        hook = np.asarray(self.hook_ms) if self.hook_ms else np.zeros(1)

        def pct(values: np.ndarray, q: float) -> float:
            return float(np.percentile(values, q)) if values.size else 0.0

        rate = self.hits / self.lookups if self.lookups else 0.0
        return (
            f"PLE NVMe rank {self.rank}: {self.steps} steps in {now - self.t0:.0f} s, "
            f"hit rate {rate:.3f} ({self.hits}/{self.lookups}), "
            f"misses/step {self.misses / max(self.steps, 1):.1f}, reads {self.reads}, "
            f"read p50 {pct(lat, 50):.3f} ms p99 {pct(lat, 99):.3f} ms, "
            f"bubble p50 {pct(hook, 50):.3f} ms p99 {pct(hook, 99):.3f} ms "
            f"max {float(hook.max()):.3f} ms"
        )


# --------------------------------------------------------------------------
# Per-rank server
# --------------------------------------------------------------------------


class PleNvmeServer:
    """Host side of one rank: hash -> own rows -> cache slots (+ reads)."""

    def __init__(
        self,
        *,
        tp_start: int,
        tp_end: int,
        multipliers: np.ndarray,
        sizes: np.ndarray,
        offsets: np.ndarray,
        eos_token_id: int,
        heads_per_ngram: int,
        cache: PleRowCache,
        store: PleNvmeRowStore | None,
        stats: PleNvmeStats,
        sync_only: bool = False,
    ) -> None:
        self.tp_start = int(tp_start)
        self.tp_end = int(tp_end)
        self.multipliers = np.asarray(multipliers, dtype=np.int64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.eos_token_id = int(eos_token_id)
        self.heads_per_ngram = int(heads_per_ngram)
        self.num_heads = (self.multipliers.shape[0] - 1) * self.heads_per_ngram
        self.own_heads = owned_heads(self.sizes, self.offsets, self.tp_start, self.tp_end)
        self.cache = cache
        self.store = store
        self.stats = stats
        self.sync_only = sync_only
        if not sync_only and store is None:
            raise ValueError("PLE NVMe server needs a row store unless SYNC_ONLY")
        self._lock = threading.Lock()

    def hash(self, tokens, qsl, ctx, heads=None) -> np.ndarray:
        return host_ngram_ids(
            tokens, qsl, ctx,
            multipliers=self.multipliers, sizes=self.sizes, offsets=self.offsets,
            eos_token_id=self.eos_token_id, heads_per_ngram=self.heads_per_ngram,
            heads=heads,
        )

    def resolve(self, tokens: np.ndarray, qsl: np.ndarray, ctx: np.ndarray,
                num_tokens_padded: int, out: np.ndarray, *, t_start: float | None = None
                ) -> np.ndarray:
        """Fill ``out`` (int64, >= num_tokens_padded * num_heads) and return it.

        NVMe mode: slot ids of this rank's rows, -1 elsewhere (other ranks'
        rows, padding tokens). SYNC_ONLY: the global ids of all heads for the
        real tokens, -1 for padding (served by the in-RAM table); the cache
        still runs its bookkeeping on the own rows so the hit rate is real.
        """
        t_start = time.perf_counter() if t_start is None else t_start
        num_tokens = int(np.asarray(tokens).shape[0])
        heads = self.num_heads
        view = out[: num_tokens_padded * heads].reshape(num_tokens_padded, heads)
        view.fill(-1)
        self.cache.begin_step()
        io_ms = 0.0
        reads = 0
        lookups = hits = misses = 0
        if num_tokens:
            if self.sync_only:
                ids = self.hash(tokens, qsl, ctx)
                view[:num_tokens] = ids
                own = ids[:, self.own_heads]
            else:
                own = self.hash(tokens, qsl, ctx, heads=self.own_heads)
            mask = (own >= self.tp_start) & (own < self.tp_end)
            local = own[mask] - self.tp_start
            slots = self.cache.lookup(local)
            miss = slots < 0
            lookups = int(local.shape[0])
            misses = int(miss.sum())
            hits = lookups - misses
            if misses:
                miss_rows = np.unique(local[miss])
                data = None
                if not self.sync_only:
                    t_io = time.perf_counter()
                    lat: list = []
                    data = self.store.read_rows(miss_rows + self.tp_start, latencies=lat)
                    io_ms = (time.perf_counter() - t_io) * 1e3
                    reads = int(miss_rows.shape[0])
                    if self.stats.enabled:
                        self.stats.read_lat.extend(lat)
                self.cache.install(miss_rows, data)
                slots = self.cache.row2slot[local].astype(np.int64)
            if not self.sync_only:
                sub = view[:num_tokens]
                cols = np.full(own.shape, -1, dtype=np.int64)
                cols[mask] = slots
                sub[:, self.own_heads] = cols
        hook_ms = (time.perf_counter() - t_start) * 1e3
        self.stats.record(lookups, hits, misses, reads, hook_ms, io_ms)
        return out


def cache_rows_per_rank(total_gib: float, tp_size: int, row_bytes: int) -> int:
    per_rank = int(total_gib * (1 << 30)) // max(1, tp_size)
    return per_rank // row_bytes


__all__ = [
    "PleNvmeRowStore",
    "PleNvmeServer",
    "PleNvmeStats",
    "PleRowCache",
    "cache_rows_per_rank",
    "host_ngram_ids",
    "owned_heads",
    "safetensors_tensor_location",
]
