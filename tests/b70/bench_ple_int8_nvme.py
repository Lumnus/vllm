# SPDX-License-Identifier: Apache-2.0
"""B70 0013 CPU micro-bench of the per-step host path (one rank).

    nice -n 19 ionice -c3 /opt/venv/bin/python tests/b70/bench_ple_int8_nvme.py

Per shape (decode 1/4/8 seqs, one 1024-token prefill chunk), per rank kind
(rank 0 = bigram heads, rank 3 = trigram heads): the host hash alone, the
resolve without I/O (hits + bookkeeping), and the full resolve with the misses
read from the real table with O_DIRECT. The miss rate is set by pre-installing
(with dummy bytes, no reads) the complement of each step's rows: 45 % miss on
bigram ranks, 70 % on trigram ranks (A-design §2.1 planning numbers). Reads
are capped (~4.5 K per run) — the disk is shared with a serving engine.

0013b: B70_PLE_INT8_NVME_READER=py|native|uring picks the reader (default
py = v1); PLE_NVME_BENCH_RANKS (default "3,0") and PLE_NVME_BENCH_SHAPES
(default "decode,prefill") narrow a run to stay under the read budget.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ple_int8_nvme as t  # noqa: E402

pn = t.pn
ROW = t.ROW
STEPS = int(os.environ.get("PLE_NVME_BENCH_STEPS", "30"))
THREADS = int(os.environ.get("B70_PLE_INT8_NVME_IO_THREADS", "16"))
READER = os.environ.get("B70_PLE_INT8_NVME_READER", "py")
RANKS = [int(r) for r in os.environ.get("PLE_NVME_BENCH_RANKS", "3,0").split(",")]
SHAPES = os.environ.get("PLE_NVME_BENCH_SHAPES", "decode,prefill").split(",")


def make_step(rng, seqs: int, length: int):
    tokens = rng.integers(0, t.VOCAB, size=seqs * length).astype(np.int32)
    qsl = (np.arange(seqs + 1) * length).astype(np.int32)
    ctx = rng.integers(0, t.VOCAB, size=(seqs, 2)).astype(np.int32)
    return tokens, qsl, ctx


def server_for(rank, store, capacity=2_000_000):
    total = 320_001_536
    per = total // t.TP
    lo, hi = rank * per, (rank + 1) * per
    cache = pn.PleRowCache(hi - lo, capacity, ROW, slab=np.zeros((capacity, ROW), np.uint8))
    return pn.PleNvmeServer(
        tp_start=lo, tp_end=hi,
        multipliers=t.LAYOUT["layer_multipliers"],
        sizes=t.LAYOUT["ngram_heads_vocab_sizes"],
        offsets=t.LAYOUT["ngram_heads_offsets"],
        eos_token_id=t.EOS, heads_per_ngram=t.HPN, cache=cache, store=store,
        stats=pn.PleNvmeStats(rank, 1e9, True),
    )


def prewarm(server, step, miss_rate, rng):
    """Install (dummy bytes) the rows of ~(1 - miss_rate) of the step's
    own-row lookups, so the timed resolve misses ~miss_rate of them."""
    tokens, qsl, ctx = step
    own = server.hash(tokens, qsl, ctx, heads=server.own_heads)
    mask = (own >= server.tp_start) & (own < server.tp_end)
    local = np.unique(own[mask] - server.tp_start)
    keep = local[rng.random(local.shape[0]) >= miss_rate]
    keep = keep[server.cache.row2slot[keep] < 0]
    if keep.size:
        server.cache.begin_step()
        server.cache.install(keep, np.zeros((keep.shape[0], ROW), np.uint8))
    return int(local.shape[0])


def pct(values, q):
    return float(np.percentile(np.asarray(values), q))


def run_shape(server, label, seqs, length, miss_rate, steps, rng, io=True):
    out = np.empty(max(seqs * length, 1) * 16 + 16, np.int64)
    hash_ms, noio_ms, full_ms, reads, lookups = [], [], [], [], []
    for _ in range(steps):
        step = make_step(rng, seqs, length)
        tokens, qsl, ctx = step
        t0 = time.perf_counter()
        server.hash(tokens, qsl, ctx, heads=server.own_heads)
        hash_ms.append((time.perf_counter() - t0) * 1e3)
        lookups.append(prewarm(server, step, miss_rate, rng))
        # Then an all-hit replay of the same step: hash + bookkeeping, no I/O.
        before = server.stats.total_reads
        t0 = time.perf_counter()
        if io:
            server.resolve(tokens, qsl, ctx, seqs * length, out)
        full_ms.append((time.perf_counter() - t0) * 1e3)
        reads.append(server.stats.total_reads - before)
        t0 = time.perf_counter()
        server.resolve(tokens, qsl, ctx, seqs * length, out)  # all hits now
        noio_ms.append((time.perf_counter() - t0) * 1e3)
    lat = np.concatenate(server.stats.read_lat) * 1e3 if server.stats.read_lat else np.zeros(1)
    server.stats.read_lat = []
    print(
        f"{label:<34} steps {steps:>3}  own rows/step {np.mean(lookups):7.1f}  "
        f"misses/step {np.mean(reads):7.1f}  hash {np.median(hash_ms):6.3f} ms  "
        f"resolve(all-hit) p50 {np.median(noio_ms):6.3f} p99 {pct(noio_ms, 99):6.3f} ms  "
        f"resolve+reads p50 {np.median(full_ms):7.3f} p99 {pct(full_ms, 99):7.3f} ms  "
        f"read lat p50 {np.median(lat):.3f} p99 {pct(lat, 99):.3f} ms"
    )
    return int(np.sum(reads))


def main():
    rng = np.random.default_rng(1)
    data_start, shape, _ = pn.safetensors_tensor_location(t.TABLE, "table")
    store = pn.PleNvmeRowStore(t.TABLE, data_start, ROW, shape[0], io_threads=THREADS,
                               reader=READER)
    total_reads = 0
    print(f"reader={READER}, io_threads/QD={THREADS}, O_DIRECT, table {t.TABLE}")
    for rank, miss in ((3, 0.70), (0, 0.45)):
        if rank not in RANKS:
            continue
        server = server_for(rank, store)
        kind = "trigram" if rank >= 2 else "bigram"
        if "decode" in SHAPES:
            for seqs in (1, 4, 8):
                total_reads += run_shape(server, f"rank {rank} ({kind}) decode {seqs} seq",
                                         seqs, 1, miss, STEPS, rng)
        if "prefill" not in SHAPES:
            continue
        if rank == 3:
            total_reads += run_shape(server, f"rank {rank} ({kind}) prefill 1024",
                                     1, 1024, miss, 1, rng)
        else:
            total_reads += run_shape(server, f"rank {rank} ({kind}) prefill 1024 no-I/O",
                                     1, 1024, 0.0, 3, rng)
    store.close()
    print(f"total O_DIRECT row reads this run: {total_reads}")


if __name__ == "__main__":
    main()
