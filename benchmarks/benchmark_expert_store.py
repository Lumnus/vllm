# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the NVMe expert store: unit latency, bursts, sustained reads.

Builds a throwaway store of random bytes in a subdirectory of each given
directory (one per drive), measures it through :class:`ExpertStore`, prints one
table and removes the store. The numbers describe the drives the directories sit
on and nothing else.

Example::

    python benchmarks/benchmark_expert_store.py \
        --dirs /mnt/nvme0/bench /mnt/nvme1/bench \
        --unit-bytes 2534400 --units 256 --pool-slots 32 --rate 200

``--stripes 1 2`` runs the store over the first one, then the first two
directories. Directories on one drive show the code's cost, not striping gain.
"""

import argparse
import os
import random
import shutil
import statistics
import time

import numpy as np

from vllm.model_executor.offloader.expert_store import (
    ExpertStore,
    ExpertStoreWriter,
    StoreSchema,
    TensorSpec,
)

IDENT = "bench"


def filesystem_of(path: str) -> str:
    best = ("", "?")
    with open("/proc/mounts") as handle:
        for line in handle:
            _dev, mount, kind, *_ = line.split()
            real = os.path.realpath(path)
            if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(
                mount
            ) >= len(best[0]):
                best = (mount, f"{kind}@{_dev}")
    return best[1]


def build_store(dirs: list[str], unit_bytes: int, units: int) -> StoreSchema:
    schema = StoreSchema.uniform(
        1, units, [TensorSpec("blob", "uint8", (unit_bytes,))], tensor_align=1
    )
    rng = np.random.default_rng(1234)
    base = rng.integers(0, 256, unit_bytes, dtype=np.uint8)
    with ExpertStoreWriter(dirs, schema, IDENT, rebuild=True) as writer:
        for expert in range(units):
            base[:8] = np.frombuffer(expert.to_bytes(8, "little"), dtype=np.uint8)
            writer.write(0, expert, {"blob": base})
        writer.finalize(checksum=False)
    return schema


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def measure(
    dirs: list[str], args: argparse.Namespace, stripes: int
) -> list[tuple[str, str]]:
    used = [os.path.join(d, "expert_store_bench") for d in dirs[:stripes]]
    for d in used:
        os.makedirs(d, exist_ok=True)
    if args.units < args.pool_slots + max(args.bursts):
        raise SystemExit("--units must exceed --pool-slots by the largest burst")
    schema = build_store(used, args.unit_bytes, args.units)
    slot = max(g.tensors[0].nbytes for g in schema.groups)
    rows: list[tuple[str, str]] = []
    pool_bytes = args.pool_slots * (-(-slot // 4096) * 4096)
    store = ExpertStore.open(
        used,
        IDENT,
        pool_bytes,
        io_threads=args.threads,
        pin=args.pin,
        direct=not args.buffered,
    )
    try:
        rng = random.Random(7)
        order = list(range(args.units))
        rng.shuffle(order)

        # 1. single-unit latency, every unit cold
        lat = []
        for expert in order[: min(args.samples, args.units)]:
            t0 = time.perf_counter()
            with store.get((0, expert)):
                pass
            lat.append((time.perf_counter() - t0) * 1e3)
        rows.append(
            (
                "single get, cold (ms)",
                f"p50 {pct(lat, 50):.2f}  p95 {pct(lat, 95):.2f}  max {max(lat):.2f}",
            )
        )

        # 2. bursts of K units, all issued together
        for burst in args.bursts:
            if burst > args.pool_slots:
                continue
            times = []
            for _ in range(args.reps):
                pool = [e for e in range(args.units) if not store.resident((0, e))]
                keys = [(0, e) for e in rng.sample(pool, burst)]
                t0 = time.perf_counter()
                futures = store.request(keys, lease=True)
                entries = [f.result(timeout=60) for f in futures]
                times.append((time.perf_counter() - t0) * 1e3)
                for entry in entries:
                    entry.release()
            med = statistics.median(times)
            mib = burst * args.unit_bytes / 2**20 / (med / 1e3)
            rows.append(
                (
                    f"burst K={burst} (ms, MiB/s)",
                    (
                        f"p50 {med:.2f}  p95 {pct(times, 95):.2f}  max {max(times):.2f}"
                        f"  {mib:,.0f} MiB/s"
                    ),
                )
            )

        # 3. sustained: one unit per tick at the target rate
        if args.rate > 0:
            interval = 1.0 / args.rate
            done_ms: list[float] = []

            def note(start):
                def cb(_f):
                    done_ms.append((time.perf_counter() - start) * 1e3)

                return cb

            futures = []
            begin = time.perf_counter()
            issued = 0
            while True:
                due = begin + issued * interval
                now = time.perf_counter()
                if now - begin >= args.duration:
                    break
                if now < due:
                    time.sleep(min(due - now, 0.001))
                    continue
                pool = [e for e in range(args.units) if not store.resident((0, e))]
                key = (0, rng.choice(pool))
                f = store.request([key])[0]
                f.add_done_callback(note(time.perf_counter()))
                futures.append(f)
                issued += 1
            for f in futures:
                f.result(timeout=60)
            elapsed = time.perf_counter() - begin
            rows.append(
                (
                    f"sustained {args.rate:g} units/s (ms)",
                    (
                        f"issued {issued} in {elapsed:.1f}s = {issued / elapsed:.0f}/s "
                        f"{issued * args.unit_bytes / 2**20 / elapsed:,.0f} MiB/s  "
                        f"latency p50 {pct(done_ms, 50):.2f}  "
                        f"p95 {pct(done_ms, 95):.2f}  max {max(done_ms):.2f}"
                    ),
                )
            )
        st = store.stats()
        rows.append(
            (
                "store counters",
                (
                    f"reads {st['reads']}  direct {st['reads_direct']}  buffered "
                    f"{st['reads_buffered']}  evictions {st['evictions']}  read p50 "
                    f"{st['read_ms']['p50']:.2f} p95 {st['read_ms']['p95']:.2f} "
                    f"max {st['read_ms']['max']:.2f} ms"
                ),
            )
        )
    finally:
        store.close()
        for d in used:
            shutil.rmtree(d, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--stripes", nargs="+", type=int, default=None)
    parser.add_argument("--unit-bytes", type=int, default=2_534_400)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--pool-slots", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bursts", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rate", type=float, default=0.0, help="units/s; 0 = skip")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--pin", action="store_true", help="pinned pool (needs a GPU)")
    parser.add_argument("--buffered", action="store_true", help="skip O_DIRECT")
    args = parser.parse_args()
    for d in args.dirs:
        os.makedirs(d, exist_ok=True)
    stripe_counts = args.stripes or list(range(1, len(args.dirs) + 1))
    print(
        f"unit {args.unit_bytes:,} B  units {args.units}  pool {args.pool_slots} slots"
        f"  threads {args.threads}  "
        f"{'buffered' if args.buffered else 'O_DIRECT requested'}"
    )
    for d in args.dirs:
        print(f"  dir {d}: {filesystem_of(d)}")
    for stripes in stripe_counts:
        print(f"\n== {stripes} stripe(s) ==")
        rows = measure(args.dirs, args, stripes)
        width = max(len(name) for name, _ in rows)
        for name, value in rows:
            print(f"  {name:<{width}}  {value}")


if __name__ == "__main__":
    main()
