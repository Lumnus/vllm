# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurements for weight components on NVMe and for per-unit checksums.

``tower``: write a synthetic tower of ``--gib`` GiB as (i) one unit, (ii) 64 MiB
units, (iii) one unit per tensor; then, for each reader thread count, time
bringing it from the drives to host memory (``Component.load``) and releasing
it again. ``integrity``: the cost of a unit check per read, for unit sizes the
expert store sees, with 0, 1 and 4 digest threads.

Pass ``--dir`` directories on the drive under test, and drop or exceed the page
cache yourself: nothing here evicts it, so on a warm cache the figures are the
code's, not the drive's. The stores written here are removed unless ``--keep``.
"""

import argparse
import math
import os
import shutil
import statistics
import time

import numpy as np
import torch

from vllm.model_executor.offloader import expert_store_integrity as integrity
from vllm.model_executor.offloader.expert_store import (
    CHUNK_BYTES,
    ExpertStore,
    ExpertStoreWriter,
    StoreSchema,
    TensorSpec,
)
from vllm.model_executor.offloader.weight_units import (
    ComponentSchema,
    ComponentSpec,
    ComponentStore,
    ComponentStoreWriter,
)

MIB = 1 << 20


def tower_specs(gib: float, blocks: int = 27) -> list[tuple[str, str, tuple[int, ...]]]:
    """A ViT-shaped list of (name, dtype, shape) of about ``gib`` GiB in bf16."""
    hidden = int(math.sqrt(gib * (1 << 30) / (blocks * 24))) // 16 * 16
    out = [("patch.weight", "bfloat16", (hidden, 3, 14, 14))]
    for i in range(blocks):
        p = f"blocks.{i}."
        out += [
            (p + "norm1.weight", "bfloat16", (hidden,)),
            (p + "norm1.bias", "bfloat16", (hidden,)),
            (p + "attn.qkv.weight", "bfloat16", (3 * hidden, hidden)),
            (p + "attn.qkv.bias", "bfloat16", (3 * hidden,)),
            (p + "attn.proj.weight", "bfloat16", (hidden, hidden)),
            (p + "attn.proj.bias", "bfloat16", (hidden,)),
            (p + "norm2.weight", "bfloat16", (hidden,)),
            (p + "norm2.bias", "bfloat16", (hidden,)),
            (p + "mlp.fc1.weight", "bfloat16", (4 * hidden, hidden)),
            (p + "mlp.fc1.bias", "bfloat16", (4 * hidden,)),
            (p + "mlp.fc2.weight", "bfloat16", (hidden, 4 * hidden)),
            (p + "mlp.fc2.bias", "bfloat16", (hidden,)),
        ]
    out.append(("merger.weight", "bfloat16", (hidden * 4, hidden)))
    return out


def make_state(specs) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(1)
    state = {}
    for name, dtype, shape in specs:
        count = math.prod(shape) * 2
        raw = torch.from_numpy(rng.integers(0, 256, count, dtype=np.uint8))
        state[name] = raw.view(getattr(torch, dtype)).reshape(shape)
    return state


def tower(args: argparse.Namespace) -> None:
    specs = tower_specs(args.gib)
    state = make_state(specs)
    total = sum(t.numel() * t.element_size() for t in state.values())
    print(f"tower: {len(state)} tensors, {total / (1 << 30):.3f} GiB")
    layouts = {
        "one unit": ("whole", CHUNK_BYTES),
        "64 MiB units": (64 * MIB, 64 * MIB),
        "per tensor": ("tensor", CHUNK_BYTES),
    }
    print(
        "layout, extents, slot MiB, write s, write MiB/s | per threads: "
        "load s (best of reps) / MiB/s / first-rep s / pool_open s"
    )
    for label, (split, cap) in layouts.items():
        dirs = [os.path.join(d, "bench-weight-units") for d in args.dir]
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        try:
            spec = ComponentSpec.from_state_dict("tower", state, split=split)
            schema = ComponentSchema((spec,), max_extent_bytes=cap)
            t0 = time.perf_counter()
            with ComponentStoreWriter(dirs, schema, "bench") as writer:
                writer.write_component("tower", state, workers=4)
                writer.finalize(checksum=False)
            wt = time.perf_counter() - t0
            cells = []
            for threads in args.threads:
                slot = None
                with ComponentStore.open(
                    dirs,
                    "bench",
                    max(threads * 2, 4) * cap,
                    io_threads=threads,
                    pin=False,
                    check_every=args.check,
                ) as store:
                    comp = store["tower"]
                    slot = comp.slot_bytes
                    runs = []
                    for _ in range(args.reps):
                        loaded = comp.load(threads=threads)
                        runs.append(loaded.timings)
                        comp.release()
                best = min(runs, key=lambda r: r["total_s"])
                cells.append(
                    f"T={threads}: {best['total_s']:.2f} s / {best['mib_s']:,.0f}"
                    f" / {runs[0]['total_s']:.2f} / {runs[0]['pool_open_s']:.2f}"
                )
            print(
                f"{label}, {comp.num_extents}, {slot / MIB:.1f}, {wt:.2f},"
                f" {total / MIB / wt:,.0f} | " + " | ".join(cells),
                flush=True,
            )
        finally:
            if not args.keep:
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)


def digest_cost() -> None:
    print("digest cost per unit (ms), BLAKE3 keyed, threads per call:")
    for size in (2_534_400, 25_171_968):
        buf = np.random.default_rng(0).integers(0, 256, size, dtype=np.uint8)
        view = memoryview(buf)
        cells = []
        for threads in (1, 2, 4, 8, 0):
            integrity.unit_digest(view, 1, threads)
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                integrity.unit_digest(view, 1, threads)
            ms = (time.perf_counter() - t0) / reps * 1e3
            cells.append(f"{'auto' if threads == 0 else threads}: {ms:.3f}")
        print(f"  {size} B: " + ", ".join(cells))


def integrity_cost(args: argparse.Namespace) -> None:
    digest_cost()
    print("end to end: sequential get (read into the slot, then the check), per unit")
    for size, count in ((2_534_400, 96), (25_171_968, 24)):
        d = os.path.join(args.dir[0], f"bench-integrity-{size}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        try:
            schema = StoreSchema.uniform(1, count, [TensorSpec("w", "uint8", (size,))])
            rng = np.random.default_rng(2)
            with ExpertStoreWriter([d], schema, "bench-i") as writer:
                for e in range(count):
                    writer.write(
                        0, e, {"w": rng.integers(0, 256, size, dtype=np.uint8)}
                    )
                writer.finalize(checksum=False)
            for mode, every, threads in (
                ("off", 0, 1),
                ("every read, 1 thread", 1, 1),
                ("every read, 4 threads", 1, 4),
                ("every 8th read, 4 threads", 8, 4),
            ):
                lat = []
                with ExpertStore.open(
                    [d],
                    "bench-i",
                    4 * ((size + 4095) // 4096 * 4096),
                    io_threads=4,
                    pin=False,
                    check_every=every,
                    check_threads=threads,
                ) as store:
                    for rep in range(2):
                        for e in range(count):
                            t0 = time.perf_counter()
                            store.get((0, e)).release()
                            if rep == 1:
                                lat.append((time.perf_counter() - t0) * 1e3)
                print(
                    f"  {size} B, {mode}: p50 {statistics.median(lat):.3f} ms,"
                    f" mean {statistics.fmean(lat):.3f} ms"
                )
        finally:
            shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tower")
    p.add_argument("--dir", nargs="+", required=True)
    p.add_argument("--gib", type=float, default=0.85)
    p.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--check", type=int, default=0, help="check_every for the reads")
    p.add_argument("--keep", action="store_true")
    p = sub.add_parser("integrity")
    p.add_argument("--dir", nargs="+", required=True)
    args = parser.parse_args()
    if args.cmd == "tower":
        tower(args)
    else:
        integrity_cost(args)


if __name__ == "__main__":
    main()
