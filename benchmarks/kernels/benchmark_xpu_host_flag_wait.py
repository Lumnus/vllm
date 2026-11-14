# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The card waiting for a host word (the CPU experts' handshake,
VLLM_CPU_EXPERTS_ASYNC=spin), against the host waiting for the card.

One iteration is one MoE layer's round trip, without the experts:

* ``host-wait`` (the path with the switch unset): a small kernel, the copy
  of a routing-sized buffer to pinned host memory, an event; the host waits
  for the event, "computes" (``--compute-us``), writes the result rows, and
  enqueues their copy to the card and a small kernel that checks them.
* ``spin-<read>``: the same commands, but the host never waits: a host
  thread waits for the event, computes, writes the rows and then the
  sequence number into a pinned word; the card waits for that word with
  ``xpu_host_wait._wait_host_flag_kernel`` (``--read volatile|atomic``)
  before the rows' copy.

Per mode: microseconds per iteration (one synchronize at the end), the card
waits that gave up (word 2), and rows the card read before the host had
written them (``stale``: must be 0; the check kernel compares every row's
first value with the iteration's number). ``--iters 1000000 --modes
spin-volatile`` is the long run (watch the kernel log for an engine reset).
Needs one XPU card and nothing else on it.

    python benchmarks/kernels/benchmark_xpu_host_flag_wait.py \
        --iters 20000 --compute-us 0,500 --rows-kb 400
"""

import argparse
import queue
import threading
import time

import torch

from vllm.model_executor.layers.fused_moe import xpu_host_wait as hw


def spin_us(us: float) -> None:
    end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < end:
        pass


def run(mode: str, iters: int, compute_us: float, rows_kb: int, spins: int) -> dict:
    dev = torch.device("xpu")
    n = max(1, rows_kb * 1024 // 4)
    x = torch.zeros(1024, dtype=torch.int32, device=dev)
    routing_host = torch.empty(1024, dtype=torch.int32, pin_memory=True)
    rows_host = torch.zeros(n, dtype=torch.int32, pin_memory=True)
    rows_np = rows_host.numpy()
    rows_dev = torch.empty(n, dtype=torch.int32, device=dev)
    stale = torch.zeros(1, dtype=torch.int32, device=dev)
    flag = hw.flag_words()
    words = flag.numpy()
    view = hw.device_view(flag, dev)
    jobs: queue.SimpleQueue = queue.SimpleQueue()

    def host_part(ev, seq):
        ev.synchronize()
        spin_us(compute_us)
        rows_np[0] = seq
        rows_np[-1] = seq

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            host_part(*job)
            words[hw.DONE] = job[1]

    thread = None
    if mode.startswith("spin"):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    atomic = mode == "spin-atomic"

    def one(seq: int) -> None:
        x.add_(1)
        routing_host.copy_(x, non_blocking=True)
        ev = torch.xpu.Event()
        ev.record()
        if thread is None:
            host_part(ev, seq)
        else:
            jobs.put((ev, seq))
            hw.wait_host_flag(flag, view, seq, spins, atomic=atomic)
        rows_dev.copy_(rows_host, non_blocking=True)
        stale.add_((rows_dev[:1] != seq).int() + (rows_dev[-1:] != seq).int())

    for s in range(1, 51):  # warm-up (compiles the kernels)
        one(s)
    torch.xpu.synchronize()
    stale.zero_()
    words[hw.TIMED_OUT] = 0
    t0 = time.perf_counter()
    for s in range(51, 51 + iters):
        one(s)
    torch.xpu.synchronize()
    took = time.perf_counter() - t0
    if thread is not None:
        jobs.put(None)
        thread.join()
    return {
        "mode": mode,
        "compute_us": compute_us,
        "us_per_iter": round(took / iters * 1e6, 2),
        "gave_up": int(words[hw.TIMED_OUT]),
        "stale": int(stale.item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--compute-us", default="0,500")
    ap.add_argument("--rows-kb", type=int, default=400)
    ap.add_argument("--spins", type=int, default=1_000_000)
    ap.add_argument("--modes", default="host-wait,spin-volatile,spin-atomic")
    args = ap.parse_args()
    for c in (float(v) for v in args.compute_us.split(",")):
        for mode in args.modes.split(","):
            print(run(mode, args.iters, c, args.rows_kb, args.spins), flush=True)


if __name__ == "__main__":
    main()
