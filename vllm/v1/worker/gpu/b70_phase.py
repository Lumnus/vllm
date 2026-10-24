# SPDX-License-Identifier: Apache-2.0
"""B70-V32-02: per-step GPU/host phase timing for the V2 runner (diagnostic, B70_PHASE_TIMING=1, rank 0 logs).

mark(name) records a timing event on the current stream plus a host timestamp. A step is the marks between two
"exec" marks. Every B70_PHASE_TIMING_EVERY steps (default 256) the medians of the GPU time between consecutive marks
(and of the host time) are logged. Events are resolved only once they have completed (query()), never synchronized."""
import collections
import os
import statistics
import time

import torch

_ON = os.environ.get("B70_PHASE_TIMING", "0") == "1"
_EVERY = int(os.environ.get("B70_PHASE_TIMING_EVERY", "256"))
_cur: list = []
_pending: collections.deque = collections.deque()
_gpu = collections.defaultdict(list)
_host = collections.defaultdict(list)
_steps = 0
_rank = None


def enabled() -> bool:
    return _ON


def mark(name: str) -> None:
    global _cur, _steps, _rank
    if not _ON:
        return
    if torch.xpu.is_current_stream_capturing():
        return
    if _rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            _rank = get_tensor_model_parallel_rank()
        except Exception:
            _rank = 0
    if name == "exec" and _cur:
        _pending.append(_cur)
        _cur = []
        _steps += 1
        _drain()
        if _steps % _EVERY == 0 and _rank == 0:
            _report()
    ev = torch.xpu.Event(enable_timing=True)
    ev.record()
    _cur.append((name, ev, time.perf_counter()))


def _drain() -> None:
    prev_last = None
    while len(_pending) >= 2:
        step, nxt = _pending[0], _pending[1]
        if not nxt[0][1].query():
            break
        _pending.popleft()
        seq = step + [nxt[0]]
        for (n0, e0, h0), (n1, e1, h1) in zip(seq, seq[1:]):
            key = f"{n0}->{n1}"
            _gpu[key].append(e0.elapsed_time(e1))
            _host[key].append((h1 - h0) * 1e3)


def _report() -> None:
    from vllm.logger import init_logger
    log = init_logger(__name__)
    parts = []
    tot_g = tot_h = 0.0
    for key in list(_gpu):
        g = statistics.median(_gpu[key]); h = statistics.median(_host[key])
        tot_g += g; tot_h += h
        parts.append(f"{key} gpu {g:.2f} host {h:.2f}")
    log.info("B70 phase timing (median ms over %d steps, sum gpu %.2f host %.2f): %s",
             len(next(iter(_gpu.values()), [])), tot_g, tot_h, " | ".join(parts))
    _gpu.clear(); _host.clear()
