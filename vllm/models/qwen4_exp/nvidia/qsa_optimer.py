# SPDX-License-Identifier: Apache-2.0
"""B70-QSA-P: per-op timer around the QSA ops. Off unless B70_QSA_OPTIMER=1.

Why not the torch profiler: on XPU it starts fine and then hangs the first decode step (graph replay under PTI,
see artifacts/P-profiler-lifecycle.md). This module uses no tracing layer: host perf counters around each op
call, plus XPU event pairs around the same span.

Each wrapped call records, per (rows-per-step T, span name):
  host   wall time of the Python call (launch cost; also any sync inside it)
  dev    device time between the two events recorded around the call (kernel time + any idle gap in the span)
and the same two numbers exclusive of nested spans. Reading: dev ~= host means the device waited on the host
(launch bound); dev >> host means the host ran ahead and the span is kernel bound.
B70_QSA_OPTIMER_MODE=sync drains the queue before and after every LEAF span: exact kernel time per leaf at the
price of overlap (use it for the split, not for step time). 'lead' is the device work queued before the span.

  B70_QSA_OPTIMER=1            on
  B70_QSA_OPTIMER_MODE         events (default) | sync
  B70_QSA_OPTIMER_EVERY        engine steps per window (default 50)
  B70_QSA_OPTIMER_WARMUP       steps ignored at the start (default 20)
  B70_QSA_OPTIMER_DIR          output dir (default /tmp/b70-qsa-optimer); one rank<r>.jsonl per worker
  B70_QSA_OPTIMER_SPANS        comma list of name or name=module:Class.attr to add or override (name=- removes)

A step is one call of the first QSA layer seen by the process (the target model; one call per layer per step).
Measurement only: nothing the model reads is changed, and an error inside the timer never reaches the engine.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time

_PKG = "vllm.models.qwen4_exp.nvidia"
# name -> (target, kind). kind: "container" holds other spans; "leaf" is a kernel group (synced in sync mode).
DEFAULT_SPECS: dict[str, tuple[str, str]] = {
    "indexer": (f"{_PKG}.indexer_qsa:QSAIndexer.forward", "container"),
    "prepare": (f"{_PKG}.indexer_qsa:qsa_prepare", "leaf"),
    "store": (f"{_PKG}.ops.qsa:qsa_store_cache_rows", "leaf"),
    "compress": (f"{_PKG}.ops.qsa:qsa_compress_groups_with_ratio", "leaf"),
    "select_decode": (f"{_PKG}.ops.qsa_indexer:qsa_select_paged_decode", "leaf"),
    "select_prefill": (f"{_PKG}.ops.qsa_indexer:qsa_select_paged_prefill", "leaf"),
    "expand": (f"{_PKG}.ops.qsa_indexer:expand_qsa_block_indices", "leaf"),
    "attn": (f"{_PKG}.qsa:Qwen4ExpQSAFlashAttentionImpl.forward_qsa", "container"),
    "sparse_attn": (f"{_PKG}.ops.qsa:qsa_sparse_paged_attention", "leaf"),
}
ANCHOR = "run_qsa"  # wrapped by the engine patch (qsa.py _qsa_eager_break), not by install()


class XpuBackend:
    def __init__(self):
        import torch

        self.torch = torch
        self.free = []  # events are recycled after a window is read: creating one costs more than recording one

    def event(self):
        e = self.free.pop() if self.free else self.torch.xpu.Event(enable_timing=True)
        e.record()
        return e

    def release(self, events):
        self.free.extend(events)

    def elapsed_ms(self, a, b) -> float:
        return float(a.elapsed_time(b))

    def sync(self) -> None:
        self.torch.xpu.synchronize()

    def capturing(self) -> bool:
        f = getattr(self.torch.xpu, "is_current_stream_capturing", None)
        try:
            return bool(f()) if f else False
        except Exception:
            return False


class _Rec:
    __slots__ = ("name", "T", "parent", "host", "e0", "e1", "lead", "wall", "kids_host", "kids_wall", "kind", "over")


class OpTimer:
    def __init__(self, backend, rank=0, out_dir="/tmp/b70-qsa-optimer", every=50, warmup=20, mode="events",
                 clock=time.perf_counter_ns, kinds=None, log=None):
        self.b, self.rank, self.every, self.warmup, self.mode = backend, rank, every, warmup, mode
        self.clock = clock
        self.kinds = kinds or {n: k for n, (_, k) in DEFAULT_SPECS.items()}
        self.log = log or (lambda s: None)
        self.dir = out_dir
        self.local = threading.local()
        self.anchor_layer = None
        self.step = 0           # engine steps seen
        self.win_steps = {}     # T -> steps in this window
        self.win_period = {}    # T -> [period_ns...]
        self.win_count = 0
        self.over_ns = 0        # time spent inside the timer itself in this window
        self.last_anchor_t = None
        self.last_anchor_T = None
        self.pending: list[_Rec] = []
        self.stats: dict[int, dict[str, dict[str, float]]] = {}
        self.errors = 0
        self.dead = False
        self.windows = 0

    # ---- state -------------------------------------------------------------------------------------------
    def _stack(self):
        s = getattr(self.local, "stack", None)
        if s is None:
            s = self.local.stack = []
        return s

    def active(self) -> bool:
        return not self.dead and self.step > self.warmup

    # ---- anchor ------------------------------------------------------------------------------------------
    def on_anchor(self, layer_name, T):
        """Called at the start of every run_qsa. Returns True if the call should be timed."""
        if self.anchor_layer is None:
            self.anchor_layer = layer_name
        if layer_name == self.anchor_layer:
            now = self.clock()
            if self.last_anchor_t is not None and self.step > self.warmup:
                self.win_period.setdefault(self.last_anchor_T, []).append(now - self.last_anchor_t)
            if self.win_count >= self.every:  # the window holds whole steps: every layer of every step is in
                self.flush()
            self.last_anchor_t, self.last_anchor_T = now, T
            self.step += 1
            if self.step > self.warmup:
                self.win_steps[T] = self.win_steps.get(T, 0) + 1
                self.win_count += 1
        self.local.T = T
        return self.active()

    # ---- span --------------------------------------------------------------------------------------------
    def call(self, name, fn, args, kwargs):
        if self.dead or self.b.capturing():
            return fn(*args, **kwargs)
        kind = "container" if name == ANCHOR else self.kinds.get(name, "leaf")
        try:
            t_in = self.clock()
            r = _Rec()
            r.name, r.kind, r.T = name, kind, getattr(self.local, "T", 0)
            r.kids_host = r.kids_wall = r.over = 0
            st = self._stack()
            r.parent = st[-1] if st else None
            sync = self.mode == "sync" and kind == "leaf"
            if sync:
                a = self.clock()
                self.b.sync()
                r.lead = self.clock() - a
                t_in = self.clock()
            else:
                r.lead = 0
            r.e0 = None if sync else self.b.event()
            st.append(r)
            h0 = self.clock()
            r.over = h0 - t_in            # instrumentation cost before the op
        except Exception:
            self._fail()
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        finally:
            try:
                h1 = self.clock()
                st.pop()
                r.host = h1 - h0
                if sync:
                    r.e1 = None
                    r.over += self.clock() - h1
                    self.b.sync()
                    r.wall = self.clock() - h0          # launch + drain
                else:
                    r.e1 = self.b.event()
                    r.over += self.clock() - h1
                    r.wall = r.host
                if r.parent is not None:
                    r.parent.kids_host += r.host + r.over
                    r.parent.kids_wall += r.wall + r.lead + r.over
                self.over_ns += r.over
                if self.active():
                    self.pending.append(r)
            except Exception:
                self._fail()

    def _fail(self):
        self.errors += 1
        if self.errors >= 5:
            self.dead = True
            self.log("B70-QSA-OPTIMER: disabled after repeated errors")

    # ---- window ------------------------------------------------------------------------------------------
    def flush(self):
        try:
            if self.mode != "sync" and self.pending:
                self.b.sync()
            kid_dev: dict[int, float] = {}
            rows = []
            for r in self.pending:
                dev = self.b.elapsed_ms(r.e0, r.e1) if r.e0 is not None else None
                rows.append((r, dev))
            # inclusive device time of children, summed onto the parent record (by identity)
            for r, dev in rows:
                if r.parent is not None and dev is not None:
                    kid_dev[id(r.parent)] = kid_dev.get(id(r.parent), 0.0) + dev
            for r, dev in rows:
                s = self.stats.setdefault(r.T, {}).setdefault(r.name, dict(
                    n=0, host_ms=0.0, host_excl_ms=0.0, dev_ms=0.0, dev_excl_ms=0.0, wall_ms=0.0, lead_ms=0.0))
                s["n"] += 1
                s["host_ms"] += r.host / 1e6
                s["host_excl_ms"] += (r.host - r.kids_host) / 1e6
                if dev is not None:
                    s["dev_ms"] += dev
                    s["dev_excl_ms"] += dev - kid_dev.get(id(r), 0.0)
                else:
                    s["wall_ms"] += r.wall / 1e6
                    s["lead_ms"] += r.lead / 1e6
            rel = getattr(self.b, "release", None)
            if rel:
                rel([e for r, _ in rows for e in (r.e0, r.e1) if e is not None])
            self.pending = []
            self._write()
        except Exception as e:  # never reach the engine
            self.log(f"B70-QSA-OPTIMER: flush failed: {e!r}")
            self.pending = []
            self._fail()

    def _write(self):
        rec = {"rank": self.rank, "window": self.windows, "t": time.time(), "mode": self.mode, "step": self.step,
               "timer_overhead_ms": self.over_ns / 1e6, "window_steps": self.win_count, "by_T": {}}
        for T in sorted(set(self.win_steps) | set(self.stats)):
            per = self.win_period.get(T, [])
            rec["by_T"][str(T)] = {
                "steps": self.win_steps.get(T, 0),
                "period_ms_mean": (sum(per) / len(per) / 1e6) if per else None,
                "spans": self.stats.get(T, {}),
            }
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, f"rank{self.rank}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"B70-QSA-OPTIMER rank {self.rank} window {self.windows} step {self.step}: wrote "
                 f"{len(rec['by_T'])} T-buckets to {self.dir}")
        self.windows += 1
        self.stats, self.win_steps, self.win_period, self.win_count = {}, {}, {}, 0
        self.over_ns = 0


_timer: OpTimer | None = None
_installed = False


def enabled() -> bool:
    return os.environ.get("B70_QSA_OPTIMER", "") not in ("", "0")


def _resolve(target):
    import importlib

    mod, _, attr = target.partition(":")
    m = importlib.import_module(mod)
    parts = attr.split(".")
    owner = m
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]


def _wrap(name, fn):
    if getattr(fn, "__b70_optimer__", False):
        return fn

    @functools.wraps(fn)
    def w(*a, **k):
        t = _timer
        if t is None or not t.active():
            return fn(*a, **k)
        return t.call(name, fn, a, k)

    w.__b70_optimer__ = True
    return w


def _specs():
    specs = dict(DEFAULT_SPECS)
    for item in filter(None, os.environ.get("B70_QSA_OPTIMER_SPANS", "").split(",")):
        name, _, tgt = item.partition("=")
        if tgt == "-":
            specs.pop(name, None)
        elif tgt:
            specs[name] = (tgt, "leaf")
    return specs


def _install_spans(log):
    for name, (target, _) in _specs().items():
        try:
            owner, attr = _resolve(target)
            setattr(owner, attr, _wrap(name, getattr(owner, attr)))
        except Exception as e:
            log(f"B70-QSA-OPTIMER: span {name} not installed ({target}): {e!r}")


def _init():
    global _timer, _installed
    if _installed:
        return
    _installed = True
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        from vllm.logger import init_logger

        lg = init_logger(__name__).info
        try:
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = int(os.environ.get("RANK", "0"))
    except Exception:
        lg, rank = print, int(os.environ.get("RANK", "0"))
    mode = os.environ.get("B70_QSA_OPTIMER_MODE", "events")
    specs = _specs()
    _timer = OpTimer(
        XpuBackend(), rank=rank, out_dir=os.environ.get("B70_QSA_OPTIMER_DIR", "/tmp/b70-qsa-optimer"),
        every=int(os.environ.get("B70_QSA_OPTIMER_EVERY", "50")),
        warmup=int(os.environ.get("B70_QSA_OPTIMER_WARMUP", "20")), mode=mode,
        kinds={n: k for n, (_, k) in specs.items()}, log=lg)
    _install_spans(lg)
    lg(f"B70-QSA-OPTIMER on (rank {rank}, mode {mode}, every {_timer.every} steps, warmup {_timer.warmup}, "
       f"spans {sorted(specs)})")


def wrap_run_qsa(fn):
    """Engine hook: wrap the inner _run_qsa body (inside @eager_break_during_capture, which records the inner
    callable and replays it eagerly every step). Returns fn unchanged when the timer is off."""
    if not enabled():
        return fn

    @functools.wraps(fn)
    def w(self, projected_qk, *a, **k):
        if not _installed:
            _init()
        t = _timer
        if t is None or t.dead:
            return fn(self, projected_qk, *a, **k)
        try:
            T = int(projected_qk.shape[0])
            timed = t.on_anchor(getattr(self, "layer_name", "?"), T)
        except Exception:
            t._fail()
            return fn(self, projected_qk, *a, **k)
        if not timed:
            return fn(self, projected_qk, *a, **k)
        return t.call(ANCHOR, fn, (self, projected_qk) + a, k)

    return w
