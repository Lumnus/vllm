# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug tracing switched on and off in a running engine, per part, bounded.

``debug.set {"part": P, "on": true, "duration_s": S, "count": N}`` over the
runtime control surface arms a part in the engine core and on every rank
(each process keeps its own window); it switches itself off after S seconds
or N lines, whichever comes first, and says so in one INFO line. Lines are
written at INFO with the prefix ``[debug <part>]`` into the engine's normal
log, so nothing new is opened on disk; the count bounds the volume and each
line is written between or inside steps by code that already ran there (a
host-side format of numbers it had), so a step is not stalled.

Parts (the code that emits them checks ``on(part)``, a dict lookup, and does
nothing else while the part is off):

* ``eplb``: each synchronous plan (slots that change, per model) and each
  commit of the maps (async passes are seen at their commit);
* ``scheduler``: each admission (tokens, cached tokens, blocks) and each
  preemption;
* ``kv``: KV cache resizes, drains and releases, block-pool fences;
* ``cpu``: each MoE layer call computed on the CPU (experts, tokens, ms);
* ``cpustep``: each MoE layer call with CPU experts, where the host's time
  went (since the previous layer call, posting, enqueuing the card's
  experts, the wait for the routing, listing, the op, the results) and two
  looks at the card the host takes without waiting: had it reached the
  routing when the host waited, had it finished its own experts when the
  CPU's were done;
* ``tiers``: each commit of a layer's tiers (EPLB's rearrangements and tier
  passes, the card tier's resize rounds), each card tier resize, each change
  of the pinned bound and each pinned-row layout that follows it;
* ``tower``: each request for a tower's modalities and each residency step
  that moves it;
* ``placement``: the per-token placement trace of the fused MoE layers
  (``VLLM_XPU_MOE_PLACEMENT_TRACE``): turned on and off by its ``.on`` file;
* ``logger:<name>``: any logger's DEBUG records, shown at INFO.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

PARTS = {
    "eplb": "EPLB plans and map commits",
    "scheduler": "admissions and preemptions",
    "kv": "KV cache resize, drain and release events",
    "cpu": "MoE layer calls computed on the CPU",
    "cpustep": "each MoE layer call with CPU experts: the host's timeline",
    "tiers": "expert tier commits, card tier resizes, pinned bound and rows",
    "tower": "encoder tower requests and residency steps",
    "placement": "per-token expert placement trace (eager steps only)",
    "logger:<name>": "a logger's DEBUG records, shown at INFO",
}
DEFAULT_DURATION_S = 300.0
MAX_DURATION_S = 3600.0
DEFAULT_COUNT = 2000
MAX_COUNT = 100_000


@dataclass
class _Window:
    part: str
    until: float
    left: int
    started: float = field(default_factory=time.monotonic)
    emitted: int = 0
    undo: Any = None


_ACTIVE: dict[str, _Window] = {}
_LAST: dict[str, dict[str, Any]] = {}


def check(args: dict[str, Any]) -> str | None:
    """Why `debug.set args` cannot apply, or None."""
    known = {"part", "on", "duration_s", "count"}
    unknown = sorted(set(args) - known)
    if unknown:
        return f"unknown debug settings {unknown}; known: {sorted(known)}"
    part = args.get("part")
    if not isinstance(part, str) or not (
        part in PARTS or (part.startswith("logger:") and len(part) > 7)
    ):
        return f"part must be one of {sorted(PARTS)}, got {part!r}"
    if part == "logger:<name>":
        return "name the logger: logger:vllm.v1.core.sched.scheduler"
    on = args.get("on", True)
    if not isinstance(on, bool):
        return "on must be true or false"
    duration = args.get("duration_s", DEFAULT_DURATION_S)
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or not 0 < duration <= MAX_DURATION_S
    ):
        return f"duration_s must be in (0, {MAX_DURATION_S:g}], got {duration!r}"
    count = args.get("count", DEFAULT_COUNT)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 0 < count <= MAX_COUNT
    ):
        return f"count must be in 1..{MAX_COUNT}, got {count!r}"
    if part == "placement" and on:
        return _placement_refusal()
    return None


def apply(args: dict[str, Any]) -> dict[str, Any]:
    """Arm or disarm a part in this process; the state after."""
    part = args["part"]
    if args.get("on", True):
        arm(
            part,
            float(args.get("duration_s", DEFAULT_DURATION_S)),
            int(args.get("count", DEFAULT_COUNT)),
        )
    else:
        disarm(part, "asked")
    return state()


def arm(part: str, duration_s: float, count: int) -> None:
    if part in _ACTIVE:
        disarm(part, "re-armed")
    window = _Window(part, time.monotonic() + duration_s, count)
    if part.startswith("logger:"):
        window.undo = _arm_logger(part[len("logger:") :], window)
    elif part == "placement":
        window.undo = _arm_placement()
    _ACTIVE[part] = window
    logger.info(
        "[debug %s] on for at most %.0f s or %d lines (%s).",
        part,
        duration_s,
        count,
        PARTS.get(part, PARTS["logger:<name>"]),
    )


def disarm(part: str, why: str) -> None:
    window = _ACTIVE.pop(part, None)
    if window is None:
        return
    if window.undo is not None:
        try:
            window.undo()
        except Exception:  # noqa: BLE001 - a debug switch
            logger.exception("[debug %s] could not be switched off cleanly", part)
    _LAST[part] = {
        "off": why,
        "lines": window.emitted,
        "seconds": round(time.monotonic() - window.started, 1),
    }
    logger.info(
        "[debug %s] off (%s) after %d lines in %.0f s.",
        part,
        why,
        window.emitted,
        time.monotonic() - window.started,
    )


def on(part: str) -> bool:
    """Whether `part` is armed now (and switches it off when its time is up)."""
    if not _ACTIVE:
        return False
    window = _ACTIVE.get(part)
    if window is None:
        return False
    if time.monotonic() >= window.until:
        disarm(part, "duration")
        return False
    return True


def armed() -> bool:
    return bool(_ACTIVE)


def sweep() -> None:
    """Switch off every part whose time is up (the engine core's busy loop
    calls this between steps while any part is on)."""
    now = time.monotonic()
    for part, window in list(_ACTIVE.items()):
        if now >= window.until:
            disarm(part, "duration")


def emit(part: str, log: logging.Logger, msg: str, *args: Any) -> None:
    """One bounded debug line of `part` (call only after `on(part)`)."""
    window = _ACTIVE.get(part)
    if window is None:
        return
    window.emitted += 1
    window.left -= 1
    log.info("[debug %s] %s", part, msg % args if args else msg)
    if window.left <= 0:
        disarm(part, "count")


def state() -> dict[str, Any]:
    sweep()
    now = time.monotonic()
    return {
        "parts": dict(PARTS),
        "on": {
            part: {
                "seconds_left": round(max(0.0, w.until - now), 1),
                "lines_left": w.left,
                "lines": w.emitted,
            }
            for part, w in _ACTIVE.items()
        },
        "last": dict(_LAST),
    }


# ---------------------------------------------------------------- loggers


class _Promote(logging.Filter):
    """Shows a logger's DEBUG records at INFO (the handlers may stop DEBUG),
    within the window's bounds."""

    def __init__(self, window: _Window) -> None:
        super().__init__()
        self.window = window

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.INFO:
            return True
        window = self.window
        if window.part not in _ACTIVE:
            return False
        if time.monotonic() >= window.until:
            disarm(window.part, "duration")
            return False
        record.levelno = logging.INFO
        record.levelname = "INFO"
        record.msg = f"[debug {window.part}] {record.msg}"
        window.emitted += 1
        window.left -= 1
        if window.left <= 0:
            disarm(window.part, "count")
        return True


def _arm_logger(name: str, window: _Window):
    target = logging.getLogger(name)
    old_level = target.level
    promote = _Promote(window)
    target.addFilter(promote)
    target.setLevel(logging.DEBUG)

    def undo() -> None:
        target.removeFilter(promote)
        target.setLevel(old_level)

    return undo


# ---------------------------------------------------------------- placement


def _placement_refusal() -> str | None:
    from vllm import envs

    if not envs.VLLM_XPU_MOE_PLACEMENT_TRACE:
        return (
            "restart_required: the placement trace's per-layer writers are "
            "built at load when VLLM_XPU_MOE_PLACEMENT_TRACE names a file "
            "prefix; with it set, this switch turns the trace on and off"
        )
    return None


def _arm_placement():
    import os

    from vllm import envs

    flag = f"{envs.VLLM_XPU_MOE_PLACEMENT_TRACE}.on"
    with open(flag, "a"):
        pass

    def undo() -> None:
        if os.path.exists(flag):
            os.remove(flag)

    return undo
