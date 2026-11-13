# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The working set of a layer's expert slots on one rank, chosen by load.

With experts computed on the CPU (``--cpu-offload-pinned-gb``) a layer's
slots sit in three tiers: card rows, pinned rows (read by the card through
UVA) and the CPU. The tier sizes are fixed per layer (the pinned one is a
runtime bound); this module decides *which* slots fill them, from the
routing the host sees anyway (the CPU tier copies every routed id of every
layer call to the host):

* **Temperature.** Per slot, the decayed call rate: an exponentially
  weighted mean of the calls per token the slot received (half-life
  ``ws_halflife_steps`` steps), divided by the mean expert's rate (top-k over
  the logical experts). 1.0 is an average expert, 3.0 three times as busy.
* **Classes.** ``hot`` at or above ``ws_hot``, ``warm`` at or above
  ``ws_warm``, ``cold`` below. A slot drops to a lower class only once it is
  below its class's threshold divided by ``ws_hysteresis``, so a slot near a
  threshold does not flap.
* **Membership.** Between two steps, a layer whose turn it is moves a batch
  of slots (``ws_batch`` per layer, ``ws_units_per_step`` over the process):
  a warm or hot slot on the CPU or in a pinned row enters a faster tier,
  taking a free row or the row of the coldest slot there (by class, then by
  temperature), which moves down. A swap needs the entrant in a higher class
  than the slot it displaces, or ``ws_hysteresis`` times its temperature, and
  the cost model to say the swap pays within one half-life (below). Nothing
  waits for EPLB's interval and nothing is bounded by EPLB's moves per rank:
  these moves change no expert of the rank, only its tier.

The cost model, per slot and step: a slot called with rate ``h`` (calls per
token) in a step of ``T`` tokens is called at least once with probability
``1 - exp(-h T)``; serving it costs nothing extra on the card,
``ws_pinned_call_us`` from a pinned row (the UVA read) and ``ws_cpu_call_us``
on the CPU (the measured cost of one expert at a few tokens: its 2.5 MB of
weights bound it, not the token count). A swap that moves the entrant up from
tier A to B and the displaced slot down from B to A gains
``(c_A - c_B) (p(h_entrant) - p(h_displaced))`` per step; it is taken when that
gain over one half-life is at least ``ws_move_us`` (one expert copied into a
row: a host copy or a drive read, and the card copy for a card row).

The bytes move like EPLB's own moves (``TierPlacement``'s rules R1-R3): a
batch takes free pool rows, fills them in the background (from the CPU tier's
host cache, else the drive), and is committed at a later step on the thread
that enqueues the forward, once every read has landed; a batch that the
placement outran (an EPLB move, a bound change) is dropped and its rows go
back. A step never waits for a read of the working set.
"""

import math
import threading
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

COLD, WARM, HOT = 0, 1, 2
CLASS_NAMES = ("cold", "warm", "hot")
# tier codes as in expert_tier_map (FAST, POOL, HOST)
_FAST, _POOL, _HOST = 0, 1, 2


@dataclass
class WorkingSetSettings:
    """The working set's settings of this process; every field but
    ``stage_rows`` may change while serving (``tiers.set``)."""

    working_set: bool = False
    ws_halflife_steps: float = 64.0
    ws_hot: float = 1.5
    ws_warm: float = 0.5
    ws_hysteresis: float = 1.25
    ws_batch: int = 4
    ws_units_per_step: int = 16
    ws_cpu_call_us: float = 110.0
    ws_pinned_call_us: float = 28.0
    ws_move_us: float = 250.0
    stage_min_calls: int = 0
    stage_rows: int = 0

    @classmethod
    def from_config(cls, config: Any) -> "WorkingSetSettings":
        out = cls()
        for f in fields(cls):
            if hasattr(config, f.name):
                setattr(out, f.name, getattr(config, f.name))
        return out

    def decay(self) -> float:
        """The per-step factor of the temperature's mean."""
        return 0.5 ** (1.0 / max(float(self.ws_halflife_steps), 1.0))


# What `tiers.set` may change, with a check per value (None: valid).
def _positive(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and v > 0


def _count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


RUNTIME_SETTINGS: dict[str, tuple[Any, str]] = {
    "working_set": (lambda v: isinstance(v, bool), "true or false"),
    "ws_halflife_steps": (_positive, "a number of steps > 0"),
    "ws_hot": (_positive, "a temperature > 0 (1.0 = the mean expert)"),
    "ws_warm": (_positive, "a temperature > 0 (1.0 = the mean expert)"),
    "ws_hysteresis": (
        lambda v: _positive(v) and v >= 1.0,
        "a factor >= 1",
    ),
    "ws_batch": (lambda v: _count(v) and v >= 1, "an integer >= 1"),
    "ws_units_per_step": (lambda v: _count(v) and v >= 1, "an integer >= 1"),
    "ws_cpu_call_us": (_positive, "microseconds > 0"),
    "ws_pinned_call_us": (
        lambda v: isinstance(v, int | float) and not isinstance(v, bool) and v >= 0,
        "microseconds >= 0",
    ),
    "ws_move_us": (
        lambda v: isinstance(v, int | float) and not isinstance(v, bool) and v >= 0,
        "microseconds >= 0",
    ),
    "stage_min_calls": (_count, "calls per expert, an integer >= 0 (0: off)"),
}


# --------------------------------------------------------------------------
# Pure decisions (numpy; tested on their own)
# --------------------------------------------------------------------------


def classify(
    temp: np.ndarray,
    prev: np.ndarray | None,
    hot: float,
    warm: float,
    hysteresis: float,
) -> np.ndarray:
    """Class per slot from its temperature: hot, warm or cold; a slot whose
    previous class was higher keeps it while it is at or above that class's
    threshold divided by ``hysteresis``."""
    temp = np.asarray(temp, dtype=np.float64)
    raw = np.where(temp >= hot, HOT, np.where(temp >= warm, WARM, COLD)).astype(np.int8)
    if prev is None:
        return raw
    prev = np.asarray(prev, dtype=np.int8)
    floor = np.where(prev == HOT, hot, np.where(prev == WARM, warm, 0.0)) / max(
        hysteresis, 1.0
    )
    keep = (raw < prev) & (temp >= floor)
    out = raw.copy()
    out[keep] = prev[keep]
    return out


def presence(rate: np.ndarray | float, tokens: float) -> np.ndarray | float:
    """Probability that a slot of call rate ``rate`` (calls per token) is
    called at least once in a step of ``tokens`` tokens."""
    return 1.0 - np.exp(-np.asarray(rate, dtype=np.float64) * max(tokens, 0.0))


@dataclass
class Swap:
    entrant: int  # slot moving up
    displaced: int | None  # slot moving down (None: it took a free row)
    lower: int  # tier the entrant leaves
    upper: int  # tier the entrant enters
    gain_us: float  # per step


def plan_swaps(
    tier: np.ndarray,
    temp: np.ndarray,
    cls: np.ndarray,
    rate: np.ndarray,
    tokens: float,
    pinned_capacity: int,
    s: WorkingSetSettings,
    max_swaps: int,
) -> tuple[np.ndarray, list[Swap]]:
    """The tier per slot after at most ``max_swaps`` swaps, best gain first,
    and the swaps. Each swap reads exactly one expert into a row."""
    t = np.asarray(tier, dtype=np.int8).copy()
    cost = (0.0, float(s.ws_pinned_call_us), float(s.ws_cpu_call_us))
    p = presence(rate, tokens)
    horizon = max(float(s.ws_halflife_steps), 1.0)
    hyst = max(float(s.ws_hysteresis), 1.0)
    # coldest first: by class, then temperature
    order_cold = np.lexsort((temp, cls))
    swaps: list[Swap] = []
    for _ in range(max(0, int(max_swaps))):
        best: Swap | None = None
        for lower, upper in ((_HOST, _FAST), (_POOL, _FAST), (_HOST, _POOL)):
            # only warm and hot slots are reinjected
            ent = np.flatnonzero((t == lower) & (cls >= WARM))
            if ent.size == 0:
                continue
            e = int(ent[np.argmax(temp[ent])])
            f: int | None = None
            if upper == _POOL and int((t == _POOL).sum()) < pinned_capacity:
                f = None  # a free pinned row within the bound
            else:
                vic = order_cold[t[order_cold] == upper]
                if vic.size == 0:
                    continue
                f = int(vic[0])
                if not (cls[e] > cls[f] or temp[e] > hyst * temp[f]):
                    continue
            gain = (cost[lower] - cost[upper]) * (
                float(p[e]) - (0.0 if f is None else float(p[f]))
            )
            if gain * horizon < float(s.ws_move_us) or gain <= 0:
                continue
            if best is None or gain > best.gain_us:
                best = Swap(e, f, lower, upper, gain)
        if best is None:
            break
        t[best.entrant] = best.upper
        if best.displaced is not None:
            t[best.displaced] = best.lower
        swaps.append(best)
    return t, swaps


def class_readout(
    cls: np.ndarray, tier: np.ndarray, num_fast: int, pinned: int
) -> dict[str, int]:
    """Counts of one layer: experts per class, slots per tier, the classes'
    placement, and the shortfalls a bound or a larger card tier would fix."""
    hot = int((cls == HOT).sum())
    warm = int((cls == WARM).sum())
    cold = int((cls == COLD).sum())
    return {
        "hot": hot,
        "warm": warm,
        "cold": cold,
        "card_rows": num_fast,
        "pinned_rows": pinned,
        "hot_on_card": int(((cls == HOT) & (tier == _FAST)).sum()),
        "warm_or_hot_on_cpu": int(((cls >= WARM) & (tier == _HOST)).sum()),
        "cold_on_card": int(((cls == COLD) & (tier == _FAST)).sum()),
        # more hot experts than card rows; more warm+hot than card+pinned
        "card_short": max(0, hot - num_fast),
        "rows_short": max(0, hot + warm - num_fast - pinned),
    }


# --------------------------------------------------------------------------
# The process's clock, budget and settings
# --------------------------------------------------------------------------


@dataclass
class WorkingSetCounters:
    batches: int = 0  # batches started
    committed: int = 0
    dropped: int = 0  # outrun by the placement (EPLB move, bound change)
    failed: int = 0  # a read failed
    units_cache: int = 0  # experts copied from the host cache into rows
    units_drive: int = 0  # experts read from the drive into rows
    swaps: dict[str, int] = field(
        default_factory=lambda: {"cpu>card": 0, "pinned>card": 0, "cpu>pinned": 0}
    )


_KIND = {(_HOST, _FAST): "cpu>card", (_POOL, _FAST): "pinned>card",
         (_HOST, _POOL): "cpu>pinned"}  # fmt: skip


class WorkingSet:
    """One per process: the settings, the step clock that shares the read
    budget between layers, the mean tokens per step, the counters."""

    def __init__(self, settings: WorkingSetSettings):
        self.settings = settings
        self.counters = WorkingSetCounters()
        self.step = 0
        self.left = int(settings.ws_units_per_step)
        self.layers = 1
        self._seen: set[int] = set()
        self.tokens = 1.0  # mean tokens per layer call (EWMA)
        self._lock = threading.Lock()
        self._executor: Any = None

    def kind(self, lower: int, upper: int) -> str:
        return _KIND[(lower, upper)]

    def tick(self, layer: int) -> None:
        """A layer's between-steps call: the second call of a layer starts a
        new step (the budget refills)."""
        if layer in self._seen:
            self.layers = max(1, len(self._seen))
            self._seen.clear()
            self.step += 1
            self.left = int(self.settings.ws_units_per_step)
        self._seen.add(layer)

    def turn(self, ordinal: int) -> int:
        """Reads this layer may start now (0: not its turn). Layers take turns
        so that the budget reaches every layer: a layer's turn comes every
        ``ceil(layers * batch / budget)`` steps."""
        s = self.settings
        batch = max(1, int(s.ws_batch))
        stride = max(1, math.ceil(self.layers * batch / max(1, s.ws_units_per_step)))
        if (ordinal + self.step) % stride != 0 or self.left <= 0:
            return 0
        return min(batch, self.left)

    def spend(self, units: int) -> None:
        self.left -= units

    def observe_tokens(self, tokens: int) -> None:
        self.tokens += (float(tokens) - self.tokens) * 0.05

    def executor(self) -> Any:
        """Two threads for host-cache-to-row copies (torch's copy releases
        the GIL)."""
        with self._lock:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="ws-copy"
                )
            return self._executor

    def snapshot(self) -> dict[str, Any]:
        c = self.counters
        return {
            "batches": c.batches,
            "committed": c.committed,
            "dropped": c.dropped,
            "failed": c.failed,
            "units_cache": c.units_cache,
            "units_drive": c.units_drive,
            "swaps": dict(c.swaps),
            "tokens_per_call": round(self.tokens, 2),
            "step": self.step,
        }


_WS: WorkingSet | None = None
_WS_LOCK = threading.Lock()


def get_working_set(config: Any) -> WorkingSet:
    """The process's working set, made from the expert store config by the
    first layer with a CPU tier."""
    global _WS
    with _WS_LOCK:
        if _WS is None:
            _WS = WorkingSet(WorkingSetSettings.from_config(config))
            if _WS.settings.working_set or _WS.settings.stage_min_calls:
                s = _WS.settings
                logger.info(
                    "Expert working set: %s; temperature half-life %g steps, hot "
                    ">= %g, warm >= %g (x the mean expert), hysteresis %g; %d "
                    "reads per layer batch, %d per step; cost model CPU %g us, "
                    "pinned %g us per expert call, a move %g us; prefill "
                    "staging %s.",
                    "on" if s.working_set else "off",
                    s.ws_halflife_steps,
                    s.ws_hot,
                    s.ws_warm,
                    s.ws_hysteresis,
                    s.ws_batch,
                    s.ws_units_per_step,
                    s.ws_cpu_call_us,
                    s.ws_pinned_call_us,
                    s.ws_move_us,
                    f"from {s.stage_min_calls} calls per expert"
                    if s.stage_min_calls
                    else "off",
                )
        return _WS


def current_working_set() -> WorkingSet | None:
    return _WS


def reset_working_set() -> None:
    """Forget the process's working set (tests; a new model load)."""
    global _WS
    with _WS_LOCK:
        _WS = None
