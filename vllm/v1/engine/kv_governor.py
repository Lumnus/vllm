# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV governor: lends accelerator memory to the experts and the encoder
towers while KV demand is low, and gives it back to the KV cache when the
demand is real and sustained.

Off unless started with ``VLLM_KV_GOVERNOR`` (``dry_run``, ``on`` or a JSON
object of settings) or switched on over the runtime control surface
(``governor.set {"mode": "dry_run"}``). It runs in the engine core's busy
loop between two steps, as the KV cache resize and the tower residency do,
and acts only through operations the engine already has: the tower's
demotion, the fast expert tier's resize (``tiers.set``, a component setting
the fast tier registers), ``kv.fit`` and ``eplb.rearrange``. Every decision
is one INFO line with its inputs; ``dry_run`` logs what it would do and
changes nothing.

Demand, every ``sample_interval_s``, from the scheduler (no worker call):

* live usage ``u``: the share of the pool running requests hold;
* cached-but-free share ``c``: blocks that hold a reusable prefix and have no
  owner (the pool's "usage" counts them as free);
* prefix-cache hit rate ``h`` over the relax window (tokens hit / queried);
* waiting requests ``w`` and preemptions since the last sample ``p``.

A sample is *hot* when ``p > 0`` or (``u >= usage_high`` and
``w >= waiting_min``): the pool already turns requests away. It is *cool*
when ``u + c' <= usage_low``, ``w == 0`` and ``p == 0``, where
``c' = c`` if ``h >= hit_rate_keep`` (cached trees that are being reused
are demand: lending their memory would evict them) and 0 otherwise.

Demand is *real and sustained* (pressure) when at least
``pressure_fraction`` of the samples of the last ``pressure_window_s`` are
hot, over a full window; it is *low* (relaxed) when at least
``relax_fraction`` of the samples of the last ``relax_window_s`` are cool,
over a full window. Otherwise it is neutral and nothing moves. Every move
clears the windows, so the next move needs a whole new window of evidence;
a move in the other direction also waits ``min_dwell_s``; two moves in the
same direction are ``action_interval_s`` apart.

Under pressure the cheapest giving-back move first: a tower on the card
that has been idle for ``tower_idle_s`` (its host copy serves the rare
image), then the fast tier, ``fast_step_slots`` slots per layer at a time,
down to its floor, then the tower even if busy; while pressure lasts the
towers' own promotion is held off (``tower_hold_s``). When relaxed, the fast
tier grows back one step at a time while the pool, after giving up the
step's blocks, would still be at or below ``usage_low``, up to its ceiling;
the tower hold is released (the tower's own controller promotes it when
images come).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

MODES = ("off", "dry_run", "on")
ENV = "VLLM_KV_GOVERNOR"
_RESIZE_ACTIVE = ("draining", "releasing", "unfinished")


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclasses.dataclass
class GovernorSettings:
    mode: str = "off"
    sample_interval_s: float = 1.0
    usage_high: float = 0.90
    usage_low: float = 0.60
    waiting_min: int = 1
    hit_rate_keep: float = 0.25
    pressure_window_s: float = 30.0
    pressure_fraction: float = 0.6
    relax_window_s: float = 300.0
    relax_fraction: float = 0.9
    min_dwell_s: float = 120.0
    action_interval_s: float = 15.0
    tower_idle_s: float = 60.0
    tower_hold_s: float = 300.0
    fast_step_slots: int = 2
    fast_min_slots: int | None = None
    fast_max_slots: int | None = None
    fast_tier_setting: str = "fast_slots"
    kv_fit_after_reclaim: bool = True
    rearrange_after_resize: bool = True
    log_inputs_interval_s: float = 0.0


_FRACTIONS = ("usage_high", "usage_low", "hit_rate_keep")
_SHARES = ("pressure_fraction", "relax_fraction")
_POSITIVE = ("sample_interval_s", "pressure_window_s", "relax_window_s")
_NON_NEGATIVE = (
    "min_dwell_s",
    "action_interval_s",
    "tower_idle_s",
    "tower_hold_s",
    "log_inputs_interval_s",
)


def check_settings(current: GovernorSettings, changes: dict[str, Any]) -> str | None:
    """Why `changes` cannot be applied to `current`, or None."""
    names = {f.name for f in dataclasses.fields(GovernorSettings)}
    unknown = sorted(set(changes) - names)
    if unknown:
        return f"unknown governor settings {unknown}; known: {sorted(names)}"
    want = dataclasses.replace(current, **changes)

    def number(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool)

    if want.mode not in MODES:
        return f"mode must be one of {list(MODES)}, got {want.mode!r}"
    for name in _FRACTIONS:
        v = getattr(want, name)
        if not number(v) or not 0.0 <= v <= 1.0:
            return f"{name} must be a number in [0, 1], got {v!r}"
    for name in _SHARES:
        v = getattr(want, name)
        if not number(v) or not 0.0 < v <= 1.0:
            return f"{name} must be a number in (0, 1], got {v!r}"
    for name in _POSITIVE:
        v = getattr(want, name)
        if not number(v) or v <= 0:
            return f"{name} must be a number > 0, got {v!r}"
    for name in _NON_NEGATIVE:
        v = getattr(want, name)
        if not number(v) or v < 0:
            return f"{name} must be a number >= 0, got {v!r}"
    if want.usage_low >= want.usage_high:
        return (
            f"usage_low ({want.usage_low}) must be below usage_high "
            f"({want.usage_high}): the gap is the hysteresis"
        )
    for name in ("waiting_min", "fast_step_slots"):
        v = getattr(want, name)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            return f"{name} must be an integer >= 1, got {v!r}"
    for name in ("fast_min_slots", "fast_max_slots"):
        v = getattr(want, name)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
            return f"{name} must be null or an integer >= 0, got {v!r}"
    if (
        want.fast_min_slots is not None
        and want.fast_max_slots is not None
        and want.fast_min_slots > want.fast_max_slots
    ):
        return "fast_min_slots must not be above fast_max_slots"
    if not isinstance(want.fast_tier_setting, str) or not want.fast_tier_setting:
        return "fast_tier_setting must be a setting name"
    for name in ("kv_fit_after_reclaim", "rearrange_after_resize"):
        if not isinstance(getattr(want, name), bool):
            return f"{name} must be true or false"
    return None


def settings_from_env(value: str) -> GovernorSettings | None:
    """``VLLM_KV_GOVERNOR``: empty or ``off`` -> None; ``dry_run`` or ``on``;
    or a JSON object of settings (``mode`` defaults to ``dry_run``)."""
    value = (value or "").strip()
    if not value or value == "off":
        return None
    if value in MODES:
        changes: dict[str, Any] = {"mode": value}
    else:
        try:
            changes = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"{ENV}: neither a mode nor a JSON object: {e}") from e
        if not isinstance(changes, dict):
            raise ValueError(f"{ENV}: the JSON value must be an object")
        changes.setdefault("mode", "dry_run")
    refusal = check_settings(GovernorSettings(), changes)
    if refusal:
        raise ValueError(f"{ENV}: {refusal}")
    settings = dataclasses.replace(GovernorSettings(), **changes)
    return None if settings.mode == "off" else settings


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


@dataclasses.dataclass
class KVSignals:
    num_blocks: int
    """Blocks requests can use now (without the null block and a fence)."""
    used: float
    """Live usage: the share of those blocks running requests hold."""
    cached_free: float
    """The share that holds a cached prefix and has no owner."""
    waiting: int
    running: int
    preemptions_total: int
    prefix_queries_total: int
    prefix_hits_total: int
    resize_state: str


def read_engine_signals(engine: Any) -> KVSignals:
    """The scheduler's view, between two steps; no worker is asked. Counting
    the cached-but-free blocks walks the block list (O(blocks))."""
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    usable = max(1, pool.num_gpu_blocks - 1 - pool.num_fenced_blocks)
    cached_free = sum(
        1
        for block in pool.blocks
        if block.ref_cnt == 0 and block.block_hash is not None and not block.is_null
    )
    running, waiting = scheduler.get_request_counts()
    waiting -= len(getattr(scheduler, "deferred_waiting", ()))
    return KVSignals(
        num_blocks=usable,
        used=float(manager.usage),
        cached_free=cached_free / usable,
        waiting=max(0, waiting),
        running=running,
        preemptions_total=int(getattr(scheduler, "num_preemptions_total", 0)),
        prefix_queries_total=int(getattr(manager, "prefix_cache_queries_total", 0)),
        prefix_hits_total=int(getattr(manager, "prefix_cache_hits_total", 0)),
        resize_state=str(engine.get_kv_cache_resize_state().get("state", "idle")),
    )


@dataclasses.dataclass
class _Sample:
    t: float
    hot: bool
    cool: bool
    preempted: int
    queries: int
    hits: int


# --------------------------------------------------------------------------
# What the governor moves (ports; the engine's are below, tests use fakes)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class FastTierView:
    slots: int
    """Card slots per layer now."""
    min_slots: int
    max_slots: int
    blocks_per_slot: float | None
    """KV blocks one slot per layer is worth (all layers), if known."""


class FastTierPort(Protocol):
    def read(self) -> tuple[FastTierView | None, str | None]:
        """(view, None), or (None, why the fast tier cannot move)."""
        ...

    def resize(self, slots: int) -> str | None:
        """Resize to `slots` per layer; None when applied on every rank,
        else why not (nothing changed)."""
        ...


@dataclasses.dataclass
class TowerView:
    name: str
    on_card: bool
    steady: bool
    idle_s: float


class TowerPort(Protocol):
    def read(self) -> list[TowerView]: ...

    def demote(self, name: str) -> str | None: ...

    def hold(self, until: float) -> None:
        """No promotion of any dynamic tower before `until` (the governor's
        clock); `hold(0)` releases a hold the governor set."""
        ...


class EngineTowerPort:
    """The dynamic towers of the engine core (`TowerResidencyController`)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._held: dict[str, float] = {}

    def _towers(self) -> dict[str, Any]:
        controller = getattr(self.engine, "_tower_residency", None)
        return {} if controller is None else controller.towers

    def read(self) -> list[TowerView]:
        from vllm.v1.engine.tower_residency import TowerPhase, TowerPlace

        out = []
        for name, tower in self._towers().items():
            now = tower._clock()
            out.append(
                TowerView(
                    name=name,
                    on_card=tower.place == TowerPlace.CARD,
                    steady=tower.phase == TowerPhase.STEADY,
                    idle_s=now - tower.last_use,
                )
            )
        return out

    def demote(self, name: str) -> str | None:
        from vllm.v1.engine.runtime_control import ControlRefused

        try:
            self.engine.runtime_control("tower.demote", {"name": name})
        except ControlRefused as e:
            return str(e)
        return None

    def hold(self, until: float) -> None:
        # The tower's own clock (monotonic) is the governor's.
        for name, tower in self._towers().items():
            if until > 0:
                if tower.hold_until < until:
                    self._held[name] = until
                    tower.hold_until = until
            elif self._held.pop(name, None) == tower.hold_until:
                tower.hold_until = 0.0


class SurfaceFastTierPort:
    """The fast tier through the runtime control surface: ``tiers.set
    {<fast_tier_setting>: n}`` (a tier setting the fast tier registers on
    every rank with ``register_tier_setting``) and the ranks' ``tiers`` and
    ``kv`` state. Contract read from the state: ``tiers.settings[name]`` is
    the slots per layer now; ``tiers.setting_bounds[name]`` (optional) is
    ``[min, max]`` (default ``[1, now]``); ``tiers.expert_bytes``, the
    number of layers and ``kv.bytes_per_block`` give the blocks one slot per
    layer is worth."""

    def __init__(self, engine: Any, setting: Callable[[], str]) -> None:
        self.engine = engine
        self._setting = setting

    def read(self) -> tuple[FastTierView | None, str | None]:
        name = self._setting()
        control = self.engine._control()
        ranks = control._rpc("state", "", {"only": ["tiers", "kv"]})
        views = []
        for reply in ranks:
            tiers = reply.get("tiers")
            if not tiers:
                return None, "no expert layer keeps its slots in tiers"
            settings = tiers.get("settings") or {}
            if name not in settings:
                return None, (
                    f"the fast tier does not move at run time in this engine "
                    f"(no tier setting {name!r}; {tiers.get('resize_blocked') or ''})"
                ).replace("; )", ")")
            now = int(settings[name])
            lo, hi = (tiers.get("setting_bounds") or {}).get(name, [1, now])
            per_block = (reply.get("kv") or {}).get("bytes_per_block")
            expert = tiers.get("expert_bytes")
            per_slot = (
                expert * int(tiers.get("layers", 0)) / per_block
                if expert and per_block
                else None
            )
            views.append(FastTierView(now, int(lo), int(hi), per_slot))
        if not views:
            return None, "no rank reports"
        return (
            FastTierView(
                slots=min(v.slots for v in views),
                min_slots=max(v.min_slots for v in views),
                max_slots=min(v.max_slots for v in views),
                blocks_per_slot=min(
                    (v.blocks_per_slot for v in views if v.blocks_per_slot),
                    default=None,
                ),
            ),
            None,
        )

    def resize(self, slots: int) -> str | None:
        from vllm.v1.engine.runtime_control import ControlRefused

        try:
            reply = self.engine.runtime_control(
                "tiers.set", {self._setting(): int(slots)}
            )
        except ControlRefused as e:
            return str(e)
        if reply.get("applied") != "all":
            return reply.get("error") or f"applied: {reply.get('applied')}"
        return None


# --------------------------------------------------------------------------
# The governor
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Decision:
    t: float
    kind: str
    """reclaim / lend / level / wait / refused / idle"""
    action: str
    reason: str
    inputs: dict[str, Any]
    dry_run: bool
    result: str | None = None

    def record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class KVGovernor:
    """Held by the engine core; `advance()` between two steps."""

    MAX_DECISIONS = 64

    def __init__(
        self,
        engine: Any,
        settings: GovernorSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        signals: Callable[[], KVSignals] | None = None,
        towers: TowerPort | None = None,
        fast_tier: FastTierPort | None = None,
        control: Callable[[str, dict], dict] | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self._clock = clock
        self._signals = signals or (lambda: read_engine_signals(engine))
        self.towers = towers or EngineTowerPort(engine)
        self.fast_tier = fast_tier or SurfaceFastTierPort(
            engine, lambda: self.settings.fast_tier_setting
        )
        self._control = control or (lambda op, args: engine.runtime_control(op, args))
        self.samples: deque[_Sample] = deque()
        self.level = "neutral"
        self.last: KVSignals | None = None
        self._next_sample = 0.0
        self._next_inputs_log = 0.0
        self._last_action_t: float | None = None
        self._last_direction: str | None = None
        self._said: set[str] = set()
        self.decisions: deque[Decision] = deque(maxlen=self.MAX_DECISIONS)
        self.counts = {"reclaims": 0, "lends": 0, "refused": 0, "samples": 0}
        # dry run: where things would be
        self._sim_fast: int | None = None
        self._sim_towers_off: set[str] = set()
        self._hold_until = 0.0
        if settings.mode != "off":
            self._log_mode()

    def _log_mode(self) -> None:
        logger.info(
            "KV governor: %s (%s).",
            "on" if self.settings.mode == "on" else "dry run, nothing is changed",
            self._settings_text(),
        )

    # ---------------------------------------------------------------- make

    @classmethod
    def maybe_create(cls, engine: Any) -> KVGovernor | None:
        from vllm import envs

        settings = settings_from_env(envs.VLLM_KV_GOVERNOR)
        if settings is None:
            return None
        return cls(engine, settings)

    # ---------------------------------------------------------------- loop

    @property
    def dry_run(self) -> bool:
        return self.settings.mode != "on"

    def seconds_to_next_event(self) -> float | None:
        if self.settings.mode == "off":
            return None
        return max(0.0, self._next_sample - self._clock()) + 0.01

    def advance(self) -> None:
        if self.settings.mode == "off":
            return
        now = self._clock()
        if now < self._next_sample:
            return
        self._next_sample = now + self.settings.sample_interval_s
        try:
            signals = self._signals()
        except Exception:  # noqa: BLE001 - a readout; serving goes on
            logger.exception("KV governor: could not read the KV signals.")
            return
        self._take_sample(now, signals)
        self._maybe_log_inputs(now)
        self._decide(now)

    # ---------------------------------------------------------------- demand

    def _take_sample(self, now: float, sig: KVSignals) -> None:
        s = self.settings
        prev = self.last
        preempted = (
            0 if prev is None else sig.preemptions_total - prev.preemptions_total
        )
        queries = (
            0 if prev is None else sig.prefix_queries_total - prev.prefix_queries_total
        )
        hits = 0 if prev is None else sig.prefix_hits_total - prev.prefix_hits_total
        self.last = sig
        self.counts["samples"] += 1
        hit_rate = self._hit_rate(extra=(queries, hits))
        valued = sig.cached_free if hit_rate >= s.hit_rate_keep else 0.0
        hot = preempted > 0 or (
            sig.used >= s.usage_high and sig.waiting >= s.waiting_min
        )
        cool = sig.used + valued <= s.usage_low and sig.waiting == 0 and preempted == 0
        self.samples.append(_Sample(now, hot, cool, max(0, preempted), queries, hits))
        horizon = now - max(s.relax_window_s, s.pressure_window_s)
        while self.samples and self.samples[0].t < horizon:
            self.samples.popleft()

    def _window(self, now: float, length: float) -> list[_Sample]:
        return [x for x in self.samples if x.t > now - length]

    def _covered(self, now: float, length: float) -> bool:
        """The samples reach back a whole window (none was cleared since)."""
        if not self.samples:
            return False
        first = self.samples[0].t
        return now - first >= length - self.settings.sample_interval_s

    def _hit_rate(self, extra: tuple[int, int] = (0, 0)) -> float:
        queries = extra[0] + sum(x.queries for x in self.samples)
        hits = extra[1] + sum(x.hits for x in self.samples)
        return hits / queries if queries else 0.0

    def _fractions(self, now: float) -> tuple[float, float, int, int]:
        s = self.settings
        hot = self._window(now, s.pressure_window_s)
        cool = self._window(now, s.relax_window_s)
        hot_f = sum(x.hot for x in hot) / len(hot) if hot else 0.0
        cool_f = sum(x.cool for x in cool) / len(cool) if cool else 0.0
        return hot_f, cool_f, len(hot), len(cool)

    def _level(self, now: float) -> str:
        s = self.settings
        hot_f, cool_f, _, _ = self._fractions(now)
        if self._covered(now, s.pressure_window_s) and hot_f >= s.pressure_fraction:
            return "pressure"
        if self._covered(now, s.relax_window_s) and cool_f >= s.relax_fraction:
            return "relaxed"
        return "neutral"

    def inputs(self, now: float | None = None) -> dict[str, Any]:
        now = self._clock() if now is None else now
        s = self.settings
        sig = self.last
        hot_f, cool_f, n_hot, n_cool = self._fractions(now)
        hit_rate = self._hit_rate()
        out: dict[str, Any] = {
            "hot": round(hot_f, 3),
            "hot_window": [n_hot, s.pressure_window_s],
            "cool": round(cool_f, 3),
            "cool_window": [n_cool, s.relax_window_s],
            "hit_rate": round(hit_rate, 3),
            "preempted_in_window": sum(
                x.preempted for x in self._window(now, s.pressure_window_s)
            ),
        }
        if sig is not None:
            valued = sig.cached_free if hit_rate >= s.hit_rate_keep else 0.0
            out.update(
                used=round(sig.used, 3),
                cached_free=round(sig.cached_free, 3),
                demand=round(sig.used + valued, 3),
                waiting=sig.waiting,
                running=sig.running,
                blocks=sig.num_blocks,
                resize=sig.resize_state,
            )
        return out

    @staticmethod
    def _inputs_text(inputs: dict[str, Any]) -> str:
        parts = []
        for key in (
            "used",
            "cached_free",
            "hit_rate",
            "demand",
            "waiting",
            "running",
            "blocks",
            "resize",
        ):
            if key in inputs:
                parts.append(f"{key} {inputs[key]}")
        parts.append(
            f"hot {inputs['hot']} of {inputs['hot_window'][0]} samples in "
            f"{inputs['hot_window'][1]:g} s"
        )
        parts.append(
            f"cool {inputs['cool']} of {inputs['cool_window'][0]} samples in "
            f"{inputs['cool_window'][1]:g} s"
        )
        parts.append(f"preempted {inputs['preempted_in_window']}")
        return ", ".join(parts)

    def _maybe_log_inputs(self, now: float) -> None:
        interval = self.settings.log_inputs_interval_s
        if interval <= 0 or now < self._next_inputs_log:
            return
        self._next_inputs_log = now + interval
        logger.info(
            "KV governor%s: level %s; %s.",
            " (dry run)" if self.dry_run else "",
            self.level,
            self._inputs_text(self.inputs(now)),
        )

    # ---------------------------------------------------------------- decide

    def _decide(self, now: float) -> None:
        level = self._level(now)
        if level != self.level:
            self._note(now, "level", f"{self.level} -> {level}", "", log=True)
            self.level = level
            self._said.clear()
        if level == "neutral":
            return
        sig = self.last
        if sig is not None and sig.resize_state in _RESIZE_ACTIVE:
            self._once(
                now, "resize", f"waiting: a KV cache resize is {sig.resize_state}"
            )
            return
        direction = "reclaim" if level == "pressure" else "lend"
        wait = self._pacing(now, direction)
        if wait is not None:
            return
        if direction == "reclaim":
            self._reclaim(now)
        else:
            self._lend(now)

    def _pacing(self, now: float, direction: str) -> str | None:
        if self._last_action_t is None:
            return None
        since = now - self._last_action_t
        s = self.settings
        if direction == self._last_direction:
            return None if since >= s.action_interval_s else "interval"
        return None if since >= s.min_dwell_s else "dwell"

    # ---- reclaim

    def _reclaim(self, now: float) -> None:
        s = self.settings
        towers = [t for t in self._tower_views() if t.on_card and t.steady]
        idle = [t for t in towers if t.idle_s >= s.tower_idle_s]
        busy = [t for t in towers if t.idle_s < s.tower_idle_s]
        self._hold_towers(now)
        # The cheapest first: an idle tower, the fast tier, a busy tower.
        if idle:
            tower = max(idle, key=lambda t: t.idle_s)
            self._demote(now, tower, f"cheapest: idle {tower.idle_s:.0f} s")
            return
        view, why = self._fast_view()
        floor = None
        if view is not None:
            floor = max(view.min_slots, s.fast_min_slots or 0)
            if view.slots > floor:
                target = max(floor, view.slots - s.fast_step_slots)
                self._resize_fast(
                    now, view, target, "reclaim", "next cheapest: the fast tier"
                )
                return
        if busy:
            tower = max(busy, key=lambda t: t.idle_s)
            self._demote(
                now,
                tower,
                f"the fast tier is at its floor; tower in use {tower.idle_s:.0f} s ago",
            )
            return
        reason = (
            f"the fast tier is at its floor ({view.slots} slots per layer)"
            if view is not None
            else f"the fast tier cannot move ({why})"
        )
        self._once(now, "reclaim-none", f"nothing left to give back: {reason}")

    def _demote(self, now: float, tower: TowerView, why: str) -> None:
        action = f"tower {tower.name} card -> host"
        if self.dry_run:
            self._sim_towers_off.add(tower.name)
            self._acted(now, "reclaim", action, why, None)
            return
        refusal = self.towers.demote(tower.name)
        self._acted(now, "reclaim", action, why, refusal)

    def _hold_towers(self, now: float) -> None:
        until = now + self.settings.tower_hold_s
        self._hold_until = until
        if not self.dry_run:
            self.towers.hold(until)

    # ---- lend

    def _lend(self, now: float) -> None:
        s = self.settings
        if self._hold_until:
            self._hold_until = 0.0
            if not self.dry_run:
                self.towers.hold(0)
            self._note(now, "hold", "tower promotion hold released", "KV demand is low")
        view, why = self._fast_view()
        if view is None:
            self._once(
                now,
                "lend-none",
                f"nothing to lend to: the fast tier cannot move ({why})",
            )
            return
        ceiling = view.max_slots
        if s.fast_max_slots is not None:
            ceiling = min(ceiling, s.fast_max_slots)
        if view.slots >= ceiling:
            self._once(
                now,
                "lend-full",
                f"the fast tier is at its ceiling ({view.slots} slots per layer)",
            )
            return
        target = min(ceiling, view.slots + s.fast_step_slots)
        sig = self.last
        if view.blocks_per_slot and sig is not None:
            give = view.blocks_per_slot * (target - view.slots)
            left = sig.num_blocks - give
            hit_rate = self._hit_rate()
            valued = sig.cached_free if hit_rate >= s.hit_rate_keep else 0.0
            held = (sig.used + valued) * sig.num_blocks
            if left <= 0 or held / left > s.usage_low:
                self._once(
                    now,
                    "lend-room",
                    f"not lending: after giving {give:.0f} blocks the demand "
                    f"would be {held / max(left, 1):.2f} of the pool, above "
                    f"usage_low {s.usage_low}",
                )
                return
        self._resize_fast(now, view, target, "lend", "KV demand is low")

    # ---- the fast tier

    def _fast_view(self) -> tuple[FastTierView | None, str | None]:
        try:
            view, why = self.fast_tier.read()
        except Exception as e:  # noqa: BLE001 - reported in the decision
            return None, f"{type(e).__name__}: {e}"
        if view is not None and self.dry_run and self._sim_fast is not None:
            view = dataclasses.replace(view, slots=self._sim_fast)
        return view, why

    def _resize_fast(
        self, now: float, view: FastTierView, target: int, kind: str, why: str
    ) -> None:
        action = f"fast tier {view.slots} -> {target} slots per layer"
        if self.dry_run:
            self._sim_fast = target
            self._acted(now, kind, action, why, None)
            return
        refusal = self.fast_tier.resize(target)
        if refusal is None:
            if kind == "reclaim" and self.settings.kv_fit_after_reclaim:
                refusal = self._call("kv.fit", {})
                if refusal:
                    refusal = f"resized; kv.fit refused: {refusal}"
            if refusal is None and self.settings.rearrange_after_resize:
                note = self._call("eplb.rearrange", {})
                if note:
                    logger.info(
                        "KV governor: no rearrangement after the resize: %s", note
                    )
        self._acted(now, kind, action, why, refusal)

    def _call(self, op: str, args: dict) -> str | None:
        from vllm.v1.engine.runtime_control import ControlRefused

        try:
            reply = self._control(op, args)
        except ControlRefused as e:
            return str(e)
        if isinstance(reply, dict) and reply.get("applied") not in (None, "all"):
            return reply.get("error") or f"applied: {reply.get('applied')}"
        return None

    def _tower_views(self) -> list[TowerView]:
        try:
            views = self.towers.read()
        except Exception:  # noqa: BLE001
            logger.exception("KV governor: could not read the towers.")
            return []
        if self.dry_run:
            views = [
                dataclasses.replace(v, on_card=False)
                if v.name in self._sim_towers_off
                else v
                for v in views
            ]
        return views

    # ---------------------------------------------------------------- record

    def _acted(
        self, now: float, kind: str, action: str, why: str, refusal: str | None
    ) -> None:
        if refusal is None:
            # Logged with the inputs that decided it.
            self._note(now, kind, action, why, log=True)
            self.counts["reclaims" if kind == "reclaim" else "lends"] += 1
            self._last_action_t = now
            self._last_direction = kind
            # The next move needs a whole new window of evidence.
            self.samples.clear()
            self.level = "neutral"
            self._said.clear()
        else:
            self.counts["refused"] += 1
            # Not again at once: the same pacing as a move.
            self._last_action_t = now
            self._last_direction = kind
            self._note(now, "refused", action, f"{why}; refused: {refusal}", log=True)

    def _once(self, now: float, key: str, text: str) -> None:
        if key in self._said:
            return
        self._said.add(key)
        self._note(now, "wait", text, "", log=True)

    def _note(
        self, now: float, kind: str, action: str, why: str, log: bool = True
    ) -> None:
        inputs = self.inputs(now)
        self.decisions.append(Decision(now, kind, action, why, inputs, self.dry_run))
        if log:
            logger.info(
                "KV governor%s: %s %s%s; inputs: %s.",
                " (dry run, would)"
                if self.dry_run and kind in ("reclaim", "lend")
                else "",
                kind,
                action,
                f" ({why})" if why else "",
                self._inputs_text(inputs),
            )

    # ---------------------------------------------------------------- control

    def _settings_text(self) -> str:
        s = self.settings
        return (
            f"pressure: >= {s.pressure_fraction:g} of {s.pressure_window_s:g} s hot "
            f"(used >= {s.usage_high:g} with >= {s.waiting_min} waiting, or a "
            f"preemption); low: >= {s.relax_fraction:g} of {s.relax_window_s:g} s "
            f"with used + cached (if hit rate >= {s.hit_rate_keep:g}) <= "
            f"{s.usage_low:g} and none waiting; dwell {s.min_dwell_s:g} s, "
            f"interval {s.action_interval_s:g} s; fast tier step "
            f"{s.fast_step_slots}"
        )

    def set(self, changes: dict[str, Any]) -> dict[str, list]:
        """Apply checked settings; returns {name: [old, new]}. Switching the
        mode resets the windows and the dry run's simulated position."""
        old = self.settings
        new = dataclasses.replace(old, **changes)
        diff = {
            f.name: [getattr(old, f.name), getattr(new, f.name)]
            for f in dataclasses.fields(GovernorSettings)
            if getattr(old, f.name) != getattr(new, f.name)
        }
        self.settings = new
        if "mode" in diff:
            self.samples.clear()
            self.level = "neutral"
            self._said.clear()
            self._sim_fast = None
            self._sim_towers_off.clear()
            if old.mode == "on" and self._hold_until:
                self.towers.hold(0)
            self._hold_until = 0.0
            self._next_sample = 0.0
        for name, (a, b) in diff.items():
            logger.info("Runtime control: governor %s %s -> %s.", name, a, b)
        if "mode" in diff and new.mode != "off":
            self._log_mode()
        return diff

    def state(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "mode": self.settings.mode,
            "settings": dataclasses.asdict(self.settings),
            "level": self.level,
            "inputs": self.inputs(now),
            "last_action_s_ago": (
                None if self._last_action_t is None else now - self._last_action_t
            ),
            "last_direction": self._last_direction,
            "tower_hold_s": max(0.0, self._hold_until - now)
            if self._hold_until
            else 0.0,
            "dry_run_position": (
                {
                    "fast_slots": self._sim_fast,
                    "towers_off": sorted(self._sim_towers_off),
                }
                if self.dry_run
                else None
            ),
            "counts": dict(self.counts),
            "decisions": [d.record() for d in self.decisions],
        }
