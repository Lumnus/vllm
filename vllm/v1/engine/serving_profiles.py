# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving profiles: named, declarative targets over the runtime control
surface's primitives, applied by an outside orchestrator.

A profile is a JSON object with a ``name``, an optional ``description`` and
any of these sections (a section left out is left as it is):

* ``kv``: ``{"tokens": N}`` or ``{"blocks": N}`` or ``{"fit": true}``, and
  ``deadline_s`` / ``preempt`` for a shrink's drain (``kv.resize``,
  ``kv.fit``);
* ``tiers``: ``fast_slots`` (card slots per layer, an integer or ``"max"`` /
  ``"min"``; the fast tier's own verb ``tiers.set fast_slots``, read and
  written through `FastTierPort`) and ``host_tier_bound`` (pinned slots);
* ``cpu``, ``eplb``, ``spec``, ``sched``: the arguments of ``cpu.set``,
  ``eplb.set``, ``spec.set``, ``sched.set``;
* ``tower``: the arguments of ``tower.set`` (thresholds, ``policy``,
  ``disk_idle_s``; ``name`` with more than one dynamic tower);
* ``defaults``: the request defaults of ``defaults.set``, which the API
  server applies (it reads them; the engine only carries them).

``profile.apply`` checks every section first (nothing changes on a refusal),
then applies in an order that frees accelerator memory before anything takes
it: settings that hold no accelerator memory and a tower whose policy moves
it off the card; the fast tier if it shrinks; the KV cache if it shrinks
(waiting for the drain); the fast tier if it grows; the KV cache if it grows
(or ``fit``); last a tower whose policy brings it to the card, which takes
its bytes from the KV cache. Each operation applies on every rank or on none
(the surface's rule). A step that fails stops the apply: ``profile.state``
then says which steps were applied and which setting stands where; nothing
is undone by itself, and the previous profile is one ``profile.apply`` away.

The engine never switches profiles by itself. ``VLLM_SERVING_PROFILES``
names a JSON file ``{"profiles": [...], "boot": "<name>"}`` whose profiles
are defined at start and whose boot profile is applied in the first steps of
the busy loop.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_SERVING_PROFILES"
SECTIONS = ("kv", "tiers", "cpu", "eplb", "spec", "sched", "tower", "defaults")
_KV_KEYS = ("tokens", "blocks", "fit", "deadline_s", "preempt")
_TIERS_KEYS = ("fast_slots", "host_tier_bound")
_RESIZE_ACTIVE = ("draining", "releasing", "unfinished")
# How long a step may wait for the engine to settle (a tower move).
TOWER_WAIT_S = 300.0


# --------------------------------------------------------------------------
# The fast expert tier (its verb is the fast tier's own; this is the reader)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class FastTierView:
    slots: int
    """Card slots per layer now (the least of the ranks)."""
    min_slots: int
    max_slots: int


class FastTierPort(Protocol):
    def read(self) -> tuple[FastTierView | None, str | None]:
        """(view, None), or (None, why the fast tier cannot move)."""
        ...

    def resize(self, slots: int) -> dict[str, Any]:
        """``tiers.set`` reply; raises ControlRefused (nothing changed)."""
        ...


class SurfaceFastTierPort:
    """The fast tier through the runtime control surface: ``tiers.set
    {"fast_slots": n}``, a tier setting the fast tier registers on every rank
    with ``register_tier_setting``. Read from each rank's ``tiers`` state:
    ``settings.fast_slots`` (slots per layer now) and the optional
    ``setting_bounds.fast_slots`` = ``[min, max]`` (default ``[1, now]``)."""

    SETTING = "fast_slots"

    def __init__(self, control: Any) -> None:
        self.control = control

    def read(self) -> tuple[FastTierView | None, str | None]:
        name = self.SETTING
        views = []
        for reply in self.control._rpc("state", "", {"only": ["tiers"]}):
            tiers = reply.get("tiers")
            if not tiers:
                return None, "no expert layer keeps its slots in tiers"
            settings = tiers.get("settings") or {}
            if name not in settings:
                blocked = tiers.get("resize_blocked")
                return None, (
                    f"the fast tier does not move at run time in this engine "
                    f"(no tier setting {name!r}" + (f"; {blocked})" if blocked else ")")
                )
            now = int(settings[name])
            lo, hi = (tiers.get("setting_bounds") or {}).get(name, [1, now])
            views.append(FastTierView(now, int(lo), int(hi)))
        if not views:
            return None, "no rank reports"
        return (
            FastTierView(
                slots=min(v.slots for v in views),
                min_slots=max(v.min_slots for v in views),
                max_slots=min(v.max_slots for v in views),
            ),
            None,
        )

    def resize(self, slots: int) -> dict[str, Any]:
        return self.control.apply("tiers.set", {self.SETTING: int(slots)})


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def check_profile(profile: Any) -> str | None:
    """Why `profile` is not a profile (its shape; values are checked when
    it is applied, against the engine as it is then), or None."""
    if not isinstance(profile, dict):
        return "a profile is a JSON object"
    name = profile.get("name")
    if not isinstance(name, str) or not name or len(name) > 64:
        return "a profile needs a name (1..64 characters)"
    unknown = sorted(set(profile) - {"name", "description", *SECTIONS})
    if unknown:
        return f"unknown profile sections {unknown}; known: {list(SECTIONS)}"
    for section in SECTIONS:
        if section in profile and not isinstance(profile[section], dict):
            return f"{section} must be an object"
    kv = profile.get("kv")
    if kv is not None:
        unknown = sorted(set(kv) - set(_KV_KEYS))
        if unknown:
            return f"unknown kv keys {unknown}; known: {list(_KV_KEYS)}"
        given = [k for k in ("tokens", "blocks", "fit") if kv.get(k)]
        if len(given) != 1:
            return "kv needs exactly one of tokens, blocks, fit"
        for key in ("tokens", "blocks"):
            v = kv.get(key)
            if v is not None and (
                not isinstance(v, int) or isinstance(v, bool) or v <= 0
            ):
                return f"kv.{key} must be a positive integer"
    tiers = profile.get("tiers")
    if tiers is not None:
        unknown = sorted(set(tiers) - set(_TIERS_KEYS))
        if unknown:
            return f"unknown tiers keys {unknown}; known: {list(_TIERS_KEYS)}"
        fast = tiers.get("fast_slots")
        if (
            fast is not None
            and fast not in ("max", "min")
            and (not isinstance(fast, int) or isinstance(fast, bool) or fast < 0)
        ):
            return 'tiers.fast_slots must be an integer >= 0, "max" or "min"'
    return None


@dataclasses.dataclass
class _Step:
    label: str
    run: Callable[[], str | None]
    """Starts the step; returns None, or "wait" to poll `done`."""
    done: Callable[[], str | None] | None = None
    """None: settled; "wait": not yet; any other text: failed with it."""
    status: str = "pending"
    detail: Any = None


@dataclasses.dataclass
class _Apply:
    name: str
    steps: list[_Step]
    started: float
    index: int = 0
    status: str = "applying"
    message: str = ""
    finished: float | None = None

    def record(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "status": self.status,
            "message": self.message,
            "seconds": (self.finished or time.monotonic()) - self.started,
            "steps": [
                {"step": s.label, "status": s.status, "detail": s.detail}
                for s in self.steps
            ],
        }


class ServingProfiles:
    """Held by the engine core; made on first use or from
    ``VLLM_SERVING_PROFILES``. The only code that applies a profile."""

    POLL_S = 0.5

    def __init__(
        self,
        engine: Any,
        *,
        control: Any = None,
        fast_tier: FastTierPort | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self._control_obj = control
        self._fast = fast_tier
        self._clock = clock
        self.profiles: dict[str, dict[str, Any]] = {}
        self.active: str | None = None
        self.previous: str | None = None
        self.current: _Apply | None = None
        self.last: _Apply | None = None

    @property
    def control(self) -> Any:
        if self._control_obj is None:
            self._control_obj = self.engine._control()
        return self._control_obj

    @property
    def fast_tier(self) -> FastTierPort:
        if self._fast is None:
            self._fast = SurfaceFastTierPort(self.control)
        return self._fast

    @classmethod
    def maybe_create(cls, engine: Any) -> ServingProfiles | None:
        from vllm import envs

        path = envs.VLLM_SERVING_PROFILES
        if not path:
            return None
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
            raise ValueError(
                f"{ENV}: {path} must hold {{'profiles': [...], 'boot': name}}"
            )
        profiles = cls(engine)
        for profile in data["profiles"]:
            refusal = check_profile(profile)
            if refusal:
                raise ValueError(f"{ENV}: {path}: {refusal}")
            profiles.profiles[profile["name"]] = profile
        boot = data.get("boot")
        logger.info(
            "Serving profiles from %s: %s%s.",
            path,
            ", ".join(profiles.profiles),
            f"; boot profile {boot}" if boot else "",
        )
        if boot is not None:
            if boot not in profiles.profiles:
                raise ValueError(f"{ENV}: the boot profile {boot!r} is not defined")
            # Applied from the busy loop's first steps (the workers answer
            # the control surface only once the engine core runs).
            profiles.pending_boot = boot
        return profiles

    pending_boot: str | None = None

    # ---------------------------------------------------------------- ops

    def operation(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        from vllm.v1.engine.runtime_control import ControlRefused

        if op == "profile.define":
            refusal = check_profile(args)
            if refusal:
                raise ControlRefused(refusal)
            name = args["name"]
            if self.current is not None and self.current.name == name:
                raise ControlRefused(f"profile {name} is being applied; redefine after")
            replaced = name in self.profiles
            self.profiles[name] = json.loads(json.dumps(args))
            logger.info(
                "Serving profile %s %s: %s.",
                name,
                "redefined" if replaced else "defined",
                ", ".join(s for s in SECTIONS if s in args) or "no sections",
            )
            return {"op": op, "applied": "all", "profile": name, "replaced": replaced}
        if op == "profile.list":
            return {
                "op": op,
                "applied": "all",
                "active": self.active,
                "profiles": [
                    {
                        "name": p["name"],
                        "description": p.get("description", ""),
                        "sections": [s for s in SECTIONS if s in p],
                    }
                    for p in self.profiles.values()
                ],
            }
        if op == "profile.apply":
            name = args.get("name")
            if name not in self.profiles:
                raise ControlRefused(
                    f"no profile {name!r}; defined: {sorted(self.profiles)}"
                )
            return {"op": op, **self.start(name)}
        if op == "profile.state":
            return {"op": op, "applied": "all", **self.state()}
        raise ControlRefused(f"unknown operation {op!r}")

    # ---------------------------------------------------------------- apply

    def start(self, name: str) -> dict[str, Any]:
        from vllm.v1.engine.runtime_control import ControlRefused

        if self.current is not None:
            raise ControlRefused(
                f"profile {self.current.name} is being applied (step "
                f"{self.current.steps[self.current.index].label}); one at a time"
            )
        profile = self.profiles[name]
        self._check(profile)  # raises ControlRefused; nothing changed
        steps = self._plan(profile)
        self.current = _Apply(name, steps, self._clock())
        logger.info(
            "Serving profile %s: applying (%s).",
            name,
            "; ".join(s.label for s in steps) or "nothing to change",
        )
        self._run()
        if self.current is not None:
            applied = "applying"  # a step waits; profile.state follows it
        elif self.last is not None and self.last.status == "converged":
            applied = "all"
        else:
            applied = "partial"
        apply = self.current or self.last
        assert apply is not None
        return {
            "applied": applied,
            "apply": apply.record(),
            "server_defaults": profile.get("defaults"),
        }

    def advance(self) -> None:
        if self.pending_boot is not None and self.current is None:
            name, self.pending_boot = self.pending_boot, None
            try:
                self.start(name)
            except Exception as e:  # noqa: BLE001 - logged; serving goes on
                logger.warning("Serving profile %s (boot) refused: %s", name, e)
            return
        if self.current is not None:
            self._run()

    def seconds_to_next_event(self) -> float | None:
        if self.current is not None or self.pending_boot is not None:
            return self.POLL_S
        return None

    def _run(self) -> None:
        """Run steps until one waits or all are done (between two steps of
        the busy loop, or inside the utility call that started the apply)."""
        from vllm.v1.engine.runtime_control import ControlRefused

        apply = self.current
        assert apply is not None
        while apply.index < len(apply.steps):
            step = apply.steps[apply.index]
            try:
                if step.status == "pending":
                    outcome = step.run()
                    step.status = "waiting" if outcome == "wait" else "done"
                if step.status == "waiting":
                    assert step.done is not None
                    outcome = step.done()
                    if outcome == "wait":
                        return
                    if outcome is not None:
                        raise ControlRefused(outcome)
                    step.status = "done"
            except ControlRefused as e:
                step.status = "failed"
                step.detail = str(e)
                self._finish(apply, "failed", f"{step.label}: {e}")
                return
            logger.info("Serving profile %s: %s done.", apply.name, step.label)
            apply.index += 1
        self._finish(apply, "converged", "")

    def _finish(self, apply: _Apply, status: str, message: str) -> None:
        apply.status = status
        apply.message = message
        apply.finished = self._clock()
        self.current = None
        self.last = apply
        if status == "converged":
            if self.active != apply.name:
                self.previous = self.active
            self.active = apply.name
            logger.info(
                "Serving profile %s: converged in %.1f s.",
                apply.name,
                apply.finished - apply.started,
            )
        else:
            self.previous = self.active or self.previous
            self.active = None
            done = [s.label for s in apply.steps if s.status == "done"]
            logger.warning(
                "Serving profile %s: stopped at %s. Applied: %s. Nothing was "
                "undone; profile.state shows where each setting stands%s.",
                apply.name,
                message,
                "; ".join(done) or "nothing",
                f"; profile.apply {self.previous} goes back" if self.previous else "",
            )

    # ---------------------------------------------------------------- check

    def _check(self, profile: dict[str, Any]) -> None:
        """Every section against the engine as it is now; raises
        ControlRefused (with RestartRequired kept) and changes nothing."""
        from vllm.v1.engine.runtime_control import ControlRefused

        control = self.control
        engine = self.engine
        refusals: list[str] = []

        def worker(op: str, args: dict[str, Any]) -> None:
            for reply in control._rpc("check", op, args):
                if reply.get("refusal"):
                    refusals.append(f"{op}: rank {reply['rank']}: {reply['refusal']}")

        for section, op in (("cpu", "cpu.set"), ("eplb", "eplb.set")):
            if profile.get(section):
                worker(op, profile[section])
        tiers = profile.get("tiers") or {}
        if "host_tier_bound" in tiers:
            worker("tiers.set", {"host_tier_bound": tiers["host_tier_bound"]})
        if "fast_slots" in tiers:
            view, why = self.fast_tier.read()
            if view is None:
                refusals.append(f"tiers.fast_slots: {why}")
            else:
                target = self._fast_target(tiers["fast_slots"], view)
                if not view.min_slots <= target <= view.max_slots:
                    refusals.append(
                        f"tiers.fast_slots {target} outside "
                        f"[{view.min_slots}, {view.max_slots}]"
                    )
        kv = profile.get("kv")
        if kv:
            try:
                engine._check_kv_cache_resizable()
            except (ValueError, NotImplementedError, RuntimeError) as e:
                refusals.append(f"kv: {e}")
            else:
                if not kv.get("fit"):
                    blocks = self._kv_target(kv)
                    floor = engine.scheduler.min_kv_cache_blocks()
                    if blocks < floor:
                        refusals.append(
                            f"kv: {blocks} blocks is below the {floor} one request "
                            f"of max_model_len needs"
                        )
        for section, check in (
            ("spec", control._spec_set),
            ("sched", control._sched_set),
        ):
            if profile.get(section):
                try:
                    check(profile[section], check_only=True)
                except ControlRefused as e:
                    refusals.append(f"{section}: {e}")
        if profile.get("tower"):
            try:
                control._tower_check(dict(profile["tower"]))
            except ControlRefused as e:
                refusals.append(f"tower: {e}")
        if refusals:
            text = "; ".join(refusals)
            if any("restart_required: " in r for r in refusals):
                text = "restart_required: " + text.replace("restart_required: ", "")
            raise ControlRefused(text)

    # ---------------------------------------------------------------- plan

    def _kv_target(self, kv: dict[str, Any]) -> int:
        if kv.get("blocks"):
            return int(kv["blocks"])
        cache_config = self.engine.vllm_config.cache_config
        blocks = cache_config.num_gpu_blocks
        tokens = getattr(cache_config, "kv_cache_size_tokens", None) or (
            blocks * cache_config.block_size
        )
        per_block = tokens / blocks
        return int(math.ceil(int(kv["tokens"]) / per_block))

    @staticmethod
    def _fast_target(value: Any, view: FastTierView) -> int:
        if value == "max":
            return view.max_slots
        if value == "min":
            return view.min_slots
        return int(value)

    def _plan(self, profile: dict[str, Any]) -> list[_Step]:
        control = self.control
        steps: list[_Step] = []

        def op_step(label: str, op: str, args: dict[str, Any]) -> _Step:
            step = _Step(label, lambda: None)

            def run() -> None:
                reply = control.apply(op, dict(args))
                step.detail = reply.get("changes") or reply.get("applied")
                if reply.get("applied") not in (None, "all"):
                    from vllm.v1.engine.runtime_control import ControlRefused

                    raise ControlRefused(
                        reply.get("error") or str(reply.get("applied"))
                    )

            step.run = run
            return step

        # 1. Settings that hold no accelerator memory.
        for section, op in (
            ("sched", "sched.set"),
            ("spec", "spec.set"),
            ("eplb", "eplb.set"),
            ("cpu", "cpu.set"),
        ):
            if profile.get(section):
                steps.append(op_step(op, op, profile[section]))
        tiers = profile.get("tiers") or {}
        if "host_tier_bound" in tiers:
            steps.append(
                op_step(
                    "tiers.set host_tier_bound",
                    "tiers.set",
                    {"host_tier_bound": tiers["host_tier_bound"]},
                )
            )
        tower = dict(profile.get("tower") or {})
        card_policy = tower.get("policy") == "card"
        if tower:
            first = {
                k: v for k, v in tower.items() if not (k == "policy" and card_policy)
            }
            if len(first) > (1 if "name" in first else 0):
                steps.append(op_step("tower.set", "tower.set", first))
            if tower.get("policy") == "host":
                steps.append(
                    self._tower_wait("tower off the accelerators", tower, False)
                )
        # 2.-5. Accelerator memory: free before take.
        if "fast_slots" in tiers:
            steps.append(self._fast_step(tiers["fast_slots"], shrink=True))
        kv = profile.get("kv")
        if kv and not kv.get("fit"):
            steps.append(self._kv_step(kv, shrink=True))
        if "fast_slots" in tiers:
            steps.append(self._fast_step(tiers["fast_slots"], shrink=False))
        if kv:
            steps.append(self._kv_step(kv, shrink=False))
        # 6. A tower that goes to the card takes its bytes from the KV cache.
        if card_policy:
            args = {"policy": "card"}
            if "name" in tower:
                args["name"] = tower["name"]
            steps.append(op_step("tower.set policy card", "tower.set", args))
            steps.append(self._tower_wait("tower on the accelerators", tower, True))
        return steps

    def _fast_step(self, value: Any, shrink: bool) -> _Step:
        step = _Step(f"fast tier {'down' if shrink else 'up'}", lambda: None)

        def run() -> None:
            view, why = self.fast_tier.read()
            if view is None:
                from vllm.v1.engine.runtime_control import ControlRefused

                raise ControlRefused(why)
            target = self._fast_target(value, view)
            if (shrink and target < view.slots) or (not shrink and target > view.slots):
                reply = self.fast_tier.resize(target)
                if reply.get("applied") not in (None, "all"):
                    from vllm.v1.engine.runtime_control import ControlRefused

                    raise ControlRefused(reply.get("error") or str(reply["applied"]))
                step.detail = [view.slots, target]
            else:
                step.detail = "nothing to do"

        step.run = run
        return step

    def _kv_step(self, kv: dict[str, Any], shrink: bool) -> _Step:
        engine = self.engine
        step = _Step(f"KV cache {'down' if shrink else 'up'}", lambda: None)

        def run() -> str | None:
            now = engine.vllm_config.cache_config.num_gpu_blocks
            if kv.get("fit"):
                reply = self.control.apply("kv.fit", {})
                step.detail = [now, reply["kv"].get("num_blocks")]
                return None
            target = self._kv_target(kv)
            if (shrink and target >= now) or (not shrink and target <= now):
                step.detail = "nothing to do"
                return None
            record = self.control.apply(
                "kv.resize",
                {
                    "num_blocks": target,
                    "deadline_s": kv.get("deadline_s"),
                    "preempt": bool(kv.get("preempt", False)),
                },
            )["kv"]
            step.detail = [now, target]
            return "wait" if record.get("state") in _RESIZE_ACTIVE else None

        def done() -> str | None:
            record = engine.get_kv_cache_resize_state()
            state = record.get("state")
            if state in _RESIZE_ACTIVE:
                return "wait"
            if state in ("finished", "unchanged"):
                return None
            return f"the KV cache shrink ended {state}: {record.get('message', '')}"

        step.run = run
        step.done = done
        return step

    def _tower_wait(self, label: str, tower: dict[str, Any], on_card: bool) -> _Step:
        step = _Step(label, lambda: "wait")
        started: list[float] = []
        failed_before: list[int] = []

        def done() -> str | None:
            from vllm.v1.engine.tower_residency import TowerPhase, TowerPlace

            t = self.control._tower(dict(tower))
            now = self._clock()
            if not started:
                started.append(now)
                failed_before.append(t.failed_promotions)
            if on_card and t.failed_promotions > failed_before[0]:
                return "the tower's promotion failed (see its WARNING line)"
            settled = (
                t.phase == TowerPhase.STEADY
                and (t.place == TowerPlace.CARD) == on_card
                and t._grow_to is None
                and not t._resize_active()
            )
            if settled:
                step.detail = t.place.value
                return None
            if now - started[0] > TOWER_WAIT_S:
                return f"the tower did not settle in {TOWER_WAIT_S:.0f} s"
            return "wait"

        step.done = done
        return step

    # ---------------------------------------------------------------- state

    def summary(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "previous": self.previous,
            "applying": None if self.current is None else self.current.name,
            "defined": sorted(self.profiles),
        }

    def state(self) -> dict[str, Any]:
        out = self.summary()
        apply = self.current or self.last
        out["apply"] = None if apply is None else apply.record()
        name = (
            self.current.name
            if self.current
            else self.active or (self.last.name if self.last else None)
        )
        out["converged"] = self.current is None and self.active is not None
        if name is not None and name in self.profiles:
            out["targets"] = self.targets(self.profiles[name])
            out["server_defaults"] = self.profiles[name].get("defaults")
        return out

    def targets(self, profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Per setting: the profile's target, the engine's value now, and
        whether they match (the API server adds its defaults)."""
        engine = self.engine
        out: dict[str, dict[str, Any]] = {}

        def put(key: str, target: Any, actual: Any) -> None:
            out[key] = {"target": target, "actual": actual, "match": target == actual}

        kv = profile.get("kv")
        if kv:
            now = engine.vllm_config.cache_config.num_gpu_blocks
            resize = engine.get_kv_cache_resize_state()
            if kv.get("fit"):
                out["kv.blocks"] = {"target": "fit", "actual": now, "match": None}
            else:
                put("kv.blocks", self._kv_target(kv), now)
            out["kv.blocks"]["resize"] = resize.get("state")
        ranks: list[dict] = []
        if (
            profile.get("cpu")
            or profile.get("eplb")
            or "host_tier_bound" in (profile.get("tiers") or {})
        ):
            ranks = self.control._rpc("state", "", {"only": ["tiers", "cpu", "eplb"]})
        first = ranks[0] if ranks else {}
        tiers = profile.get("tiers") or {}
        if "fast_slots" in tiers:
            view, why = self.fast_tier.read()
            if view is None:
                out["tiers.fast_slots"] = {
                    "target": tiers["fast_slots"],
                    "actual": None,
                    "match": False,
                    "why": why,
                }
            else:
                put(
                    "tiers.fast_slots",
                    self._fast_target(tiers["fast_slots"], view),
                    view.slots,
                )
        if "host_tier_bound" in tiers:
            settings = (first.get("tiers") or {}).get("settings") or {}
            put(
                "tiers.host_tier_bound",
                tiers["host_tier_bound"],
                settings.get("host_tier_bound"),
            )
        for key, value in (profile.get("cpu") or {}).items():
            settings = (first.get("cpu") or {}).get("settings") or {}
            put(f"cpu.{key}", value, settings.get(key))
        for key, value in (profile.get("eplb") or {}).items():
            settings = (first.get("eplb") or {}).get("settings") or {}
            put(f"eplb.{key}", value, settings.get(key))
        scheduler = getattr(engine, "scheduler", None)
        spec = profile.get("spec") or {}
        if scheduler is not None and spec:
            now = scheduler.num_spec_tokens_now
            if "num_speculative_tokens" in spec:
                put("spec.num_speculative_tokens", spec["num_speculative_tokens"], now)
            if "enabled" in spec:
                put("spec.enabled", spec["enabled"], now > 0)
        sched = profile.get("sched") or {}
        if scheduler is not None:
            if "max_num_active_seqs" in sched:
                put(
                    "sched.max_num_active_seqs",
                    sched["max_num_active_seqs"],
                    scheduler.max_num_active_reqs,
                )
            if "max_num_scheduled_tokens" in sched:
                put(
                    "sched.max_num_scheduled_tokens",
                    sched["max_num_scheduled_tokens"],
                    scheduler.max_num_scheduled_tokens,
                )
        tower = profile.get("tower") or {}
        if tower:
            try:
                t = self.control._tower(dict(tower))
            except Exception as e:  # noqa: BLE001 - a readout
                out["tower"] = {
                    "target": tower,
                    "actual": None,
                    "match": False,
                    "why": str(e),
                }
            else:
                for key, value in tower.items():
                    if key == "name":
                        continue
                    actual = (
                        getattr(t, key, None)
                        if key in ("policy", "disk_idle_s")
                        else getattr(t.options, key, None)
                    )
                    put(f"tower.{key}", value, actual)
                if "policy" in tower and tower["policy"] in ("host", "card"):
                    put(
                        "tower.place",
                        "card" if tower["policy"] == "card" else "not card",
                        t.place.value if t.place.value == "card" else "not card",
                    )
        return out
