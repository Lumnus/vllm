# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's side of the runtime control surface.

An outside controller reads the state of every dynamic subsystem and changes
it in a running engine, through `EngineCore.runtime_control_state` and
`EngineCore.runtime_control(op, args)`. Both are utilities: the busy loop
runs them between two steps, never while a step waits for its output (as
`resize_kv_cache`). What they ask of the workers goes through
`collective_rpc` to `GPUWorker.runtime_control`, which each worker runs on
its main thread after every batch launched before it; so a change is in
place on every rank before the next step, and no step sees a half-applied
change.

Operations that the workers carry (EPLB, the expert store and tiers, the KV
pool's settings, the routing trace, the PLE row cache, fault injection) run
in two phases: every rank checks, and only if no rank refuses does every
rank apply. A rank that then fails to apply (an allocation, a file) is
reported, and the ranks that did apply undo it where the operation can be
undone (a written state file or a scrub cannot); the reply says `all`,
`none` or `partial`. Operations of the engine core itself (KV resize, the
encoder towers) run here.
"""

import dataclasses
import time
from typing import Any

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Operations the workers carry; see vllm/v1/worker/runtime_control.py.
WORKER_OPS = (
    "eplb.set",
    "eplb.pause",
    "eplb.resume",
    "eplb.rearrange",
    "eplb.save",
    "eplb.load",
    "store.set",
    "store.scrub",
    "tiers.set",
    "cpu.set",
    "cpu.stats",
    "kv.set",
    "trace.set",
    "ple.set",
    "fault.arm",
    "fault.clear",
)
ENGINE_OPS = (
    "kv.resize",
    "kv.abort",
    "kv.fit",
    "tower.promote",
    "tower.demote",
    "tower.set",
)
DEV_ONLY_OPS = ("fault.arm", "fault.clear")

_TOWER_SETTINGS = {
    "promote_after": int,
    "promote_window_s": float,
    "demote_idle_s": float,
    "idle_location": str,
    "card_reserve_mib": int,
    "host_available_floor_gib": float,
    "shrink_deadline_s": float,
}


class ControlRefused(Exception):
    """An operation the engine will not apply now; nothing was changed."""


class RuntimeControl:
    """Held by the engine core (created on first use)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _rpc(self, phase: str, op: str = "", args: dict | None = None) -> list[dict]:
        return self.engine.collective_rpc(
            "runtime_control",
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            args=(phase, op, args or {}),
        )

    def _worker_op(self, op: str, args: dict) -> dict[str, Any]:
        checks = self._rpc("check", op, args)
        refusals = [c for c in checks if c.get("refusal")]
        if refusals:
            raise ControlRefused(
                "; ".join(f"rank {c['rank']}: {c['refusal']}" for c in refusals)
            )
        replies = self._rpc("apply", op, args)
        done = [r for r in replies if r.get("ok")]
        failed = [r for r in replies if not r.get("ok")]
        reply: dict[str, Any] = {
            "op": op,
            "applied": "all" if not failed else "none" if not done else "partial",
            "ranks": replies,
        }
        if failed and done:
            if all(r.get("undoable") for r in done):
                undone = self._rpc("restore", op, {"ranks": [r["rank"] for r in done]})
                done_ranks = {r["rank"] for r in done}
                reply["rolled_back"] = all(
                    u.get("ok") for u in undone if u["rank"] in done_ranks
                )
                if reply["rolled_back"]:
                    reply["applied"] = "none"
            else:
                reply["rolled_back"] = False
        if failed:
            reply["error"] = "; ".join(
                f"rank {r['rank']}: {r.get('error')}" for r in failed
            )
            logger.warning(
                "Runtime control: %s %s on %d of %d ranks (%s)%s.",
                op,
                "applied" if done else "not applied",
                len(done),
                len(replies),
                reply["error"],
                ""
                if "rolled_back" not in reply
                else "; undone on the others"
                if reply["rolled_back"]
                else "; NOT undone on the others",
            )
        else:
            changes = done[0].get("changes", {}) if done else {}
            logger.info(
                "Runtime control: %s applied on %d of %d ranks%s.",
                op,
                len(done),
                len(replies),
                "".join(f"; {k} {v[0]} -> {v[1]}" for k, v in changes.items()),
            )
        return reply

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state(self, only: list[str] | None = None) -> dict[str, Any]:
        engine = self.engine
        cache_config = engine.vllm_config.cache_config
        kv: dict[str, Any] = {
            "extensible": bool(cache_config.enable_extensible_kv_cache),
            "num_blocks": cache_config.num_gpu_blocks,
            "release_chunk_mib": cache_config.kv_cache_release_chunk_mib,
            "grow_floor_mib": cache_config.kv_cache_grow_torch_floor_mib,
        }
        if kv["extensible"]:
            kv["resize"] = engine.get_kv_cache_resize_state()
        ranks = self._rpc("state", "", {"only": only} if only else None)
        return {
            "dev_mode": bool(envs.VLLM_SERVER_DEV_MODE),
            "operations": sorted(WORKER_OPS + ENGINE_OPS),
            "kv": kv,
            "towers": self._towers_state(),
            "ranks": ranks,
            "ranks_agree": _agreement(ranks),
        }

    def _towers_state(self) -> dict[str, Any]:
        controller = self.engine._tower_residency
        if controller is None:
            return {}
        out = {}
        stats: dict[str, Any] = {}
        delegate = None
        for tower in controller.towers.values():
            delegate = tower._delegate
            break
        if delegate is not None and hasattr(delegate, "host_tower_stats"):
            try:
                stats = delegate.host_tower_stats()
            except Exception:  # noqa: BLE001 - a readout only
                stats = {}
        now = controller._clock()
        for name, tower in controller.towers.items():
            out[name] = {
                **tower.record(),
                "idle_s": now - tower.last_use,
                "hold_s": max(0.0, tower.hold_until - now),
                "options": {k: getattr(tower.options, k) for k in _TOWER_SETTINGS},
                "stats": stats.get(name, {}),
            }
        return out

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def apply(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        if op in DEV_ONLY_OPS and not envs.VLLM_SERVER_DEV_MODE:
            raise ControlRefused(
                f"{op} is available in development mode only (VLLM_SERVER_DEV_MODE=1)"
            )
        if op == "kv.set":
            reply = self._worker_op(op, args)
            if reply["applied"] == "all":
                cache_config = self.engine.vllm_config.cache_config
                if "release_chunk_mib" in args:
                    cache_config.kv_cache_release_chunk_mib = args["release_chunk_mib"]
                if "grow_floor_mib" in args:
                    cache_config.kv_cache_grow_torch_floor_mib = args["grow_floor_mib"]
            return reply
        if op in WORKER_OPS:
            return self._worker_op(op, args)
        if op == "kv.resize":
            num_blocks = args.get("num_blocks")
            if not isinstance(num_blocks, int) or num_blocks <= 0:
                raise ControlRefused("num_blocks must be a positive integer")
            record = self.engine.resize_kv_cache(
                num_blocks, args.get("deadline_s"), bool(args.get("preempt", False))
            )
            return {"op": op, "applied": "all", "kv": record}
        if op == "kv.abort":
            record = self.engine.abort_kv_cache_resize(
                "aborted over the runtime control surface"
            )
            return {"op": op, "applied": "all", "kv": record}
        if op == "kv.fit":
            return self._kv_fit()
        if op.startswith("tower."):
            return self._tower_op(op, args)
        raise ControlRefused(
            f"unknown operation {op!r}; one of {sorted(WORKER_OPS + ENGINE_OPS)}"
        )

    def _kv_fit(self) -> dict[str, Any]:
        """Grow toward the reserved capacity: every rank votes the most its
        memory measurement and the torch headroom floor allow (the grow's own
        vote, see `grow_kv_cache`), so the pool ends at what the floor
        allows on the poorest rank."""
        self.engine._check_kv_cache_resizable()
        ranks = self._rpc("state", "", {"only": ["kv"]})
        capacities = [r["kv"]["capacity_blocks"] for r in ranks if r.get("kv")]
        if not capacities:
            raise ControlRefused("no rank reports an extensible KV cache")
        record = self.engine.resize_kv_cache(min(capacities))
        return {"op": "kv.fit", "applied": "all", "kv": record}

    def _tower(self, args: dict[str, Any]) -> Any:
        controller = self.engine._tower_residency
        if controller is None:
            raise ControlRefused(
                "no encoder tower with residency 'dynamic' (--mm-encoder-host-towers)"
            )
        name = args.get("name")
        if name is None:
            if len(controller.towers) != 1:
                raise ControlRefused(
                    f"name one of the towers {sorted(controller.towers)}"
                )
            name = next(iter(controller.towers))
        tower = controller.towers.get(name)
        if tower is None:
            raise ControlRefused(
                f"no dynamic tower {name!r}; towers: {sorted(controller.towers)}"
            )
        return tower

    def _tower_op(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        from vllm.v1.engine.tower_residency import TowerPhase, TowerPlace

        tower = self._tower(args)
        now = tower._clock()
        if op == "tower.set":
            return self._tower_set(
                tower, {k: v for k, v in args.items() if k != "name"}
            )
        if tower.phase != TowerPhase.STEADY:
            raise ControlRefused(
                f"tower {tower.name} is {tower.phase.value}; wait until it settles"
            )
        if op == "tower.promote":
            if tower.place == TowerPlace.CARD:
                raise ControlRefused(f"tower {tower.name} is on the accelerators")
            if tower._resize_active():
                raise ControlRefused("a KV cache resize is in progress")
            logger.info(
                "Runtime control: tower %s: promotion asked (it is in %s).",
                tower.name,
                tower.place.value,
            )
            tower._start_promotion(now, "asked over the runtime control surface")
        elif op == "tower.demote":
            if tower.place != TowerPlace.CARD:
                raise ControlRefused(
                    f"tower {tower.name} is in {tower.place.value}, not on the "
                    "accelerators"
                )
            logger.info("Runtime control: tower %s: demotion asked.", tower.name)
            tower._demote(now, now - tower.last_use)
        else:
            raise ControlRefused(f"unknown operation {op!r}")
        return {"op": op, "applied": "all", "tower": tower.record()}

    def _tower_set(self, tower: Any, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(changes) - set(_TOWER_SETTINGS))
        if unknown:
            raise ControlRefused(
                f"unknown tower settings {unknown}; known: {sorted(_TOWER_SETTINGS)}"
            )
        old = tower.options
        try:
            # Validated like the start flag (ranges, idle_location's values).
            new = dataclasses.replace(old, **changes)
        except Exception as e:  # noqa: BLE001 - pydantic's validation error
            raise ControlRefused(f"invalid tower settings: {e}") from e
        tower.options = new
        diff = {
            k: [getattr(old, k), getattr(new, k)]
            for k in _TOWER_SETTINGS
            if getattr(old, k) != getattr(new, k)
        }
        for name, (before, after) in diff.items():
            logger.info(
                "Runtime control: tower %s %s %s -> %s.",
                tower.name,
                name,
                before,
                after,
            )
        return {
            "op": "tower.set",
            "applied": "all",
            "changes": diff,
            "tower": tower.record(),
        }


class ExpertTierStatsPoller:
    """The CPU expert tier's counters (--cpu-offload-pinned-gb), per rank, for
    the engine's metrics: read from the workers through the runtime control
    surface's state (part "cpu" and "tiers") every ``interval_s`` seconds,
    between two steps of the busy loop, and handed to the scheduler, whose
    stats carry them to the `vllm:expert_tier_*` gauges. One worker RPC per
    interval; made only when experts are computed on the CPU and stats are
    logged."""

    INTERVAL_S = 10.0

    def __init__(self, engine: Any, interval_s: float = INTERVAL_S) -> None:
        self.engine = engine
        self.interval_s = interval_s
        self._next = 0.0
        self._clock = time.monotonic

    @classmethod
    def maybe_create(cls, engine: Any) -> "ExpertTierStatsPoller | None":
        config = engine.vllm_config
        uva = getattr(getattr(config, "offload_config", None), "uva", None)
        if uva is None or not uva.host_experts_on():
            return None
        if not getattr(engine.scheduler, "log_stats", False):
            return None
        return cls(engine)

    def maybe_poll(self) -> None:
        now = self._clock()
        if now < self._next:
            return
        self._next = now + self.interval_s
        ranks = self.engine.collective_rpc(
            "runtime_control",
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            args=("state", "", {"only": ["cpu", "tiers"]}),
        )
        self.engine.scheduler.expert_tier_stats = expert_tier_series(ranks)


# Series of the CPU expert tier per rank; the keys of
# SchedulerStats.expert_tier_stats and the `vllm:expert_tier_<key>` gauges.
EXPERT_TIER_GAUGES = {
    "routed_card": "Routed (token, expert) calls of this rank on the card's "
    "own rows since start.",
    "routed_pinned": "Routed calls of this rank served from pinned host rows.",
    "routed_cpu": "Routed calls of this rank computed on the CPU.",
    "cpu_layer_calls": "MoE layer calls with work on the CPU.",
    "cpu_idle_layer_calls": "MoE layer calls with nothing on the CPU.",
    "cpu_experts": "Distinct experts per layer call on the CPU, summed.",
    "cpu_compute_seconds": "Seconds the CPU computed experts.",
    "cpu_read_seconds": "Seconds layer calls waited for experts from the drive.",
    "cpu_wait_seconds": "Seconds the host waited for the routing.",
    "host_cache_hits": "Host cache hits.",
    "host_cache_drive_reads": "Experts read from the drive while a step waited.",
    "host_cache_read_ahead": "Experts read into the host cache ahead of use.",
    "host_cache_evicted": "Experts evicted from the host cache.",
    "host_cache_rows": "Experts the host cache holds at most.",
    "promotions": "Slots moved from the CPU into pinned rows.",
    "demotions": "Slots moved from pinned rows to the CPU.",
    "cpu_threads": "Threads of the CPU expert op.",
    "pinned_bound": "Pinned slots per layer (the bound; host_tier_bound).",
    "slots_card": "Slots of every layer on the card now.",
    "slots_pinned": "Slots of every layer in pinned rows now.",
    "slots_cpu": "Slots of every layer computed on the CPU now.",
}


def expert_tier_series(ranks: list[dict]) -> dict[str, dict[str, float]]:
    """The gauges' values per rank (the rank as text) from the workers'
    runtime control state; ranks without a CPU tier are left out."""
    out: dict[str, dict[str, float]] = {}
    for reply in ranks:
        cpu = reply.get("cpu")
        if not cpu:
            continue
        c = cpu["counters"]
        series = {
            "routed_card": c["routed_card"],
            "routed_pinned": c["routed_pinned"],
            "routed_cpu": c["routed_cpu"],
            "cpu_layer_calls": c["layers_with_cpu_calls"],
            "cpu_idle_layer_calls": c["layers_without"],
            "cpu_experts": c["cpu_experts"],
            "cpu_compute_seconds": c["compute_ms"] / 1e3,
            "cpu_read_seconds": c["read_ms"] / 1e3,
            "cpu_wait_seconds": c["wait_ms"] / 1e3,
            "host_cache_hits": c["cache_hits"],
            "host_cache_drive_reads": c["cache_misses"],
            "host_cache_read_ahead": c["cache_prefetched"],
            "host_cache_evicted": c["cache_evicted"],
            "host_cache_rows": cpu["cache"]["rows"],
            "promotions": c["promotions"],
            "demotions": c["demotions"],
            "cpu_threads": cpu["cpu_threads"],
        }
        tiers = reply.get("tiers") or {}
        if tiers:
            bound = tiers.get("settings", {}).get("host_tier_bound")
            if bound is not None:
                series["pinned_bound"] = bound
            now = tiers.get("slots_now", {})
            for tier in ("card", "pinned", "cpu"):
                if tier in now:
                    series[f"slots_{tier}"] = now[tier]
        out[str(reply.get("rank", len(out)))] = {k: float(v) for k, v in series.items()}
    return out


def _agreement(ranks: list[dict]) -> dict[str, bool]:
    """Whether the ranks report the same EPLB settings and step counter (they
    must, or a collective would hang) and the same KV block count."""

    def same(values: list) -> bool:
        return all(v == values[0] for v in values)

    out: dict[str, bool] = {}
    eplb = [r.get("eplb") for r in ranks]
    if all(e is not None for e in eplb) and eplb:
        out["eplb_settings"] = same([e["settings"] for e in eplb])
        out["eplb_step"] = same([e["step"] for e in eplb])
        out["eplb_paused"] = same([e["paused"] for e in eplb])
    kv = [r.get("kv") for r in ranks]
    if all(k is not None for k in kv) and kv:
        out["kv_blocks"] = same([k["committed_blocks"] for k in kv])
    return out
