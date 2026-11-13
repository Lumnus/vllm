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
    "spec.set",
    "sched.set",
    "profile.define",
    "profile.list",
    "profile.apply",
    "profile.state",
    "profile.get",
    "debug.set",
    "debug.state",
    "schema",
)
# Answered by the API server process itself (request defaults it reads).
SERVER_OPS = ("defaults.set", "defaults.state")
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
# Placement settings of a dynamic tower that live on its controller (not in
# its start options): the policy and the second idle timeout (to the drive).
TOWER_POLICIES = ("dynamic", "host", "card")
_TOWER_PLACEMENT = ("policy", "disk_idle_s")


class ControlRefused(Exception):
    """An operation the engine will not apply now; nothing was changed."""


class RestartRequired(ControlRefused):
    """A setting that cannot change in a running engine; the reason says
    why. Its text starts with `restart_required: `, which the HTTP route
    turns into `"restart_required": true`."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"restart_required: {reason}")


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
        out = {
            "dev_mode": bool(envs.VLLM_SERVER_DEV_MODE),
            "operations": sorted(WORKER_OPS + ENGINE_OPS + SERVER_OPS),
            "kv": kv,
            "towers": self._towers_state(),
            "ranks": ranks,
            "ranks_agree": _agreement(ranks),
        }
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            out["kv_demand"] = kv_demand(engine)
            out["spec"] = self._spec_state(scheduler)
            out["sched"] = self._sched_state(scheduler)
        profiles = getattr(engine, "_serving_profiles", None)
        if profiles is not None:
            out["profile"] = profiles.summary()
        return out

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
                **{k: getattr(tower, k, None) for k in _TOWER_PLACEMENT},
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
            unknown = sorted(
                set(args) - {"num_blocks", "tokens", "deadline_s", "preempt"}
            )
            if unknown:
                raise ControlRefused(f"unknown kv.resize arguments {unknown}")
            num_blocks = args.get("num_blocks")
            tokens = args.get("tokens")
            if (num_blocks is None) == (tokens is None):
                raise ControlRefused("give exactly one of num_blocks and tokens")
            if tokens is not None:
                if (
                    not isinstance(tokens, int)
                    or isinstance(tokens, bool)
                    or tokens <= 0
                ):
                    raise ControlRefused("tokens must be a positive integer")
                num_blocks = kv_tokens_to_blocks(self.engine, tokens)
            if not isinstance(num_blocks, int) or num_blocks <= 0:
                raise ControlRefused("num_blocks must be a positive integer")
            record = self.engine.resize_kv_cache(
                num_blocks, args.get("deadline_s"), bool(args.get("preempt", False))
            )
            reply = {"op": op, "applied": "all", "kv": record, "num_blocks": num_blocks}
            if tokens is not None:
                reply["tokens"] = tokens
            return reply
        if op == "kv.abort":
            record = self.engine.abort_kv_cache_resize(
                "aborted over the runtime control surface"
            )
            return {"op": op, "applied": "all", "kv": record}
        if op == "kv.fit":
            if set(args) - {"fit"} or args.get("fit", True) is not True:
                raise ControlRefused('kv.fit takes no arguments (or {"fit": true})')
            return self._kv_fit()
        if op.startswith("tower."):
            return self._tower_op(op, args)
        if op.startswith("profile."):
            return self._profile_op(op, args)
        if op.startswith("debug."):
            return self._debug_op(op, args)
        if op == "schema":
            from vllm.v1.engine.runtime_params import schema

            return {"op": op, "applied": "all", **schema(self.engine, self)}
        if op == "spec.set":
            return self._spec_set(args)
        if op == "sched.set":
            return self._sched_set(args)
        raise ControlRefused(
            f"unknown operation {op!r}; one of "
            f"{sorted(WORKER_OPS + ENGINE_OPS + SERVER_OPS)}"
        )

    # ------------------------------------------------------------------
    # Serving profiles (vllm/v1/engine/serving_profiles.py)
    # ------------------------------------------------------------------

    def _profile_op(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        from vllm.v1.engine.serving_profiles import ServingProfiles

        engine = self.engine
        profiles = getattr(engine, "_serving_profiles", None)
        if profiles is None:
            # Made on first use: an engine nobody gives a profile never has one.
            profiles = ServingProfiles(engine, control=self)
            engine._serving_profiles = profiles
        return profiles.operation(op, args)

    # ------------------------------------------------------------------
    # Debug tracing (vllm/v1/runtime_debug.py)
    # ------------------------------------------------------------------

    # Parts whose lines come from the workers (the rest from the engine core).
    _DEBUG_ON_WORKERS = ("eplb", "cpu")

    def _debug_op(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        from vllm.v1 import runtime_debug

        if op == "debug.state":
            ranks = self._rpc("state", "", {"only": ["debug"]})
            return {
                "op": op,
                "applied": "all",
                "debug": runtime_debug.state(),
                "ranks": [{"rank": r["rank"], **(r.get("debug") or {})} for r in ranks],
            }
        if op != "debug.set":
            raise ControlRefused(f"unknown operation {op!r}")
        refusal = runtime_debug.check(args)
        if refusal:
            if refusal.startswith("restart_required: "):
                raise RestartRequired(refusal[len("restart_required: ") :])
            raise ControlRefused(refusal)
        part = args["part"]
        reply: dict[str, Any] = {"op": op, "applied": "all"}
        if part in self._DEBUG_ON_WORKERS or part.startswith("logger:"):
            reply = {**self._worker_op(op, args), "op": op}
            if reply["applied"] != "all":
                return reply
        if part not in self._DEBUG_ON_WORKERS:
            reply["debug"] = runtime_debug.apply(args)
        return reply

    # ------------------------------------------------------------------
    # The drafter (speculative decoding)
    # ------------------------------------------------------------------

    def _decode_graph_ks(self, k_max: int) -> list[int] | None:
        """Spec token counts whose uniform decode batches have captured
        decode graphs (the runner captures them per query length:
        `VllmConfig.get_decode_cudagraph_query_lens`); None if unknown."""
        from vllm.config.compilation import CUDAGraphMode

        config = self.engine.vllm_config
        try:
            mode = config.compilation_config.cudagraph_mode
            if (
                mode is None
                or not mode.separate_routine()
                or mode.decode_mode() != CUDAGraphMode.FULL
            ):
                return []
            widest = config.uniform_decode_query_len
            lens = config.get_decode_cudagraph_query_lens(widest)
            sampled = widest - config.num_speculative_tokens
        except Exception:  # noqa: BLE001 - a readout only
            return None
        return [k for k in range(k_max + 1) if k + sampled in lens]

    def _spec_state(self, scheduler: Any) -> dict[str, Any]:
        k_max = int(getattr(scheduler, "num_spec_tokens", 0))
        if k_max <= 0:
            return {"drafter": False}
        config = self.engine.vllm_config
        spec_config = getattr(config, "speculative_config", None)
        totals = getattr(scheduler, "spec_totals", None)
        acceptance: dict[str, Any] | None = None
        if totals is not None and totals.num_drafts:
            drafted = totals.num_draft_tokens
            acceptance = {
                "drafts": totals.num_drafts,
                "draft_tokens": drafted,
                "accepted_tokens": totals.num_accepted_tokens,
                "acceptance_rate": (
                    totals.num_accepted_tokens / drafted if drafted else None
                ),
                "mean_acceptance_length": 1
                + totals.num_accepted_tokens / totals.num_drafts,
                "per_position": [
                    (a / d if d else None)
                    for a, d in zip(
                        totals.num_accepted_tokens_per_pos,
                        totals.num_draft_tokens_per_pos,
                    )
                ],
            }
        return {
            "drafter": True,
            "method": getattr(spec_config, "method", None),
            "num_speculative_tokens": scheduler.num_spec_tokens_now,
            "num_speculative_tokens_max": k_max,
            "cap": scheduler.spec_tokens_cap,
            "per_batch_size_schedule": scheduler.dynamic_sd_lookup is not None,
            "decode_graph_tokens": self._decode_graph_ks(k_max),
            "acceptance": acceptance,
            "acceptance_note": (
                None
                if getattr(scheduler, "log_stats", False)
                else "acceptance is counted with stats logging only"
            ),
        }

    def _spec_set(
        self, args: dict[str, Any], check_only: bool = False
    ) -> dict[str, Any]:
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        scheduler = self.engine.scheduler
        known = {"num_speculative_tokens", "enabled", "reset_stats"}
        unknown = sorted(set(args) - known)
        if unknown:
            raise ControlRefused(
                f"unknown drafter settings {unknown}; known: {sorted(known)}"
            )
        k_max = int(scheduler.num_spec_tokens)
        if k_max <= 0:
            raise RestartRequired(
                "no drafter in this engine: the drafter is loaded, and its "
                "buffers and graphs are sized, at start (--speculative-config)"
            )
        k = args.get("num_speculative_tokens")
        enabled = args.get("enabled")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 0):
            raise ControlRefused(
                f"num_speculative_tokens must be an integer >= 0, got {k!r}"
            )
        if enabled is not None and not isinstance(enabled, bool):
            raise ControlRefused(f"enabled must be true or false, got {enabled!r}")
        if k is not None and k > k_max:
            raise RestartRequired(
                f"num_speculative_tokens {k} is above {k_max}, the number the "
                f"drafter's buffers, the rejection sampler and the captured "
                f"graphs were sized for at start; restart with "
                f"--speculative-config num_speculative_tokens={k}"
            )
        if check_only:
            return {}
        old_cap = scheduler.spec_tokens_cap
        if enabled is False:
            cap: int | None = 0
        elif k is not None:
            cap = k
        elif enabled is True:
            cap = None if old_cap == 0 else old_cap
        else:
            cap = old_cap
        if cap == k_max:
            cap = None
        old_now = scheduler.num_spec_tokens_now
        scheduler.spec_tokens_cap = cap
        new_now = scheduler.num_spec_tokens_now
        changes: dict[str, list] = {}
        if old_now != new_now:
            changes["num_speculative_tokens"] = [old_now, new_now]
        if args.get("reset_stats"):
            scheduler.spec_totals = SpecDecodingStats.new(k_max)
            changes["acceptance"] = ["counted", "reset"]
        graphs = self._decode_graph_ks(k_max)
        note = ""
        if graphs is not None and new_now not in graphs:
            note = (
                f"; uniform decode batches of {new_now} draft tokens have no "
                f"captured decode graph (captured: {graphs}) and run on the "
                f"token-keyed graphs or eagerly: the same tokens, more time"
            )
        for name, (a, b) in changes.items():
            logger.info("Runtime control: spec.set %s %s -> %s%s.", name, a, b, note)
        if not changes:
            logger.info("Runtime control: spec.set unchanged (%d).", new_now)
        return {
            "op": "spec.set",
            "applied": "all",
            "changes": changes,
            "spec": self._spec_state(scheduler),
        }

    # ------------------------------------------------------------------
    # Scheduler limits
    # ------------------------------------------------------------------

    def _sched_state(self, scheduler: Any) -> dict[str, Any]:
        config = scheduler.scheduler_config
        return {
            "max_num_active_seqs": scheduler.max_num_active_reqs,
            "max_num_seqs": scheduler.max_num_running_reqs,
            "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "long_prefill_token_threshold": config.long_prefill_token_threshold,
            "chunked_prefill": bool(config.enable_chunked_prefill),
        }

    def _sched_set(
        self, args: dict[str, Any], check_only: bool = False
    ) -> dict[str, Any]:
        scheduler = self.engine.scheduler
        config = scheduler.scheduler_config
        known = {
            "max_num_active_seqs",
            "max_num_scheduled_tokens",
            "long_prefill_token_threshold",
        }
        unknown = sorted(set(args) - known)
        if unknown:
            if "max_num_seqs" in unknown or "max_num_batched_tokens" in unknown:
                raise RestartRequired(
                    "max_num_seqs and max_num_batched_tokens size the model "
                    "runner's request slots, input buffers and captured graphs "
                    "at start; max_num_active_seqs and max_num_scheduled_tokens "
                    "move below them while serving"
                )
            raise ControlRefused(
                f"unknown scheduler settings {unknown}; known: {sorted(known)}"
            )
        seqs = args.get("max_num_active_seqs")
        tokens = args.get("max_num_scheduled_tokens")
        long_prefill = args.get("long_prefill_token_threshold")

        def integer(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        if long_prefill is not None and (
            not integer(long_prefill)
            or not 0 <= long_prefill <= scheduler.max_model_len
        ):
            raise ControlRefused(
                "long_prefill_token_threshold must be an integer in 0..max_model_len "
                f"({scheduler.max_model_len}; 0 = off), got {long_prefill!r}"
            )

        if seqs is not None:
            if not integer(seqs) or seqs < 1:
                raise ControlRefused(
                    f"max_num_active_seqs must be an integer >= 1, got {seqs!r}"
                )
            if seqs > scheduler.max_num_running_reqs:
                raise RestartRequired(
                    f"max_num_active_seqs {seqs} is above max_num_seqs "
                    f"{scheduler.max_num_running_reqs}, which sizes the model "
                    f"runner's request slots and captured graphs at start; "
                    f"restart with --max-num-seqs {seqs}"
                )
        if tokens is not None:
            if not integer(tokens) or tokens < 1:
                raise ControlRefused(
                    f"max_num_scheduled_tokens must be an integer >= 1, got {tokens!r}"
                )
            if tokens > config.max_num_batched_tokens:
                raise RestartRequired(
                    f"max_num_scheduled_tokens {tokens} is above "
                    f"max_num_batched_tokens {config.max_num_batched_tokens}, "
                    f"which sizes the runner's input buffers at start; restart "
                    f"with --max-num-batched-tokens {tokens}"
                )
            floor = 1 + int(getattr(scheduler, "num_spec_tokens", 0))
            if tokens < floor:
                raise ControlRefused(
                    f"max_num_scheduled_tokens must hold one decode step "
                    f"({floor} tokens with the drafter)"
                )
            if not config.enable_chunked_prefill and tokens < scheduler.max_model_len:
                raise ControlRefused(
                    "without chunked prefill a prompt must fit one step: "
                    f"max_num_scheduled_tokens must stay >= max_model_len "
                    f"{scheduler.max_model_len}"
                )
        if check_only:
            return {}
        changes: dict[str, list] = {}
        if seqs is not None and seqs != scheduler.max_num_active_reqs:
            changes["max_num_active_seqs"] = [scheduler.max_num_active_reqs, seqs]
            # Admission reads it every step; requests running above a lower
            # cap finish, new ones wait.
            scheduler.max_num_active_reqs = seqs
        if tokens is not None and tokens != scheduler.max_num_scheduled_tokens:
            changes["max_num_scheduled_tokens"] = [
                scheduler.max_num_scheduled_tokens,
                tokens,
            ]
            scheduler.max_num_scheduled_tokens = tokens
        if (
            long_prefill is not None
            and long_prefill != config.long_prefill_token_threshold
        ):
            # Read from the engine core's scheduler config at every step.
            changes["long_prefill_token_threshold"] = [
                config.long_prefill_token_threshold,
                long_prefill,
            ]
            config.long_prefill_token_threshold = long_prefill
        for name, (a, b) in changes.items():
            logger.info("Runtime control: sched.set %s %s -> %s.", name, a, b)
        if not changes:
            logger.info("Runtime control: sched.set unchanged.")
        return {
            "op": "sched.set",
            "applied": "all",
            "changes": changes,
            "sched": self._sched_state(scheduler),
        }

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
            if getattr(tower, "policy", "dynamic") == "host":
                raise ControlRefused(
                    f"tower {tower.name} has policy host; set the policy instead"
                )
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
            if getattr(tower, "policy", "dynamic") == "card":
                raise ControlRefused(
                    f"tower {tower.name} has policy card; set the policy instead"
                )
            logger.info("Runtime control: tower %s: demotion asked.", tower.name)
            tower._demote(now, now - tower.last_use)
        else:
            raise ControlRefused(f"unknown operation {op!r}")
        return {"op": op, "applied": "all", "tower": tower.record()}

    def _tower_check(self, args: dict[str, Any]) -> None:
        """Raise ControlRefused if `tower.set args` would be refused; changes
        nothing (a serving profile checks before it applies)."""
        tower = self._tower(args)
        changes = {k: v for k, v in args.items() if k != "name"}
        old_options = tower.options
        placement = {
            k: getattr(tower, k) for k in _TOWER_PLACEMENT if hasattr(tower, k)
        }
        try:
            self._tower_set(tower, changes, log=False)
        finally:
            tower.options = old_options
            for k in _TOWER_PLACEMENT:
                if k in placement:
                    setattr(tower, k, placement[k])
                elif hasattr(tower, k):
                    delattr(tower, k)

    def _tower_set(
        self, tower: Any, changes: dict[str, Any], log: bool = True
    ) -> dict[str, Any]:
        known = sorted([*_TOWER_SETTINGS, *_TOWER_PLACEMENT])
        unknown = sorted(set(changes) - set(known))
        if unknown:
            raise ControlRefused(f"unknown tower settings {unknown}; known: {known}")
        placement = {k: changes.pop(k) for k in _TOWER_PLACEMENT if k in changes}
        policy = placement.get("policy", getattr(tower, "policy", "dynamic"))
        if policy not in TOWER_POLICIES:
            raise ControlRefused(
                f"policy must be one of {list(TOWER_POLICIES)}, got {policy!r}"
            )
        disk = placement.get("disk_idle_s", getattr(tower, "disk_idle_s", None))
        if disk is not None and (
            not isinstance(disk, int | float) or isinstance(disk, bool) or disk <= 0
        ):
            raise ControlRefused(
                f"disk_idle_s must be null or a number > 0, got {disk!r}"
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
        # The controller acts on a new policy at its next step (between two
        # steps of the busy loop): "host" demotes, "card" promotes.
        for name, value, default in (
            ("policy", policy, "dynamic"),
            ("disk_idle_s", disk, None),
        ):
            before = getattr(tower, name, default)
            if before != value:
                diff[name] = [before, value]
                setattr(tower, name, value)
        for name, (before, after) in diff.items() if log else ():
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


def kv_tokens_to_blocks(engine: Any, tokens: int) -> int:
    """KV cache blocks that hold `tokens`, at the pool's tokens per block
    now (hybrid models: the size in tokens the pool reports over its
    blocks)."""
    import math

    cache_config = engine.vllm_config.cache_config
    blocks = cache_config.num_gpu_blocks
    size = getattr(cache_config, "kv_cache_size_tokens", None) or (
        blocks * cache_config.block_size
    )
    return int(math.ceil(int(tokens) / (size / blocks)))


def kv_demand(engine: Any) -> dict[str, Any]:
    """What the KV pool is asked for now, from the scheduler (no worker
    call): the measurements an outside orchestrator chooses a serving profile
    by. Live usage counts blocks held by requests; cached-but-free blocks
    hold a reusable prefix and have no owner (the pool's usage counts them as
    free); the prefix-cache hit rate and the preemptions are since start."""
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    usable = max(1, pool.num_gpu_blocks - 1 - pool.num_fenced_blocks)
    running, waiting = scheduler.get_request_counts()
    waiting -= len(getattr(scheduler, "deferred_waiting", ()))
    queries = int(getattr(manager, "prefix_cache_queries_total", 0))
    hits = int(getattr(manager, "prefix_cache_hits_total", 0))
    cached_free = pool.num_cached_free_blocks()
    return {
        "blocks": usable,
        "used": round(float(manager.usage), 4),
        "cached_free_blocks": cached_free,
        "cached_free": round(cached_free / usable, 4),
        "running": running,
        "waiting": max(0, waiting),
        "preemptions": int(getattr(scheduler, "num_preemptions_total", 0)),
        "prefix_cache_queries": queries,
        "prefix_cache_hits": hits,
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
    }


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
