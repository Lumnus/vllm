# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker's side of the runtime control surface (`GPUWorker.runtime_control`).

The engine core calls this on every rank through `collective_rpc`, as a
utility between two steps (`vllm/v1/engine/runtime_control.py`); on the
worker the call runs on the main thread after every batch launched before
it, so whatever it changes is in place before the next step on every rank.
Four phases:

- `state`: what the dynamic subsystems of this rank are and hold now;
- `check`: whether an operation can be applied here, changing nothing;
- `apply`: apply it; never raises: an error is the reply, with nothing of
  the operation left applied on this rank where the operation allows;
- `restore`: undo the last `apply` of an operation on this rank (the engine
  core asks when another rank could not apply it).

One INFO line per changed setting, naming the old and the new value.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# Settings other components contribute (e.g. a host tier bound)
# --------------------------------------------------------------------------


@dataclass
class RuntimeSetting:
    """A setting of the expert tiers that a component lets change while
    serving: `get()` returns its value, `check(value)` why it cannot take
    `value` now (or None), `apply(value)` makes the change."""

    get: Callable[[], Any]
    check: Callable[[Any], str | None]
    apply: Callable[[Any], None]


_TIER_SETTINGS: dict[str, RuntimeSetting] = {}

# Why the expert tiers' sizes are fixed (see the readout). The pinned bound
# (host_tier_bound) moves at run time in an engine started with experts
# computed on the CPU, which registers it; the fast tier never does.
TIER_RESIZE_BLOCKED = (
    "the tier sizes are fixed at load in this engine: the fast tier is each "
    "layer's own device parameter of num_fast rows (its storage is what the "
    "kernels and any captured graph read) and the device memory a larger fast "
    "tier needs is not taken from the KV cache by any path; the pinned bound "
    "moves at run time only when the engine started with "
    "--cpu-offload-pinned-gb, because slots served on the CPU need the MoE op "
    "outside the captured graphs, which is decided at start"
)


def register_tier_setting(name: str, setting: RuntimeSetting) -> None:
    """Let `tiers.set` change `name` on this rank (e.g. `host_tier_bound`)."""
    _TIER_SETTINGS[name] = setting


# --------------------------------------------------------------------------
# Where things are
# --------------------------------------------------------------------------


def _rank(worker: Any) -> int:
    return int(getattr(worker, "rank", 0))


def _ep_rank(worker: Any) -> int:
    from vllm.distributed import parallel_state

    try:
        return parallel_state.get_ep_group().device_group.rank()
    except (AssertionError, AttributeError):
        return _rank(worker)


def _eplb(worker: Any) -> Any:
    return getattr(getattr(worker, "model_runner", None), "eplb_state", None)


def _kv_cache(worker: Any) -> Any:
    return getattr(getattr(worker, "model_runner", None), "extensible_kv_cache", None)


def _model(worker: Any) -> Any:
    runner = getattr(worker, "model_runner", None)
    get_model = getattr(runner, "get_model", None)
    if get_model is None:
        return None
    try:
        return get_model()
    except Exception:  # noqa: BLE001 - no model loaded
        return None


def _row_tables(worker: Any) -> list[tuple[str, Any]]:
    """The PLE row tables of the model (modules with a `row_table`)."""
    cached = worker.__dict__.get("_runtime_control_row_tables")
    if cached is not None:
        return cached
    tables: list[tuple[str, Any]] = []
    model = _model(worker)
    if model is not None and hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            table = getattr(module, "row_table", None)
            if table is not None:
                tables.append((name, table))
    worker.__dict__["_runtime_control_row_tables"] = tables
    return tables


def _row_stats(table: Any) -> Any:
    server = getattr(table, "_server", None)
    return getattr(server, "stats", None)


def _store_sessions() -> list[Any]:
    try:
        from vllm.model_executor.offloader.expert_tier_store import sessions
    except ImportError:
        return []
    return sessions()


def _tiers() -> list[Any]:
    try:
        from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import live_tiers
    except ImportError:
        return []
    return live_tiers()


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def _tiers_state() -> dict[str, Any] | None:
    tiers = _tiers()
    if not tiers:
        return None
    from vllm.model_executor.layers.fused_moe.xpu_expert_tiers import (
        pinned_arena_gib,
    )

    def count(tier: int) -> int:
        # slots of every layer in the tier now (moves and promotions included)
        return sum(
            int((t.placement.tier == tier).sum())
            if t.placement is not None
            else (t.num_fast if tier == 0 else t.num_slow if tier == 1 else 0)
            for t in tiers
        )

    return {
        "layers": len(tiers),
        "fast_slots_per_layer": sorted({t.num_fast for t in tiers}),
        # every slot off the card (pinned rows and the CPU together)
        "host_slots_per_layer": sorted({t.num_slow for t in tiers}),
        # the bound on the slots in pinned rows, and the rest on the CPU
        "pinned_slots_per_layer": sorted(
            {
                t.placement.num_pinned if t.placement is not None else t.num_slow
                for t in tiers
            }
        ),
        "cpu_slots_per_layer": sorted(
            {t.placement.num_host if t.placement is not None else 0 for t in tiers}
        ),
        "slots_now": {"card": count(0), "pinned": count(1), "cpu": count(2)},
        "pool_rows_per_layer": sorted({t.num_pool_rows for t in tiers}),
        "expert_bytes": tiers[0].slot_bytes() if tiers[0].slow else None,
        "store_backed_layers": sum(t.store_binding is not None for t in tiers),
        "moves_committed": sum(len(t.move_log) for t in tiers),
        "pinned_gib": round(pinned_arena_gib(), 3),
        "settings": {name: s.get() for name, s in _TIER_SETTINGS.items()},
        "resizable": "host_tier_bound" in _TIER_SETTINGS,
        "resize_blocked": (
            None if "host_tier_bound" in _TIER_SETTINGS else TIER_RESIZE_BLOCKED
        ),
    }


def _cpu_runtime() -> Any:
    """The process's CPU expert tier (--cpu-offload-pinned-gb), or None."""
    try:
        from vllm.model_executor.layers.fused_moe.xpu_cpu_experts import (
            current_runtime,
        )
    except ImportError:
        return None
    return current_runtime()


def _cpu_state() -> dict[str, Any] | None:
    """The CPU expert tier: its settings, its host cache and its counters
    since start (the ones its periodic log line prints)."""
    rt = _cpu_runtime()
    if rt is None:
        return None
    cfg = rt.config
    cache = rt.cache
    with cache._lock:
        cached = len(cache._rows)
    return {
        "isa": rt.isa,
        "settings": {name: getattr(cfg, name) for name in _CPU_SETTINGS},
        "cpu_threads": rt.threads,
        "cache": {
            "rows": cache.num_rows,
            "gib": round(cache.gib, 3),
            "row_bytes": cache.layout.row_bytes,
            "experts_held": cached,
        },
        "counters": rt.snapshot(),
        "max_experts_per_layer_call": rt.stats.max_experts,
    }


def _store_state() -> list[dict[str, Any]]:
    out = []
    for session in _store_sessions():
        cfg = session.config
        entry: dict[str, Any] = {
            "dirs": list(cfg.dirs),
            "mode": session.mode,
            "status": session.status,
            "check_every": cfg.check_every,
            "scrub_every": cfg.scrub_every,
            "stats_interval_s": cfg.stats_interval_s,
            "pool_rows": session.pool_rows,
            "pool_gib_pinned": round(session.pool_bytes_pinned / 2**30, 3),
        }
        if session.is_open:
            store = session.store
            stats = store.stats()
            entry["check_every"] = store.check_every
            entry["stats"] = {
                k: stats[k]
                for k in (
                    "reads",
                    "bytes_read",
                    "read_errors",
                    "hits",
                    "misses",
                    "evictions",
                    "checked",
                    "check_failures",
                    "into_read_mode",
                    "into_reads",
                    "read_ms",
                )
                if k in stats
            }
        out.append(entry)
    return out


def _kv_state(worker: Any) -> dict[str, Any] | None:
    kv_cache = _kv_cache(worker)
    if kv_cache is None:
        return None
    chunk = kv_cache.buffer.max_chunk_bytes
    cache_config = worker.cache_config
    return {
        "committed_blocks": kv_cache.num_committed_blocks,
        "capacity_blocks": kv_cache.capacity_blocks,
        "bytes_per_block": kv_cache.bytes_per_block,
        "physical_bytes": kv_cache.buffer.physical_bytes,
        "release_chunk_bytes": chunk,
        "release_chunk_mib": cache_config.kv_cache_release_chunk_mib,
        "grow_floor_mib": cache_config.kv_cache_grow_torch_floor_mib,
        "torch_memory_limit": kv_cache.torch_memory_limit,
        "last_grow": kv_cache.last_grow,
    }


def _trace_state(eplb: Any) -> dict[str, Any] | None:
    if eplb is None:
        return None
    config = eplb.parallel_config.eplb_config
    trace = eplb._trace
    return {
        "enabled": trace is not None,
        "path": None if trace is None else trace.path,
        "interval": config.trace_interval,
        "sample": config.trace_sample,
        "rotate_mb": config.trace_rotate_mb,
        "keep": config.trace_keep,
        "steps": None if trace is None else trace.steps,
        "dropped": None if trace is None else trace.dropped,
    }


def _ple_state(worker: Any) -> list[dict[str, Any]]:
    out = []
    for name, table in _row_tables(worker):
        stats = _row_stats(table)
        entry: dict[str, Any] = {
            "module": name,
            "kind": type(table).__name__,
            "on_disk": stats is not None,
        }
        slab = getattr(table, "_slab", None)
        if slab is not None:
            entry["cache_rows"] = int(slab.shape[0])
        if stats is not None:
            lookups = stats.lookups
            entry.update(
                stats_interval_s=stats.interval_s,
                steps=stats.steps,
                lookups=lookups,
                hits=stats.hits,
                misses=stats.misses,
                reads=stats.reads,
                from_lookahead=stats.from_lookahead,
                hit_rate=(stats.hits / lookups) if lookups else None,
                seconds=time.monotonic() - stats.started,
            )
        out.append(entry)
    return out


def state(worker: Any, only: list[str] | None = None) -> dict[str, Any]:
    """This rank's readout; `only` names the parts wanted (all by default)."""
    from vllm.distributed.eplb import fault_injection

    eplb = _eplb(worker)

    def eplb_summary() -> dict | None:
        if eplb is None or not eplb.model_states:
            return None
        from vllm.distributed.eplb.runtime_control import summary

        return summary(eplb)

    parts: dict[str, Callable[[], Any]] = {
        "eplb": eplb_summary,
        "tiers": _tiers_state,
        "cpu": _cpu_state,
        "store": _store_state,
        "kv": lambda: _kv_state(worker),
        "trace": lambda: _trace_state(eplb),
        "ple": lambda: _ple_state(worker),
        "faults": lambda: {
            "armed": fault_injection.armed(),
            "fired": fault_injection.fired(),
        },
    }
    out: dict[str, Any] = {"rank": _rank(worker), "ep_rank": _ep_rank(worker)}
    for name, read in parts.items():
        if only is None or name in only:
            out[name] = read()
    return out


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


class Refused(Exception):
    """The operation cannot be applied on this rank now."""


@dataclass
class _Op:
    check: Callable[[Any, dict], str | None]
    apply: Callable[[Any, dict], tuple[dict, dict, Any]]
    """(changes {name: [old, new]}, result, undo kept on this rank)"""
    restore: Callable[[Any, Any], None] | None = None
    """None: the operation cannot be undone (a scrub, a file written)."""
    dev_only: bool = False


def _need_eplb(worker: Any) -> str | None:
    eplb = _eplb(worker)
    if eplb is None or not eplb.model_states:
        return "EPLB is not on in this engine (--enable-eplb)"
    return None


# ---- eplb


def _check_eplb_set(worker: Any, args: dict) -> str | None:
    from vllm.distributed.eplb.runtime_control import check_settings

    return _need_eplb(worker) or check_settings(_eplb(worker), args)


def _apply_eplb_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb.runtime_control import apply_settings, settings

    eplb = _eplb(worker)
    before = settings(eplb)
    changes = apply_settings(eplb, args)
    return changes, {"settings": settings(eplb)}, before


def _restore_eplb_set(worker: Any, before: dict) -> None:
    from vllm.distributed.eplb.runtime_control import apply_settings

    apply_settings(_eplb(worker), before)


def _apply_pause(paused: bool) -> Callable[[Any, dict], tuple[dict, dict, Any]]:
    def apply(worker: Any, args: dict) -> tuple[dict, dict, Any]:
        eplb = _eplb(worker)
        old = eplb.rearrangements_paused
        eplb.rearrangements_paused = paused
        changes = {"paused": [old, paused]} if old != paused else {}
        return changes, {"paused": paused}, old

    return apply


def _restore_pause(worker: Any, old: bool) -> None:
    _eplb(worker).rearrangements_paused = old


def _check_rearrange(worker: Any, args: dict) -> str | None:
    from vllm.distributed.eplb.runtime_control import check_trigger

    return _need_eplb(worker) or check_trigger(_eplb(worker))


def _apply_rearrange(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb.runtime_control import trigger

    eplb = _eplb(worker)
    old = trigger(eplb)
    return (
        {"step": [old, eplb.expert_rearrangement_step]},
        {"at_next_step": True, "window_fill": eplb.window_fill},
        old,
    )


def _restore_rearrange(worker: Any, old: int) -> None:
    from vllm.distributed.eplb.runtime_control import untrigger

    untrigger(_eplb(worker), old)


def _check_save(worker: Any, args: dict) -> str | None:
    from vllm.distributed.eplb.runtime_control import check_save

    return _need_eplb(worker) or check_save(
        _eplb(worker), args.get("path"), args.get("source", "last")
    )


def _apply_save(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb.runtime_control import save

    result = save(_eplb(worker), args.get("path"), args.get("source", "last"))
    return {}, result, None


def _check_load(worker: Any, args: dict) -> str | None:
    from vllm.distributed.eplb.runtime_control import check_load

    return _need_eplb(worker) or check_load(_eplb(worker), args.get("path"))


def _apply_load(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb.runtime_control import load

    eplb = _eplb(worker)
    old_step = eplb.expert_rearrangement_step
    result = load(eplb, args.get("path"))
    undo = result.pop("undo")
    return (
        {"step": [old_step, eplb.expert_rearrangement_step]},
        {**result, "at_next_step": True},
        undo,
    )


def _restore_load(worker: Any, undo: dict) -> None:
    from vllm.distributed.eplb.runtime_control import unload

    unload(_eplb(worker), undo)


# ---- expert store and tiers

_STORE_SETTINGS = ("check_every", "scrub_every", "stats_interval_s")


def _check_store_set(worker: Any, args: dict) -> str | None:
    sessions = _store_sessions()
    if not sessions:
        return "no expert store in this engine (--expert-store-config)"
    unknown = sorted(set(args) - set(_STORE_SETTINGS))
    if unknown:
        return f"unknown store settings {unknown}; known: {list(_STORE_SETTINGS)}"
    for name in ("check_every", "scrub_every"):
        value = args.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            return f"{name} must be an integer >= 0, got {value!r}"
    interval = args.get("stats_interval_s")
    if interval is not None and (
        not isinstance(interval, int | float)
        or isinstance(interval, bool)
        or interval < 0
    ):
        return f"stats_interval_s must be a number >= 0, got {interval!r}"
    if args.get("check_every"):
        for session in sessions:
            if session.is_open and not session.store.can_check_units():
                return (
                    "this store has no per-unit checksums: unit checks cannot be "
                    "turned on (add them with expert_store_build checksums)"
                )
    return None


def _store_values(session: Any) -> dict[str, Any]:
    cfg = session.config
    check = session.store.check_every if session.is_open else cfg.check_every
    return {
        "check_every": check,
        "scrub_every": cfg.scrub_every,
        "stats_interval_s": cfg.stats_interval_s,
    }


def _set_store(session: Any, values: dict[str, Any]) -> None:
    cfg = session.config
    if "check_every" in values:
        if session.is_open:
            session.store.set_check_every(int(values["check_every"]), cfg.check_threads)
        cfg.check_every = int(values["check_every"])
    if "scrub_every" in values:
        cfg.scrub_every = int(values["scrub_every"])
    if "stats_interval_s" in values:
        session.set_stats_interval(float(values["stats_interval_s"]))


def _apply_store_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    undo = []
    changes: dict[str, list] = {}
    for session in _store_sessions():
        before = _store_values(session)
        undo.append((session, before))
        _set_store(session, args)
        after = _store_values(session)
        for name in args:
            if before[name] != after[name]:
                changes[name] = [before[name], after[name]]
    return changes, {"store": _store_state()}, undo


def _restore_store_set(worker: Any, undo: list) -> None:
    for session, before in undo:
        _set_store(session, before)


def _check_scrub(worker: Any, args: dict) -> str | None:
    refusal = _need_eplb(worker)
    if refusal:
        return refusal
    if not any(ms.movers for ms in _eplb(worker).model_states.values()):
        return "no model here places its experts from the expert store"
    return None


def _apply_scrub(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    summaries = _eplb(worker).scrub_expert_store(full=bool(args.get("full", True)))
    return {}, {"scrub": summaries}, None


def _check_tiers_set(worker: Any, args: dict) -> str | None:
    if not _tiers():
        return "no expert layer keeps its slots in tiers (--cpu-offload-per-expert)"
    eplb = _eplb(worker)
    if eplb is not None and any(
        ms.rebalanced for ms in getattr(eplb, "model_states", {}).values()
    ):
        return (
            "an EPLB rearrangement is in flight (its layers are being prepared "
            "and committed one by one); ask after it completes"
        )
    for name, value in args.items():
        setting = _TIER_SETTINGS.get(name)
        if setting is None:
            if name == "host_tier_bound":
                return (
                    "host_tier_bound cannot change in this engine: "
                    f"{TIER_RESIZE_BLOCKED}"
                )
            return f"unknown tier setting {name!r}"
        refusal = setting.check(value)
        if refusal:
            return f"{name}: {refusal}"
    return None


def _apply_tiers_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    undo = {}
    changes = {}
    for name, value in args.items():
        setting = _TIER_SETTINGS[name]
        old = setting.get()
        setting.apply(value)
        undo[name] = old
        changes[name] = [old, setting.get()]
    return changes, {"tiers": _tiers_state()}, undo


def _restore_tiers_set(worker: Any, undo: dict) -> None:
    for name, value in undo.items():
        _TIER_SETTINGS[name].apply(value)


# ---- the CPU expert tier (--cpu-offload-pinned-gb)

# What a running engine may change, with its type and lower bound; the
# promotion settings are read by every between-steps call, the threads by
# every host op call, the cache size by a resize of the host cache.
_CPU_SETTINGS: dict[str, tuple[type, float]] = {
    "cpu_threads": (int, 1),
    "host_cache_gib": (float, 0.0),
    "host_promote_misses": (int, 0),
    "host_promote_window": (int, 1),
}
_NO_CPU_TIER = (
    "no experts are computed on the CPU in this engine (--cpu-offload-pinned-gb)"
)
# Fixed at start, and why.
_CPU_FIXED = {
    "cpu_isa": "the self-check against the card's arithmetic runs at start "
    "for the chosen code",
    "host_promote_rows": "the promotion rows are pinned rows sized at load",
    "read_timeout_s": "an expert store setting; not part of the CPU tier",
}


def _check_cpu_set(worker: Any, args: dict) -> str | None:
    rt = _cpu_runtime()
    if rt is None:
        return _NO_CPU_TIER
    for name, value in args.items():
        if name in _CPU_FIXED:
            return f"{name} cannot change while serving: {_CPU_FIXED[name]}"
        spec = _CPU_SETTINGS.get(name)
        if spec is None:
            return f"unknown CPU tier setting {name!r}; known: {sorted(_CPU_SETTINGS)}"
        kind, low = spec
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if kind is int and not (number and isinstance(value, int) and value >= low):
            return f"{name} must be an integer >= {low:g}, got {value!r}"
        if kind is float and not (number and value > low):
            return f"{name} must be a number > {low:g}, got {value!r}"
        if name == "cpu_threads" and value > 1024:
            return "cpu_threads must be at most 1024"
    if "host_cache_gib" in args:
        rows = rt.cache.rows_for_gib(float(args["host_cache_gib"]))
        need = max(1, rt.stats.max_experts)
        if rows < need:
            return (
                f"host_cache_gib {args['host_cache_gib']} holds {rows} experts; "
                f"one layer call has already needed {need} at once"
            )
        refusal = rt.cache.resize_refusal()
        if refusal:
            return refusal
    return None


def _cpu_values(rt: Any) -> dict[str, Any]:
    return {name: getattr(rt.config, name) for name in _CPU_SETTINGS}


def _set_cpu(rt: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        if name == "host_cache_gib":
            rt.resize_cache(float(value))
        elif name == "cpu_threads":
            rt.set_threads(int(value))
        else:
            setattr(rt.config, name, int(value))


def _apply_cpu_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    rt = _cpu_runtime()
    before = _cpu_values(rt)
    _set_cpu(rt, args)
    after = _cpu_values(rt)
    changes = {k: [before[k], after[k]] for k in args if before[k] != after[k]}
    return changes, {"cpu": _cpu_state()}, {k: before[k] for k in args}


def _restore_cpu_set(worker: Any, before: dict) -> None:
    _set_cpu(_cpu_runtime(), before)


def _check_cpu_stats(worker: Any, args: dict) -> str | None:
    if _cpu_runtime() is None:
        return _NO_CPU_TIER
    return None


def _apply_cpu_stats(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    """Log the CPU tier's line now and return its counters."""
    rt = _cpu_runtime()
    rt.log_stats()
    return {}, {"cpu": _cpu_state()}, None


# ---- KV cache

_KV_SETTINGS = ("release_chunk_mib", "grow_floor_mib")


def _check_kv_set(worker: Any, args: dict) -> str | None:
    if _kv_cache(worker) is None:
        return "the KV cache is not extensible (--enable-extensible-kv-cache)"
    unknown = sorted(set(args) - set(_KV_SETTINGS))
    if unknown:
        return f"unknown KV settings {unknown}; known: {list(_KV_SETTINGS)}"
    for name, low in (("release_chunk_mib", 1), ("grow_floor_mib", 0)):
        if name in args:
            value = args[name]
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < low
            ):
                return f"{name} must be null or an integer >= {low}, got {value!r}"
    if args.get("release_chunk_mib") is not None:
        from vllm.v1.worker.extensible_kv_cache import registers_kv_cache_memory

        if registers_kv_cache_memory(worker.vllm_config):
            return (
                "the KV connector registers the KV cache memory, which needs one "
                "allocation per segment: no release chunk"
            )
    return None


def _kv_values(worker: Any) -> dict[str, Any]:
    cache_config = worker.cache_config
    return {
        "release_chunk_mib": cache_config.kv_cache_release_chunk_mib,
        "grow_floor_mib": cache_config.kv_cache_grow_torch_floor_mib,
    }


def _set_kv(worker: Any, values: dict[str, Any]) -> None:
    """The release chunk applies to the grows from now on: blocks committed
    before keep the physical objects they were mapped with, and a shrink
    gives back every object that lies wholly above its new size, whatever
    its size. The floor is read at every grow's vote."""
    cache_config = worker.cache_config
    if "release_chunk_mib" in values:
        mib = values["release_chunk_mib"]
        buffer = _kv_cache(worker).buffer._buffer
        if mib is None:
            buffer.max_chunk_bytes = None
        else:
            granule = buffer.granularity
            buffer.max_chunk_bytes = -(-(int(mib) << 20) // granule) * granule
        cache_config.kv_cache_release_chunk_mib = mib
    if "grow_floor_mib" in values:
        cache_config.kv_cache_grow_torch_floor_mib = values["grow_floor_mib"]


def _apply_kv_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    before = _kv_values(worker)
    _set_kv(worker, args)
    after = _kv_values(worker)
    changes = {k: [before[k], after[k]] for k in args if before[k] != after[k]}
    return changes, {"kv": _kv_state(worker)}, before


def _restore_kv_set(worker: Any, before: dict) -> None:
    _set_kv(worker, before)


# ---- routing trace

_TRACE_SETTINGS = ("enabled", "path", "interval", "sample", "rotate_mb", "keep")


def _trace_values(eplb: Any) -> dict[str, Any]:
    config = eplb.parallel_config.eplb_config
    return {
        "enabled": eplb._trace is not None,
        "path": config.trace_path,
        "interval": config.trace_interval,
        "sample": config.trace_sample,
        "rotate_mb": config.trace_rotate_mb,
        "keep": config.trace_keep,
    }


def _check_trace_set(worker: Any, args: dict) -> str | None:
    refusal = _need_eplb(worker)
    if refusal:
        return refusal
    unknown = sorted(set(args) - set(_TRACE_SETTINGS))
    if unknown:
        return f"unknown trace settings {unknown}; known: {list(_TRACE_SETTINGS)}"
    want = {**_trace_values(_eplb(worker)), **args}
    if want["enabled"] and not want["path"]:
        return "the trace needs a path (none given, no --eplb-config trace_path)"
    for name, low in (("interval", 1), ("rotate_mb", 0), ("keep", 0)):
        value = want[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            return f"{name} must be an integer >= {low}, got {value!r}"
    sample = want["sample"]
    if sample is not None and (
        not isinstance(sample, int) or not 0 < sample <= want["interval"]
    ):
        return (
            f"sample must be null or 1..interval ({want['interval']}), got {sample!r}"
        )
    return None


def _set_trace(eplb: Any, values: dict[str, Any]) -> None:
    """Interval and sample are in each trace file's meta record: changing
    them (or the path) ends the trace and begins a new one, whose first
    file is new (the old one is rotated aside); the interval being summed
    is dropped. Rotation and the files kept change in place."""
    import torch

    from vllm.distributed.eplb.eplb_trace import ExpertLoadTracer

    config = eplb.parallel_config.eplb_config
    old = _trace_values(eplb)
    want = {**old, **values}
    trace = eplb._trace
    restart = trace is None or any(
        want[k] != old[k] for k in ("path", "interval", "sample")
    )
    config.trace_path = want["path"]
    config.trace_interval = want["interval"]
    config.trace_sample = want["sample"]
    config.trace_rotate_mb = want["rotate_mb"]
    config.trace_keep = want["keep"]
    if trace is not None and (not want["enabled"] or restart):
        eplb._drop_trace()
        trace.close()
        trace = None
    if trace is not None:
        trace.rotate_bytes = want["rotate_mb"] << 20
        trace.keep = want["keep"]
        return
    if not want["enabled"]:
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    trace = ExpertLoadTracer(
        want["path"],
        want["interval"],
        rank,
        sample=want["sample"],
        rotate_bytes=want["rotate_mb"] << 20,
        keep=want["keep"],
    )
    eplb._trace = trace
    for model_state in eplb.model_states.values():
        model_state.map_change_hook = trace.before_map_change
    eplb._should_record_value = None
    eplb._update_layer_should_record(log_stats=config.log_balancedness)


def _apply_trace_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    eplb = _eplb(worker)
    before = _trace_values(eplb)
    _set_trace(eplb, args)
    after = _trace_values(eplb)
    changes = {k: [before[k], after[k]] for k in after if before[k] != after[k]}
    return changes, {"trace": _trace_state(eplb)}, before


def _restore_trace_set(worker: Any, before: dict) -> None:
    _set_trace(_eplb(worker), before)


# ---- PLE row cache


def _check_ple_set(worker: Any, args: dict) -> str | None:
    if not any(_row_stats(t) is not None for _, t in _row_tables(worker)):
        return "no PLE row table on disk in this model"
    unknown = sorted(set(args) - {"stats_interval_s"})
    if unknown:
        return f"unknown PLE settings {unknown}; known: ['stats_interval_s']"
    value = args.get("stats_interval_s")
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        return f"stats_interval_s must be a number >= 0, got {value!r}"
    return None


def _set_ple_interval(worker: Any, interval_s: float) -> None:
    for _, table in _row_tables(worker):
        stats = _row_stats(table)
        if stats is not None:
            stats.interval_s = float(interval_s)
            stats.enabled = stats.interval_s > 0
            stats._reset(time.monotonic())


def _apply_ple_set(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    tables = [t for _, t in _row_tables(worker) if _row_stats(t) is not None]
    old = _row_stats(tables[0]).interval_s
    new = float(args["stats_interval_s"])
    _set_ple_interval(worker, new)
    changes = {"stats_interval_s": [old, new]} if old != new else {}
    return changes, {"ple": _ple_state(worker)}, old


def _restore_ple_set(worker: Any, old: float) -> None:
    _set_ple_interval(worker, old)


# ---- fault injection (development mode only)


def _check_fault_arm(worker: Any, args: dict) -> str | None:
    from vllm.distributed.eplb import fault_injection

    refusal = _need_eplb(worker)
    if refusal:
        return refusal
    stage = args.get("stage")
    if stage not in fault_injection.STAGES:
        return f"stage must be one of {list(fault_injection.STAGES)}, got {stage!r}"
    count = args.get("count", 1)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return f"count must be a positive integer, got {count!r}"
    rank = args.get("rank")
    if rank is not None and not isinstance(rank, int):
        return f"rank must be an EP rank or null (every rank), got {rank!r}"
    return None


def _apply_fault_arm(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb import fault_injection

    before = fault_injection.armed()
    rank = args.get("rank")
    here = rank is None or rank == _ep_rank(worker)
    if here:
        fault_injection.arm(args["stage"], int(args.get("count", 1)))
    after = fault_injection.armed()
    changes = {"armed": [before, after]} if before != after else {}
    return changes, {"armed_here": here, "armed": after}, before


def _restore_faults(worker: Any, before: dict) -> None:
    from vllm.distributed.eplb import fault_injection

    fault_injection.clear()
    for stage, count in before.items():
        fault_injection.arm(stage, count)


def _apply_fault_clear(worker: Any, args: dict) -> tuple[dict, dict, Any]:
    from vllm.distributed.eplb import fault_injection

    before = fault_injection.armed()
    fault_injection.clear(args.get("stage"))
    after = fault_injection.armed()
    changes = {"armed": [before, after]} if before != after else {}
    return changes, {"armed": after}, before


OPS: dict[str, _Op] = {
    "eplb.set": _Op(_check_eplb_set, _apply_eplb_set, _restore_eplb_set),
    "eplb.pause": _Op(lambda w, a: _need_eplb(w), _apply_pause(True), _restore_pause),
    "eplb.resume": _Op(lambda w, a: _need_eplb(w), _apply_pause(False), _restore_pause),
    "eplb.rearrange": _Op(_check_rearrange, _apply_rearrange, _restore_rearrange),
    "eplb.save": _Op(_check_save, _apply_save),
    "eplb.load": _Op(_check_load, _apply_load, _restore_load),
    "store.set": _Op(_check_store_set, _apply_store_set, _restore_store_set),
    "store.scrub": _Op(_check_scrub, _apply_scrub),
    "tiers.set": _Op(_check_tiers_set, _apply_tiers_set, _restore_tiers_set),
    "cpu.set": _Op(_check_cpu_set, _apply_cpu_set, _restore_cpu_set),
    "cpu.stats": _Op(_check_cpu_stats, _apply_cpu_stats),
    "kv.set": _Op(_check_kv_set, _apply_kv_set, _restore_kv_set),
    "trace.set": _Op(_check_trace_set, _apply_trace_set, _restore_trace_set),
    "ple.set": _Op(_check_ple_set, _apply_ple_set, _restore_ple_set),
    "fault.arm": _Op(_check_fault_arm, _apply_fault_arm, _restore_faults, True),
    "fault.clear": _Op(lambda w, a: None, _apply_fault_clear, _restore_faults, True),
}


def _check(worker: Any, op: str, args: dict) -> str | None:
    spec = OPS.get(op)
    if spec is None:
        return f"unknown operation {op!r}"
    if spec.dev_only and not envs.VLLM_SERVER_DEV_MODE:
        return f"{op} is available in development mode only (VLLM_SERVER_DEV_MODE=1)"
    return spec.check(worker, args)


def handle(
    worker: Any, phase: str, op: str = "", args: dict | None = None
) -> dict[str, Any]:
    """`GPUWorker.runtime_control`: see the module docstring."""
    return _plain(_handle(worker, phase, op, dict(args or {})))


def _plain(reply: dict[str, Any]) -> dict[str, Any]:
    """The reply as plain JSON types (numpy scalars and the like become
    Python numbers or strings): it crosses the RPC and the engine core's
    utility reply to an HTTP client."""

    def default(value: Any) -> Any:
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        if isinstance(value, set | tuple):
            return list(value)
        return str(value)

    return json.loads(json.dumps(reply, default=default))


def _handle(worker: Any, phase: str, op: str, args: dict) -> dict[str, Any]:
    rank = _rank(worker)
    if phase == "state":
        return state(worker, args.get("only"))
    if phase == "check":
        try:
            refusal = _check(worker, op, args)
        except Exception as e:  # noqa: BLE001 - the reply carries it
            refusal = f"{type(e).__name__}: {e}"
        return {"rank": rank, "refusal": refusal}
    undo_store: dict[str, Any] = worker.__dict__.setdefault("_runtime_control_undo", {})
    if phase == "apply":
        # An older change's undo must never be taken for this one's.
        undo_store.pop(op, None)
        refusal = _check(worker, op, args)
        if refusal is not None:
            return {"rank": rank, "ok": False, "error": refusal, "changes": {}}
        spec = OPS[op]
        try:
            changes, result, undo = spec.apply(worker, args)
        except Exception as e:  # noqa: BLE001 - the reply carries it
            logger.warning(
                "Runtime control (rank %d): %s could not be applied: %s", rank, op, e
            )
            return {
                "rank": rank,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "changes": {},
            }
        undo_store[op] = undo
        for name, (old, new) in changes.items():
            logger.info(
                "Runtime control (rank %d): %s %s %s -> %s.", rank, op, name, old, new
            )
        if not changes:
            logger.info("Runtime control (rank %d): %s done.", rank, op)
        return {
            "rank": rank,
            "ok": True,
            "error": None,
            "changes": changes,
            "result": result,
            "undoable": spec.restore is not None,
        }
    if phase == "restore":
        spec = OPS.get(op)
        ranks = args.get("ranks")
        if ranks is not None and rank not in ranks:
            return {"rank": rank, "ok": True, "error": None, "skipped": True}
        if spec is None or spec.restore is None or op not in undo_store:
            return {"rank": rank, "ok": False, "error": "nothing to restore"}
        try:
            spec.restore(worker, undo_store.pop(op))
        except Exception as e:  # noqa: BLE001 - the reply carries it
            logger.warning(
                "Runtime control (rank %d): %s could not be undone: %s", rank, op, e
            )
            return {"rank": rank, "ok": False, "error": f"{type(e).__name__}: {e}"}
        logger.info("Runtime control (rank %d): %s undone.", rank, op)
        return {"rank": rank, "ok": True, "error": None}
    return {"rank": rank, "ok": False, "error": f"unknown phase {phase!r}"}
