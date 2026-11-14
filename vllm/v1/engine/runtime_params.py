# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The registrations below are a table: one parameter per line reads best.
# ruff: noqa: E501
"""The runtime control surface's self-description, for a UI or an agent.

Every operation (``register_verb``) and every parameter (``register_param``)
is registered with its name, type, unit, range or choices, default, whether
it changes live (and through which verb) or needs a restart (and why), and a
one-line description. ``GET /runtime_control/schema`` (the operation
``schema``) returns them with each parameter's current value; serving
profiles name parameters the same way (``<section>.<key>``). A component
registers its own verbs and parameters where it is defined: nothing here is
a central list another component has to edit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class Param:
    name: str
    """``<section>.<key>``; the profile section and key when ``verb`` is set."""
    type: str
    """int | float | bool | str | enum | json"""
    description: str
    verb: str | None = None
    """The operation that changes it live; None: start only."""
    key: str | None = None
    """Its key in the verb's body (default: the name's last part)."""
    unit: str = ""
    min: float | None = None
    max: float | None = None
    choices: list | None = None
    nullable: bool = False
    default: Any = None
    restart_reason: str | None = None
    """Why it needs a restart (start-only), or what part of its range does."""
    start: str | None = None
    """Where it is set at start (flag, config key or environment variable)."""
    side: str = "engine"
    """engine | server (the API server process answers it)."""
    read: Callable[[Any, dict], Any] | None = None
    """(engine, state) -> its value now."""

    def describe(self, engine: Any, state: dict) -> dict[str, Any]:
        # Every field that is set, 0 and False included (only None and an
        # empty text mean "not given").
        out = {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k != "read" and v is not None and not (isinstance(v, str) and v == "")
        }
        out["nullable"] = self.nullable
        out["live"] = self.verb is not None and self.verb in VERBS
        if self.verb is not None and self.verb not in VERBS:
            out["unavailable"] = f"no operation {self.verb} in this engine"
        out["key"] = self.key or self.name.rsplit(".", 1)[-1]
        if self.read is not None:
            try:
                out["value"] = self.read(engine, state)
            except Exception:  # noqa: BLE001 - a readout
                out["value"] = None
        return out


@dataclasses.dataclass
class Verb:
    name: str
    description: str
    params: list[str] = dataclasses.field(default_factory=list)
    """Parameter names its body takes (or free-form keys listed in `args`)."""
    args: dict[str, str] = dataclasses.field(default_factory=dict)
    """Arguments that are not parameters (e.g. ``num_blocks``): name -> doc."""
    side: str = "engine"
    kind: str = "action"
    """set (changes parameters) | action | read"""


PARAMS: dict[str, Param] = {}
VERBS: dict[str, Verb] = {}


def register_param(param: Param) -> None:
    PARAMS[param.name] = param
    if param.verb is not None:
        verb = VERBS.get(param.verb)
        if verb is not None and param.name not in verb.params:
            verb.params.append(param.name)


def register_verb(verb: Verb) -> None:
    existing = VERBS.get(verb.name)
    if existing is not None:
        verb.params = list(dict.fromkeys([*existing.params, *verb.params]))
    VERBS[verb.name] = verb
    for param in PARAMS.values():
        if param.verb == verb.name and param.name not in verb.params:
            verb.params.append(param.name)


def schema(engine: Any, control: Any) -> dict[str, Any]:
    """Every verb and parameter, with values read from one state readout."""
    try:
        state = control.state()
    except Exception as e:  # noqa: BLE001 - values are best effort
        logger.warning("Runtime control schema: the state could not be read: %s", e)
        state = {}
    from vllm.v1.engine.serving_profiles import SECTIONS

    return {
        "verbs": [dataclasses.asdict(v) for v in VERBS.values()],
        "params": [p.describe(engine, state) for p in PARAMS.values()],
        "profile_sections": list(SECTIONS),
    }


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _rank0(part: str, *keys: str) -> Callable[[Any, dict], Any]:
    def read(engine: Any, state: dict) -> Any:
        ranks = state.get("ranks") or [{}]
        value: Any = ranks[0].get(part)
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    return read


def _attr(*path: str) -> Callable[[Any, dict], Any]:
    def read(engine: Any, state: dict) -> Any:
        value: Any = engine
        for name in path:
            value = getattr(value, name, None)
        return value

    return read


def _tower(key: str) -> Callable[[Any, dict], Any]:
    def read(engine: Any, state: dict) -> Any:
        towers = state.get("towers") or {}
        if not towers:
            return None
        tower = next(iter(towers.values()))
        return tower.get(key, (tower.get("options") or {}).get(key))

    return read


# --------------------------------------------------------------------------
# The surface's own verbs and parameters (other components register theirs)
# --------------------------------------------------------------------------

_GRAPH = "captured graphs and buffers are sized for it at start"
_LOAD = "allocated or built at load"


def _register_builtin() -> None:
    V, P = register_verb, register_param
    V(Verb("state", "Read every part's state (GET /runtime_control).", kind="read"))
    V(Verb("schema", "This description (GET /runtime_control/schema).", kind="read"))
    # -- KV cache
    V(
        Verb(
            "kv.resize",
            "Grow or shrink the KV cache to num_blocks.",
            args={
                "num_blocks": "int, blocks (or tokens)",
                "tokens": "int, converted to blocks at the pool's tokens per block now; the reply states both",
                "deadline_s": "float, s, drain limit of a shrink",
                "preempt": "bool, preempt holders at the deadline",
            },
        )
    )
    V(Verb("kv.fit", "Grow the KV cache to what the memory and the torch floor allow."))
    V(Verb("kv.abort", "Abort a draining KV shrink."))
    V(Verb("kv.set", "KV cache settings.", kind="set"))
    P(
        Param(
            "kv.release_chunk_mib",
            "int",
            "Physical object size of KV grows from now on (a shrink releases whole objects).",
            "kv.set",
            unit="MiB",
            min=1,
            nullable=True,
            start="--kv-cache-release-chunk-mib",
            read=_attr("vllm_config", "cache_config", "kv_cache_release_chunk_mib"),
        )
    )
    P(
        Param(
            "kv.grow_floor_mib",
            "int",
            "Torch headroom every grow keeps.",
            "kv.set",
            unit="MiB",
            min=0,
            nullable=True,
            start="--kv-cache-grow-torch-floor-mib",
            read=_attr("vllm_config", "cache_config", "kv_cache_grow_torch_floor_mib"),
        )
    )
    P(
        Param(
            "kv.blocks",
            "int",
            "KV cache size (kv.resize num_blocks; a profile's kv.blocks).",
            "kv.resize",
            key="num_blocks",
            unit="blocks",
            min=1,
            restart_reason="above the reserved capacity: the address range is reserved at start",
            read=_attr("vllm_config", "cache_config", "num_gpu_blocks"),
        )
    )
    P(
        Param(
            "kv.tokens",
            "int",
            "KV cache size in tokens (a profile's kv.tokens; converted with the pool's tokens per block now).",
            "kv.resize",
            unit="tokens",
            min=1,
            read=_attr("vllm_config", "cache_config", "kv_cache_size_tokens"),
        )
    )
    P(
        Param(
            "kv.fit",
            "bool",
            "KV cache as large as memory and the torch floor allow (kv.fit; a profile's kv.fit).",
            "kv.fit",
        )
    )
    # The fast tier's own verb (tiers.fast) is registered by the fast tier.
    P(
        Param(
            "tiers.fast_rows",
            "int",
            "Card rows per MoE layer per rank (a profile's tiers.fast_rows, also max/min).",
            "tiers.fast",
            key="rows",
            restart_reason="outside the range reserved at start (fast_rows_max)",
            read=lambda e, s: (s.get("fast_tier") or {}).get("rows"),
        )
    )
    P(
        Param(
            "kv.offloading_size",
            "float",
            "Host KV tier (native offloading connector).",
            unit="GiB",
            restart_reason="the host buffer and the connector are " + _LOAD,
            start="--kv-offloading-size",
            read=_attr("vllm_config", "cache_config", "kv_offloading_size"),
        )
    )
    P(
        Param(
            "kv.cache_dtype",
            "str",
            "KV cache element type.",
            restart_reason="the attention kernels and the pool layout are chosen at start",
            start="--kv-cache-dtype",
            read=_attr("vllm_config", "cache_config", "cache_dtype"),
        )
    )
    P(
        Param(
            "kv.prefix_caching",
            "bool",
            "Prefix caching.",
            restart_reason="the block hashes are computed per request from the start",
            start="--enable-prefix-caching",
            read=_attr("vllm_config", "cache_config", "enable_prefix_caching"),
        )
    )
    # -- scheduler
    V(Verb("sched.set", "Scheduler limits below their start values.", kind="set"))
    P(
        Param(
            "sched.max_num_active_seqs",
            "int",
            "Requests admitted to run at once (running ones above a lower cap finish).",
            "sched.set",
            min=1,
            restart_reason="above max_num_seqs: " + _GRAPH,
            start="--max-num-active-seqs",
            read=_attr("scheduler", "max_num_active_reqs"),
        )
    )
    P(
        Param(
            "sched.max_num_scheduled_tokens",
            "int",
            "Token budget of one step.",
            "sched.set",
            unit="tokens",
            min=1,
            restart_reason="above max_num_batched_tokens: the input buffers are sized at start",
            start="--max-num-scheduled-tokens",
            read=_attr("scheduler", "max_num_scheduled_tokens"),
        )
    )
    P(
        Param(
            "sched.long_prefill_token_threshold",
            "int",
            "Prefill tokens one request may take in a step (0 = off): a long prompt is chunked so that running decodes keep their pace.",
            "sched.set",
            unit="tokens",
            min=0,
            start="--long-prefill-token-threshold",
            read=_attr("scheduler", "scheduler_config", "long_prefill_token_threshold"),
        )
    )
    P(
        Param(
            "sched.max_num_seqs",
            "int",
            "Runner request slots.",
            restart_reason=_GRAPH,
            start="--max-num-seqs",
            read=_attr("scheduler", "max_num_running_reqs"),
        )
    )
    P(
        Param(
            "sched.max_num_batched_tokens",
            "int",
            "Runner token slots.",
            unit="tokens",
            restart_reason="the input buffers are sized at start",
            start="--max-num-batched-tokens",
            read=_attr("scheduler", "scheduler_config", "max_num_batched_tokens"),
        )
    )
    P(
        Param(
            "sched.max_model_len",
            "int",
            "Longest request.",
            unit="tokens",
            restart_reason="the KV floor, the rope tables and the graphs are sized at start",
            start="--max-model-len",
            read=_attr("scheduler", "max_model_len"),
        )
    )
    P(
        Param(
            "sched.async_scheduling",
            "bool",
            "Scheduling overlapped with execution.",
            restart_reason="the engine core's step loop is chosen at start",
            start="--async-scheduling",
        )
    )
    # -- drafter
    V(
        Verb(
            "spec.set",
            "Speculative decoding width, on/off, acceptance counters.",
            kind="set",
            args={"reset_stats": "bool, zero the acceptance counters"},
        )
    )
    P(
        Param(
            "spec.num_speculative_tokens",
            "int",
            "Draft tokens per step (0 = off).",
            "spec.set",
            unit="tokens",
            min=0,
            restart_reason="above the start number: drafter buffers, rejection sampler and graphs are sized for it",
            start="--speculative-config num_speculative_tokens",
            read=_attr("scheduler", "num_spec_tokens_now"),
        )
    )
    P(
        Param(
            "spec.enabled",
            "bool",
            "Drafter on (its start width) or off.",
            "spec.set",
            read=lambda e, s: (
                getattr(getattr(e, "scheduler", None), "num_spec_tokens_now", 0) > 0
            ),
        )
    )
    P(
        Param(
            "spec.method",
            "str",
            "Drafter model and method.",
            restart_reason="the drafter is loaded at start",
            start="--speculative-config method/model",
        )
    )
    # -- EPLB
    V(Verb("eplb.set", "EPLB settings.", kind="set"))
    V(Verb("eplb.pause", "Intervals end without a rearrangement."))
    V(Verb("eplb.resume", "Rearrangements again."))
    V(Verb("eplb.rearrange", "A rearrangement at the next step."))
    V(
        Verb(
            "eplb.save",
            "Write the load state file.",
            args={"path": "str", "source": "last | window"},
        )
    )
    V(
        Verb(
            "eplb.load",
            "Rearrange from a state file at the next step.",
            args={"path": "str"},
        )
    )
    P(
        Param(
            "eplb.step_interval",
            "int",
            "Steps between periodic rearrangements (0 = none: EPLB observes; "
            "eplb.rearrange still moves).",
            "eplb.set",
            unit="steps",
            min=0,
            default=0,
            start="--eplb-config step_interval",
            read=_rank0("eplb", "settings", "step_interval"),
        )
    )
    P(
        Param(
            "eplb.min_imbalance",
            "float",
            "A periodic rearrangement starts only when the busiest rank carries "
            "at least 1 + this times the mean rank load of the window (0 = always).",
            "eplb.set",
            min=0,
            default=0.0,
            start="--eplb-config min_imbalance",
            read=_rank0("eplb", "settings", "min_imbalance"),
        )
    )
    P(
        Param(
            "eplb.move_min_gain",
            "float",
            "Incremental policy: the least a swap must lower the busiest rank's "
            "load, as a fraction of the mean rank load (null = 0.001).",
            "eplb.set",
            min=0,
            nullable=True,
            start="--eplb-config move_min_gain",
            read=_rank0("eplb", "settings", "move_min_gain"),
        )
    )
    P(
        Param(
            "eplb.interval_from_end",
            "bool",
            "Count step_interval from the end of the previous rearrangement.",
            "eplb.set",
            default=False,
            start="--eplb-config interval_from_end",
            read=_rank0("eplb", "settings", "interval_from_end"),
        )
    )
    P(
        Param(
            "eplb.window_size",
            "int",
            "Steps of load a rearrangement plans from.",
            "eplb.set",
            unit="steps",
            min=1,
            start="--eplb-config window_size",
            read=_rank0("eplb", "settings", "window_size"),
        )
    )
    P(
        Param(
            "eplb.max_moves_per_rank",
            "int",
            "Expert moves per rank per rearrangement.",
            "eplb.set",
            min=1,
            restart_reason="above the rows set aside at load",
            start="--eplb-config max_moves_per_rank",
            read=_rank0("eplb", "settings", "max_moves_per_rank"),
        )
    )
    P(
        Param(
            "eplb.policy",
            "enum",
            "Rearrangement policy.",
            "eplb.set",
            choices=["default", "incremental"],
            start="--eplb-config policy",
            read=_rank0("eplb", "settings", "policy"),
        )
    )
    P(
        Param(
            "eplb.num_redundant_experts",
            "int",
            "Redundant expert slots.",
            restart_reason="the expert weights and maps are " + _LOAD,
            start="--eplb-config num_redundant_experts",
        )
    )
    P(
        Param(
            "eplb.use_async",
            "bool",
            "Weights moved by a worker thread.",
            restart_reason="the async worker and its buffers are " + _LOAD,
            start="--eplb-config use_async",
        )
    )
    P(
        Param(
            "eplb.tier_interval",
            "int",
            "Steps between EPLB's tier passes (the card, pinned and CPU tiers re-cut by load inside each rank; 0 = off).",
            "eplb.set",
            unit="steps",
            min=0,
            start="--eplb-config tier_interval",
            read=_rank0("eplb", "settings", "tier_interval"),
        )
    )
    P(
        Param(
            "eplb.tier_max_moves",
            "int",
            "Experts a tier pass reads per layer (null = the layer's spare rows).",
            "eplb.set",
            min=1,
            nullable=True,
            restart_reason="above the spare rows set aside at load",
            start="--eplb-config tier_max_moves",
            read=_rank0("eplb", "settings", "tier_max_moves"),
        )
    )
    P(
        Param(
            "eplb.tier_min_gain",
            "float",
            "A tier swap only when the call cost it saves over the window repays its move this many times (null = hysteresis alone).",
            "eplb.set",
            min=0,
            nullable=True,
            start="--eplb-config tier_min_gain",
            read=_rank0("eplb", "settings", "tier_min_gain"),
        )
    )
    P(
        Param(
            "eplb.tier_min_changes",
            "int",
            "A tier pass only when its cut moves at least this many slots over the layers and ranks (0 = every interval).",
            "eplb.set",
            min=0,
            start="--eplb-config tier_min_changes",
            read=_rank0("eplb", "settings", "tier_min_changes"),
        )
    )
    P(
        Param(
            "eplb.tier_window_moves",
            "int",
            "Experts the tier passes may read per layer and rank over the last window_size steps (null = no cap).",
            "eplb.set",
            min=1,
            nullable=True,
            start="--eplb-config tier_window_moves",
            read=_rank0("eplb", "settings", "tier_window_moves"),
        )
    )
    P(
        Param(
            "eplb.tier_hot",
            "float",
            "Temperature (load over the mean expert's) from which an expert counts as hot.",
            "eplb.set",
            min=0,
            start="--eplb-config tier_hot",
            read=_rank0("eplb", "settings", "tier_hot"),
        )
    )
    P(
        Param(
            "eplb.tier_warm",
            "float",
            "Temperature from which an expert counts as warm (at most tier_hot).",
            "eplb.set",
            min=0,
            start="--eplb-config tier_warm",
            read=_rank0("eplb", "settings", "tier_warm"),
        )
    )
    P(
        Param(
            "eplb.tier_hysteresis",
            "float",
            "Load ratio before a tier choice or a class changes.",
            min=1,
            restart_reason="read into the EPLB state at start",
            start="--eplb-config tier_hysteresis",
        )
    )
    P(
        Param(
            "eplb.seed_path",
            "str",
            "Expert loads that place the tiers at boot when the state file has none (EPLB state, 0.32 rankings, learned state or a trace's loads).",
            restart_reason="read once, at load",
            start="--eplb-config seed_path",
        )
    )
    # -- tiers, store, CPU tier
    V(
        Verb(
            "tiers.set",
            "Expert tier settings a tier component registers; while EPLB moves "
            "experts the change waits for it (answer `waiting`).",
            args={
                "deadline_s": "float, how long the change may wait for EPLB "
                "(default 60)",
            },
            kind="set",
        )
    )
    P(
        Param(
            "tiers.host_tier_bound",
            "int",
            "Pinned slots per layer per rank (the rest computed on the CPU).",
            "tiers.set",
            min=1,
            restart_reason="above the pinned rows sized at load (--cpu-offload-pinned-gb, host_promote_rows)",
            start="--cpu-offload-pinned-gb",
            read=_rank0("tiers", "settings", "host_tier_bound"),
        )
    )
    V(Verb("cpu.set", "The CPU expert tier.", kind="set"))
    V(Verb("cpu.stats", "Log the CPU tier's counters now."))
    P(
        Param(
            "cpu.cpu_threads",
            "int",
            "Threads of the CPU expert op.",
            "cpu.set",
            min=1,
            max=1024,
            start="--expert-store-config cpu_threads",
            read=_rank0("cpu", "settings", "cpu_threads"),
        )
    )
    P(
        Param(
            "cpu.host_cache_gib",
            "float",
            "Host cache of experts for the CPU tier.",
            "cpu.set",
            unit="GiB",
            min=0,
            start="--expert-store-config host_cache_gib",
            read=_rank0("cpu", "settings", "host_cache_gib"),
        )
    )
    P(
        Param(
            "cpu.stage_min_calls",
            "int",
            "Prefill staging: CPU-tier experts a layer call routes at least this many tokens to are computed on the card through staging rows (0 = off).",
            "cpu.set",
            min=0,
            start="--expert-store-config stage_min_calls",
            read=_rank0("cpu", "settings", "stage_min_calls"),
        )
    )
    P(
        Param(
            "cpu.stage_min_tokens",
            "int",
            "Prefill staging by cost: a layer call of at least this many tokens stages the CPU-tier experts that make its host-side time smallest (0 = off).",
            "cpu.set",
            min=0,
            start="--expert-store-config stage_min_tokens",
            read=_rank0("cpu", "settings", "stage_min_tokens"),
        )
    )
    P(
        Param(
            "cpu.stage_copy_threads",
            "int",
            "Threads copying units from the host cache into the staging rows.",
            "cpu.set",
            min=1,
            max=256,
            start="--expert-store-config stage_copy_threads",
            read=_rank0("cpu", "settings", "stage_copy_threads"),
        )
    )
    P(
        Param(
            "cpu.stage_card_us",
            "float",
            "The card's time per staged expert the staging cost model assumes.",
            "cpu.set",
            unit="us",
            min=0,
            start="--expert-store-config stage_card_us",
            read=_rank0("cpu", "settings", "stage_card_us"),
        )
    )
    P(
        Param(
            "cpu.stage_device_rows",
            "int",
            "Device staging rows in use: the card reads staged CPU-tier experts "
            "from device memory filled by DMA; 0 = it reads the pinned rows "
            "through UVA. At most the rows reserved at load "
            "(--expert-store-config stage_device_rows); more needs a restart.",
            "cpu.set",
            min=0,
            start="--expert-store-config stage_device_rows",
            read=_rank0("cpu", "settings", "stage_device_rows"),
        )
    )
    P(
        Param(
            "cpu.stage_dma_us",
            "float",
            "The time of one unit's host-to-card copy the staging cost model "
            "assumes with device staging rows.",
            "cpu.set",
            unit="us",
            min=0,
            start="--expert-store-config stage_dma_us",
            read=_rank0("cpu", "settings", "stage_dma_us"),
        )
    )
    P(
        Param(
            "cpu.stage_rows",
            "int",
            "Staging rows (0 = a layer's CPU slots).",
            min=0,
            restart_reason="the staging rows are pinned memory allocated at first use",
            start="--expert-store-config stage_rows",
        )
    )
    P(
        Param(
            "cpu.cpu_isa",
            "str",
            "CPU code path.",
            restart_reason="the self-check against the card's arithmetic runs at start",
            start="--expert-store-config cpu_isa",
        )
    )
    P(
        Param(
            "cpu.cpu_arith",
            "str",
            "CPU experts' arithmetic: the card kernel's per call, or the GEMV's.",
            restart_reason="the self-check against the card's arithmetic runs at start",
            start="--expert-store-config cpu_arith",
        )
    )
    V(Verb("store.set", "Expert store settings.", kind="set"))
    V(
        Verb(
            "store.scrub",
            "Read back every mapped slot and compare.",
            args={"full": "bool"},
        )
    )
    P(
        Param(
            "store.check_every",
            "int",
            "Check every N-th unit read (0 = off).",
            "store.set",
            min=0,
            start="--expert-store-config check_every",
            read=lambda e, s: (
                ((s.get("ranks") or [{}])[0].get("store") or [{}])[0]
            ).get("check_every"),
        )
    )
    P(
        Param(
            "store.scrub_every",
            "int",
            "Scrub after every N-th rearrangement (0 = off).",
            "store.set",
            min=0,
            start="--expert-store-config scrub_every",
        )
    )
    P(
        Param(
            "store.stats_interval_s",
            "float",
            "Seconds between the store's (and the CPU tier's) stats lines (0 = off).",
            "store.set",
            unit="s",
            min=0,
            start="--expert-store-config stats_interval_s",
        )
    )
    P(
        Param(
            "store.pool_gib",
            "float",
            "Pinned pool of the store.",
            unit="GiB",
            restart_reason="pinned host memory " + _LOAD,
            start="--expert-store-config pool_gib",
        )
    )
    # -- tower
    V(
        Verb(
            "tower.set",
            "A dynamic encoder tower's policy and thresholds.",
            kind="set",
            args={"name": "str, with more than one dynamic tower"},
        )
    )
    V(
        Verb(
            "tower.promote",
            "Bring a dynamic tower to the accelerators now.",
            args={"name": "str"},
        )
    )
    V(Verb("tower.demote", "Send it back to host memory now.", args={"name": "str"}))
    P(
        Param(
            "tower.policy",
            "enum",
            "Placement: dynamic (by requests and idle time), host (never on the card), card (always).",
            "tower.set",
            choices=["dynamic", "host", "card"],
            default="dynamic",
            read=_tower("policy"),
        )
    )
    P(
        Param(
            "tower.promote_after",
            "int",
            "Requests within promote_window_s that bring it to the card.",
            "tower.set",
            min=1,
            default=2,
            start="--mm-encoder-host-towers promote_after",
            read=_tower("promote_after"),
        )
    )
    P(
        Param(
            "tower.promote_window_s",
            "float",
            "The window of promote_after.",
            "tower.set",
            unit="s",
            min=0,
            default=600.0,
            read=_tower("promote_window_s"),
        )
    )
    P(
        Param(
            "tower.demote_idle_s",
            "float",
            "Idle time on the card before it goes back to host memory.",
            "tower.set",
            unit="s",
            min=0,
            default=1800.0,
            read=_tower("demote_idle_s"),
        )
    )
    P(
        Param(
            "tower.disk_idle_s",
            "float",
            "Idle time in host memory before its host copy is released (the drive tier); null = idle_location decides.",
            "tower.set",
            unit="s",
            min=0,
            nullable=True,
            read=_tower("disk_idle_s"),
        )
    )
    P(
        Param(
            "tower.idle_location",
            "enum",
            "Where it rests idle when disk_idle_s is null.",
            "tower.set",
            choices=["host", "disk"],
            default="host",
            read=_tower("idle_location"),
        )
    )
    P(
        Param(
            "tower.card_reserve_mib",
            "int",
            "Activation room asked of the KV cache on top of its weights.",
            "tower.set",
            unit="MiB",
            min=0,
            default=128,
            read=_tower("card_reserve_mib"),
        )
    )
    P(
        Param(
            "tower.host_available_floor_gib",
            "float",
            "Host MemAvailable a promotion must leave.",
            "tower.set",
            unit="GiB",
            min=0,
            default=0.0,
            read=_tower("host_available_floor_gib"),
        )
    )
    P(
        Param(
            "tower.shrink_deadline_s",
            "float",
            "Drain limit of the KV shrink a promotion asks for (0 = none).",
            "tower.set",
            unit="s",
            min=0,
            default=60.0,
            read=_tower("shrink_deadline_s"),
        )
    )
    P(
        Param(
            "tower.card_layout",
            "enum",
            "Replicated or sharded on the cards.",
            choices=["replicated", "sharded"],
            restart_reason="built into the copies",
            start="--mm-encoder-host-towers card_layout",
        )
    )
    P(
        Param(
            "tower.residency",
            "enum",
            "Whether a tower is dynamic at all.",
            choices=["host", "dynamic"],
            restart_reason="the residency controller and the host tower connector are set up at start",
            start="--mm-encoder-host-towers residency",
        )
    )
    # -- tracing and debug
    V(Verb("trace.set", "The routing (expert load) trace.", kind="set"))
    P(
        Param(
            "trace.enabled",
            "bool",
            "Routing trace on/off.",
            "trace.set",
            start="--eplb-config trace_path",
            read=_rank0("trace", "enabled"),
        )
    )
    P(
        Param(
            "trace.interval",
            "int",
            "Steps summed per trace record.",
            "trace.set",
            unit="steps",
            min=1,
            read=_rank0("trace", "interval"),
        )
    )
    P(
        Param(
            "trace.rotate_mb",
            "int",
            "Trace file rotation size.",
            "trace.set",
            unit="MB",
            min=0,
            read=_rank0("trace", "rotate_mb"),
        )
    )
    P(
        Param(
            "trace.keep",
            "int",
            "Rotated trace files kept.",
            "trace.set",
            min=0,
            read=_rank0("trace", "keep"),
        )
    )
    V(Verb("ple.set", "PLE row cache stats.", kind="set"))
    P(
        Param(
            "ple.stats_interval_s",
            "float",
            "Seconds between the PLE row cache's stats lines (0 = off).",
            "ple.set",
            unit="s",
            min=0,
            start="--engram-config disk_stats_interval_s",
        )
    )
    P(
        Param(
            "ple.disk_row_cache_gib",
            "float",
            "PLE row cache on the host.",
            unit="GiB",
            restart_reason="the row cache is " + _LOAD,
            start="--engram-config disk_row_cache_gib",
        )
    )
    V(
        Verb(
            "debug.set",
            "Bounded debug lines of one part; off by itself after duration_s or count.",
            args={
                "part": "eplb | scheduler | kv | cpu | tower | placement | logger:<name>",
                "on": "bool",
                "duration_s": "float, s, at most 3600",
                "count": "int, lines, at most 100000",
            },
        )
    )
    V(Verb("debug.state", "Which debug parts are on, and their bounds.", kind="read"))
    V(
        Verb(
            "fault.arm",
            "Development mode: fail the next EPLB allocations of a stage.",
            args={
                "stage": "load | plan | copy_rows",
                "rank": "int or null",
                "count": "int",
            },
        )
    )
    V(
        Verb(
            "fault.clear",
            "Development mode: clear armed faults.",
            args={"stage": "str"},
        )
    )
    # -- profiles
    V(
        Verb(
            "profile.define",
            "Define or replace a serving profile (JSON; sections name these parameters).",
            args={
                "name": "str",
                "description": "str",
                "<section>": "object of <section>.<key> values",
            },
        )
    )
    V(
        Verb(
            "profile.list",
            "Defined profiles (name, description, sections) and the active one; profile.get gives a body.",
            kind="read",
        )
    )
    V(
        Verb(
            "profile.get",
            "One profile's full body as defined, and whether it is active or being applied.",
            kind="read",
            args={"name": "str"},
        )
    )
    V(
        Verb(
            "profile.apply",
            "Apply a profile: checked first, memory freed before taken.",
            args={"name": "str"},
        )
    )
    V(
        Verb(
            "profile.state",
            "Active profile, apply progress, each target against its value.",
            kind="read",
        )
    )
    # -- request defaults (the API server process)
    V(
        Verb(
            "defaults.set",
            "Request defaults of this API server process; null = start value.",
            kind="set",
            side="server",
        )
    )
    V(
        Verb(
            "defaults.state",
            "Request defaults now and at start.",
            kind="read",
            side="server",
        )
    )
    for name, typ, desc, lo, hi in (
        (
            "temperature",
            "float",
            "Sampling temperature when a request has none.",
            0,
            None,
        ),
        ("top_p", "float", "Nucleus mass when a request has none.", 0, 1),
        ("top_k", "int", "Top-k when a request has none (0 or -1 off).", -1, None),
        ("min_p", "float", "Min-p when a request has none.", 0, 1),
        (
            "repetition_penalty",
            "float",
            "Repetition penalty when a request has none.",
            0,
            None,
        ),
        (
            "presence_penalty",
            "float",
            "Presence penalty when a request has none.",
            -2,
            2,
        ),
        (
            "frequency_penalty",
            "float",
            "Frequency penalty when a request has none.",
            -2,
            2,
        ),
        (
            "thinking_token_budget",
            "int",
            "Thinking tokens before the end-of-thinking token, when a request has none.",
            0,
            None,
        ),
    ):
        P(
            Param(
                f"defaults.{name}",
                typ,
                desc,
                "defaults.set",
                min=lo,
                max=hi,
                nullable=True,
                side="server",
                start="--generation-config / --override-generation-config",
            )
        )
    P(
        Param(
            "defaults.cjk_mask_default",
            "enum",
            "Han mask for chat requests without an explicit flag.",
            "defaults.set",
            choices=["auto", "on", "off"],
            default="auto",
            nullable=True,
            side="server",
            restart_reason="needs the Han mask's env switch at start",
        )
    )
    P(
        Param(
            "defaults.reasoning_parser",
            "str",
            "Reasoning parser.",
            restart_reason="the parser is built with the serving handlers at start",
            side="server",
            start="--reasoning-parser",
        )
    )
    # -- logging and stats
    P(
        Param(
            "stats.log_stats",
            "bool",
            "Stats and metrics collection.",
            restart_reason="the stat loggers are built at start",
            start="--disable-log-stats",
        )
    )


_register_builtin()


def _register_components() -> None:
    """Components whose module registers its own verbs and parameters."""
    from vllm.v1.engine import fast_tier

    fast_tier.register_runtime_params(register_verb, register_param, Verb, Param)


_register_components()
