# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB's part of the runtime control surface: what a running engine's EPLB
state is, and the changes an outside controller may make to it.

Every function here runs on a worker's main thread inside an RPC from the
engine core, which the engine core issues between two steps (as a utility,
see `EngineCore.runtime_control`): the batches launched before it have run
`execute_model` and `sample_tokens` (where `EplbState.step` runs), and the
next one is launched after it. So a change lands between two calls of
`EplbState.step`, on every EP rank before the same step. Nothing here is in
a captured graph: the router reads `logical_to_physical_map`,
`logical_replica_count` and the shared record flag, and only `step` writes
them.

The functions that check a change (`check_*`) change nothing, so the engine
core can ask every rank before any rank applies; the functions that apply
return what they replaced so that a change one rank could not make can be
undone on the others.
"""

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from .eplb_state_file import read_expert_loads, write_expert_loads
from .policy import EPLB_POLICIES, IncrementalEplbPolicy

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)

SETTINGS = ("step_interval", "window_size", "max_moves_per_rank", "policy")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def in_flight(state: "EplbState") -> str | None:
    """Why a rearrangement is in flight (an async one between its plan and
    its last layer's commit), or None. The same on every EP rank at a step
    boundary: `rebalanced` is set and cleared at the same step everywhere."""
    for model_state in state.model_states.values():
        if model_state.rebalanced:
            pending = model_state.pending_result
            at = "" if pending is None else f" at layer {pending.layer_idx}"
            return (
                f"an async rearrangement of {model_state.model_name} is in "
                f"flight{at} of {model_state.model.num_moe_layers}"
            )
    return None


def settings(state: "EplbState") -> dict[str, Any]:
    config = state.parallel_config.eplb_config
    policy = state.policy
    return {
        "step_interval": state.expert_rearrangement_step_interval,
        "window_size": state.expert_load_window_size,
        "max_moves_per_rank": (
            policy.max_moves_per_rank
            if issubclass(policy, IncrementalEplbPolicy)
            else config.max_moves_per_rank
        ),
        "policy": config.policy,
    }


def _balancedness(model_state: "EplbModelState", num_ranks: int) -> float | None:
    """Mean over layers with load of (mean rank load) / (max rank load), for
    `last_expert_load` (the loads the last rearrangement used) under the
    placement in force; None before any load is known."""
    with gpu_sync_allowed():
        load = model_state.last_expert_load.float().cpu()
        p2l = model_state.physical_to_logical_map.cpu().long()
    if load.numel() == 0 or num_ranks < 1 or p2l.shape[1] % num_ranks:
        return None
    valid = p2l >= 0
    index = p2l.clamp(min=0)
    counts = torch.zeros_like(load).scatter_add_(1, index, valid.float())
    per_slot = torch.gather(load / counts.clamp(min=1), 1, index) * valid
    per_rank = per_slot.reshape(p2l.shape[0], num_ranks, -1).sum(-1)
    peak = per_rank.max(dim=1).values
    used = peak > 0
    if not bool(used.any()):
        return None
    return float((per_rank.mean(dim=1)[used] / peak[used]).mean())


def model_summary(state: "EplbState", model_state: "EplbModelState") -> dict:
    model = model_state.model
    num_ranks = get_ep_group().device_group.size()
    with gpu_sync_allowed():
        replicas = model_state.logical_replica_count.cpu()
    pending = model_state.pending_result
    movers = model_state.movers
    return {
        "model": model_state.model_name,
        "layers": model.num_moe_layers,
        "logical_experts": model.num_logical_experts,
        "physical_experts": model.num_physical_experts,
        "redundant_experts": model.num_redundant_experts,
        "local_slots": model.num_physical_experts // max(num_ranks, 1),
        "replicas_max": int(replicas.max()) if replicas.numel() else 0,
        "replicated_experts_per_layer": (
            float((replicas > 1).sum(dim=1).float().mean()) if replicas.numel() else 0.0
        ),
        "balancedness": _balancedness(model_state, num_ranks),
        "fast_slots": (
            None
            if model_state.fast_slots is None
            else sorted({f for f in model_state.fast_slots if f is not None})
        ),
        "store_backed": movers is not None,
        "in_flight": model_state.rebalanced,
        "pending_layer": None if pending is None else pending.layer_idx,
        "completed_async": model_state.completed_rearrangements,
        "abandoned": model_state.abandoned_rearrangements,
        "last_async_seconds": model_state.last_rearrangement_seconds,
    }


def summary(state: "EplbState") -> dict[str, Any]:
    """Everything an outside controller reads about EPLB on this rank."""
    config = state.parallel_config.eplb_config
    return {
        "settings": settings(state),
        "use_async": state.is_async,
        "paused": state.rearrangements_paused,
        "paused_intervals": state.paused_intervals,
        "step": state.expert_rearrangement_step,
        "steps_until_next": max(
            0,
            state.expert_rearrangement_step_interval - state.expert_rearrangement_step,
        ),
        "window_fill": state.window_fill,
        "next_uses_saved_load": state.next_uses_saved_load,
        "rearrangements": state.rearrangements,
        "skipped": state.skipped_rearrangements,
        "last_sync_seconds": state.last_rearrange_seconds,
        "in_flight": in_flight(state),
        "state_path": config.state_path,
        "tier_hysteresis": config.tier_hysteresis,
        "residency_mask": config.residency_mask,
        "models": [model_summary(state, ms) for ms in state.model_states.values()],
    }


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def _bounded_rows(state: "EplbState") -> int | None:
    """The most slots one rearrangement may change on a rank with the memory
    this engine set aside for it: the transfer buffer's rows when it was
    sized for the incremental policy, the spare pool rows of a store-backed
    model; None when nothing bounds it (a buffer row per local slot)."""
    bound: int | None = None
    for model_state in state.model_states.values():
        if model_state.movers is not None:
            spares = [
                m.store_binding.spare_rows
                for m in model_state.movers
                if m is not None and getattr(m, "store_binding", None) is not None
            ]
            if spares:
                rows = min(spares)
                bound = rows if bound is None else min(bound, rows)
            continue
        buffer = model_state.expert_buffer
        if not buffer:
            continue
        local = model_state.model.expert_weights[0][0].shape[0]
        rows = int(buffer[0].shape[0])
        if rows < local:
            bound = rows if bound is None else min(bound, rows)
    return bound


def check_settings(state: "EplbState", changes: dict[str, Any]) -> str | None:
    """Why `changes` cannot be made now, or None. Changes nothing."""
    unknown = sorted(set(changes) - set(SETTINGS))
    if unknown:
        return f"unknown EPLB settings {unknown}; known: {list(SETTINGS)}"
    for name in ("step_interval", "window_size", "max_moves_per_rank"):
        if name in changes:
            value = changes[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                return f"{name} must be a positive integer, got {value!r}"
    policy_name = changes.get("policy", state.parallel_config.eplb_config.policy)
    if policy_name not in EPLB_POLICIES:
        return f"unknown policy {policy_name!r}; one of {sorted(EPLB_POLICIES)}"
    busy = in_flight(state)
    if busy and ("policy" in changes or "max_moves_per_rank" in changes):
        # The async worker reads the policy when it plans a cycle.
        return f"{busy}; the policy changes between rearrangements only"
    bound = _bounded_rows(state)
    incremental = issubclass(EPLB_POLICIES[policy_name], IncrementalEplbPolicy)
    if "policy" in changes and not incremental:
        if any(ms.movers is not None for ms in state.model_states.values()):
            return (
                "the expert store moves a bounded number of experts per "
                "rearrangement: the policy stays 'incremental'"
            )
        if bound is not None:
            return (
                f"the transfer buffer holds {bound} rows per tensor (sized for "
                f"the incremental policy's max_moves_per_rank): the "
                f"'{policy_name}' policy may move every slot of a rank"
            )
    moves = changes.get("max_moves_per_rank")
    if moves is not None and bound is not None and moves > bound:
        return (
            f"max_moves_per_rank {moves} is above the {bound} rows set aside "
            "at load for one rearrangement (transfer buffer or spare pool rows)"
        )
    return None


def apply_settings(state: "EplbState", changes: dict[str, Any]) -> dict[str, list]:
    """Make `changes` (checked) and return {name: [old, new]}. A new load
    window is allocated before anything changes, so an allocation failure
    leaves this rank as it was."""
    old = settings(state)
    config = state.parallel_config.eplb_config
    windows: list[tuple[EplbModelState, torch.Tensor]] = []
    new_size = changes.get("window_size", state.expert_load_window_size)
    if new_size != state.expert_load_window_size:
        for model_state in state.model_states.values():
            window = model_state.expert_load_window
            windows.append(
                (model_state, window.new_zeros((new_size, *window.shape[1:])))
            )
    if "policy" in changes or "max_moves_per_rank" in changes:
        name = changes.get("policy", config.policy)
        moves = changes.get("max_moves_per_rank", old["max_moves_per_rank"])
        policy = EPLB_POLICIES[name]
        if issubclass(policy, IncrementalEplbPolicy):
            policy = policy.configured(moves)
        state.policy = policy
        config.policy = name
        config.max_moves_per_rank = moves
    if windows:
        _resize_windows(state, windows, new_size)
        config.window_size = new_size
    if "step_interval" in changes:
        state.expert_rearrangement_step_interval = changes["step_interval"]
        config.step_interval = changes["step_interval"]
    # The record flag for the next step follows the new interval and window.
    state._update_layer_should_record(log_stats=config.log_balancedness)
    new = settings(state)
    return {k: [old[k], new[k]] for k in SETTINGS if old[k] != new[k]}


def _resize_windows(
    state: "EplbState",
    windows: list[tuple["EplbModelState", torch.Tensor]],
    new_size: int,
) -> None:
    """Keep the most recent steps of each window, oldest first."""
    old_size = state.expert_load_window_size
    head = state.expert_load_window_step
    order = list(range(head, old_size)) + list(range(head))  # oldest first
    keep = order[-min(old_size, new_size) :]
    for model_state, window in windows:
        old = model_state.expert_load_window
        window[: len(keep)].copy_(old[keep])
        model_state.expert_load_window = window
    weights = state._window_token_weights
    state._window_token_weights = [weights[i] for i in keep] + [1.0] * (
        new_size - len(keep)
    )
    state.expert_load_window_size = new_size
    state.expert_load_window_step = len(keep) % new_size
    state.window_fill = min(state.window_fill, len(keep))


# --------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------


def check_trigger(state: "EplbState") -> str | None:
    if state.rearrangements_paused:
        return "rearrangements are paused: resume them first"
    return in_flight(state)


def trigger(state: "EplbState") -> int:
    """Start a rearrangement at the next step, on every rank alike (every
    rank's step counter is the same). Returns the counter it replaced."""
    old = state.expert_rearrangement_step
    state.expert_rearrangement_step = max(
        state.expert_rearrangement_step_interval - 1, 0
    )
    state._update_layer_should_record(
        log_stats=state.parallel_config.eplb_config.log_balancedness
    )
    return old


def untrigger(state: "EplbState", old_step: int) -> None:
    state.expert_rearrangement_step = old_step
    state.next_uses_saved_load = False
    state._update_layer_should_record(
        log_stats=state.parallel_config.eplb_config.log_balancedness
    )


def _path(state: "EplbState", path: str | None) -> str | None:
    return path or state.parallel_config.eplb_config.state_path


def check_save(state: "EplbState", path: str | None, source: str) -> str | None:
    if source not in ("last", "window"):
        return f"source must be 'last' or 'window', got {source!r}"
    target = _path(state, path)
    if target is None:
        return "no path given and no --eplb-config state_path"
    directory = os.path.dirname(os.path.abspath(target))
    if get_ep_group().device_group.rank() == 0 and not os.path.isdir(directory):
        return f"{directory} is not a directory"
    return None


def _all_ranks_ok(ok: bool) -> bool:
    """Host agreement of the EP ranks (no device memory)."""
    group = get_ep_group()
    if group.device_group.size() == 1:
        return ok
    flags = torch.tensor([int(not ok)], dtype=torch.int32, device="cpu")
    group.cpu_all_reduce_(flags)
    return int(flags.item()) == 0


def save(state: "EplbState", path: str | None, source: str) -> dict[str, Any]:
    """Write the loads to the state file (EP rank 0 writes). `last`: the
    loads the last rearrangement used, the same on every rank, no
    collective. `window`: the load window as it stands, all-reduced over the
    EP ranks like a rearrangement's (every rank must call this)."""
    target = _path(state, path)
    assert target is not None
    rank = get_ep_group().device_group.rank()
    models = list(state.model_states.values())
    if source == "window":
        error = None
        loads: list[torch.Tensor] = []
        try:
            loads = state._logical_load_windows(False, rank == 0)
        except torch.OutOfMemoryError as e:
            error = e
        if not _all_ranks_ok(error is None):
            raise RuntimeError(
                "a rank could not allocate the load window"
                + ("" if error is None else f": {str(error).splitlines()[0]}")
            )
        loads = state._allreduce_list(loads)
    else:
        loads = [ms.last_expert_load for ms in models]
    layers: dict[str, list[int]] = {}
    for model_state, window in zip(models, loads):
        with gpu_sync_allowed():
            values = window.cpu().tolist()
        for layer, layer_loads in zip(model_state.model.moe_layers, values):
            if sum(layer_loads) > 0:
                layers[layer.layer_name] = [round(x) for x in layer_loads]
    if rank == 0 and layers:
        write_expert_loads(target, layers)
    return {"path": target, "source": source, "layers": len(layers)}


def _saved_loads(state: "EplbState", path: str) -> tuple[dict, str | None]:
    if not os.path.isfile(path):
        return {}, f"{path} does not exist"
    saved = read_expert_loads(path)
    found = 0
    for model_state in state.model_states.values():
        model = model_state.model
        for layer in model.moe_layers:
            loads = saved.get(layer.layer_name)
            if loads is not None and len(loads) == model.num_logical_experts:
                found += 1
    if not found:
        return saved, f"{path} has no loads for the MoE layers of this model"
    return saved, None


def check_load(state: "EplbState", path: str | None) -> str | None:
    target = _path(state, path)
    if target is None:
        return "no path given and no --eplb-config state_path"
    refusal = check_trigger(state)
    if refusal is not None:
        return refusal
    if get_ep_group().device_group.rank() == 0:
        return _saved_loads(state, target)[1]
    return None


def load(state: "EplbState", path: str | None) -> dict[str, Any]:
    """Place the experts by the loads in the state file at the next step:
    EP rank 0 reads them into `last_expert_load` (the rearrangement takes
    rank 0's, as at boot), every rank starts the rearrangement. Returns the
    undo record under `undo`."""
    target = _path(state, path)
    assert target is not None
    undo: dict[str, Any] = {"loads": None}
    layers = 0
    if get_ep_group().device_group.rank() == 0:
        saved, refusal = _saved_loads(state, target)
        if refusal is not None:
            raise RuntimeError(refusal)
        undo["loads"] = [
            ms.last_expert_load.clone() for ms in state.model_states.values()
        ]
        for model_state in state.model_states.values():
            model = model_state.model
            for idx, layer in enumerate(model.moe_layers):
                loads = saved.get(layer.layer_name)
                if loads is None or len(loads) != model.num_logical_experts:
                    continue
                model_state.last_expert_load[idx].copy_(
                    torch.tensor(loads, dtype=torch.int64)
                )
                layers += 1
    undo["step"] = trigger(state)
    state.next_uses_saved_load = True
    return {"path": target, "layers": layers, "undo": undo}


def unload(state: "EplbState", undo: dict[str, Any]) -> None:
    if undo.get("loads") is not None:
        for model_state, loads in zip(state.model_states.values(), undo["loads"]):
            model_state.last_expert_load = loads
    untrigger(state, undo["step"])


# --------------------------------------------------------------------------
# The card tier's size (runtime control tiers.fast)
# --------------------------------------------------------------------------


def check_card_tier(state: "EplbState | None", rows: Any) -> str | None:
    """Why the card tier of the expert tiers cannot move to ``rows`` card
    rows per layer now, or None. EPLB owns the placement: no resize while a
    rearrangement of its is in flight; the tiers say the rest."""
    from vllm.model_executor.layers.fused_moe import xpu_fast_tier

    if state is not None:
        why = in_flight(state)
        if why is not None:
            return f"{why}; ask after it completes"
    return xpu_fast_tier.check(rows)


def resize_card_tier(state: "EplbState | None", rows: int) -> dict[str, Any]:
    """Change the card tier's capacity to ``rows`` rows per layer and let
    EPLB's movers (the layers' tiers) fill or vacate them: the rows' memory
    is mapped or released by the tiers, which slots enter or leave is the
    placement's tier choice by EPLB's load (``TierPlacement.plan_card_round``,
    ``plan_pinned_refill``), and every move is committed through the path a
    rearrangement commits through (``ExpertTiers.commit_move``). Called only
    by the runtime control surface, between two steps; never by EPLB's own
    step (the size of the card tier is decided outside the engine)."""
    from vllm.model_executor.layers.fused_moe import xpu_fast_tier

    why = check_card_tier(state, rows)
    if why is not None:
        raise RuntimeError(why)
    return xpu_fast_tier.apply(rows)
