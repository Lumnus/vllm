# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An EPLB policy that improves the current placement by a few expert moves.

The default policy places every expert from scratch, so most experts change
rank at every rearrangement even when the load barely changed (on a 512-expert,
4-rank layer about 70 % of them, for a balance gain of about 1 %). Each moved
expert is a weight transfer, and the transfer buffers are sized for the moves.

This policy starts from the current placement and, per MoE layer, applies the
moves that lower the most-loaded rank's load the most, until no move gains or
`max_moves_per_rank` slots of a rank have changed:

1. With redundant experts: a replica of the expert that loses least by giving
   one up is re-targeted to the expert with the highest load per replica.
2. Swaps of two experts between the most-loaded rank and another rank.

The bound makes the bytes of one rearrangement, and the buffer it needs, a
small fixed fraction of the model. A placement far from balanced (a fresh
start) reaches balance over several rearrangements.
"""

import numpy as np
import torch

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy


class IncrementalEplbPolicy(AbstractEplbPolicy):
    max_moves_per_rank: int = 8
    """Slots of one rank a rearrangement may change, per MoE layer."""

    min_gain: float = 1e-3
    """Smallest reduction of the most-loaded rank's load a move must bring,
    as a fraction of the mean rank load."""

    @classmethod
    def configured(
        cls, max_moves_per_rank: int, min_gain: float | None = None
    ) -> type["IncrementalEplbPolicy"]:
        """This policy with its move bound (and, when given, its smallest
        gain per swap) set."""
        attrs: dict[str, float | int] = {"max_moves_per_rank": int(max_moves_per_rank)}
        if min_gain is not None:
            attrs["min_gain"] = float(min_gain)
        return type(cls.__name__, (cls,), attrs)

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """See `AbstractEplbPolicy.rebalance_experts`. Without a current
        placement of the same shape (or with empty slots) this places from
        scratch like the default policy; the move bound then does not hold."""
        num_layers, num_logical = weight.shape
        if (
            old_global_expert_indices is None
            or tuple(old_global_expert_indices.shape) != (num_layers, num_replicas)
            or bool((old_global_expert_indices < 0).any())
            or num_replicas % num_ranks != 0
        ):
            return DefaultEplbPolicy.rebalance_experts(
                weight,
                num_replicas,
                num_groups,
                num_nodes,
                num_ranks,
                old_global_expert_indices,
            )
        load = weight.float().cpu().numpy().astype(np.float64)
        phy2log = old_global_expert_indices.cpu().numpy().astype(np.int64).copy()
        for layer in range(num_layers):
            cls._improve_layer(phy2log[layer], load[layer], num_logical, num_ranks)
        return torch.from_numpy(phy2log)

    @classmethod
    def _improve_layer(
        cls,
        slots: np.ndarray,
        load: np.ndarray,
        num_logical: int,
        num_ranks: int,
    ) -> None:
        """Improve one layer's `slots` (physical slot -> logical expert) in
        place."""
        per_rank = slots.shape[0] // num_ranks
        budget = np.full(num_ranks, cls.max_moves_per_rank, dtype=np.int64)
        count = np.bincount(slots, minlength=num_logical).astype(np.float64)
        if (count == 0).any():
            return  # not a valid placement to start from; leave it
        rank_of = np.repeat(np.arange(num_ranks), per_rank)

        def rank_loads() -> np.ndarray:
            return np.bincount(
                rank_of, weights=load[slots] / count[slots], minlength=num_ranks
            )

        mean = float(load.sum()) / num_ranks
        if mean <= 0:
            return
        tol = cls.min_gain * mean

        # 1. Re-target replicas towards the experts with the most load each.
        if slots.shape[0] > num_logical:
            while True:
                per_replica = load / count
                hot = int(np.argmax(per_replica))
                spare = np.flatnonzero(count > 1)
                spare = spare[spare != hot]
                if spare.size == 0:
                    break
                # The expert whose replicas get the least loaded by giving one up.
                after = load[spare] / (count[spare] - 1)
                cold = int(spare[np.argmin(after)])
                if max(load[hot] / (count[hot] + 1), float(after.min())) >= (
                    per_replica[hot] * (1 - cls.min_gain)
                ):
                    break
                loads = rank_loads()
                holds_hot = np.zeros(num_ranks, dtype=bool)
                holds_hot[rank_of[slots == hot]] = True
                candidates = np.flatnonzero(
                    (slots == cold) & (budget[rank_of] > 0) & ~holds_hot[rank_of]
                )
                if candidates.size == 0:
                    break
                # The replica on the least-loaded rank takes the hot expert.
                slot = int(candidates[np.argmin(loads[rank_of[candidates]])])
                slots[slot] = hot
                count[cold] -= 1
                count[hot] += 1
                budget[rank_of[slot]] -= 1

        # 2. Swap experts between the most-loaded rank and the others.
        while True:
            loads = rank_loads()
            hi = int(np.argmax(loads))
            if budget[hi] <= 0:
                break
            slot_load = load[slots] / count[slots]
            hi_slots = np.arange(hi * per_rank, (hi + 1) * per_rank)
            hi_experts = slots[hi_slots]
            best = (tol, -1, -1)
            for lo in range(num_ranks):
                if lo == hi or budget[lo] <= 0:
                    continue
                lo_slots = np.arange(lo * per_rank, (lo + 1) * per_rank)
                lo_experts = slots[lo_slots]
                others = np.delete(loads, [hi, lo])
                rest = float(others.max()) if others.size else 0.0
                delta = slot_load[hi_slots][:, None] - slot_load[lo_slots][None, :]
                new_max = np.maximum(
                    np.maximum(loads[hi] - delta, loads[lo] + delta), rest
                )
                gain = loads[hi] - new_max
                # No two replicas of one expert on one rank.
                gain[np.isin(hi_experts, lo_experts), :] = -np.inf
                gain[:, np.isin(lo_experts, hi_experts)] = -np.inf
                idx = int(np.argmax(gain))
                i, j = divmod(idx, per_rank)
                if gain[i, j] > best[0]:
                    best = (float(gain[i, j]), int(hi_slots[i]), int(lo_slots[j]))
            _, a, b = best
            if a < 0:
                break
            slots[a], slots[b] = slots[b], slots[a]
            budget[rank_of[a]] -= 1
            budget[rank_of[b]] -= 1
