# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residency mask: keep the router away from experts no tier can compute.

With expert weights in tiers (card rows, host rows the card reads), an expert
id that no tier maps is skipped by the expert kernels and contributes nothing
to its token, without an error. The residency mask is the cheapest way to
keep that from happening silently: the router's logits of the masked experts
are set to -inf before top-k, so the router picks among the experts that can
be computed and renormalises over them as it always does.

The mask is a device tensor per MoE layer over logical experts (the router's
logits are logical; EPLB maps to physical slots after top-k). It is written
in place between steps, at the step hook where EPLB writes its maps, and it
must be the same on every rank (every rank routes the same tokens), so it is
always derived from all-reduced or replicated state.

Two ways to choose it (`EPLBConfig.residency_mask`):

- ``"nonresident"``: the experts the layers report as held by no tier
  (`nonresident_physical()` on the layer's expert mover). A logical expert is
  masked when every physical replica of it is non-resident.
- ``"coldest"``: an experiment mode. Per layer, the least-routed experts that
  together carry at most `residency_mask_share` of the recent load are
  masked, so the fidelity cost of not computing an expert can be measured on
  a model whose experts are all resident.

Either way the router also counts, from the logits before the mask, what the
mask displaced (a top-k of the unmasked logits beside the masked one): how
many top-k calls went to masked experts and how much renormalised routing
weight they carried, and how often each expert would have been routed. The
choice of the coldest experts uses those unmasked counts. Choosing from the
routing after the mask would lock the mask in: a masked expert is never
routed, so it would stay the coldest forever.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Rows of `ResidencyMask.stats`, per MoE layer.
DISPLACED_CALLS = 0
"""Top-k calls of the unmasked logits that went to a masked expert."""
ROUTED_CALLS = 1
"""Top-k calls of the unmasked logits (real tokens times top_k)."""
DISPLACED_WEIGHT = 2
"""Renormalised routing weight the displaced calls carried, summed."""
NUM_STATS = 3


def coldest_share_mask(load: np.ndarray, share: float, keep: int) -> np.ndarray:
    """Per layer, the least-loaded experts whose loads sum to at most `share`
    of the layer's load.

    Ties in load are broken by expert index (lower index masked first), so
    the result is a pure function of `load`: ranks that all-reduced the same
    load choose the same mask. A layer with no load masks nothing.

    Args:
        load: `[layers, experts]` load per logical expert (non-negative).
        share: in [0, 1). 0 masks nothing.
        keep: at least this many experts stay unmasked per layer (the
            router's top_k: fewer would leave top-k to pick masked experts).

    """
    load = np.asarray(load, dtype=np.float64)
    if load.ndim != 2:
        raise ValueError(f"load must be [layers, experts], got {load.shape}")
    if not 0.0 <= share < 1.0:
        raise ValueError(f"share must be in [0, 1), got {share}")
    num_layers, num_experts = load.shape
    mask = np.zeros((num_layers, num_experts), dtype=bool)
    if share == 0.0:
        return mask
    limit = max(0, num_experts - keep)
    index = np.arange(num_experts)
    for layer in range(num_layers):
        row = load[layer]
        total = row.sum()
        if total <= 0:
            continue
        order = np.lexsort((index, row))  # ascending load, then index
        cumulative = np.cumsum(row[order])
        count = int(np.searchsorted(cumulative, share * total, side="right"))
        mask[layer, order[: min(count, limit)]] = True
    return mask


def logical_mask_from_residency(
    nonresident_count: np.ndarray,
    physical_to_logical: np.ndarray,
    num_logical: int,
    keep: int,
) -> np.ndarray:
    """Logical experts with no resident physical replica.

    Args:
        nonresident_count: `[layers, physical]`, summed over the EP ranks: > 0
            where the slot's owner reports it held by no tier.
        physical_to_logical: `[layers, physical]` EPLB's placement (-1 for an
            unused slot).
        num_logical: logical experts per layer.
        keep: the router's top_k; a layer with fewer resident logical experts
            cannot be routed and raises.

    """
    nonresident = np.asarray(nonresident_count) > 0
    phys2log = np.asarray(physical_to_logical)
    num_layers = phys2log.shape[0]
    resident_replicas = np.zeros((num_layers, num_logical), dtype=np.int64)
    for layer in range(num_layers):
        used = phys2log[layer] >= 0
        logical = phys2log[layer][used]
        resident = ~nonresident[layer][used]
        np.add.at(resident_replicas[layer], logical, resident.astype(np.int64))
    mask = resident_replicas == 0
    too_few = (num_logical - mask.sum(axis=1)) < keep
    if too_few.any():
        layer = int(np.flatnonzero(too_few)[0])
        raise RuntimeError(
            f"MoE layer {layer} has {num_logical - int(mask[layer].sum())} "
            f"resident experts, fewer than top_k={keep}: the router cannot "
            "route around the experts no tier holds"
        )
    return mask


class ResidencyMask:
    """The device tensors of one model's residency mask and the host state
    that chooses it.

    `mask[l]`, `stats[l]` and `shadow_load[l]` are handed to MoE layer `l`'s
    `EplbLayerState`; the router reads `mask[l]` and adds to the other two.
    Their addresses never change (captured graphs read them), so every update
    is in place.
    """

    def __init__(
        self,
        mode: str,
        num_layers: int,
        num_logical: int,
        top_k: int,
        share: float,
        interval: int,
        window_steps: int,
        count_displaced: bool,
        device: torch.device,
    ):
        if mode not in ("nonresident", "coldest"):
            raise ValueError(f"unknown residency mask mode {mode!r}")
        if top_k <= 0 or top_k > num_logical:
            raise ValueError(f"top_k {top_k} for {num_logical} experts")
        self.mode = mode
        self.num_layers = num_layers
        self.num_logical = num_logical
        self.top_k = top_k
        self.share = share
        self.interval = interval
        self.count_displaced = count_displaced
        self.mask = torch.zeros(
            (num_layers, num_logical), dtype=torch.bool, device=device
        )
        self.stats = torch.zeros(
            (num_layers, NUM_STATS), dtype=torch.float32, device=device
        )
        self.shadow_load = torch.zeros(
            (num_layers, num_logical), dtype=torch.float32, device=device
        )
        # Unmasked load of the last intervals (all-reduced, host), newest last;
        # together they cover about `window_steps` steps.
        self._history: deque[np.ndarray] = deque(
            maxlen=max(1, -(-window_steps // interval))
        )
        self.steps = 0
        self.masked = np.zeros((num_layers, num_logical), dtype=bool)
        self.refreshes = 0

    def layer_tensors(
        self, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """What MoE layer `layer`'s router reads and writes."""
        if not self.count_displaced:
            return self.mask[layer], None, None
        return self.mask[layer], self.stats[layer], self.shadow_load[layer]

    def take_interval_load(self) -> torch.Tensor:
        """The unmasked load the routers counted since the last call (device,
        this rank's), and start counting again."""
        load = self.shadow_load.clone()
        self.shadow_load.zero_()
        return load

    def add_interval_load(self, load: np.ndarray) -> np.ndarray:
        """Add one interval's all-reduced unmasked load; return the window's."""
        self._history.append(np.asarray(load, dtype=np.float64))
        return np.sum(np.stack(list(self._history)), axis=0)

    def choose_coldest(self, window_load: np.ndarray) -> np.ndarray:
        return coldest_share_mask(window_load, self.share, self.top_k)

    def write(self, mask: np.ndarray) -> bool:
        """Write the mask in place (enqueued on the current stream, so every
        kernel enqueued after it sees the new mask and none before it does).
        Returns whether it changed."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tuple(self.mask.shape):
            raise ValueError(f"mask shape {mask.shape} != {tuple(self.mask.shape)}")
        changed = not np.array_equal(mask, self.masked)
        if changed:
            self.mask.copy_(torch.from_numpy(mask.copy()))
            self.masked = mask.copy()
        self.refreshes += 1
        return changed

    def take_stats(self, stats: np.ndarray) -> str | None:
        """The log line for the stats of the last interval (host copy of
        `stats`), or None when nothing was routed."""
        displaced = float(stats[:, DISPLACED_CALLS].sum())
        routed = float(stats[:, ROUTED_CALLS].sum())
        weight = float(stats[:, DISPLACED_WEIGHT].sum())
        per_layer = self.masked.sum(axis=1)
        if routed <= 0:
            return None
        token_layers = routed / self.top_k  # each carries routed weight 1
        return (
            f"masked experts per layer mean {per_layer.mean():.1f}, "
            f"max {int(per_layer.max())} of {self.num_logical}; "
            f"of {int(routed)} unmasked top-k calls {int(displaced)} "
            f"({100.0 * displaced / routed:.3f} %) went to masked experts, "
            f"carrying {100.0 * weight / token_layers:.3f} % of "
            "the routed weight"
        )
