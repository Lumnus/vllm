# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot, warm and cold: thresholds over EPLB's load window, for the readout
of a layer's tiers.

One signal: the load of each expert in the window EPLB's last move (a full
rearrangement, a tier pass, or the restore's fill) planned with. One order:
the slots of a rank sorted by that load. The plan cuts that order into tiers
at the tiers' capacities (card rows, then the pinned bound, the rest on the
CPU), with ``EPLBConfig.tier_hysteresis`` between near-equal experts. The
classes are thresholds over the same order: an expert at least ``tier_hot``
times the mean expert's load is hot, at least ``tier_warm`` times warm, the
rest cold; a drop to a lower class waits until the load is below the
threshold divided by ``tier_hysteresis``. Comparing the two says whether the
capacities fit the load: more warm and hot experts than card and pinned rows
is ``rows_short``, a number to act on (raise the pinned bound, or give the
card tier memory).
"""

import numpy as np

COLD, WARM, HOT = 0, 1, 2
# tier codes as in expert_tier_map (FAST, POOL, HOST)
_FAST, _POOL, _HOST = 0, 1, 2

CLASS_KEYS = (
    "hot",
    "warm",
    "cold",
    "hot_on_card",
    "warm_or_hot_on_cpu",
    "cold_on_card",
    "card_short",
    "rows_short",
)


def classify(
    temp: np.ndarray,
    prev: np.ndarray | None,
    hot: float,
    warm: float,
    hysteresis: float,
) -> np.ndarray:
    """Class per slot from its temperature (load over the mean expert's): hot,
    warm or cold; a slot whose previous class was higher keeps it while it is
    at or above that class's threshold divided by ``hysteresis``."""
    temp = np.asarray(temp, dtype=np.float64)
    raw = np.where(temp >= hot, HOT, np.where(temp >= warm, WARM, COLD)).astype(np.int8)
    if prev is None or len(prev) != len(raw):
        return raw
    prev = np.asarray(prev, dtype=np.int8)
    floor = np.where(prev == HOT, hot, np.where(prev == WARM, warm, 0.0)) / max(
        hysteresis, 1.0
    )
    keep = (raw < prev) & (temp >= floor)
    out = raw.copy()
    out[keep] = prev[keep]
    return out


def class_readout(
    cls: np.ndarray, tier: np.ndarray, num_fast: int, pinned: int
) -> dict[str, int]:
    """Counts of one layer: experts per class, the rows per tier, where the
    classes sit, and the shortfalls a larger bound or card tier would fix."""
    hot = int((cls == HOT).sum())
    warm = int((cls == WARM).sum())
    return {
        "hot": hot,
        "warm": warm,
        "cold": int((cls == COLD).sum()),
        "card_rows": int(num_fast),
        "pinned_rows": int(pinned),
        "hot_on_card": int(((cls == HOT) & (tier == _FAST)).sum()),
        "warm_or_hot_on_cpu": int(((cls >= WARM) & (tier == _HOST)).sum()),
        "cold_on_card": int(((cls == COLD) & (tier == _FAST)).sum()),
        "card_short": max(0, hot - int(num_fast)),
        "rows_short": max(0, hot + warm - int(num_fast) - int(pinned)),
    }
