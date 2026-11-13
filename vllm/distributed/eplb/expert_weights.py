# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert weights as EPLB addresses them: one row per local expert slot.

A plain expert tensor is viewed as `[num_local_slots, -1]`, one storage. A
layer may instead keep its slots in tiers of different memory, for example
the first slots in device memory and the rest in pinned host memory that the
kernels read through UVA (`--cpu-offload-per-expert`). Such a tensor is a
`TieredExpertWeight`: EPLB indexes it by slot like a plain tensor, so its
transfer plan, buffers and maps stay the same, and the slot order doubles as
the tier order. Its policy step `order_slots_by_tier` keeps the most-loaded
experts of each rank in the first (fast) tier.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import numpy as np
import torch


class TieredExpertWeight:
    """One expert tensor of a layer whose local slots span several storages.

    Slots `[0, n0)` are the rows of `parts[0]`, `[n0, n0 + n1)` those of
    `parts[1]`, and so on. Tier 0 is the fast tier. Every part is a contiguous
    `[n_i, row]` tensor of the same dtype and row length, readable and writable
    from the device (a host part through a UVA view).

    `on_rows_written(slots)` is called after EPLB wrote rows into the slower
    tiers, so a cache of their contents (such as a staged device copy) can be
    dropped.
    """

    def __init__(
        self,
        parts: Sequence[torch.Tensor],
        on_rows_written: Callable[[list[int]], None] | None = None,
    ):
        assert parts, "a tiered expert weight needs at least one part"
        first = parts[0]
        for part in parts:
            assert part.dim() == 2 and part.is_contiguous(), tuple(part.shape)
            assert part.dtype == first.dtype and part.shape[1] == first.shape[1], (
                f"tier parts differ: {part.dtype} {tuple(part.shape)} vs "
                f"{first.dtype} {tuple(first.shape)}"
            )
        self.parts = tuple(parts)
        self.on_rows_written = on_rows_written
        self._starts = np.cumsum([0] + [p.shape[0] for p in self.parts])

    @property
    def tier_sizes(self) -> tuple[int, ...]:
        return tuple(p.shape[0] for p in self.parts)

    @property
    def shape(self) -> torch.Size:
        return torch.Size((int(self._starts[-1]), self.parts[0].shape[1]))

    @property
    def dtype(self) -> torch.dtype:
        return self.parts[0].dtype

    @property
    def device(self) -> torch.device:
        return self.parts[0].device

    def element_size(self) -> int:
        return self.parts[0].element_size()

    def __len__(self) -> int:
        return int(self._starts[-1])

    def tier_of(self, slot: int) -> int:
        if not 0 <= slot < len(self):
            raise IndexError(f"slot {slot} of {len(self)}")
        return int(np.searchsorted(self._starts, slot, side="right") - 1)

    def __getitem__(self, slot: int) -> torch.Tensor:
        tier = self.tier_of(slot)
        return self.parts[tier][slot - int(self._starts[tier])]

    def view(self, dtype: torch.dtype) -> "TieredExpertWeight":
        """The same slots reinterpreted as `dtype` (equal element size)."""
        return TieredExpertWeight(
            [p.view(dtype) for p in self.parts], self.on_rows_written
        )

    def rows_written(self, slots: Iterable[int]) -> None:
        slow = [s for s in slots if s >= self.parts[0].shape[0]]
        if slow and self.on_rows_written is not None:
            self.on_rows_written(slow)


class ExpertMover(Protocol):
    """A layer that places its own experts, instead of EPLB sending them.

    The weights of an expert never change and each has a durable home (the
    expert store), so a rearrangement is, for the layer, reading what it newly
    holds from there and then switching which row serves which slot. EPLB keeps
    the decisions (loads, policy, the replicated maps) and the timing: it calls
    :meth:`prepare` off the step thread and :meth:`commit` between two steps,
    in the window where it commits its own maps, on every rank at the same step.
    """

    def prepare(
        self,
        old_phys2log: np.ndarray,
        new_phys2log: np.ndarray,
        logical_load: np.ndarray,
        hysteresis: float,
    ) -> Any:
        """Read what this rank's slots newly hold. Any thread; touches no
        device state. ``*_phys2log``: the logical expert of every physical
        slot of the layer; ``logical_load``: load per logical expert."""

    def commit(self, prepared: Any) -> None:
        """Switch to the prepared placement (enqueue its device work). The
        thread that enqueues the forward, between two steps."""

    def discard(self, prepared: Any) -> None:
        """Give up a prepared placement that will not be committed."""

    def fill_to(
        self, phys2log: np.ndarray, logical_load: np.ndarray, hysteresis: float
    ) -> None:
        """Put the layer straight into a placement, reading every slot from its
        home. Only before serving: nothing reads the rows."""

    def prepared_stats(self, prepared: Any) -> dict[str, float]:
        """Units and bytes read and milliseconds spent, for the logs."""

    @property
    def scrub_every(self) -> int:
        """A debugging aid of the layer: check it after every Nth
        rearrangement (and once after load); 0 for never."""

    def scrub(self, phys2log: np.ndarray | None = None, full: bool = True) -> dict:
        """Compare what the layer's kernels read with the experts' home and
        the device's maps with the placement (and with ``phys2log``, EPLB's
        map of the layer). The thread that enqueues the forward, between two
        steps; blocking. Keys: ``slots``, ``mismatches``, ``maps_equal``,
        ``placement_equal``, ``first``."""


class StoreBackedWeight(TieredExpertWeight):
    """A tiered expert tensor of a layer whose slots are placed by an
    :class:`ExpertMover`. EPLB reads shapes and sizes from it as from any
    tiered tensor; the bytes are moved by ``mover``, never by EPLB.

    It cannot be addressed by slot. In such a layer the row that serves a
    slot is the mover's state and changes while serving, so "slot ``s`` is
    row ``s`` of a part" stops being true after the first tier move: reading
    or writing a row through this object would silently touch another
    expert. Every by-slot access therefore raises; EPLB's transfer paths
    (which index weights by slot) must not be reached for these layers.

    ``unbacked`` slots beyond the parts have no row at all (a layer that
    computes some experts on the CPU from the store): they count in the
    shape and as a last tier, with no tensor behind them.
    """

    def __init__(
        self,
        parts: Sequence[torch.Tensor],
        on_rows_written: Callable[[list[int]], None] | None,
        mover: ExpertMover,
        unbacked: int = 0,
    ):
        super().__init__(parts, on_rows_written)
        self.mover = mover
        assert unbacked >= 0
        self.unbacked = unbacked

    @property
    def tier_sizes(self) -> tuple[int, ...]:
        sizes = super().tier_sizes
        return sizes + (self.unbacked,) if self.unbacked else sizes

    @property
    def shape(self) -> torch.Size:
        return torch.Size((len(self), self.parts[0].shape[1]))

    def __len__(self) -> int:
        return int(self._starts[-1]) + self.unbacked

    def view(self, dtype: torch.dtype) -> "StoreBackedWeight":
        return StoreBackedWeight(
            [p.view(dtype) for p in self.parts],
            self.on_rows_written,
            self.mover,
            self.unbacked,
        )

    def _by_slot(self, what: str) -> RuntimeError:
        return RuntimeError(
            f"{what}: the rows of this layer are placed by its expert mover "
            "and are not addressable by slot (a store-backed layer is "
            "rearranged through ExpertMover.prepare and commit only)"
        )

    def tier_of(self, slot: int) -> int:
        raise self._by_slot("tier_of")

    def __getitem__(self, slot: int) -> torch.Tensor:
        raise self._by_slot("row access")

    def rows_written(self, slots: Iterable[int]) -> None:
        raise self._by_slot("rows_written")


ExpertWeight = torch.Tensor | TieredExpertWeight


def expert_mover(weights: Sequence[Any]) -> ExpertMover | None:
    """The mover of a layer whose experts are placed by the layer itself."""
    for weight in weights:
        if isinstance(weight, StoreBackedWeight):
            return weight.mover
    return None


def new_expert_buffer(weight: ExpertWeight) -> torch.Tensor:
    """A device buffer with one row per slot of `weight`."""
    if isinstance(weight, torch.Tensor):
        return torch.empty_like(weight)
    return torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)


def notify_rows_written(weights: Sequence[ExpertWeight], slots: list[int]) -> None:
    """Tell tiered weights which slots EPLB wrote."""
    for weight in weights:
        if isinstance(weight, TieredExpertWeight):
            weight.rows_written(slots)


def fast_slot_count(weights: Sequence[ExpertWeight]) -> int | None:
    """Slots of the fast tier of a layer's experts, or None without tiers."""
    for weight in weights:
        if isinstance(weight, TieredExpertWeight):
            return weight.tier_sizes[0]
    return None


def order_slots_by_tier(
    phy2log: np.ndarray,
    load: np.ndarray,
    fast_slots: Sequence[int | None],
    num_ranks: int,
    hysteresis: float,
) -> np.ndarray:
    """Keep the most-loaded experts of each rank in its fast slots.

    A permutation of each rank's slots, after the policy decided which experts
    a rank holds: the first `fast_slots[layer]` local slots of a rank are its
    fast tier. The most-loaded expert in a slow slot trades places with the
    least-loaded one in a fast slot while it carries more than `hysteresis`
    times that load, so near-equal experts do not move back and forth.

    Args:
        phy2log: [layers, num_physical] logical expert of each physical slot.
        load: [layers, num_logical] load per logical expert.
        fast_slots: per layer, fast slots per rank, or None (no tiers).
        num_ranks: EP ranks; each owns `num_physical // num_ranks` slots.
        hysteresis: >= 1.

    Returns:
        The new [layers, num_physical] map.

    """
    out = phy2log.copy()
    num_layers, num_physical = phy2log.shape
    per_rank = num_physical // num_ranks
    for layer in range(num_layers):
        fast = fast_slots[layer]
        if fast is None or fast >= per_rank:
            continue
        row = out[layer]
        valid = row >= 0
        replicas = np.bincount(row[valid], minlength=load.shape[1])
        slot_load = np.full(num_physical, -1.0)
        slot_load[valid] = load[layer][row[valid]] / replicas[row[valid]]
        for rank in range(num_ranks):
            base = rank * per_rank
            fast_idx = np.arange(base, base + fast)
            slow_idx = np.arange(base + fast, base + per_rank)
            hot = slow_idx[np.argsort(-slot_load[slow_idx], kind="stable")]
            cold = fast_idx[np.argsort(slot_load[fast_idx], kind="stable")]
            for s, f in zip(hot, cold):
                if slot_load[s] <= hysteresis * max(slot_load[f], 0.0):
                    break
                row[s], row[f] = row[f], row[s]
                slot_load[s], slot_load[f] = slot_load[f], slot_load[s]
    return out
