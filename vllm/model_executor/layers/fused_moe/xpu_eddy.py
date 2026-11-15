# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The eddy: a row whose MoE layer needs CPU-tier experts leaves the flow
instead of making the batch wait (VLLM_CPU_EXPERTS_EDDY=1, off by default).

At the MoE layer L of step t a row that routes to a CPU-tier expert is
*parked*: the card computes its card-tier pairs (held per pair), the CPU
computes its CPU pairs on a host thread while the step goes on, and the row
continues through the remaining layers as a substitute with no side effects
(its KV slots and recurrent-state indices are redirected to the padding
slot, its routing to a substitute that reaches no CPU and no host memory).
The step produces no token for it. At the next step the row is a
substitute up to layer L; at layer L's MoE op its real value re-enters: the
per-pair sum over the held card pairs and the CPU's rows, in top-k order,
fp32, one rounding (the same pairs and order as the waiting path, so the
same bits), with its held residual and shared-expert output. If the CPU
has not finished, the row waits in the eddy another step; the card never
waits for the CPU.

What is held is small and per row: device masks (``active``, ``resume``,
``parked_at``), per layer the parked rows' card pairs, map and router
weights, and the tensors a decoder hands to :meth:`EddyBook.hold`
(residual, shared-expert output). The host keeps per-row vectors (the
layer and step a row waits for) and reads one small tensor per step.

This module is the cell's mechanism: :meth:`EddyBook.moe_layer` is called
from ``XPUExperts._apply_tiered`` while a book is open; a driver (a test
harness here; the model runner's step hooks in an engine) opens the book
per step with the rows' index buffers, calls :meth:`EddyBook.hold` after
each MoE layer, and reads which rows produced a token at the step's end.
Decode only, one row per sequence (no drafter), the rows at fixed
positions across steps.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV = "VLLM_CPU_EXPERTS_EDDY"
_DEPTH_ENV = "VLLM_CPU_EXPERTS_EDDY_DEPTH"
PAD = -1
# resume value of a row that waits in the eddy this whole step
_WAITING = np.int32(1 << 30)
_MAX_LAYERS = 1024


def eddy_enabled() -> bool:
    """VLLM_CPU_EXPERTS_EDDY set to 1 (unset or 0: off)."""
    v = os.environ.get(_ENV, "").strip().lower()
    if v in ("", "0", "off"):
        return False
    if v != "1":
        raise ValueError(f"{_ENV} is 1 or unset, not {v!r}")
    return True


def _depth() -> int:
    d = int(os.environ.get(_DEPTH_ENV, "") or 2)
    if not 1 <= d <= 3:
        raise ValueError(f"{_DEPTH_ENV} is 1..3 (steps a CPU job may lag), not {d}")
    return d


_BOOK: EddyBook | None = None


def current_book() -> EddyBook | None:
    """The book of the step in flight, or None (always None with the switch
    unset: no book is ever opened)."""
    return _BOOK


class EddyError(RuntimeError):
    pass


@dataclass
class _Hold:
    """Per MoE layer: the parked rows' card pairs ([rows * top_k, hidden]),
    their map (pair -> held row, -1 for a CPU pair) and router weights; this
    step's park and rejoin masks; tensors held by name (:meth:`EddyBook.hold`)."""

    pairs: torch.Tensor
    pair_map: torch.Tensor
    weights: torch.Tensor
    park: torch.Tensor
    rejoin: torch.Tensor
    named: dict[str, torch.Tensor] = field(default_factory=dict)


class _EddyHost:
    """The CPU parts of parked rows, one layer call at a time, on a thread
    of its own. ``done[layer, slot]`` is the step whose job finished."""

    def __init__(self, runtime: Any, done: np.ndarray):
        self.runtime = runtime
        self.done = done
        self._jobs: list[tuple[Any, int, int, int] | None] = []
        self._cond = threading.Condition()
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._loop, name="cpu-experts-eddy", daemon=True
        )
        self._thread.start()

    def submit(self, pending: Any, layer: int, slot: int, step: int) -> None:
        with self._cond:
            self._jobs.append((pending, layer, slot, step))
            self._cond.notify_all()

    def wait_done(self, layer: np.ndarray, slot: int, step: np.ndarray) -> None:
        """Block until ``done[layer, slot] >= step`` for every entry (the
        ring is full: the CPU is ``depth`` steps behind)."""
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    self.error is not None
                    or bool((self.done[layer, slot] >= step).all())
                )
            )
        self.check()

    def check(self) -> None:
        if self.error is not None:
            raise EddyError(
                f"CPU experts (eddy): a parked layer call failed ({self.error})"
            ) from self.error

    def stop(self) -> None:
        with self._cond:
            self._jobs.append(None)
            self._cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._jobs))
                job = self._jobs.pop(0)
            if job is None:
                return
            pending, layer, slot, step = job
            try:
                if self.error is None:
                    self.runtime._pairs_into_buffers(pending, fill_map=True)
            except BaseException as e:  # noqa: BLE001 - raised on the model's thread
                self.error = e
            finally:
                with self._cond:
                    self.done[layer, slot] = step
                    self._cond.notify_all()


class EddyBook:
    """The substitutes' bookkeeping for one engine (rows = batch rows)."""

    def __init__(self, max_rows: int, depth: int | None = None):
        self.max_rows = max_rows
        self.depth = _depth() if depth is None else depth
        self.step = -1
        self.num_rows = 0
        # host: the layer and step (and ring slot) a row waits for
        self.wait_layer = np.full(max_rows, -1, dtype=np.int32)
        self.wait_step = np.full(max_rows, -1, dtype=np.int64)
        self.done = np.full((_MAX_LAYERS, self.depth), -1, dtype=np.int64)
        self.posted = np.full((_MAX_LAYERS, self.depth), -1, dtype=np.int64)
        self._layers: dict[str, int] = {}
        self._holds: dict[int, _Hold] = {}
        self._rejoin_slots: dict[int, list[int]] = {}
        self._index: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._tables: dict[str, tuple[int, torch.Tensor, int]] = {}
        self.host: _EddyHost | None = None
        self.device: torch.device | None = None
        # counters: rows parked, layer calls that took rejoining rows, and
        # row-steps a row waited in the eddy because its CPU job was not done
        self.parks_seen = 0
        self.rejoins = 0
        self.waited_steps = 0

    # -- per step -------------------------------------------------------------

    def begin_step(
        self,
        step: int,
        num_rows: int,
        device: torch.device,
        index_buffers: list[torch.Tensor] = (),  # type: ignore[assignment]
        padding: np.ndarray | None = None,
    ) -> None:
        """Open the book for ``step`` over rows ``0 .. num_rows-1``.

        Rows whose CPU job is done rejoin at their layer this step; rows whose
        job is not done wait the whole step (substitutes everywhere). The
        ``index_buffers`` (KV slot mappings, recurrent-state indices: one entry
        per row, ``PAD`` = touch nothing) are edited in place for the step and
        restored by :meth:`end_step`. ``padding`` (bool per row): rows that are
        not a sequence this step (graph padding, a finished sequence's place):
        substitutes everywhere, never parked, never producing."""
        global _BOOK
        if num_rows > self.max_rows:
            raise EddyError(f"{num_rows} rows above the book's {self.max_rows}")
        if self.host is not None:
            self.host.check()
        self.step, self.num_rows = step, num_rows
        self.device = device
        # The ring slot this step posts into must hold no unfinished job, and
        # it is waited for before the rows are sorted: a job finished now has
        # its rows rejoin this step, before the slot takes new inputs.
        slot = step % self.depth
        busy = np.flatnonzero(self.posted[:, slot] > self.done[:, slot])
        if busy.size:
            assert self.host is not None
            logger.info_once(
                "CPU experts (eddy): the CPU is %d steps behind the card; this "
                "step waits for its oldest layer calls (VLLM_CPU_EXPERTS_EDDY_DEPTH).",
                self.depth,
            )
            self.host.wait_done(busy, slot, self.posted[busy, slot])
        n = num_rows
        wl, ws = self.wait_layer[:n], self.wait_step[:n]
        waiting = wl >= 0
        slot_of = (ws % self.depth).astype(np.int64)
        ready = waiting & (self.done[np.maximum(wl, 0), slot_of] >= ws)
        resume = np.where(ready, wl, np.where(waiting, _WAITING, -1)).astype(np.int32)
        self.waited_steps += int((waiting & ~ready).sum())
        if padding is not None:
            resume[padding[:n]] = _WAITING
        self._rejoin_slots = {}
        if ready.any():
            pairs = np.unique(np.stack([wl[ready], slot_of[ready]], 1), axis=0)
            for layer, slot in pairs.tolist():
                self._rejoin_slots.setdefault(layer, []).append(slot)
        self._alloc(device)
        self.resume[:n].copy_(torch.from_numpy(resume))
        self.active[:n].copy_(self.resume[:n] < 0)
        self.parked_at[:n].fill_(-1)
        self._index = []
        inactive = ~self.active[:n]
        for buf in index_buffers:
            saved = buf[:n].clone()
            buf[:n].copy_(torch.where(inactive, buf.new_full((), PAD), saved))
            self._index.append((buf, saved))
        _BOOK = self

    def end_step(self) -> np.ndarray:
        """Close the step: restore the index buffers, record new parks, and
        return whether each row produced a token (bool [num_rows])."""
        global _BOOK
        _BOOK = None
        n = self.num_rows
        for buf, saved in self._index:
            buf[:n].copy_(saved)
        self._index = []
        assert self.active is not None and self.parked_at is not None
        state = torch.stack([self.active[:n].int(), self.parked_at[:n].int()]).cpu()
        produced = state[0].numpy().astype(bool)
        parked_at = state[1].numpy()
        parked = parked_at >= 0
        self.wait_layer[:n][produced] = -1
        self.wait_layer[:n][parked] = parked_at[parked]
        self.wait_step[:n][parked] = self.step
        self.parks_seen += int(parked.sum())
        if self.host is not None:
            self.host.check()
        return produced

    def release(self, rows: np.ndarray) -> None:
        """Rows whose sequence ended (finished, aborted) while parked: their
        jobs' results are dropped."""
        self.wait_layer[rows] = -1

    def stop(self) -> None:
        if self.host is not None:
            self.host.stop()
            self.host = None

    # -- per MoE layer ----------------------------------------------------------

    def _alloc(self, device: torch.device) -> None:
        if getattr(self, "active", None) is not None and self.active.device == device:
            return
        m = self.max_rows
        self.active = torch.zeros(m, dtype=torch.bool, device=device)
        self.resume = torch.full((m,), -1, dtype=torch.int32, device=device)
        self.parked_at = torch.full((m,), -1, dtype=torch.int32, device=device)

    def layer_of(self, name: str) -> int:
        """The layer's index in the book (the order of first appearance)."""
        i = self._layers.get(name)
        if i is None:
            i = self._layers[name] = len(self._layers)
            if i >= _MAX_LAYERS:
                raise EddyError(f"more than {_MAX_LAYERS} MoE layers")
        return i

    def _tables_for(self, tiers: Any, device: torch.device) -> tuple[torch.Tensor, int]:
        """(bool [num_global]: physical id computed on this rank's CPU,
        a card-tier id of this rank), rebuilt when the placement moves."""
        from vllm.model_executor.offloader.expert_tier_map import FAST, HOST

        version = int(getattr(tiers, "moves_committed", 0))
        got = self._tables.get(tiers.layer_name)
        if got is not None and got[0] == version:
            return got[1], got[2]
        tier = np.asarray(tiers._tier_of_global)
        fast = np.flatnonzero(tier == FAST)
        if fast.size == 0:
            raise EddyError(f"{tiers.layer_name}: no card-tier expert for substitutes")
        table = torch.from_numpy(tier == HOST).to(device)
        self._tables[tiers.layer_name] = (version, table, int(fast[0]))
        return table, int(fast[0])

    def _hold_for(
        self, layer: int, topk: int, hidden: int, dtype: torch.dtype
    ) -> _Hold:
        h = self._holds.get(layer)
        if h is None:
            m, dev = self.max_rows, self.device
            h = self._holds[layer] = _Hold(
                pairs=torch.zeros((m * topk, hidden), dtype=dtype, device=dev),
                pair_map=torch.full((m, topk), -1, dtype=torch.int32, device=dev),
                weights=torch.zeros((m, topk), dtype=torch.float32, device=dev),
                park=torch.zeros(m, dtype=torch.bool, device=dev),
                rejoin=torch.zeros(m, dtype=torch.bool, device=dev),
            )
        return h

    def moe_layer(
        self,
        experts: Any,
        fast: Any,
        slow: Any,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """``XPUExperts._apply_tiered`` while the book is open (per-pair sum
        only). Never waits for the CPU."""
        from vllm.model_executor.layers.fused_moe.experts import xpu_moe

        tiers = experts.tiers
        cpu = tiers.cpu
        T, K = topk_ids.shape
        H = output.shape[-1]
        dev = output.device
        if self.num_rows < T:
            raise EddyError(
                f"{tiers.layer_name}: {T} rows, the step has {self.num_rows}"
            )
        layer = self.layer_of(tiers.layer_name)
        table, fast_id = self._tables_for(tiers, dev)
        active = self.active[:T]
        rejoin = self.resume[:T] == layer
        # Substitute routing: another rank's sentinel with EP (both maps -1),
        # one card-tier id with weight 0 on a single rank. No CPU, no pinned.
        cfg = experts.moe_config
        if cfg.ep_size == 1:
            sub = fast_id
        else:  # remap_unrouted's sentinel: an id this rank does not hold
            sub = tiers.num_global - 1 if cfg.ep_rank == 0 else 0
        ids_m = torch.where(active[:, None], topk_ids, topk_ids.new_full((), sub))
        w_m = torch.where(active[:, None], topk_weights, topk_weights.new_zeros(()))
        is_cpu = table[ids_m.long()]
        hit = active & is_cpu.any(1)
        h = self._hold_for(layer, K, H, output.dtype)

        # Rejoin sources first: the ring slot a rejoining job used may be the
        # one this step posts into (its rows leave before the new inputs land:
        # the copies below are enqueued before post's, and the host thread
        # writes rows only after post's event).
        rejoin_sources = []
        for slot in self._rejoin_slots.get(layer, ()):
            b = cpu._keyed_buffers[(layer, slot)]
            rows = torch.empty((T * K, H), dtype=b.pair_rows.dtype, device=dev)
            rows.copy_(b.pair_rows[: T * K], non_blocking=True)
            rmap = torch.empty((T, K), dtype=torch.int32, device=dev)
            rmap.copy_(b.row_map[:T], non_blocking=True)
            rejoin_sources.append((rows, rmap))

        # The CPU part of the rows that park here: handed over, not awaited.
        slot = self.step % self.depth
        pending = cpu.post(
            tiers, hidden_states, ids_m, w_m, weights=False, buffers_key=(layer, slot)
        )
        cpu._pair_buffers(pending.buffers, T * K)
        if self.host is None:
            self.host = _EddyHost(cpu, self.done)
        self.posted[layer, slot] = self.step
        self.host.submit(pending, layer, slot, self.step)

        # The card's pairs of every active row; the ordinary sum.
        captured = (xpu_moe._Captured(), xpu_moe._Captured())
        fast.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=w_m,
            topk_ids=ids_m,
            output_hook=captured[0],
            **experts.impl_apply_kwargs,
        )
        xpu_moe.staged_impl(slow, T, "slow").apply(
            output=torch.empty_like(output),
            hidden_states=hidden_states,
            topk_weights=w_m,
            topk_ids=ids_m,
            output_hook=captured[1],
            **experts.impl_apply_kwargs,
        )
        sources = [c.source() for c in captured]
        xpu_moe.sum_topk(output, w_m, sources)

        # Hold the parked rows' card pairs (per pair: the sum at rejoin is the
        # same single sum), their map and weights.
        per = _pick(sources, T, K, H, output.dtype, dev)
        pair_hit = hit.repeat_interleave(K)[:, None]
        h.pairs[: T * K].copy_(torch.where(pair_hit, per, h.pairs[: T * K]))
        iota = torch.arange(T * K, dtype=torch.int32, device=dev).view(T, K)
        card_map = torch.where(is_cpu, iota.new_full((), -1), iota)
        h.pair_map[:T].copy_(torch.where(hit[:, None], card_map, h.pair_map[:T]))
        h.weights[:T].copy_(torch.where(hit[:, None], w_m.float(), h.weights[:T]))

        # Rejoin: the held card pairs and the CPU's rows, one sum.
        if rejoin_sources:
            back = torch.empty_like(output)
            xpu_moe.sum_topk(
                back,
                h.weights[:T],
                [(h.pairs[: T * K], h.pair_map[:T]), *rejoin_sources],
            )
            output.copy_(torch.where(rejoin[:, None], back, output))
            self.rejoins += len(rejoin_sources)

        # Masks and the side-effect redirects for the layers after this one.
        h.park[:T].copy_(hit)
        h.rejoin[:T].copy_(rejoin)
        self.parked_at[:T].copy_(
            torch.where(hit, self.parked_at.new_full((), layer), self.parked_at[:T])
        )
        active.copy_((active & ~hit) | rejoin)
        for buf, saved in self._index:
            cur = buf[:T]
            cur.copy_(
                torch.where(
                    hit, cur.new_full((), PAD), torch.where(rejoin, saved[:T], cur)
                )
            )
        logger.info_once(
            "CPU experts: eddy on (VLLM_CPU_EXPERTS_EDDY=1): rows that need "
            "CPU-tier experts park at that layer and rejoin there at a later "
            "step; the card does not wait for the CPU (ring depth %d).",
            self.depth,
        )

    def hold(self, layer_name: str, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """After the MoE op of ``layer_name``: keep ``tensor``'s rows that
        parked there, and give back the held rows that rejoined there."""
        layer = self._layers.get(layer_name)
        if layer is None or layer not in self._holds:
            return tensor
        h = self._holds[layer]
        T = tensor.shape[0]
        buf = h.named.get(name)
        if buf is None:
            buf = h.named[name] = torch.zeros(
                (self.max_rows, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        shape = (T,) + (1,) * (tensor.dim() - 1)
        buf[:T].copy_(torch.where(h.park[:T].view(shape), tensor, buf[:T]))
        return torch.where(h.rejoin[:T].view(shape), buf[:T], tensor)


def _pick(
    sources: list[tuple[torch.Tensor, torch.Tensor]],
    T: int,
    K: int,
    H: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """[T * K, H]: each pair's row from the first source holding it, zeros
    where none does (``pick_rows`` in torch: fixed shapes, no host sync)."""
    per = torch.zeros((T * K, H), dtype=dtype, device=device)
    taken = torch.zeros(T * K, dtype=torch.bool, device=device)
    for rows, rmap in sources:
        if rows.shape[0] == 0:
            continue
        m = rmap.reshape(-1)
        ok = (m >= 0) & ~taken
        vals = rows.index_select(0, m.clamp(min=0).long()).to(dtype)
        per = torch.where(ok[:, None], vals, per)
        taken |= ok
    return per


def drain_host_for_sync_path() -> None:
    """Before a layer call computes its CPU part on the model's thread while
    an eddy book has jobs in flight: they share the runtime's host buffers."""
    book = _ENGINE_BOOK
    if book is None or book.host is None:
        return
    for slot in range(book.depth):
        layers = np.flatnonzero(book.posted[:, slot] > book.done[:, slot])
        if layers.size:
            book.host.wait_done(layers, slot, book.posted[layers, slot])


_ENGINE_BOOK: EddyBook | None = None


def engine_book(max_rows: int) -> EddyBook:
    """The process's book (created once)."""
    global _ENGINE_BOOK
    if _ENGINE_BOOK is None or _ENGINE_BOOK.max_rows < max_rows:
        if _ENGINE_BOOK is not None:
            _ENGINE_BOOK.stop()
        _ENGINE_BOOK = EddyBook(max_rows)
    return _ENGINE_BOOK


def reset() -> None:
    global _ENGINE_BOOK, _BOOK
    if _ENGINE_BOOK is not None:
        _ENGINE_BOOK.stop()
    _ENGINE_BOOK = None
    _BOOK = None
