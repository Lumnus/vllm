# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV grow floor across give-and-take cycles, on a model of one card.

The card's journal of one serving engine (two cards, one rank shown), from
boot to four hours later: the card tier took 251 blocks and gave them back
(+8 / -8 rows), then the vision tower came to the card on an image request,
taking 287 blocks, and left again after 31 minutes idle. Torch's numbers at
each return are the journal's; the model reproduces every number those lines
print (the measurement's fits, the headroom at the asked count, the floor's
block count). Before the serving windows the floor was torch's all-time peak
less its allocation at the moment of the grow, so the tower's own 0.84 GiB,
gone by then, counted as a transient and the KV cache stayed at 1,819 blocks.
"""

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker import extensible_kv_cache as ekv
from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache, ServingTransient

GIB = 1 << 30
DEV = torch.device("cpu")
# The card's KV cache maps 3.45 MiB per block (6.74 GiB at 2,003 blocks,
# 6.12 GiB at 1,819).
BLK = int(3.45 * (1 << 20))
# Torch's budget (its cap plus what the KV cache maps): "cap 9.25 GiB of
# budget 15.9.." at 2,003 blocks; the same budget gives "cap after: 8.95" at
# 2,090 blocks in the second cycle.
BUDGET = int(15.99 * GIB)
# Device memory torch and the KV cache share (free = this - reserved - KV).
DEVICE = int(16.25 * GIB)
STARTUP = int(1.13 * GIB)


def gib(x: float) -> int:
    return int(x * GIB)


class Torch:
    """Torch's caching allocator, as far as the floor reads it. `stuck` is
    what stays reserved after `empty_cache` (free blocks in segments that
    still hold live ones)."""

    def __init__(self, allocated: int) -> None:
        self.allocated = allocated
        self.reserved = allocated
        self.peak = allocated
        self.stuck = 0
        self.resets = 0

    def stats(self, device=None) -> dict[str, int]:
        return {
            "allocated_bytes.all.current": self.allocated,
            "allocated_bytes.all.peak": self.peak,
        }

    def alloc(self, nbytes: int) -> None:
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)
        self.reserved = max(self.reserved, self.allocated)

    def free(self, nbytes: int) -> None:
        self.allocated -= nbytes

    def step(self, transient: int) -> None:
        """A step that needs `transient` above the allocation and frees it."""
        self.peak = max(self.peak, self.allocated + transient)
        self.reserved = max(self.reserved, self.allocated + transient)

    def empty_cache(self) -> None:
        self.reserved = self.allocated + self.stuck

    def reset_peak(self, device=None) -> None:
        self.resets += 1
        self.peak = self.allocated


class CardKV:
    """The KV cache's side of the floor on the card: committed blocks, the
    budget torch shares with it, the startup transient, the serving windows
    (the real `ExtensibleKVCache` methods)."""

    serving_transient_bytes = ExtensibleKVCache.serving_transient_bytes
    steady_allocation_change = ExtensibleKVCache.steady_allocation_change
    record_sizing_baseline = ExtensibleKVCache.record_sizing_baseline

    def __init__(self, blocks: int) -> None:
        self.bytes_per_block = BLK
        self.capacity_blocks = 4096
        self.num_committed_blocks = blocks
        self.reserved_headroom_bytes = STARTUP
        self.serving_transient = ServingTransient()
        self.torch_memory_budget = BUDGET
        self.torch_memory_limit = BUDGET - self.physical_bytes
        self.external_mapped_bytes: list = []
        self.unfinished_shrink = None
        self.unfinished_grow = None
        self.last_grow = None

    @property
    def physical_bytes(self) -> int:
        return self.num_committed_blocks * BLK

    def physical_bytes_for(self, n: int) -> int:
        return n * BLK

    def blocks_within(self, nbytes: int) -> int:
        return nbytes // BLK

    def back(self, n: int) -> None:
        pass

    def commit(self, n: int) -> None:
        self.num_committed_blocks = n

    def refresh_torch_memory_limit(self) -> int:
        self.torch_memory_limit = BUDGET - self.physical_bytes
        return self.torch_memory_limit

    def shrink_to(self, n: int) -> None:
        self.num_committed_blocks = n
        self.refresh_torch_memory_limit()


class TodayKV(CardKV):
    """The cache as it was before the serving windows: the floor's serving
    term is peak - allocated (`_serving_transient` without windows)."""

    serving_transient_bytes = None


class Card:
    def __init__(self, monkeypatch, kv_cls: type[CardKV], blocks: int) -> None:
        self.torch = Torch(gib(7.27))
        self.kv = kv_cls(blocks)
        t = self.torch
        monkeypatch.setattr(
            ekv,
            "current_platform",
            SimpleNamespace(
                free_device_memory=lambda device=None: (
                    DEVICE - t.reserved - self.kv.physical_bytes
                )
            ),
        )
        # The card's margins (XPU keeps none).
        monkeypatch.setattr(
            ekv,
            "measure_kv_cache_bytes",
            partial(
                ekv.measure_kv_cache_bytes, margin_floor_bytes=0, margin_fraction=0.0
            ),
        )
        monkeypatch.setattr(ekv, "has_kv_transfer_group", lambda: False)
        monkeypatch.setattr(ekv, "rebind_committed_views", lambda runner: None)
        monkeypatch.setattr(ekv, "_allocator_total_memory", lambda device: DEVICE)
        monkeypatch.setattr(ekv, "_device_properties_total_memory", lambda device: None)
        monkeypatch.setattr(torch.accelerator, "memory_stats", t.stats)
        monkeypatch.setattr(
            torch.accelerator, "memory_reserved", lambda device=None: t.reserved
        )
        monkeypatch.setattr(
            torch.accelerator, "memory_allocated", lambda device=None: t.allocated
        )
        monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", t.reset_peak)
        monkeypatch.setattr(torch.accelerator, "empty_cache", t.empty_cache)
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: None)
        self.runner = SimpleNamespace(
            extensible_kv_cache=self.kv,
            vllm_config=SimpleNamespace(
                attention_config=SimpleNamespace(hisparse_config=None),
                model_config=SimpleNamespace(multimodal_config=None),
                parallel_config=SimpleNamespace(),
                cache_config=SimpleNamespace(),
            ),
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                    is_block_compact=True
                )
            ),
            kv_cache_config=SimpleNamespace(num_blocks=blocks),
            device=DEV,
            kv_caches_by_layer={},
        )

    def grow(self, target: int, restore: bool = False) -> int:
        return ekv.grow_kv_cache(
            self.runner,
            target,
            init_free_memory=64 * GIB,
            requested_memory=64 * GIB,
            reduce_min=lambda v: v,
            restore=restore,
        )

    def move(self, windows: bool, what: str):
        """What the worker does around a tower move: tell the KV cache (with
        windows), or nothing (as before)."""
        if not windows:
            return nullcontext()
        return self.kv.steady_allocation_change(DEV, what)

    def torch_room_under_cap(self) -> int:
        return BUDGET - self.kv.physical_bytes - self.torch.reserved


def _journal(card: Card, windows: bool) -> tuple[int, int]:
    """Boot to the tower's return, the serving engine's sequence. Returns the
    block counts after the card tier's return and after the tower's."""
    t, kv = card.torch, card.kv
    kv.record_sizing_baseline(DEV)
    # Serving at boot (2K x 4 loads): 1.02 GiB above the allocation, and
    # torch keeps 0.50 GiB it cannot give back.
    t.step(gib(1.02))
    t.stuck = gib(7.77) - gib(7.27)
    t.empty_cache()
    # tiers fast +8: the KV cache gives 251 blocks; -8: they come back as a
    # restore grow (the card tier's shrink).
    kv.shrink_to(2003)
    tier_return = card.grow(2253, restore=True)
    # Second image request: the KV cache gives 287 blocks, then every rank
    # builds the tower's copy on the device (torch memory) and loads it.
    kv.shrink_to(1819)
    with card.move(windows, "tower vision_tower put on the device"):
        t.step(gib(0.10))
        t.alloc(gib(0.84))
    # Image requests while the tower is on the card; serving keeps 0.03 GiB.
    t.step(gib(0.84))
    t.alloc(gib(0.03))
    # 31 minutes without images: the copy leaves; the segments it shared with
    # allocations made meanwhile stay reserved (8.29 GiB after emptying).
    with card.move(windows, "tower vision_tower taken off the device"):
        t.free(gib(0.84))
    t.stuck = gib(8.29) - gib(7.30)
    t.empty_cache()
    tower_return = card.grow(2106)
    return tier_return, tower_return


def test_the_model_prints_the_journal_numbers(monkeypatch):
    """Today's floor on the model gives the journal's lines: the tier's
    return to about 2,106 of 2,253 (measurement fits 2,252), the tower's
    return not grown from 1,819 (measurement fits 2,090; floor 1.65 GiB)."""
    card = Card(monkeypatch, TodayKV, 2254)
    t = card.torch
    tier_return, tower_return = _journal(card, windows=False)
    assert 2100 <= tier_return <= 2110
    assert tower_return == 1819  # (a): today's code ends at 1,819
    report = card.kv.last_grow
    assert 2085 <= report["measured_blocks"] <= 2092
    assert report["floor_blocks"] == 1819
    assert abs(report["floor_bytes"] - gib(1.65)) < gib(0.01)
    # Torch as the tower's return line reads it.
    assert abs(t.allocated - gib(7.30)) < gib(0.01)
    assert abs(t.reserved - gib(8.29)) < gib(0.01)
    assert abs(t.peak - gib(8.95)) < gib(0.01)


def test_the_tower_round_trip_leaves_the_floor_at_what_serving_needs(monkeypatch):
    """(b) With the serving windows the tower's own bytes and its move's peak
    stay out of the floor: the floor is the startup transient (1.13 GiB, more
    than serving ever needed above its steady allocation), and the KV cache
    comes back to the most blocks that leave torch that much under its cap
    with its reservation as it is (about 1,950), up from 1,819."""
    card = Card(monkeypatch, CardKV, 2254)
    tier_return, tower_return = _journal(card, windows=True)
    assert 2100 <= tier_return <= 2110  # the tier's return did not change
    assert 1945 <= tower_return <= 1955
    report = card.kv.last_grow
    assert report["floor_bytes"] == STARTUP
    assert report["floor_blocks"] == tower_return
    # The floor holds after the grow, under the cap and on the device.
    assert card.torch_room_under_cap() >= STARTUP
    # Serving's own transients were kept: 1.02 before the tier, 0.84 with
    # the tower on the card (above its steady allocation then).
    assert abs(card.kv.serving_transient.closed_bytes - gib(1.02)) < gib(0.01)


def test_the_floor_still_counts_what_torch_keeps_reserved(monkeypatch):
    """(c) The S26 shape: a return to the exact earlier size past what torch's
    allocator has room for is still refused. After the tower's round trip
    torch holds 0.52 GiB more reserved than before it (fragments it cannot
    give back), so even a restore grow stops at the floor, short of 2,106."""
    card = Card(monkeypatch, CardKV, 2254)
    _, tower_return = _journal(card, windows=True)
    assert card.grow(2106, restore=True) == tower_return
    assert card.kv.last_grow["floor_blocks"] == tower_return
    assert card.torch_room_under_cap() >= STARTUP


def test_the_tier_return_is_a_restore_that_keeps_the_floor(monkeypatch):
    """(c) The card tier's -8: the restore asks for 2,253 blocks; the floor
    gives about 2,106, and torch keeps the startup transient under its cap
    (the 295a2cdff6 code mapped all 2,253 and the next 2K x 4 load died)."""
    card = Card(monkeypatch, CardKV, 2254)
    tier_return, _ = _journal(card, windows=True)
    assert tier_return < 2253
    # At 2,253 torch would have had 0.63 GiB under its cap (journal 0.64).
    room_at_asked = BUDGET - 2253 * BLK - gib(7.77)
    assert abs(room_at_asked - gib(0.63)) < gib(0.02)


def test_a_transient_serving_showed_stays_in_the_floor(monkeypatch):
    """A load that needed more than the startup transient (1.5 GiB above the
    steady allocation) keeps the floor there through a tower round trip: the
    windows drop only the move's bytes, never what serving has shown."""
    card = Card(monkeypatch, CardKV, 2254)
    t, kv = card.torch, card.kv
    kv.record_sizing_baseline(DEV)
    t.step(gib(1.5))
    kv.shrink_to(1819)
    with card.move(True, "tower vision_tower put on the device"):
        t.alloc(gib(0.84))
    t.step(gib(0.5))
    with card.move(True, "tower vision_tower taken off the device"):
        t.free(gib(0.84))
    t.empty_cache()
    card.grow(2106)
    assert abs(kv.last_grow["floor_bytes"] - gib(1.5)) < gib(0.001)
    assert card.torch_room_under_cap() >= gib(1.5)


def test_a_transient_on_top_of_the_tower_counts_from_the_tower(monkeypatch):
    """With the tower on the card, a step's transient counts above the
    allocation that includes the tower; one larger than the startup
    transient sets the floor after the tower left."""
    card = Card(monkeypatch, CardKV, 2254)
    t, kv = card.torch, card.kv
    kv.record_sizing_baseline(DEV)
    with card.move(True, "tower vision_tower put on the device"):
        t.alloc(gib(0.84))
    t.step(gib(1.4))
    with card.move(True, "tower vision_tower taken off the device"):
        t.free(gib(0.84))
    allocated, peak = t.allocated, t.peak
    floor, source = ekv.grow_torch_floor_bytes(
        kv, card.runner.vllm_config, allocated, peak
    )
    assert source == "transient"
    assert abs(floor - gib(1.4)) < gib(0.001)


def test_serving_transient_windows():
    s = ServingTransient()
    g = GIB
    # No window open: peak - allocated, as before.
    assert s.bytes(7 * g, 8 * g) == g
    s.open(7 * g)
    assert s.bytes(7 * g, 8 * g) == g
    # The allocation fell inside the window (a holder freed without a move):
    # counted from the lower one, the conservative side.
    assert s.bytes(6 * g, 8 * g) == 2 * g
    s.close(7 * g, 8 * g)
    assert s.closed_bytes == g and s.base is None
    # A higher steady allocation in the next window counts from its base.
    s.open(9 * g)
    assert s.bytes(9 * g, 9 * g + g // 2) == g  # the closed window's 1 GiB
    assert s.bytes(9 * g, 11 * g) == 2 * g


def test_a_move_resets_the_peak_and_opens_a_window_from_its_allocation(monkeypatch):
    card = Card(monkeypatch, CardKV, 2254)
    t, kv = card.torch, card.kv
    kv.record_sizing_baseline(DEV)
    t.step(gib(0.5))
    resets = t.resets
    with kv.steady_allocation_change(DEV, "a test move"):
        t.alloc(gib(2))
        t.step(gib(3))
    assert t.resets == resets + 1
    assert kv.serving_transient.base == t.allocated
    assert kv.serving_transient.closed_bytes == gib(0.5)
    # A move that raises still opens the next window.
    with (
        pytest.raises(RuntimeError),
        kv.steady_allocation_change(DEV, "a failing move"),
    ):
        raise RuntimeError("commit failed")
    assert kv.serving_transient.base == t.allocated


def test_the_worker_tells_the_kv_cache_around_a_tower_move(monkeypatch):
    gpu_worker = pytest.importorskip("vllm.v1.worker.gpu_worker")
    seen: list[str] = []

    class Kv:
        def steady_allocation_change(self, device, what):
            seen.append(f"{what} on {device}")
            return nullcontext()

    tower = SimpleNamespace(
        commit=lambda wait_s: seen.append("commit") or {"bytes": 1},
        release=lambda: seen.append("release") or {"bytes": 1},
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(extensible_kv_cache=Kv(), device=DEV),
        device=DEV,
        _card_tower=lambda name: tower,
    )
    worker._steady_allocation_change = partial(
        gpu_worker.Worker._steady_allocation_change, worker
    )
    gpu_worker.Worker.tower_residency_commit(worker, "vision_tower")
    gpu_worker.Worker.tower_residency_release(worker, "vision_tower")
    assert seen == [
        "tower vision_tower put on the device on cpu",
        "commit",
        "tower vision_tower taken off the device on cpu",
        "release",
    ]
    # Without an extensible KV cache the moves run as before.
    worker.model_runner = SimpleNamespace(extensible_kv_cache=None, device=DEV)
    seen.clear()
    gpu_worker.Worker.tower_residency_commit(worker, "vision_tower")
    assert seen == ["commit"]
