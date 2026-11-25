# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV grow floor after a vision-tower round trip on a four-card engine,
from the engine's own records of two such round trips (rank 0's numbers).

First record: boot sized the cache at 8,897 blocks; the tower took 1,122 (to
7,775); when it left, torch's allocation was back (17.61 GiB, 17.58 before
the tower) but its reservation was not: 18.46 GiB after emptying the cache,
0.85 GiB held in segments the tower's tensors had shared with allocations
that outlived it. Under the cap that reservation is taken, and the grow back
was refused at 7,775.

Second record, with the tower's copy in a memory pool of its own
(`card_tower`): boot 9,004 blocks, the tower took 1,122 (to 7,882), and after
it left torch held exactly what it held at startup (17.58 GiB allocated,
17.94 reserved). The grow back stopped at 8,419 all the same: the floor held
the whole startup transient (1.54 GiB) under torch's cap, where startup
sizing itself had left torch 1.03 GiB under it (the rest is the memory kept
for allocations outside torch). With the floor under the cap no stricter
than startup sizing, the cache goes back to 9,004.

Blocks map in 2 MiB granules.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker import extensible_kv_cache as ekv

GIB = 1 << 30
GRANULE = 2 << 20
STARTUP = 1_648_974_848


def mapped_for(blk: int):
    def mapped(n: int) -> int:
        return -(-n * blk // GRANULE) * GRANULE

    return mapped


class KV:
    reserved_headroom_bytes = STARTUP
    external_mapped_bytes: list = []

    def __init__(self, blk, committed, budget, serving, startup_room=None):
        self.mapped = mapped_for(blk)
        self.blk = blk
        self.num_committed_blocks = committed
        self.torch_memory_budget = budget
        self.serving = serving
        self.startup_torch_cap_room = startup_room

    @property
    def physical_bytes(self) -> int:
        return self.mapped(self.num_committed_blocks)

    def physical_bytes_for(self, n: int) -> int:
        return self.mapped(n)

    def blocks_within(self, nbytes: int) -> int:
        n = nbytes // self.blk
        while self.mapped(n + 1) <= nbytes:
            n += 1
        while n and self.mapped(n) > nbytes:
            n -= 1
        return n

    def serving_transient_bytes(self, allocated: int, peak: int) -> int:
        return self.serving


def _floor(monkeypatch, kv, *, allocated, reserved, free, asked, setting=None):
    monkeypatch.setattr(
        ekv, "_torch_memory_now", lambda device: (allocated, allocated, reserved)
    )
    monkeypatch.setattr(
        ekv,
        "current_platform",
        SimpleNamespace(free_device_memory=lambda device=None: free),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_grow_torch_floor_mib=setting)
    )
    return ekv.grow_floor(kv, cfg, torch.device("cpu"), asked)


# ---------------------------------------------------------------- first record
A_BLK = 921_500
A_ALLOCATED = 18_910_072_320
A_RESERVED = 19_824_377_856
A_STRANDED = A_RESERVED - A_ALLOCATED
A_SERVING = 1_555_345_920
A_MAPPED = mapped_for(A_BLK)
A_BUDGET = 21_409_767_424 + A_MAPPED(7775)
# Rank 0's free memory at 7,775 blocks, after emptying torch's cache; ranks
# 1-3 read 1.81 GiB.
A_FREE = 736_432_128 + 374 * GRANULE
A_FREE_OTHERS = int(1.81 * GIB)


def _a_kv():
    return KV(A_BLK, 7775, A_BUDGET, A_SERVING)


def test_the_refused_grow_as_recorded(monkeypatch):
    f = _floor(
        monkeypatch,
        _a_kv(),
        allocated=A_ALLOCATED,
        reserved=A_RESERVED,
        free=A_FREE,
        asked=8626,
    )
    assert A_MAPPED(8897) - A_MAPPED(7775) == 1_033_895_936  # the shrink's release
    assert (f.floor, f.source, f.cap_floor) == (STARTUP, "transient", STARTUP)
    assert f.cap_after == 20_625_432_576
    assert f.free_after == 736_432_128
    assert f.headroom_after == 801_054_720
    assert f.max_blocks == 7775


def test_8897_blocks_on_the_recorded_state_would_leave_torch_short():
    cap = A_BUDGET - A_MAPPED(8897)
    free = A_FREE - (A_MAPPED(8897) - A_MAPPED(7775))
    # New segments torch could still reserve under the cap.
    assert cap - A_RESERVED == 551_493_632
    # Even counting the stranded 0.85 GiB as usable, less than serving needed.
    assert A_SERVING - (cap - A_ALLOCATED) == 89_546_752
    assert A_SERVING - (free + A_STRANDED) == 154_169_344


@pytest.mark.parametrize(
    ("free", "expected"),
    [(A_FREE, 8627), (A_FREE_OTHERS, 8698)],  # the device binds; the cap does
)
def test_with_the_towers_segments_back_the_grow_reaches_the_floor(
    monkeypatch, free, expected
):
    # The tower's segments went back to the device with its pool.
    free += A_STRANDED
    f = _floor(
        monkeypatch,
        _a_kv(),
        allocated=A_ALLOCATED,
        reserved=A_ALLOCATED,
        free=free,
        asked=8897,
    )
    assert f.max_blocks == expected
    # Torch keeps its whole startup transient on the device and under the cap.
    assert A_BUDGET - A_MAPPED(expected) - A_ALLOCATED >= STARTUP
    assert free - (A_MAPPED(expected) - A_MAPPED(7775)) >= STARTUP


# --------------------------------------------------------------- second record
B_BLK = 921_600
B_ALLOCATED = 18_880_546_304
B_RESERVED = 19_264_438_272
B_SERVING = 23_769_600
B_MAPPED = mapped_for(B_BLK)
# Torch's cap at startup, with 9,004 blocks mapped.
B_STARTUP_LIMIT = 20_376_227_840
B_BUDGET = B_STARTUP_LIMIT + B_MAPPED(9004)
# Rank 0's free memory at 7,882 blocks.
B_FREE = 1_472_204_800 + (B_MAPPED(8835) - B_MAPPED(7882))


def _b_kv(startup_room=None):
    return KV(B_BLK, 7882, B_BUDGET, B_SERVING, startup_room)


def _b_floor(monkeypatch, kv, reserved=B_RESERVED, setting=None):
    return _floor(
        monkeypatch,
        kv,
        allocated=B_ALLOCATED,
        reserved=reserved,
        free=B_FREE,
        asked=9004,
        setting=setting,
    )


def test_the_second_record_stops_at_8419_with_the_whole_floor_under_the_cap(
    monkeypatch,
):
    assert B_MAPPED(9004) == 8_298_430_464  # the startup mapping
    assert B_BUDGET - B_MAPPED(8835) == 20_531_417_088  # recorded cap after
    f = _b_floor(monkeypatch, _b_kv())
    assert f.cap_floor == STARTUP
    assert f.max_blocks == 8419


def test_back_at_its_startup_state_torch_lets_the_cache_regrow_to_startup(
    monkeypatch,
):
    # Startup left torch this much under its cap; torch reserves what it did
    # then.
    room = B_STARTUP_LIMIT - B_RESERVED
    assert room < STARTUP
    f = _b_floor(monkeypatch, _b_kv(room))
    assert f.cap_floor == room
    assert f.max_blocks == 9004
    # The device side still keeps the whole floor.
    device_after = B_FREE - (B_MAPPED(9004) - B_MAPPED(7882))
    assert device_after + (B_RESERVED - B_ALLOCATED) - STARTUP == 51_932_672


def test_a_reservation_beyond_the_startup_one_still_counts_as_taken(monkeypatch):
    room = B_STARTUP_LIMIT - B_RESERVED
    f = _b_floor(monkeypatch, _b_kv(room), reserved=B_RESERVED + 64 * (1 << 20))
    assert f.max_blocks < 9004
    assert B_BUDGET - B_MAPPED(f.max_blocks) - (B_RESERVED + 64 * (1 << 20)) >= room


def test_a_serving_transient_above_the_startup_one_is_kept_in_full_beyond_it(
    monkeypatch,
):
    room = B_STARTUP_LIMIT - B_RESERVED
    kv = _b_kv(room)
    kv.serving = STARTUP + 256 * (1 << 20)
    f = _b_floor(monkeypatch, kv)
    assert f.floor == kv.serving
    assert f.cap_floor == room + 256 * (1 << 20)
    assert f.max_blocks < 9004


def test_a_floor_set_by_hand_is_held_whole_under_the_cap(monkeypatch):
    room = B_STARTUP_LIMIT - B_RESERVED
    f = _b_floor(monkeypatch, _b_kv(room), setting=1024)
    assert (f.source, f.floor, f.cap_floor) == ("setting", 1 << 30, 1 << 30)
