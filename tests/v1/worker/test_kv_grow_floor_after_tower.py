# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV grow floor after a vision-tower round trip on a four-card engine,
from the engine's own record of a grow it refused.

Boot sized the cache at 8,897 blocks; the tower took 1,122 (to 7,775); when it
left, torch's allocation was back (17.61 GiB, 17.58 before the tower) but its
reservation was not: 18.46 GiB after emptying the cache, 0.85 GiB held in
segments the tower's tensors had shared with allocations that outlived it.
Under the cap that reservation is taken, and the grow back was refused at
7,775. Blocks map in 2 MiB granules (1,122 blocks = 493 granules, 851 = 374).

With the tower built in a memory pool of its own (`card_tower`), its segments
leave with it: the reservation comes back to the allocation and the grow
reaches what the floor allows on the device and under the cap.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker import extensible_kv_cache as ekv

GIB = 1 << 30
GRANULE = 2 << 20
BLK = 921_500
ALLOCATED = 18_910_072_320
RESERVED = 19_824_377_856
STRANDED = RESERVED - ALLOCATED
STARTUP = 1_648_974_848
SERVING = 1_555_345_920
LIMIT_AT_7775 = 21_409_767_424
# Rank 0's free memory at 7,775 blocks, after emptying torch's cache.
FREE_AT_7775 = 736_432_128 + 374 * GRANULE
# Ranks 1-3 read 1.81 GiB.
FREE_AT_7775_OTHERS = int(1.81 * GIB)


def mapped(n: int) -> int:
    return -(-n * BLK // GRANULE) * GRANULE


BUDGET = LIMIT_AT_7775 + mapped(7775)


class KV:
    reserved_headroom_bytes = STARTUP
    num_committed_blocks = 7775
    torch_memory_budget = BUDGET
    external_mapped_bytes: list = []

    @property
    def physical_bytes(self) -> int:
        return mapped(self.num_committed_blocks)

    def physical_bytes_for(self, n: int) -> int:
        return mapped(n)

    def blocks_within(self, nbytes: int) -> int:
        n = nbytes // BLK
        while mapped(n + 1) <= nbytes:
            n += 1
        while n and mapped(n) > nbytes:
            n -= 1
        return n

    def serving_transient_bytes(self, allocated: int, peak: int) -> int:
        return SERVING


def _floor(monkeypatch, *, reserved: int, free: int, asked: int):
    monkeypatch.setattr(
        ekv, "_torch_memory_now", lambda device: (ALLOCATED, ALLOCATED, reserved)
    )
    monkeypatch.setattr(
        ekv,
        "current_platform",
        SimpleNamespace(free_device_memory=lambda device=None: free),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    cfg = SimpleNamespace(cache_config=SimpleNamespace())
    return ekv.grow_floor(KV(), cfg, torch.device("cpu"), asked)


def _keeps_the_floor(n: int, *, reserved: int, free: int) -> bool:
    cap_after = BUDGET - mapped(n)
    free_after = free - (mapped(n) - mapped(7775))
    return (
        cap_after - reserved >= STARTUP
        and free_after + (reserved - ALLOCATED) >= STARTUP
    )


def test_the_refused_grow_as_recorded(monkeypatch):
    f = _floor(monkeypatch, reserved=RESERVED, free=FREE_AT_7775, asked=8626)
    assert mapped(8897) - mapped(7775) == 1_033_895_936  # the shrink's release
    assert (f.floor, f.source) == (STARTUP, "transient")
    assert f.cap_after == 20_625_432_576
    assert f.free_after == 736_432_128
    assert f.headroom_after == 801_054_720
    assert f.max_blocks == 7775


def test_8897_blocks_on_the_recorded_state_would_leave_torch_short():
    cap = BUDGET - mapped(8897)
    free = FREE_AT_7775 - (mapped(8897) - mapped(7775))
    # New segments torch could still reserve under the cap.
    assert cap - RESERVED == 551_493_632
    # Even counting the stranded 0.85 GiB as usable, less than serving needed.
    assert SERVING - (cap - ALLOCATED) == 89_546_752
    assert SERVING - (free + STRANDED) == 154_169_344


@pytest.mark.parametrize(
    ("free", "expected", "bound"),
    [
        # Rank 0: the device binds (it has 0.39 GiB less free than the others).
        (FREE_AT_7775, 8627, "device"),
        # Ranks 1-3: the cap binds.
        (FREE_AT_7775_OTHERS, 8698, "cap"),
    ],
)
def test_with_the_towers_pool_dropped_the_grow_goes_back_to_the_floor(
    monkeypatch, free, expected, bound
):
    # The tower's segments went back to the device with its pool.
    free += STRANDED
    f = _floor(monkeypatch, reserved=ALLOCATED, free=free, asked=8897)
    assert f.max_blocks == expected
    assert _keeps_the_floor(expected, reserved=ALLOCATED, free=free)
    assert not _keeps_the_floor(expected + 1, reserved=ALLOCATED, free=free)
    cap_room = BUDGET - mapped(expected + 1) - ALLOCATED
    device_room = free - (mapped(expected + 1) - mapped(7775))
    assert (cap_room < STARTUP) == (bound == "cap")
    assert (device_room < STARTUP) == (bound == "device")
    # Torch keeps its whole startup transient above the allocation, on the
    # device and under the cap.
    assert BUDGET - mapped(expected) - ALLOCATED >= STARTUP
    assert free - (mapped(expected) - mapped(7775)) >= STARTUP


def test_the_floor_keeps_more_under_the_cap_than_startup_sizing_left():
    # Startup capped torch at 18.98 GiB with 8,897 blocks mapped: the budget
    # less the cache. A grow back to 8,897 would leave torch the same cap,
    # 1.36 GiB above its allocation, short of the 1.54 GiB floor by the
    # 199 blocks between 8,698 and 8,897.
    cap = BUDGET - mapped(8897)
    assert round(cap / GIB, 2) == 18.98
    assert STARTUP - (cap - ALLOCATED) == 183_175_680
