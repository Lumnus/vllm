# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shrinking the extensible KV cache on the workers while serving, on the
fake of the XPU VMM ops (unmapped pages read back as 0xA5)."""

import copy
import threading
from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.test_xpu_extensible_kv_cache import (  # noqa: F401
    MIB,
    MinGroup,
    _kv_cache_config,
    fake_xpu,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import extensible_kv_cache
from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.utils import AttentionGroup

POISON = 0xA5
FILL = 0x11
CHUNK = 4 * MIB  # four blocks of each 1 MiB layer segment
COMMITTED = 40
KERNEL_SPLIT = 4  # kernel blocks per block
MAX_REQS, WIDTH, MAX_TOKENS = 4, 16, 32


def _block_tables(stale_id: int, block_size: int) -> BlockTables:
    """Forward tables as a run leaves them: a live row up to its own blocks,
    stale ids from earlier rows past them, and a finished request's row."""
    tables = object.__new__(BlockTables)
    state = torch.zeros(MAX_REQS, WIDTH, dtype=torch.int32)
    state[0, :3] = torch.tensor([1, 2, 3])  # live
    state[1, :4] = torch.tensor([stale_id, stale_id + 1, 5, 6])  # finished
    forward = torch.zeros(MAX_REQS, WIDTH, dtype=torch.int32)
    forward[0, :3] = torch.tensor([1, 2, 3])
    forward[0, 3:6] = torch.tensor([stale_id, 4, stale_id + 2])  # stale columns
    tables.block_tables = [SimpleNamespace(gpu=state)]
    tables.input_block_tables = [forward]
    tables.block_sizes = [block_size]
    slots = torch.full((1, MAX_TOKENS), PAD_SLOT_ID, dtype=torch.int64)
    slots[0, :4] = torch.tensor(
        [block_size, stale_id * block_size, stale_id * block_size + 3, 7]
    )
    tables.slot_mappings = slots
    return tables


def _attention_group(spec, stale_id: int) -> AttentionGroup:
    group = AttentionGroup(
        backend=None,  # type: ignore[arg-type]
        layer_names=["layer.0"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        kernel_block_size=spec.block_size // KERNEL_SPLIT,
    )
    table = torch.zeros(1, MAX_REQS, WIDTH * KERNEL_SPLIT, dtype=torch.int32)
    table[0, 0, :8] = torch.arange(4, 12)  # blocks 1 and 2
    table[0, 0, 8:12] = torch.arange(4) + stale_id * KERNEL_SPLIT
    group.kernel_block_table = table
    return group


def _speculator(stale_id: int, block_size: int):
    tables = SimpleNamespace(block_sizes=[block_size])
    context = torch.full((1, MAX_TOKENS), PAD_SLOT_ID, dtype=torch.int64)
    context[0, :2] = torch.tensor([3, stale_id * block_size + 1])
    speculator = SimpleNamespace(
        draft_kv_cache_group_ids=[0],
        block_tables=tables,
        _context_slot_mappings=context,
        _kernel_context_slot_mappings=context.clone(),
        attn_groups=[],
    )
    speculator.forget_block_ids_from = (
        lambda n: DFlashSpeculator.forget_block_ids_from(speculator, n)  # type: ignore[arg-type]
    )
    return speculator


def _block_ids_named(runner) -> set[int]:
    """Every block id a kernel could be handed by the runner's tables."""
    ids: set[int] = set()
    tables = runner.block_tables
    for t in [tables.block_tables[0].gpu, tables.input_block_tables[0]]:
        ids.update(int(i) for i in t.flatten())
    block_size = tables.block_sizes[0]
    for slots in [tables.slot_mappings[0], runner.speculator._context_slot_mappings[0]]:
        ids.update(int(s) // block_size for s in slots if s >= 0)
    for groups in runner.attn_groups:
        for group in groups:
            ids.update(
                int(k) // KERNEL_SPLIT for k in group.kernel_block_table.flatten()
            )
    return ids


class ShrinkRanks:
    """Ranks of one engine on the fake driver, each an `ExtensibleKVCache`
    committed in release chunks, with a runner carrying the persistent tables
    a serving run leaves behind. The view rebind is recorded."""

    def __init__(self, monkeypatch, fake_xpu, world_size: int, chunk=CHUNK):  # noqa: F811
        self.fake_xpu = fake_xpu
        config, self.spec = _kv_cache_config(KVCacheLayout.LBNHC)
        self.layout = KVCacheLayout.LBNHC
        self.caches: list[ExtensibleKVCache] = []
        self.runners = []
        self.rebinds: list[tuple[int, int, int]] = []
        for rank in range(world_size):
            cache = ExtensibleKVCache(
                copy.copy(config),
                torch.device("xpu", rank),
                self.layout,
                release_chunk_bytes=chunk,
            )
            cache.allocate(cache.size)
            cache.commit(COMMITTED)
            full = cache.buffer.full_view()
            for offset, stride in zip(cache.segment_offsets, cache.segment_strides):
                full[offset + stride : offset + COMMITTED * stride].fill_(FILL)
            self.caches.append(cache)
            runner = SimpleNamespace(
                extensible_kv_cache=cache,
                vllm_config=SimpleNamespace(
                    attention_config=SimpleNamespace(hisparse_config=None),
                ),
                cache_config=SimpleNamespace(
                    get_resolved_kv_cache_layout=lambda: self.layout
                ),
                kv_cache_config=copy.copy(config),
                device=torch.device("xpu", rank),
                kv_caches_by_layer={},
                block_tables=_block_tables(30, self.spec.block_size),
                attn_groups=[[_attention_group(self.spec, 33)]],
                speculator=_speculator(36, self.spec.block_size),
            )
            runner.kv_cache_config.num_blocks = COMMITTED
            self.runners.append(runner)
        monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: False)

        def record_rebind(runner):
            self.rebinds.append(
                (
                    runner.device.index,
                    runner.kv_cache_config.num_blocks,
                    runner.extensible_kv_cache.num_committed_blocks,
                )
            )

        monkeypatch.setattr(
            extensible_kv_cache, "rebind_committed_views", record_rebind
        )

    def shrink_all(self, num_blocks: int) -> list[object]:
        group = MinGroup(len(self.runners))
        results: list[object] = [None] * len(self.runners)

        def run(rank: int) -> None:
            try:
                results[rank] = extensible_kv_cache.shrink_kv_cache(
                    self.runners[rank], num_blocks, reduce_min=group.reducer(rank)
                )
            except BaseException as e:  # noqa: BLE001
                results[rank] = e

        threads = [
            threading.Thread(target=run, args=(r,)) for r in range(len(self.runners))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert group.rounds == 2  # both agreements, on every rank
        return results

    def block_bytes(self, rank: int, block_id: int) -> set[int]:
        cache = self.caches[rank]
        full = cache.buffer.full_view()
        values: set[int] = set()
        for offset, stride in zip(cache.segment_offsets, cache.segment_strides):
            start = offset + block_id * stride
            values.update(full[start : start + stride].unique().tolist())
        return values

    def close(self) -> None:
        for cache in self.caches:
            cache.free()


@pytest.fixture
def shrink_ranks(monkeypatch, fake_xpu):  # noqa: F811
    made: list[ShrinkRanks] = []

    def make(world_size: int = 1, **kwargs) -> ShrinkRanks:
        ranks = ShrinkRanks(monkeypatch, fake_xpu, world_size, **kwargs)
        made.append(ranks)
        return ranks

    yield make
    for ranks in made:
        ranks.close()


def test_shrink_keeps_the_blocks_below_and_releases_whole_chunks(shrink_ranks):
    ranks = shrink_ranks()
    cache = ranks.caches[0]
    ops = ranks.fake_xpu.ops
    mapped = ops.physical_bytes
    chunks = cache.num_physical_chunks

    (released,) = ranks.shrink_all(24)

    # Blocks 24..39 are four 4-block chunks in each of the three segments.
    assert released == 3 * 16 * MIB
    assert ops.physical_bytes == mapped - released
    assert cache.num_physical_chunks == chunks - 12
    assert cache.num_committed_blocks == 24
    assert ranks.runners[0].kv_cache_config.num_blocks == 24
    assert ranks.rebinds == [(0, 24, 24)]
    for block in (1, 12, 23):
        assert ranks.block_bytes(0, block) == {FILL}
    for block in (24, 31, 39):
        assert ranks.block_bytes(0, block) == {POISON}


def test_shrink_inside_a_chunk_keeps_that_chunk(shrink_ranks):
    ranks = shrink_ranks()
    (released,) = ranks.shrink_all(26)
    # 24..27 straddles the new size and stays mapped.
    assert released == 3 * 12 * MIB
    assert ranks.block_bytes(0, 25) == {FILL}
    assert ranks.block_bytes(0, 27) == {FILL}  # not committed, still mapped
    assert ranks.block_bytes(0, 28) == {POISON}


@pytest.mark.parametrize("handled", [True, False])
def test_no_table_names_a_released_block(shrink_ranks, monkeypatch, handled):
    """Every persistent table a kernel reads block ids from is rewritten
    before the unmap. Without that (handled=False, the code before it) the
    stale ids of the tables name released blocks, which on a card faults."""
    ranks = shrink_ranks()
    runner = ranks.runners[0]
    if not handled:
        monkeypatch.setattr(
            extensible_kv_cache, "forget_block_ids_from", lambda runner, n: None
        )
    assert max(_block_ids_named(runner)) >= 24

    ranks.shrink_all(24)

    named = _block_ids_named(runner)
    poisoned = {b for b in named if POISON in ranks.block_bytes(0, b)}
    if handled:
        assert max(named) < 24
        assert poisoned == set()
        # Live entries are untouched.
        tables = runner.block_tables
        assert tables.block_tables[0].gpu[0, :3].tolist() == [1, 2, 3]
        assert tables.input_block_tables[0][0, :3].tolist() == [1, 2, 3]
        assert runner.attn_groups[0][0].kernel_block_table[0, 0, :8].tolist() == list(
            range(4, 12)
        )
        assert tables.slot_mappings[0, 3].item() == 7
        assert runner.speculator._context_slot_mappings[0, 1].item() == PAD_SLOT_ID
    else:
        assert poisoned >= {30, 31, 33, 36}


def test_torch_cap_rises_by_what_was_released(shrink_ranks, monkeypatch):
    ranks = shrink_ranks()
    cache = ranks.caches[0]
    total = 32 << 30
    fractions: list[float] = []
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (total, total)
    )
    monkeypatch.setattr(
        torch.xpu,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: fractions.append(fraction),
    )
    cache.set_torch_memory_limit(8 << 30)
    cache.anchor_torch_memory_budget()

    (released,) = ranks.shrink_all(24)

    assert cache.torch_memory_limit == (8 << 30) + released
    assert fractions[-1] == pytest.approx(((8 << 30) + released) / total)
    # Recomputed, not moved: a repeat changes nothing.
    assert ranks.shrink_all(24) == [0]
    assert cache.torch_memory_limit == (8 << 30) + released


def test_a_rank_that_cannot_start_stops_every_rank(shrink_ranks, monkeypatch):
    ranks = shrink_ranks(2)
    ranks.caches[1].unfinished_grow = (COMMITTED, 44, None)

    def finish_grow(runner, *args, **kwargs):
        raise RuntimeError("device error finishing a grow")

    monkeypatch.setattr(extensible_kv_cache, "_finish_grow", finish_grow)
    mapped = ranks.fake_xpu.ops.physical_bytes

    results = ranks.shrink_all(24)

    assert isinstance(results[0], RuntimeError)
    assert "no rank shrank" in str(results[0])
    assert isinstance(results[1], RuntimeError)
    assert [c.num_committed_blocks for c in ranks.caches] == [COMMITTED] * 2
    assert ranks.fake_xpu.ops.physical_bytes == mapped
    assert ranks.rebinds == []


@pytest.mark.parametrize("step", ["rebind", "release"])
def test_a_failure_past_the_agreement_is_finished_by_the_next_call(
    shrink_ranks, monkeypatch, step
):
    ranks = shrink_ranks(2)
    failed = ranks.caches[1]
    if step == "rebind":
        recorded = extensible_kv_cache.rebind_committed_views
        fail = {1}

        def rebind(runner):
            if runner.device.index in fail:
                fail.discard(runner.device.index)
                raise RuntimeError("device error while rebinding")
            recorded(runner)

        monkeypatch.setattr(extensible_kv_cache, "rebind_committed_views", rebind)
    else:
        release = failed.release_tail
        calls = iter(range(10))

        def release_once_failing():
            if next(calls) == 0:
                raise RuntimeError("UR_RESULT_ERROR_UNKNOWN in unmap")
            return release()

        failed.release_tail = release_once_failing  # type: ignore[method-assign]

    results = ranks.shrink_all(24)

    assert results[0] is not None and isinstance(results[0], RuntimeError)
    assert "unfinished on another rank" in str(results[0])
    assert isinstance(results[1], RuntimeError)
    assert ranks.caches[0].unfinished_shrink is None
    assert failed.unfinished_shrink == 24
    if step == "rebind":
        # The failed rank still serves its old views over mapped memory.
        assert ranks.runners[1].kv_cache_config.num_blocks == COMMITTED
        assert ranks.block_bytes(1, 30) == {FILL}

    # A grow is refused while the shrink is unfinished.
    with pytest.raises(RuntimeError, match="unfinished"):
        extensible_kv_cache.grow_kv_cache(
            ranks.runners[1],
            60,
            init_free_memory=0,
            requested_memory=0,
            reduce_min=lambda v: v,
        )

    results = ranks.shrink_all(24)

    assert results[0] == 0  # done already: nothing more to release
    assert results[1] == 3 * 16 * MIB
    assert failed.unfinished_shrink is None
    assert ranks.runners[1].kv_cache_config.num_blocks == 24
    assert ranks.block_bytes(1, 30) == {POISON}
    assert ranks.fake_xpu.ops.physical_bytes == sum(
        c.physical_bytes for c in ranks.caches
    )


def test_shrink_finishes_an_unfinished_grow_first(shrink_ranks, monkeypatch):
    ranks = shrink_ranks()
    finished: list[tuple[int, int]] = []

    def finish_grow(runner, start, end, limit, recovering=False):
        assert recovering
        finished.append((start, end))
        runner.extensible_kv_cache.unfinished_grow = None

    monkeypatch.setattr(extensible_kv_cache, "_finish_grow", finish_grow)
    ranks.caches[0].unfinished_grow = (36, COMMITTED, None)

    assert ranks.shrink_all(24) == [3 * 16 * MIB]
    assert finished == [(36, COMMITTED)]


def test_shrink_follows_the_connector_after_the_views_and_before_the_unmap(
    shrink_ranks, monkeypatch
):
    ranks = shrink_ranks()
    events: list[str] = []
    ops = ranks.fake_xpu.ops

    class Connector:
        def supports_kv_cache_resize(self) -> bool:
            return True

        def resize_kv_caches(self, kv_caches, num_blocks) -> None:
            # Its pending stores still read the tail: it must be mapped.
            assert ranks.block_bytes(0, 30) == {FILL}
            events.append(f"connector:{num_blocks}:unmaps={ops.calls.count('unmap')}")

    connector = Connector()
    monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(extensible_kv_cache, "get_kv_transfer_group", lambda: connector)
    unmaps = ops.calls.count("unmap")

    ranks.shrink_all(24)

    assert events == [f"connector:24:unmaps={unmaps}"]
    assert ops.calls.count("unmap") == unmaps + 12


def test_shrink_refuses_an_unsupported_connector_before_any_agreement(
    shrink_ranks, monkeypatch
):
    ranks = shrink_ranks()

    class Connector:
        def supports_kv_cache_resize(self) -> bool:
            return False

    monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        extensible_kv_cache, "get_kv_transfer_group", lambda: Connector()
    )
    with pytest.raises(NotImplementedError, match="resizing"):
        extensible_kv_cache.shrink_kv_cache(
            ranks.runners[0], 24, reduce_min=lambda v: pytest.fail("no agreement")
        )
    assert ranks.caches[0].num_committed_blocks == COMMITTED


def test_shrink_plan_reports_what_a_shrink_would_release(shrink_ranks):
    chunked = shrink_ranks()
    plan = extensible_kv_cache.kv_cache_shrink_plan(chunked.runners[0], 24)
    assert plan["releasable_bytes"] == 3 * 16 * MIB
    assert plan["release_chunk_bytes"] == CHUNK
    assert plan["num_committed_blocks"] == COMMITTED
    # Nothing changed.
    assert chunked.caches[0].num_committed_blocks == COMMITTED

    whole = shrink_ranks(chunk=None)
    plan = extensible_kv_cache.kv_cache_shrink_plan(whole.runners[0], 24)
    assert plan["releasable_bytes"] == 0
    assert plan["release_chunk_bytes"] is None
