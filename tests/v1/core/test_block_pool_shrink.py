# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shrinking the KV cache block pool while requests run: the fence on the
tail of the pool, the drain, and the cut."""

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.request import RequestStatus

BLOCK_SIZE = 16


def _pool(num_blocks: int) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )


def _key(name: bytes):
    return make_block_hash_with_group_id(BlockHash(name * 32), 0)


def _cache(pool: BlockPool, block_id: int, name: bytes) -> None:
    """Make block_id a prefix-cache entry under `name`."""
    pool._insert_block_hash(_key(name), pool.blocks[block_id], num_tokens=None)


def _ids_in_map(pool: BlockPool) -> set[int]:
    cache = pool.cached_block_hash_to_block
    ids = set(cache._cache.values())
    for dup in cache._dup_ids.values():
        ids.update(dup)
    return ids


def test_free_tail_blocks_leave_the_queue_at_once():
    pool = _pool(10)
    held = pool.get_new_blocks(3)
    assert [b.block_id for b in held] == [1, 2, 3]
    assert pool.get_num_free_blocks() == 6

    pool.begin_shrink(6)

    # 6..9 were free: retired at once. 4 and 5 stay free.
    assert pool.get_num_free_blocks() == 2
    assert pool.num_fenced_blocks == 4
    assert pool.num_retiring_blocks == 0
    assert pool.shrink_drained()
    # Usage is relative to the pool the fence leaves: 3 of 5 usable blocks.
    assert pool.get_usage() == pytest.approx(3 / 5)
    assert [b.block_id for b in pool.get_new_blocks(2)] == [4, 5]
    with pytest.raises(ValueError):
        pool.get_new_blocks(1)

    pool.finish_shrink()

    assert pool.num_gpu_blocks == 6
    assert [b.block_id for b in pool.blocks] == list(range(6))
    assert pool.retire_from is None
    assert pool.get_usage() == pytest.approx(1.0)


def test_tail_blocks_in_use_retire_as_their_holders_free_them():
    pool = _pool(8)
    blocks = pool.get_new_blocks(7)  # ids 1..7
    pool.begin_shrink(4)
    assert pool.num_retiring_blocks == 4
    assert not pool.shrink_drained()
    assert pool.get_num_free_blocks() == 0
    assert pool.fenced_block_holders() == {4, 5, 6, 7}

    # One holder frees a head and two tail blocks: the head block is free
    # again, the tail ones retire.
    pool.free_blocks([blocks[1], blocks[4], blocks[5]])  # ids 2, 5, 6
    assert pool.get_num_free_blocks() == 1
    assert pool.num_retiring_blocks == 2
    # In use: 1, 3, 4 and 7, of the 5 blocks the fence leaves usable.
    assert pool.get_usage() == pytest.approx(4 / 5)
    assert [b.block_id for b in pool.get_new_blocks(1)] == [2]

    pool.free_blocks([blocks[3], blocks[6]])  # ids 4, 7
    assert pool.shrink_drained()
    pool.finish_shrink()
    assert pool.num_gpu_blocks == 4
    assert pool.get_num_free_blocks() == 0


def test_finish_before_the_drain_is_refused():
    pool = _pool(6)
    pool.get_new_blocks(5)
    pool.begin_shrink(3)
    with pytest.raises(RuntimeError, match="still in use"):
        pool.finish_shrink()
    assert pool.num_gpu_blocks == 6


def test_cached_tail_blocks_are_dropped_and_never_hit():
    pool = _pool(10)
    head, tail = pool.get_new_blocks(8)[2], pool.blocks[8]  # head id 3
    _cache(pool, head.block_id, b"h")
    # Free and cached, in use and cached.
    pool.free_blocks([pool.blocks[7]])
    _cache(pool, 7, b"f")
    _cache(pool, 8, b"u")
    assert tail.ref_cnt == 1

    pool.begin_shrink(6)

    assert pool.get_cached_block(BlockHash(b"f" * 32), [0]) is None
    assert _ids_in_map(pool) == {3}
    assert pool.blocks[7].block_hash is None
    assert pool.blocks[8].block_hash is None
    # The running holder of block 8 caches it again during the drain: no
    # other request may find it.
    _cache(pool, 8, b"u")
    assert 8 in _ids_in_map(pool)
    assert pool.get_cached_block(BlockHash(b"u" * 32), [0]) is None
    assert pool.get_cached_block(BlockHash(b"h" * 32), [0]) == [head]
    # It retires without its entry.
    pool.free_blocks([pool.blocks[6], pool.blocks[8]])
    assert _ids_in_map(pool) == {3}
    assert pool.shrink_drained()
    pool.finish_shrink()
    assert max(_ids_in_map(pool)) < 6


def test_abort_restores_the_pool_exactly():
    pool = _pool(10)
    blocks = pool.get_new_blocks(7)  # ids 1..7
    pool.free_blocks([blocks[0]])  # id 1 free again
    free_before = pool.get_num_free_blocks()
    usage_before = pool.get_usage()
    pool.begin_shrink(5)
    pool.free_blocks([blocks[5]])  # id 6 retires
    assert pool.get_num_free_blocks() == free_before - 2  # 8, 9 retired

    pool.abort_shrink()

    assert pool.retire_from is None
    assert pool.num_gpu_blocks == 10
    # The retired ids are free again: 1, 6, 8, 9.
    assert pool.get_num_free_blocks() == free_before + 1
    assert pool.get_usage() == pytest.approx(usage_before - 1 / 9)
    assert sorted(b.block_id for b in pool.get_new_blocks(4)) == [1, 6, 8, 9]
    # Blocks still held stay with their holders.
    assert all(pool.blocks[i].ref_cnt == 1 for i in (2, 3, 4, 5, 7))


def test_one_resize_at_a_time_and_bounds():
    pool = _pool(10)
    with pytest.raises(ValueError):
        pool.begin_shrink(10)
    with pytest.raises(ValueError):
        pool.begin_shrink(1)  # the null block must stay
    pool.begin_shrink(5)
    with pytest.raises(RuntimeError, match="already shrinking"):
        pool.begin_shrink(4)
    with pytest.raises(RuntimeError, match="while it is shrinking"):
        pool.grow(12)


def test_grow_after_shrink_and_shrink_after_grow():
    pool = _pool(6)
    pool.grow(10)
    pool.begin_shrink(6)
    pool.finish_shrink()
    assert pool.num_gpu_blocks == 6
    pool.grow(9)
    assert [b.block_id for b in pool.blocks] == list(range(9))
    assert pool.get_num_free_blocks() == 8
    assert sorted(b.block_id for b in pool.get_new_blocks(8)) == list(range(1, 9))


def test_prefix_cache_reset_during_a_fence():
    pool = _pool(8)
    pool.begin_shrink(4)
    assert pool.reset_prefix_cache()
    blocks = pool.get_new_blocks(1)
    assert not pool.reset_prefix_cache()
    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache()


def _kv_cache_config(num_blocks: int) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                layers=["layer"],
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def test_manager_watermark_and_config_through_a_shrink():
    config = _kv_cache_config(31)
    manager = KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        watermark=0.1,
    )
    page = config.kv_cache_tensors[0].size // 31
    assert manager.watermark_blocks == 3

    manager.begin_shrink(11)
    assert manager.watermark_blocks == 1
    assert config.num_blocks == 31
    manager.abort_shrink()
    assert manager.watermark_blocks == 3

    manager.begin_shrink(11)
    manager.finish_shrink()
    assert config.num_blocks == 11
    assert config.kv_cache_tensors[0].size == 11 * page
    assert manager.block_pool.num_gpu_blocks == 11
    assert manager.watermark_blocks == 1


def test_manager_floor_follows_max_model_len_and_the_watermark():
    from types import SimpleNamespace

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=160),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
    )

    def manager(watermark: float) -> KVCacheManager:
        return KVCacheManager(
            _kv_cache_config(64),
            max_model_len=160,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            watermark=watermark,
        )

    # 160 tokens are 10 blocks of 16, beside the null block.
    assert manager(0.0).min_num_blocks(vllm_config) == 11
    # With a 10 % watermark the pool needs one more block to keep it free.
    assert manager(0.1).min_num_blocks(vllm_config) == 12


def _finish(scheduler, request) -> None:
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)


def test_scheduler_admits_on_head_blocks_and_finishes_when_drained():
    scheduler = create_scheduler(
        max_num_batched_tokens=400,
        block_size=BLOCK_SIZE,
        num_blocks=13,
        max_model_len=96,
        enable_prefix_caching=True,
    )
    first, second = create_requests(
        num_requests=2, num_tokens=80, block_size=BLOCK_SIZE
    )
    scheduler.add_request(first)
    out = scheduler.schedule()
    assert out.scheduled_new_reqs[0].block_ids[0] == [1, 2, 3, 4, 5]

    # 96 tokens need 6 blocks beside the null block: 7 is the floor.
    with pytest.raises(ValueError, match="max_model_len"):
        scheduler.begin_kv_cache_shrink(6)
    scheduler.begin_kv_cache_shrink(7)
    assert scheduler.kv_cache_shrink_progress() == (6, 0)

    scheduler.add_request(second)
    out = scheduler.schedule()
    assert not out.scheduled_new_reqs  # one head block free, five needed

    _finish(scheduler, first)
    out = scheduler.schedule()
    (new_req,) = out.scheduled_new_reqs
    assert max(new_req.block_ids[0]) < 7

    scheduler.finish_kv_cache_shrink()
    assert scheduler.cache_config.num_gpu_blocks == 7
    assert scheduler.kv_cache_config.num_blocks == 7
    assert scheduler.kv_cache_manager.block_pool.num_gpu_blocks == 7


def test_scheduler_preempts_the_holders_of_fenced_blocks():
    scheduler = create_scheduler(
        max_num_batched_tokens=400,
        block_size=BLOCK_SIZE,
        num_blocks=20,
        max_model_len=96,
    )
    first, second = create_requests(
        num_requests=2, num_tokens=80, block_size=BLOCK_SIZE
    )
    scheduler.add_request(first)
    scheduler.add_request(second)
    out = scheduler.schedule()
    blocks = {r.req_id: r.block_ids[0] for r in out.scheduled_new_reqs}
    assert blocks[second.request_id] == [6, 7, 8, 9, 10]

    scheduler.begin_kv_cache_shrink(8)
    assert scheduler.kv_cache_shrink_progress() == (12, 3)

    assert scheduler.preempt_kv_cache_shrink_holders() == 1

    assert scheduler.kv_cache_shrink_progress() == (12, 0)
    assert second.status == RequestStatus.PREEMPTED
    assert [r.request_id for r in scheduler.running] == [first.request_id]
    scheduler.finish_kv_cache_shrink()
    assert scheduler.kv_cache_manager.block_pool.num_gpu_blocks == 8


def test_scheduler_abort_restores_admission():
    scheduler = create_scheduler(
        max_num_batched_tokens=400,
        block_size=BLOCK_SIZE,
        num_blocks=13,
        max_model_len=96,
    )
    first, second = create_requests(
        num_requests=2, num_tokens=80, block_size=BLOCK_SIZE
    )
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.begin_kv_cache_shrink(7)
    scheduler.add_request(second)
    assert not scheduler.schedule().scheduled_new_reqs

    scheduler.abort_kv_cache_shrink()

    (new_req,) = scheduler.schedule().scheduled_new_reqs
    assert len(new_req.block_ids[0]) == 5
    assert scheduler.kv_cache_manager.block_pool.num_gpu_blocks == 13
