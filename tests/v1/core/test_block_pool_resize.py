# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Growing the KV cache block pool while requests run."""

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import ModelRunnerOutput

BLOCK_SIZE = 16


def _pool(num_blocks: int) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )


def test_grown_blocks_are_allocated_after_the_existing_ones():
    pool = _pool(4)
    assert pool.get_num_free_blocks() == 3  # block 0 is the null block
    held = pool.get_new_blocks(2)
    assert [b.block_id for b in held] == [1, 2]

    pool.grow(8)

    assert pool.num_gpu_blocks == 8
    assert len(pool.blocks) == 8
    assert [b.block_id for b in pool.blocks] == list(range(8))
    assert pool.get_num_free_blocks() == 1 + 4
    # Blocks already handed out keep their identity and state.
    assert [pool.blocks[b.block_id] for b in held] == held
    assert all(b.ref_cnt == 1 for b in held)
    # The free block that was free before the grow goes first, then the new ones.
    new = pool.get_new_blocks(5)
    assert [b.block_id for b in new] == [3, 4, 5, 6, 7]
    assert pool.get_num_free_blocks() == 0
    with pytest.raises(ValueError):
        pool.get_new_blocks(1)


def test_freed_blocks_and_grown_blocks_share_one_queue():
    pool = _pool(4)
    blocks = pool.get_new_blocks(3)
    pool.grow(6)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 5
    allocated = {b.block_id for b in pool.get_new_blocks(5)}
    assert allocated == {1, 2, 3, 4, 5}


def test_hash_map_finds_blocks_beyond_the_old_size():
    """The cached-block map holds the pool's block list, so ids that exist only
    after a grow resolve through it."""
    pool = _pool(3)
    pool.grow(6)
    grown = pool.blocks[5]
    key = make_block_hash_with_group_id(b"h" * 32, 0)
    pool.cached_block_hash_to_block.insert(key, grown)
    assert pool.cached_block_hash_to_block.get_one_block(key) is grown
    assert pool.cached_block_hash_to_block.contain(key, 5)
    assert pool.cached_block_hash_to_block.pop(key, 5) is grown


def test_usage_follows_the_new_size():
    pool = _pool(5)
    pool.get_new_blocks(4)  # all four usable blocks
    assert pool.get_usage() == pytest.approx(1.0)
    pool.grow(9)
    assert pool.get_usage() == pytest.approx(1.0 - 4 / 8)


def test_prefix_cache_reset_sees_the_grown_pool():
    pool = _pool(3)
    pool.grow(6)
    assert pool.reset_prefix_cache()
    assert pool.get_num_free_blocks() == 5


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


def test_manager_grow_recomputes_watermark_and_config():
    config = _kv_cache_config(11)
    manager = KVCacheManager(
        config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        watermark=0.1,
    )
    assert manager.watermark_blocks == 1
    page = config.kv_cache_tensors[0].size // 11

    manager.grow(31)

    assert manager.watermark_blocks == 3
    assert config.num_blocks == 31
    assert config.kv_cache_tensors[0].size == 31 * page
    assert manager.block_pool.num_gpu_blocks == 31
    assert manager.block_pool.get_num_free_blocks() == 30
    assert manager.usage == pytest.approx(0.0)


def test_scheduler_uses_grown_blocks_with_requests_running():
    """A request that does not fit waits; once the pool grows it runs on ids the
    first request never held, and the first request's blocks are unchanged."""
    scheduler = create_scheduler(
        max_num_batched_tokens=200,
        block_size=BLOCK_SIZE,
        num_blocks=7,  # 6 usable blocks
        enable_prefix_caching=False,
    )
    first, second = create_requests(
        num_requests=2, num_tokens=80, block_size=BLOCK_SIZE
    )
    scheduler.add_request(first)
    out0 = scheduler.schedule()
    first_blocks = out0.scheduled_new_reqs[0].block_ids[0]
    assert len(first_blocks) == 5

    scheduler.add_request(second)
    out1 = scheduler.schedule()
    assert not out1.scheduled_new_reqs  # one free block, five needed
    assert len(scheduler.waiting) == 1

    scheduler.grow_kv_cache(13)

    assert scheduler.cache_config.num_gpu_blocks == 13
    out2 = scheduler.schedule()
    (new_req,) = out2.scheduled_new_reqs
    second_blocks = new_req.block_ids[0]
    assert len(second_blocks) == 5
    assert not set(second_blocks) & set(first_blocks)
    assert max(second_blocks) >= 7  # at least one block from the grown range
    assert [r.request_id for r in scheduler.running] == [
        first.request_id,
        second.request_id,
    ]
    # The first request goes on decoding on its own blocks.
    scheduler.update_from_output(
        out0,
        ModelRunnerOutput(
            req_ids=[first.request_id],
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert scheduler.kv_cache_manager.get_block_ids(first.request_id)[0][:5] == (
        first_blocks
    )


def test_scheduler_stats_report_the_pool_size():
    scheduler = create_scheduler(num_blocks=11, block_size=BLOCK_SIZE)
    scheduler.log_stats = True
    assert scheduler.make_stats().num_gpu_blocks == 11
    scheduler.grow_kv_cache(21)
    assert scheduler.make_stats().num_gpu_blocks == 21
