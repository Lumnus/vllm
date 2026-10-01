# SPDX-License-Identifier: Apache-2.0
"""B70-0024: per-group scratch blocks for graph-padding rows of recurrent state.

The full-CUDA-graph metadata builders pad unused request rows (padding rows of
a FULL replay) with NULL_BLOCK_ID. In the unified hybrid layout every KV cache
group views the whole pool from byte 0, so block 0 is ONE page shared by every
recurrent layer and by the attention null page. Padding rows of the GDN / PLE
short-conv kernels read and roll their state in that page, each layer in its
own dtype/layout, so the page fills with non-finite bytes during a replay
(measured on B70: block-0 views 0 -> 13 non-finite within one FULL replay).

With B70_MAMBA_PAD_SCRATCH=1 (the default on XPU) the scheduler's block pool withholds one
extra block per recurrent (Mamba-spec) KV cache group, right after the null
block, so the ids are deterministic: the i-th Mamba group (in kv_cache_groups
order) owns block ``1 + i``. These blocks are never allocated, freed, hashed or
offloaded. The worker zeroes them once after allocating the KV cache and the
metadata builders point padding rows at their own group's scratch block instead
of block 0. Each scratch page is therefore written only by padding rows of one
layer, in that layer's own layout; the attention null page is left alone.
"""

import os

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

ENV = "B70_MAMBA_PAD_SCRATCH"


def enabled() -> bool:
    """B70_MAMBA_PAD_SCRATCH=1/0 forces on/off; unset = on for XPU only."""
    value = os.environ.get(ENV)
    if value is not None:
        return value == "1"
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


def mamba_group_ids(kv_cache_config: KVCacheConfig) -> list[int]:
    ids = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, MambaSpec):
            ids.append(gid)
    return ids


def pad_scratch_block_ids(kv_cache_config: KVCacheConfig) -> dict[int, int]:
    """KV cache group id -> its reserved padding-row scratch block id."""
    if not enabled():
        return {}
    return {gid: 1 + i for i, gid in enumerate(mamba_group_ids(kv_cache_config))}


def reserve_in_block_pool(block_pool, kv_cache_config: KVCacheConfig) -> list:
    """Withhold the scratch blocks from the pool's free queue (scheduler side).

    Must run right after BlockPool construction (the null block is the only
    block popped so far), so the popped ids are exactly 1..n.
    """
    scratch = pad_scratch_block_ids(kv_cache_config)
    if not scratch:
        return []
    want = sorted(scratch.values())
    if block_pool.num_gpu_blocks <= len(want) + 1:
        return []
    reserved = [block_pool.free_block_queue.popleft() for _ in want]
    got = [b.block_id for b in reserved]
    assert got == want, f"B70-0024: scratch blocks {got} != expected {want}"
    return reserved


def zero_scratch_pages(kv_caches: dict, kv_cache_config: KVCacheConfig) -> None:
    """Zero each recurrent layer's view of its group's scratch block (worker)."""
    import torch

    for gid, block_id in pad_scratch_block_ids(kv_cache_config).items():
        for name in kv_cache_config.kv_cache_groups[gid].layer_names:
            cache = kv_caches.get(name)
            if cache is None:
                continue
            tensors = cache if isinstance(cache, (list, tuple)) else [cache]
            for t in tensors:
                if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] > block_id:
                    t[block_id].zero_()
