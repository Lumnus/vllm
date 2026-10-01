# SPDX-License-Identifier: Apache-2.0
"""B70-0024: FULL-graph padding rows must not touch block 0 or a live block.

CPU-only. The staging-kernel case runs on the Triton interpreter
(TRITON_INTERPRET=1) when no GPU is present.
"""

import types

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.v1.core import b70_pad_scratch
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, MambaSpec

BS = 832
K = 3


def _hybrid_groups():
    fa = FullAttentionSpec(
        block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    page = fa.page_size_bytes
    gdn = MambaSpec(
        block_size=BS,
        shapes=((page // 4,),),
        dtypes=(torch.bfloat16,),
        page_size_padded=page,
        mamba_cache_mode="align",
        num_speculative_blocks=K,
    )
    ple = MambaSpec(
        block_size=BS,
        shapes=((page // 8,),),
        dtypes=(torch.float32,),
        page_size_padded=page,
        mamba_cache_mode="align",
        num_speculative_blocks=K,
    )
    spec = {}
    # Qwen3.8-Flash-Next shape, shortened: 3 x GDN then 1 x full attention.
    for i in range(16):
        spec[f"l.{i}"] = fa if (i + 1) % 4 == 0 else gdn
        if i == 1:
            spec["l.1.ple"] = ple
    spec["draft.16"] = fa
    return ku._get_kv_cache_groups_uniform_page_size(spec)


def _manager(num_blocks=400):
    groups = _hybrid_groups()
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    mgr = KVCacheManager(
        cfg,
        max_model_len=16384,
        scheduler_block_size=BS,
        hash_block_size=BS,
        enable_caching=False,
        use_eagle=True,
        num_prefill_lookahead=1,
    )
    return cfg, mgr


def _allocate_until_full(mgr):
    seen = set()
    reqs = create_requests(64, num_tokens=1700, block_size=BS, max_tokens=64)
    for r in reqs:
        if mgr.allocate_slots(r, 1700, num_lookahead_tokens=K) is None:
            break
        for ids in mgr.get_block_ids(r.request_id):
            seen.update(ids)
    return seen


def test_scratch_blocks_are_reserved_and_never_allocated(monkeypatch):
    monkeypatch.setenv(b70_pad_scratch.ENV, "1")
    cfg, mgr = _manager()
    scratch = b70_pad_scratch.pad_scratch_block_ids(cfg)
    mamba = b70_pad_scratch.mamba_group_ids(cfg)
    assert len(mamba) == 13  # 12 GDN + 1 PLE
    assert sorted(scratch.values()) == list(range(1, len(mamba) + 1))
    assert [b.block_id for b in mgr.coordinator.b70_pad_scratch_blocks] == sorted(
        scratch.values()
    )
    seen = _allocate_until_full(mgr)
    assert seen, "nothing allocated"
    assert not (seen & set(scratch.values())), "a scratch block was handed out"


def test_without_reservation_low_blocks_are_allocatable(monkeypatch):
    # Control: with the switch off the same ids are ordinary blocks.
    monkeypatch.setenv(b70_pad_scratch.ENV, "0")
    _, mgr = _manager()
    seen = _allocate_until_full(mgr)
    assert set(range(1, 14)) & seen


@pytest.mark.parametrize("pad", [None, 5])
def test_uniform_spec_staging_padding_rows(pad):
    """3 real spec-decode requests padded to 4 rows / 16 tokens (FULL replay)."""
    from vllm.v1.attention.backends.gdn_spec_metadata import (
        try_build_uniform_spec_metadata,
    )

    maxbs, k = 16, K + 1
    b = types.SimpleNamespace(
        use_spec_decode=True,
        use_full_cuda_graph=True,
        num_spec=K,
        decode_cudagraph_max_bs=64,
        vllm_config=types.SimpleNamespace(
            cache_config=types.SimpleNamespace(mamba_cache_mode="align")
        ),
        kv_cache_spec=types.SimpleNamespace(num_speculative_blocks=K, block_size=BS),
        spec_state_indices_tensor=torch.full((maxbs, k), 777, dtype=torch.int32),
        spec_sequence_masks=torch.ones(maxbs, dtype=torch.bool),
        spec_token_indx=torch.zeros(maxbs * k, dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        spec_query_start_loc=torch.zeros(maxbs + 1, dtype=torch.int32),
        num_accepted_tokens=torch.zeros(maxbs, dtype=torch.int32),
    )
    if pad is not None:
        b.b70_pad_state_index = pad
    bt = torch.zeros((4, 8), dtype=torch.int32)
    bt[0, :7] = torch.tensor([0, 0, 0, 101, 102, 103, 104])
    bt[1, :7] = torch.tensor([0, 0, 0, 201, 202, 203, 204])
    bt[2, :7] = torch.tensor([0, 0, 0, 301, 302, 303, 304])
    qsl = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
    m = types.SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=16,
        block_table_tensor=bt,
        seq_lens=torch.tensor([3004, 3004, 3004, 0], dtype=torch.int32),
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
    )
    md = try_build_uniform_spec_metadata(
        b,
        m,
        torch.tensor([2, 4, 1, 1], dtype=torch.int32),
        torch.tensor([K, K, K, -1], dtype=torch.int32),
    )
    assert md is not None and md.num_spec_decodes == 3
    idx = md.spec_state_indices_tensor
    assert idx[:3].tolist() == [
        [101, 102, 103, 104],
        [201, 202, 203, 204],
        [301, 302, 303, 304],
    ]
    live = {int(x) for x in idx[:3].flatten()}
    want = 0 if pad is None else pad
    assert idx[3].tolist() == [want] * k
    if pad is not None:
        # The B70-0024 contract: no padding-row write targets block 0 or a live block.
        assert 0 not in idx[3].tolist()
        assert not (set(idx[3].tolist()) & live)


def test_zero_scratch_pages(monkeypatch):
    monkeypatch.setenv(b70_pad_scratch.ENV, "1")
    cfg, _ = _manager()
    scratch = b70_pad_scratch.pad_scratch_block_ids(cfg)
    caches = {}
    for g in cfg.kv_cache_groups:
        for name in g.layer_names:
            caches[name] = [torch.ones(40, 3)]
    b70_pad_scratch.zero_scratch_pages(caches, cfg)
    for gid, g in enumerate(cfg.kv_cache_groups):
        for name in g.layer_names:
            t = caches[name][0]
            if gid in scratch:
                assert t[scratch[gid]].abs().sum() == 0
                assert t.sum() == t.numel() - t.shape[1]
            else:
                assert t.sum() == t.numel()
