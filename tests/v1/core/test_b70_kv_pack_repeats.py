# SPDX-License-Identifier: Apache-2.0
# B70-K2-0001: split the packed full-attention bucket (QSA) behind
# B70_KV_SPLIT_GROUPS / B70_KV_PACK_REPEATS. CPU only: real-shaped Flash-Next
# QSA specs, shuffled insertion orders; checks partition, per-layer pairing of
# compressed-key caches with their own attention layer, and block sizes.
import random
import re
import types

import pytest
import torch

import vllm.v1.core.kv_cache_utils as U
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator as C
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        SlidingWindowSpec)
from vllm.v1.kv_cache_layout import KVCacheLayout

BF = torch.bfloat16
BS = 832
K = 7


def _items():
    gdn = C.gated_delta_net_state_shape(4, 16, 48, 128, 128, 4, K)
    ple = C.short_conv_state_shape(1, 10240, 10, K)
    items = []
    for L in range(48):
        p = f"language_model.model.layers.{L}"
        if L % 4 == 3:
            items.append((f"{p}.self_attn.attn", FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=256, head_size_v=256, dtype=BF)))
            items.append((f"{p}.self_attn.indexer.compressed_key_cache", MLAAttentionSpec(block_size=BS, num_kv_heads=1, head_size=128, dtype=BF, tokens_per_state=4)))
            items.append((f"{p}.self_attn.indexer.raw_key_cache", CircularBufferSpec(block_size=16, num_kv_heads=1, head_size=128, head_size_v=0, dtype=BF)))
        else:
            items.append((f"{p}.linear_attn", MambaSpec(shapes=tuple(gdn), dtypes=(BF, torch.float32), block_size=BS, page_size_padded=851968, mamba_cache_mode="align", num_speculative_blocks=K)))
    items.append(("language_model.model.layers.2.ple.short_conv", MambaSpec(shapes=tuple(ple), dtypes=(BF,), block_size=BS, page_size_padded=851968, mamba_cache_mode="align", num_speculative_blocks=K, tp_replicated=True)))
    for i in range(5):
        items.append((f"drafter.model.layers.{i}.self_attn.attn", SlidingWindowSpec(block_size=BS, num_kv_heads=2, head_size=128, dtype=BF, sliding_window=2048)))
    return items


def _cfg():
    cc = types.SimpleNamespace(get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLNHC, mamba_cache_mode="align")
    return types.SimpleNamespace(cache_config=cc, model_config=types.SimpleNamespace(max_model_len=262144),
                                 max_in_flight_tokens=8192,
                                 parallel_config=types.SimpleNamespace(decode_context_parallel_size=1),
                                 num_speculative_tokens=K)


def _idx(n):
    return re.search(r"layers\.(\d+)\.", n).group(1)


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(U, "_annotate_eagle_groups", lambda *a, **k: None)
    monkeypatch.setattr(U, "_warn_if_unannotated_eagle_mamba", lambda *a, **k: None)
    monkeypatch.setattr(U, "_uses_trailing_mtp_layers", lambda *a, **k: False)
    monkeypatch.delenv("B70_KV_PACK_REPEATS", raising=False)
    monkeypatch.delenv("B70_KV_SPLIT_GROUPS", raising=False)


# env -> (groups, bytes per block); unset = split (default on)
CASES = [
    ({}, 66, 905_216),  # default ON
    ({"B70_KV_SPLIT_GROUPS": "0"}, 7, 10_862_592),  # opt-out = upstream packing
    ({"B70_KV_SPLIT_GROUPS": "1"}, 66, 905_216),
    ({"B70_KV_PACK_REPEATS": "1"}, 66, 905_216),
    ({"B70_KV_PACK_REPEATS": "3"}, 23, 2_715_648),
    ({"B70_KV_SPLIT_GROUPS": "1", "B70_KV_PACK_REPEATS": "3"}, 23, 2_715_648),
]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("env,groups,bpb", CASES)
def test_pack_repeats(monkeypatch, seed, env, groups, bpb):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    it = _items()
    random.seed(seed)
    random.shuffle(it)
    sp = dict(it)
    g = U._get_packed_kv_cache_groups(_cfg(), sp)
    for x in g:
        cmp_ = [_idx(n) for n in x.layer_names if n.endswith("compressed_key_cache")]
        att = [_idx(n) for n in x.layer_names if n.endswith("self_attn.attn") and not n.startswith("drafter")]
        assert sorted(cmp_) == sorted(att), x.layer_names
    assert sorted(n for x in g for n in x.layer_names) == sorted(sp)
    assert len(g) == groups
    assert U._get_kv_cache_bytes_per_block(g, KVCacheLayout.BLNHC) == bpb
