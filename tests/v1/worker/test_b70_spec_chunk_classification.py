# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 batch order and row classification: spec verify rows beside a continuing chunk.

vllm#55894 (V1 runner): the drafter's builder lowers the runner's
reorder_batch_threshold to 1, so 1+k-token verify rows can sit behind a
continuing prefill chunk, and a POSITIONAL split_decodes_and_prefills then runs
them as prefills. These tests pin how Model Runner V2 orders and classifies the
same batch shape (MTP k=3, decode_query_len 4): ≥2 verify streams, a continuing
3K chunk, a fresh prefill, a plain 1-token decode.
"""

import inspect

import numpy as np
import torch

from tests.v1.worker.test_gpu_batch_ordering import _make_common_attn_metadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids
from vllm.v1.worker.gpu.model_states import mamba_hybrid

K = 3
DQL = 1 + K
DRAFT = [-1] * K  # async scheduling places placeholder drafts


def _batch(num_verify: int) -> tuple[dict[str, int], dict[str, list[int]]]:
    # Scheduler dict order is arbitrary; put the chunk first on purpose.
    tokens = {"chunk": 1024 - DQL * num_verify, "fresh": 300, "dec1": 1}
    tokens |= {f"v{i}": DQL for i in range(num_verify)}
    drafts = {f"v{i}": list(DRAFT) for i in range(num_verify)}
    return tokens, drafts


def _spec_mask(query_lens: list[int], drafts_per_req: list[int]) -> np.ndarray:
    """The expression mamba_hybrid.prepare_attn feeds GDN and PLE (pinned below)."""
    num_scheduled = np.array(query_lens)
    num_draft = np.array(drafts_per_req)
    is_decode = num_scheduled == num_draft + 1
    return (num_draft > 0) & is_decode


def test_mask_expression_is_the_runner_expression():
    src = inspect.getsource(mamba_hybrid.MambaHybridModelState.prepare_attn)
    assert "num_draft_tokens_per_req + 1" in src
    assert "(num_draft_tokens_per_req > 0) & is_decode" in src


def test_v2_orders_verify_rows_before_a_continuing_chunk():
    for n in (2, 3, 8):
        tokens, drafts = _batch(n)
        order = sort_batch_req_ids(tokens, drafts, DQL)
        assert order[:n] == [f"v{i}" for i in range(n)], order
        assert order[n:] == ["dec1", "fresh", "chunk"], order


def test_positional_split_after_v2_order_keeps_verify_rows_as_decodes():
    # QSA side-cache builder: threshold 1+k, require_uniform (qsa_cache.py).
    for n in (2, 3):
        tokens, drafts = _batch(n)
        order = sort_batch_req_ids(tokens, drafts, DQL)
        q = [tokens[r] for r in order]
        nd, npf, ndt, _ = split_decodes_and_prefills(
            _make_common_attn_metadata(q), decode_threshold=DQL, require_uniform=True
        )
        assert (nd, ndt) == (n, n * DQL)
        assert npf == 3


def test_mask_split_is_position_independent():
    # GDN / PLE builders classify by the spec mask, not by position.
    for n in (2, 3):
        tokens, drafts = _batch(n)
        for order in (
            sort_batch_req_ids(tokens, drafts, DQL),
            ["chunk", *[f"v{i}" for i in range(n)], "fresh", "dec1"],  # V1-style
        ):
            q = [tokens[r] for r in order]
            d = [len(drafts.get(r, [])) for r in order]
            mask = _spec_mask(q, d)
            assert [r for r, m in zip(order, mask) if m] == [
                r for r in order if r.startswith("v")
            ]


def test_negative_control_v1_misorder_breaks_a_positional_split():
    # The #55894 shape: verify rows behind a continuing chunk. A positional
    # split at 1+k then counts every verify row as a prefill.
    tokens, _ = _batch(3)
    order = ["dec1", "chunk", "v0", "v1", "v2", "fresh"]
    q = [tokens[r] for r in order]
    nd, npf, _, _ = split_decodes_and_prefills(
        _make_common_attn_metadata(q), decode_threshold=DQL
    )
    assert (nd, npf) == (1, 5)
