# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU expert-parallel sentinel kernel: ids as ``torch.where`` maps them,
and a zeroed ``rows_per_expert``."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


@pytest.mark.parametrize("num_tokens", [1, 7, 300, 4096])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("num_local_experts", [4, 64, 300])
@pytest.mark.parametrize("sentinel", [0, 255])
def test_sentinel_ids_and_counts(num_tokens, top_k, num_local_experts, sentinel):
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        sentinel_ids_and_counts,
    )

    ids = torch.randint(-1, 256, (num_tokens, top_k), dtype=torch.int32, device="xpu")
    out, counts = sentinel_ids_and_counts(ids, sentinel, num_local_experts)
    torch.testing.assert_close(out, torch.where(ids < 0, sentinel, ids), rtol=0, atol=0)
    assert out.shape == ids.shape and out.dtype == ids.dtype
    assert counts.dtype == torch.int32 and counts.shape == (num_local_experts,)
    assert int(counts.abs().sum()) == 0
