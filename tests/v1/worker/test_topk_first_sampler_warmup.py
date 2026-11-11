# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The top-k-first sampler's warm-up compiles the specialisations serving uses.

Needs an accelerator with compiled (not interpreted) Triton; kept apart from
the CPU tests of test_topk_first_sampler.py, which import the sampler under the
Triton interpreter. Run it in its own pytest process."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.sample import topk_first_sampler as sm


def _num_compiled(fn) -> int:
    return sum(len(c[0]) for c in fn.device_caches.values())


@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="needs an accelerator for Triton"
)
def test_warmup_covers_serving_specialisations():
    """After warmup(), calls with the dtypes the model runner passes (int64
    request indices, int32 draft tokens, bf16 or fp32 logits, both KP values)
    compile nothing new."""
    kernels = (
        sm._split_topk_kernel,
        sm._merge_kernel,
        sm._fast_reject_kernel,
        sm._fast_sample_kernel,
    )
    if not all(hasattr(k, "device_caches") for k in kernels):
        pytest.skip("Triton runs interpreted here (no per-device kernel cache)")
    dev = torch.accelerator.current_accelerator()
    v, steps, nc = 3 * sm.BLOCK + 17, 7, 16
    sm.warmup(dev, v, steps, nc, logits_dtype=torch.bfloat16)
    before = [_num_compiled(k) for k in kernels]

    r = 3
    n = r * (steps + 1)
    states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.tensor([0.7, 0.0, 1.0], device=dev)),
        top_k=SimpleNamespace(
            gpu=torch.tensor([20, 1, 5], dtype=torch.int32, device=dev)
        ),
        top_p=SimpleNamespace(gpu=torch.tensor([0.9, 1.0, 0.95], device=dev)),
        seeds=SimpleNamespace(gpu=torch.arange(r, dtype=torch.int64, device=dev)),
    )
    idx_mapping = torch.arange(r, dtype=torch.int64, device=dev)
    eim = idx_mapping.repeat_interleave(steps + 1)
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    cu = torch.arange(0, n + 1, steps + 1, dtype=torch.int32, device=dev)
    draft = torch.randn(r, steps, v, device=dev)
    cand = torch.randint(0, v, (r, steps, nc), dtype=torch.int64, device=dev)
    ds = torch.randint(0, v, (n,), dtype=torch.int32, device=dev)
    for dtype in (torch.bfloat16, torch.float32):
        logits = torch.randn(n, v, device=dev).to(dtype)
        for kmax in (20, 1):
            sm.fast_reject(
                logits, draft, cand, ds, pos, cu, idx_mapping, eim, states, steps, kmax
            )
            sm.fast_sample(logits, eim, states, pos, kmax)
    torch.accelerator.synchronize()
    assert [_num_compiled(k) for k in kernels] == before
