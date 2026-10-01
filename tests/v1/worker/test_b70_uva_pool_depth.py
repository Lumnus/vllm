# SPDX-License-Identifier: Apache-2.0
"""B70-0025: UVA pool depth on XPU with spec decoding, and round-robin reuse."""

import numpy as np
import pytest

import vllm.v1.worker.gpu.buffer_utils as bu


def test_depth_policy(monkeypatch):
    monkeypatch.delenv("B70_UVA_POOL_DEPTH", raising=False)
    assert bu.b70_uva_pool_depth(2, is_xpu=True, uses_spec_decode=True) == 8
    assert bu.b70_uva_pool_depth(2, is_xpu=True, uses_spec_decode=False) == 2
    assert bu.b70_uva_pool_depth(2, is_xpu=False, uses_spec_decode=True) == 2
    assert bu.b70_uva_pool_depth(12, is_xpu=True, uses_spec_decode=True) == 12
    monkeypatch.setenv("B70_UVA_POOL_DEPTH", "0")
    assert bu.b70_uva_pool_depth(2, is_xpu=True, uses_spec_decode=True) == 2
    monkeypatch.setenv("B70_UVA_POOL_DEPTH", "16")
    assert bu.b70_uva_pool_depth(2, is_xpu=True, uses_spec_decode=True) == 16


class _HostBuf:
    """Stand-in for UvaBuffer on a host-only box: the 'device view' IS the
    host memory, exactly like UVA, so a later host write is visible to an
    earlier, not-yet-consumed reader."""

    def __init__(self, size, dtype):
        import torch

        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()

    def uva(self, n=None):
        return self.cpu[:n] if n is not None else self.cpu


@pytest.mark.parametrize("depth,writes_before_consume,clobbered", [
    (2, 2, True),   # stock depth: two stagings later the first view is overwritten
    (8, 2, False),  # B70-0025 depth: still intact
    (8, 7, False),
    (8, 8, True),
])
def test_round_robin_reuse(monkeypatch, depth, writes_before_consume, clobbered):
    import torch

    monkeypatch.setattr(bu, "is_uva_available", lambda: True)
    monkeypatch.setattr(bu, "UvaBuffer", _HostBuf)
    pool = bu.UvaBufferPool(4, torch.int32, max_concurrency=depth)
    first = pool.copy_to_uva(np.array([1, 1, 1, 1], dtype=np.int32))
    for i in range(writes_before_consume):
        pool.copy_to_uva(np.array([i + 2] * 4, dtype=np.int32))
    # The "late kernel" of the first step reads its view now.
    assert (first.tolist() != [1, 1, 1, 1]) == clobbered
