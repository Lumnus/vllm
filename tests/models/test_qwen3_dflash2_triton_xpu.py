# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 Triton kernels (grouped convolution, candidate edge scoring) on XPU.

The device gate is checked directly. The kernels themselves run here under the
Triton interpreter on CPU (a subprocess with TRITON_INTERPRET=1) against the
torch references, at the served drafter's shapes; that checks the kernel
programs, not the XPU backend's code generation."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_dflash2


@pytest.mark.parametrize(
    "is_cuda,is_xpu,switch,expected",
    [
        (True, False, False, True),
        (False, True, True, True),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_triton_gate(monkeypatch, is_cuda, is_xpu, switch, expected):
    monkeypatch.setattr(qwen3_dflash2, "XPU_DFLASH2_TRITON", switch)
    tensor = SimpleNamespace(is_cuda=is_cuda, is_xpu=is_xpu)
    assert qwen3_dflash2._use_triton(tensor) is expected


def test_cpu_tensors_take_the_reference():
    assert not qwen3_dflash2._use_triton(torch.zeros(1))


def test_switch_defaults_on(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_XPU_DFLASH2_TRITON", raising=False)
    assert envs.VLLM_XPU_DFLASH2_TRITON is True
    monkeypatch.setenv("VLLM_XPU_DFLASH2_TRITON", "0")
    assert envs.VLLM_XPU_DFLASH2_TRITON is False


def test_kernels_match_reference_under_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    result = subprocess.run(
        # -P: the test directory must not shadow installed packages.
        [sys.executable, "-P", __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "interpreter checks passed" in result.stdout


def _grouped_conv_reference(hidden, delta, base, block_size, num_groups, group_size):
    """fp32 sequential reference of the grouped convolution."""
    taps = base.shape[0]
    batch = hidden.shape[0] // block_size
    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]
    return expected.flatten(0, 1).flatten(-2).to(hidden.dtype)


def _interpreter_checks() -> None:
    torch.manual_seed(0)
    # Served drafter: hidden 2560, conv_group_size 16, query block 1 + 7
    # speculative tokens; plus an odd shape. One tap only: the interpreter
    # rejects the later taps' mask (a vector mask and-ed with the program's
    # scalar `position >= tap`: "mask has unsupported dtype"), which compiled
    # Triton accepts; the served kernel_size 2 runs on the card only.
    for dtype in (torch.float32, torch.bfloat16):
        for batch, num_groups, group_size, block_size, taps in (
            (2, 160, 16, 8, 1),
            (3, 13, 11, 5, 1),
        ):
            rows = batch * block_size
            hidden = torch.randn(rows, num_groups * group_size, dtype=dtype)
            base = torch.randn(taps, num_groups * group_size, dtype=dtype)
            # A strided view, as DFlashGroupedConv.prepare passes it.
            delta = torch.randn(rows, 2, taps, num_groups, dtype=dtype)[:, 1]
            actual = qwen3_dflash2.dflash2_grouped_conv_impl(
                hidden, delta, base, block_size, group_size
            )
            expected = _grouped_conv_reference(
                hidden, delta, base, block_size, num_groups, group_size
            )
            tol = 1e-2 if dtype is torch.bfloat16 else 1e-5
            torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    # Served selector: rank 256, top-k 16, 7 steps; anchors past the vocabulary
    # in a strided view. FP32 codebooks only: the interpreter's tl.dot returns
    # garbage for BF16 operands (even at 1 x 3 x 5), so the served BF16 dot runs
    # on the card only.
    vocab = 257
    for batch, steps, top_k, rank in ((2, 7, 16, 256), (3, 5, 17, 65)):
        predecessors = torch.randn(vocab, rank)
        successors = torch.randn_like(predecessors)
        candidates = torch.randint(vocab, (batch, steps, top_k))
        unary = torch.randn(batch, steps, top_k, dtype=torch.bfloat16)
        hidden = torch.randn(batch, steps, rank)
        anchors = torch.randint(2 * vocab, (batch * 8,), dtype=torch.int64)[::8]
        args = (predecessors, successors, candidates, unary, hidden, anchors)
        actual = qwen3_dflash2.dflash2_score_edges_impl(*args)
        expected = qwen3_dflash2._score_edges_reference(*args, top_k)
        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015625)
    print("interpreter checks passed")


if __name__ == "__main__":
    _interpreter_checks()
