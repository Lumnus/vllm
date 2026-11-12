# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A host tower in an engine whose platform is an accelerator.

vLLM registers its custom ops under the dispatch key of the current platform
(`direct_register_custom_op`). A tower that computes on host tensors inside
such an engine must not reach an op that has no kernel for the host. These
tests register the ops under a foreign key (`tests/multimodal/
foreign_dispatch_key_plugin.py`) in a child process, as the XPU platform
does, and run the tower there.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.multimodal import foreign_dispatch_key_plugin as plugin

pytestmark = pytest.mark.cpu_test

REPO = Path(__file__).resolve().parents[2]
FOREIGN_KEY = "XPU"
MODEL = "Qwen/Qwen3.5-0.8B"

# Child-process scripts: the plugin is imported first, as `-p` would.
PREMISE = """
import tests.multimodal.foreign_dispatch_key_plugin
import torch
q = torch.randn(1, 6, 2, 8)
cu = torch.tensor([0, 3, 6])
import vllm.v1.attention.ops.vit_attn_wrappers
try:
    torch.ops.vllm.torch_sdpa_wrapper(q, q, q, None, cu, False)
except NotImplementedError as e:
    assert "'CPU' backend" in str(e), str(e)
    print("PREMISE-HOLDS")
"""

ATTENTION = """
import tests.multimodal.foreign_dispatch_key_plugin
from types import SimpleNamespace
from functools import partial
import torch
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.v1.attention.ops.vit_attn_wrappers import torch_sdpa_wrapper

heads, dim = 2, 8
self = SimpleNamespace(
    num_heads=heads, num_kv_heads=heads, head_size=dim, scale=dim**-0.5
)
self.view_qkv_to_4d = partial(MMEncoderAttention.view_qkv_to_4d, self)
self._forward_sdpa = partial(MMEncoderAttention._forward_sdpa, self)
q, k, v = (torch.randn(1, 6, heads, dim) for _ in range(3))
cu = torch.tensor([0, 2, 6])
out = MMEncoderAttention.forward_cpu(self, q, k, v, cu)
ref = torch_sdpa_wrapper(q, k, v, dim**-0.5, cu)
assert torch.equal(out, ref)
print("ATTENTION-ON-HOST-OK")
"""


def _child(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = {**os.environ, plugin.ENV: FOREIGN_KEY, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_the_foreign_key_leaves_the_registered_op_without_a_host_kernel():
    # The stand-in is only worth its name while it reproduces the failure.
    done = _child("-c", PREMISE)
    assert "PREMISE-HOLDS" in done.stdout, done.stdout + done.stderr


def test_attention_computes_on_host_tensors_under_the_foreign_key():
    done = _child("-c", ATTENTION)
    assert "ATTENTION-ON-HOST-OK" in done.stdout, done.stdout + done.stderr


def test_a_real_tower_encodes_under_the_foreign_key():
    """The tower of the checkpoint suite, end to end, with every op of its
    forward registered under the foreign key."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} is not in the local cache")

    done = _child(
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "tests.multimodal.foreign_dispatch_key_plugin",
        "tests/multimodal/test_host_tower_checkpoint.py",
    )
    out = done.stdout + done.stderr
    assert done.returncode == 0, out[-6000:]
    assert "3 passed" in out, out[-2000:]


@pytest.mark.parametrize(
    ("forward", "on_host"),
    [
        ("forward_native", False),
        ("forward_cuda", False),
        ("forward_xpu", False),
        ("forward_cpu", True),
    ],
)
def test_only_the_cpu_forward_takes_the_host_entry(monkeypatch, forward, on_host):
    """With no host tower in play the accelerator forwards reach the
    registered op, exactly as before; the CPU forward alone takes the host
    entry."""
    from functools import partial
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.layers.attention import mm_encoder_attention as mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    calls = []
    monkeypatch.setattr(
        mod, "vit_torch_sdpa_wrapper", lambda q, **kw: calls.append("op") or q
    )
    monkeypatch.setattr(
        mod, "vit_torch_sdpa_wrapper_on_host", lambda q, **kw: calls.append("host") or q
    )
    self = SimpleNamespace(
        num_heads=2,
        num_kv_heads=2,
        head_size=8,
        scale=1.0,
        fp8_enabled=False,
        is_flash_attn_backend=False,
        attn_backend=AttentionBackendEnum.TORCH_SDPA,
    )
    self.view_qkv_to_4d = partial(mod.MMEncoderAttention.view_qkv_to_4d, self)
    self._forward_sdpa = partial(mod.MMEncoderAttention._forward_sdpa, self)

    x = torch.zeros(1, 4, 2, 8)
    getattr(mod.MMEncoderAttention, forward)(self, x, x, x)

    assert calls == ["host" if on_host else "op"]
