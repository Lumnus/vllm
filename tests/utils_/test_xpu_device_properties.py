# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""xpu_get_device_properties reads directly when XPU is initialized, and in a
small child process otherwise."""

import subprocess
import sys
import types

import pytest
import torch

from vllm.utils import platform_utils


def test_reads_directly_once_initialized(monkeypatch):
    props = types.SimpleNamespace(name="Device A", local_mem_size=131072)
    monkeypatch.setattr(platform_utils, "xpu_is_initialized", lambda: True)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: props)
    monkeypatch.setattr(subprocess, "run", pytest.fail)
    assert platform_utils.xpu_get_device_properties(0, ("name", "local_mem_size")) == (
        "Device A",
        131072,
    )


def test_reads_in_a_small_child_when_not_initialized(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        out = 'a library warning\n["Device A", 34359738368]\n'
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    def fail(device):
        raise AssertionError("the XPU runtime must not be initialized here")

    monkeypatch.setattr(platform_utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(torch.xpu, "get_device_properties", fail)
    monkeypatch.setattr(subprocess, "run", run)
    assert platform_utils.xpu_get_device_properties(1, ("name", "total_memory")) == (
        "Device A",
        34359738368,
    )

    argv = seen["argv"]
    assert argv[:2] == [sys.executable, "-c"]
    assert argv[3:] == ["1", "name", "total_memory"]
    # The child must stay small: torch only, not vLLM, not the parent's main.
    assert "import json, sys, torch" in argv[2]
    assert "vllm" not in argv[2]
    assert seen["kwargs"]["timeout"] == platform_utils._XPU_PROPERTIES_TIMEOUT_S
    assert seen["kwargs"]["check"] is True


def test_child_script_reads_the_named_properties(monkeypatch, capsys):
    props = types.SimpleNamespace(name="Device A", local_mem_size=131072)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: props)
    monkeypatch.setattr(sys, "argv", ["-c", "0", "name", "local_mem_size"])
    exec(platform_utils._XPU_PROPERTIES_SCRIPT, {})
    assert capsys.readouterr().out.strip() == '["Device A", 131072]'


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(cmd="python", timeout=1),
        subprocess.CalledProcessError(returncode=1, cmd="python"),
    ],
)
def test_child_failure_raises(monkeypatch, error):
    def run(argv, **kwargs):
        raise error

    monkeypatch.setattr(platform_utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(type(error)):
        platform_utils.xpu_get_device_properties(0, ("name",))
