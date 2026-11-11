# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU platform answers device questions without initializing XPU."""

import pytest

pytest.importorskip("vllm_xpu_kernels")

from vllm.platforms import xpu as xpu_platform  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_cache():
    xpu_platform._device_name_and_memory.cache_clear()
    yield
    xpu_platform._device_name_and_memory.cache_clear()


def test_name_and_memory_come_from_the_properties_helper(monkeypatch):
    calls = []

    def properties(device, names, init_xpu=False):
        calls.append((device, names, init_xpu))
        return ("Device A", 32 << 30)

    monkeypatch.setattr(xpu_platform, "xpu_get_device_properties", properties)
    platform = xpu_platform.XPUPlatform
    assert platform.get_device_name() == "Device A"
    assert platform.get_device_total_memory() == 32 << 30
    # One read answers both, and it does not ask for an initialization.
    assert calls == [(0, ("name", "total_memory"), False)]


def test_falls_back_to_the_runtime_when_the_child_fails(monkeypatch):
    calls = []

    def properties(device, names, init_xpu=False):
        calls.append(init_xpu)
        if not init_xpu:
            raise RuntimeError("no child")
        return ("from runtime", 1)

    monkeypatch.setattr(xpu_platform, "xpu_get_device_properties", properties)
    assert xpu_platform.XPUPlatform.get_device_name(1) == "from runtime"
    assert calls == [False, True]
