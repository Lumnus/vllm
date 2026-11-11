# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU platform answers device questions without initializing XPU."""

import pytest

pytest.importorskip("vllm_xpu_kernels")

import torch  # noqa: E402

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


def test_linear_attention_import_does_not_query_triton(monkeypatch):
    """Importing the linear attention ops must not reach the Triton driver,
    whose utilities need a stream and so initialize the XPU runtime."""
    import importlib

    import triton

    from vllm.platforms import current_platform
    from vllm.utils import platform_utils

    if not current_platform.is_xpu():
        pytest.skip("needs the XPU platform")

    class NoDriver:
        @property
        def active(self):
            raise AssertionError("the Triton driver must not be queried")

    def fail(*args, **kwargs):
        raise AssertionError("the XPU runtime must not be initialized")

    def properties(device, names, init_xpu=False):
        assert not init_xpu
        return {
            ("name", "total_memory"): ("Intel(R) Arc(TM) Pro B70 Graphics", 1),
            ("local_mem_size",): (131072,),
        }[tuple(names)]

    monkeypatch.setattr(triton.runtime, "driver", NoDriver())
    monkeypatch.setattr(torch.xpu, "get_device_name", fail)
    monkeypatch.setattr(torch.xpu, "get_device_properties", fail)
    monkeypatch.setattr(platform_utils, "xpu_get_device_properties", properties)
    monkeypatch.setattr(xpu_platform, "xpu_get_device_properties", properties)
    from vllm.third_party.flash_linear_attention.ops import utils

    try:
        utils = importlib.reload(utils)
        assert utils.device == "xpu"
        assert utils.is_intel and not utils.is_intel_alchemist
        assert utils.check_shared_mem() is True
        assert utils.check_shared_mem("hopper") is False

        def no_answer(device, names, init_xpu=False):
            raise RuntimeError("no child")

        monkeypatch.setattr(utils, "xpu_get_device_properties", no_answer)
        assert utils.check_shared_mem("ada") is False
    finally:
        monkeypatch.undo()
        importlib.reload(utils)
