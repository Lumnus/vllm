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


def test_name_and_memory_come_from_one_cached_read(monkeypatch):
    calls = []

    def facts(device):
        calls.append(device)
        return ("Device A", 32 << 30, 131072)

    monkeypatch.setattr(xpu_platform, "xpu_device_facts", facts)
    platform = xpu_platform.XPUPlatform
    assert platform.get_device_name() == "Device A"
    assert platform.get_device_total_memory() == 32 << 30
    assert calls == [0]


def test_initializes_in_process_only_when_no_child_can_start(monkeypatch):
    calls = []

    def facts(device):
        raise OSError("cannot start a process")

    def properties(device, names, init_xpu=False):
        calls.append((names, init_xpu))
        return ("from runtime", 1)

    monkeypatch.setattr(xpu_platform, "xpu_device_facts", facts)
    monkeypatch.setattr(xpu_platform, "xpu_get_device_properties", properties)
    assert xpu_platform.XPUPlatform.get_device_name(1) == "from runtime"
    assert calls == [(("name", "total_memory"), True)]


def test_does_not_initialize_in_process_after_a_failed_child(monkeypatch):
    from vllm.utils.platform_utils import XPUPropertiesChildError

    def facts(device):
        raise XPUPropertiesChildError("no answer in 120 s")

    def fail(*args, **kwargs):
        raise AssertionError("the XPU runtime must not be initialized")

    monkeypatch.setattr(xpu_platform, "xpu_device_facts", facts)
    monkeypatch.setattr(xpu_platform, "xpu_get_device_properties", fail)
    with pytest.raises(XPUPropertiesChildError):
        xpu_platform.XPUPlatform.get_device_total_memory()


def test_a_real_child_timeout_does_not_reach_the_in_process_path(monkeypatch):
    """End to end through the helper: subprocess times out, nothing initializes."""
    import subprocess

    from vllm.utils import platform_utils

    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    def fail(*args, **kwargs):
        raise AssertionError("the XPU runtime must not be initialized")

    platform_utils._xpu_device_facts.clear()
    monkeypatch.setattr(platform_utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(torch.xpu, "get_device_properties", fail)
    try:
        with pytest.raises(platform_utils.XPUPropertiesChildError):
            xpu_platform.XPUPlatform.get_device_name()
    finally:
        platform_utils._xpu_device_facts.clear()


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

    def facts(device):
        return ("Intel(R) Arc(TM) Pro B70 Graphics", 1, 131072)

    monkeypatch.setattr(triton.runtime, "driver", NoDriver())
    monkeypatch.setattr(torch.xpu, "get_device_name", fail)
    monkeypatch.setattr(torch.xpu, "get_device_properties", fail)
    monkeypatch.setattr(platform_utils, "xpu_device_facts", facts)
    monkeypatch.setattr(xpu_platform, "xpu_device_facts", facts)
    from vllm.third_party.flash_linear_attention.ops import utils

    try:
        utils = importlib.reload(utils)
        assert utils.device == "xpu"
        assert utils.is_intel and not utils.is_intel_alchemist
        assert utils.check_shared_mem() is True
        assert utils.check_shared_mem("hopper") is False

        def no_answer(device):
            raise RuntimeError("no child")

        warned = []
        monkeypatch.setattr(utils, "xpu_device_facts", no_answer)
        monkeypatch.setattr(
            utils.logger, "warning", lambda *args, **kwargs: warned.append(args[0])
        )
        assert utils.check_shared_mem("ada") is False
        assert len(warned) == 1 and "small autotune block sizes" in warned[0]
    finally:
        monkeypatch.undo()
        importlib.reload(utils)


def test_set_device_selects_before_runtime_init(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch._C, "_xpu_setDevice", lambda i: calls.append(("select", i))
    )
    monkeypatch.setattr(torch.xpu, "set_device", lambda d: calls.append(("set", d)))
    device = torch.device("xpu", 2)
    xpu_platform.XPUPlatform.set_device(device)
    assert calls == [("select", 2), ("set", device)]

    calls.clear()
    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: True)
    xpu_platform.XPUPlatform.set_device(device)
    assert calls == [("set", device)]

    # A torch whose selector is missing or fails keeps today's behaviour.
    calls.clear()
    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: False)
    monkeypatch.delattr(torch._C, "_xpu_setDevice")
    xpu_platform.XPUPlatform.set_device(device)
    assert calls == [("set", device)]

    def broken(index):
        raise RuntimeError("no such device")

    calls.clear()
    monkeypatch.setattr(torch._C, "_xpu_setDevice", broken, raising=False)
    xpu_platform.XPUPlatform.set_device(device)
    assert calls == [("set", device)]
