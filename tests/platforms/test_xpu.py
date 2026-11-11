# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU device facts come from Level Zero Sysman until the runtime is initialized."""

import ctypes
import sys
import types

import pytest

pytest.importorskip("vllm_xpu_kernels")

import torch  # noqa: E402

from vllm.platforms import xpu as xpu_platform  # noqa: E402

DEVICES = [("Device A", 8 << 30), ("Device B", 16 << 30), ("Device C", 32 << 30)]


def _fake_pyzes() -> types.ModuleType:
    """A Sysman with one driver and len(DEVICES) devices, one memory module each."""

    class Core(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 64)]

    class DeviceProps(ctypes.Structure):
        _fields_ = [("stype", ctypes.c_int32), ("core", Core)]

    class MemProps(ctypes.Structure):
        _fields_ = [("stype", ctypes.c_int32), ("location", ctypes.c_int32)]

    class MemState(ctypes.Structure):
        _fields_ = [("stype", ctypes.c_int32), ("size", ctypes.c_uint64)]

    def enumerate_into(count, array, handles):
        if array is None:
            count._obj.value = len(handles)
        else:
            for i, handle in enumerate(handles):
                array[i] = handle
        return 0

    handles = list(range(1, len(DEVICES) + 1))
    zes = types.ModuleType("pyzes")
    zes.ZE_RESULT_SUCCESS = 0
    zes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 1
    zes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 2
    zes.ZES_STRUCTURE_TYPE_MEM_STATE = 3
    zes.ZES_MEM_LOC_DEVICE = 1
    zes.zes_driver_handle_t = ctypes.c_void_p
    zes.zes_device_handle_t = ctypes.c_void_p
    zes.zes_mem_handle_t = ctypes.c_void_p
    zes.zes_device_properties_t = DeviceProps
    zes.zes_mem_properties_t = MemProps
    zes.zes_mem_state_t = MemState
    zes.zesInit = lambda flags: 0
    zes.zesDriverGet = lambda count, array: enumerate_into(count, array, [1])
    zes.zesDeviceGet = lambda driver, count, array: enumerate_into(
        count, array, handles
    )
    # A device's only memory module carries the device's handle.
    zes.zesDeviceEnumMemoryModules = lambda device, count, array: enumerate_into(
        count, array, [device]
    )

    def get_properties(device, props):
        props._obj.core.name = DEVICES[device - 1][0].encode()
        return 0

    def get_mem_properties(module, props):
        props._obj.location = zes.ZES_MEM_LOC_DEVICE
        return 0

    def get_mem_state(module, state):
        state._obj.size = DEVICES[module - 1][1]
        return 0

    zes.zesDeviceGetProperties = get_properties
    zes.zesMemoryGetProperties = get_mem_properties
    zes.zesMemoryGetState = get_mem_state
    return zes


@pytest.fixture
def sysman(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyzes", _fake_pyzes())
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)


def _forbid_runtime(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the XPU runtime must not be initialized")

    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.xpu, "get_device_name", fail)
    monkeypatch.setattr(torch.xpu, "get_device_properties", fail)


def test_reads_name_and_memory_without_runtime(sysman, monkeypatch):
    _forbid_runtime(monkeypatch)
    platform = xpu_platform.XPUPlatform
    assert platform.get_device_name() == "Device A"
    assert platform.get_device_total_memory() == 8 << 30
    assert platform.get_device_name(2) == "Device C"


def test_follows_affinity_mask(sysman, monkeypatch):
    _forbid_runtime(monkeypatch)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "2,0")
    platform = xpu_platform.XPUPlatform
    assert platform.get_device_name(0) == "Device C"
    assert platform.get_device_total_memory(1) == 8 << 30


def test_falls_back_to_runtime(monkeypatch):
    monkeypatch.setattr(xpu_platform, "_sysman_device_name_and_memory", lambda i: None)
    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda i: "from runtime")
    assert xpu_platform.XPUPlatform.get_device_name() == "from runtime"


def test_uses_runtime_once_initialized(sysman, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda i: "from runtime")
    assert xpu_platform.XPUPlatform.get_device_name() == "from runtime"


def test_device_properties_use_runtime_once_initialized(monkeypatch):
    from vllm.utils import platform_utils

    props = types.SimpleNamespace(local_mem_size=131072, name="Device A")
    monkeypatch.setattr(platform_utils, "xpu_is_initialized", lambda: True)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i: props)
    assert platform_utils.xpu_get_device_properties(0, ("local_mem_size", "name")) == (
        131072,
        "Device A",
    )


def test_linear_attention_import_does_not_query_triton(monkeypatch):
    """Importing the linear attention ops must not reach the Triton driver,
    whose utilities need a stream and so initialize the XPU runtime."""
    import importlib

    import triton

    from vllm.platforms import current_platform
    from vllm.utils import platform_utils

    # Resolve the platform with the real Sysman before faking it.
    if not current_platform.is_xpu():
        pytest.skip("needs the XPU platform")
    monkeypatch.setitem(sys.modules, "pyzes", _fake_pyzes())
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    _forbid_runtime(monkeypatch)

    class NoDriver:
        @property
        def active(self):
            raise AssertionError("the Triton driver must not be queried")

    monkeypatch.setattr(triton.runtime, "driver", NoDriver())
    monkeypatch.setattr(
        platform_utils, "xpu_get_device_properties", lambda device, names: (131072,)
    )
    from vllm.third_party.flash_linear_attention.ops import utils

    try:
        utils = importlib.reload(utils)
        assert utils.device == "xpu"
        assert utils.is_intel and not utils.is_intel_alchemist
        assert utils.check_shared_mem() is True
        assert utils.check_shared_mem("hopper") is False
    finally:
        monkeypatch.undo()
        importlib.reload(utils)
