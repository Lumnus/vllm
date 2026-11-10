# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The extensible KV cache's XPU path, run on the CPU against a fake of the
vllm-xpu-kernels VMM ops (`torch.ops._xpu_C.vmm_*`) backed by host memory.

The fake follows the ops' contract: a reservation is address space only, a
mapped range reads as zero until written, unmapping drops its contents, and
every call on an unknown handle or an out-of-range device raises.
"""

import ctypes
import mmap
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.utils import extensible_tensor, vmm_driver
from vllm.utils.extensible_tensor import ExtensibleTensor, granule_size
from vllm.utils.vmm_driver import XpuVmmDriver
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import extensible_kv_cache
from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache

GRANULE = 2 << 20
NUM_DEVICES = 2


class FakeXpuVmmOps:
    """Host-memory stand-in for the `_xpu_C` VMM ops."""

    def __init__(self) -> None:
        self._reservations: dict[int, tuple[mmap.mmap, int]] = {}
        self._handles: dict[int, tuple[int, int]] = {}
        self._mapped: dict[int, tuple[int, int]] = {}
        self._next_handle = 1
        self.calls: list[str] = []
        self.fail_create = False

    def _device(self, device_index: int) -> None:
        if not 0 <= device_index < NUM_DEVICES:
            raise RuntimeError(f"Invalid XPU device index {device_index}")

    def vmm_granularity(self, device_index: int) -> int:
        self._device(device_index)
        return GRANULE

    def vmm_reserve(self, size: int) -> int:
        self.calls.append("reserve")
        if size <= 0:
            raise RuntimeError("size must be positive")
        # Over-allocate so the reservation starts on a granule, as the
        # driver's does.
        region = mmap.mmap(-1, size + GRANULE)
        raw = ctypes.addressof(ctypes.c_char.from_buffer(region))
        ptr = -(-raw // GRANULE) * GRANULE
        self._reservations[ptr] = (region, size)
        return ptr

    def vmm_free_reserved(self, ptr: int, size: int) -> None:
        self.calls.append("free_reserved")
        _, reserved = self._reservations.pop(ptr)
        assert reserved == size

    def vmm_create(self, size: int, device_index: int) -> int:
        self.calls.append("create")
        self._device(device_index)
        if self.fail_create:
            raise RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY")
        assert size % GRANULE == 0
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = (size, device_index)
        return handle

    def vmm_map(self, ptr: int, size: int, handle: int) -> None:
        self.calls.append("map")
        if handle not in self._handles:
            raise RuntimeError(f"Unknown VMM handle {handle}")
        assert self._handles[handle][0] == size
        assert ptr % GRANULE == 0
        ctypes.memset(ptr, 0, size)
        self._mapped[ptr] = (size, handle)

    def vmm_set_access(self, ptr: int, size: int, device_index: int) -> None:
        self.calls.append("set_access")
        self._device(device_index)

    def vmm_unmap(self, ptr: int, size: int) -> None:
        self.calls.append("unmap")
        mapped_size, _ = self._mapped.pop(ptr)
        assert mapped_size == size
        # Physical pages go back to the driver: their contents do not survive.
        ctypes.memset(ptr, 0xA5, size)

    def vmm_release(self, handle: int) -> None:
        self.calls.append("release")
        if self._handles.pop(handle, None) is None:
            raise RuntimeError(f"Unknown VMM handle {handle}")

    def vmm_tensor_from_ptr(
        self, ptr: int, num_bytes: int, device_index: int
    ) -> torch.Tensor:
        self._device(device_index)
        if num_bytes == 0:
            return torch.empty(0, dtype=torch.uint8)
        return torch.frombuffer(
            (ctypes.c_uint8 * num_bytes).from_address(ptr), dtype=torch.uint8
        )

    @property
    def physical_bytes(self) -> int:
        return sum(size for size, _ in self._mapped.values())


@pytest.fixture
def fake_xpu(monkeypatch):
    """XpuVmmDriver over FakeXpuVmmOps, installed where the cache finds it."""
    ops = FakeXpuVmmOps()
    package = types.ModuleType("vllm_xpu_kernels")
    module = types.ModuleType("vllm_xpu_kernels._xpu_C")
    package._xpu_C = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", package)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels._xpu_C", module)
    monkeypatch.setattr(torch.ops, "_xpu_C", ops, raising=False)
    driver = XpuVmmDriver()
    monkeypatch.setattr(extensible_tensor, "get_vmm_driver", lambda: driver)
    # Nothing is in flight on the fake device.
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args: None)
    return SimpleNamespace(ops=ops, driver=driver)


def test_driver_calls_the_xpu_c_ops(fake_xpu):
    """The driver binds `torch.ops._xpu_C`, where vllm-xpu-kernels#692
    registers the ops, and passes their errors through."""
    driver = fake_xpu.driver
    assert driver.granularity(0) == GRANULE
    assert granule_size(0) == GRANULE
    base = driver.reserve(2 * GRANULE)
    handle = driver.create(GRANULE, 1)
    driver.map(base + GRANULE, GRANULE, handle)
    driver.set_access(base + GRANULE, GRANULE, 1)
    view = driver.uint8_tensor(base + GRANULE, GRANULE, 1)
    assert view.data_ptr() == base + GRANULE
    assert view.numel() == GRANULE
    driver.unmap(base + GRANULE, GRANULE)
    driver.release(handle)
    driver.free_reserved(base, 2 * GRANULE)
    with pytest.raises(RuntimeError, match="Unknown VMM handle"):
        driver.release(handle)
    with pytest.raises(RuntimeError, match="Invalid XPU device index"):
        driver.granularity(NUM_DEVICES)


def test_driver_names_the_missing_kernels(monkeypatch):
    """Kernels without the VMM ops make the driver unusable with a reason the
    engine logs when it falls back to estimate-based sizing."""
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", types.ModuleType("k"))
    monkeypatch.setitem(
        sys.modules, "vllm_xpu_kernels._xpu_C", types.ModuleType("k._xpu_C")
    )
    monkeypatch.setattr(torch.ops, "_xpu_C", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="vllm-xpu-kernels#692"):
        XpuVmmDriver()


def test_vmm_driver_dispatches_to_xpu(monkeypatch, fake_xpu):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    vmm_driver.get_vmm_driver.cache_clear()
    try:
        assert isinstance(vmm_driver.get_vmm_driver(), XpuVmmDriver)
    finally:
        vmm_driver.get_vmm_driver.cache_clear()


@pytest.mark.parametrize("fail_create", [False, True])
def test_vmm_probe_on_xpu(monkeypatch, fake_xpu, fail_create):
    """The startup probe does one round trip through the ops, and reports
    the op's error when the device cannot back an allocation."""
    monkeypatch.setattr(vmm_driver, "get_vmm_driver", lambda: fake_xpu.driver)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    fake_xpu.ops.fail_create = fail_create
    vmm_driver.vmm_unavailable_reason.cache_clear()
    try:
        reason = vmm_driver.vmm_unavailable_reason()
    finally:
        vmm_driver.vmm_unavailable_reason.cache_clear()
    if fail_create:
        assert "OUT_OF_DEVICE_MEMORY" in reason
    else:
        assert reason is None
        assert fake_xpu.ops.physical_bytes == 0
        assert fake_xpu.ops.calls.count("reserve") == fake_xpu.ops.calls.count(
            "free_reserved"
        )


def test_extensible_tensor_on_xpu_grows_in_place(fake_xpu):
    """Growth keeps the base address and the bytes already written; new bytes
    read as zero; releasing the pages keeps the addresses (sleep/wake)."""
    et = ExtensibleTensor(4 * GRANULE, device="xpu:1", num_segments=2)
    try:
        assert et.device == torch.device("xpu", 1)
        base = et.base_ptr
        et.resize_per_segment_(GRANULE // 2, zero_new=True)
        assert fake_xpu.ops.physical_bytes == 2 * GRANULE
        seg0 = et.segment_view(0)
        seg0[:16].fill_(7)
        et.resize_per_segment_(2 * GRANULE, zero_new=True)
        assert et.base_ptr == base
        assert fake_xpu.ops.physical_bytes == 4 * GRANULE
        grown = et.segment_view(0)
        assert grown.data_ptr() == seg0.data_ptr()
        assert torch.all(grown[:16] == 7)
        assert torch.count_nonzero(grown[16:]) == 0

        et.release_physical()
        assert fake_xpu.ops.physical_bytes == 0
        et.resize_per_segment_(GRANULE, zero_new=True)
        assert et.base_ptr == base
        assert torch.count_nonzero(et.segment_view(0)) == 0
    finally:
        et.free()
    assert not fake_xpu.ops._reservations


NUM_BLOCKS = 64
NUM_LAYERS = 3


def _kv_cache_config(layout: KVCacheLayout):
    from vllm.config import ModelConfig, VllmConfig

    # 1 MiB per layer and block, so commits cross granules.
    spec = FullAttentionSpec(
        block_size=1024, num_kv_heads=4, head_size=64, dtype=torch.float16
    )
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    vllm_config.cache_config.kv_cache_layout = layout.name
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec([f"layer.{i}" for i in range(NUM_LAYERS)], spec)],
        available_memory=NUM_LAYERS * spec.page_size_bytes * NUM_BLOCKS,
    )
    assert config.num_blocks == NUM_BLOCKS
    return config, spec


@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_kv_cache_commits_on_xpu(fake_xpu, monkeypatch, layout):
    """The KV cache's growth path runs on the XPU driver: the null block first,
    then the warmup budget, then the measured size, all without moving."""
    config, spec = _kv_cache_config(layout)
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), layout)
    try:
        assert kv_cache.buffer.device == torch.device("xpu", 0)
        view = kv_cache.allocate(kv_cache.size)
        base = view.data_ptr()
        expected_segments = NUM_LAYERS if layout == KVCacheLayout.LBNHC else 1
        assert kv_cache.buffer.num_segments == expected_segments
        assert kv_cache.num_committed_blocks >= 1

        kv_cache.commit(8)
        assert kv_cache.physical_bytes == kv_cache.physical_bytes_for(8)
        assert fake_xpu.ops.physical_bytes == kv_cache.physical_bytes
        kv_cache.commit(NUM_BLOCKS)
        assert kv_cache.buffer.full_view().data_ptr() == base
        assert kv_cache.physical_bytes >= NUM_LAYERS * spec.page_size_bytes * 64
        # Shrinking remaps a smaller prefix (the warmup overshoot case).
        kv_cache.commit(16, shrink=True)
        assert kv_cache.num_committed_blocks == 16
        assert fake_xpu.ops.physical_bytes == kv_cache.physical_bytes_for(16)
    finally:
        kv_cache.free()
    assert fake_xpu.ops.physical_bytes == 0


def test_torch_memory_cap_uses_the_xpu_allocator(fake_xpu, monkeypatch):
    """The cap goes to the XPU caching allocator, not torch.cuda's, and is
    lifted when the cache is freed."""
    config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), KVCacheLayout.LBNHC)
    fractions: list[tuple[float, torch.device]] = []
    total = 32 << 30
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (total, total)
    )
    monkeypatch.setattr(
        torch.xpu,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: fractions.append((fraction, device)),
    )

    def cuda_cap(*args, **kwargs):
        raise AssertionError("the XPU path must not cap torch.cuda")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", cuda_cap)
    kv_cache.set_torch_memory_limit(8 << 30)
    kv_cache.free()
    assert fractions == [
        (0.25, torch.device("xpu", 0)),
        (1.0, torch.device("xpu", 0)),
    ]


def test_committable_blocks_on_xpu_keep_the_margin(fake_xpu, monkeypatch):
    """Warmup's commit budget on XPU reads the platform's free memory and
    keeps the margin floor and the reserved headroom free."""
    config, spec = _kv_cache_config(KVCacheLayout.LBNHC)
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), KVCacheLayout.LBNHC)
    try:
        bytes_per_block = NUM_LAYERS * spec.page_size_bytes
        floor = extensible_kv_cache.KV_CACHE_MARGIN_FLOOR_BYTES
        assert floor == 1 << 30
        free = floor + 120 * bytes_per_block
        monkeypatch.setattr(
            extensible_kv_cache,
            "device_memory_info",
            lambda device: (free, 32 << 30),
        )
        # The profiled activation peak: 100 blocks' worth, above 10% of the
        # headroom, so it is the reserve; the margin is the 1 GiB floor.
        kv_cache.reserved_headroom_bytes = 100 * bytes_per_block
        assert kv_cache.committable_blocks() == 20
    finally:
        kv_cache.free()


@pytest.mark.parametrize("requested", [None, True, False])
def test_extensible_kv_cache_resolves_on_xpu(monkeypatch, requested):
    """Unset, the extensible KV cache is on for XPU as for CUDA; an explicit
    choice is kept."""
    import vllm.platforms
    from vllm.config.vllm import VllmConfig

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda_alike=lambda: False,
            is_rocm=lambda: False,
            is_xpu=lambda: True,
        ),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            _extensible_kv_cache_resolved=False,
            enable_extensible_kv_cache=requested,
            kv_cache_memory_bytes=None,
            num_gpu_blocks_override=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(enable_elastic_ep=False),
        kv_transfer_config=None,
    )
    VllmConfig._resolve_extensible_kv_cache(config)
    assert config.cache_config.enable_extensible_kv_cache is (requested is not False)
