# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The extensible KV cache's XPU path, run on the CPU against a fake of the
vllm-xpu-kernels VMM ops (`torch.ops._xpu_C.vmm_*`) backed by host memory.

The fake follows the ops' contract: a reservation is address space only, a
mapped range reads as zero until written, unmapping drops its contents, and
every call on an unknown handle or an out-of-range device raises.
"""

import copy
import ctypes
import importlib.util
import mmap
import sys
import threading
import types
from functools import partial
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
        # Devices whose creates fail, the rest of the ranks' being fine.
        self.fail_create_devices: set[int] = set()

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
        if self.fail_create or device_index in self.fail_create_devices:
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
        # Pin the margin so the expectation does not depend on the platform
        # the module was imported under.
        floor = 1 << 30
        monkeypatch.setattr(extensible_kv_cache, "KV_CACHE_MARGIN_FLOOR_BYTES", floor)
        monkeypatch.setattr(extensible_kv_cache, "KV_CACHE_MARGIN_FRACTION", 0.75)
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


def _import_with_platform(monkeypatch, *, rocm: bool, xpu: bool):
    """A fresh copy of the sizing module, imported as `current_platform` says."""
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: rocm, is_xpu=lambda: xpu),
    )
    spec = importlib.util.spec_from_file_location(
        "_extensible_kv_cache_probe", extensible_kv_cache.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# platform: (margin floor in GiB, margin fraction)
PLATFORM_MARGINS = {
    "cuda": (1, 0.75),
    "rocm": (3, 0.75),
    "xpu": (0, 0.0),
}


@pytest.mark.parametrize("platform", list(PLATFORM_MARGINS))
def test_margin_constants_follow_the_platform(monkeypatch, platform):
    module = _import_with_platform(
        monkeypatch, rocm=platform == "rocm", xpu=platform == "xpu"
    )
    floor_gib, fraction = PLATFORM_MARGINS[platform]
    assert floor_gib << 30 == module.KV_CACHE_MARGIN_FLOOR_BYTES
    assert fraction == module.KV_CACHE_MARGIN_FRACTION
    # The keyword defaults of the sizing function are bound to the same values.
    defaults = module.measure_kv_cache_bytes.__kwdefaults__
    assert defaults["margin_floor_bytes"] == floor_gib << 30
    assert defaults["margin_fraction"] == fraction
    # Not platform dependent for XPU: the non-torch reserve and the warmup share.
    reserve_bytes = (2 if platform == "rocm" else 1) << 29
    assert reserve_bytes == module.KV_CACHE_NON_TORCH_RESERVE_BYTES
    assert module.KV_CACHE_WARMUP_RESERVE_FRACTION == 0.1


@pytest.mark.parametrize("platform", list(PLATFORM_MARGINS))
def test_measure_kv_cache_bytes_at_the_card_log_numbers(monkeypatch, platform):
    """Rank 1 of a TP4 hybrid-MoE boot on four Arc Pro B70 cards: 2.96 GiB
    free, 4.3 GiB committed, 1.11 GiB transient peak, 28.18 GiB requested,
    28.64 GiB free at init."""
    module = _import_with_platform(
        monkeypatch, rocm=platform == "rocm", xpu=platform == "xpu"
    )
    gib = 1 << 30
    free, committed, transient = int(2.96 * gib), int(4.3 * gib), int(1.11 * gib)
    requested, init_free = int(28.18 * gib), int(28.64 * gib)
    result = module.measure_kv_cache_bytes(
        init_free_memory=init_free,
        free_memory=free,
        committed_bytes=committed,
        requested_memory=requested,
        transient_peak_bytes=transient,
    )
    # available = min(requested - non_kv_used, free + committed - margin)
    #             - transient_peak, margin = max(floor, fraction * transient).
    floor_gib, fraction = PLATFORM_MARGINS[platform]
    non_kv_used = init_free - free - committed
    margin = max(floor_gib << 30, int(transient * fraction))
    expected = min(requested - non_kv_used, free + committed - margin) - transient
    assert result == expected
    # The device headroom binds on CUDA and ROCm (5.14 GiB in the card log for
    # CUDA's margin); the budget binds on XPU, where no margin is held back.
    if platform == "xpu":
        assert result == requested - non_kv_used - transient
    else:
        assert result == free + committed - margin - transient


def test_xpu_margin_gives_the_larger_cache(monkeypatch):
    gib = 1 << 30
    kwargs = dict(
        init_free_memory=int(28.64 * gib),
        free_memory=int(2.96 * gib),
        committed_bytes=int(4.3 * gib),
        requested_memory=int(28.18 * gib),
        transient_peak_bytes=int(1.11 * gib),
    )
    sizes = {
        name: _import_with_platform(
            monkeypatch, rocm=name == "rocm", xpu=name == "xpu"
        ).measure_kv_cache_bytes(**kwargs)
        for name in PLATFORM_MARGINS
    }
    assert sizes["xpu"] > sizes["cuda"] > sizes["rocm"]
    # The transient peak is still kept free in full on XPU.
    assert sizes["xpu"] <= int(28.18 * gib) - kwargs["transient_peak_bytes"]


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


def test_backing_segments_maps_memory_without_committing(fake_xpu):
    """`back_segments_` maps the physical pages a growth needs and leaves the
    committed prefixes alone; the resize that follows maps nothing more."""
    et = ExtensibleTensor(4 * GRANULE, device="xpu:0", num_segments=2)
    try:
        et.resize_per_segment_(GRANULE // 2, zero_new=True)
        et.segment_view(0)[:16].fill_(7)
        creates = fake_xpu.ops.calls.count("create")

        et.back_segments_([2 * GRANULE] * 2)
        assert fake_xpu.ops.physical_bytes == 4 * GRANULE
        assert et.bytes_per_segment == GRANULE // 2
        assert et.segment_view(0).numel() == GRANULE // 2
        backed = fake_xpu.ops.calls.count("create")
        assert backed > creates

        et.resize_segments_([2 * GRANULE] * 2, zero_new=True)
        assert fake_xpu.ops.calls.count("create") == backed
        assert et.bytes_per_segment == 2 * GRANULE
        grown = et.segment_view(0)
        assert torch.all(grown[:16] == 7)
        assert torch.count_nonzero(grown[16:]) == 0
        with pytest.raises(ValueError, match="grow-only"):
            et.back_segments_([GRANULE] * 2)
    finally:
        et.free()


# Growing the committed prefix while serving.

MIB = 1 << 20


class MinGroup:
    """`reduce_min` for ranks running in threads of one process."""

    def __init__(self, world_size: int) -> None:
        self._barrier = threading.Barrier(world_size, timeout=30)
        self._lock = threading.Lock()
        self._values: dict[int, list[int]] = {}
        self.rounds = 0

    def reducer(self, rank: int):
        calls = iter(range(10**6))

        def reduce_min(value: int) -> int:
            call = next(calls)
            with self._lock:
                self._values.setdefault(call, []).append(value)
                self.rounds = max(self.rounds, call + 1)
            self._barrier.wait()
            return min(self._values[call])

        return reduce_min


class GrowFixture:
    """Ranks of one engine on the fake driver: each an `ExtensibleKVCache` on
    its own device with a stand-in runner, a platform that reports each
    device's free memory, and the seams of the grow path recorded."""

    def __init__(self, monkeypatch, fake_xpu, free_memory: list[int], committed: int):
        self.fake_xpu = fake_xpu
        self.free_memory = free_memory
        self.rebinds: list[tuple[int, int]] = []
        self.torch_fractions: list[float] = []
        config, self.spec = _kv_cache_config(KVCacheLayout.LBNHC)
        self.config = config
        self.layout = KVCacheLayout.LBNHC
        self.caches: list[ExtensibleKVCache] = []
        self.runners = []
        self.initial_physical: list[int] = []
        for rank in range(len(free_memory)):
            cache = ExtensibleKVCache(
                copy.copy(config), torch.device("xpu", rank), self.layout
            )
            cache.allocate(cache.size)
            cache.commit(committed)
            self.caches.append(cache)
            runner = SimpleNamespace(
                extensible_kv_cache=cache,
                vllm_config=SimpleNamespace(
                    attention_config=SimpleNamespace(hisparse_config=None),
                    model_config=SimpleNamespace(multimodal_config=None),
                    parallel_config=SimpleNamespace(),
                ),
                cache_config=SimpleNamespace(
                    get_resolved_kv_cache_layout=lambda: self.layout
                ),
                kv_cache_config=copy.copy(config),
                device=torch.device("xpu", rank),
                kv_caches_by_layer={},
            )
            runner.kv_cache_config.num_blocks = committed
            self.runners.append(runner)
            self.initial_physical.append(cache.physical_bytes)
        # Free memory falls by what is mapped after the setup.
        monkeypatch.setattr(
            extensible_kv_cache,
            "current_platform",
            SimpleNamespace(
                free_device_memory=lambda device: (
                    self.free_memory[device.index]
                    - (
                        self.caches[device.index].physical_bytes
                        - self.initial_physical[device.index]
                    )
                )
            ),
        )
        # The margin the platform keeps free is the sizing tests' concern.
        monkeypatch.setattr(
            extensible_kv_cache,
            "measure_kv_cache_bytes",
            partial(
                extensible_kv_cache.measure_kv_cache_bytes,
                margin_floor_bytes=0,
                margin_fraction=0.0,
            ),
        )
        monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: False)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
        # What torch reports to a measurement while serving.
        self.torch_stats: dict[str, int] = {}
        self.torch_reserved = 0
        monkeypatch.setattr(
            torch.accelerator, "memory_stats", lambda device=None: self.torch_stats
        )
        monkeypatch.setattr(
            torch.accelerator,
            "memory_reserved",
            lambda device=None: self.torch_reserved,
        )

        def record_rebind(runner):
            self.rebinds.append(
                (
                    runner.kv_cache_config.num_blocks,
                    runner.extensible_kv_cache.num_committed_blocks,
                )
            )

        monkeypatch.setattr(
            extensible_kv_cache, "rebind_committed_views", record_rebind
        )

    def grow(self, rank: int, target: int, reduce_min=lambda value: value) -> int:
        return extensible_kv_cache.grow_kv_cache(
            self.runners[rank],
            target,
            init_free_memory=1024 * MIB,
            requested_memory=1024 * MIB,
            reduce_min=reduce_min,
        )

    def grow_all(self, target: int) -> list[int]:
        group = MinGroup(len(self.runners))
        results: list[int | BaseException] = [0] * len(self.runners)

        def run(rank: int) -> None:
            try:
                results[rank] = self.grow(rank, target, group.reducer(rank))
            except BaseException as e:  # noqa: BLE001
                results[rank] = e

        threads = [
            threading.Thread(target=run, args=(r,)) for r in range(len(self.runners))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results  # type: ignore[return-value]

    def close(self) -> None:
        for cache in self.caches:
            cache.free()


def _free_for(committed_blocks: int, kv_cache: ExtensibleKVCache, fits: int) -> int:
    """Free device memory at which `fits` blocks fit in total, given
    `committed_blocks` committed, with 1 GiB at init, all of it requested and
    no transient peak or margin."""
    # available = min(requested - non_kv_used, free + committed), and
    # non_kv_used = init_free - free - committed, so both are free + committed.
    return kv_cache.physical_bytes_for(fits) - kv_cache.physical_bytes_for(
        committed_blocks
    )


@pytest.fixture
def grow_ranks(monkeypatch, fake_xpu):
    made: list[GrowFixture] = []

    def make(free_blocks: list[int], committed: int = 8) -> GrowFixture:
        # Sizes that fit exactly `free_blocks[r]` blocks on rank r.
        probe_config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
        probe = ExtensibleKVCache(
            probe_config, torch.device("xpu", 0), KVCacheLayout.LBNHC
        )
        free = [_free_for(committed, probe, blocks) for blocks in free_blocks]
        probe.free()
        fixture = GrowFixture(monkeypatch, fake_xpu, free, committed)
        made.append(fixture)
        return fixture

    yield make
    for fixture in made:
        fixture.close()


def test_grow_commits_what_the_measurement_fits(grow_ranks):
    ranks = grow_ranks([40])
    cache = ranks.caches[0]
    base = cache.buffer.base_ptr
    # Memory mapped by an earlier attempt holds garbage the grow must not expose.
    cache.back(24)
    full = cache.buffer.full_view()
    for offset, stride in zip(cache.segment_offsets, cache.segment_strides):
        full[offset + 8 * stride : offset + 24 * stride].fill_(0xFF)
    cache.buffer.segment_view(0)[:64].fill_(7)

    assert ranks.grow(0, 24) == 24

    assert cache.num_committed_blocks == 24
    assert cache.buffer.base_ptr == base
    view = cache.buffer.segment_view(0)
    assert view.numel() == 24 * ranks.spec.page_size_bytes
    assert torch.all(view[:64] == 7)
    assert torch.count_nonzero(view[8 * ranks.spec.page_size_bytes :]) == 0
    assert ranks.rebinds == [(24, 24)]
    assert ranks.fake_xpu.ops.physical_bytes == cache.physical_bytes_for(24)


def test_grow_stops_at_what_fits_and_at_capacity(grow_ranks):
    ranks = grow_ranks([20])
    assert ranks.grow(0, 50) == 20
    assert ranks.caches[0].num_committed_blocks == 20
    assert ranks.grow(0, 500) == 20  # nothing more fits

    ranks = grow_ranks([10_000])
    assert ranks.grow(0, 500) == NUM_BLOCKS


def test_grow_to_the_current_size_or_less_changes_nothing(grow_ranks):
    ranks = grow_ranks([40])
    creates = ranks.fake_xpu.ops.calls.count("create")
    reduce_calls: list[int] = []

    def reduce_min(value: int) -> int:
        reduce_calls.append(value)
        return value

    assert ranks.grow(0, 8, reduce_min) == 8
    assert ranks.grow(0, 3, reduce_min) == 8
    assert ranks.caches[0].num_committed_blocks == 8
    assert ranks.fake_xpu.ops.calls.count("create") == creates
    assert ranks.rebinds == []
    # Both agreements run on every call, so that no rank can leave one out.
    assert reduce_calls == [8, 1, 3, 1]


def test_grow_measures_inside_the_call_and_takes_the_minimum(grow_ranks):
    """Ranks with different free memory agree on the smaller count."""
    ranks = grow_ranks([40, 20])
    assert ranks.grow_all(50) == [20, 20]
    assert [c.num_committed_blocks for c in ranks.caches] == [20, 20]
    assert ranks.rebinds == [(20, 20), (20, 20)]


def test_failed_map_on_one_rank_leaves_every_rank_at_the_old_size(grow_ranks):
    ranks = grow_ranks([40, 40])
    ops = ranks.fake_xpu.ops
    # The second rank's device cannot back an allocation; the first can.
    ops.fail_create_devices = {1}

    assert ranks.grow_all(24) == [8, 8]

    assert [c.num_committed_blocks for c in ranks.caches] == [8, 8]
    assert [r.kv_cache_config.num_blocks for r in ranks.runners] == [8, 8]
    assert ranks.rebinds == []
    # Committed bytes did not move; granules mapped by the attempt stay mapped.
    for cache in ranks.caches:
        assert cache.buffer.segment_view(0).numel() == 8 * ranks.spec.page_size_bytes
        assert cache.physical_bytes >= cache.physical_bytes_for(8)

    # The next attempt reuses them and every rank reaches the new size.
    ops.fail_create_devices = set()
    assert ranks.grow_all(24) == [24, 24]
    assert [c.num_committed_blocks for c in ranks.caches] == [24, 24]
    assert ranks.fake_xpu.ops.physical_bytes == sum(
        c.physical_bytes_for(24) for c in ranks.caches
    )


def test_grow_lowers_the_torch_cap_by_the_bytes_it_commits(grow_ranks, monkeypatch):
    ranks = grow_ranks([40])
    cache = ranks.caches[0]
    fractions: list[float] = []
    total = 32 << 30
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (total, total)
    )
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(
        torch.xpu,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: fractions.append(fraction),
    )
    cache.set_torch_memory_limit(8 << 30)
    added = cache.physical_bytes_for(24) - cache.physical_bytes_for(8)

    assert ranks.grow(0, 24) == 24

    assert cache.torch_memory_limit == (8 << 30) - added
    assert fractions[-1] == pytest.approx(((8 << 30) - added) / total)


class FakeConnector:
    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.resized: list[tuple[object, int]] = []

    def supports_kv_cache_resize(self) -> bool:
        return self.supported

    def resize_kv_caches(self, kv_caches, num_blocks) -> None:
        self.resized.append((kv_caches, num_blocks))


@pytest.mark.parametrize("supported", [True, False])
def test_grow_follows_the_connector_or_refuses_before_changing_anything(
    grow_ranks, monkeypatch, supported
):
    ranks = grow_ranks([40])
    connector = FakeConnector(supported)
    monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(extensible_kv_cache, "get_kv_transfer_group", lambda: connector)
    creates = ranks.fake_xpu.ops.calls.count("create")

    if supported:
        assert ranks.grow(0, 24) == 24
        assert connector.resized == [(ranks.runners[0].kv_caches_by_layer, 24)]
    else:
        with pytest.raises(NotImplementedError, match="resizing the KV cache"):
            ranks.grow(0, 24)
        assert connector.resized == []
        assert ranks.caches[0].num_committed_blocks == 8
        assert ranks.fake_xpu.ops.calls.count("create") == creates


def test_grow_refuses_hisparse(grow_ranks):
    ranks = grow_ranks([40])
    ranks.runners[0].vllm_config.attention_config.hisparse_config = object()
    with pytest.raises(NotImplementedError, match="HiSparse"):
        ranks.grow(0, 24)
    assert ranks.caches[0].num_committed_blocks == 8


def test_rebinding_gives_views_over_the_grown_blocks_at_the_same_base(
    fake_xpu, monkeypatch
):
    config, spec = _kv_cache_config(KVCacheLayout.LBNHC)
    layout = KVCacheLayout.LBNHC
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), layout)
    bound: list[dict[str, torch.Tensor]] = []
    monkeypatch.setattr(
        extensible_kv_cache,
        "bind_kv_caches",
        lambda committed, *args: bound.append(committed),
    )
    runner = SimpleNamespace(
        extensible_kv_cache=kv_cache,
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout, use_replayssm=False
        ),
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None),
            cache_config=SimpleNamespace(use_replayssm=False),
        ),
        kv_cache_config=copy.copy(config),
        kernel_block_sizes=None,
        attn_groups=[],
        speculator=None,
        device=torch.device("cpu"),
        kv_caches_by_layer={},
    )
    try:
        kv_cache.allocate(kv_cache.size)
        ptrs = []
        for num_blocks in (8, 24):
            kv_cache.commit(num_blocks)
            runner.kv_cache_config.num_blocks = num_blocks
            extensible_kv_cache.rebind_committed_views(runner)
            views = bound[-1]
            assert len(views) == NUM_LAYERS
            assert all(v.shape[0] == num_blocks for v in views.values())
            ptrs.append({name: v.data_ptr() for name, v in views.items()})
        assert ptrs[0] == ptrs[1]
    finally:
        kv_cache.free()


def test_extend_kv_cache_rebinds_through_the_shared_function(fake_xpu, monkeypatch):
    """Startup extension commits, rebinds the views, then creates the
    connector, in that order."""
    config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
    layout = KVCacheLayout.LBNHC
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), layout)
    order: list[str] = []
    monkeypatch.setattr(
        extensible_kv_cache,
        "ensure_kv_transfer_initialized",
        lambda *a: order.append("connector config"),
    )
    monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(
        extensible_kv_cache,
        "rebind_committed_views",
        lambda runner: order.append(f"rebind {runner.kv_cache_config.num_blocks}"),
    )
    monkeypatch.setattr(
        extensible_kv_cache,
        "get_kv_connector",
        lambda *a: order.append("connector") or "connector",
    )
    runner = SimpleNamespace(
        extensible_kv_cache=kv_cache,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
        kv_cache_config=copy.copy(config),
        kv_caches_by_layer={},
    )
    try:
        kv_cache.allocate(kv_cache.size)
        extensible_kv_cache.extend_kv_cache(runner, config, 12)
        assert order == ["connector config", "rebind 12", "connector"]
        assert kv_cache.num_committed_blocks == 12
        assert runner.kv_connector == "connector"
    finally:
        kv_cache.free()


def _measure_startup(monkeypatch, kv_cache, initial_blocks, free_memory):
    monkeypatch.setattr(
        extensible_kv_cache,
        "current_platform",
        SimpleNamespace(free_device_memory=lambda device: free_memory),
    )
    monkeypatch.setattr(
        extensible_kv_cache,
        "measure_kv_cache_bytes",
        partial(
            extensible_kv_cache.measure_kv_cache_bytes,
            margin_floor_bytes=0,
            margin_fraction=0.0,
        ),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (1 << 40, 1 << 40)
    )
    monkeypatch.setattr(
        torch.xpu, "set_per_process_memory_fraction", lambda fraction, device=None: None
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
    )
    return extensible_kv_cache.measure_kv_cache_blocks(
        kv_cache,
        vllm_config,
        torch.device("xpu", 0),
        init_free_memory=1024 * MIB,
        requested_memory=1024 * MIB,
        initial_blocks=initial_blocks,
    )


@pytest.mark.parametrize(
    ("initial_blocks", "expected"),
    [(None, 40), (60, 40), (40, 40), (12, 12), (1, 1)],
)
def test_startup_commits_no_more_than_the_initial_blocks_setting(
    fake_xpu, monkeypatch, initial_blocks, expected
):
    """Unset, startup commits what the measurement allows (40 blocks here); set,
    the smaller of the two."""
    config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), KVCacheLayout.LBNHC)
    try:
        free = _free_for(0, kv_cache, 40)
        assert _measure_startup(monkeypatch, kv_cache, initial_blocks, free) == expected
    finally:
        kv_cache.free()


@pytest.mark.parametrize(
    ("enabled", "initial_blocks", "refused"),
    [(True, 100, False), (False, 100, True), (False, None, False), (True, None, False)],
)
def test_initial_blocks_needs_the_extensible_kv_cache(enabled, initial_blocks, refused):
    """The setting is checked once the extensible KV cache is settled on."""
    from vllm.config.vllm import VllmConfig

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_extensible_kv_cache=enabled,
            extensible_kv_cache_initial_blocks=initial_blocks,
        )
    )
    if refused:
        with pytest.raises(ValueError, match="extensible_kv_cache_initial_blocks"):
            VllmConfig._validate_extensible_kv_cache_initial_blocks(config)
    else:
        VllmConfig._validate_extensible_kv_cache_initial_blocks(config)


def test_failed_access_grant_leaves_nothing_mapped(fake_xpu, monkeypatch):
    """A chunk whose access grant fails is unmapped and released again, so a
    later commit of the same range maps it afresh instead of over it."""
    et = ExtensibleTensor(4 * GRANULE, device="xpu:0", num_segments=2)
    try:
        grant = fake_xpu.driver.set_access
        grants: list[int] = []

        def set_access(ptr: int, size: int, device_index: int) -> None:
            grants.append(size)
            if len(grants) == 1:
                raise RuntimeError("UR_RESULT_ERROR_UNKNOWN")
            grant(ptr, size, device_index)

        monkeypatch.setattr(fake_xpu.driver, "set_access", set_access)
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            et.back_segments_([GRANULE] * 2)
        assert fake_xpu.ops.physical_bytes == 0
        assert et.num_physical_chunks == 0

        et.resize_segments_([GRANULE] * 2, zero_new=True)
        assert fake_xpu.ops.physical_bytes == 2 * GRANULE
        assert et.num_physical_chunks == 2
        assert torch.count_nonzero(et.segment_view(0)) == 0
    finally:
        et.free()


# Chunked commit and tail release.

POISON = 0xA5


def _objects(et: ExtensibleTensor) -> list[tuple[int, int]]:
    """(offset, size) of every mapped physical object, in granules."""
    return sorted(
        (offset // GRANULE, size // GRANULE) for _, offset, size in et._buffer._handles
    )


def test_chunked_commit_maps_objects_of_the_chunk_size(fake_xpu):
    """A run is split into objects of at most the chunk size, on a grid
    counted from the segment start whatever sizes the commits come in; a chunk
    size that is not a granule multiple rounds up."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE - 1)
    try:
        assert et.max_chunk_bytes == 2 * GRANULE
        et.resize_per_segment_(3 * GRANULE, zero_new=True)
        assert _objects(et) == [(0, 2), (2, 1)]
        et.resize_per_segment_(7 * GRANULE, zero_new=True)
        # The partial chunk [2, 3) is not extended; the new run starts there.
        assert _objects(et) == [(0, 2), (2, 1), (3, 1), (4, 2), (6, 1)]
        assert fake_xpu.ops.calls.count("create") == 5
        assert fake_xpu.ops.calls.count("map") == 5
        assert et.physical_bytes == fake_xpu.ops.physical_bytes == 7 * GRANULE
    finally:
        et.free()


def _grow_each(ranks: GrowFixture, target: int) -> list[object]:
    """Every rank's result of one grow, an exception where the rank raised."""
    group = MinGroup(len(ranks.runners))
    results: list[object] = [None] * len(ranks.runners)

    def run(rank: int) -> None:
        try:
            results[rank] = ranks.grow(rank, target, group.reducer(rank))
        except BaseException as e:  # noqa: BLE001
            results[rank] = e

    threads = [
        threading.Thread(target=run, args=(r,)) for r in range(len(ranks.runners))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert group.rounds == 2
    return results


def test_rank_that_cannot_measure_still_joins_both_agreements(grow_ranks, monkeypatch):
    """A rank whose measurement raises votes for no growth and raises only
    after both agreements, so the other ranks are not left waiting in one."""
    ranks = grow_ranks([40, 40])
    measured = extensible_kv_cache.current_platform.free_device_memory

    def free_device_memory(device):
        if device.index == 1:
            raise RuntimeError("UR_RESULT_ERROR_UNKNOWN")
        return measured(device)

    monkeypatch.setattr(
        extensible_kv_cache,
        "current_platform",
        SimpleNamespace(free_device_memory=free_device_memory),
    )
    creates = ranks.fake_xpu.ops.calls.count("create")

    results = _grow_each(ranks, 24)

    assert results[0] == 8
    assert isinstance(results[1], RuntimeError)
    assert [c.num_committed_blocks for c in ranks.caches] == [8, 8]
    assert ranks.fake_xpu.ops.calls.count("create") == creates
    assert ranks.rebinds == []


def test_grow_that_failed_after_the_agreement_is_finished_by_the_next_resize(
    grow_ranks, monkeypatch
):
    """A rank that fails once every rank agreed and mapped reports the failure
    and keeps serving its old views; the next resize finishes its grow, zeroing
    included, before any rank reports the new count."""
    ranks = grow_ranks([40, 40])
    rebound: list[tuple[int, int]] = []
    fail = {1}

    def rebind(runner):
        rank = runner.device.index
        if rank in fail:
            fail.discard(rank)
            raise RuntimeError("device error while rebinding")
        rebound.append((rank, runner.kv_cache_config.num_blocks))

    monkeypatch.setattr(extensible_kv_cache, "rebind_committed_views", rebind)

    results = _grow_each(ranks, 24)

    assert results[0] == 24
    assert isinstance(results[1], RuntimeError)
    failed = ranks.caches[1]
    assert failed.unfinished_grow == (8, 24, None)
    assert ranks.caches[0].unfinished_grow is None
    # The failed rank still describes the views it serves with.
    assert ranks.runners[1].kv_cache_config.num_blocks == 8
    # Whatever the failed attempt left in the new blocks must not survive.
    full = failed.buffer.full_view()
    for offset, stride in zip(failed.segment_offsets, failed.segment_strides):
        full[offset + 8 * stride : offset + 24 * stride].fill_(0xFF)
    failed.buffer.segment_view(0)[:64].fill_(7)

    assert ranks.grow_all(24) == [24, 24]

    assert rebound == [(0, 24), (1, 24)]
    assert failed.unfinished_grow is None
    assert ranks.runners[1].kv_cache_config.num_blocks == 24
    view = failed.buffer.segment_view(0)
    assert torch.all(view[:64] == 7)
    assert torch.count_nonzero(view[8 * ranks.spec.page_size_bytes :]) == 0


def test_rank_that_cannot_finish_an_earlier_grow_stops_every_rank(
    grow_ranks, monkeypatch
):
    """While one rank still fails to finish its grow, a resize toward more
    grows no rank, and that rank raises after the agreements."""
    ranks = grow_ranks([40, 40])

    def rebind(runner):
        if runner.device.index == 1:
            raise RuntimeError("device error while rebinding")

    monkeypatch.setattr(extensible_kv_cache, "rebind_committed_views", rebind)
    results = _grow_each(ranks, 24)
    assert results[0] == 24
    assert isinstance(results[1], RuntimeError)

    results = _grow_each(ranks, 32)

    assert results[0] == 24
    assert isinstance(results[1], RuntimeError)
    assert [c.num_committed_blocks for c in ranks.caches] == [24, 24]
    assert ranks.caches[1].unfinished_grow == (8, 24, None)


# Measuring while serving.


def _sized_with_reserve(ranks: GrowFixture, transient: int) -> ExtensibleKVCache:
    """Rank 0 as its startup sizing left it: `transient` bytes kept free beyond
    the blocks that fit, with torch at 512 MiB reserved and 500 MiB allocated."""
    cache = ranks.caches[0]
    cache.reserved_headroom_bytes = transient
    cache.torch_reserved_at_sizing = 512 * MIB
    cache.torch_allocated_at_sizing = 500 * MIB
    ranks.free_memory[0] += transient
    ranks.torch_reserved = 512 * MIB
    ranks.torch_stats = {
        "allocated_bytes.all.current": 500 * MIB,
        "allocated_bytes.all.peak": 500 * MIB,
    }
    return cache


def test_serving_measurement_counts_torch_cache_as_reclaimable(grow_ranks, monkeypatch):
    """Torch's cache that is reserved and not allocated is room a grow may
    take, as long as torch keeps the startup reserve above its allocation.
    One line reports every term."""
    ranks = grow_ranks([40])
    transient, taken = 64 * MIB, 24 * MIB
    cache = _sized_with_reserve(ranks, transient)
    ranks.torch_reserved += taken
    ranks.torch_stats = {
        "allocated_bytes.all.current": 510 * MIB,
        "allocated_bytes.all.peak": 530 * MIB,
    }
    ranks.free_memory[0] -= taken
    lines: list[str] = []
    monkeypatch.setattr(
        extensible_kv_cache.logger,
        "info",
        lambda message, *args: lines.append(message % args),
    )
    # Free: 40 blocks' worth plus 40 MiB; reclaimable 26 MiB; keep 64 MiB.
    expected = cache.blocks_within(cache.physical_bytes_for(40) + 2 * MIB)
    assert expected >= 40

    assert ranks.grow(0, 50) == min(50, expected)

    measured = [line for line in lines if "resize measurement" in line]
    assert len(measured) == 1
    assert "(0.03 reclaimable)" in measured[0]
    assert "keep 0.06 GiB above the allocation" in measured[0]
    assert f"{expected} blocks fit" in measured[0]


def test_serving_measurement_keeps_free_the_peak_serving_has_shown(grow_ranks):
    """A peak above the allocation larger than the startup reserve raises
    the room torch keeps."""
    ranks = grow_ranks([40])
    transient, above = 64 * MIB, 8 * MIB
    cache = _sized_with_reserve(ranks, transient)
    ranks.torch_stats["allocated_bytes.all.peak"] = 500 * MIB + transient + above
    # Free 40 blocks' worth + 64 MiB, reclaimable 12 MiB, keep 72 MiB.
    expected = cache.blocks_within(cache.physical_bytes_for(40) + 4 * MIB)
    without_peak = cache.blocks_within(cache.physical_bytes_for(40) + 12 * MIB)
    assert expected < without_peak

    assert ranks.grow(0, 50) == expected


def test_serving_measurement_counts_memory_taken_outside_torch(grow_ranks):
    """Device memory that went elsewhere since the sizing is simply not free."""
    ranks = grow_ranks([40])
    cache = _sized_with_reserve(ranks, 64 * MIB)
    elsewhere = 8 * MIB
    ranks.free_memory[0] -= elsewhere
    expected = cache.blocks_within(cache.physical_bytes_for(40) + 12 * MIB - elsewhere)

    assert ranks.grow(0, 50) == expected


GIB = 1 << 30


def _card_room(**overrides):
    """The terms of the measurement on four B70 at 5,993 blocks after a load
    (allocated 19.5 GiB, reserved 22.34, peak 20.52, startup reserve 1.11,
    cap 21.54 over 5.14 GiB mapped), with the device's free memory as given."""
    terms = dict(
        free_memory=0,
        mapped_bytes=int(5.14 * GIB),
        torch_allocated=int(19.5 * GIB),
        torch_reserved=int(22.34 * GIB),
        torch_peak=int(20.52 * GIB),
        startup_reserve=int(1.11 * GIB),
        torch_budget=int((21.54 + 5.14) * GIB),
    )
    terms.update(overrides)
    return extensible_kv_cache.serving_kv_room(**terms)


def test_serving_room_at_the_card_numbers():
    """With nothing free on the device the room is in torch's cache: none
    until it is emptied; afterwards the cap, not the device, binds."""
    bytes_per_block = 6.14 * GIB / 7158
    before = _card_room()
    assert before.reclaimable == pytest.approx(2.84 * GIB, rel=1e-3)
    assert before.keep == pytest.approx(1.11 * GIB, rel=1e-3)
    assert before.room == 0
    # Emptying the cache returned 1.76 GiB on the card that bound before the
    # fix pass (its 6,744 = 5.14 + 1.76 - 1.11 GiB); 1.08 GiB stays cached.
    after = _card_room(
        free_memory=int(1.76 * GIB), torch_reserved=int((19.5 + 1.08) * GIB)
    )
    assert after.device_room == pytest.approx(1.73 * GIB, rel=1e-3)
    assert after.cap_room == pytest.approx(0.93 * GIB, rel=1e-3)
    assert after.room == after.cap_room
    assert 5993 + int(after.room // bytes_per_block) == pytest.approx(7077, abs=2)


def test_serving_room_refuses_when_torch_holds_its_reservation():
    room = _card_room(torch_allocated=int(22.2 * GIB))
    assert room.room == 0
    # And no cap: the device alone decides, still nothing.
    assert _card_room(torch_allocated=int(22.2 * GIB), torch_budget=None).room == 0


def test_serving_measurement_empties_torch_cache_only_when_the_grow_needs_it(
    grow_ranks, monkeypatch
):
    ranks = grow_ranks([40])
    cache = _sized_with_reserve(ranks, 64 * MIB)
    empties: list[int] = []
    # Torch caches 100 MiB it does not use; emptying gives it to the device.
    # Blocks are 3 MiB: free memory then covers 20 blocks beyond the 8.
    ranks.torch_reserved += 100 * MIB
    ranks.free_memory[0] -= 100 * MIB

    def empty_cache() -> None:
        empties.append(1)
        ranks.torch_reserved -= 100 * MIB
        ranks.free_memory[0] += 100 * MIB

    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)

    # A grow the free memory covers leaves the cache alone.
    assert ranks.grow(0, 12) == 12
    assert empties == []
    # One that needs more empties it once, then measures again.
    assert ranks.grow(0, 40) == 40
    assert empties == [1]
    assert cache.num_committed_blocks == 40


def test_serving_measurement_refuses_when_torch_has_nothing_to_give(grow_ranks):
    ranks = grow_ranks([40])
    cache = _sized_with_reserve(ranks, 64 * MIB)
    # Torch took the whole startup reserve and more, and allocated it all.
    ranks.free_memory[0] = 0
    ranks.torch_reserved = 600 * MIB
    ranks.torch_stats = {
        "allocated_bytes.all.current": 600 * MIB,
        "allocated_bytes.all.peak": 600 * MIB,
    }
    assert ranks.grow(0, 40) == 8
    assert cache.num_committed_blocks == 8


def test_grow_cap_follows_from_the_budget_and_stops_the_grow(grow_ranks, monkeypatch):
    """The cap after a grow is the budget less what the KV cache maps; a grow
    that would put it below the allocation plus the reserve stops short."""
    ranks = grow_ranks([40])
    cache = _sized_with_reserve(ranks, 0)
    total = 32 << 30
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device=None: (total, total)
    )
    monkeypatch.setattr(
        torch.xpu, "set_per_process_memory_fraction", lambda fraction, device=None: None
    )
    # Torch holds 500 MiB; the budget leaves it room up to 20 blocks.
    cap = 500 * MIB + cache.physical_bytes_for(20) - cache.physical_bytes
    cache.set_torch_memory_limit(cap)
    assert cache.torch_memory_budget is None
    cache.anchor_torch_memory_budget()
    budget = cache.torch_memory_budget
    assert budget == cap + cache.physical_bytes

    assert ranks.grow(0, 40) == 20
    assert cache.torch_memory_limit == budget - cache.physical_bytes
    assert cache.torch_memory_limit == 500 * MIB


def test_sizing_baseline_records_what_torch_holds(fake_xpu, monkeypatch):
    config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
    device = torch.device("xpu", 0)
    kv_cache = ExtensibleKVCache(config, device, KVCacheLayout.LBNHC)
    resets: list[torch.device] = []
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda device=None: 7)
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda device=None: 5)
    monkeypatch.setattr(
        torch.accelerator,
        "reset_peak_memory_stats",
        lambda device=None: resets.append(device),
    )
    try:
        assert kv_cache.torch_reserved_at_sizing is None
        kv_cache.record_sizing_baseline(device)
        assert kv_cache.torch_reserved_at_sizing == 7
        assert kv_cache.torch_allocated_at_sizing == 5
        assert resets == [device]
    finally:
        kv_cache.free()


def test_chunk_grid_follows_the_segment_start(fake_xpu):
    """Each segment's chunks are counted from its own start, so a segment that
    begins mid-chunk of the reservation still has whole chunks of its own."""
    capacities = [3 * GRANULE, 4 * GRANULE]
    et = ExtensibleTensor(
        7 * GRANULE,
        device="xpu:0",
        segment_capacities=capacities,
        max_chunk_bytes=2 * GRANULE,
    )
    try:
        et.resize_segments_([3 * GRANULE, 4 * GRANULE], zero_new=True)
        assert _objects(et) == [(0, 2), (2, 1), (3, 2), (5, 2)]
    finally:
        et.free()


def test_no_chunk_size_keeps_one_object_per_run(fake_xpu):
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0")
    try:
        assert et.max_chunk_bytes is None
        et.resize_per_segment_(3 * GRANULE, zero_new=True)
        et.resize_per_segment_(7 * GRANULE, zero_new=True)
        assert _objects(et) == [(0, 3), (3, 4)]
        assert fake_xpu.ops.calls.count("create") == 2
        assert et.num_physical_chunks == 2
    finally:
        et.free()


def test_whole_runs_override_the_chunk_size(fake_xpu):
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        et.resize_segments_([7 * GRANULE], zero_new=True, whole_runs=True)
        assert _objects(et) == [(0, 7)]
    finally:
        et.free()


@pytest.mark.parametrize("chunk", [0, -GRANULE])
def test_chunk_size_must_be_positive(fake_xpu, chunk):
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        ExtensibleTensor(4 * GRANULE, device="xpu:0", max_chunk_bytes=chunk)
    assert not fake_xpu.ops._reservations


def test_shrink_releases_the_whole_chunks_above_the_new_size(fake_xpu):
    """Bytes below the new size keep their contents; the objects above it are
    unmapped whole (the fake poisons their pages) and released; a chunk that
    straddles the new size stays."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        base = et.base_ptr
        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        full = et.full_view()
        full[: 12 * GRANULE].fill_(11)
        ops = fake_xpu.ops
        calls = len(ops.calls)

        # 5 granules: chunk [4, 6) straddles, [6, 12) goes: three objects.
        assert et.shrink_segments_([5 * GRANULE]) == 6 * GRANULE
        assert ops.calls[calls:] == ["unmap", "release"] * 3
        assert ops.physical_bytes == et.physical_bytes == 6 * GRANULE
        assert et.num_bytes == 5 * GRANULE
        assert et.base_ptr == base
        assert et.segment_view(0).numel() == 5 * GRANULE
        assert torch.all(et.segment_view(0) == 11)
        assert torch.all(full[6 * GRANULE : 12 * GRANULE] == POISON)
        assert _objects(et) == [(0, 2), (2, 2), (4, 2)]

        # The same shrink again changes nothing.
        calls = len(ops.calls)
        assert et.shrink_segments_([5 * GRANULE]) == 0
        assert ops.calls[calls:] == []
    finally:
        et.free()
    assert ops.physical_bytes == 0


def test_shrink_to_a_chunk_boundary_releases_exactly_the_chunks_above(fake_xpu):
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        before = fake_xpu.ops.physical_bytes
        released = et.shrink_segments_([4 * GRANULE])
        assert released == 8 * GRANULE
        assert before - fake_xpu.ops.physical_bytes == released
        assert et.physical_bytes == 4 * GRANULE
        assert et.releasable_bytes([2 * GRANULE]) == 2 * GRANULE
        assert et.shrink_segments_([0]) == 4 * GRANULE
        assert fake_xpu.ops.physical_bytes == 0
    finally:
        et.free()


def test_grow_after_shrink_maps_fresh_zeroed_chunks(fake_xpu):
    """Everything above the old committed size reads as zero after a regrow,
    whether its pages were released and mapped afresh or kept in a straddling
    chunk; the bytes below are untouched."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        et.full_view()[: 12 * GRANULE].fill_(11)
        et.shrink_segments_([5 * GRANULE])
        creates = fake_xpu.ops.calls.count("create")

        # Granule 5 is still mapped, in the straddling chunk, holding 11s.
        et.resize_per_segment_(6 * GRANULE, zero_new=True)
        assert fake_xpu.ops.calls.count("create") == creates
        view = et.segment_view(0)
        assert torch.all(view[: 5 * GRANULE] == 11)
        assert torch.count_nonzero(view[5 * GRANULE :]) == 0

        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        assert fake_xpu.ops.calls.count("create") == creates + 3
        assert et.physical_bytes == fake_xpu.ops.physical_bytes == 12 * GRANULE
        view = et.segment_view(0)
        assert torch.all(view[: 5 * GRANULE] == 11)
        assert torch.count_nonzero(view[5 * GRANULE :]) == 0
    finally:
        et.free()


def test_shrink_keeps_a_granule_shared_with_the_next_segment(fake_xpu):
    """Segments need not start on a granule. A chunk holding the end of one
    segment and the start of the next is not wholly inside either's tail."""
    half = GRANULE // 2
    capacities = [3 * GRANULE + half, 3 * GRANULE]
    et = ExtensibleTensor(
        sum(capacities),
        device="xpu:0",
        segment_capacities=capacities,
        max_chunk_bytes=GRANULE,
    )
    try:
        et.resize_segments_(list(capacities), zero_new=True)
        full = et.full_view()
        full.fill_(2)
        # The first segment gives back everything above its first granule: the
        # granule it shares with the second segment (granule 3) stays.
        released = et.shrink_segments_([GRANULE, capacities[1]])
        assert released == 2 * GRANULE
        assert _objects(et)[:2] == [(0, 1), (3, 1)]
        second = et.segment_view(1)
        assert second.numel() == capacities[1]
        assert torch.all(second == 2)
    finally:
        et.free()


def test_shrink_rejects_a_bad_size(fake_xpu):
    et = ExtensibleTensor(
        8 * GRANULE, device="xpu:0", num_segments=2, max_chunk_bytes=GRANULE
    )
    try:
        et.resize_per_segment_(2 * GRANULE, zero_new=True)
        before = (fake_xpu.ops.physical_bytes, et.bytes_per_segment)
        for sizes in ([GRANULE], [3 * GRANULE, GRANULE], [GRANULE, -1]):
            with pytest.raises(ValueError):
                et.shrink_segments_(sizes)
            with pytest.raises(ValueError):
                et.releasable_bytes(sizes)
        assert (fake_xpu.ops.physical_bytes, et.bytes_per_segment) == before
    finally:
        et.free()


def test_release_range_takes_any_range_and_a_recommit_fills_the_hole(fake_xpu):
    """The generic primitive: objects wholly inside a range go, whatever else
    is mapped on both sides; a partial object at either end stays."""
    et = ExtensibleTensor(8 * GRANULE, device="xpu:0", max_chunk_bytes=GRANULE)
    try:
        buffer = et._buffer
        buffer.ensure_committed_range(0, 8 * GRANULE)
        full = et.full_view()
        full.fill_(3)
        # [2.5, 5.5) granules wholly contains the objects at granules 3 and 4.
        start, end = 2 * GRANULE + GRANULE // 2, 5 * GRANULE + GRANULE // 2
        assert buffer.releasable_bytes(start, end) == 2 * GRANULE
        assert buffer.release_range(start, end) == 2 * GRANULE
        assert buffer.committed_bytes == 6 * GRANULE
        assert torch.all(full[: 3 * GRANULE] == 3)
        assert torch.all(full[5 * GRANULE :] == 3)
        assert torch.all(full[3 * GRANULE : 5 * GRANULE] == POISON)
        assert buffer.release_range(start, end) == 0

        buffer.ensure_committed_range(2 * GRANULE, 6 * GRANULE)
        assert buffer.committed_bytes == 8 * GRANULE
        assert torch.all(full[3 * GRANULE : 5 * GRANULE] == 0)
        assert torch.all(full[: 3 * GRANULE] == 3)
        with pytest.raises(ValueError):
            buffer.release_range(5, 4)
    finally:
        et.free()


def test_a_failed_chunk_undoes_the_whole_run(fake_xpu, monkeypatch):
    """A commit that fails on its third chunk unmaps and releases the two it
    mapped and records nothing, so the retry maps the whole run afresh."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        ops = fake_xpu.ops
        real_create = ops.vmm_create
        creates = []

        def failing_create(size, device_index):
            creates.append(size)
            if len(creates) in (3, 4):  # the retry after empty_cache fails too
                raise RuntimeError("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY")
            return real_create(size, device_index)

        monkeypatch.setattr(ops, "vmm_create", failing_create)
        monkeypatch.setattr(
            torch.accelerator, "empty_cache", lambda: None, raising=False
        )
        with pytest.raises(RuntimeError, match="OUT_OF_DEVICE_MEMORY"):
            et.back_segments_([12 * GRANULE])
        assert et._buffer._mapped_granules == set()
        assert et._buffer._handles == []
        assert ops.physical_bytes == 0
        assert ops.calls.count("set_access") == 0
        assert et.bytes_per_segment == 0

        monkeypatch.setattr(ops, "vmm_create", real_create)
        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        assert _objects(et) == [(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)]
    finally:
        et.free()


def _fail_grants(fake_xpu, monkeypatch, which):
    """Make the access grants numbered `which` (from 1) raise."""
    grant = fake_xpu.driver.set_access
    count = [0]

    def set_access(ptr: int, size: int, device_index: int) -> None:
        count[0] += 1
        if count[0] in which:
            raise RuntimeError("UR_RESULT_ERROR_UNKNOWN")
        grant(ptr, size, device_index)

    monkeypatch.setattr(fake_xpu.driver, "set_access", set_access)
    return grant


def test_a_failed_grant_undoes_a_three_chunk_run(fake_xpu, monkeypatch):
    """A run of three chunks whose access grant fails: nothing is recorded,
    the physical bytes are back to what they were before the call, objects of
    earlier calls are untouched, and a retry maps three fresh chunks."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        ops = fake_xpu.ops
        buffer = et._buffer
        buffer.ensure_committed_range(0, 2 * GRANULE)  # an earlier call
        before = (
            ops.physical_bytes,
            list(buffer._handles),
            set(buffer._mapped_granules),
        )
        assert before[0] == 2 * GRANULE

        grant = _fail_grants(fake_xpu, monkeypatch, {1})
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            buffer.ensure_committed_range(2 * GRANULE, 8 * GRANULE)
        assert (
            ops.physical_bytes,
            list(buffer._handles),
            set(buffer._mapped_granules),
        ) == before

        monkeypatch.setattr(fake_xpu.driver, "set_access", grant)
        creates = ops.calls.count("create")
        buffer.ensure_committed_range(2 * GRANULE, 8 * GRANULE)
        assert ops.calls.count("create") == creates + 3
        assert ops.physical_bytes == 8 * GRANULE
        assert len(buffer._handles) == 4
    finally:
        et.free()


def test_a_failed_second_run_undoes_the_first_run_of_the_call(fake_xpu, monkeypatch):
    """The undo covers everything one call created, not only the failing run:
    a call over two unmapped runs whose second grant fails leaves neither."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        ops = fake_xpu.ops
        buffer = et._buffer
        buffer.ensure_committed_range(4 * GRANULE, 6 * GRANULE)  # a mapped hole
        _fail_grants(fake_xpu, monkeypatch, {2})  # the grants of run 1 and run 2
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            buffer.ensure_committed_range(0, 10 * GRANULE)
        assert buffer._mapped_granules == {4, 5}
        assert len(buffer._handles) == 1
        assert ops.physical_bytes == 2 * GRANULE
    finally:
        et.free()


def test_a_failed_undo_keeps_what_it_could_not_release(fake_xpu, monkeypatch):
    """If the undo itself fails on an object, that object stays in the books
    (handle and granules) so a later call can reuse or release it, and the
    original error is raised with the undo failure chained."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        ops = fake_xpu.ops
        buffer = et._buffer
        grant = _fail_grants(fake_xpu, monkeypatch, {1})
        real_unmap = ops.vmm_unmap
        unmaps = []

        def failing_unmap(ptr, size):
            unmaps.append(ptr)
            if len(unmaps) == 2:
                raise RuntimeError("unmap refused")
            real_unmap(ptr, size)

        monkeypatch.setattr(ops, "vmm_unmap", failing_unmap)
        with pytest.raises(RuntimeError, match="UNKNOWN") as raised:
            buffer.ensure_committed_range(0, 6 * GRANULE)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert "unmap refused" in str(raised.value.__cause__)
        # Three objects were created; the second one to be undone (the middle
        # one, undone newest first) is the one left behind.
        assert len(buffer._handles) == 1
        assert ops.physical_bytes == 2 * GRANULE
        _, offset, size = buffer._handles[0]
        assert buffer._mapped_granules == set(
            range(offset // GRANULE, (offset + size) // GRANULE)
        )

        monkeypatch.setattr(ops, "vmm_unmap", real_unmap)
        monkeypatch.setattr(fake_xpu.driver, "set_access", grant)
        assert buffer.release_range(0, 16 * GRANULE) == 2 * GRANULE
        assert ops.physical_bytes == 0
    finally:
        et.free()


def test_a_failed_release_keeps_the_books_straight(fake_xpu, monkeypatch):
    """If an unmap fails part-way, the objects already released are gone from
    the books and the rest stay mapped; the committed size is the new one."""
    et = ExtensibleTensor(16 * GRANULE, device="xpu:0", max_chunk_bytes=2 * GRANULE)
    try:
        et.resize_per_segment_(12 * GRANULE, zero_new=True)
        ops = fake_xpu.ops
        real_unmap = ops.vmm_unmap
        unmaps = []

        def failing_unmap(ptr, size):
            unmaps.append(ptr)
            if len(unmaps) == 2:
                raise RuntimeError("unmap refused")
            real_unmap(ptr, size)

        monkeypatch.setattr(ops, "vmm_unmap", failing_unmap)
        with pytest.raises(RuntimeError, match="unmap refused"):
            et.shrink_segments_([4 * GRANULE])
        assert et.num_bytes == 4 * GRANULE
        assert et.physical_bytes == ops.physical_bytes
        assert len(et._buffer._handles) == 5
        monkeypatch.setattr(ops, "vmm_unmap", real_unmap)
        # Repeating the shrink finishes it.
        assert et.shrink_segments_([4 * GRANULE]) == 6 * GRANULE
        assert et.physical_bytes == ops.physical_bytes == 4 * GRANULE
    finally:
        et.free()


def _chunked_kv_cache(layout, chunk_blocks=8):
    config, spec = _kv_cache_config(layout)
    kv_cache = ExtensibleKVCache(
        config,
        torch.device("xpu", 0),
        layout,
        release_chunk_bytes=chunk_blocks * spec.page_size_bytes,
    )
    return config, spec, kv_cache


def test_kv_cache_commits_in_release_chunks(fake_xpu):
    _, _, kv_cache = _chunked_kv_cache(KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        # 48 MiB per layer is 24 granules in chunks of four, the first one
        # shortened by the null block's granule.
        per_segment = _objects(kv_cache.buffer)
        assert len(per_segment) == NUM_LAYERS * 7
        assert fake_xpu.ops.physical_bytes == kv_cache.physical_bytes
        assert kv_cache.physical_bytes == kv_cache.physical_bytes_for(48)
    finally:
        kv_cache.free()


def test_kv_cache_defragment_still_gives_one_object_per_segment(fake_xpu):
    _, _, kv_cache = _chunked_kv_cache(KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        assert not all(kv_cache.buffer.segments_backed_by_one_chunk([48 * MIB] * 3))
        kv_cache.commit(48, defragment=True)
        assert kv_cache.buffer.num_physical_chunks == NUM_LAYERS
        assert all(kv_cache.buffer.segments_backed_by_one_chunk([48 * MIB] * 3))
        # Later growth is chunked again.
        kv_cache.commit(64)
        assert kv_cache.buffer.num_physical_chunks > NUM_LAYERS + 1
    finally:
        kv_cache.free()


@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_kv_cache_shrinks_to_a_smaller_prefix(fake_xpu, layout):
    """`shrink_to` keeps the contents of the kept blocks, gives back whole
    chunks of the tail (poisoned in the fake), and a regrow reads as zero."""
    _, _, kv_cache = _chunked_kv_cache(layout)
    try:
        view = kv_cache.allocate(kv_cache.size)
        base = view.data_ptr()
        kv_cache.commit(48)
        full = kv_cache.buffer.full_view()
        segments = list(zip(kv_cache.segment_offsets, kv_cache.segment_strides))
        for index, (offset, stride) in enumerate(segments):
            full[offset : offset + 48 * stride].fill_(index + 1)
        before = fake_xpu.ops.physical_bytes

        released = kv_cache.shrink_to(16)
        assert kv_cache.num_committed_blocks == 16
        assert released > 0
        assert before - fake_xpu.ops.physical_bytes == released
        assert kv_cache.physical_bytes == fake_xpu.ops.physical_bytes
        for index, (offset, stride) in enumerate(segments):
            assert torch.all(full[offset : offset + 16 * stride] == index + 1)
            assert torch.all(
                full[offset + 40 * stride : offset + 48 * stride] == POISON
            )
        assert kv_cache.buffer.full_view().data_ptr() == base

        kv_cache.commit(48)
        assert kv_cache.num_committed_blocks == 48
        for index, (offset, stride) in enumerate(segments):
            assert torch.all(full[offset : offset + 16 * stride] == index + 1)
            assert (
                torch.count_nonzero(full[offset + 16 * stride : offset + 48 * stride])
                == 0
            )
        assert kv_cache.physical_bytes == kv_cache.physical_bytes_for(48)
    finally:
        kv_cache.free()


def test_kv_cache_keeps_the_straddling_chunk_when_shrinking(fake_xpu):
    _, _, kv_cache = _chunked_kv_cache(KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        kv_cache.shrink_to(10)
        # 10 MiB per layer is five granules; the chunk of granules [4, 8)
        # holds the new size and stays, so more than five granules remain.
        assert kv_cache.physical_bytes > kv_cache.physical_bytes_for(10)
        assert kv_cache.physical_bytes == kv_cache.physical_bytes_for(16)
        assert fake_xpu.ops.physical_bytes == kv_cache.physical_bytes
    finally:
        kv_cache.free()


def test_kv_cache_blocks_releasable(fake_xpu):
    _, spec, kv_cache = _chunked_kv_cache(KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        chunk = 8 * spec.page_size_bytes * NUM_LAYERS
        assert kv_cache.blocks_releasable(0) == 0
        assert kv_cache.blocks_releasable(1) == 8
        assert kv_cache.blocks_releasable(chunk) == 8
        assert kv_cache.blocks_releasable(chunk + 1) == 16
        assert kv_cache.blocks_releasable(1 << 40) is None
        for need in (1, chunk, 3 * chunk + 1, 5 * chunk):
            k = kv_cache.blocks_releasable(need)
            assert k is not None
            assert kv_cache._releasable_bytes(48 - k) >= need
            assert kv_cache._releasable_bytes(48 - k + 1) < need
        # What it predicts is what shrinking then frees.
        k = kv_cache.blocks_releasable(2 * chunk)
        before = fake_xpu.ops.physical_bytes
        assert kv_cache.shrink_to(48 - k) == before - fake_xpu.ops.physical_bytes
        assert before - fake_xpu.ops.physical_bytes >= 2 * chunk
    finally:
        kv_cache.free()


def test_kv_cache_without_a_chunk_size_releases_little(fake_xpu):
    """Today's commit maps one object per run, so a tail inside it cannot be
    given back: only a shrink to (almost) nothing frees an object."""
    config, _ = _kv_cache_config(KVCacheLayout.LBNHC)
    kv_cache = ExtensibleKVCache(config, torch.device("xpu", 0), KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        assert kv_cache.blocks_releasable(1) == 46
        before = fake_xpu.ops.physical_bytes
        assert kv_cache.shrink_to(16) == 0
        assert fake_xpu.ops.physical_bytes == before
        assert kv_cache.num_committed_blocks == 16
        # The committed prefix is smaller; its pages are still there.
        kv_cache.commit(32)
        assert kv_cache.num_committed_blocks == 32
    finally:
        kv_cache.free()


@pytest.mark.parametrize(
    ("initial_blocks", "expected"), [(12, 24), (24, 24), (30, 30), (60, 40)]
)
def test_startup_cap_keeps_the_blocks_warmup_committed(
    fake_xpu, monkeypatch, initial_blocks, expected
):
    """Warmup leaves ids of the blocks it committed in the block tables, so a
    cap below that count must not unmap them. Two ranks with different free
    memory and no KV connector: both keep the warmup count, the extension
    unmaps nothing and what warmup wrote in its last block is still there."""
    layout = KVCacheLayout.LBNHC
    warmup_blocks = 24
    fits = [50, 40]
    caches = []
    counts = []
    order: list[str] = []
    monkeypatch.setattr(
        extensible_kv_cache, "ensure_kv_transfer_initialized", lambda *a: None
    )
    monkeypatch.setattr(extensible_kv_cache, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(
        extensible_kv_cache,
        "rebind_committed_views",
        lambda runner: order.append(f"rebind {runner.kv_cache_config.num_blocks}"),
    )
    monkeypatch.setattr(extensible_kv_cache, "get_kv_connector", lambda *a: None)
    try:
        for rank in range(2):
            config, _ = _kv_cache_config(layout)
            kv_cache = ExtensibleKVCache(config, torch.device("xpu", rank), layout)
            caches.append((config, kv_cache))
            kv_cache.allocate(kv_cache.size)
            kv_cache.commit(warmup_blocks)
            last = kv_cache.segment_strides[0] * (warmup_blocks - 1)
            kv_cache.buffer.segment_view(0)[last : last + 64].fill_(7)
            free = _free_for(warmup_blocks, kv_cache, fits[rank])
            counts.append(_measure_startup(monkeypatch, kv_cache, initial_blocks, free))
        assert counts == [min(max(initial_blocks, warmup_blocks), f) for f in fits]
        num_blocks = min(counts)
        assert num_blocks == expected

        unmaps = fake_xpu.ops.calls.count("unmap")
        for config, kv_cache in caches:
            runner = SimpleNamespace(
                extensible_kv_cache=kv_cache,
                vllm_config=SimpleNamespace(kv_transfer_config=None),
                cache_config=SimpleNamespace(
                    get_resolved_kv_cache_layout=lambda: layout
                ),
                kv_cache_config=copy.copy(config),
                kv_caches_by_layer={},
            )
            extensible_kv_cache.extend_kv_cache(runner, config, num_blocks)
            assert kv_cache.num_committed_blocks == num_blocks
            last = kv_cache.segment_strides[0] * (warmup_blocks - 1)
            assert torch.all(kv_cache.buffer.segment_view(0)[last : last + 64] == 7)
        assert fake_xpu.ops.calls.count("unmap") == unmaps
        assert order == [f"rebind {num_blocks}"] * 2
    finally:
        for _, kv_cache in caches:
            kv_cache.free()


@pytest.mark.parametrize(
    ("layout", "minimum"), [(KVCacheLayout.LBNHC, 1), (KVCacheLayout.BLNHC, 2)]
)
def test_kv_cache_shrink_bounds(fake_xpu, layout, minimum):
    _, _, kv_cache = _chunked_kv_cache(layout)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(16)
        assert kv_cache.min_committed_blocks == minimum
        for bad in (minimum - 1, 17):
            with pytest.raises(ValueError, match="Cannot shrink"):
                kv_cache.shrink_to(bad)
        assert kv_cache.shrink_to(16) == 0
        kv_cache.shrink_to(minimum)
        assert kv_cache.num_committed_blocks == minimum
    finally:
        kv_cache.free()


def test_grow_after_shrink_on_the_kv_cache_reuses_back_and_commit(fake_xpu):
    """`back` then `commit` (the grow path) work on a shrunk, chunked cache:
    the memory is mapped by `back` and nothing more by `commit`."""
    _, _, kv_cache = _chunked_kv_cache(KVCacheLayout.LBNHC)
    try:
        kv_cache.allocate(kv_cache.size)
        kv_cache.commit(48)
        kv_cache.shrink_to(16)
        kv_cache.back(40)
        assert kv_cache.num_committed_blocks == 16
        creates = fake_xpu.ops.calls.count("create")
        kv_cache.commit(40)
        assert fake_xpu.ops.calls.count("create") == creates
        assert kv_cache.num_committed_blocks == 40
    finally:
        kv_cache.free()


def test_release_chunk_setting(fake_xpu):
    from vllm.config import CacheConfig
    from vllm.engine.arg_utils import EngineArgs

    assert CacheConfig().kv_cache_release_chunk_mib is None
    assert CacheConfig(kv_cache_release_chunk_mib=32).kv_cache_release_chunk_mib == 32
    with pytest.raises(ValueError):
        CacheConfig(kv_cache_release_chunk_mib=0)
    assert EngineArgs().kv_cache_release_chunk_mib is None


@pytest.mark.parametrize(
    ("enabled", "chunk", "refused"),
    [(True, 32, False), (False, 32, True), (False, None, False), (True, None, False)],
)
def test_release_chunk_needs_the_extensible_kv_cache(enabled, chunk, refused):
    from vllm.config.vllm import VllmConfig

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_extensible_kv_cache=enabled, kv_cache_release_chunk_mib=chunk
        )
    )
    if refused:
        with pytest.raises(ValueError, match="kv_cache_release_chunk_mib"):
            VllmConfig._validate_kv_cache_release_chunk(config)
    else:
        VllmConfig._validate_kv_cache_release_chunk(config)


@pytest.mark.parametrize(
    ("chunk_mib", "registers", "expected"),
    [(None, False, None), (32, False, 32 << 20), (32, True, None), (None, True, None)],
)
def test_release_chunk_bytes_follows_the_setting_and_the_connector(
    chunk_mib, registers, expected
):
    """The setting becomes bytes, except where a connector registers the cache
    for RDMA and needs one allocation per segment."""
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_release_chunk_mib=chunk_mib),
        kv_transfer_config=SimpleNamespace(registers_kv_cache_memory=lambda: registers),
    )
    assert extensible_kv_cache.release_chunk_bytes(vllm_config) == expected
