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
    assert reduce_calls == [8, 3]  # only the measurement round ran


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
