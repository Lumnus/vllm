# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of weight components on NVMe (``weight_units``)."""

import os

import numpy as np
import pytest
import torch

from vllm.model_executor.offloader import expert_store_build as ebuild
from vllm.model_executor.offloader.expert_store import (
    CHUNK_BYTES,
    DATA_FILE,
    PAGE,
    ExpertStore,
    StoreCorruptError,
    StoreIncompleteError,
    StoreMismatchError,
    StoreSchema,
    TensorSpec,
)
from vllm.model_executor.offloader.weight_units import (
    CatalogLayout,
    ComponentSchema,
    ComponentSpec,
    ComponentStore,
    ComponentStoreWriter,
    allocate_host,
)

IDENT = "tower-test/v1"
MIB = 1 << 20


def dirs_of(tmp_path, n: int) -> list[str]:
    out = []
    for i in range(n):
        d = tmp_path / f"drive{i}"
        d.mkdir()
        out.append(str(d))
    return out


def rand_tensor(gen, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=gen).bool()
    itemsize = torch.empty((), dtype=dtype).element_size()
    count = 1
    for n in shape:
        count *= n
    raw = torch.randint(0, 256, (count * itemsize,), generator=gen, dtype=torch.uint8)
    return raw.view(dtype).reshape(shape)


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        )
    )


def rand_int8(seed: int, count: int) -> torch.Tensor:
    return rand_tensor(torch.Generator().manual_seed(seed), torch.int8, (count,))


def tower_state(big_mib: int = 300) -> dict[str, torch.Tensor]:
    """A synthetic tower: mixed dtypes, sizes from 1 KB to ``big_mib`` MiB."""
    gen = torch.Generator().manual_seed(7)
    state = {
        "patch.bias": rand_tensor(gen, torch.float32, (256,)),  # 1 KB
        "pos.weight": rand_tensor(gen, torch.bfloat16, (3, 512)),  # 3 KB
        "ln.weight": rand_tensor(gen, torch.float16, (1000,)),
        "qkv.weight": rand_tensor(gen, torch.int8, (1000, 1000)),  # 1 MB
        "fp8.weight": rand_tensor(gen, torch.float8_e4m3fn, (4097,)),  # odd size
        "mask": rand_tensor(gen, torch.bool, (13,)),
        "mlp.weight": rand_tensor(gen, torch.bfloat16, (2048, 4096)),  # 16 MiB
        "tail.bias": rand_tensor(gen, torch.float32, (5,)),
    }
    rows = big_mib * MIB // 2 // 1024
    state["embed.weight"] = rand_tensor(gen, torch.bfloat16, (rows, 1024))
    return state


def write_components(dirs, schema, states, **kw) -> None:
    with ComponentStoreWriter(dirs, schema, IDENT) as writer:
        for name, state in states.items():
            writer.write_component(name, state, **kw)
        writer.finalize()


def open_store(dirs, pool=64 * MIB, **kw) -> ComponentStore:
    kw.setdefault("pin", False)
    kw.setdefault("io_threads", 4)
    return ComponentStore.open(dirs, IDENT, pool, **kw)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def plan_of(split, tensors, cap=64 * 1024):
    spec = ComponentSpec("c", tuple(tensors), split)
    layout = CatalogLayout(ComponentSchema((spec,), max_extent_bytes=cap), 1, IDENT)
    return layout, layout.plans[0]


@pytest.mark.parametrize("split", ["whole", "tensor", 8192, 4096 * 5])
def test_every_tensor_byte_is_in_exactly_one_extent(split):
    rng = np.random.default_rng(1)
    tensors = [
        TensorSpec(f"t{i}", "int8", (int(n),))
        for i, n in enumerate(rng.integers(0, 70_000, 25))
    ]
    layout, plan = plan_of(split, tensors)
    covered = [np.zeros(t.nbytes, dtype=np.int32) for t in tensors]
    for extent in plan.extents:
        assert extent.unit_bytes % PAGE == 0
        assert extent.unit_bytes <= 64 * 1024
        assert extent.payload_bytes <= extent.unit_bytes
        prev_end = 0
        for p in extent.pieces:
            assert p.unit_offset >= prev_end  # pieces do not overlap
            prev_end = p.unit_offset + p.nbytes
            covered[p.tensor][p.tensor_start : p.tensor_start + p.nbytes] += 1
        assert prev_end <= extent.payload_bytes
    for c in covered:
        assert (c == 1).all()


def test_cut_modes():
    tensors = [TensorSpec("a", "int8", (100,)), TensorSpec("b", "int8", (150_000,))]
    _, whole = plan_of("whole", tensors)
    assert len(whole.extents) == 3  # 150,100 B in extents of at most 64 KiB
    _, per_tensor = plan_of("tensor", tensors)
    assert len(per_tensor.extents) == 1 + 3  # a; b cut in 3
    assert [len(e.pieces) for e in per_tensor.extents] == [1, 1, 1, 1]
    _, fixed = plan_of(4096 * 4, tensors)
    assert all(e.payload_bytes == 4096 * 4 for e in fixed.extents[:-1])
    for bad in (100, 4096 + 1, 128 * 1024, "nonsense"):
        with pytest.raises(ValueError):
            plan_of(bad, tensors)
    with pytest.raises(ValueError, match="no bytes"):
        plan_of("whole", [TensorSpec("a", "int8", (0,))])


def test_extent_cap_never_exceeds_the_pinned_chunk():
    spec = ComponentSpec("c", (TensorSpec("a", "int8", (10,)),), "whole")
    with pytest.raises(ValueError, match="max_extent_bytes"):
        CatalogLayout(
            ComponentSchema((spec,), max_extent_bytes=CHUNK_BYTES * 2), 1, "x"
        )
    CatalogLayout(ComponentSchema((spec,), max_extent_bytes=CHUNK_BYTES), 1, "x")


def test_stripes_are_balanced_by_bytes_with_mixed_sizes():
    comps = tuple(
        ComponentSpec(f"c{i}", (TensorSpec("w", "int8", (n,)),), 64 * 1024)
        for i, n in enumerate([1 << 20, 3 << 20, 10_000, 600_000, 2 << 20])
    )
    layout = CatalogLayout(ComponentSchema(comps), 3, IDENT)
    spread = max(layout.stripe_bytes) - min(layout.stripe_bytes)
    assert spread <= 64 * 1024
    # offsets inside a stripe never overlap
    spans = {}
    for i in range(layout.total_units):
        comp, unit = layout.key_of(i)
        stripe, offset, size = layout.locate(comp, unit)
        spans.setdefault(stripe, []).append((offset, offset + size))
    for ranges in spans.values():
        ranges.sort()
        assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        assert ranges[0][0] == PAGE


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stripes", [1, 3])
def test_round_trip_mixed_tower_bit_exact(tmp_path, stripes):
    state = tower_state(300)
    spec = ComponentSpec.from_state_dict("tower", state, split=64 * MIB)
    dirs = dirs_of(tmp_path, stripes)
    write_components(dirs, ComponentSchema((spec,)), {"tower": state}, workers=3)
    with open_store(dirs, 256 * MIB, check_every=1) as store:
        comp = store["tower"]
        assert comp.num_extents == 5  # 301.x MiB in 64 MiB extents
        assert comp.state == "disk"
        loaded = comp.load(threads=4)
        assert comp.state == "host"
        assert set(loaded.tensors) == set(state)
        for name, tensor in state.items():
            assert same_bytes(loaded.tensors[name], tensor), name
        assert store.layout.slot_bytes <= 64 * MIB
        assert loaded.timings["bytes"] == comp.nbytes
        # the pool was closed after the call; nothing is held but the tensors
        assert store.stats()["tower"] is None


@pytest.mark.parametrize("split", ["whole", "tensor"])
def test_other_cuts_round_trip(tmp_path, split):
    state = tower_state(20)
    spec = ComponentSpec.from_state_dict("tower", state, split=split)
    dirs = dirs_of(tmp_path, 2)
    write_components(
        dirs, ComponentSchema((spec,), max_extent_bytes=8 * MIB), {"tower": state}
    )
    with open_store(dirs, 32 * MIB, check_every=1) as store:
        extents = store["tower"].num_extents
        assert extents >= 3  # a "unit" above the cap is several extents
        loaded = store["tower"].load()
        assert all(same_bytes(loaded.tensors[n], t) for n, t in state.items())
    if split == "tensor":
        # the 16 MiB tensor is cut in 2, the 20 MiB one in 3
        assert extents == len(state) + 3


def test_load_into_given_destinations(tmp_path):
    state = tower_state(4)
    spec = ComponentSpec.from_state_dict("tower", state, split=MIB)
    dirs = dirs_of(tmp_path, 1)
    write_components(dirs, ComponentSchema((spec,)), {"tower": state})
    dest = {n: torch.zeros_like(t) for n, t in state.items()}
    with open_store(dirs, 8 * MIB) as store:
        loaded = store["tower"].load(dest)
        assert loaded.tensors["mask"] is dest["mask"]
        assert all(same_bytes(dest[n], t) for n, t in state.items())
        assert not loaded.owned
        bad = dict(dest, **{"mask": torch.zeros(14, dtype=torch.bool)})
        with pytest.raises(ValueError, match="mask"):
            store["tower"].load(bad)
        with pytest.raises(ValueError, match="needs tensors"):
            store["tower"].load({"mask": dest["mask"]})


def test_numpy_destinations(tmp_path):
    state = {"w": rand_tensor(torch.Generator().manual_seed(8), torch.uint8, (5000,))}
    spec = ComponentSpec.from_state_dict("c", state, split=4096)
    dirs = dirs_of(tmp_path, 1)
    write_components(dirs, ComponentSchema((spec,)), {"c": state})
    with open_store(dirs, 1 << 20) as store:
        out = np.zeros(5000, dtype=np.uint8)
        store["c"].load({"w": out})
        assert out.tobytes() == state["w"].numpy().tobytes()
        with pytest.raises(ValueError, match="matching array"):
            store["c"].load({"w": np.zeros(4999, dtype=np.uint8)})


def test_extents_iterator_leases_and_releases(tmp_path):
    state = tower_state(6)
    spec = ComponentSpec.from_state_dict("tower", state, split=MIB)
    dirs = dirs_of(tmp_path, 2)
    write_components(dirs, ComponentSchema((spec,)), {"tower": state})
    names = [t.name for t in spec.tensors]
    seen = {n: bytearray(state[n].numel() * state[n].element_size()) for n in names}
    with open_store(dirs, 4 * MIB) as store:
        comp = store["tower"]
        count = 0
        for extent in comp.extents(prefetch=3):
            with extent:
                for name, start, view in extent.pieces():
                    seen[name][start : start + len(view)] = bytes(view)
            count += 1
        assert count == comp.num_extents
        stats = comp.pool_stats()
        assert stats["slots_leased"] == 0
        comp.close_pool()
        assert comp.pool_stats() is None
    for name in names:
        want = state[name].reshape(-1).view(torch.uint8).numpy().tobytes()
        assert bytes(seen[name]) == want


def test_per_group_pools_are_sized_by_their_own_extents(tmp_path):
    small_state = {
        "a": rand_tensor(torch.Generator().manual_seed(1), torch.bfloat16, (5 * MIB,))
    }  # 10 MiB
    big_state = {
        "b": rand_tensor(torch.Generator().manual_seed(2), torch.int8, (200 * MIB,))
    }
    small = ComponentSpec.from_state_dict("small", small_state, split=2560 * 1024)
    big = ComponentSpec.from_state_dict("big", big_state, split=100 * MIB)
    schema = ComponentSchema((small, big), max_extent_bytes=128 * MIB)
    dirs = dirs_of(tmp_path, 2)
    write_components(dirs, schema, {"small": small_state, "big": big_state})
    with open_store(dirs, 256 * MIB) as store:
        slots = store.slot_bytes()
        assert slots == {"small": 2560 * 1024, "big": 100 * MIB}
        assert store.stats() == {"small": None, "big": None}  # nothing allocated yet
        sub_small = store["small"]._pool_acquire()
        try:
            stats = sub_small.stats()
            assert stats["pool_bytes"] == 4 * 2560 * 1024  # all of it, nothing more
            assert stats["slots"] == 4
            assert store.stats()["big"] is None  # the big pool is not there
        finally:
            store["small"]._pool_release()
        sub_big = store["big"]._pool_acquire()
        try:
            assert sub_big.stats()["slots"] == 2
            assert sub_big.stats()["pool_bytes"] == 200 * MIB
        finally:
            store["big"]._pool_release()
        # one shared slot size (the expert store's rule) would be this much:
        shared = 4 * 100 * MIB  # the small component's 4 slots at the big slot size
        assert shared / (4 * 2560 * 1024) > 30
        loaded = store["small"].load()
        assert same_bytes(loaded.tensors["a"], small_state["a"])
        assert same_bytes(store["big"].load().tensors["b"], big_state["b"])


def test_release_to_disk_and_back(tmp_path):
    state = tower_state(8)
    spec = ComponentSpec.from_state_dict("tower", state, split=MIB)
    dirs = dirs_of(tmp_path, 2)
    write_components(dirs, ComponentSchema((spec,)), {"tower": state})
    with open_store(dirs, 8 * MIB) as store:
        comp = store["tower"]
        first = comp.load(keep_pool=True)
        assert comp.state == "host" and comp.load() is first  # idempotent
        assert comp.pool_stats() is not None  # kept for the next call
        comp.release()
        assert comp.state == "disk"
        assert first.tensors == {} and first.released
        again = comp.load()
        assert again is not first and comp.state == "host"
        assert all(same_bytes(again.tensors[n], t) for n, t in state.items())
        for key in ("alloc_s", "pool_open_s", "read_copy_s", "total_s", "mib_s"):
            assert again.timings[key] >= 0
        assert comp.last_timings is again.timings
        with comp.load(pin=False):
            pass
        assert comp.state == "disk"  # leaving the block released it


def test_corrupt_extent_is_named_and_leaves_no_lease(tmp_path):
    state = tower_state(4)
    spec = ComponentSpec.from_state_dict("tower", state, split=MIB)
    dirs = dirs_of(tmp_path, 2)
    write_components(dirs, ComponentSchema((spec,)), {"tower": state})
    layout = ebuild._load_layout(ebuild._manifest(dirs))
    stripe, offset, _ = layout.locate(0, 2)
    with open(os.path.join(dirs[stripe], DATA_FILE), "r+b") as handle:
        handle.seek(offset + 9)
        handle.write(b"\xff")
    with open_store(dirs, 4 * MIB, check_every=1) as store:
        with pytest.raises(StoreCorruptError, match=r"\(0, 2\)"):
            store["tower"].load(keep_pool=True)
        assert store["tower"].state == "disk"
        stats = store["tower"].pool_stats()
        assert stats["slots_leased"] == 0 and stats["check_failures"] == 1
    with open_store(dirs, 4 * MIB) as store:  # unchecked: no error
        store["tower"].load()


def test_stores_of_the_two_kinds_refuse_each_other(tmp_path):
    state = {"w": rand_tensor(torch.Generator().manual_seed(3), torch.int8, (4096,))}
    spec = ComponentSpec.from_state_dict("c", state)
    comp_dirs = dirs_of(tmp_path, 1)
    write_components(comp_dirs, ComponentSchema((spec,)), {"c": state})
    with pytest.raises(StoreMismatchError, match="takes"):
        ExpertStore.open(comp_dirs, IDENT, 1 << 20, pin=False)
    with pytest.raises(StoreMismatchError, match="identity"):
        ComponentStore.open(comp_dirs, "other", 1 << 20, pin=False)
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    source = ebuild.SyntheticExperts(layers=1, experts=2)
    ebuild.build_source([str(expert_dir)], source, finalize=True)
    with pytest.raises(StoreMismatchError):
        ComponentStore.open([str(expert_dir)], source.identity, 1 << 20, pin=False)
    assert isinstance(source.schema, StoreSchema)


def test_incomplete_store_is_not_opened_and_writer_resumes(tmp_path):
    states = {
        "one": {"w": rand_int8(4, 50_000)},
        "two": {"w": rand_int8(5, 70_000)},
    }
    schema = ComponentSchema(
        tuple(ComponentSpec.from_state_dict(n, s, 16384) for n, s in states.items())
    )
    dirs = dirs_of(tmp_path, 2)
    writer = ComponentStoreWriter(dirs, schema, IDENT, writer_id="w")
    assert writer.write_component("one", states["one"]) == 4
    writer.flush()
    del writer  # killed after the flush
    with pytest.raises(StoreIncompleteError):
        ComponentStore.open(dirs, IDENT, 1 << 20, pin=False)
    with ComponentStoreWriter(dirs, schema, IDENT, writer_id="w") as writer:
        assert writer.has_component("one") and not writer.has_component("two")
        assert writer.write_component("one", states["one"]) == 0  # nothing to redo
        assert writer.write_component("two", states["two"]) == 5
        with pytest.raises(ValueError, match="no component"):
            writer.write_component("three", {})
        with pytest.raises(ValueError, match="needs tensors"):
            writer.write_component("two", {})
        with pytest.raises(ValueError, match="dtype"):
            writer.write_component("two", {"w": torch.zeros(70_000, dtype=torch.int16)})
        writer.finalize()
    with open_store(dirs, MIB, check_every=1) as store:
        for name, state in states.items():
            assert same_bytes(store[name].load().tensors["w"], state["w"])


def test_host_allocation_plan():
    specs = [
        TensorSpec("a", "bfloat16", (1000, 7)),
        TensorSpec("b", "int8", (13,)),
        TensorSpec("c", "float32", (5, 5)),
    ]
    tensors, owners = allocate_host(specs, pin=False)
    assert owners == []
    assert {n: (t.dtype, tuple(t.shape)) for n, t in tensors.items()} == {
        "a": (torch.bfloat16, (1000, 7)),
        "b": (torch.int8, (13,)),
        "c": (torch.float32, (5, 5)),
    }
    with pytest.raises(ValueError, match="cannot be pinned"):
        allocate_host([TensorSpec("huge", "int8", (CHUNK_BYTES + 1,))], pin=True)
    try:
        tensors, owners = allocate_host(specs, pin=True)
    except (RuntimeError, ImportError):
        pytest.skip("no accelerator to pin host memory with")
    assert all(t.is_pinned() for t in tensors.values())
    assert all(o.numel() <= CHUNK_BYTES for o in owners)
