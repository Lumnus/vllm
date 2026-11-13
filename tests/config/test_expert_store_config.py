# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``ExpertStoreConfig`` and ``--expert-store-config``."""

import dataclasses
from types import SimpleNamespace

import pytest

from vllm.config import ExpertStoreConfig, VllmConfig
from vllm.config.offload import OffloadConfig, UVAOffloadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _args(*argv: str) -> EngineArgs:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(list(argv)))


def test_off_by_default():
    assert ExpertStoreConfig().enabled is False
    field = {f.name: f for f in dataclasses.fields(VllmConfig)}["expert_store_config"]
    assert field.default is None
    assert _args().expert_store_config is None


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--expert-store-config",
            '{"dirs": ["/a", "/b"], "pool_gib": 8, "spare_rows": 4}',
        ],
        [
            "--expert-store-config.dirs",
            '["/a", "/b"]',
            "--expert-store-config.pool_gib",
            "8",
            "--expert-store-config.spare_rows",
            "4",
        ],
    ],
)
def test_cli(argv):
    config = _args(*argv).expert_store_config
    assert config is not None and config.enabled
    assert config.dirs == ["/a", "/b"] and config.pool_gib == 8
    assert config.spare_rows == 4
    # the documented defaults
    assert (config.io_threads, config.direct, config.build) == (16, True, "auto")
    assert (config.verify, config.read_mode, config.decay_interval) == (
        False,
        "auto",
        256,
    )
    assert config.stats_interval_s == 0.0
    assert config.pool_pinned is False
    assert (config.check_every, config.check_threads) == (8, 2)


def test_validation():
    with pytest.raises(ValueError, match="distinct"):
        ExpertStoreConfig(dirs=["/a", "/a"])
    with pytest.raises(ValueError):
        ExpertStoreConfig(dirs=["/a"], build="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExpertStoreConfig(dirs=["/a"], read_mode="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExpertStoreConfig(dirs=["/a"], pool_gib=0)


def _parallel(**eplb):
    eplb.setdefault("policy", "incremental")
    eplb.setdefault("use_async", True)
    eplb.setdefault("max_moves_per_rank", 8)
    return SimpleNamespace(enable_elastic_ep=False, eplb_config=SimpleNamespace(**eplb))


def _offload(per_expert=True):
    return OffloadConfig(
        uva=UVAOffloadConfig(cpu_offload_gb=4, cpu_offload_per_expert=per_expert)
    )


def test_needs_the_tiers_a_bounded_policy_and_async_eplb():
    config = ExpertStoreConfig(dirs=["/a"])
    config.verify_parallel_config(_parallel(), _offload())
    ExpertStoreConfig().verify_parallel_config(
        _parallel(policy="default"), _offload(False)
    )
    with pytest.raises(ValueError, match="cpu-offload-per-expert"):
        config.verify_parallel_config(_parallel(), _offload(False))
    with pytest.raises(ValueError, match="incremental"):
        config.verify_parallel_config(_parallel(policy="default"), _offload())
    with pytest.raises(ValueError, match="use_async"):
        config.verify_parallel_config(_parallel(use_async=False), _offload())
    elastic = _parallel()
    elastic.enable_elastic_ep = True
    with pytest.raises(ValueError, match="elastic"):
        config.verify_parallel_config(elastic, _offload())
    staged = _parallel()
    staged.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline"):
        config.verify_parallel_config(staged, _offload())
    with pytest.raises(ValueError, match="spare_rows=2"):
        ExpertStoreConfig(dirs=["/a"], spare_rows=2).verify_parallel_config(
            _parallel(), _offload()
        )


def test_spare_rows_default_to_the_moves_one_rearrangement_makes():
    assert ExpertStoreConfig(dirs=["/a"]).spare_rows_for(8) == 8
    assert ExpertStoreConfig(dirs=["/a"], spare_rows=12).spare_rows_for(8) == 12


def test_only_the_shape_of_the_rows_is_a_compile_factor():
    a = ExpertStoreConfig(dirs=["/a"], pool_gib=1, build="never").compile_factors()
    b = ExpertStoreConfig(dirs=["/b", "/c"], pool_gib=9, io_threads=2).compile_factors()
    assert a == b
    assert ExpertStoreConfig(dirs=["/a"], spare_rows=3).compile_factors() != a


def test_the_card_tier_settings_leave_the_hash_alone_until_set():
    """Unset, the factors are the ones before the settings existed (the
    same keys); fast_rows_max, the capacity the kernels see, enters when
    set; the run-time settings never do."""
    base = ExpertStoreConfig(dirs=["/a"]).compile_factors()
    assert not any(k.startswith(("fast_", "pinned_release")) for k in base)
    tuned = ExpertStoreConfig(
        dirs=["/a"], fast_rows_min=2, fast_chunk_mib=8, pinned_release=True
    ).compile_factors()
    assert tuned == base
    sized = ExpertStoreConfig(dirs=["/a"], fast_rows_max=64).compile_factors()
    assert sized == {**base, "fast_rows_max": 64}
    assert ExpertStoreConfig(dirs=["/a"], fast_rows_max=64).fast_tier_resizable
    assert not ExpertStoreConfig(dirs=["/a"]).fast_tier_resizable
    assert not ExpertStoreConfig(fast_rows_max=64).fast_tier_resizable  # no store


def test_the_card_tier_floor_is_below_its_capacity():
    tiers = SimpleNamespace(cpu_offload_per_expert=True)
    offload = OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=4, **vars(tiers)))
    eplb = SimpleNamespace(policy="incremental", use_async=True, max_moves_per_rank=8)
    parallel = SimpleNamespace(
        enable_elastic_ep=False, pipeline_parallel_size=1, eplb_config=eplb
    )
    ExpertStoreConfig(
        dirs=["/a"], fast_rows_max=8, fast_rows_min=2
    ).verify_parallel_config(parallel, offload)
    with pytest.raises(ValueError, match="fast_rows_min=9 is above fast_rows_max=8"):
        ExpertStoreConfig(
            dirs=["/a"], fast_rows_max=8, fast_rows_min=9
        ).verify_parallel_config(parallel, offload)
    with pytest.raises(ValueError):
        ExpertStoreConfig(dirs=["/a"], fast_rows_max=-1)
