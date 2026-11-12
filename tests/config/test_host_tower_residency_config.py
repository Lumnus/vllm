# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--mm-encoder-host-towers` with residency "dynamic": the options, their
defaults, and what is refused."""

from types import SimpleNamespace

import pytest

from tests.config.test_host_tower_config import _vllm_config
from vllm.config import VllmConfig
from vllm.config.multimodal import HostTowerOptions, MultiModalConfig
from vllm.config.utils import hash_config

pytestmark = pytest.mark.cpu_test


def test_the_short_format_and_the_default_stay_in_host_memory():
    short = MultiModalConfig(mm_encoder_host_towers={"vision_tower": 16})
    assert short.mm_encoder_host_towers["vision_tower"].residency == "host"
    assert HostTowerOptions().residency == "host"


def test_dynamic_options_and_their_defaults():
    config = MultiModalConfig(
        mm_encoder_host_towers={
            "vision_tower": {"threads": 16, "residency": "dynamic"},
        }
    )
    options = config.mm_encoder_host_towers["vision_tower"]
    assert (
        options.residency,
        options.promote_after,
        options.promote_window_s,
        options.demote_idle_s,
        options.idle_location,
        options.card_layout,
        options.card_reserve_mib,
        options.shrink_deadline_s,
    ) == ("dynamic", 2, 600.0, 1800.0, "host", "replicated", 128, 60.0)


def test_the_full_dynamic_format():
    config = MultiModalConfig(
        mm_encoder_host_towers={
            "vision_tower": {
                "threads": 16,
                "residency": "dynamic",
                "promote_after": 3,
                "promote_window_s": 120,
                "demote_idle_s": 180,
                "idle_location": "disk",
                "card_layout": "sharded",
                "card_reserve_mib": 64,
                "shrink_deadline_s": 0,
            }
        }
    )
    options = config.mm_encoder_host_towers["vision_tower"]
    assert options.promote_after == 3
    assert options.promote_window_s == 120
    assert options.demote_idle_s == 180
    assert options.idle_location == "disk"
    assert options.card_layout == "sharded"
    assert options.card_reserve_mib == 64
    assert options.shrink_deadline_s == 0


@pytest.mark.parametrize(
    "options",
    [
        {"residency": "card"},
        {"promote_after": 0},
        {"promote_window_s": 0},
        {"demote_idle_s": -1},
        {"idle_location": "ram"},
        {"card_layout": "data"},
        {"card_reserve_mib": -1},
    ],
)
def test_invalid_residency_options_are_refused(options):
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_host_towers={"vision_tower": options})


def test_residency_does_not_change_the_compile_hash():
    assert hash_config(MultiModalConfig()) == hash_config(
        MultiModalConfig(
            mm_encoder_host_towers={"vision_tower": {"residency": "dynamic"}}
        )
    )


def test_dynamic_needs_a_releasable_kv_cache():
    with pytest.raises(ValueError, match="kv-cache-release-chunk-mib"):
        _vllm_config({"vision_tower": {"residency": "dynamic"}})


def _deployment(tp_mode="weights", pp=1, dp=1, chunk=2):
    """What `_check_dynamic_host_towers` reads, without a platform that
    serves the extensible KV cache (this CPU run has none)."""
    mm_config = SimpleNamespace(mm_encoder_tp_mode=tp_mode)
    return SimpleNamespace(
        model_config=SimpleNamespace(get_multimodal_config=lambda: mm_config),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp, data_parallel_size=dp
        ),
        cache_config=SimpleNamespace(kv_cache_release_chunk_mib=chunk),
    )


def test_a_deployment_that_can_move_a_tower_passes():
    VllmConfig._check_dynamic_host_towers(_deployment(), "--flag", ["vision_tower"])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tp_mode": "data"}, "mm_encoder_tp_mode='weights'"),
        ({"pp": 2}, "pipeline parallelism"),
        ({"dp": 2}, "data parallelism"),
        ({"chunk": None}, "kv-cache-release-chunk-mib"),
    ],
)
def test_deployments_that_cannot_move_a_tower_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VllmConfig._check_dynamic_host_towers(
            _deployment(**kwargs), "--flag", ["vision_tower"]
        )


def test_host_residency_is_not_checked_for_the_kv_cache():
    config = _vllm_config({"vision_tower": {"threads": 4, "residency": "host"}})
    assert config.ec_transfer_config is not None
