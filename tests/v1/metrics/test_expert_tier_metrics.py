# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU expert tier's counters as metrics: the engine core's poller reads
them from the workers through the runtime control surface, the scheduler's
stats carry them, and `vllm:expert_tier_<key>` gauges (one series per rank)
show them, in the family of the encoder towers' gauges."""

from types import SimpleNamespace

import prometheus_client
import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.config.observability import ObservabilityConfig
from vllm.v1.engine.runtime_control import (
    EXPERT_TIER_GAUGES,
    ExpertTierStatsPoller,
)
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test


def build_logger_config(observability_config: ObservabilityConfig) -> VllmConfig:
    # the model of test_histogram_buckets.py
    return VllmConfig(
        model_config=ModelConfig(model="distilbert/distilgpt2"),
        observability_config=observability_config,
    )


def _engine(host_experts: bool, log_stats: bool = True):
    calls = []

    def collective_rpc(method, timeout=None, args=()):
        calls.append((method, args))
        return [
            {
                "rank": 0,
                "cpu": {
                    "counters": dict.fromkeys(
                        [
                            "routed_card",
                            "routed_pinned",
                            "routed_cpu",
                            "layers_with_cpu_calls",
                            "layers_without",
                            "cpu_experts",
                            "cache_hits",
                            "cache_misses",
                            "cache_prefetched",
                            "cache_evicted",
                            "promotions",
                            "demotions",
                        ],
                        3,
                    )
                    | {"compute_ms": 1500.0, "read_ms": 0.0, "wait_ms": 2.0},
                    "cache": {"rows": 64},
                    "cpu_threads": 8,
                },
                "tiers": {
                    "settings": {"host_tier_bound": 22},
                    "slots_now": {"card": 98, "pinned": 22, "cpu": 136},
                },
            }
        ]

    uva = SimpleNamespace(host_experts_on=lambda: host_experts)
    return SimpleNamespace(
        vllm_config=SimpleNamespace(offload_config=SimpleNamespace(uva=uva)),
        scheduler=SimpleNamespace(log_stats=log_stats, expert_tier_stats=None),
        collective_rpc=collective_rpc,
        calls=calls,
    )


def test_the_poller_exists_only_with_cpu_experts_and_stats():
    assert ExpertTierStatsPoller.maybe_create(_engine(False)) is None
    assert ExpertTierStatsPoller.maybe_create(_engine(True, log_stats=False)) is None
    assert ExpertTierStatsPoller.maybe_create(_engine(True)) is not None


def test_the_poller_reads_the_workers_at_its_interval():
    engine = _engine(True)
    poller = ExpertTierStatsPoller.maybe_create(engine)
    now = [100.0]
    poller._clock = lambda: now[0]
    poller.maybe_poll()
    poller.maybe_poll()  # within the interval: no RPC
    assert len(engine.calls) == 1
    method, args = engine.calls[0]
    assert method == "runtime_control" and args == (
        "state",
        "",
        {"only": ["cpu", "tiers"]},
    )
    series = engine.scheduler.expert_tier_stats["0"]
    assert set(series) == set(EXPERT_TIER_GAUGES)
    assert series["cpu_compute_seconds"] == 1.5 and series["slots_cpu"] == 136.0
    now[0] += ExpertTierStatsPoller.INTERVAL_S
    poller.maybe_poll()
    assert len(engine.calls) == 2


def test_the_gauges_carry_the_series_per_rank():
    assert SchedulerStats().expert_tier_stats == {}
    config = build_logger_config(ObservabilityConfig())
    try:
        logger = PrometheusStatLogger(config)
        families = {m.name for m in prometheus_client.REGISTRY.collect()}
        assert {f"vllm:expert_tier_{k}" for k in EXPERT_TIER_GAUGES} <= families
        # no series before a rank reports
        assert _samples("vllm:expert_tier_routed_cpu") == []
        stats = SchedulerStats(
            expert_tier_stats={"1": {"routed_cpu": 7.0, "pinned_bound": 22.0}}
        )
        logger.record(scheduler_stats=stats, iteration_stats=None)
        samples = _samples("vllm:expert_tier_routed_cpu")
        assert [(s.labels["rank"], s.value) for s in samples] == [("1", 7.0)]
        assert _samples("vllm:expert_tier_cpu_threads") == []  # not reported
    finally:
        unregister_vllm_metrics()


def _samples(name: str) -> list:
    return [
        s
        for m in prometheus_client.REGISTRY.collect()
        for s in m.samples
        if s.name == name
    ]
