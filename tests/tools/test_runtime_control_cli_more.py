# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runtime control client: serving profiles, the tower's policy, the
drafter, scheduler and request default commands."""

import json

import pytest

from tools.runtime_control import _op_and_args, _parser, _print_schema, _print_state


def test_the_schema_prints_one_line_per_parameter(capsys):
    _print_schema(
        {
            "params": [
                {
                    "name": "sched.max_num_active_seqs",
                    "value": 4,
                    "live": True,
                    "verb": "sched.set",
                    "min": 1,
                },
                {
                    "name": "sched.max_num_seqs",
                    "value": 4,
                    "live": False,
                    "restart_reason": "graphs",
                },
            ]
        }
    )
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "sched.max_num_active_seqs = 4 [1..]; live via sched.set",
        "sched.max_num_seqs = 4; restart: graphs",
    ]


@pytest.mark.parametrize(
    "argv, op, args",
    [
        (["profile", "list"], "profile.list", {}),
        (["profile", "state"], "profile.state", {}),
        (["profile", "get", "long-sessions"], "profile.get", {"name": "long-sessions"}),
        (
            ["profile", "apply", "long-sessions"],
            "profile.apply",
            {"name": "long-sessions"},
        ),
        (
            ["tower", "set", "--policy", "host", "--disk-idle", "1800"],
            "tower.set",
            {"policy": "host", "disk_idle_s": 1800.0},
        ),
        (["tower", "set", "--disk-idle", "none"], "tower.set", {"disk_idle_s": None}),
        (["spec", "set", "--tokens", "2"], "spec.set", {"num_speculative_tokens": 2}),
        (["spec", "set", "--off"], "spec.set", {"enabled": False}),
        (
            ["spec", "set", "--on", "--reset-stats"],
            "spec.set",
            {"enabled": True, "reset_stats": True},
        ),
        (
            [
                "sched",
                "set",
                "--max-active-seqs",
                "4",
                "--max-scheduled-tokens",
                "2048",
            ],
            "sched.set",
            {"max_num_active_seqs": 4, "max_num_scheduled_tokens": 2048},
        ),
        (["defaults", "state"], "defaults.state", {}),
        (
            ["debug", "set", "eplb", "--count", "10"],
            "debug.set",
            {"part": "eplb", "on": True, "count": 10},
        ),
        (["debug", "set", "cpu", "--off"], "debug.set", {"part": "cpu", "on": False}),
        (["debug", "state"], "debug.state", {}),
        (
            ["sched", "set", "--long-prefill", "512"],
            "sched.set",
            {"long_prefill_token_threshold": 512},
        ),
        (
            [
                "defaults",
                "set",
                "--temperature",
                "0.7",
                "--presence-penalty",
                "1.5",
                "--thinking-budget",
                "none",
                "--cjk-mask",
                "off",
            ],
            "defaults.set",
            {
                "temperature": 0.7,
                "presence_penalty": 1.5,
                "thinking_token_budget": None,
                "cjk_mask_default": "off",
            },
        ),  # fmt: skip
    ],
)
def test_commands_map_to_operations(argv, op, args):
    assert _op_and_args(_parser().parse_args(argv)) == (op, args)


def test_profile_define_reads_the_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"name": "p", "kv": {"fit": True}}))
    op, args = _op_and_args(_parser().parse_args(["profile", "define", str(path)]))
    assert op == "profile.define" and args == {"name": "p", "kv": {"fit": True}}


def test_the_state_prints_the_new_parts(capsys):
    _print_state(
        {
            "kv": {"num_blocks": 10, "resize": {"state": "idle"}},
            "towers": {},
            "kv_demand": {
                "used": 0.93,
                "cached_free_blocks": 12,
                "cached_free": 0.05,
                "running": 4,
                "waiting": 3,
                "preemptions": 2,
                "prefix_cache_hit_rate": 0.4,
            },
            "profile": {"active": "long-sessions", "applying": None, "defined": ["a"]},
            "spec": {
                "drafter": True,
                "method": "dflash",
                "num_speculative_tokens": 8,
                "num_speculative_tokens_max": 15,
                "decode_graph_tokens": [15],
                "acceptance": {"acceptance_rate": 0.41, "mean_acceptance_length": 4.2},
            },
            "sched": {
                "max_num_active_seqs": 4,
                "max_num_seqs": 16,
                "max_num_scheduled_tokens": 2048,
                "max_num_batched_tokens": 8192,
            },
            "ranks": [],
        }
    )
    out = capsys.readouterr().out
    assert "KV demand: used 0.93, cached-free 12 blocks" in out
    assert "profile: active long-sessions" in out
    assert "drafter dflash: 8 of 15 tokens" in out and "acceptance 0.410" in out
    assert "active seqs 4 of 16" in out
