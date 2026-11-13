# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runtime control client: the governor, drafter, scheduler and request
default commands."""

import pytest

from tools.runtime_control import _op_and_args, _parser, _print_state


@pytest.mark.parametrize(
    "argv, op, args",
    [
        (["governor", "state"], "governor.state", {}),
        (["governor", "set", "--mode", "dry_run"], "governor.set", {"mode": "dry_run"}),
        (
            [
                "governor",
                "set",
                "--set",
                "usage_high=0.85",
                "--set",
                "fast_min_slots=null",
                "--set",
                "fast_tier_setting=fast_slots",
            ],
            "governor.set",
            {
                "usage_high": 0.85,
                "fast_min_slots": None,
                "fast_tier_setting": "fast_slots",
            },
        ),  # fmt: skip
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


def test_a_bad_set_item_is_bad_usage():
    with pytest.raises(SystemExit):
        _op_and_args(_parser().parse_args(["governor", "set", "--set", "nope"]))


def test_the_state_prints_the_new_parts(capsys):
    _print_state(
        {
            "kv": {"num_blocks": 10, "resize": {"state": "idle"}},
            "towers": {},
            "governor": {
                "mode": "dry_run",
                "level": "pressure",
                "inputs": {"used": 0.93, "waiting": 3, "hot": 0.8, "cool": 0.0},
                "counts": {"reclaims": 1},
            },
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
    assert "governor: dry_run, level pressure" in out
    assert "drafter dflash: 8 of 15 tokens" in out and "acceptance 0.410" in out
    assert "active seqs 4 of 16" in out
