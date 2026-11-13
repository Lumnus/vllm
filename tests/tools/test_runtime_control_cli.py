# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runtime control client maps each command to its operation."""

import pytest

from tools.runtime_control import _op_and_args, _parser, _print_state


@pytest.mark.parametrize(
    "argv, op, args",
    [
        (["eplb", "pause"], "eplb.pause", {}),
        (["eplb", "rearrange"], "eplb.rearrange", {}),
        (
            ["eplb", "set", "--step-interval", "300", "--max-moves", "4"],
            "eplb.set",
            {"step_interval": 300, "max_moves_per_rank": 4},
        ),
        (["eplb", "save"], "eplb.save", {"source": "last"}),
        (["eplb", "load", "--path", "/s.json"], "eplb.load", {"path": "/s.json"}),
        (["store", "scrub", "--moved-only"], "store.scrub", {"full": False}),
        (["store", "set", "--check-every", "1"], "store.set", {"check_every": 1}),
        (
            ["tiers", "set", "--host-tier-bound", "8"],
            "tiers.set",
            {"host_tier_bound": 8},
        ),
        (
            [
                "cpu",
                "set",
                "--threads",
                "6",
                "--cache-gib",
                "8",
                "--promote-misses",
                "4",
            ],
            "cpu.set",
            {"cpu_threads": 6, "host_cache_gib": 8.0, "host_promote_misses": 4},
        ),  # fmt: skip
        (["cpu", "stats"], "cpu.stats", {}),
        (
            ["kv", "resize", "2000", "--deadline", "30", "--preempt"],
            "kv.resize",
            {"num_blocks": 2000, "deadline_s": 30.0, "preempt": True},
        ),
        (["kv", "fit"], "kv.fit", {}),
        (
            ["kv", "set", "--release-chunk-mib", "none", "--grow-floor-mib", "1024"],
            "kv.set",
            {"release_chunk_mib": None, "grow_floor_mib": 1024},
        ),
        (["tower", "promote"], "tower.promote", {}),
        (
            ["tower", "set", "--name", "vt", "--demote-idle", "60"],
            "tower.set",
            {"name": "vt", "demote_idle_s": 60.0},
        ),
        (
            ["trace", "on", "--path", "/t", "--interval", "20"],
            "trace.set",
            {"enabled": True, "path": "/t", "interval": 20},
        ),
        (["trace", "off"], "trace.set", {"enabled": False}),
        (
            ["ple", "set", "--stats-interval", "60"],
            "ple.set",
            {"stats_interval_s": 60.0},
        ),
        (
            ["fault", "arm", "--stage", "load", "--rank", "1"],
            "fault.arm",
            {"stage": "load", "count": 1, "rank": 1},
        ),
        (["fault", "clear"], "fault.clear", {}),
    ],
)
def test_commands_map_to_operations(argv, op, args):
    assert _op_and_args(_parser().parse_args(argv)) == (op, args)


def test_the_state_prints(capsys):
    _print_state(
        {
            "kv": {"num_blocks": 10, "resize": {"state": "idle"}},
            "towers": {},
            "ranks": [
                {
                    "rank": 0,
                    "ep_rank": 0,
                    "eplb": {
                        "settings": {
                            "policy": "incremental",
                            "step_interval": 300,
                            "window_size": 100,
                            "max_moves_per_rank": 8,
                        },
                        "window_fill": 3,
                        "step": 5,
                        "steps_until_next": 295,
                        "paused": True,
                        "rearrangements": 2,
                        "skipped": 0,
                        "in_flight": None,
                        "models": [],
                    },
                    "kv": None,
                }
            ],
            "ranks_agree": {"eplb_step": False},
        }
    )
    out = capsys.readouterr().out
    assert "PAUSED" in out and "RANKS DISAGREE" in out
