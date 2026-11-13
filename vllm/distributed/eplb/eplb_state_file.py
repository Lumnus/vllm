# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert loads EPLB keeps across restarts (`EPLBConfig.state_path`).

The file maps each MoE layer's name to the load per logical expert of the
window the last rearrangement used: `{"<layer name>": [int, ...], ...}`.
Layer names are global, so the pipeline stages, each with its own EPLB state,
merge their layers into one file under a lock.
"""

import fcntl
import json
import os
import re
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse(raw: Any, path: str) -> dict[str, list[int]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of layer name -> loads")
    return {str(k): [int(x) for x in v] for k, v in raw.items()}


def read_expert_loads(path: str) -> dict[str, list[int]]:
    """The saved loads, or none when the file is missing or unreadable: they
    only decide where experts start, so a bad file must not fail the start."""
    try:
        with open(path) as handle:
            return _parse(json.load(handle), path)
    except FileNotFoundError:
        logger.info("EPLB: no saved expert loads at %s yet", path)
    except (OSError, ValueError, TypeError) as err:
        logger.warning("EPLB: ignoring the saved expert loads in %s: %s", path, err)
    return {}


def write_expert_loads(path: str, layers: dict[str, list[int]]) -> None:
    """Replace `layers` in the file, keeping the other layers, atomically."""
    with open(f"{path}.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # released when the file closes
        merged: dict[str, list[int]] = {}
        try:
            with open(path) as handle:
                merged = _parse(json.load(handle), path)
        except (OSError, ValueError, TypeError):
            pass
        merged.update(layers)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as handle:
            json.dump(dict(sorted(merged.items())), handle)
        os.replace(tmp, path)


_LAYER_NUMBER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def read_seed_loads(
    path: str, layer_names: list[str], num_logical: int
) -> dict[str, list[int]]:
    """Loads to start from when EPLB has none of its own (the expert store's
    ``tier_seed``), per MoE layer name. Three shapes are read:

    - an EPLB state file: layer name -> load per logical expert (as written
      by `write_expert_loads`, or from a routing trace by
      ``tools/analyze_expert_trace.py --write-loads``);
    - a rankings file: decoder layer number -> logical experts, hottest
      first; an expert at position i of n gets load n - i, unlisted ones 0;
    - a learned-state file: ``{"layers": {layer number: {expert: score}}}``;
      scores are scaled to integers (the hottest 10^6).

    A layer whose loads do not cover ``num_logical`` experts is left out. A
    missing or unreadable file gives none (a seed only decides where experts
    start)."""
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as err:
        logger.warning("EPLB: ignoring the tier seed %s: %s", path, err)
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("layers"), dict):
        raw = {
            k: {int(e): float(s) for e, s in v.items()}
            for k, v in raw["layers"].items()
        }
    if not isinstance(raw, dict):
        logger.warning("EPLB: ignoring the tier seed %s: not a JSON object", path)
        return {}
    by_number: dict[int, str] = {}
    for name in layer_names:
        match = _LAYER_NUMBER.search(name)
        if match is not None:
            by_number[int(match.group(1))] = name
    out: dict[str, list[int]] = {}
    for key, value in raw.items():
        name = key if key in layer_names else None
        if name is None and str(key).isdigit():
            name = by_number.get(int(key))
        if name is None:
            continue
        loads = [0] * num_logical
        if isinstance(value, dict):  # learned scores
            top = max([float(s) for s in value.values()] + [0.0])
            for e, score in value.items():
                if 0 <= int(e) < num_logical and top > 0:
                    loads[int(e)] = round(float(score) / top * 1e6)
        elif isinstance(value, list) and key in layer_names:  # loads
            if len(value) != num_logical:
                logger.warning(
                    "EPLB: the tier seed %s has %d loads for %s, which has %d "
                    "logical experts; not used",
                    path,
                    len(value),
                    name,
                    num_logical,
                )
                continue
            loads = [int(x) for x in value]
        elif isinstance(value, list):  # rankings, hottest first
            n = len(value)
            for i, e in enumerate(value):
                if 0 <= int(e) < num_logical:
                    loads[int(e)] = max(loads[int(e)], n - i)
        else:
            continue
        out[name] = loads
    return out
