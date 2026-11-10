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
