# SPDX-License-Identifier: Apache-2.0
"""B70-V32-0054: loader for the 0048 fused decode MoE op (libb70_moe_decode.so, b70moe.moe_decode).

B70_MOE_DECODE_OP=<path> loads the library once; B70_MOE_DECODE_OP_MAX_ROWS (default 16, at most 16) bounds the
batches it takes. 0 = off."""
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_ROWS: list = [None]


def decode_op_max_rows() -> int:
    rows = _ROWS[0]
    if rows is None:
        rows, lib = 0, os.environ.get("B70_MOE_DECODE_OP", "0")
        if lib not in ("", "0"):
            try:
                torch.ops.load_library(lib)
                rows = min(16, int(os.environ.get("B70_MOE_DECODE_OP_MAX_ROWS", "16")))
                logger.info("B70 decode op: %s, batches of <= %d tokens", lib, rows)
            except (OSError, RuntimeError) as e:
                logger.warning("B70 decode op: cannot load %s (%s); stock path", lib, e)
        _ROWS[0] = rows
    return rows
