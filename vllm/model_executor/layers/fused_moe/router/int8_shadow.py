# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing agreement of an INT8 router gate against a BF16 shadow (debug).

``VLLM_ROUTER_INT8_SHADOW`` (default off) makes every checkpoint-INT8
``GateLinear`` also evaluate a BF16 router on the same input and count the
routed slots whose expert differs: per token, the experts in the INT8 top-k
that are not in the shadow's top-k. Values:

* ``dequant``: the shadow weight is the INT8 weight times its row scales, in
  the activation dtype. The difference is then the activation rounding and the
  INT8 GEMM alone.
* a checkpoint directory (a BF16 checkpoint with ``model.safetensors.index.json``):
  the shadow is that checkpoint's ``mlp.gate`` of the same layer, so the count
  is calibrated INT8 against the original router.

The counts accumulate on the device (graph-capturable in-place adds) and are
logged, rank 0 only, each time the first gate of the model runs outside graph
capture: one line per eager forward, and after graph replays one line for the
tokens they covered. A boot with ``--enforce-eager`` gives one line per
forward. The extra GEMM, two top-k and a host sync per report make this a
measurement switch, never a serving setting.
"""

from __future__ import annotations

import atexit
import json
import os
import re

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_SHADOWS: list[RouterInt8Shadow] = []


def _capturing() -> bool:
    # The XPU model runner points this at torch.xpu's check.
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no device runtime: never capturing
        return False


def _rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001 - no process group (tests, CPU tools)
        return True


def _layer_index(prefix: str) -> int:
    match = re.search(r"layers\.(\d+)\.", prefix)
    return int(match.group(1)) if match else -1


def dequantized_weight(layer: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """``[N, K]`` = int8 rows times their scales. The INT8 kernels keep the
    weight as ``[N, K]`` or as a ``[K, N]`` view; the layer's sizes decide."""
    weight = layer.weight
    n, k = layer.output_size, layer.input_size
    if tuple(weight.shape) == (k, n) and (n != k or not weight.is_contiguous()):
        weight = weight.t()
    if tuple(weight.shape) != (n, k):
        raise ValueError(
            f"{layer.prefix}: INT8 weight shape {tuple(layer.weight.shape)} is "
            f"neither [{n}, {k}] nor [{k}, {n}]"
        )
    scale = layer.weight_scale.reshape(-1, 1).float()
    return (weight.float() * scale).to(dtype)


def checkpoint_weight(directory: str, prefix: str) -> torch.Tensor:
    """The ``mlp.gate`` weight of ``prefix``'s layer in a checkpoint directory
    (an MTP gate only from MTP names, a target gate only from the others)."""
    from safetensors import safe_open

    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    layer = _layer_index(prefix)
    is_mtp = "mtp" in prefix.split(".")
    suffix = re.compile(rf"(^|\.)layers\.{layer}\.mlp\.gate\.weight$")
    names = [
        name
        for name in weight_map
        if suffix.search(name) and ("mtp" in name.split(".")) == is_mtp
    ]
    if len(names) != 1:
        raise ValueError(
            f"VLLM_ROUTER_INT8_SHADOW: {len(names)} router weights for layer "
            f"{layer} ({prefix}) in {index_path}: {names[:4]}"
        )
    with safe_open(
        os.path.join(directory, weight_map[names[0]]), framework="pt"
    ) as handle:
        return handle.get_tensor(names[0])


class RouterInt8Shadow:
    """Counts the routed slots where an INT8 gate and its BF16 shadow differ."""

    def __init__(self, source: str, top_k: int, prefix: str) -> None:
        self.source = source
        self.top_k = top_k
        self.prefix = prefix
        self.layer = _layer_index(prefix)
        self.weight: torch.Tensor | None = None
        # [differing slots, rows]
        self.counts: torch.Tensor | None = None
        _SHADOWS.append(self)
        _SHADOWS.sort(key=lambda s: s.layer)

    def _ensure(self, layer: torch.nn.Module, x: torch.Tensor) -> bool:
        if self.weight is not None:
            return True
        if _capturing():
            # Never allocate inside a capture; the profile run comes first.
            return False
        if self.source == "dequant":
            weight = dequantized_weight(layer, x.dtype)
        else:
            weight = checkpoint_weight(self.source, self.prefix)
        self.weight = weight.to(device=x.device, dtype=x.dtype)
        self.counts = torch.zeros(2, dtype=torch.int64, device=x.device)
        return True

    def observe(
        self, layer: torch.nn.Module, x: torch.Tensor, logits: torch.Tensor
    ) -> None:
        if not self._ensure(layer, x):
            return
        if _SHADOWS and _SHADOWS[0] is self and not _capturing():
            report()
        assert self.weight is not None and self.counts is not None
        rows = x.reshape(-1, x.shape[-1])
        ref = F.linear(rows.to(self.weight.dtype), self.weight)
        k = min(self.top_k, ref.shape[-1])
        got = torch.topk(logits.reshape(rows.shape[0], -1).float(), k).indices
        want = torch.topk(ref.float(), k).indices
        same = (got.unsqueeze(-1) == want.unsqueeze(-2)).any(-1).sum()
        self.counts[0].add_(rows.shape[0] * k - same)
        self.counts[1].add_(rows.shape[0])


def report() -> None:
    """Log and clear the counts of every shadow (one line)."""
    live = [s for s in _SHADOWS if s.counts is not None]
    if not live:
        return
    counts = torch.stack([s.counts for s in live]).cpu()
    for s in live:
        s.counts.zero_()
    rows = counts[:, 1]
    if int(rows.max()) == 0 or not _rank0():
        return
    diff = counts[:, 0]
    slots = int((rows * torch.tensor([s.top_k for s in live])).sum())
    worst = int(torch.argmax(diff))
    logger.info(
        "Router INT8 shadow (%s): %d of %d routed slots differ (%.4f%%) over %d "
        "tokens in %d layers; layers with differences %d; most in layer %d (%d).",
        live[0].source,
        int(diff.sum()),
        slots,
        100.0 * int(diff.sum()) / max(slots, 1),
        int(rows[0]),
        len(live),
        int((diff > 0).sum()),
        live[worst].layer,
        int(diff[worst]),
    )


def make_shadow(top_k: int, prefix: str) -> RouterInt8Shadow | None:
    """The shadow ``VLLM_ROUTER_INT8_SHADOW`` asks for, or None."""
    import vllm.envs as envs

    value = (envs.VLLM_ROUTER_INT8_SHADOW or "").strip()
    if value in ("", "0", "off", "false"):
        return None
    source = "dequant" if value in ("1", "dequant") else value
    if source != "dequant" and not os.path.isfile(
        os.path.join(source, "model.safetensors.index.json")
    ):
        raise ValueError(
            "VLLM_ROUTER_INT8_SHADOW must be 'dequant' or a checkpoint directory "
            f"with model.safetensors.index.json, got {value!r}"
        )
    logger.info_once(
        "Router INT8 shadow on (VLLM_ROUTER_INT8_SHADOW=%s): every INT8 router "
        "also runs a BF16 evaluation and counts differing routed slots.",
        source,
    )
    return RouterInt8Shadow(source, top_k, prefix)


def clear() -> None:
    """Forget every shadow (tests)."""
    _SHADOWS.clear()


atexit.register(lambda: report() if _SHADOWS else None)
