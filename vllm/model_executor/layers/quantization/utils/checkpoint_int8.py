# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 for small linears that only a calibrated checkpoint may quantize.

Router gates, shared-expert gates and similar small projections are built
without a quantization config in most models: rounded at load they would change
which experts run. A checkpoint can still carry them as calibrated INT8 W8A8
(compressed-tensors, per-channel int8 weights, dynamic per-token int8
activations). ``checkpoint_int8_quant_config`` hands such a layer the quant
config only when the checkpoint itself declares it INT8 W8A8. Otherwise the
layer is built unquantized, as before: an online ``targets`` entry or a
``--quantization`` method never rounds it at load.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

# The attribute a layer built through this helper carries: the part name, and
# whether it took the checkpoint's INT8 method.
PART_ATTR = "checkpoint_int8_part"
INT8_ATTR = "checkpoint_int8"


def _probe() -> torch.nn.Module:
    """An uninitialised linear whose class name matches no class-name target
    (``Linear``): only a target that names the layer (a path or a regex, or
    every shard of a fused layer) declares it INT8. A catch-all class target
    never pulls one of these layers in; checkpoints that mean it name it, and
    a catch-all that did quantize it would fail to load as before. The
    lookups only test the type and the class name, so nothing is allocated."""
    cls = _probe_class()
    return cls.__new__(cls)


@functools.cache
def _probe_class() -> type[torch.nn.Module]:
    from vllm.model_executor.layers.linear import LinearBase

    class _CheckpointNameProbe(LinearBase):
        pass

    return _CheckpointNameProbe


def _is_w8a8_int8(quant_config: QuantizationConfig, probe, prefix: str) -> bool:
    get_scheme_dict = getattr(quant_config, "get_scheme_dict", None)
    if get_scheme_dict is None:
        return False
    try:
        scheme = get_scheme_dict(probe, prefix)
    except ValueError:
        return False
    if not scheme:
        return False
    weights, acts = scheme.get("weights"), scheme.get("input_activations")
    if weights is None or acts is None:
        return False

    def is_int8(args) -> bool:
        kind = getattr(args, "type", None)
        kind = getattr(kind, "value", kind)
        return kind == "int" and int(getattr(args, "num_bits", 0)) == 8

    return is_int8(weights) and is_int8(acts) and bool(getattr(acts, "dynamic", False))


def checkpoint_int8_quant_config(
    quant_config: QuantizationConfig | None,
    prefix: str,
    part: str,
) -> QuantizationConfig | None:
    """``quant_config`` if the checkpoint carries ``prefix`` as INT8 W8A8,
    else None (the layer stays BF16; nothing is rounded at load).

    A checkpoint that declares the layer INT8 in another scheme (weight-only,
    static activations) is refused: its int8 tensors would not fit a BF16
    layer, and this path only runs the dynamic per-token W8A8 rule."""
    if quant_config is None:
        return None
    get_format = getattr(quant_config, "get_checkpoint_weight_format", None)
    if get_format is None:
        return None
    probe = _probe()
    try:
        weight_format = get_format(probe, prefix)
    except (AttributeError, TypeError, NotImplementedError):
        # A config that cannot answer for a bare linear keeps the old path.
        return None
    if weight_format != ("int", 8):
        return None
    if not _is_w8a8_int8(quant_config, probe, prefix):
        raise ValueError(
            f"{prefix} ({part}) is INT8 in the checkpoint but not W8A8 with "
            "dynamic int8 activations; only that scheme is supported for it."
        )
    logger.debug("%s (%s): INT8 W8A8 from the checkpoint", prefix, part)
    return quant_config


def mark(layer: torch.nn.Module, part: str) -> torch.nn.Module:
    """Tag a layer built through ``checkpoint_int8_quant_config`` for the
    load-time summary line."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    setattr(layer, PART_ATTR, part)
    setattr(
        layer,
        INT8_ATTR,
        not isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod),
    )
    return layer


def summarize(model: torch.nn.Module) -> dict[str, tuple[int, int, str]]:
    """{part: (int8 count, total, method names)} over the tagged layers."""
    totals: Counter[str] = Counter()
    int8: Counter[str] = Counter()
    methods: dict[str, set[str]] = {}
    for module in model.modules():
        part = getattr(module, PART_ATTR, None)
        if part is None:
            continue
        totals[part] += 1
        if getattr(module, INT8_ATTR, False):
            int8[part] += 1
            scheme = getattr(module, "scheme", None)
            name = type(scheme).__name__ if scheme is not None else "?"
            methods.setdefault(part, set()).add(name)
    return {
        part: (int8[part], totals[part], ",".join(sorted(methods.get(part, ()))))
        for part in totals
    }


def log_checkpoint_int8_parts(model: torch.nn.Module, label: str) -> None:
    """One line per model: how many of each tagged part run INT8."""
    summary = summarize(model)
    if not summary:
        return
    parts = "; ".join(
        f"{part} {n} of {total}" + (f" INT8 ({methods})" if n else " INT8")
        for part, (n, total, methods) in sorted(summary.items())
    )
    logger.info(
        "INT8 checkpoint parts (%s): %s. The others run BF16 as the checkpoint "
        "keeps them; none is rounded at load.",
        label,
        parts,
    )
