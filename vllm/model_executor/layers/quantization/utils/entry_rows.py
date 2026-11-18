# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One activation row block quantized once, read by every INT8 consumer.

Inside an ``EntryRows`` scope (the XPU MoE entry, see
``fused_moe/xpu_moe_entry.py``) an INT8 W8A8 linear with dynamic symmetric
per-token activations that is about to quantize the scope's input tensor
takes the scope's int8 rows and scales instead. They were made by the same
call the linear makes (``ops.scaled_int8_quant(x, None, None,
symmetric=True)``) on the same tensor, so the linear computes what it would
have. A scope can also hold a layer's whole output, computed ahead by a
merged GEMM of the same rows (``outputs``).

The tensor must be the very input (same storage, shape, strides, dtype)
and unchanged since it was quantized; anything else quantizes as before.
"Unchanged" is proved by the version counter where the tensor has one. The
engine's forwards (profile run, graph capture, every serving step) run under
``torch.inference_mode``, whose tensors carry no version counter: there the
proof is the scope itself. It holds the input, so its storage cannot be freed
and reused while consumers look for it; consumers find the rows only inside
the forward that made them; and nothing in the scope writes its input in place
(the CPU check runs the layer with version counting on and asserts that the
input's version does not move). A normal tensor never matches an inference
one, or the other way round."""

from contextlib import contextmanager

import torch

# Open scopes, innermost last. Empty outside the XPU MoE entry.
ACTIVE: list["EntryRows"] = []


def _version(t: torch.Tensor) -> int | None:
    """``t``'s version counter; None for an inference tensor, which has
    none (reading ``_version`` raises)."""
    return None if t.is_inference() else t._version


class EntryRows:
    def __init__(
        self,
        x: torch.Tensor,
        q: torch.Tensor | None,
        s: torch.Tensor | None,
    ) -> None:
        self.x = x
        self.version = _version(x)
        self.q = q
        self.s = s
        # Whether consumers other than a merged GEMM read q / s.
        self.share_rows = True
        # id(layer) -> that layer's output for ``x``, computed ahead.
        self.outputs: dict[int, torch.Tensor] = {}
        # Names of the consumers that read the rows or an output (boot proof).
        self.readers: list[str] = []
        # Shared expert whose sigmoid gate is applied later (in the routed
        # gather), and that gate once the shared expert has computed it.
        self.fold_module: torch.nn.Module | None = None
        self.shared_gate: torch.Tensor | None = None
        self.shared_pending: tuple | None = None
        self.single_round = False
        # The gather applies the sigmoid: shared_gate holds the gate's logits.
        self.gate_logits = False
        self.folded = False

    def holds(self, t: torch.Tensor) -> bool:
        x = self.x
        return (
            t is x
            or (
                t.data_ptr() == x.data_ptr()
                and t.shape == x.shape
                and t.stride() == x.stride()
                and t.dtype == x.dtype
                and t.device == x.device
            )
        ) and _version(t) == self.version


def current() -> EntryRows | None:
    return ACTIVE[-1] if ACTIVE else None


@contextmanager
def scope(rows: EntryRows):
    ACTIVE.append(rows)
    try:
        yield rows
    finally:
        ACTIVE.pop()


def int8_input(layer: torch.nn.Module, x: torch.Tensor):
    """For an INT8 linear (dynamic symmetric per-token activations) about to
    quantize ``x``: ``(output, None)`` when the scope computed this layer's
    output ahead, ``(None, (q, s))`` when it holds ``x``'s int8 rows, else
    None (quantize as usual)."""
    rows = current()
    if rows is None or not rows.holds(x):
        return None
    out = rows.outputs.pop(id(layer), None)
    name = getattr(layer, "prefix", "") or type(layer).__name__
    if out is not None:
        rows.readers.append(name + " (merged GEMM)")
        return out, None
    if rows.q is None or not rows.share_rows:
        return None
    rows.readers.append(name)
    return None, (rows.q, rows.s)


def defer_shared_gate(module: torch.nn.Module, gate: torch.Tensor) -> bool:
    """For a gated shared expert: True when the open scope adds
    ``gate * out`` later (in the routed experts' gather), so the expert
    returns its output ungated; ``gate`` is the sigmoid column."""
    rows = current()
    if rows is None or rows.fold_module is not module or rows.gate_logits:
        return False
    rows.shared_gate = gate
    return True


def defer_shared_gate_logits(module: torch.nn.Module, logits: torch.Tensor) -> bool:
    """``defer_shared_gate`` before the sigmoid, when the open scope's gather
    applies it (VLLM_XPU_MOE_SHARED_GATE_IN_GATHER): True when the expert
    returns its output ungated and ``logits`` (the expert gate's output) go to
    the gather."""
    rows = current()
    if rows is None or rows.fold_module is not module or not rows.gate_logits:
        return False
    rows.shared_gate = logits
    return True
