# SPDX-License-Identifier: Apache-2.0
"""C1 graphs for stateless HC math; communication and hybrid state stay eager."""

import os
import threading
from contextlib import contextmanager

import torch

_scope = threading.local()


@contextmanager
def hc_graph_scope(width):
    previous = getattr(_scope, "active", False)
    _scope.active = width in (4, 8) and os.getenv("VLLM_XPU_C1_HC_GRAPH", "1") == "1"
    try:
        yield
    finally:
        _scope.active = previous


def replay_hc(module, name, forward, args):
    """Return a fixed-address HC result, or use eager outside the C1 scope."""
    if not getattr(_scope, "active", False) or torch.xpu.is_current_stream_capturing():
        return forward(*args)
    entries = getattr(module, "_c1_hc_graphs", None)
    if entries is None:
        entries = module._c1_hc_graphs = {}
    signature = tuple(
        (tuple(x.shape), x.dtype, x.device) if x is not None else None for x in args
    )
    entry = entries.get(name)
    if entry is not None and entry["signature"] != signature:
        return forward(*args)
    if entry is None:
        if not hasattr(_scope, "pool"):
            _scope.pool = torch.xpu.graph_pool_handle()
        inputs = tuple(x.clone() if x is not None else None for x in args)
        # HC is stateless: warmup and capture cannot advance target caches.
        forward(*inputs)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, pool=_scope.pool):
            output = forward(*inputs)
        entry = entries[name] = dict(
            signature=signature, inputs=inputs, output=output, graph=graph, replays=0
        )
    else:
        for destination, source in zip(entry["inputs"], args):
            if destination is not None:
                destination.copy_(source)
    entry["graph"].replay()
    entry["replays"] += 1
    return entry["output"]
