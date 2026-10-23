# SPDX-License-Identifier: Apache-2.0
"""L32: PleNvmeServer.resolve_ids (device-hashed ids -> cache slots) and the stage gather of B70-L32-0013."""
import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "ple_nvme", os.path.join(HERE, "../../vllm/models/qwen4_exp/nvidia/ple_nvme.py"))
ple_nvme = importlib.util.module_from_spec(spec)
sys.modules["ple_nvme"] = ple_nvme
spec.loader.exec_module(ple_nvme)

ROWS, DIM = 4096, 164


def _table(tmp):
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, size=(ROWS, DIM), dtype=np.uint8)
    hdr = json.dumps({"table": {"dtype": "U8", "shape": [ROWS, DIM], "data_offsets": [0, data.nbytes]},
                      "__metadata__": {"format": "lumnus-ple-int8-rowscale/v1"}}).encode()
    hdr += b" " * (-len(hdr) % 8)
    path = os.path.join(tmp, "t.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + data.tobytes())
    return path, data


def _server(path, tp_start, tp_end, cap=512):
    start, shape, dt = ple_nvme.safetensors_tensor_location(path, "table")
    store = ple_nvme.PleNvmeRowStore(path, start, DIM, shape[0], io_threads=2, reader="py")
    slab = np.zeros((cap, DIM), np.uint8)
    cache = ple_nvme.PleRowCache(tp_end - tp_start, cap, DIM, slab=slab)
    stats = ple_nvme.PleNvmeStats(0, 60.0, False)
    srv = ple_nvme.PleNvmeServer(tp_start=tp_start, tp_end=tp_end, multipliers=np.ones(3, np.int64),
                                 sizes=np.array([2048, 2048]), offsets=np.array([0, 2048]),
                                 eos_token_id=0, heads_per_ngram=1, cache=cache, store=store, stats=stats)
    return srv, slab


def test_resolve_ids_gathers_owned_rows_only():
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp)
        srv, slab = _server(path, 1024, 3072)
        rng = np.random.default_rng(1)
        for _ in range(5):  # repeated steps: hits + misses + evictions
            ids = rng.integers(0, ROWS, size=(40, 2))
            slots, mask = srv.resolve_ids(ids)
            assert np.array_equal(mask, (ids >= 1024) & (ids < 3072))
            t, c = np.nonzero(mask)
            assert np.array_equal(slab[slots[t, c]], data[ids[t, c]])


def test_stage_fill_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp)
        srv, slab = _server(path, 0, 2048)
        own_heads = np.array([0])  # only head 0's rows [0, 2048) are owned
        ids = np.stack([np.arange(10) * 7, 2048 + np.arange(10)], axis=1)
        stage = np.full((16, 2, DIM), 9, np.uint8)
        stage[:12].fill(0)
        slots, mask = srv.resolve_ids(ids[:, own_heads])
        t, c = np.nonzero(mask)
        stage[t, own_heads[c]] = slab[slots[t, c]]
        assert np.array_equal(stage[:10, 0], data[ids[:, 0]])
        assert not stage[:10, 1].any() and not stage[10:12].any()
