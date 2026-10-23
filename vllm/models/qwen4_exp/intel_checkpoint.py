# Copyright Wulf. Qwen model data remains under the Qwen Community License 1.0.
"""Pinned Intel PLE layout and bounded private mapping; no model conversion."""
import json
import mmap
import os
import struct

PLE_FILE = 'model-00016-of-00017.safetensors'
PLE_PREFIX = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_'
PLE_ROWS = 2500012
PLE_WIDTH = 160
PLE_SHARDS = 128
PLE_BYTES = PLE_ROWS * PLE_WIDTH * 2
PLE_OFFSET = 20736
PLE_FILE_BYTES = 102400512256


def ple_layout(path):
    """Reject changed dtype, numeric shard order, gaps, overlaps and truncation."""
    with open(path, 'rb') as stream:
        if os.fstat(stream.fileno()).st_size != PLE_FILE_BYTES:
            raise ValueError('Intel PLE file size differs from the pinned layout')
        size = struct.unpack('<Q', stream.read(8))[0]
        if size != PLE_OFFSET - 8:
            raise ValueError('Intel PLE header size differs from the pinned layout')
        header = json.loads(stream.read(size))
    names = {PLE_PREFIX + str(i) + '.weight' for i in range(PLE_SHARDS)}
    if set(header) - {'__metadata__'} != names:
        raise ValueError('Intel PLE tensor inventory differs from the pinned layout')
    for i in range(PLE_SHARDS):
        entry = header[PLE_PREFIX + str(i) + '.weight']
        if (entry['dtype'] != 'BF16' or entry['shape'] != [PLE_ROWS, PLE_WIDTH]
                or entry['data_offsets'] != [i * PLE_BYTES, (i + 1) * PLE_BYTES]):
            raise ValueError('Invalid Intel PLE numeric shard layout: ' + str(i))
    if PLE_OFFSET + PLE_SHARDS * PLE_BYTES > PLE_FILE_BYTES:
        raise ValueError('Intel PLE payload is truncated')
    return PLE_OFFSET, PLE_SHARDS * PLE_ROWS, PLE_WIDTH


def map_ple_table(path):
    """Return tensor plus mapping owner; caller protects the mapping before use."""
    import torch
    offset, rows, width = ple_layout(path)
    aligned = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    delta = offset - aligned
    with open(path, 'rb') as stream:
        owner = mmap.mmap(stream.fileno(), delta + rows * width * 2,
                          flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                          offset=aligned)
    table = torch.frombuffer(owner, dtype=torch.bfloat16, count=rows * width,
                             offset=delta).view(rows, width)
    return {'mapping': owner, 'table': table}
