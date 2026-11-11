# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel warm-up reaches the layer hooks for mixed-batch specialisations."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu import warmup


def _alignment_free(kernel, names: list[str]) -> bool:
    """Whether a Triton kernel does not specialise on these pointers' alignment."""
    params = {p.name: p for p in getattr(kernel, "params", [])}
    if not params:
        pytest.skip("Triton kernels are not JIT functions here")
    return all(params[name].do_not_specialize_on_alignment for name in names)


def test_dequant_kernel_ignores_group_slice_alignment():
    """Later prefill groups pass row slices of the block table and seq_lens;
    their alignment must not select another compiled variant."""
    from vllm.v1.attention.ops import xpu_int8_per_token_head as ops

    assert _alignment_free(ops._dequant_blocks_kernel, ["bt_ptr", "sl_ptr"])


def test_short_conv_kernels_ignore_index_slice_alignment():
    """A mixed batch splits its state and token indices at the decode count;
    the slices' alignment must not select another compiled variant."""
    from vllm.models.qwen4_exp.common.ops import ple

    names = ["state_idx_ptr", "qsl_ptr", "num_acc_ptr", "has_init_ptr"]
    names.append("token_idx_ptr")
    assert _alignment_free(ple._ple_conv_kernel, names)
    assert _alignment_free(ple._ple_conv_writeback_kernel, names)


def test_short_conv_warmup_covers_every_mapped_mode(monkeypatch):
    """The hook launches the spec, prefill and plain-decode modes with int64
    token maps, the decode one with and without initial states."""
    from vllm.models.qwen4_exp.common.ops import ple

    seen = []

    def record(*args, mode, token_indices=None, has_initial_states=None, **kw):
        seen.append((mode, token_indices.dtype, has_initial_states is not None))

    monkeypatch.setattr(ple, "ple_conv", record)
    ple.warmup_token_mapped_ple_conv(
        torch.zeros(4, 16, 16),
        torch.zeros(16, 4),
        dilation=3,
        spec_query_len=8,
        dtype=torch.float32,
    )
    assert seen == [
        ("spec", torch.int64, False),
        ("prefill", torch.int64, True),
        ("decode", torch.int64, False),
        ("decode", torch.int64, True),
    ]


def _runner(static_forward_context, groups, block_tables):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context=static_forward_context
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=names) for names in groups]
        ),
        block_tables=SimpleNamespace(input_block_tables=block_tables),
    )


def _int8_attention(calls, name, numel=6):
    def warmup_int8_kv_prefill(kv_cache, block_table):
        calls.append((name, tuple(kv_cache.shape), tuple(block_table.shape)))

    return SimpleNamespace(
        impl=SimpleNamespace(
            kv_cache_dtype="int8_per_token_head",
            warmup_int8_kv_prefill=warmup_int8_kv_prefill,
        ),
        kv_cache=torch.zeros(numel),
    )


def test_hooks_run_once_with_the_layers_block_table():
    calls: list = []

    def warmup_short_conv():
        calls.append("short conv")
        return True

    context = {
        "model.layers.1.ple": SimpleNamespace(warmup_short_conv=warmup_short_conv),
        "model.layers.3.attn": _int8_attention(calls, "layer 3"),
        "model.layers.7.attn": _int8_attention(calls, "layer 7"),
        "model.layers.0.attn": SimpleNamespace(
            impl=SimpleNamespace(kv_cache_dtype="auto"), kv_cache=torch.zeros(6)
        ),
    }
    runner = _runner(
        context,
        [["model.layers.1.ple"], ["model.layers.3.attn", "model.layers.7.attn"]],
        [torch.zeros(4, 1, dtype=torch.int32), torch.zeros(4, 37, dtype=torch.int32)],
    )
    warmup.warmup_mixed_batch_kernels(runner)
    assert calls == ["short conv", ("layer 3", (6,), (4, 37))]


def test_unbound_or_ungrouped_layers_are_skipped():
    calls: list = []
    context = {
        "a": _int8_attention(calls, "unbound", numel=0),
        "b": _int8_attention(calls, "ungrouped"),
        "c": SimpleNamespace(warmup_short_conv=lambda: False),
    }
    runner = _runner(context, [["a"]], [torch.zeros(4, 37, dtype=torch.int32)])
    warmup.warmup_mixed_batch_kernels(runner)
    assert calls == []
