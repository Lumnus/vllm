# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-K: compile the QSA sparse attention kernels for BMG-G31 with NO device.

Ahead-of-time: Triton's Intel backend lowers to SPIR-V and, with
``generate_native_code``, runs ``ocloc`` for the zebin; the spill size is read
from the zebin's ``.ze_info`` (the same field Triton uses to decide its 256-GRF
retry). Nothing opens /dev/dri, so this runs while every card is busy:

    unshare -m bash -c 'mount -t tmpfs none /dev/dri; \
        TRITON_INTEL_DEVICE_ARCH=bmg <venv>/bin/python -I aot_attn_regs.py \
        --ops-dir <tree>/vllm/models/qwen4_exp/nvidia/ops --target <kernel.json>'

``--target`` is any Triton cache ``<kernel>.json`` compiled on a B70 (it holds
the device properties dict); without it the built-in B70 dict below is used.

Prints one json line per (kernel, config, grf_mode): spill bytes per thread,
GRF mode the zebin was built with, threads per warp, TTGIR flags (dpas, 2D
block IO, FMA dot) and shared memory. The shapes are the engine's TP4 shapes
(6 q heads / 1 KV head / head_dim 256 / 2,051 selected tokens / 832-token
blocks / page table at max_model_len 262,144).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import types

B70_ARCH = {
    "architecture": 21483225088,
    "device_id": 57891,
    "driver_version": "1.17.39758+10",
    "gpu_eu_count": 256,
    "gpu_subslice_count": 32,
    "has_atomic64": True,
    "has_bfloat16_conversions": True,
    "has_fp16": True,
    "has_fp64": True,
    "has_subgroup_2d_block_io": True,
    "has_subgroup_matrix_multiply_accumulate": True,
    "has_subgroup_matrix_multiply_accumulate_tensor_float32": False,
    "is_integrated_gpu": False,
    "last_level_cache_size": 25165824,
    "local_mem_size": 131072,
    "max_compute_units": 256,
    "max_num_sub_groups": 64,
    "max_work_group_size": 1024,
    "memory_bus_width": 64,
    "memory_clock_rate": 19000,
    "name": "Intel(R) Arc(TM) Pro B70 Graphics",
    "platform_name": "Intel(R) oneAPI Unified Runtime over Level-Zero V2",
    "sub_group_sizes": [16, 32],
    "total_memory": 34242297856,
    "type": "gpu",
    "vendor": "Intel(R) Corporation",
    "version": "20.2.0",
    "has_2d_block_io": True,
    "has_bfloat16_conversion": True,
    "__intel_already_queried_extensions__": True,
    "arch": "bmg",
}

# TP4 engine shapes
HEADS, KV_HEADS, HEAD_DIM = 6, 1, 256
SEL_WIDTH = 2051
BLOCK = 832
WIDTH = 316  # ceil(262144 / 832)
NBLOCKS = 4000


class FakeTensor:
    """Enough of a tensor for Triton's binder: dtype + a 16-aligned data_ptr."""

    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 1 << 20


def _stub_modules(ops_dir: str):
    import triton
    from triton import language as tl

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Plat:
        def is_xpu(self):
            return True

        def get_device_capability(self):
            return None

        def has_device_capability(self, *_):
            return False

    mod("vllm")
    mod("vllm.triton_utils", tl=tl, triton=triton, HAS_TRITON=True)
    mod("vllm.platforms", current_platform=_Plat())
    mod("vllm.model_executor")
    mod("vllm.model_executor.warmup")
    mod(
        "vllm.model_executor.warmup.jit_warmup_triton_helper",
        TritonWarmupTensor=object,
        triton_scalar_specialization_rep=lambda x: x,
    )
    pkg = mod("qsaops")
    pkg.__path__ = [ops_dir]
    tune = {}
    mod(
        "qsaops.qsa_xpu_tuning",
        get=lambda k, f, d: tune.get(k, {}).get(f, d),
        launch_extra=lambda k: {},
        _tune=tune,
    )

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"qsaops.{name}", os.path.join(ops_dir, f"{name}.py")
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"qsaops.{name}"] = m
        spec.loader.exec_module(m)
        return m

    return load


SPILL_RE = re.compile(r"spill_size\s*[:=]\s*(\d+)")
GRF_RE = re.compile(r"(grf_count|num_grf|grf_size)\s*[:=]\s*(\d+)", re.I)


def zeinfo(zebin: bytes) -> dict:
    from elftools.elf.elffile import ELFFile

    with tempfile.NamedTemporaryFile(suffix=".zebin") as f:
        f.write(zebin)
        f.flush()
        with open(f.name, "rb") as fh:
            sec = ELFFile(fh).get_section_by_name(".ze_info")
            text = sec.data().decode() if sec is not None else ""
    out = {}
    m = SPILL_RE.search(text)
    out["spill_bytes"] = int(m.group(1)) if m else None
    m = GRF_RE.search(text)
    out["grf_field"] = int(m.group(2)) if m else None
    for key in ("private_size", "slm_size", "has_stack_calls", "simd_size"):
        mm = re.search(rf"{key}\s*[:=]\s*(\S+)", text)
        if mm:
            out[key] = mm.group(1)
    return out


DUMP = {"dir": ""}


def compile_one(jit_fn, args, kwargs, target, grf_mode, label=""):
    from triton.compiler import compile as tcompile
    from triton.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature

    backend = make_backend(target)
    binder = create_function_from_signature(jit_fn.signature, jit_fn.params, backend)
    kw = dict(kwargs)
    kw["generate_native_code"] = True
    kw["grf_mode"] = grf_mode
    kw.setdefault("debug", False)
    kw.setdefault("instrumentation_mode", "")
    bound, spec, _opts = binder(*args, **kw)
    options, signature, constexprs, attrs = jit_fn._pack_args(
        backend, kw, bound, spec, _opts
    )
    src = (
        jit_fn.ASTSource(jit_fn, signature, constexprs, attrs)
        if hasattr(jit_fn, "ASTSource")
        else None
    )
    if src is None:
        from triton.compiler import ASTSource

        src = ASTSource(jit_fn, signature, constexprs, attrs)
    ck = tcompile(src, target=target, options=options.__dict__)
    md = ck.metadata
    info = zeinfo(ck.asm["zebin"])
    ttgir = ck.asm.get("ttgir", "")
    if DUMP["dir"] and label:
        os.makedirs(DUMP["dir"], exist_ok=True)
        for stage in ("ttgir", "llir"):
            with open(os.path.join(DUMP["dir"], f"{label}.{stage}"), "w") as f:
                f.write(ck.asm.get(stage, ""))
    return {
        **info,
        "build_flags": getattr(md, "build_flags", ""),
        "threads_per_warp": getattr(md, "threads_per_warp", None),
        "shared": getattr(md, "shared", None),
        "dpas": "dpas" in ttgir,
        "block_io": "block_io" in ttgir,
        "fma_dot": ("tt.dot" in ttgir) and "dpas" not in ttgir,
        "ttgir_bytes": len(ttgir),
    }


def splitk_args(
    q, k, ind, out, part, lse, gate, mod, rows, splits, block_n, warps, stages, block_m
):
    num_tiles = -(-SEL_WIDTH // block_n)
    a = (
        q,
        k,
        k,
        ind,
        FakeTensor(ind.dtype),
        FakeTensor(ind.dtype),
        part,
        lse,
        out,
        0.0625,
        1.0,
        gate,
        HEADS * HEAD_DIM,
        HEAD_DIM,
        BLOCK * 512,
        512,
        512,
        BLOCK * 512,
        512,
        512,
        SEL_WIDTH + 1,
        WIDTH,
        HEADS * HEAD_DIM,
        HEAD_DIM,
        HEADS * HEAD_DIM,
        HEAD_DIM,
        rows,
        NBLOCKS,
        max(rows // 8, 1),
    )
    kw = dict(
        TOPK=SEL_WIDTH,
        PAGE_SIZE=BLOCK,
        PAGE_TABLE_WIDTH=WIDTH,
        GROUP_SIZE=HEADS,
        HEAD_DIM=HEAD_DIM,
        NUM_QUERY_HEADS=HEADS,
        NUM_SPLITS=splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=stages,
    )
    return a, kw


def attn2_args(q, k, ind, part, rows, block_n, num_dchunk, tpp, warps, stages, block_m):
    a = (
        q,
        k,
        k,
        ind,
        FakeTensor(ind.dtype),
        FakeTensor(ind.dtype),
        part,
        part,
        0.0625,
        1.0,
        HEADS * HEAD_DIM,
        HEAD_DIM,
        BLOCK * 512,
        512,
        512,
        BLOCK * 512,
        512,
        512,
        SEL_WIDTH + 1,
        WIDTH,
        rows,
        NBLOCKS,
        max(rows // 8, 1),
    )
    kw = dict(
        TOPK=SEL_WIDTH,
        PAGE_SIZE=BLOCK,
        PAGE_TABLE_WIDTH=WIDTH,
        GROUP_SIZE=HEADS,
        HEAD_DIM=HEAD_DIM,
        NUM_QUERY_HEADS=HEADS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_DCHUNK=num_dchunk,
        TILES_PER_PROG=tpp,
        num_warps=warps,
        num_stages=stages,
    )
    return a, kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops-dir", required=True)
    ap.add_argument("--target", default="")
    ap.add_argument("--kernels", default="splitk,attn2,merge2")
    ap.add_argument("--grf", default="default,256")
    ap.add_argument("--grid", default="", help="json list of attn2 configs to try")
    ap.add_argument("--dump", default="", help="write ttgir/llir per config here")
    args = ap.parse_args()
    os.environ.setdefault("TRITON_INTEL_DEVICE_ARCH", "bmg")
    DUMP["dir"] = args.dump

    import torch
    from triton.backends.compiler import GPUTarget

    arch = B70_ARCH
    if args.target:
        with open(args.target) as f:
            arch = json.load(f)["target"]["arch"]
    target = GPUTarget("xpu", arch, 16)
    load = _stub_modules(args.ops_dir)
    which = args.kernels.split(",")
    q = FakeTensor(torch.bfloat16)
    k = FakeTensor(torch.bfloat16)
    ind = FakeTensor(torch.int32)
    out = FakeTensor(torch.bfloat16)
    part = FakeTensor(torch.float32)
    grfs = args.grf.split(",")

    def emit(row):
        print(json.dumps(row), flush=True)

    if "splitk" in which:
        qsa = load("qsa")
        # (label, rows, splits, block_n, warps, stages, block_m): defaults + C1's best so far
        cfgs = [
            ("dec8-default", 8, 64, 32, 4, 2, 8),
            ("dec8-c1best", 8, 4, 16, 8, 1, 16),
            ("dec64-default", 64, 8, 32, 1, 2, 8),
            ("pre-default", 4096, 1, 32, 1, 2, 8),
            ("pre-c1best", 4096, 1, 16, 1, 3, 8),
        ]
        for label, rows, splits, bn, w, st, bm in cfgs:
            for grf in grfs:
                a, kw = splitk_args(
                    q,
                    k,
                    ind,
                    out,
                    part if splits > 1 else out,
                    part if splits > 1 else out,
                    out,
                    qsa,
                    rows,
                    splits,
                    bn,
                    w,
                    st,
                    bm,
                )
                row = {
                    "kernel": "splitk",
                    "cfg": label,
                    "block_n": bn,
                    "num_warps": w,
                    "num_stages": st,
                    "block_m": bm,
                    "splits": splits,
                    "grf_mode": grf,
                }
                try:
                    row.update(
                        compile_one(
                            qsa._qsa_sparse_paged_gqa_splitk_kernel,
                            a,
                            kw,
                            target,
                            grf,
                            f"splitk-{label}-grf{grf}",
                        )
                    )
                except Exception as e:  # record, do not crash
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                emit(row)

    if "attn2" in which or "merge2" in which:
        m2 = load("qsa_attn2")
    if "attn2" in which:
        if args.grid:
            grid = json.loads(args.grid)
        else:
            grid = [
                {
                    "block_n": bn,
                    "num_dchunk": nd,
                    "tiles_per_prog": tpp,
                    "num_warps": w,
                    "num_stages": st,
                }
                for bn in (16, 32, 64, 128)
                for nd in (2, 4, 8)
                for tpp in (1, 4)
                for w in (1,)
                for st in (1, 2)
            ]
        for c in grid:
            for grf in grfs:
                a, kw = attn2_args(
                    q,
                    k,
                    ind,
                    part,
                    c.get("rows", 8),
                    c["block_n"],
                    c.get("num_dchunk", 4),
                    c.get("tiles_per_prog", 1),
                    c.get("num_warps", 1),
                    c.get("num_stages", 1),
                    c.get("block_m", 8),
                )
                row = {"kernel": "attn2", **c, "grf_mode": grf}
                try:
                    label = (
                        "attn2-"
                        + "-".join(f"{k}{v}" for k, v in sorted(c.items()))
                        + f"-grf{grf}"
                    )
                    row.update(
                        compile_one(
                            m2._qsa_attn2_tile_kernel, a, kw, target, grf, label
                        )
                    )
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                emit(row)
    if "merge2" in which:
        for bn in (16, 32, 64):
            for w in (1, 2, 4):
                a = (
                    part,
                    part,
                    ind,
                    out,
                    out,
                    SEL_WIDTH + 1,
                    HEADS * HEAD_DIM,
                    HEAD_DIM,
                    HEADS * HEAD_DIM,
                    HEAD_DIM,
                    8,
                )
                kw = dict(
                    TOPK=SEL_WIDTH,
                    BLOCK_N=bn,
                    HEAD_DIM=HEAD_DIM,
                    NUM_QUERY_HEADS=HEADS,
                    num_warps=w,
                    num_stages=1,
                )
                row = {
                    "kernel": "merge2",
                    "block_n": bn,
                    "num_warps": w,
                    "grf_mode": "default",
                }
                try:
                    row.update(
                        compile_one(
                            m2._qsa_attn2_merge_kernel, a, kw, target, "default"
                        )
                    )
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                emit(row)


if __name__ == "__main__":
    main()
