# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-IP-K: W4A8 routed experts for the XPU int4 MoE (B70_MOE_W4A8=1).

The int4 expert weights stay exactly the bytes ``XpuFusedMoe`` holds (GPTQ
nibble order, signed int4 after ``implement_zp``: byte p of a row holds input
k = 2p in the low nibble and k = 2p + 1 in the high nibble; bf16 scales
[E, N, K/128]). No copy, no repack. Activations are quantized per token to
symmetric int8 (dynamic). The GEMM unpacks int4 -> int8 in registers (exact)
and runs int8 x int8 -> int32 DPAS per 128-wide K group; each group's int32
partial sum is scaled by the weight scale of (row of W, group) into an fp32
accumulator, and the token's activation scale is applied once at the end.

To avoid an interleave shuffle in registers, the activation quantizer writes
each 128-wide K group de-interleaved: the 64 even k first, then the 64 odd k.
The low nibbles of a 64-byte weight block then pair with the first half and
the high nibbles with the second half: two K=64 dots per group.

Routing (remap_hidden_states), SiLU-and-mul semantics and the weighted gather
(moe_gather) are the stock XpuFusedMoe ops; only the two grouped GEMMs (and the
activation quantization feeding them) change.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

GROUP = 128

# (BLOCK_M, BLOCK_N, num_warps, large GRF) per GEMM; override with
# B70_MOE_W4A8_CFG1 / B70_MOE_W4A8_CFG2 = "BM,BN,warps,grf".
_DEFAULT_CFG1 = (64, 128, 8, 0)
_DEFAULT_CFG2 = (64, 128, 8, 0)


def enabled() -> bool:
    return os.environ.get("B70_MOE_W4A8", "0") == "1"


def min_tokens() -> int:
    return int(os.environ.get("B70_MOE_W4A8_MIN_TOKENS", "64"))


def uva_enabled() -> bool:
    """Hot/cold: also run the cold experts read over UVA (pinned host) through W4A8."""
    return os.environ.get("B70_MOE_W4A8_UVA", "0") == "1"


def _cfg(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    vals = [int(x) for x in raw.split(",")]
    return tuple(vals + [0] * (4 - len(vals)))


@triton.jit
def _b70_w4a8_pair(base, c, j, SILU: tl.constexpr, INTER: tl.constexpr):
    """Even and odd elements (64 each) of 128-wide group ``c`` of one row, fp32.
    SILU: the row is [gate(INTER) | up(INTER)] and the value is silu(gate) * up."""
    off = c * 128 + 2 * j
    if SILU:
        ge = tl.load(base + off).to(tl.float32)
        go = tl.load(base + off + 1).to(tl.float32)
        ue = tl.load(base + INTER + off).to(tl.float32)
        uo = tl.load(base + INTER + off + 1).to(tl.float32)
        ve = ge / (1.0 + tl.exp(-ge)) * ue
        vo = go / (1.0 + tl.exp(-go)) * uo
    else:
        ve = tl.load(base + off).to(tl.float32)
        vo = tl.load(base + off + 1).to(tl.float32)
    return ve, vo


@triton.jit
def _b70_w4a8_round(v):
    # round half away from zero, clamp to the symmetric int8 range
    r = tl.where(v >= 0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))
    return tl.minimum(tl.maximum(r, -127.0), 127.0).to(tl.int8)


@triton.jit
def _b70_w4a8_quant_kernel(
    X, Xq, Xs, n_valid_ptr, stride_x,
    K: tl.constexpr, SILU: tl.constexpr, INTER: tl.constexpr, HAS_N_VALID: tl.constexpr,
):
    """Per-row symmetric int8: Xq[row] (de-interleaved per 128 group), Xs[row] = amax / 127."""
    row = tl.program_id(0)
    if HAS_N_VALID:
        if row >= tl.load(n_valid_ptr):
            return
    j = tl.arange(0, 64)
    base = X + row.to(tl.int64) * stride_x
    amax = tl.zeros([64], dtype=tl.float32)
    for c in tl.static_range(K // 128):
        ve, vo = _b70_w4a8_pair(base, c, j, SILU, INTER)
        amax = tl.maximum(amax, tl.maximum(tl.abs(ve), tl.abs(vo)))
    m = tl.max(amax, 0)
    scale = tl.maximum(m, 1e-20) / 127.0
    tl.store(Xs + row, scale)
    out = Xq + row.to(tl.int64) * K
    for c in tl.static_range(K // 128):
        ve, vo = _b70_w4a8_pair(base, c, j, SILU, INTER)
        tl.store(out + c * 128 + j, _b70_w4a8_round(ve / scale))
        tl.store(out + c * 128 + 64 + j, _b70_w4a8_round(vo / scale))


@triton.jit
def _b70_w4a8_gemm_kernel(
    A, As, W, Ws, Out, tile_e_ptr, tile_r0_ptr, seg_end_ptr,
    N, stride_we, stride_se,
    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    """Out[r, n] = As[r] * sum_g Ws[e, n, g] * sum_{k in g} A[r, k] * W[e, n, k] for the rows r of
    one expert tile. A int8 [R, K] (groups de-interleaved), W int4 packed [E, N, K/2]."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    e = tl.load(tile_e_ptr + pid_t)
    if e < 0:
        return
    r0 = tl.load(tile_r0_ptr + pid_t)
    r_end = tl.load(seg_end_ptr + e)
    n0 = pid_n * BN
    KP: tl.constexpr = K // 2
    G: tl.constexpr = K // 128
    a_lo = tl.make_block_ptr(A, shape=(r_end, K), strides=(K, 1), offsets=(r0, 0),
                             block_shape=(BM, 64), order=(1, 0))
    a_hi = tl.make_block_ptr(A, shape=(r_end, K), strides=(K, 1), offsets=(r0, 64),
                             block_shape=(BM, 64), order=(1, 0))
    w_p = tl.make_block_ptr(W + e.to(tl.int64) * stride_we, shape=(KP, N), strides=(1, KP),
                            offsets=(0, n0), block_shape=(64, BN), order=(0, 1))
    offs_n = n0 + tl.arange(0, BN)
    s_ptr = Ws + e.to(tl.int64) * stride_se + offs_n * G
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in range(G):
        al = tl.load(a_lo, boundary_check=(0,))
        ah = tl.load(a_hi, boundary_check=(0,))
        pw = tl.load(w_p)
        wl = (pw << 4) >> 4  # low nibble, sign-extended: even k
        wh = pw >> 4  # high nibble (arithmetic shift): odd k
        part = tl.dot(al, wl, out_dtype=tl.int32)
        part = tl.dot(ah, wh, acc=part, out_dtype=tl.int32)
        sc = tl.load(s_ptr + g).to(tl.float32)
        acc += part.to(tl.float32) * sc[None, :]
        a_lo = tl.advance(a_lo, (0, 128))
        a_hi = tl.advance(a_hi, (0, 128))
        w_p = tl.advance(w_p, (64, 0))
    offs_m = r0 + tl.arange(0, BM)
    asc = tl.load(As + offs_m, mask=offs_m < r_end, other=0.0)
    acc = acc * asc[:, None]
    o = tl.make_block_ptr(Out, shape=(r_end, N), strides=(N, 1), offsets=(r0, n0),
                          block_shape=(BM, BN), order=(1, 0))
    tl.store(o, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))


def tile_map(rows_per_expert: torch.Tensor, max_rows: int, bm: int):
    """Device-side (no host sync) tile -> (expert, first row) map over expert-contiguous rows.
    Returns tile_e [T] (-1 = idle program), tile_r0 [T], seg_end [E] (int32)."""
    counts = rows_per_expert.to(torch.int64)
    n_exp = counts.numel()
    tiles = (counts + bm - 1) // bm
    cum = torch.cumsum(tiles, 0)
    seg_end = torch.cumsum(counts, 0)
    n_tiles = (max_rows + bm - 1) // bm + n_exp
    t = torch.arange(n_tiles, device=counts.device, dtype=torch.int64)
    e = torch.searchsorted(cum, t, right=True)
    ec = e.clamp(max=n_exp - 1)
    local = t - (cum - tiles)[ec]
    r0 = (seg_end - counts)[ec] + local * bm
    tile_e = torch.where(e < n_exp, ec, torch.full_like(ec, -1))
    return tile_e.to(torch.int32), r0.to(torch.int32), seg_end.to(torch.int32)


def quant_rows(x: torch.Tensor, n_valid: torch.Tensor | None = None, silu_inter: int = 0):
    """int8 de-interleaved rows + fp32 per-row scales. silu_inter > 0: x is [R, 2*I] gate|up."""
    rows = x.shape[0]
    k = silu_inter if silu_inter else x.shape[1]
    assert k % GROUP == 0 and x.stride(1) == 1
    xq = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    xs = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows:
        _b70_w4a8_quant_kernel[(rows,)](
            x, xq, xs, n_valid if n_valid is not None else xs, x.stride(0),
            K=k, SILU=bool(silu_inter), INTER=silu_inter, HAS_N_VALID=n_valid is not None,
            num_warps=1,
        )
    return xq, xs


def grouped_gemm(aq, a_s, w, ws, out, tile_e, tile_r0, seg_end, cfg):
    """out[R, N] (bf16) for expert-contiguous int8 rows ``aq`` [R, K] and int4 weights ``w`` [E, N, K/2]."""
    bm, bn, warps, grf = (tuple(cfg) + (0,))[:4]
    extra = {"grf_mode": "256"} if grf and os.environ.get("TRITON_INTERPRET") != "1" else {}
    n_exp, n, kp = w.shape
    k = kp * 2
    assert ws.shape == (n_exp, n, k // GROUP) and n % bn == 0 and aq.shape[1] == k
    grid = (n // bn, tile_e.numel())
    _b70_w4a8_gemm_kernel[grid](
        aq, a_s, w, ws, out, tile_e, tile_r0, seg_end,
        n, w.stride(0), ws.stride(0),
        K=k, BM=bm, BN=bn, num_warps=warps, **extra,
    )
    return out


def eligible(impl) -> str | None:
    """None if ``impl`` (an XpuFusedMoe) can run W4A8, else the reason."""
    if not getattr(impl, "is_int4", False):
        return "not an int4 MoE"
    if impl.activation != "silu" or impl.gemm1_clamp_limit:
        return f"activation {impl.activation} / clamp {impl.gemm1_clamp_limit}"
    if impl.w13_bias is not None or impl.w2_bias is not None:
        return "expert biases"
    if getattr(impl, "_astra_groups", None) is not None:
        return "astra group ownership re-laid out the weights"
    for w, s in ((impl.w13, impl.gemm1_wei_scales), (impl.w2, impl.gemm2_wei_scales)):
        if w.element_size() != 1 or w.dim() != 3 or not w.is_contiguous():
            return f"weight {w.dtype} {tuple(w.shape)}"
        if s is None or s.dim() != 3 or s.shape[-1] * GROUP != w.shape[-1] * 2 or not s.is_contiguous():
            return "scales are not [E, N, K/128] contiguous"
    if impl.w13.shape[1] != 2 * impl.inter_size or impl.inter_size % GROUP:
        return f"inter size {impl.inter_size}"
    return None


def apply(impl, output, hidden_states, topk_weights, topk_ids, expert_map=None) -> None:
    """Drop-in for XpuFusedMoe._apply_kernel (stock routing ops, W4A8 GEMMs)."""
    num_rows, hidden = hidden_states.shape
    topk = impl.n_experts_per_token
    n_moe = num_rows * topk
    if expert_map is None and impl.ep_size > 1:
        expert_map = impl.expert_map
    dev = hidden_states.device
    # Same as stock: rows routed nowhere (all -1) are zeroed before the remap.
    invalid = (topk_ids < 0).all(dim=1)
    hs = hidden_states.masked_fill(invalid.unsqueeze(1), 0)
    # Quantize the M token rows once, then route the int8 bytes with their scales through the
    # stock remap (as e4m3 bytes: the remap copies, it does not convert).
    hq, hs_scale = quant_rows(hs)
    remapped = torch.empty((n_moe, hidden), dtype=torch.float8_e4m3fn, device=dev)
    remapped_scales = torch.empty((n_moe, 1), dtype=torch.float32, device=dev)
    rows_per_expert = torch.zeros((impl.num_experts,), dtype=torch.int32, device=dev)
    unp = torch.empty((num_rows, topk), dtype=torch.int32, device=dev)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hq.view(torch.float8_e4m3fn),
        hidden_states_scales=hs_scale.view(num_rows, 1),
        remapped_hidden_states=remapped,
        remapped_hidden_states_scales=remapped_scales,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unp,
        topk_ids=topk_ids,
        total_experts_num=impl.total_experts_num,
        local_experts_num=impl.local_experts_num,
    )
    cfg1 = _cfg("B70_MOE_W4A8_CFG1", _DEFAULT_CFG1)
    cfg2 = _cfg("B70_MOE_W4A8_CFG2", _DEFAULT_CFG2)
    tile_e, tile_r0, seg_end = tile_map(rows_per_expert, n_moe, cfg1[0])
    inter = impl.inter_size
    g1 = torch.empty((n_moe, 2 * inter), dtype=output.dtype, device=dev)
    grouped_gemm(remapped.view(torch.int8), remapped_scales.view(-1), impl.w13,
                 impl.gemm1_wei_scales, g1, tile_e, tile_r0, seg_end, cfg1)
    aq2, as2 = quant_rows(g1, seg_end[-1:], silu_inter=inter)
    if cfg2[0] != cfg1[0]:
        tile_e, tile_r0, seg_end = tile_map(rows_per_expert, n_moe, cfg2[0])
    g2 = torch.empty((n_moe, hidden), dtype=output.dtype, device=dev)
    grouped_gemm(aq2, as2, impl.w2, impl.gemm2_wei_scales, g2, tile_e, tile_r0, seg_end, cfg2)
    torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp, impl.num_experts)


_LOGGED = False


def maybe_apply(impl, output, hidden_states, topk_weights, topk_ids) -> bool:
    """W4A8 when B70_MOE_W4A8=1, the batch has >= B70_MOE_W4A8_MIN_TOKENS tokens and ``impl`` is a
    plain int4 g128 SiLU XpuFusedMoe; False = caller runs the stock path."""
    global _LOGGED
    if not enabled() or hidden_states.shape[0] < min_tokens():
        return False
    ok = getattr(impl, "_b70_w4a8_ok", None)
    if ok is None:
        why = eligible(impl)
        ok = why is None
        impl._b70_w4a8_ok = ok
        if why is not None:
            logger.warning_once("B70-IP-K: MoE W4A8 refused (%s); stock W4A16 path", why)
    if not ok:
        return False
    if not _LOGGED:
        _LOGGED = True
        logger.info("B70-IP-K: MoE W4A8 on: int4 g128 weights as stored, int8 per-token activations, "
                    "int8 x int8 -> int32 DPAS, batches >= %d tokens (B70_MOE_W4A8_MIN_TOKENS); "
                    "cfg1 %s cfg2 %s", min_tokens(), _cfg("B70_MOE_W4A8_CFG1", _DEFAULT_CFG1),
                    _cfg("B70_MOE_W4A8_CFG2", _DEFAULT_CFG2))
    apply(impl, output, hidden_states, topk_weights, topk_ids)
    return True
