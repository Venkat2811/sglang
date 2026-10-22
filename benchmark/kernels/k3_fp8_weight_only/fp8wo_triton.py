"""Weight-only block-FP8 GEMM for small M (decode), Triton.

out[m, n] = sum_k x[m, k] * (w_fp8[n, k] * s[n // 128, k // 128])

- x: bf16 [M, K] (row stride arbitrary, unit column stride)
- w: float8_e4m3fn [N, K] contiguous, 128x128 block scales s: fp32 [ceil(N/128), K/128]
- out: bf16 or fp32 [M, N] (row stride arbitrary, unit column stride)

Swap-AB: each program owns BLOCK_N weight rows and one K slice; the weight tile is widened to bf16 in registers and
multiplied with the (tiny) activation tile on the tensor cores; the 128x128 scale is applied per 128-wide K step to
the fp32 partial (BLOCK_K == 128, so one scale per weight row per step). Split-K partials go to an fp32 workspace and
the last program of each N tile (self-resetting per-tile counter, graph-safe) reduces them and writes the output, so a
call is one launch with no memset and no separate reduce kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8wo_kernel(
    x_ptr, w_ptr, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_wn, stride_sn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IS_SPLIT: tl.constexpr, EVEN_N: tl.constexpr,
    SCALE_W: tl.constexpr = True,
):
    BLOCK_K: tl.constexpr = 128
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * K_PER_SPLIT
    if EVEN_N:
        n_ld = offs_n
    else:
        n_ld = tl.minimum(offs_n, N - 1)
    w_ptrs = w_ptr + n_ld[:, None] * stride_wn + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + (k0 + offs_k)[:, None]
    s_ptrs = s_ptr + (n_ld // 128) * stride_sn + k0 // BLOCK_K
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, K_PER_SPLIT, BLOCK_K):
        w = tl.load(w_ptrs, cache_modifier=".cg", eviction_policy="evict_first")
        xa = tl.load(x_ptrs, mask=m_mask, other=0.0)
        s = tl.load(s_ptrs)
        if SCALE_W:
            # scale the widened weight tile before the MMA so the accumulator never leaves the tensor-core path
            wb = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
            acc = tl.dot(wb, xa, acc)
        else:
            part = tl.dot(w.to(tl.bfloat16), xa)
            acc += part * s[:, None]
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        s_ptrs += 1
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        # partial tile -> workspace[pid_k, pid_n, BLOCK_N, BLOCK_M]
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for j in range(0, SPLIT_K):
                t2 = j * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
            tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def block_m_for(m: int) -> int:
    return 16 if m <= 16 else (32 if m <= 32 else (64 if m <= 64 else 128))


class Fp8WoWorkspace:
    """Per-device split-K workspace + counters (counters self-reset; zeroed once)."""

    def __init__(self, device, ws_floats: int = 8 << 20, n_counters: int = 1 << 16):
        self.ws = torch.empty(ws_floats, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(n_counters, dtype=torch.int32, device=device)


_WS: dict = {}


def get_ws(device) -> Fp8WoWorkspace:
    """One workspace per (device, stream): split-K counters must not be shared by kernels that can run concurrently."""
    key = (str(device), torch.cuda.current_stream(device).cuda_stream)
    if key not in _WS:
        _WS[key] = Fp8WoWorkspace(device)
    return _WS[key]


def fp8wo_gemm(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor | None = None,
               out_dtype: torch.dtype | None = None, block_n: int = 32, split_k: int = 1,
               num_warps: int = 4, num_stages: int = 6, ws: Fp8WoWorkspace | None = None,
               scale_w: bool = True) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert K % 128 == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    if nb % split_k != 0:
        raise ValueError(f"split_k {split_k} does not divide {nb} K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    _fp8wo_kernel[(grid_n, split_k)](
        x, w, s, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * 128, split_k,
        x.stride(0), w.stride(0), s.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, IS_SPLIT=split_k > 1, EVEN_N=(N % block_n == 0), SCALE_W=scale_w,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def quant_block_fp8(w: torch.Tensor, block: int = 128):
    """bf16 [N, K] -> (fp8 e4m3 [N, K], fp32 scales [ceil(N/128), K/128]); scale = amax / 448 per 128x128 block."""
    N, K = w.shape
    assert K % block == 0
    npad = (-N) % block
    wf = w.float()
    if npad:
        wf = torch.cat([wf, wf.new_zeros(npad, K)], 0)
    nb, kb = wf.shape[0] // block, K // block
    t = wf.view(nb, block, kb, block)
    amax = t.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / 448.0
    q = (t / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.view(nb * block, K)[:N].contiguous()
    return q, scale.contiguous()


def dequant_block_fp8(q: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
    N, K = q.shape
    sf = s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)
    return q.float() * sf


# ---------------------------------------------------------------------------------------------------------------------
# TMA variant: the weight tile streams through a host-side tensor descriptor (one bulk copy per stage), BLOCK_K 128 or
# 256 (two 128-wide scale columns per tile). The activation tile is a plain (L2-resident) load.
# ---------------------------------------------------------------------------------------------------------------------
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: E402


@triton.jit
def _fp8wo_tma_kernel(
    x_ptr, w_desc, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_sn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, IS_SPLIT: tl.constexpr,
    W_FP8: tl.constexpr,
):
    KB: tl.constexpr = BLOCK_K // 128
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * K_PER_SPLIT
    n_ld = tl.minimum(offs_n, N - 1)
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        k = k0 + kk
        w = w_desc.load([n0, k])
        xa = tl.load(x_ptr + offs_m[None, :] * stride_xm + (k + offs_k)[:, None], mask=m_mask, other=0.0)
        if W_FP8:
            if KB == 1:
                s = tl.load(s_ptr + (n_ld // 128) * stride_sn + k // 128)
                wb = (w.to(tl.float32) * s[:, None]).to(tl.bfloat16)
            else:
                s2 = tl.load(s_ptr + (n_ld // 128)[:, None] * stride_sn + (k // 128 + tl.arange(0, KB))[None, :])
                s3 = tl.broadcast_to(s2[:, :, None], (BLOCK_N, KB, 128))
                sk = tl.reshape(s3, (BLOCK_N, BLOCK_K))
                wb = (w.to(tl.float32) * sk).to(tl.bfloat16)
        else:
            wb = w
        acc = tl.dot(wb, xa, acc)
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for j in range(0, SPLIT_K):
                t2 = j * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
            tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


_DESC_CACHE: dict = {}


def weight_desc(w: torch.Tensor, block_n: int, block_k: int) -> TensorDescriptor:
    key = (w.data_ptr(), tuple(w.shape), block_n, block_k)
    d = _DESC_CACHE.get(key)
    if d is None:
        d = TensorDescriptor(w, list(w.shape), [w.stride(0), 1], [block_n, block_k])
        _DESC_CACHE[key] = d
    return d


def fp8wo_gemm_tma(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor | None, out: torch.Tensor | None = None,
                   out_dtype: torch.dtype | None = None, block_n: int = 64, block_k: int = 128, split_k: int = 1,
                   num_warps: int = 4, num_stages: int = 4, ws: Fp8WoWorkspace | None = None) -> torch.Tensor:
    """w fp8 [N, K] with block scales s, or bf16 [N, K] with s None (same kernel, no widening)."""
    M, K = x.shape
    N = w.shape[0]
    assert K % block_k == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // block_k
    if nb % split_k != 0:
        raise ValueError("split_k does not divide the K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    w_fp8 = w.dtype == torch.float8_e4m3fn
    _fp8wo_tma_kernel[(grid_n, split_k)](
        x, weight_desc(w, block_n, block_k), s if w_fp8 else x, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * block_k, split_k,
        x.stride(0), s.stride(0) if w_fp8 else 0, out.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, BLOCK_K=block_k, IS_SPLIT=split_k > 1, W_FP8=w_fp8,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# v3: fp16 tensor-core path. The fp8 weight widens to fp16 with one cvt per two values (e4m3 is exact in fp16); the
# activation tile is converted to fp16 once per 128-wide K step. Scale placement:
#   SCALE_X=True : the per-(row block, K block) scale multiplies the activation tile before the MMA (BLOCK_N <= 128 so
#                  one scalar per step) and the accumulator stays in the MMA;
#   SCALE_X=False: the partial of each K step is scaled in registers (acc += dot * s).
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def _fp8wo_v3_kernel(
    x_ptr, w_desc, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_sn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, IS_SPLIT: tl.constexpr,
    SCALE_X: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    k0 = pid_k * K_PER_SPLIT
    m_mask = offs_m[None, :] < M
    s_row = s_ptr + (n0 // 128) * stride_sn
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, 128 * KB):
        for j in tl.static_range(KB):
            k = k0 + kk + j * 128
            w = w_desc.load([n0, k])
            xa = tl.load(x_ptr + offs_m[None, :] * stride_xm + (k + offs_k)[:, None], mask=m_mask, other=0.0)
            s = tl.load(s_row + k // 128)
            if SCALE_X:
                acc = tl.dot(w.to(tl.float16), (xa.to(tl.float32) * s).to(tl.float16), acc)
            else:
                acc += tl.dot(w.to(tl.float16), xa.to(tl.float16)) * s
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for jj in range(0, SPLIT_K):
                t2 = jj * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
            tl.store(o_ptrs, tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def fp8wo_gemm_v3(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor | None = None,
                  out_dtype: torch.dtype | None = None, block_n: int = 64, kb: int = 1, split_k: int = 1,
                  num_warps: int = 4, num_stages: int = 4, scale_x: bool = True,
                  ws: Fp8WoWorkspace | None = None) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert block_n <= 128 and 128 % block_n == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    if K % 128 or nb % (split_k * kb):
        raise ValueError("split_k * kb must divide the 128-wide K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    _fp8wo_v3_kernel[(grid_n, split_k)](
        x, weight_desc(w, block_n, 128), s, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * 128, split_k,
        x.stride(0), s.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, KB=kb, IS_SPLIT=split_k > 1, SCALE_X=scale_x,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# v4: bf16 tensor-core path with a two-instruction widen: cvt.rn.bf16x2.e4m3x2 (SM100+) then mul.rn.bf16x2 by the
# block scale, so the accumulator stays in the MMA and the activation tile is used as is (bf16).
# ---------------------------------------------------------------------------------------------------------------------
@triton.jit
def _widen_scale(w_u8, s_bf16):
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b16 lo, hi;
        mov.b32 {lo, hi}, $2;
        cvt.rn.bf16x2.e4m3x2 $0, lo;
        cvt.rn.bf16x2.e4m3x2 $1, hi;
        mul.rn.bf16x2 $0, $0, $3;
        mul.rn.bf16x2 $1, $1, $4;
        }""",
        constraints="=r,=r,r,r,r",
        args=[w_u8, s_bf16],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _fp8wo_v4_kernel(
    x_ptr, w_desc, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_sn, stride_om, col_gap_at, col_gap_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, IS_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    k0 = pid_k * K_PER_SPLIT
    m_mask = offs_m[None, :] < M
    s_row = s_ptr + (n0 // 128) * stride_sn
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, 128 * KB):
        for j in tl.static_range(KB):
            k = k0 + kk + j * 128
            w = w_desc.load([n0, k])
            xa = tl.load(x_ptr + offs_m[None, :] * stride_xm + (k + offs_k)[:, None], mask=m_mask, other=0.0)
            s = tl.load(s_row + k // 128).to(tl.bfloat16)
            sb = tl.full((BLOCK_N, 128), 0.0, tl.bfloat16) + s
            wb = _widen_scale(w.to(tl.uint8, bitcast=True), sb)
            acc = tl.dot(wb, xa, acc)
    # output column remap (lets one launch skip a column gap, e.g. the bf16 router inside the MoE front)
    oc = offs_n + tl.where(offs_n >= col_gap_at, col_gap_len, 0)
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for jj in range(0, SPLIT_K):
                t2 = jj * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def fp8wo_gemm_v4(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor | None = None,
                  out_dtype: torch.dtype | None = None, block_n: int = 128, kb: int = 1, split_k: int = 1,
                  num_warps: int = 4, num_stages: int = 4, ws: Fp8WoWorkspace | None = None,
                  col_gap_at: int = 1 << 30, col_gap_len: int = 0) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert block_n <= 128 and 128 % block_n == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N + col_gap_len), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    if K % 128 or nb % (split_k * kb):
        raise ValueError("split_k * kb must divide the 128-wide K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    _fp8wo_v4_kernel[(grid_n, split_k)](
        x, weight_desc(w, block_n, 128), s, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * 128, split_k,
        x.stride(0), s.stride(0), out.stride(0), col_gap_at, col_gap_len,
        BLOCK_M=bm, BLOCK_N=block_n, KB=kb, IS_SPLIT=split_k > 1,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _fp8wo_v5_kernel(
    x_ptr, w_desc, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_sn, stride_om, col_gap_at, col_gap_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, IS_SPLIT: tl.constexpr,
    WS: tl.constexpr, NSTAGE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    k0 = pid_k * K_PER_SPLIT
    m_mask = offs_m[None, :] < M
    s_row = s_ptr + (n0 // 128) * stride_sn
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in tl.range(0, K_PER_SPLIT, 128, warp_specialize=WS, num_stages=NSTAGE):
        if True:
            k = k0 + kk
            w = w_desc.load([n0, k])
            xa = tl.load(x_ptr + offs_m[None, :] * stride_xm + (k + offs_k)[:, None], mask=m_mask, other=0.0)
            s = tl.load(s_row + k // 128).to(tl.bfloat16)
            sb = tl.full((BLOCK_N, 128), 0.0, tl.bfloat16) + s
            wb = _widen_scale(w.to(tl.uint8, bitcast=True), sb)
            acc = tl.dot(wb, xa, acc)
    # output column remap (lets one launch skip a column gap, e.g. the bf16 router inside the MoE front)
    oc = offs_n + tl.where(offs_n >= col_gap_at, col_gap_len, 0)
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for jj in range(0, SPLIT_K):
                t2 = jj * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def fp8wo_gemm_v5(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor | None = None,
                  out_dtype: torch.dtype | None = None, block_n: int = 128, kb: int = 1, split_k: int = 1,
                  num_warps: int = 4, num_stages: int = 4, ws: Fp8WoWorkspace | None = None,
                  col_gap_at: int = 1 << 30, col_gap_len: int = 0, ws_loop: bool = True) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert block_n <= 128 and 128 % block_n == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N + col_gap_len), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    if K % 128 or nb % (split_k * kb):
        raise ValueError("split_k * kb must divide the 128-wide K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    _fp8wo_v5_kernel[(grid_n, split_k)](
        x, weight_desc(w, block_n, 128), s, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * 128, split_k,
        x.stride(0), s.stride(0), out.stride(0), col_gap_at, col_gap_len,
        BLOCK_M=bm, BLOCK_N=block_n, KB=kb, IS_SPLIT=split_k > 1, WS=ws_loop, NSTAGE=num_stages,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out



@triton.jit
def _fp8wo_v4p_kernel(
    x_ptr, w_desc, s_ptr, out_ptr, ws_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_sn, stride_om, col_gap_at, col_gap_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, IS_SPLIT: tl.constexpr,
    PDL_EARLY: tl.constexpr = True,
):
    tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    k0 = pid_k * K_PER_SPLIT
    m_mask = offs_m[None, :] < M
    s_row = s_ptr + (n0 // 128) * stride_sn
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, 128 * KB):
        for j in tl.static_range(KB):
            k = k0 + kk + j * 128
            w = w_desc.load([n0, k])
            xa = tl.load(x_ptr + offs_m[None, :] * stride_xm + (k + offs_k)[:, None], mask=m_mask, other=0.0)
            s = tl.load(s_row + k // 128).to(tl.bfloat16)
            sb = tl.full((BLOCK_N, 128), 0.0, tl.bfloat16) + s
            wb = _widen_scale(w.to(tl.uint8, bitcast=True), sb)
            acc = tl.dot(wb, xa, acc)
    if PDL_EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    # output column remap (lets one launch skip a column gap, e.g. the bf16 router inside the MoE front)
    oc = offs_n + tl.where(offs_n >= col_gap_at, col_gap_len, 0)
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wsp = ws_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wsp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for jj in range(0, SPLIT_K):
                t2 = jj * tl.num_programs(0) + pid_n
                p2 = ws_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], tot.to(out_ptr.dtype.element_ty), mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + offs_m[None, :] * stride_om + oc[:, None], acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def fp8wo_gemm_v4p(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, out: torch.Tensor | None = None,
                  out_dtype: torch.dtype | None = None, block_n: int = 128, kb: int = 1, split_k: int = 1,
                  num_warps: int = 4, num_stages: int = 4, ws: Fp8WoWorkspace | None = None,
                  col_gap_at: int = 1 << 30, col_gap_len: int = 0) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    assert block_n <= 128 and 128 % block_n == 0 and x.stride(1) == 1 and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N + col_gap_len), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    if K % 128 or nb % (split_k * kb):
        raise ValueError("split_k * kb must divide the 128-wide K blocks")
    bm = block_m_for(M)
    grid_n = triton.cdiv(N, block_n)
    ws = ws or get_ws(x.device)
    if split_k > 1:
        assert grid_n * split_k * block_n * bm <= ws.ws.numel() and grid_n <= ws.cnt.numel()
    _fp8wo_v4p_kernel[(grid_n, split_k)](
        x, weight_desc(w, block_n, 128), s, out, ws.ws, ws.cnt,
        M, N, (nb // split_k) * 128, split_k,
        x.stride(0), s.stride(0), out.stride(0), col_gap_at, col_gap_len,
        BLOCK_M=bm, BLOCK_N=block_n, KB=kb, IS_SPLIT=split_k > 1,
        num_warps=num_warps, num_stages=num_stages, launch_pdl=True,
    )
    return out
