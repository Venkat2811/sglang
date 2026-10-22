"""FP8 weights on the native block-scaled FP8 tensor-core MMA, bf16 activations split in-CTA into two E4M3 terms.

out[m, n] = sum_k x[m, k] * w[n, k],  w = q_w * 2^e_w  (e4m3 with a power-of-two scale per 128x128 block)

The weight tile goes from TMA (shared memory) straight into tcgen05's block-scaled MMA (kind::mxf8f6f4), so it is never
widened in registers. The activation tile x [BLOCK_M, 128] (bf16, L2-resident) is split per (row, 32-wide K group):
    x = 2^a * q_hi + 2^b * q_lo + r,   q_hi = e4m3(x / 2^a),  q_lo = e4m3((x - 2^a q_hi) / 2^b)
with a, b power-of-two (UE8M0) scales from the group amax; two MMAs (q_hi, q_lo) accumulate into one fp32 accumulator.
TERMS=1 drops q_lo (plain W8A8 with in-CTA activation quantization, reference only).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from fp8wo_triton import Fp8WoWorkspace, block_m_for, get_ws  # noqa: F401


@triton.jit
def _e8m0_of(amax):
    # smallest power of two 2^e with amax / 2^e <= 448; returns (e as f32, ue8m0 byte)
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    return e, (e + 127.0).to(tl.uint8)


@triton.jit
def _fp8ds_kernel(
    x_ptr, w_desc, ws8_ptr, out_ptr, wsp_ptr, cnt_ptr,
    M, N, K_PER_SPLIT, SPLIT_K,
    stride_xm, stride_s8n, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IS_SPLIT: tl.constexpr, TERMS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    k0 = pid_k * K_PER_SPLIT
    m_mask = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, 128):
        k = k0 + kk
        w = w_desc.load([n0, k])                                         # [BLOCK_N, 128] e4m3
        # weight UE8M0 exponent of this 128x128 block, broadcast to the 4 groups of 32
        e_w = tl.load(ws8_ptr + (n0 // 128) * stride_s8n + k // 128)    # uint8 scalar
        sw = tl.full((BLOCK_N, 4), 0, tl.uint8) + e_w
        xa = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :], mask=m_mask, other=0.0)
        xf = tl.reshape(xa.to(tl.float32), (BLOCK_M, 4, 32))
        e_hi, s_hi = _e8m0_of(tl.max(tl.abs(xf), axis=2))               # [BLOCK_M, 4]
        q_hi = (xf * tl.exp2(-e_hi)[:, :, None]).to(tl.float8e4nv)
        acc = tl.dot_scaled(w, sw, "e4m3", tl.trans(tl.reshape(q_hi, (BLOCK_M, 128))), s_hi, "e4m3", acc)
        if TERMS == 2:
            r = xf - q_hi.to(tl.float32) * tl.exp2(e_hi)[:, :, None]
            e_lo, s_lo = _e8m0_of(tl.max(tl.abs(r), axis=2))
            q_lo = (r * tl.exp2(-e_lo)[:, :, None]).to(tl.float8e4nv)
            acc = tl.dot_scaled(w, sw, "e4m3", tl.trans(tl.reshape(q_lo, (BLOCK_M, 128))), s_lo, "e4m3", acc)
    o_mask = (offs_m[None, :] < M) & (offs_n[:, None] < N)
    if IS_SPLIT:
        tile = pid_k * tl.num_programs(0) + pid_n
        wp = wsp_ptr + tile * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        tl.store(wp, acc, cache_modifier=".cg")
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for jj in range(0, SPLIT_K):
                t2 = jj * tl.num_programs(0) + pid_n
                p2 = wsp_ptr + t2 * (BLOCK_N * BLOCK_M) + tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
                tot += tl.load(p2, cache_modifier=".cg")
            tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None], tot.to(out_ptr.dtype.element_ty),
                     mask=o_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
    else:
        tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None], acc.to(out_ptr.dtype.element_ty), mask=o_mask)


def quant_block_fp8_pow2(w: torch.Tensor, block: int = 128):
    """bf16 [N, K] -> (e4m3 [N, K], uint8 UE8M0 exponents [ceil(N/128), K/128]); scale = 2^ceil(log2(amax/448))."""
    N, K = w.shape
    npad = (-N) % block
    wf = w.float()
    if npad:
        wf = torch.cat([wf, wf.new_zeros(npad, K)], 0)
    nb, kb = wf.shape[0] // block, K // block
    t = wf.view(nb, block, kb, block)
    amax = t.abs().amax(dim=(1, 3)).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (t / torch.exp2(e)[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(nb * block, K)[:N].contiguous(), (e + 127).to(torch.uint8).contiguous()


def dequant_pow2(q: torch.Tensor, e8: torch.Tensor, block: int = 128) -> torch.Tensor:
    N, K = q.shape
    s = torch.exp2(e8.float() - 127.0)
    return q.float() * s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)


_DESC: dict = {}


def fp8ds_gemm(x, w, e8, out=None, out_dtype=None, split_k=1, num_warps=4, num_stages=4, terms=2, ws=None,
               block_n=128):
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    nb = K // 128
    assert K % 128 == 0 and nb % split_k == 0
    bm = max(16, block_m_for(M))
    grid_n = triton.cdiv(N, block_n)
    key = (w.data_ptr(), block_n)
    d = _DESC.get(key)
    if d is None:
        d = _DESC[key] = TensorDescriptor(w, list(w.shape), [w.stride(0), 1], [block_n, 128])
    ws = ws or get_ws(x.device)
    _fp8ds_kernel[(grid_n, split_k)](
        x, d, e8, out, ws.ws, ws.cnt, M, N, (nb // split_k) * 128, split_k,
        x.stride(0), e8.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=block_n, IS_SPLIT=split_k > 1, TERMS=terms, num_warps=num_warps, num_stages=num_stages)
    return out
