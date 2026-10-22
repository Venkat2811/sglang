"""Adapter kernels between DeepEP low-latency dispatch and FlashInfer's
trtllm-gen routed MXFP4 MoE.

DeepEP LL (FP8, ``round_scale`` + ``use_ue8m0``) hands each rank a per-expert
packed slab: ``x[num_local_experts, R, hidden]`` in FP8 e4m3 with UE8M0 scales
per 128 columns, packed four to an int32 (``[E, R, hidden // 512]``, column-major
in the last two dims), plus ``masked_m[E]`` valid rows per expert. Rows at or
past ``masked_m[e]`` are stale.

trtllm-gen's routed MoE wants dense token rows ``[T, hidden]`` (MXFP8: UE8M0 per
32 columns, linear layout) and top-k expert ids per row. The adapter

1. gathers the valid ``(expert, slot)`` pairs into ``T`` compact rows, each with
   top-1 routing to its (rank-local) expert at weight 1.0; the per-128 scale
   byte is repeated over its four 32-column blocks, which is exact (same UE8M0
   exponent, same FP8 payload). Rows past the valid count get a non-local id,
   which trtllm-gen skips;
2. after the expert GEMMs, scatters the per-row outputs back into DeepEP's
   ``[E, R, hidden]`` slab for the combine (which applies the real top-k
   weights). With ``do_finalize=False`` the scatter reads the permuted GEMM2
   output directly through ``expanded_idx_to_permuted_idx``, so the finalize
   kernel and its full-width writes are skipped.

``T`` is a static bound (CUDA-graph safe): every valid pair fits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# bf16 1.0 in the low 16 bits of a packed (expert_id << 16) | weight entry.
_BF16_ONE_BITS = 0x3F80


@triton.jit
def _deepep_ll_gather_mxfp8_kernel(
    x_ptr,
    xs_ptr,
    cnt_ptr,
    xc_ptr,
    sc_ptr,
    ids_ptr,
    rowmap_ptr,
    stride_xe,
    stride_xr,
    stride_se,
    stride_sr,
    stride_sg,
    R,
    invalid_packed,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    NSF: tl.constexpr,
    NSF_PAD: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
):
    j = tl.program_id(0)
    eo = tl.arange(0, E_PAD)
    cnt = tl.load(cnt_ptr + eo, mask=eo < E, other=0)
    cnt = tl.minimum(tl.maximum(cnt, 0), R)
    incl = tl.cumsum(cnt, axis=0)
    # expert of compact row j: number of experts whose inclusive end is <= j
    e = tl.sum((incl <= j).to(tl.int32), axis=0)
    if e < E:
        start = tl.sum(tl.where(eo == e, incl - cnt, 0), axis=0)
        s = j - start
        ho = tl.arange(0, H_PAD)
        hm = ho < H
        src = x_ptr + e.to(tl.int64) * stride_xe + s.to(tl.int64) * stride_xr
        v = tl.load(src + ho, mask=hm)
        tl.store(xc_ptr + j.to(tl.int64) * H + ho, v, mask=hm)
        # per-32 scale column c -> 128-group q = c // 4 -> int32 word q // 4,
        # byte q % 4 (little endian)
        co = tl.arange(0, NSF_PAD)
        cm = co < NSF
        q = co // 4
        w = tl.load(
            xs_ptr + e * stride_se + s * stride_sr + (q // 4) * stride_sg,
            mask=cm,
            other=0,
        )
        b = (w >> ((q % 4) * 8)) & 0xFF
        tl.store(sc_ptr + j.to(tl.int64) * NSF + co, b.to(tl.uint8), mask=cm)
        tl.store(ids_ptr + j, (e << 16) | WEIGHT_BITS)
        tl.store(rowmap_ptr + j, e * R + s)
    else:
        tl.store(ids_ptr + j, invalid_packed)
        tl.store(rowmap_ptr + j, -1)


@triton.jit
def _deepep_ll_scatter_kernel(
    src_ptr,
    perm_ptr,
    rowmap_ptr,
    out_ptr,
    stride_src,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    USE_PERM: tl.constexpr,
):
    j = tl.program_id(0)
    r = tl.load(rowmap_ptr + j)
    if r >= 0:
        if USE_PERM:
            p = tl.load(perm_ptr + j)
        else:
            p = j
        ho = tl.arange(0, H_PAD)
        hm = ho < H
        v = tl.load(src_ptr + p.to(tl.int64) * stride_src + ho, mask=hm)
        tl.store(out_ptr + r.to(tl.int64) * H + ho, v, mask=hm)


@triton.jit
def _deepep_ll_gather_mxfp8_by_expert_kernel(
    x_ptr,
    xs_ptr,
    cnt_ptr,
    xc_ptr,
    sc_ptr,
    ids_ptr,
    rowmap_ptr,
    stride_xe,
    stride_xr,
    stride_se,
    stride_sr,
    stride_sg,
    R,
    num_rows,
    invalid_packed,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    NSF: tl.constexpr,
    NSF_PAD: tl.constexpr,
    WEIGHT_BITS: tl.constexpr,
    FILL_BLOCK: tl.constexpr,
):
    # programs [0, E): copy expert e's valid slots; programs [E, ...): write the padding ids
    pid = tl.program_id(0)
    eo = tl.arange(0, E_PAD)
    cnt = tl.load(cnt_ptr + eo, mask=eo < E, other=0)
    cnt = tl.minimum(tl.maximum(cnt, 0), R)
    incl = tl.cumsum(cnt, axis=0)
    if pid < E:
        n = tl.sum(tl.where(eo == pid, cnt, 0), axis=0)
        start = tl.sum(tl.where(eo == pid, incl - cnt, 0), axis=0)
        ho = tl.arange(0, H_PAD)
        hm = ho < H
        co = tl.arange(0, NSF_PAD)
        cm = co < NSF
        q = co // 4
        for s in range(0, n):
            j = start + s
            if j < num_rows:
                src = x_ptr + pid.to(tl.int64) * stride_xe + s.to(tl.int64) * stride_xr
                v = tl.load(src + ho, mask=hm)
                tl.store(xc_ptr + j.to(tl.int64) * H + ho, v, mask=hm)
                w = tl.load(
                    xs_ptr + pid * stride_se + s * stride_sr + (q // 4) * stride_sg,
                    mask=cm,
                    other=0,
                )
                b = (w >> ((q % 4) * 8)) & 0xFF
                tl.store(sc_ptr + j.to(tl.int64) * NSF + co, b.to(tl.uint8), mask=cm)
                tl.store(ids_ptr + j, (pid << 16) | WEIGHT_BITS)
                tl.store(rowmap_ptr + j, pid * R + s)
    else:
        total = tl.sum(cnt, axis=0)
        fo = (pid - E) * FILL_BLOCK + tl.arange(0, FILL_BLOCK)
        fm = (fo >= total) & (fo < num_rows)
        tl.store(ids_ptr + fo, tl.full([FILL_BLOCK], 0, tl.int32) + invalid_packed, mask=fm)
        tl.store(rowmap_ptr + fo, tl.full([FILL_BLOCK], -1, tl.int32), mask=fm)


@triton.jit
def _deepep_ll_scatter_by_expert_kernel(
    src_ptr,
    perm_ptr,
    cnt_ptr,
    out_ptr,
    stride_src,
    R,
    num_rows,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
):
    e = tl.program_id(0)
    eo = tl.arange(0, E_PAD)
    cnt = tl.load(cnt_ptr + eo, mask=eo < E, other=0)
    cnt = tl.minimum(tl.maximum(cnt, 0), R)
    incl = tl.cumsum(cnt, axis=0)
    n = tl.sum(tl.where(eo == e, cnt, 0), axis=0)
    start = tl.sum(tl.where(eo == e, incl - cnt, 0), axis=0)
    ho = tl.arange(0, H_PAD)
    hm = ho < H
    for s in range(0, n):
        j = start + s
        if j < num_rows:
            p = tl.load(perm_ptr + j)
            v = tl.load(src_ptr + p.to(tl.int64) * stride_src + ho, mask=hm)
            tl.store(out_ptr + (e * R + s).to(tl.int64) * H + ho, v, mask=hm)


def deepep_ll_gather_mxfp8(
    x: torch.Tensor,
    x_scale_packed: torch.Tensor,
    masked_m: torch.Tensor,
    num_rows: int,
    invalid_id: int,
    by_row: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact DeepEP LL FP8 slab rows into trtllm-gen MXFP8 rows.

    Returns ``(x_compact [num_rows, H] fp8, scale [num_rows, H // 32] (UE8M0
    bytes viewed as float8_e4m3fn), packed top-1 ids [num_rows, 1] int32,
    row_map [num_rows] int32 = e * R + slot, or -1 for padding rows)``.
    """
    E, R, H = x.shape
    assert x.dtype == torch.float8_e4m3fn, x.dtype
    assert x_scale_packed.dtype == torch.int32, x_scale_packed.dtype
    assert x_scale_packed.shape[:2] == (E, R), (x_scale_packed.shape, x.shape)
    assert H % 512 == 0 and x_scale_packed.shape[2] == H // 512
    assert x.stride(2) == 1
    nsf = H // 32
    dev = x.device
    xc = torch.empty((num_rows, H), dtype=torch.float8_e4m3fn, device=dev)
    sc = torch.empty((num_rows, nsf), dtype=torch.uint8, device=dev)
    ids = torch.empty((num_rows, 1), dtype=torch.int32, device=dev)
    row_map = torch.empty((num_rows,), dtype=torch.int32, device=dev)
    invalid_packed = (invalid_id << 16) | _BF16_ONE_BITS
    if not by_row:
        fill_block = 1024
        _deepep_ll_gather_mxfp8_by_expert_kernel[(E + triton.cdiv(num_rows, fill_block),)](
            x.view(torch.uint8),
            x_scale_packed,
            masked_m,
            xc.view(torch.uint8),
            sc,
            ids,
            row_map,
            x.stride(0),
            x.stride(1),
            x_scale_packed.stride(0),
            x_scale_packed.stride(1),
            x_scale_packed.stride(2),
            R,
            num_rows,
            invalid_packed,
            E=E,
            E_PAD=triton.next_power_of_2(E),
            H=H,
            H_PAD=triton.next_power_of_2(H),
            NSF=nsf,
            NSF_PAD=triton.next_power_of_2(nsf),
            WEIGHT_BITS=_BF16_ONE_BITS,
            FILL_BLOCK=fill_block,
            num_warps=4,
        )
        return xc, sc.view(torch.float8_e4m3fn), ids, row_map
    _deepep_ll_gather_mxfp8_kernel[(num_rows,)](
        x.view(torch.uint8),
        x_scale_packed,
        masked_m,
        xc.view(torch.uint8),
        sc,
        ids,
        row_map,
        x.stride(0),
        x.stride(1),
        x_scale_packed.stride(0),
        x_scale_packed.stride(1),
        x_scale_packed.stride(2),
        R,
        invalid_packed,
        E=E,
        E_PAD=triton.next_power_of_2(E),
        H=H,
        H_PAD=triton.next_power_of_2(H),
        NSF=nsf,
        NSF_PAD=triton.next_power_of_2(nsf),
        WEIGHT_BITS=_BF16_ONE_BITS,
        num_warps=4,
    )
    return xc, sc.view(torch.float8_e4m3fn), ids, row_map


def deepep_ll_scatter(
    src: torch.Tensor,
    row_map: torch.Tensor,
    out: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
    masked_m: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out.view(-1, H)[row_map[j]] = src[perm[j] if perm is not None else j]
    for every compact row j with row_map[j] >= 0. With masked_m (and perm), one
    program per expert walks only its valid slots."""
    H = out.shape[-1]
    assert out.is_contiguous() and src.shape[-1] == H and src.stride(-1) == 1
    assert src.dtype == out.dtype
    n = row_map.shape[0]
    if masked_m is not None and perm is not None:
        E, R, _ = out.shape
        _deepep_ll_scatter_by_expert_kernel[(E,)](
            src,
            perm,
            masked_m,
            out,
            src.stride(0),
            R,
            n,
            E=E,
            E_PAD=triton.next_power_of_2(E),
            H=H,
            H_PAD=triton.next_power_of_2(H),
            num_warps=4,
        )
        return out
    _deepep_ll_scatter_kernel[(n,)](
        src,
        perm if perm is not None else row_map,
        row_map,
        out,
        src.stride(0),
        H=H,
        H_PAD=triton.next_power_of_2(H),
        USE_PERM=perm is not None,
        num_warps=4,
    )
    return out
