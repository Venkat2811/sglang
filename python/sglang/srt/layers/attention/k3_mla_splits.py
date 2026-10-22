# Copyright 2026 Venkat Raman. Licensed under the Apache License, Version 2.0.
"""k3_mla_splits: balanced MLA decode waves for batches larger than one wave of 2-CTA clusters (Kimi-K3, Blackwell).

FlashInfer's cute-dsl monolithic MLA decode runs one 2-CTA cluster per request (K3's 96 heads fit one 128-row tile).
Its split heuristic gives split 1 once a rank has more than SMs / 4 requests, so with more requests than resident
clusters (SMs // 2: 74 on B300, 76 on GB300) the requests that do not fit run as a second, nearly empty wave: at 116K
on one B300 node the MLA kernel goes from 3.7 ms per step at 74 rows to 5.0-5.4 ms at 76 rows.

This path keeps FlashInfer's call and kernel unchanged and changes only its input. For B > C (C = resident clusters),
the r = B mod C overflow requests are cut into s pieces along their (rank-local) KV; each piece becomes a virtual row:
its page-table row is the request's row shifted by the piece's first page, its length is the piece length, its query
is the request's query. The B real rows come first (an overflow request's own row keeps piece 0), the extra pieces
after them, so the stock launch (split 1 for B' rows) runs the first q * C requests as full waves and the hardware
hands the small pieces to clusters as they free up. A Triton merge then folds each overflow request's pieces back into
its row with the usual LSE-weighted merge (the same merge DCP already does across ranks).

Correctness: a decode query sees every stored key, so a piece needs no causal bound of its own (its keys sit below
the request's global bound whatever local offset the kernel assumes). Pieces that end up empty (short contexts) store
O = 0, LSE = -inf (the kernel's DCP empty-row rule) and drop out of the merge. Only q_len == 1 under DCP takes this path.

s depends only on (B, C) (graph-safe); pieces are at least MIN_PIECE_TILES x 128 tokens long. Env:
  SGLANG_K3_MLA_WAVE_BALANCE=1    turn the path on (default off)
  SGLANG_K3_MLA_WAVE_MAX_SPLIT    cap on pieces per overflow request (default 32)
  SGLANG_K3_MLA_WAVE_SAT_CLUSTERS clusters that saturate HBM in this kernel (default 32)
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

TILE_KV = 128
MIN_PIECE_TILES = int(os.environ.get("SGLANG_K3_MLA_WAVE_MIN_PIECE_TILES", "2"))


def enabled() -> bool:
    return os.environ.get("SGLANG_K3_MLA_WAVE_BALANCE", "0") == "1"


@functools.lru_cache(maxsize=None)
def plan_rows(B: int, C: int, cap: int, n_sat: int) -> tuple:
    """(q, r, s, B'): q full waves, r overflow requests cut into s pieces, B' = B + r * (s - 1) launch rows.

    The kernel is bandwidth-bound once about n_sat clusters stream at the same time (one cluster alone reaches ~160 GB/s
    against ~4.1 TB/s for the whole B300), so the stock second wave only hurts while it holds fewer than n_sat requests.
    The overflow is cut into just enough pieces to put n_sat clusters to work: s = ceil(n_sat / r), none when r >= n_sat.
    """
    q, r = divmod(B, C)
    if B <= C or r == 0 or r >= n_sat:
        return q, r, 1, B
    s = min(cap, -(-n_sat // r))
    return q, r, s, B + r * (s - 1)


@triton.jit
def _build_rows_kernel(
    q_ptr, q_stride, qv_ptr, Q_BYTES: tl.constexpr, Q_CHUNK: tl.constexpr, NQ: tl.constexpr,
    bt_ptr, bt_stride, vt_ptr, vt_stride,
    len_ptr, glob_ptr, vlen_ptr, vglob_ptr,
    B, QC, S, MIN_TILES: tl.constexpr, PAGES_PER_TILE: tl.constexpr, PAGE: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """Program (v, j): j < NQ copies chunk j of row v's query; j >= NQ copies page chunk j - NQ of its page table."""
    tl.extra.cuda.gdc_wait()  # programmatic dependent launch: the query comes from the previous kernel
    v = tl.program_id(0)
    j = tl.program_id(1)
    # source request and piece index of virtual row v
    is_extra = v >= B
    e = tl.maximum(v - B, 0)
    sm1 = tl.maximum(S - 1, 1)
    src = tl.where(is_extra, QC + e // sm1, v)
    if j < NQ:
        qo = j * Q_CHUNK + tl.arange(0, Q_CHUNK)
        qm = qo < Q_BYTES
        qb = tl.load(q_ptr + src.to(tl.int64) * q_stride + qo, mask=qm)
        tl.store(qv_ptr + v.to(tl.int64) * Q_BYTES + qo, qb, mask=qm)
    else:
        k = tl.where(is_extra, e % sm1 + 1, 0)
        is_piece = (src >= QC) & (S > 1)
        L = tl.load(len_ptr + src)
        T = (L + 127) // 128
        per = tl.maximum((T + S - 1) // S, MIN_TILES)
        start_tile = k * per
        plen = tl.minimum(tl.maximum(L - start_tile * 128, 0), per * 128)
        vlen = tl.where(is_piece, plen, L)
        shift = tl.where(is_piece, start_tile * PAGES_PER_TILE, 0)
        npages = (vlen + PAGE - 1) // PAGE
        offs = (j - NQ) * BLOCK_P + tl.arange(0, BLOCK_P)
        m = offs < npages
        pages = tl.load(bt_ptr + src.to(tl.int64) * bt_stride + shift + offs, mask=m, other=0)
        tl.store(vt_ptr + v.to(tl.int64) * vt_stride + offs, pages, mask=m)
        if j == NQ:
            tl.store(vlen_ptr + v, vlen)
            tl.store(vglob_ptr + v, tl.load(glob_ptr + src))


@triton.jit
def _merge_kernel(o_ptr, lse_ptr, H, QC, B, S, D: tl.constexpr, BLOCK_S: tl.constexpr):
    """Fold pieces 1..S-1 (rows B + i*(S-1) + k-1) of overflow request i into its own row QC + i (piece 0)."""
    tl.extra.cuda.gdc_wait()  # launched early behind the MLA kernel; waits for its completion
    i = tl.program_id(0)
    h = tl.program_id(1)
    row0 = QC + i
    ks = tl.arange(0, BLOCK_S)
    km = ks < S
    rows = tl.where(ks == 0, row0, B + i * (S - 1) + ks - 1)
    lse = tl.load(lse_ptr + rows.to(tl.int64) * H + h, mask=km, other=-float("inf"))
    mx = tl.max(lse, axis=0)
    valid = mx > -float("inf")
    mx_safe = tl.where(valid, mx, 0.0)
    w = tl.where(km & (lse > -float("inf")), tl.exp(lse - mx_safe), 0.0)
    wsum = tl.sum(w, axis=0)
    d = tl.arange(0, D)
    o = tl.load(o_ptr + (rows.to(tl.int64) * H + h)[:, None] * D + d[None, :], mask=km[:, None], other=0.0)
    acc = tl.sum(o.to(tl.float32) * w[:, None], axis=0)
    res = tl.where(valid, acc / tl.where(wsum > 0, wsum, 1.0), 0.0)
    tl.store(o_ptr + (row0.to(tl.int64) * H + h) * D + d, res.to(o_ptr.dtype.element_ty))
    tl.store(lse_ptr + row0.to(tl.int64) * H + h, tl.where(valid, mx + tl.log(wsum), -float("inf")))


@functools.lru_cache(maxsize=None)
def _clusters(device_index: int) -> int:
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    return get_max_active_clusters(2)


def mla_decode(
    query: torch.Tensor,  # [B, 1, H, 576] fp8
    kv_cache: torch.Tensor,  # [pages, 1, page, 576] or [pages, page, 576] fp8
    workspace: torch.Tensor,  # FlashInfer cute-dsl workspace (bytes)
    block_tables: torch.Tensor,  # [B, P] int32, rank-local pages
    seq_lens: torch.Tensor,  # [B] int32, rank-local lengths
    max_seq_len: int,
    softmax_scale: float,
    causal_seqlens_kv_global: Optional[torch.Tensor],  # [B] int32
    cp_world: int,
    cp_rank: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    output_scale: float = 1.0,
    force: bool = False,
):
    """Rank-local MLA decode returning (out [B, 1, H, 512] bf16, lse [B, 1, H] natural log)."""
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as M

    B, q_len, H, D_qk = query.shape
    kv_lora_rank = 512
    enable_dcp = cp_world > 1
    C = _clusters(query.device.index if query.device.index is not None else 0)
    cap = int(os.environ.get("SGLANG_K3_MLA_WAVE_MAX_SPLIT", "32"))
    n_sat = int(os.environ.get("SGLANG_K3_MLA_WAVE_SAT_CLUSTERS", "32"))
    qn, r, s, Bv = plan_rows(B, C, cap, n_sat)

    def stock(qq, bt, sl, gl):
        return M.cute_dsl_mla_decode(
            qq, kv_cache, workspace, kv_lora_rank, D_qk - kv_lora_rank, bt, sl, max_seq_len, softmax_scale,
            output_scale, return_lse=True, enable_dcp=enable_dcp, cp_world=cp_world, cp_rank=cp_rank,
            causal_seqlens_kv_global=gl if enable_dcp else None)

    if s <= 1 or q_len != 1 or not enable_dcp:
        o, l = stock(query, block_tables, seq_lens, causal_seqlens_kv_global)
    else:
        page = kv_cache.shape[-2]
        P = block_tables.shape[1]
        dev = query.device
        qv = torch.empty((Bv,) + tuple(query.shape[1:]), dtype=query.dtype, device=dev)
        vt = torch.empty((Bv, P), dtype=torch.int32, device=dev)
        vlen = torch.empty((Bv,), dtype=torch.int32, device=dev)
        vglob = torch.empty((Bv,), dtype=torch.int32, device=dev)
        qc = query if query.is_contiguous() else query.contiguous()
        q_bytes = q_len * H * D_qk
        q_chunk = 8192
        nq = triton.cdiv(q_bytes, q_chunk)
        max_pages = min(P, (max_seq_len + page - 1) // page)
        BLOCK_P = 1024
        _build_rows_kernel[(Bv, nq + triton.cdiv(max_pages, BLOCK_P))](
            qc.view(torch.uint8), q_bytes, qv.view(torch.uint8), q_bytes, q_chunk, nq,
            block_tables, block_tables.stride(0), vt, vt.stride(0),
            seq_lens, causal_seqlens_kv_global, vlen, vglob,
            B, qn * C, s, MIN_PIECE_TILES, TILE_KV // page, page, BLOCK_P,
            num_warps=4,
            launch_pdl=True,
        )
        o, l = stock(qv, vt, vlen, vglob)  # o [Bv, 1, H, 512] bf16, l [Bv, 1, H] natural log
        _merge_kernel[(r, H)](
            o, l, H, qn * C, B, s, kv_lora_rank, triton.next_power_of_2(s), num_warps=8, launch_pdl=True
        )
        o, l = o[:B], l[:B]
    if out is not None:
        out.copy_(o.view_as(out))
        o = out
    if lse is not None:
        lse.copy_(l.reshape(lse.shape))
        l = lse
    return o, l
