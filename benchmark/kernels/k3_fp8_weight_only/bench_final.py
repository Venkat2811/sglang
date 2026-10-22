#!/usr/bin/env python3
"""bench_final.py --shapes A,B --out FILE: best-config timing of the weight-only FP8 kernels we built, per K3 GEMM and
rows (8..64), in the same harness as bench_gemm.py (one CUDA graph over distinct weight copies, >= 1.5 GB).
Arms: triton_v4 (bf16 MMA, A widened into tensor memory), triton_v3 (fp16 MMA), cuda_tc (fp8wo_tc.cu, tiled weights),
router_tma_bf16 (the 896-row router kept in bf16, our TMA kernel, fp32 out). bf16 cuBLAS and Marlin come from
bench_gemm.py (base.jsonl)."""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gemm import SHAPES, time_graph  # noqa: E402
from fp8wo_triton import (Fp8WoWorkspace, dequant_block_fp8, fp8wo_gemm_tma, fp8wo_gemm_v3, fp8wo_gemm_v4,  # noqa: E402
                          quant_block_fp8)
from fp8wo_tc import TcWorkspace, fp8wo_tc, tile_weight  # noqa: E402


def splits(nb, gn, lo=24, hi=1200):
    return [s for s in (1, 2, 3, 4, 6, 7, 8, 12, 14) if nb % s == 0 and lo <= gn * s <= hi] or [1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--ms", default="8,16,24,32,48,64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="triton_v4,triton_v3,cuda_tc")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(1)
    ws = Fp8WoWorkspace(dev)
    tws = TcWorkspace(dev)
    fo = open(a.out, "a")
    arms = a.arms.split(",")
    for name, N, K, count, out32, stream in [s for s in SHAPES if s[0] in a.shapes.split(",")]:
        ncopy = int(min(48, max(4, math.ceil(1.5e9 / (N * K * 2)))))
        wb = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
        odt = torch.float32 if out32 else torch.bfloat16
        nb = K // 128
        if name == "moe_router":
            nbytes = N * K * 2
            for M in [int(m) for m in a.ms.split(",")]:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                o = torch.empty(M, N, device=dev, dtype=odt)
                best = None
                for bn in (32, 64, 128):
                    for sk in splits(nb, math.ceil(N / bn)):
                        for nw in (4, 8):
                            for ns in (3, 4, 6):
                                c = dict(block_n=bn, block_k=128, split_k=sk, num_warps=nw, num_stages=ns)
                                try:
                                    fns = [lambda w=w: fp8wo_gemm_tma(x, w, None, out=o, ws=ws, **c) for w in wb]
                                    us = time_graph(fns, reps=4, batches=3)
                                    if best is None or us < best[0]:
                                        best = (us, c)
                                except Exception:  # noqa: BLE001
                                    pass
                fns = [lambda w=w: fp8wo_gemm_tma(x, w, None, out=o, ws=ws, **best[1]) for w in wb]
                us = time_graph(fns)
                rec = dict(shape=name, N=N, K=K, count=count, M=M, arm="router_tma_bf16", us=us,
                           GBps=nbytes / us / 1e3, cfg=best[1], t=time.strftime("%H:%M:%S", time.gmtime()))
                fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
            del wb
            torch.cuda.empty_cache()
            continue
        q = [quant_block_fp8(w) for w in wb]
        del wb
        nbytes = N * K + q[0][1].numel() * 4
        ref_w = dequant_block_fp8(*q[0])
        tiled = [(tile_weight(qq), s) for qq, s in q] if "cuda_tc" in arms else None
        for M in [int(m) for m in a.ms.split(",")]:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            o = torch.empty(M, N, device=dev, dtype=odt)
            ref = x.float() @ ref_w.t()
            for arm in arms:
                best = None
                if arm in ("triton_v4", "triton_v3"):
                    fn = fp8wo_gemm_v4 if arm == "triton_v4" else fp8wo_gemm_v3
                    cfgs = [dict(block_n=bn, kb=1, split_k=sk, num_warps=nw, num_stages=ns)
                            for bn in (64, 128) for sk in splits(nb, math.ceil(N / bn))
                            for nw in (4, 8) for ns in (2, 3, 4, 6)]
                    for c in cfgs:
                        try:
                            fn(x, q[0][0], q[0][1], out=o, ws=ws, **c)
                            torch.cuda.synchronize()
                            err = float((o.float() - ref).abs().max() / ref.abs().max())
                            if not err < 2e-2:
                                continue
                            fns = [lambda w=w, s=s: fn(x, w, s, out=o, ws=ws, **c) for w, s in q]
                            us = time_graph(fns, reps=4, batches=3)
                            if best is None or us < best[0]:
                                best = (us, c, err)
                        except Exception:  # noqa: BLE001
                            pass
                    if best is None:
                        continue
                    c = best[1]
                    us = time_graph([lambda w=w, s=s: fn(x, w, s, out=o, ws=ws, **c) for w, s in q])
                else:  # cuda_tc
                    for sk in (1, 2, 3, 4, 6, 8):
                        if sk > nb:
                            continue
                        for st in (3, 4, 6, 8):
                            if M > 64 and st > 4:
                                continue
                            try:
                                fp8wo_tc(x, tiled[0][0], tiled[0][1], out=o, split_k=sk, stages=st, ws=tws, n=N)
                                torch.cuda.synchronize()
                                err = float((o.float() - ref).abs().max() / ref.abs().max())
                                if not err < 2e-2:
                                    continue
                                fns = [lambda w=w, s=s: fp8wo_tc(x, w, s, out=o, split_k=sk, stages=st, ws=tws, n=N)
                                       for w, s in tiled]
                                us = time_graph(fns, reps=4, batches=3)
                                if best is None or us < best[0]:
                                    best = (us, dict(split_k=sk, stages=st), err)
                            except Exception:  # noqa: BLE001
                                pass
                    if best is None:
                        continue
                    c = best[1]
                    us = time_graph([lambda w=w, s=s: fp8wo_tc(x, w, s, out=o, ws=tws, n=N, **c) for w, s in tiled])
                rec = dict(shape=name, N=N, K=K, count=count, out_fp32=out32, stream=stream, M=M, arm=arm, us=us,
                           GBps=nbytes / us / 1e3, err=best[2], cfg=c, t=time.strftime("%H:%M:%S", time.gmtime()))
                fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
        del q, tiled
        torch.cuda.empty_cache()
    fo.close()


if __name__ == "__main__":
    main()
