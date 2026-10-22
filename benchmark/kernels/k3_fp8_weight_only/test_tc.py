#!/usr/bin/env python3
"""test_tc.py: build fp8wo_tc, check correctness on small and K3 shapes, then time the K3 shapes per split-K/stages."""
import argparse, math, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8wo_triton import dequant_block_fp8, quant_block_fp8  # noqa: E402
from fp8wo_tc import TcWorkspace, fp8wo_tc, mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--check-only", action="store_true")
ap.add_argument("--shape", default="6144,7168")
ap.add_argument("--ms", default="24")
a = ap.parse_args()
dev = torch.device("cuda:0")
t0 = time.time(); mod(); print(f"built in {time.time() - t0:.0f} s", flush=True)
ws = TcWorkspace(dev)
torch.manual_seed(0)
for (N, K, M, sk, f32) in [(128, 128, 16, 1, False), (256, 512, 8, 1, False), (384, 1024, 24, 2, False),
                           (2112, 7168, 24, 4, False), (6144, 7168, 32, 4, True), (7168, 1536, 64, 1, False),
                           (7168, 3584, 5, 7, False), (512, 1024, 100, 1, False)]:
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    q, s = quant_block_fp8(w)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    o = fp8wo_tc(x, q, s, out_dtype=torch.float32 if f32 else torch.bfloat16, split_k=sk, ws=ws)
    torch.cuda.synchronize()
    ref = x.float() @ dequant_block_fp8(q, s).t()
    err = float((o.float() - ref).abs().max() / ref.abs().max())
    print(f"check N={N} K={K} M={M} split={sk} f32={f32}: rel err {err:.2e} {'OK' if err < 1e-2 else 'BAD'}", flush=True)
if a.check_only:
    sys.exit(0)
from bench_gemm import time_graph  # noqa: E402
N, K = [int(v) for v in a.shape.split(",")]
ncopy = max(4, math.ceil(1.5e9 / (N * K * 2)))
qs = [quant_block_fp8(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02) for _ in range(ncopy)]
nbytes = N * K + qs[0][1].numel() * 4
nb = K // 128
for M in [int(v) for v in a.ms.split(",")]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    o = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    res = []
    for sk in sorted({1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}):
        if sk > nb:
            continue
        for stg in (2, 3, 4, 6):
            try:
                fns = [lambda q=q, s=s: fp8wo_tc(x, q, s, out=o, split_k=sk, stages=stg, ws=ws) for q, s in qs]
                us = time_graph(fns, reps=4, batches=3)
                res.append((us, sk, stg))
            except Exception as e:  # noqa: BLE001
                print("fail", sk, stg, repr(e)[:200], flush=True)
    res.sort()
    for us, sk, stg in res[:5]:
        print(f"N={N} K={K} M={M}: {us:.2f} us {nbytes / us / 1e3:.0f} GB/s split {sk} stages {stg}", flush=True)
