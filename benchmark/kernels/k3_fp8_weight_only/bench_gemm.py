#!/usr/bin/env python3
"""bench_gemm.py: Kimi-K3's per-GPU bf16 projection GEMMs (TP8) at decode rows, bf16 cuBLAS vs FP8 weight-only kernels.

Arms per (shape, M):
  bf16      torch F.linear / torch.mm(out_dtype=fp32) on the bf16 weight (what the engine runs today, cuBLAS nvjet)
  marlin    sglang apply_fp8_marlin_linear (W8A16, block 128x128 -> group 128), SGLANG_FORCE_FP8_MARLIN's kernel
  triton    fp8wo_triton.fp8wo_gemm (W8A16, ours), best config of a sweep (config recorded)
  cake      FlashInfer 0.7.1 experimental kimi_k3_fp8_projection (W8A8: in-CTA or separate per-token E4M3 quant,
            UE8M0-requantized weight), reference only: it is not weight-only

Timing: one CUDA graph of n_copies back-to-back GEMMs on distinct weight copies (>= ~1.5 GB of weights, so nothing is
L2-resident, like the engine's layer stream), median of 5 timed batches of replays; us per GEMM. Effective GB/s =
weight bytes read (fp8 + fp32 scales for the FP8 arms) / time. Correctness: max |out - ref| / max |ref| against the fp32
product with the dequantized FP8 weight (FP8 arms) or the bf16 weight (bf16 arm).
Writes one JSON line per (shape, M, arm) to --out.
"""
import argparse, json, math, os, statistics, sys, time, traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8wo_triton import Fp8WoWorkspace, dequant_block_fp8, fp8wo_gemm, quant_block_fp8  # noqa: E402

# name, N, K, count per decode step (one B300 node, TP8), output dtype fp32?, stream
SHAPES = [
    ("kda_qkvg_in", 6144, 7168, 69, False, "main"),
    ("kda_o_proj", 7168, 1536, 69, False, "main"),
    ("mla_qa_kva", 2112, 7168, 24, False, "main"),
    ("mla_q_b", 2304, 1536, 24, False, "main"),
    ("mla_g_proj", 1536, 7168, 24, False, "side"),
    ("mla_o_proj", 7168, 1536, 24, False, "main"),
    ("moe_front", 6016, 7168, 92, True, "main"),
    ("moe_front_norouter", 5120, 7168, 92, True, "main"),
    ("moe_router", 896, 7168, 92, True, "main"),
    ("latent_up", 7168, 3584, 92, False, "main"),
    ("shared_down", 7168, 768, 92, False, "side"),
    ("dense_gate_up", 8448, 7168, 1, False, "main"),
    ("dense_down", 7168, 4224, 1, False, "main"),
    ("lm_head", 20480, 7168, 1, False, "main"),
]


def time_graph(fns, reps=8, batches=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(batches):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / (reps * len(fns)))
    del g
    return statistics.median(ts)


def relerr(out, ref):
    return float((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-30))


def triton_configs(N, K, M):
    nb = K // 128
    for bn in (16, 32, 64, 128):
        gn = math.ceil(N / bn)
        for sk in (1, 2, 3, 4, 6, 7, 8):
            if nb % sk or gn * sk < 24 or gn * sk > 1600:
                continue
            for nw in (4, 8):
                for ns in (3, 4, 6, 8):
                    yield dict(block_n=bn, split_k=sk, num_warps=nw, num_stages=ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--ms", default="8,16,24,32,48,64")
    ap.add_argument("--arms", default="bf16,marlin,triton,cake")
    ap.add_argument("--out", default="/work/bench/gemm.jsonl")
    ap.add_argument("--min-bytes", type=float, default=1.5e9)
    ap.add_argument("--quick-sweep", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    ms = [int(x) for x in a.ms.split(",")]
    arms = a.arms.split(",")
    shapes = SHAPES if a.shapes == "all" else [s for s in SHAPES if s[0] in a.shapes.split(",")]
    ws = Fp8WoWorkspace(dev)
    fo = open(a.out, "a")

    def emit(rec):
        rec["t"] = time.strftime("%H:%M:%S", time.gmtime())
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        print(json.dumps(rec), flush=True)

    for name, N, K, count, out32, stream in shapes:
        bf16_bytes = N * K * 2
        ncopy = int(min(48, max(4, math.ceil(a.min_bytes / bf16_bytes))))
        wb = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
        q = [quant_block_fp8(w) for w in wb]
        fp8_bytes = N * K + q[0][1].numel() * 4
        xs = {M: torch.randn(M, K, device=dev, dtype=torch.bfloat16) for M in ms}
        odt = torch.float32 if out32 else torch.bfloat16
        outs = {M: torch.empty(M, N, device=dev, dtype=odt) for M in ms}
        ref_q = dequant_block_fp8(*q[0])
        base = dict(shape=name, N=N, K=K, count=count, out_fp32=out32, stream=stream, ncopy=ncopy,
                    bf16_MB=bf16_bytes / 1e6, fp8_MB=fp8_bytes / 1e6)
        # ---- bf16 cuBLAS
        if "bf16" in arms:
            for M in ms:
                x, o = xs[M], outs[M]
                if out32:
                    fns = [lambda w=w: torch.mm(x, w.t(), out=o, out_dtype=torch.float32) for w in wb]
                else:
                    fns = [lambda w=w: torch.mm(x, w.t(), out=o) for w in wb]
                us = time_graph(fns)
                fns[0]()
                err = relerr(o, x.float() @ wb[0].float().t())
                emit(dict(base, M=M, arm="bf16", us=us, GBps=bf16_bytes / us / 1e3, err=err))
        # ---- marlin
        if "marlin" in arms:
            try:
                from sglang.srt.layers.quantization.marlin_utils_fp8 import (apply_fp8_marlin_linear,
                                                                            prepare_fp8_layer_for_marlin)
                layers = []
                for wq, sc in q:
                    L = torch.nn.Module()
                    L.weight = torch.nn.Parameter(wq.clone(), requires_grad=False)
                    L.weight_scale_inv = torch.nn.Parameter(sc.clone(), requires_grad=False)
                    L.weight_block_size = [128, 128]
                    L.output_size_per_partition = N
                    L.input_size_per_partition = K
                    L.orig_dtype = torch.bfloat16
                    prepare_fp8_layer_for_marlin(L, size_k_first=False)
                    layers.append(L)
                for M in ms:
                    x = xs[M]
                    res = {}
                    fns = [lambda L=L: res.__setitem__("o", apply_fp8_marlin_linear(x, L.weight, L.weight_scale,
                                                                                   L.workspace, N, K, None))
                           for L in layers]
                    us = time_graph(fns)
                    fns[0]()
                    err = relerr(res["o"], x.float() @ ref_q.t())
                    emit(dict(base, M=M, arm="marlin", us=us, GBps=fp8_bytes / us / 1e3, err=err))
                del layers
            except Exception as e:
                emit(dict(base, arm="marlin", error=repr(e)[:400]))
                traceback.print_exc()
        # ---- triton (ours), config sweep
        if "triton" in arms:
            wqs = [t[0] for t in q]
            scs = [t[1] for t in q]
            for M in ms:
                x, o = xs[M], outs[M]
                best = None
                cfgs = list(triton_configs(N, K, M))
                if a.quick_sweep:
                    cfgs = [c for c in cfgs if c["num_stages"] in (4, 6) and c["num_warps"] == 4]
                tried = 0
                for c in cfgs:
                    try:
                        fp8wo_gemm(x, wqs[0], scs[0], out=o, ws=ws, **c)
                        torch.cuda.synchronize()
                        e = relerr(o, x.float() @ ref_q.t())
                        if not e < 2e-2:
                            continue
                        fns = [lambda wq=wq, sc=sc: fp8wo_gemm(x, wq, sc, out=o, ws=ws, **c) for wq, sc in zip(wqs, scs)]
                        us = time_graph(fns, reps=4, batches=3)
                        tried += 1
                        if best is None or us < best[0]:
                            best = (us, c, e)
                    except Exception as ex:  # noqa: BLE001
                        msg = repr(ex)
                        if "OutOfResources" not in msg and "shared memory" not in msg:
                            print("cfg fail", c, msg[:200], flush=True)
                if best is None:
                    emit(dict(base, M=M, arm="triton", error="no config"))
                    continue
                c = best[1]
                fns = [lambda wq=wq, sc=sc: fp8wo_gemm(x, wq, sc, out=o, ws=ws, **c) for wq, sc in zip(wqs, scs)]
                us = time_graph(fns)
                emit(dict(base, M=M, arm="triton", us=us, GBps=fp8_bytes / us / 1e3, err=best[2], cfg=c, n_cfg=tried))
        # ---- FlashInfer cake (W8A8 reference)
        if "cake" in arms:
            try:
                from flashinfer.gemm import (allocate_kimi_k3_fp8_projection_workspace,
                                             prepare_kimi_k3_fp8_projection,
                                             prepare_kimi_k3_fp8_projection_weights)
                npad = (-N) % 128
                preps = []
                for wq, sc in q:
                    wp = torch.cat([wq, wq.new_zeros(npad, K)], 0) if npad else wq
                    preps.append(prepare_kimi_k3_fp8_projection_weights(wp, sc, N))
                for M in ms:
                    try:
                        x = xs[M]
                        o = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                        wsp = allocate_kimi_k3_fp8_projection_workspace(preps[0], M)
                        runners = [prepare_kimi_k3_fp8_projection(x, p, o, wsp) for p in preps]
                        fns = [(r.launch if hasattr(r, "launch") else r) for r in runners]
                        us = time_graph(fns)
                        fns[0]()
                        err = relerr(o, x.float() @ ref_q.t())
                        emit(dict(base, M=M, arm="cake", us=us, GBps=fp8_bytes / us / 1e3, err=err))
                    except Exception as e:
                        emit(dict(base, M=M, arm="cake", error=repr(e)[:300]))
                del preps
            except Exception as e:
                emit(dict(base, arm="cake", error=repr(e)[:400]))
        del wb, q, xs, outs, ref_q
        torch.cuda.empty_cache()
    fo.close()


if __name__ == "__main__":
    main()
