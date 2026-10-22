#!/usr/bin/env python3
"""real_quant_err.py: block-FP8 (e4m3, 128x128, scale = amax/448) quantization error on real Kimi-K3 bf16 weights.
Per tensor: relative Frobenius error of the weight, and relative error of y = x W^T for x ~ N(0, 1) (64 rows)."""
import json, sys
import torch
from safetensors import safe_open
sys.path.insert(0, "/work/bench")
from fp8wo_triton import dequant_block_fp8, quant_block_fp8

D = "/data00/models/moonshotai/Kimi-K3/f831ab6"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
P = "language_model.model.layers."
names = [P + "2.self_attn.q_proj.weight", P + "2.self_attn.k_proj.weight", P + "2.self_attn.v_proj.weight",
         P + "2.self_attn.g_proj.weight", P + "2.self_attn.o_proj.weight",
         P + "3.self_attn.q_a_proj.weight", P + "3.self_attn.kv_a_proj_with_mqa.weight", P + "3.self_attn.q_b_proj.weight",
         P + "3.self_attn.o_proj.weight", P + "3.self_attn.g_proj.weight",
         P + "2.block_sparse_moe.gate.weight", P + "2.block_sparse_moe.routed_expert_down_proj.weight",
         P + "2.block_sparse_moe.routed_expert_up_proj.weight", P + "2.block_sparse_moe.shared_experts.gate_proj.weight",
         P + "2.block_sparse_moe.shared_experts.up_proj.weight", P + "2.block_sparse_moe.shared_experts.down_proj.weight",
         P + "60.self_attn.q_proj.weight", P + "60.block_sparse_moe.routed_expert_up_proj.weight",
         P + "0.mlp.gate_proj.weight", P + "0.mlp.down_proj.weight", "language_model.lm_head.weight"]
dev = torch.device("cuda:0")
torch.manual_seed(0)
for n in names:
    if n not in idx:
        print(json.dumps(dict(name=n, missing=True)), flush=True); continue
    with safe_open(f"{D}/{idx[n]}", framework="pt", device="cuda:0") as f:
        w = f.get_tensor(n)
    if w.dtype != torch.bfloat16:
        print(json.dumps(dict(name=n, dtype=str(w.dtype))), flush=True); continue
    q, s = quant_block_fp8(w)
    wq = dequant_block_fp8(q, s)
    wf = w.float()
    werr = float((wq - wf).norm() / wf.norm())
    x = torch.randn(64, w.shape[1], device=dev)
    y, yq = x @ wf.t(), x @ wq.t()
    yerr = float((yq - y).norm() / y.norm())
    ymax = float(((yq - y).abs().max(1).values / y.abs().max(1).values).max())
    print(json.dumps(dict(name=n.replace(P, "L"), shape=list(w.shape), w_rel_fro=round(werr, 5), y_rel_fro=round(yerr, 5),
                          y_rel_max=round(ymax, 5), amax=round(float(wf.abs().max()), 4))), flush=True)
    del w, q, s, wq, wf
