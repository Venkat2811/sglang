#!/usr/bin/env python3
"""repro_nvidia_kda_nan.py: NaN from SGLang v0.5.21's NVIDIA KDA prefill pipeline (Kimi-K3, B300).

The server sends a prefill batch of two or more sequences to NvidiaKDAKernel.extend (kda_nvidia.py), which l2-norms
q/k, pads each sequence into a 2K/4K/8K/16K bucket (pad rows: q/k/v/beta 0, gate -1000), converts the state to the
vendor layout and calls the vendored chunk_kda_fwd (sglang.kernels.ops.attention.linear.kda_nvidia_prefill). A
single sequence goes to the Triton chunk_kda instead.

Needs: torch and the sglang package of lmsysorg/sglang:v0.5.21-cu130 (no other code), one SM100/SM103 GPU.

  python3 repro_nvidia_kda_nan.py minimal.pt                 # the small saved case
  python3 repro_nvidia_kda_nan.py inputs-layer5-dev7.pt --full --sweep --make-minimal minimal.pt
                                                             # the captured production call (model layer 5, TP rank 7)

For each case it prints, as token rows with any NaN (and per head where some are NaN):
  vendored pipeline, batched      the wrapper's own steps on the batch as captured (2 rows)
  vendored pipeline, B=1 each     the same steps, every sequence alone
  triton chunk_kda                SGLang's Triton path on the same batch
  reference fp32                  a token-by-token KDA recurrence in fp32
and the largest difference of each finite result against the reference. With --full it also calls
NvidiaKDAKernel.extend and TritonKDAKernel.extend exactly as the server did, and compares with the server's output."""
import argparse

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.linear.kernels.kda_nvidia import (
    NvidiaKDAKernel,
    _from_nvidia_kda_state_layout,
    _to_nvidia_kda_state_layout,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

BUCKETS = (2048, 4096, 8192, 16384)
KERN = None


def kern():
    global KERN
    if KERN is None:
        KERN = NvidiaKDAKernel()
        KERN._ensure_loaded()
    return KERN


def nan_info(o):
    o = o.reshape(-1, o.shape[-2], o.shape[-1])
    ph = torch.isnan(o).any(-1)                                    # [T, H]
    rows = int(ph.any(-1).sum())
    first = int(ph.any(-1).nonzero()[0]) if rows else None
    heads = [(h, int(ph[:, h].sum()), int(ph[:, h].nonzero()[0])) for h in range(ph.shape[1]) if ph[:, h].any()]
    return rows, first, heads


def reference(q, k, v, g, beta, A_log, dt_bias, lb, s0_vk):
    """S <- diag(alpha_t) S; S <- S + beta_t k_t (v_t - S^T k_t)^T; o_t = K^-0.5 S^T q_t (q, k l2-normed, fp32).
    log alpha = lb * sigmoid(exp(A_log) * (g + dt_bias)) (safe gate, as fla/kda.py) or -exp(A_log)*softplus(g+dt_bias).
    q, k, v, g: [T, H, D]; beta: [T, H] (already in [0, 1]); s0_vk: [H, V, K] (SGLang layout)."""
    T, H, K = q.shape
    f = lambda x: x.float()
    q, k, v, g, beta = f(q), f(k), f(v), f(g), f(beta)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    a = torch.exp(f(A_log).reshape(1, H, 1)); x = g + f(dt_bias).reshape(1, H, K)
    la = lb * torch.sigmoid(a * x) if lb is not None else -a * torch.nn.functional.softplus(x)
    alpha = torch.exp(la)
    S = f(s0_vk).transpose(-1, -2).contiguous()
    out = torch.empty(T, H, v.shape[-1], device=q.device)
    for t in range(T):
        S = alpha[t].unsqueeze(-1) * S
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        S = S + (beta[t].reshape(H, 1, 1) * k[t].unsqueeze(-1)) * (v[t] - kv).unsqueeze(1)
        out[t] = (K ** -0.5) * torch.einsum("hk,hkv->hv", q[t], S)
    return out, S


def vendored(q, k, v, g, beta, A_log, dt_bias, lb, s0_vk, lens, bucket=None):
    """kda_nvidia.py extend's steps for one group: l2norm, pad rows to the bucket, layout, chunk_kda_fwd, unpack."""
    kn = kern()
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    B = len(lens); bucket = bucket or next(b for b in BUCKETS if b >= max(lens)); dev = q.device
    st = {n: torch.zeros(B, bucket, H, K if n != "v" else V, dtype=torch.bfloat16, device=dev) for n in ("q", "k", "v")}
    st["g"] = torch.full((B, bucket, H, K), -1000.0, dtype=torch.bfloat16, device=dev)
    st["beta"] = torch.zeros(B, bucket, H, dtype=torch.bfloat16, device=dev)
    s = 0
    for r, L in enumerate(lens):
        st["q"][r, :L].copy_(kn._l2norm(q[0, s:s + L].contiguous()))
        st["k"][r, :L].copy_(kn._l2norm(k[0, s:s + L].contiguous()))
        st["v"][r, :L].copy_(v[0, s:s + L])
        st["g"][r, :L].copy_(g[0, s:s + L].reshape(L, H, K))
        st["beta"][r, :L].copy_(beta[0, s:s + L])
        s += L
    s0 = _to_nvidia_kda_state_layout(s0_vk, head_k_dim=K, head_v_dim=V)
    res = kn._fwd(st["q"], st["k"], st["v"], st["g"], st["beta"], scale=K ** -0.5, initial_state=s0,
                  output_final_state=True, cu_seqlens=None, safe_gate=lb is not None, lower_bound=lb,
                  use_gate_in_kernel=True, A_log=A_log.reshape(-1).float().contiguous(),
                  dt_bias=dt_bias.reshape(-1).float().contiguous())
    o, fs = res[0], res[1]
    return (torch.cat([o[r, :L] for r, L in enumerate(lens)], 0).unsqueeze(0),
            _from_nvidia_kda_state_layout(fs, head_k_dim=K, head_v_dim=V, dtype=torch.float32))


def triton(q, k, v, g, beta, A_log, dt_bias, lb, s0_vk, lens):
    dev = q.device; T, H, K = q.shape[1], q.shape[2], q.shape[3]
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    ssm = s0_vk.clone()
    o = chunk_kda(q=q, k=k, v=v, g=g.reshape(1, T, H, K), beta=beta, initial_state=ssm,
                  initial_state_indices=torch.arange(len(lens), dtype=torch.int32, device=dev),
                  use_qk_l2norm_in_kernel=True, cu_seqlens=cu, A_log=A_log, dt_bias=dt_bias, lower_bound=lb)
    return (o[0] if isinstance(o, tuple) else o), ssm


def line(name, o, ref):
    rows, first, heads = nan_info(o)
    a = o.reshape(ref.shape).float()
    m = torch.isfinite(a).all(-1).all(-1)
    err = (a[m] - ref[m]).abs().max().item() if m.any() else float("nan")
    msg = f"  {name:34s} NaN rows {rows:6d}/{ref.shape[0]}"
    msg += f", first NaN row {first}" if rows else ""
    msg += f", max |diff| vs fp32 ref on finite rows {err:.3g} (ref max {ref.abs().max().item():.3g})"
    if heads:
        msg += f"\n  {'':34s} heads with NaN (head, rows, first row): {heads[:12]}"
    print(msg, flush=True)
    return rows


def gate_cumsum_report(q, g, A_log, dt_bias, lb, lens):
    """Per sequence: cumulative log-decay G[t] = sum_{s<=t} log alpha_s per (head, key dim), from the reference gate.
    Prints the first row where some (head, dim) has G below -88.72 (exp(-G) overflows fp32) and below -103.97
    (exp(G) underflows to 0 in fp32), and the per-head row of the first -88.72 crossing."""
    H, K = q.shape[2], q.shape[3]
    a = torch.exp(A_log.float().reshape(1, H, 1)); s = 0
    for r, L in enumerate(lens):
        x = g[0, s:s + L].float().reshape(L, H, K) + dt_bias.float().reshape(1, H, K)
        la = lb * torch.sigmoid(a * x) if lb is not None else -a * torch.nn.functional.softplus(x)
        G = la.cumsum(0)                                                     # [L, H, K]
        mn = G.amin(-1)                                                      # [L, H]
        def first(th):
            m = (mn < th).any(-1)
            return int(m.nonzero()[0]) if m.any() else None
        ph = [(int((mn[:, h] < -88.72).nonzero()[0]) if (mn[:, h] < -88.72).any() else None) for h in range(H)]
        print(f"  gate: seq {r} ({L} rows) first row with cumulative log-decay < -88.72: {first(-88.72)}, < -103.97: "
              f"{first(-103.97)}; mean log-decay per token {la.mean().item():.4f}; per head first < -88.72: {ph}")
        s += L


def case(d, dev, title, b1=True):
    """Every kernel call gets its own fresh copy of the inputs (some calls are not side-effect free)."""
    def fresh():
        return ([d[n].to(dev).clone() for n in ("q", "k", "v", "g", "beta")],
                d["A_log"].to(dev).clone(), d["dt_bias"].to(dev).clone(), d["slot_states_before"].to(dev).clone())
    (q, k, v, g, beta), A_log, dt_bias, s0 = fresh()
    lb = d["lower_bound"]
    lens = [int(x) for x in d["extend_seq_lens_cpu"]]
    H = q.shape[2]
    print(f"== {title}: sequences {lens}, heads {H}, beta range [{beta.float().min().item():.4f}, "
          f"{beta.float().max().item():.4f}], lower_bound {lb}", flush=True)
    gate_cumsum_report(q, g, A_log, dt_bias, lb, lens)
    refs, s = [], 0
    for r, L in enumerate(lens):
        refs.append(reference(q[0, s:s + L], k[0, s:s + L], v[0, s:s + L], g[0, s:s + L].reshape(L, H, -1),
                              beta[0, s:s + L], A_log, dt_bias, lb, s0[r])[0]); s += L
    ref = torch.cat(refs, 0)
    (q, k, v, g, beta), A_log, dt_bias, s0 = fresh()
    rb = line("vendored pipeline, batched", vendored(q, k, v, g, beta, A_log, dt_bias, lb, s0, lens)[0], ref)
    if b1 and len(lens) > 1:
        s = 0
        for r, L in enumerate(lens):
            (q, k, v, g, beta), A_log, dt_bias, s0 = fresh()
            sl = slice(s, s + L)
            line(f"vendored pipeline, B=1 seq {r} ({L})",
                 vendored(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], A_log, dt_bias, lb, s0[r:r + 1], [L])[0], refs[r])
            s += L
    (q, k, v, g, beta), A_log, dt_bias, s0 = fresh()
    line("triton chunk_kda", triton(q, k, v, g, beta, A_log, dt_bias, lb, s0, lens)[0], ref)
    line("reference fp32 (self)", ref, ref)
    return rb


def sub(d, heads=None, seqs=None, prefix=None):
    """Slice a capture: some heads, some sequences (by index), each cut to `prefix` tokens."""
    lens = [int(x) for x in d["extend_seq_lens_cpu"]]
    starts = [0] + torch.tensor(lens).cumsum(0).tolist()
    seqs = list(range(len(lens))) if seqs is None else seqs
    idx = []; nl = []
    for r in seqs:
        L = lens[r] if prefix is None else min(prefix, lens[r])
        idx.append(torch.arange(starts[r], starts[r] + L)); nl.append(L)
    idx = torch.cat(idx)
    hs = slice(None) if heads is None else heads
    K = d["q"].shape[3]
    out = {n: d[n][:, idx][:, :, hs].contiguous() for n in ("q", "k", "v")}
    g = d["g"].reshape(d["g"].shape[0], d["g"].shape[1], d["q"].shape[2], K)
    out["g"] = g[:, idx][:, :, hs].contiguous()
    out["beta"] = d["beta"][:, idx][:, :, hs].contiguous()
    H = d["q"].shape[2]
    out["A_log"] = d["A_log"].reshape(H)[hs].contiguous()
    out["dt_bias"] = d["dt_bias"].reshape(H, K)[hs].reshape(-1).contiguous()
    out["lower_bound"] = d["lower_bound"]
    out["slot_states_before"] = d["slot_states_before"][seqs][:, hs].contiguous()
    out["extend_seq_lens_cpu"] = nl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--make-minimal", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    d = torch.load(a.path)
    with torch.no_grad():
        case(d, dev, "as given")
        if not a.full:
            return
        lens = [int(x) for x in d["extend_seq_lens_cpu"]]
        # exactly the server's two kernel classes, with the captured keyword arguments
        ci = d["cache_indices"].to(dev)
        s0 = d["slot_states_before"].to(dev)
        n = int(ci.max()) + 2
        ssm = torch.zeros((n,) + tuple(s0.shape[1:]), dtype=getattr(torch, d["ssm_dtype"].split(".")[-1]), device=dev)
        slots = torch.where(ci >= 0, ci, n - 1).long()
        ssm[slots] = s0.to(ssm.dtype)
        okw = {k2: (v2.to(dev) if isinstance(v2, torch.Tensor) else v2) for k2, v2 in d["other_kwargs"].items()}
        okw.pop("track_state", None); okw.pop("track_chunk_idx", None)
        kw = dict(ssm_states=ssm.clone(), cache_indices=ci, query_start_loc=d["query_start_loc"].to(dev),
                  A_log=d["A_log"].to(dev), dt_bias=d["dt_bias"].to(dev), lower_bound=d["lower_bound"], **okw)
        print(f"== server kernel classes, keyword arguments as captured: {sorted(okw)}")
        H = d["q"].shape[2]
        ref = None
        try:
            ref = torch.cat([reference(*(d[nm].to(dev)[0, st:st + L] for nm in ("q", "k", "v")),
                                       d["g"].to(dev)[0, st:st + L].reshape(L, H, -1), d["beta"].to(dev)[0, st:st + L],
                                       d["A_log"].to(dev), d["dt_bias"].to(dev), d["lower_bound"], s0[r])[0]
                             for r, (st, L) in enumerate(zip([0] + torch.tensor(lens).cumsum(0).tolist(), lens))], 0)
        except Exception as e:
            print("reference failed", e)
        qkvgb = [d[nm].to(dev) for nm in ("q", "k", "v", "g", "beta")]
        o1 = NvidiaKDAKernel().extend(*qkvgb, **kw)
        line("NvidiaKDAKernel.extend (server path)", o1[0] if isinstance(o1, tuple) else o1, ref)
        kw["ssm_states"] = ssm.clone()
        o2 = TritonKDAKernel().extend(*qkvgb, **kw)
        line("TritonKDAKernel.extend", o2[0] if isinstance(o2, tuple) else o2, ref)
        try:
            cap = torch.load(a.path.replace("inputs-", "outputs-"))["output"].to(dev)
            line("server output (captured)", cap, ref)
            o1v = o1[0] if isinstance(o1, tuple) else o1
            same = torch.equal(torch.nan_to_num(o1v.float(), nan=12345.0), torch.nan_to_num(cap.float(), nan=12345.0))
            print(f"  standalone NvidiaKDAKernel.extend output bit-identical to the server's: {same}")
        except Exception as e:
            print("no captured output", e)
        # which heads fail in the batch
        _, _, heads = nan_info(vendored(*qkvgb, d["A_log"].to(dev), d["dt_bias"].to(dev), d["lower_bound"], s0, lens)[0])
        bad = [h for h, _, _ in heads]
        print(f"== failing heads in the captured batch: {bad}")
        if a.sweep and bad:
            long_r = max(range(len(lens)), key=lambda r: lens[r]); short_r = min(range(len(lens)), key=lambda r: lens[r])
            print("== length sweep (all heads of the capture; sequences cut from the captured ones)")
            for P in (1024, 2048, 3072, 4096, 8192, lens[long_r]):
                dd = sub(d, seqs=[long_r], prefix=P)
                rows1 = case(dd, dev, f"long sequence alone, first {P} tokens (B=1 through the vendored pipeline)", b1=False)
                if len(lens) > 1:
                    dd2 = sub(d, seqs=[long_r, short_r], prefix=P)
                    case(dd2, dev, f"long first {P} + short ({dd2['extend_seq_lens_cpu'][1]}), B=2", b1=False)
        if a.make_minimal and bad:
            h = bad[0]
            long_r = max(range(len(lens)), key=lambda r: lens[r])
            for P in (1024, 2048, 3072, 4096, 8192, lens[long_r]):
                dd = sub(d, heads=[h], seqs=[long_r], prefix=P)
                rows = case(dd, dev, f"candidate minimal: head {h}, long sequence, first {P} tokens, B=1", b1=False)
                if rows:
                    dd["note"] = (f"Kimi-K3 model layer 5 KDA, TP rank 7 local head {h}, long sequence of the captured "
                                  f"production batch, first {P} tokens, initial state as captured")
                    torch.save(dd, a.make_minimal)
                    print(f"saved {a.make_minimal}")
                    break


if __name__ == "__main__":
    main()
