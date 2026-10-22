"""Python wrapper for fp8wo_tc.cu (built with torch cpp_extension for sm_103a / sm_100a)."""
import os
import torch

_MOD = None


def mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load
        cc = torch.cuda.get_device_capability()
        arch = f"{cc[0]}{cc[1]}a"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cc[0]}.{cc[1]}a")
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp8wo_tc.cu")
        dbg = int(os.environ.get("FP8WO_DBG", "0"))
        _MOD = load(name=f"fp8wo_tc_{arch}_d{dbg}", sources=[src], verbose=False,
                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                                       "--expt-relaxed-constexpr", "-lineinfo", f"-DFP8WO_DBG={dbg}"],
                    build_directory=os.environ.get("FP8WO_BUILD_DIR"))
    return _MOD


class TcWorkspace:
    def __init__(self, device, floats=16 << 20, counters=1 << 14):
        self.ws = torch.empty(floats, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(counters, dtype=torch.int32, device=device)


def tile_weight(q: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] -> uint8 [ceil(N/128), K/128, 128*128]: each 128x128 tile contiguous, chunk-major
    ([8 chunks of 16 K][128 rows][16 B]) so one bulk copy streams it and the widen warps read it without bank conflicts."""
    N, K = q.shape
    npad = (-N) % 128
    u = q.view(torch.uint8)
    if npad:
        u = torch.cat([u, u.new_zeros(npad, K)], 0)
    nt, kb = u.shape[0] // 128, K // 128
    t = u.view(nt, 128, kb, 8, 16).permute(0, 2, 3, 1, 4).contiguous()
    return t.view(nt, kb, 128 * 128)


def fp8wo_tc(x, w, s, out=None, out_dtype=None, split_k=1, stages=4, ws=None, gap_at=1 << 30, gap_len=0, n=None,
             cluster=True, pdl=False):
    M = x.shape[0]
    N = n if w.dim() == 3 else w.shape[0]
    if out is None:
        out = torch.empty((M, N + gap_len), dtype=out_dtype or x.dtype, device=x.device)
    if ws is None:
        ws = _default_ws(x.device)
    mod().run(x, w, s, out, ws.ws, ws.cnt, split_k, gap_at, gap_len, stages, N, int(cluster), int(pdl))
    return out


_WS = {}


def _default_ws(device):
    k = str(device)
    if k not in _WS:
        _WS[k] = TcWorkspace(device)
    return _WS[k]
