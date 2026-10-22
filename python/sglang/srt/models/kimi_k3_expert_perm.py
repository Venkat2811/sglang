"""Kimi-K3 expert placement helpers (env-gated, off by default).

1. Load-time per-layer expert permutation (``SGLANG_K3_EXPERT_PERM=<file.npy>``).
   The file holds ``slot_to_expert[num_hidden_layers, num_experts]`` (int): row l
   says which checkpoint expert lands in slot s of layer l. Slot s lives on EP
   rank ``s // (num_experts / ep_size)``, so a row decides each expert's rank.
   Loading puts expert e's weights (and block scales) in slot ``pos[l][e]`` and
   reorders the router gate rows and ``e_score_correction_bias`` the same way,
   so the router emits slot ids directly: no runtime remap, the fused routing
   path is untouched. Top-k selection, the renormalized weights and each
   token's weighted expert sum are unchanged by a relabel; only which rank
   computes which expert moves (the TP all-reduce then adds the per-rank
   partial sums in a different grouping, so outputs can differ in the last
   bits).

2. Padded-row masking for the routed MoE (``SGLANG_K3_MASK_PAD_ROUTING=1``).
   CUDA-graph padding rows carry stale activations; their router logits are
   not finite, the radix top-k then picks experts 0..15 (all on EP rank 0),
   and rank 0 runs expert GEMMs for rows nobody reads. With the flag the fused
   route+quant kernel reads the graph's ``num_token_non_padded`` scalar and
   writes id -1 / weight 0 for rows past it (packed 0xFFFF0000), which every
   rank treats as a non-local expert: no expert work, no rank-0 hot spot.

3. A routing counter for building the permutation
   (``SGLANG_K3_ROUTE_COUNTS_DIR=<dir>``): rank 0, decode forwards only, counts
   per layer and expert for real rows and padded rows separately, plus a
   per-step ring of per-layer expert hits; dumped to the directory when a
   ``dump.req`` file appears there.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forward context: the model forward stashes the batch facts the MoE layers
# need (decode or not, the device scalar num_token_non_padded). The scalar is
# the CUDA-graph buffer itself during capture, so captured kernels read the
# replay-time value.
# ---------------------------------------------------------------------------
_FWD_IS_DECODE = False
_FWD_NUM_TOKEN_NON_PADDED: Optional[torch.Tensor] = None


def set_forward_ctx(forward_batch) -> None:
    global _FWD_IS_DECODE, _FWD_NUM_TOKEN_NON_PADDED
    try:
        _FWD_IS_DECODE = bool(forward_batch.forward_mode.is_decode())
    except Exception:
        _FWD_IS_DECODE = False
    _FWD_NUM_TOKEN_NON_PADDED = getattr(forward_batch, "num_token_non_padded", None)


def num_token_non_padded() -> Optional[torch.Tensor]:
    return _FWD_NUM_TOKEN_NON_PADDED


def mask_pad_routing_enabled() -> bool:
    return os.environ.get("SGLANG_K3_MASK_PAD_ROUTING", "0") == "1"


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------
_PERM_LOADED = False
_SLOT_TO_EXPERT: Optional[np.ndarray] = None  # [L, E]
_EXPERT_TO_SLOT: Optional[np.ndarray] = None  # [L, E]


def _load_perm() -> None:
    global _PERM_LOADED, _SLOT_TO_EXPERT, _EXPERT_TO_SLOT
    if _PERM_LOADED:
        return
    _PERM_LOADED = True
    path = os.environ.get("SGLANG_K3_EXPERT_PERM", "")
    if not path:
        return
    s2e = np.load(path).astype(np.int64)
    assert s2e.ndim == 2, f"{path}: expected [layers, experts], got {s2e.shape}"
    num_experts = s2e.shape[1]
    e2s = np.empty_like(s2e)
    for l in range(s2e.shape[0]):
        row = s2e[l]
        assert np.array_equal(
            np.sort(row), np.arange(num_experts)
        ), f"{path}: row {l} is not a permutation of 0..{num_experts - 1}"
        e2s[l, row] = np.arange(num_experts)
    _SLOT_TO_EXPERT, _EXPERT_TO_SLOT = s2e, e2s
    moved = int((s2e != np.arange(num_experts)[None, :]).any(axis=1).sum())
    logger.info(
        "K3 expert permutation loaded from %s: %d layers x %d experts, %d layers "
        "not identity",
        path,
        s2e.shape[0],
        num_experts,
        moved,
    )


def perm_active() -> bool:
    _load_perm()
    return _SLOT_TO_EXPERT is not None


def slot_of(layer_id: Optional[int], expert_id: int) -> int:
    """Slot (physical global expert id) that checkpoint expert ``expert_id`` of
    layer ``layer_id`` loads into."""
    _load_perm()
    if _EXPERT_TO_SLOT is None or layer_id is None or layer_id >= len(_EXPERT_TO_SLOT):
        return expert_id
    return int(_EXPERT_TO_SLOT[layer_id, expert_id])


def permute_router_(moe_module, layer_id: int) -> bool:
    """Reorder the router gate rows and the correction bias of one K3 MoE layer
    so router output index s scores expert slot_to_expert[layer][s]. Must run
    before the front weights are merged (the merge views the gate rows).
    Idempotent per module."""
    _load_perm()
    if _SLOT_TO_EXPERT is None or getattr(moe_module, "_k3_perm_applied", False):
        return False
    if layer_id >= len(_SLOT_TO_EXPERT):
        return False
    s2e = _SLOT_TO_EXPERT[layer_id]
    if np.array_equal(s2e, np.arange(len(s2e))):
        moe_module._k3_perm_applied = True
        return False
    gate = moe_module.gate
    idx = torch.as_tensor(s2e, dtype=torch.long, device=gate.weight.device)
    w = gate.weight.data
    assert w.shape[0] == len(s2e), (w.shape, len(s2e))
    w.copy_(w.index_select(0, idx))
    bias = gate.e_score_correction_bias
    if bias is not None:
        b = bias.data
        b.copy_(b.index_select(0, idx.to(b.device)))
    moe_module._k3_perm_applied = True
    return True


# ---------------------------------------------------------------------------
# Routing counter
# ---------------------------------------------------------------------------
class _RouteRecorder:
    def __init__(self, out_dir: str, num_layers: int, num_experts: int, ring: int):
        self.out_dir = out_dir
        self.L, self.E, self.S = num_layers, num_experts, ring
        dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.counts = torch.zeros(num_layers, num_experts, dtype=torch.int64, device=dev)
        self.pad_counts = torch.zeros(
            num_layers, num_experts, dtype=torch.int64, device=dev
        )
        self.ring = torch.zeros(ring, num_layers, num_experts, dtype=torch.uint8, device=dev)
        self.ring_rows = torch.full((ring,), -1, dtype=torch.int32, device=dev)
        self.ring_bs = torch.full((ring,), -1, dtype=torch.int32, device=dev)
        self.step = torch.zeros(1, dtype=torch.int64, device=dev)
        self.slot = torch.zeros(1, dtype=torch.int64, device=dev)
        self.first_layer: Optional[int] = None
        self._thread = threading.Thread(target=self._dump_loop, daemon=True)
        self._thread.start()
        logger.info(
            "K3 route recorder on (rank 0): %d layers x %d experts, ring %d steps, "
            "dir %s",
            num_layers,
            num_experts,
            ring,
            out_dir,
        )

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if self.first_layer is None or layer_id < self.first_layer:
            self.first_layer = layer_id
        M, K = topk_ids.shape
        dev = topk_ids.device
        if layer_id == self.first_layer:
            torch.remainder(self.step, self.S, out=self.slot)
            self.step.add_(1)
            ntnp = num_token_non_padded()
            rows = (
                ntnp.to(torch.int32).view(1)
                if ntnp is not None
                else torch.full((1,), M, dtype=torch.int32, device=dev)
            )
            self.ring_rows.index_copy_(0, self.slot, rows)
            self.ring_bs.index_copy_(
                0, self.slot, torch.full((1,), M, dtype=torch.int32, device=dev)
            )
        ntnp = num_token_non_padded()
        ar = torch.arange(M, device=dev)
        valid = ar < ntnp.to(ar.dtype) if ntnp is not None else torch.ones_like(ar, dtype=torch.bool)
        ids = topk_ids.to(torch.int64)
        ok = ids >= 0
        flat = ids.clamp(min=0).view(-1)
        w_real = (ok & valid[:, None]).to(torch.int64).view(-1)
        w_pad = (ok & ~valid[:, None]).to(torch.int64).view(-1)
        self.counts[layer_id].index_add_(0, flat, w_real)
        self.pad_counts[layer_id].index_add_(0, flat, w_pad)
        hits = torch.zeros(self.E, dtype=torch.int64, device=dev)
        hits.index_add_(0, flat, w_real)
        self.ring.select(1, layer_id).index_copy_(
            0, self.slot, hits.clamp(max=255).to(torch.uint8).view(1, -1)
        )

    def _dump_loop(self) -> None:
        req = os.path.join(self.out_dir, "dump.req")
        torch.cuda.set_device(self.device)
        stream = torch.cuda.Stream(device=self.device)
        while True:
            time.sleep(2.0)
            if not os.path.exists(req):
                continue
            try:
                tag = open(req).read().strip() or "dump"
                with torch.cuda.stream(stream):
                    data = dict(
                        counts=self.counts.cpu().numpy(),
                        pad_counts=self.pad_counts.cpu().numpy(),
                        ring=self.ring.cpu().numpy(),
                        ring_rows=self.ring_rows.cpu().numpy(),
                        ring_bs=self.ring_bs.cpu().numpy(),
                        step=self.step.cpu().numpy(),
                    )
                tmp = os.path.join(self.out_dir, f"{tag}.tmp.npz")
                np.savez(tmp, **data)
                os.replace(tmp, os.path.join(self.out_dir, f"{tag}.npz"))
                logger.info("K3 route recorder dumped %s (step %d)", tag, int(data["step"][0]))
            except Exception as e:  # pragma: no cover - diagnostics only
                logger.warning("K3 route recorder dump failed: %r", e)
            finally:
                try:
                    os.remove(req)
                except OSError:
                    pass


_RECORDER: Optional[_RouteRecorder] = None
_RECORDER_INIT = False


def recorder(num_layers: int, num_experts: int) -> Optional[_RouteRecorder]:
    """The rank-0 routing counter, or None when off (env unset, other ranks)."""
    global _RECORDER, _RECORDER_INIT
    if _RECORDER_INIT:
        return _RECORDER
    _RECORDER_INIT = True
    out_dir = os.environ.get("SGLANG_K3_ROUTE_COUNTS_DIR", "")
    if not out_dir:
        return None
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    ring = int(os.environ.get("SGLANG_K3_ROUTE_RING", "16384"))
    _RECORDER = _RouteRecorder(out_dir, num_layers, num_experts, ring)
    return _RECORDER


def maybe_record(layer_id: int, num_layers: int, num_experts: int, topk_ids) -> None:
    """Count one MoE layer's routing (decode forwards, rank 0, env-gated)."""
    if topk_ids is None or not _FWD_IS_DECODE:
        return
    rec = recorder(num_layers, num_experts)
    if rec is not None:
        rec.record(layer_id, topk_ids)
