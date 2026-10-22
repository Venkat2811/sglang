"""DeepEP low-latency dispatch -> FlashInfer trtllm-gen MXFP4 experts (SM100).

The trtllm-gen routed MoE consumes dense token rows with top-k ids, while DeepEP
LL delivers a per-expert packed slab. The adapter in
``sglang.kernels.ops.moe.deepep_ll_mxfp4`` compacts the slab into top-1 rows
(rank-local expert id, weight 1.0), runs the routed MoE with the finalize
skipped, and scatters the permuted GEMM2 rows back into the slab for DeepEP's
combine, which applies the real top-k weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
    )


def _compact_rows(*, slab_rows: int, num_local_experts: int, top_k: int, local_tokens: int) -> int:
    """Static bound on valid (expert, slot) pairs on this rank.

    Each dispatched token occupies at most min(top_k, E) local expert slots, and
    the slab holds R = cap * num_ranks token slots per expert. With
    SGLANG_DEEPEP_MXFP4_UNIFORM_ROWS every rank dispatches at most as many
    tokens as this one (one DP group, equal SP-MoE shards), which tightens R.
    """
    per_token = min(top_k, num_local_experts)
    rows = slab_rows * per_token
    if envs.SGLANG_DEEPEP_MXFP4_UNIFORM_ROWS.get() and local_tokens > 0:
        num_ranks = get_parallel().moe_ep_size
        rows = min(rows, num_ranks * local_tokens * per_token)
    return max(16, -(-rows // 16) * 16)


def fused_experts_deepep_ll_to_flashinfer_mxfp4(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: FlashInferTrtllmGenMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    from sglang.kernels.ops.moe.deepep_ll_mxfp4 import (
        deepep_ll_gather_mxfp8,
        deepep_ll_scatter,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, _ = (
        dispatch_output
    )
    assert runner_config.activation == "situ", runner_config.activation
    assert quant_info.flashinfer_mxfp4_moe_precision == "default"
    assert hidden_states_scale is not None, (
        "DeepEP LL -> FlashInfer MXFP4 needs the FP8 dispatch (UE8M0 scales)"
    )
    num_local_experts, slab_rows, hidden = hidden_states.shape
    assert num_local_experts == quant_info.local_num_experts
    assert hidden == quant_info.hidden_size
    top_k = topk_ids.shape[1]

    rows = _compact_rows(
        slab_rows=slab_rows,
        num_local_experts=num_local_experts,
        top_k=top_k,
        local_tokens=topk_ids.shape[0],
    )
    # Rank-local ids: [0, E) local, E = the padding id trtllm-gen skips; the
    # expert count is rounded to FlashInfer's multiple of 4.
    x_rows, x_scale, packed_ids, row_map = deepep_ll_gather_mxfp8(
        hidden_states,
        hidden_states_scale,
        masked_m,
        num_rows=rows,
        invalid_id=num_local_experts,
    )
    gemm2_out, _, permuted_idx = trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed_ids,
        routing_bias=None,
        hidden_states=x_rows,
        hidden_states_scale=x_scale,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale,
        gemm1_bias=None,
        gemm1_alpha=quant_info.gemm1_alpha,
        gemm1_beta=quant_info.gemm1_clamp_limit,
        gemm1_clamp_limit=None,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=-(-(num_local_experts + 1) // 4) * 4,
        top_k=1,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=0,
        local_num_experts=num_local_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        activation_type=ActivationType.Situ.value,
        tune_max_num_tokens=next_power_of_2(rows),
        do_finalize=False,
        enable_pdl=trtllm_moe_enable_pdl(rows),
    )
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        out = torch.empty(
            (num_local_experts, slab_rows, hidden),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
    deepep_ll_scatter(gemm2_out, row_map, out, perm=permuted_idx, masked_m=masked_m)
    return DeepEPLLCombineInput(
        hidden_states=out, topk_ids=topk_ids, topk_weights=topk_weights
    )
