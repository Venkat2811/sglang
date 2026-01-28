"""
Python bindings for broker queue CUDA kernels.
"""

import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Try to load compiled kernel
_has_broker_kernel = False
try:
    # Set library path for PyTorch dependencies
    import sys
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if torch_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Add user site-packages to path
    user_site = os.path.expanduser('~/.local/lib/python3.12/site-packages')
    if user_site not in sys.path:
        sys.path.insert(0, user_site)

    import broker_moe_kernel
    _has_broker_kernel = True
    logger.info("✓ Loaded broker queue CUDA kernels")
except (ImportError, OSError) as e:
    logger.warning(f"Broker queue CUDA kernels not available: {e}")
    logger.warning("Using fallback mode")


def init_expert_queues(
    num_experts: int,
    queue_capacity: int = 256,
    device: str = 'cuda'
) -> List[int]:
    """
    Initialize broker queues for each expert.

    Args:
        num_experts: Number of local experts
        queue_capacity: Maximum tokens per queue
        device: Device to allocate queues on

    Returns:
        List of queue pointers (as int64)
    """

    if not _has_broker_kernel:
        # Fallback: return dummy pointers
        logger.debug(f"Fallback: allocated {num_experts} dummy queues")
        return [0] * num_experts

    # Use CUDA kernel to initialize queues
    queue_ptrs = broker_moe_kernel.broker_init_queues(num_experts, queue_capacity)
    logger.debug(f"Initialized {num_experts} broker queues (capacity={queue_capacity})")
    return queue_ptrs


def dispatch_tokens_to_queues(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_queue_ptrs: List[int],
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Dispatch tokens to expert queues using broker queue algorithm.

    Args:
        hidden_states: (num_tokens, hidden_size) token embeddings
        topk_ids: (num_tokens, top_k) expert IDs
        topk_weights: (num_tokens, top_k) router weights
        expert_queue_ptrs: List of expert queue pointers
        stream: CUDA stream for async execution
    """

    if not _has_broker_kernel:
        logger.debug(f"Fallback dispatch: {hidden_states.shape[0]} tokens")
        return

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    top_k = topk_ids.shape[1]
    num_experts = len(expert_queue_ptrs)

    # Get CUDA stream pointer
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    # Call CUDA kernel with int64 pointers
    broker_moe_kernel.broker_dispatch(
        hidden_states.data_ptr(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        expert_queue_ptrs,  # Already list of int64
        num_tokens,
        num_experts,
        hidden_size,
        top_k,
        stream_ptr,
    )

    logger.debug(f"Dispatched {num_tokens} tokens to {num_experts} experts")


def launch_persistent_experts(
    expert_queue_ptrs: List[int],
    expert_weights_w1: torch.Tensor,
    expert_weights_w2: torch.Tensor,
    output_buffer: torch.Tensor,
    num_tokens_expected: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """
    Launch persistent expert kernels that consume from queues.

    Args:
        expert_queue_ptrs: List of expert queue pointers
        expert_weights_w1: Expert gate projection weights
        expert_weights_w2: Expert down projection weights
        output_buffer: Output buffer for results
        num_tokens_expected: Expected tokens per expert
        stream: CUDA stream for async execution
    """

    if not _has_broker_kernel:
        logger.debug(f"Fallback expert computation")
        return

    num_experts = len(expert_queue_ptrs)
    hidden_size = output_buffer.shape[1]
    intermediate_size = expert_weights_w1.shape[0] // num_experts

    # Get CUDA stream pointer
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    # Call CUDA kernel with int64 pointers
    broker_moe_kernel.broker_launch_experts(
        expert_queue_ptrs,  # Already list of int64
        expert_weights_w1.data_ptr(),
        expert_weights_w2.data_ptr(),
        output_buffer.data_ptr(),
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens_expected,
        stream_ptr,
    )

    logger.debug(f"Launched persistent kernels for {num_experts} experts")
