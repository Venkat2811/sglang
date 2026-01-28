/**
 * Broker Queue MoE Token Dispatcher - CUDA Kernel
 *
 * Dispatches tokens from router to per-expert broker queues.
 * Replaces DeepEP's RDMA-based dispatch with GPU-resident lock-free queues.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "broker_queue_v2.cuh"

// MoEToken is defined in broker_queue_v2.cuh

// Expert queue metadata
struct ExpertQueueMeta {
    BrokerQueueV2<MoEToken, 256>* queue;  // Broker queue pointer
    uint32_t num_enqueued;                // Tokens enqueued this batch
    uint32_t num_dequeued;                // Tokens processed this batch
};

/**
 * Dispatch kernel: Route tokens to expert queues
 *
 * Each thread block handles a subset of tokens, cooperatively enqueuing
 * them to the appropriate expert queues using work-group operations.
 *
 * @param hidden_states: (num_tokens, hidden_size) token embeddings
 * @param topk_ids: (num_tokens, top_k) expert IDs for each token
 * @param topk_weights: (num_tokens, top_k) router weights
 * @param expert_queues: Array of broker queue pointers (one per expert)
 * @param num_tokens: Total number of tokens to dispatch
 * @param num_experts: Number of local experts
 * @param hidden_size: Hidden dimension
 * @param top_k: Number of experts per token
 */
__global__ void broker_dispatch_kernel(
    const float* __restrict__ hidden_states,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    BrokerQueueV2<MoEToken, 256>** expert_queues,
    int num_tokens,
    int num_experts,
    int hidden_size,
    int top_k
) {
    // Each block processes a subset of tokens
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Each token goes to top_k experts
    for (int k = 0; k < top_k; k++) {
        int expert_id = topk_ids[token_idx * top_k + k];
        if (expert_id < 0 || expert_id >= num_experts) continue;

        float weight = topk_weights[token_idx * top_k + k];

        // Get expert's queue
        auto* queue = expert_queues[expert_id];

        // Create token metadata
        MoEToken token;
        token.hidden_state_ptr = const_cast<float*>(
            &hidden_states[token_idx * hidden_size]
        );
        token.weight = weight;
        token.token_id = token_idx;
        token.expert_id = expert_id;

        // Enqueue to expert's broker queue (single-thread for now)
        // TODO: Use work-group enqueue for batches
        if (lane_id == 0) {
            bool success = queue->enqueue(token);
            if (!success) {
                // Queue full - this shouldn't happen with proper sizing
                printf("WARNING: Expert %d queue full for token %d\\n",
                       expert_id, token_idx);
            }
        }
        __syncwarp();
    }
}

/**
 * Persistent expert kernel: Consumes tokens from queue and computes
 *
 * Each block is assigned to one expert and runs persistently, consuming
 * tokens from the expert's broker queue as they arrive.
 *
 * @param expert_queues: Array of broker queue pointers
 * @param expert_weights_w1: Expert's w1 weights (gate projection)
 * @param expert_weights_w2: Expert's w2 weights (down projection)
 * @param output_buffer: Output buffer for processed tokens
 * @param num_experts: Number of local experts
 * @param hidden_size: Hidden dimension
 * @param intermediate_size: Intermediate dimension (FFN)
 * @param num_tokens_expected: Expected number of tokens per expert
 */
__global__ void broker_expert_persistent_kernel(
    BrokerQueueV2<MoEToken, 256>** expert_queues,
    const float* __restrict__ expert_weights_w1,
    const float* __restrict__ expert_weights_w2,
    float* __restrict__ output_buffer,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    int num_tokens_expected
) {
    // Each block handles one expert
    int expert_id = blockIdx.x;
    if (expert_id >= num_experts) return;

    auto* queue = expert_queues[expert_id];

    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;

    // Process tokens from queue until batch complete
    int tokens_processed = 0;
    while (tokens_processed < num_tokens_expected) {
        MoEToken token;

        // Try to dequeue token
        bool got_token = false;
        if (lane_id == 0) {
            got_token = queue->dequeue(token);
        }

        // Broadcast success to warp
        got_token = __shfl_sync(0xffffffff, got_token, 0);

        if (!got_token) {
            // No token available yet, yield and retry
            // TODO: Use proper backoff strategy
            __nanosleep(100);
            continue;
        }

        // Token available, process it
        // Broadcast token metadata to all threads
        float* input_ptr = (float*)__shfl_sync(0xffffffff,
                                               (unsigned long long)token.hidden_state_ptr, 0);
        float weight = __shfl_sync(0xffffffff, token.weight, 0);
        int token_id = __shfl_sync(0xffffffff, token.token_id, 0);

        // TODO: Actual expert computation (FFN)
        // For now, just placeholder to show structure

        // Simplified: Copy input to output (scaled by weight)
        for (int i = lane_id; i < hidden_size; i += 32) {
            float val = input_ptr[i];
            output_buffer[token_id * hidden_size + i] += val * weight;
        }
        __syncwarp();

        tokens_processed++;
    }
}

// Python module definition
#include <torch/extension.h>

/**
 * Python-compatible wrapper: Initialize expert queues
 * Returns list of queue pointer addresses as int64
 */
std::vector<int64_t> broker_init_queues_py(
    int num_experts,
    int queue_capacity
) {
    std::vector<int64_t> queue_ptrs;

    for (int i = 0; i < num_experts; i++) {
        // Allocate queue on device
        BrokerQueueV2<MoEToken, 256>* d_queue;
        cudaMalloc(&d_queue, sizeof(BrokerQueueV2<MoEToken, 256>));

        // Initialize queue
        cudaMemset(d_queue, 0, sizeof(BrokerQueueV2<MoEToken, 256>));

        queue_ptrs.push_back(reinterpret_cast<int64_t>(d_queue));
    }

    return queue_ptrs;
}

/**
 * Python-compatible wrapper: Dispatch tokens to expert queues
 */
void broker_dispatch_py(
    int64_t hidden_states_ptr,
    int64_t topk_ids_ptr,
    int64_t topk_weights_ptr,
    std::vector<int64_t> expert_queue_ptrs,
    int num_tokens,
    int num_experts,
    int hidden_size,
    int top_k,
    int64_t stream_ptr
) {
    // Cast pointers
    const float* hidden_states = reinterpret_cast<const float*>(hidden_states_ptr);
    const int32_t* topk_ids = reinterpret_cast<const int32_t*>(topk_ids_ptr);
    const float* topk_weights = reinterpret_cast<const float*>(topk_weights_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Build queue pointer array on host
    std::vector<BrokerQueueV2<MoEToken, 256>*> h_queues;
    for (auto ptr : expert_queue_ptrs) {
        h_queues.push_back(reinterpret_cast<BrokerQueueV2<MoEToken, 256>*>(ptr));
    }

    // Allocate device memory for queue pointer array
    BrokerQueueV2<MoEToken, 256>** d_queues;
    cudaMalloc(&d_queues, num_experts * sizeof(BrokerQueueV2<MoEToken, 256>*));

    // Copy queue pointers to device
    cudaMemcpy(d_queues, h_queues.data(),
               num_experts * sizeof(BrokerQueueV2<MoEToken, 256>*),
               cudaMemcpyHostToDevice);

    // Launch one block per token
    int num_blocks = num_tokens;
    int threads_per_block = 32;

    broker_dispatch_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        hidden_states,
        topk_ids,
        topk_weights,
        d_queues,
        num_tokens,
        num_experts,
        hidden_size,
        top_k
    );

    // Sync and free
    cudaStreamSynchronize(stream);
    cudaFree(d_queues);
}

/**
 * Python-compatible wrapper: Launch persistent expert kernels
 */
void broker_launch_experts_py(
    std::vector<int64_t> expert_queue_ptrs,
    int64_t expert_weights_w1_ptr,
    int64_t expert_weights_w2_ptr,
    int64_t output_buffer_ptr,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    int num_tokens_expected,
    int64_t stream_ptr
) {
    // Cast pointers
    const float* expert_weights_w1 = reinterpret_cast<const float*>(expert_weights_w1_ptr);
    const float* expert_weights_w2 = reinterpret_cast<const float*>(expert_weights_w2_ptr);
    float* output_buffer = reinterpret_cast<float*>(output_buffer_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Build queue pointer array on host
    std::vector<BrokerQueueV2<MoEToken, 256>*> h_queues;
    for (auto ptr : expert_queue_ptrs) {
        h_queues.push_back(reinterpret_cast<BrokerQueueV2<MoEToken, 256>*>(ptr));
    }

    // Allocate device memory for queue pointer array
    BrokerQueueV2<MoEToken, 256>** d_queues;
    cudaMalloc(&d_queues, num_experts * sizeof(BrokerQueueV2<MoEToken, 256>*));

    // Copy queue pointers to device
    cudaMemcpy(d_queues, h_queues.data(),
               num_experts * sizeof(BrokerQueueV2<MoEToken, 256>*),
               cudaMemcpyHostToDevice);

    // Launch one persistent block per expert
    int num_blocks = num_experts;
    int threads_per_block = 256;

    broker_expert_persistent_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_queues,
        expert_weights_w1,
        expert_weights_w2,
        output_buffer,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens_expected
    );

    // Sync and free
    cudaStreamSynchronize(stream);
    cudaFree(d_queues);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broker_init_queues", &broker_init_queues_py, "Initialize broker queues",
          py::arg("num_experts"),
          py::arg("queue_capacity"));

    m.def("broker_dispatch", &broker_dispatch_py, "Dispatch tokens to expert queues",
          py::arg("hidden_states_ptr"),
          py::arg("topk_ids_ptr"),
          py::arg("topk_weights_ptr"),
          py::arg("expert_queue_ptrs"),
          py::arg("num_tokens"),
          py::arg("num_experts"),
          py::arg("hidden_size"),
          py::arg("top_k"),
          py::arg("stream_ptr"));

    m.def("broker_launch_experts", &broker_launch_experts_py, "Launch persistent expert kernels",
          py::arg("expert_queue_ptrs"),
          py::arg("expert_weights_w1_ptr"),
          py::arg("expert_weights_w2_ptr"),
          py::arg("output_buffer_ptr"),
          py::arg("num_experts"),
          py::arg("hidden_size"),
          py::arg("intermediate_size"),
          py::arg("num_tokens_expected"),
          py::arg("stream_ptr"));
}
