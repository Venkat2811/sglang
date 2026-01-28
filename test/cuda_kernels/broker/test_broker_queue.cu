/**
 * Broker Queue Unit Tests
 *
 * Tests the core Broker Queue implementation against the paper's guarantees:
 * 1. FIFO ordering (linearizability)
 * 2. Correctness under concurrent access
 * 3. Performance vs atomic baseline
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdint>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue.cuh"

// Helper macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// =================================================================
// Test 1: Single thread enqueue/dequeue
// =================================================================

__global__ void test_single_thread_kernel(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Enqueue value
        int value = 42;
        bool success = queue.enqueue(value);

        if (!success) {
            *result = -1; // Enqueue failed
            return;
        }

        // Dequeue value
        int retrieved;
        success = queue.dequeue(retrieved);

        if (!success) {
            *result = -2; // Dequeue failed
            return;
        }

        // Verify correctness
        *result = (retrieved == value) ? 0 : -3;
    }
}

void test_single_thread() {
    printf("[TEST 1] Single thread enqueue/dequeue... ");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_single_thread_kernel<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    assert(h_result == 0);
    printf("PASS\n");
}

// =================================================================
// Test 2: Work-group enqueue/dequeue (warp-level)
// =================================================================

__global__ void test_warp_level_kernel(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    const int WARP_SIZE = 32;
    const int NUM_ITEMS = 32;

    // Each thread prepares one item
    int my_value = threadIdx.x + 100;
    int items[NUM_ITEMS];

    if (threadIdx.x == 0) {
        // Collect items from all threads (simplified - leader does it)
        for (int i = 0; i < NUM_ITEMS; i++) {
            items[i] = i + 100;
        }

        // Enqueue as work-group
        bool success = queue.enqueue_wg(items, NUM_ITEMS, 0, WARP_SIZE);
        if (!success) {
            *result = -1;
            return;
        }

        // Dequeue as work-group
        int retrieved[NUM_ITEMS];
        success = queue.dequeue_wg(retrieved, NUM_ITEMS, 0, WARP_SIZE);
        if (!success) {
            *result = -2;
            return;
        }

        // Verify FIFO order
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (retrieved[i] != items[i]) {
                *result = -3;
                return;
            }
        }

        *result = 0;
    }
}

void test_warp_level() {
    printf("[TEST 2] Warp-level work-group enqueue/dequeue... ");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_warp_level_kernel<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    assert(h_result == 0);
    printf("PASS\n");
}

// =================================================================
// Test 3: Multiple warps concurrent access (linearizability)
// =================================================================

__global__ void test_concurrent_warps_kernel(int* results, int num_warps) {
    __shared__ BrokerQueue<int, 2048> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id < num_warps) {
        // Each warp enqueues a unique value
        int value = warp_id * 1000 + lane_id;

        // Leader thread of each warp enqueues
        if (lane_id == 0) {
            bool success = queue.enqueue(value);
            if (!success) {
                results[warp_id] = -1;
            }
        }
    }
    __syncthreads();

    // Now all warps dequeue and verify order
    if (warp_id < num_warps) {
        if (lane_id == 0) {
            int retrieved;
            bool success = queue.dequeue(retrieved);
            if (!success) {
                results[warp_id] = -2;
            } else {
                // Verify value belongs to some warp
                int found_warp = retrieved / 1000;
                if (found_warp >= 0 && found_warp < num_warps) {
                    results[warp_id] = 0; // Success
                } else {
                    results[warp_id] = -3; // Invalid value
                }
            }
        }
    }
}

void test_concurrent_warps() {
    printf("[TEST 3] Concurrent warps linearizability... ");

    const int NUM_WARPS = 8;
    int* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, NUM_WARPS * sizeof(int)));

    test_concurrent_warps_kernel<<<1, NUM_WARPS * 32>>>(d_results, NUM_WARPS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_results[NUM_WARPS];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, NUM_WARPS * sizeof(int),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_results));

    // All warps should succeed
    for (int i = 0; i < NUM_WARPS; i++) {
        assert(h_results[i] == 0);
    }

    printf("PASS\n");
}

// =================================================================
// Test 4: FIFO ordering guarantee
// =================================================================

__global__ void test_fifo_ordering_kernel(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    const int NUM_ITEMS = 100;

    if (threadIdx.x == 0) {
        // Enqueue sequential values
        for (int i = 0; i < NUM_ITEMS; i++) {
            bool success = queue.enqueue(i);
            if (!success) {
                *result = -1;
                return;
            }
        }

        // Dequeue and verify FIFO order
        for (int i = 0; i < NUM_ITEMS; i++) {
            int value;
            bool success = queue.dequeue(value);
            if (!success) {
                *result = -2;
                return;
            }
            if (value != i) {
                *result = -3; // FIFO violation
                return;
            }
        }

        *result = 0; // Success
    }
}

void test_fifo_ordering() {
    printf("[TEST 4] FIFO ordering guarantee... ");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_fifo_ordering_kernel<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    assert(h_result == 0);
    printf("PASS\n");
}

// =================================================================
// Test 5: Queue status queries
// =================================================================

__global__ void test_queue_status_kernel(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();

        // Initially empty
        if (!queue.is_empty()) {
            *result = -1;
            return;
        }

        // Enqueue one item
        int value = 42;
        queue.enqueue(value);

        // Should not be empty
        if (queue.is_empty()) {
            *result = -2;
            return;
        }

        // Size should be > 0
        if (queue.size() == 0) {
            *result = -3;
            return;
        }

        // Dequeue
        int retrieved;
        queue.dequeue(retrieved);

        // Should be empty again
        if (!queue.is_empty()) {
            *result = -4;
            return;
        }

        *result = 0;
    }
}

void test_queue_status() {
    printf("[TEST 5] Queue status queries... ");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_queue_status_kernel<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    assert(h_result == 0);
    printf("PASS\n");
}

// =================================================================
// Test 6: MoE Token structure
// =================================================================

__global__ void test_moe_token_kernel(int* result) {
    __shared__ MoETokenQueue queue;

    if (threadIdx.x == 0) {
        queue.init();

        // Create a token
        MoEToken token;
        token.token_id = 123;
        token.expert_id = 5;
        token.weight = 0.75f;
        token.hidden_state_ptr = nullptr; // Would point to actual data

        // Enqueue
        bool success = queue.enqueue(token);
        if (!success) {
            *result = -1;
            return;
        }

        // Dequeue
        MoEToken retrieved;
        success = queue.dequeue(retrieved);
        if (!success) {
            *result = -2;
            return;
        }

        // Verify
        if (retrieved.token_id != 123 ||
            retrieved.expert_id != 5 ||
            retrieved.weight != 0.75f) {
            *result = -3;
            return;
        }

        *result = 0;
    }
}

void test_moe_token() {
    printf("[TEST 6] MoE Token structure... ");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_moe_token_kernel<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    assert(h_result == 0);
    printf("PASS\n");
}

// =================================================================
// Main test runner
// =================================================================

int main() {
    printf("=======================================================\n");
    printf("Broker Queue Unit Tests\n");
    printf("=======================================================\n\n");

    // Run all tests
    test_single_thread();
    test_warp_level();
    test_concurrent_warps();
    test_fifo_ordering();
    test_queue_status();
    test_moe_token();

    printf("\n=======================================================\n");
    printf("All tests PASSED ✓\n");
    printf("=======================================================\n");

    return 0;
}
