/**
 * Test for Broker Queue V2 (simplified sequential tickets)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue_v2.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test 1: Single thread enqueue/dequeue
__global__ void test_single(int* result) {
    __shared__ BrokerQueueV2<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();

        int value = 42;
        bool success = queue.enqueue(value);
        if (!success) {
            *result = -1;
            return;
        }

        int retrieved;
        success = queue.dequeue(retrieved);
        if (!success) {
            *result = -2;
            return;
        }

        *result = (retrieved == value) ? 0 : -3;
    }
}

// Test 2: Warp-level batch operations
__global__ void test_warp_batch(int* result) {
    __shared__ BrokerQueueV2<int, 1024> queue;
    __shared__ int items[16];
    __shared__ int retrieved[16];

    if (threadIdx.x == 0) {
        queue.init();
        *result = 0;  // Initialize result
    }
    __syncthreads();

    const int NUM_ITEMS = 16;
    int lane_id = threadIdx.x % 32;

    // Leader prepares test data in shared memory
    if (lane_id == 0) {
        for (int i = 0; i < NUM_ITEMS; i++) {
            items[i] = 100 + i;
        }
    }
    __syncwarp();

    // ALL warp threads call enqueue_wg together
    bool success = queue.enqueue_wg(items, NUM_ITEMS, lane_id, 32);
    if (!success && lane_id == 0) {
        printf("GPU: Enqueue failed\n");
        *result = -1;
        return;
    }

    if (lane_id == 0) {
        printf("GPU: Enqueue success, size=%llu\n", queue.size());
    }
    __syncwarp();

    // ALL warp threads call dequeue_wg together
    success = queue.dequeue_wg(retrieved, NUM_ITEMS, lane_id, 32);
    if (!success && lane_id == 0) {
        printf("GPU: Dequeue failed, size=%llu\n", queue.size());
        *result = -2;
        return;
    }
    __syncwarp();

    // Leader verifies results
    if (lane_id == 0) {
        printf("GPU: Verifying %d items\n", NUM_ITEMS);
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (retrieved[i] != items[i]) {
                printf("GPU: Mismatch at index %d: expected %d, got %d\n",
                       i, items[i], retrieved[i]);
                *result = -3;
                return;
            }
        }

        printf("GPU: All items verified\n");
    }
}

// Test 3: Multiple operations
__global__ void test_multiple(int* result) {
    __shared__ BrokerQueueV2<int, 1024> queue;
    __shared__ int items[8];
    __shared__ int retrieved[8];

    if (threadIdx.x == 0) {
        queue.init();
        *result = 0;  // Initialize result
    }
    __syncthreads();

    int lane_id = threadIdx.x % 32;

    // Enqueue 5 batches
    for (int batch = 0; batch < 5; batch++) {
        // Leader prepares data
        if (lane_id == 0) {
            for (int i = 0; i < 8; i++) {
                items[i] = batch * 100 + i;
            }
        }
        __syncwarp();

        // ALL warp threads call enqueue_wg
        bool success = queue.enqueue_wg(items, 8, lane_id, 32);
        if (!success && lane_id == 0) {
            *result = -1;
            return;
        }
        __syncwarp();
    }

    // Dequeue and verify
    for (int batch = 0; batch < 5; batch++) {
        // ALL warp threads call dequeue_wg
        bool success = queue.dequeue_wg(retrieved, 8, lane_id, 32);
        if (!success && lane_id == 0) {
            *result = -2;
            return;
        }
        __syncwarp();

        // Leader verifies
        if (lane_id == 0) {
            for (int i = 0; i < 8; i++) {
                if (retrieved[i] != batch * 100 + i) {
                    *result = -3;
                    return;
                }
            }
        }
        __syncwarp();
    }
}

int main() {
    printf("Broker Queue V2 Tests\n");
    printf("=====================\n\n");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    // Test 1
    printf("[TEST 1] Single thread... ");
    test_single<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%s\n", h_result == 0 ? "PASS ✓" : "FAIL ✗");
    if (h_result != 0) return 1;

    // Test 2
    printf("[TEST 2] Warp-level batch... ");
    test_warp_batch<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%s\n", h_result == 0 ? "PASS ✓" : "FAIL ✗");
    if (h_result != 0) return 1;

    // Test 3
    printf("[TEST 3] Multiple operations... ");
    test_multiple<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%s\n", h_result == 0 ? "PASS ✓" : "FAIL ✗");
    if (h_result != 0) return 1;

    CUDA_CHECK(cudaFree(d_result));

    printf("\n=====================\n");
    printf("All tests PASSED ✓\n");
    printf("=====================\n");

    return 0;
}
