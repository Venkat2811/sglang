/**
 * Minimal test using the broker_queue.cuh header
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void test_single_enqueue_dequeue(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
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

int main() {
    printf("Testing broker_queue.cuh...\n");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_single_enqueue_dequeue<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));

    if (h_result == 0) {
        printf("✓ TEST PASSED\n");
        return 0;
    } else {
        printf("✗ TEST FAILED (code=%d)\n", h_result);
        return 1;
    }
}
