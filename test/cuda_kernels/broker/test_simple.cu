/**
 * Simple Broker Queue Test - Minimal version for debugging
 */

#include <cuda_runtime.h>
#include <cstdio>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue.cuh"

__global__ void simple_test() {
    printf("GPU: Test starting\n");

    // Allocate queue in global memory for simplicity
    BrokerQueue<int, 1024>* queue = new BrokerQueue<int, 1024>();
    queue->init();

    printf("GPU: Queue initialized\n");

    // Single thread test
    if (threadIdx.x == 0) {
        int value = 42;
        printf("GPU: Enqueuing %d\n", value);

        // Manual enqueue to debug
        uint64_t ticket = atomicAdd((unsigned long long*)&queue->head, 1ULL);
        printf("GPU: Got ticket %llu, count=%llu\n", ticket, queue->count);

        // Wait for turn
        while (queue->count != ticket) {
            // Spin
        }
        printf("GPU: Got turn\n");

        // Write data
        queue->data[ticket % 1024] = value;

        // Advance count
        atomicAdd((unsigned long long*)&queue->count, 1ULL);
        printf("GPU: Enqueue complete, count now=%llu\n", queue->count);

        // Dequeue
        printf("GPU: Dequeuing\n");
        uint64_t d_ticket = atomicAdd((unsigned long long*)&queue->tail, 1ULL);
        printf("GPU: Dequeue ticket %llu, count=%llu\n", d_ticket, queue->count);

        while (queue->count != d_ticket) {
            // Spin
        }
        printf("GPU: Got dequeue turn\n");

        int retrieved = queue->data[d_ticket % 1024];
        printf("GPU: Retrieved %d\n", retrieved);

        atomicAdd((unsigned long long*)&queue->count, 1ULL);
        printf("GPU: Dequeue complete\n");

        if (retrieved == value) {
            printf("GPU: TEST PASSED\n");
        } else {
            printf("GPU: TEST FAILED: expected %d, got %d\n", value, retrieved);
        }
    }

    delete queue;
}

int main() {
    printf("Starting simple broker queue test\n");

    simple_test<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Test complete\n");
    return 0;
}
