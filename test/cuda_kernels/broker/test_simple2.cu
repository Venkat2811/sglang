/**
 * Simple Broker Queue Test - Even/Odd ticket version
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void simple_test() {
    printf("GPU: Test starting\n");

    // Manual implementation to verify algorithm
    __shared__ uint64_t head;    // Even tickets: 0, 2, 4, 6, ...
    __shared__ uint64_t tail;    // Odd tickets: 1, 3, 5, 7, ...
    __shared__ uint64_t count;   // Alternates: 0, 1, 2, 3, 4, ...
    __shared__ int data[1024];

    if (threadIdx.x == 0) {
        head = 0;
        tail = 1;
        count = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int value = 42;

        // ENQUEUE (even ticket)
        printf("GPU: Enqueuing %d\n", value);
        uint64_t enq_ticket = atomicAdd((unsigned long long*)&head, 2ULL);  // Get even ticket
        printf("GPU: Enqueue ticket=%llu, count=%llu\n", enq_ticket, count);

        while (count != enq_ticket) {
            // Wait for turn
        }
        printf("GPU: Got enqueue turn\n");

        data[(enq_ticket / 2) % 1024] = value;  // Divide by 2 for actual slot
        atomicAdd((unsigned long long*)&count, 1ULL);  // Advance by 1
        printf("GPU: Enqueue complete, count now=%llu\n", count);

        // DEQUEUE (odd ticket)
        printf("GPU: Dequeuing\n");
        uint64_t deq_ticket = atomicAdd((unsigned long long*)&tail, 2ULL);  // Get odd ticket
        printf("GPU: Dequeue ticket=%llu, count=%llu\n", deq_ticket, count);

        while (count != deq_ticket) {
            // Wait for turn
        }
        printf("GPU: Got dequeue turn\n");

        int retrieved = data[(deq_ticket / 2) % 1024];  // Divide by 2 for actual slot
        printf("GPU: Retrieved %d (expected %d)\n", retrieved, value);

        atomicAdd((unsigned long long*)&count, 1ULL);  // Advance by 1
        printf("GPU: Dequeue complete, count now=%llu\n", count);

        if (retrieved == value) {
            printf("GPU: TEST PASSED ✓\n");
        } else {
            printf("GPU: TEST FAILED ✗\n");
        }
    }
}

int main() {
    printf("Starting simple broker queue test (even/odd tickets)\n");

    simple_test<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Test complete\n");
    return 0;
}
