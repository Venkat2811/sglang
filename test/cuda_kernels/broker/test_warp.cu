/**
 * Warp-level Broker Queue test
 * Tests work-group cooperative operations
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test 1: Single warp enqueue/dequeue with work-group API
__global__ void test_warp_enqueue_dequeue(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
        printf("GPU: Queue initialized (head=%llu, tail=%llu, count=%llu)\n",
               queue.head, queue.tail, queue.count);
    }
    __syncthreads();

    const int WARP_SIZE = 32;
    const int NUM_ITEMS = 16; // Start small

    // Prepare items (each thread contributes)
    int items[NUM_ITEMS];
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        // Leader prepares all items
        for (int i = 0; i < NUM_ITEMS; i++) {
            items[i] = 100 + i;
        }

        printf("GPU: Enqueuing %d items\n", NUM_ITEMS);

        // Enqueue using work-group API
        bool success = queue.enqueue_wg(items, NUM_ITEMS, lane_id, WARP_SIZE);

        if (!success) {
            printf("GPU: Enqueue FAILED\n");
            *result = -1;
            return;
        }

        printf("GPU: Enqueue SUCCESS (head=%llu, tail=%llu, count=%llu)\n",
               queue.head, queue.tail, queue.count);

        // Dequeue using work-group API
        int retrieved[NUM_ITEMS];
        printf("GPU: Dequeuing %d items\n", NUM_ITEMS);

        success = queue.dequeue_wg(retrieved, NUM_ITEMS, lane_id, WARP_SIZE);

        if (!success) {
            printf("GPU: Dequeue FAILED\n");
            *result = -2;
            return;
        }

        printf("GPU: Dequeue SUCCESS (head=%llu, tail=%llu, count=%llu)\n",
               queue.head, queue.tail, queue.count);

        // Verify FIFO order
        bool all_correct = true;
        for (int i = 0; i < NUM_ITEMS; i++) {
            if (retrieved[i] != items[i]) {
                printf("GPU: Mismatch at index %d: expected %d, got %d\n",
                       i, items[i], retrieved[i]);
                all_correct = false;
            }
        }

        if (all_correct) {
            printf("GPU: All items matched - TEST PASSED ✓\n");
            *result = 0;
        } else {
            printf("GPU: Items mismatch - TEST FAILED ✗\n");
            *result = -3;
        }
    }
}

// Test 2: Full warp cooperation (all 32 threads participate)
__global__ void test_full_warp_cooperation(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    const int WARP_SIZE = 32;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Each thread in the warp has one item
    int my_item = threadIdx.x + 200;
    int items[32];

    // Gather items from all threads
    items[lane_id] = my_item;
    __syncwarp();

    // Enqueue cooperatively
    if (lane_id == 0) {
        bool success = queue.enqueue_wg(items, WARP_SIZE, lane_id, WARP_SIZE);
        if (!success) {
            *result = -1;
            return;
        }

        // Dequeue cooperatively
        int retrieved[32];
        success = queue.dequeue_wg(retrieved, WARP_SIZE, lane_id, WARP_SIZE);
        if (!success) {
            *result = -2;
            return;
        }

        // Verify
        for (int i = 0; i < WARP_SIZE; i++) {
            if (retrieved[i] != items[i]) {
                *result = -3;
                return;
            }
        }

        *result = 0;
    }
}

// Test 3: Multiple sequential enqueue/dequeue
__global__ void test_multiple_operations(int* result) {
    __shared__ BrokerQueue<int, 1024> queue;

    if (threadIdx.x == 0) {
        queue.init();
    }
    __syncthreads();

    const int NUM_BATCHES = 5;
    const int BATCH_SIZE = 8;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        for (int batch = 0; batch < NUM_BATCHES; batch++) {
            int items[BATCH_SIZE];

            // Prepare batch
            for (int i = 0; i < BATCH_SIZE; i++) {
                items[i] = batch * 100 + i;
            }

            // Enqueue
            bool success = queue.enqueue_wg(items, BATCH_SIZE, lane_id, 32);
            if (!success) {
                printf("GPU: Batch %d enqueue failed\n", batch);
                *result = -1;
                return;
            }
        }

        printf("GPU: All %d batches enqueued\n", NUM_BATCHES);

        // Dequeue all
        for (int batch = 0; batch < NUM_BATCHES; batch++) {
            int retrieved[BATCH_SIZE];

            bool success = queue.dequeue_wg(retrieved, BATCH_SIZE, lane_id, 32);
            if (!success) {
                printf("GPU: Batch %d dequeue failed\n", batch);
                *result = -2;
                return;
            }

            // Verify
            for (int i = 0; i < BATCH_SIZE; i++) {
                int expected = batch * 100 + i;
                if (retrieved[i] != expected) {
                    printf("GPU: Batch %d, item %d: expected %d, got %d\n",
                           batch, i, expected, retrieved[i]);
                    *result = -3;
                    return;
                }
            }
        }

        printf("GPU: All batches verified - TEST PASSED ✓\n");
        *result = 0;
    }
}

int main() {
    printf("=======================================================\n");
    printf("Broker Queue Warp-Level Tests\n");
    printf("=======================================================\n\n");

    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    // Test 1
    printf("[TEST 1] Single warp enqueue/dequeue... ");
    test_warp_enqueue_dequeue<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_result == 0) {
        printf("PASS ✓\n\n");
    } else {
        printf("FAIL ✗ (code=%d)\n\n", h_result);
        CUDA_CHECK(cudaFree(d_result));
        return 1;
    }

    // Test 2
    printf("[TEST 2] Full warp cooperation... ");
    test_full_warp_cooperation<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_result == 0) {
        printf("PASS ✓\n\n");
    } else {
        printf("FAIL ✗ (code=%d)\n\n", h_result);
        CUDA_CHECK(cudaFree(d_result));
        return 1;
    }

    // Test 3
    printf("[TEST 3] Multiple sequential operations... ");
    test_multiple_operations<<<1, 32>>>(d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_result == 0) {
        printf("PASS ✓\n\n");
    } else {
        printf("FAIL ✗ (code=%d)\n\n", h_result);
        CUDA_CHECK(cudaFree(d_result));
        return 1;
    }

    CUDA_CHECK(cudaFree(d_result));

    printf("=======================================================\n");
    printf("All warp-level tests PASSED ✓\n");
    printf("=======================================================\n");

    return 0;
}
