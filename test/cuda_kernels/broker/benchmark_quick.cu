/**
 * Quick throughput sanity check: Broker Queue vs atomicCAS baseline
 * Goal: Verify we're >2x faster before SGLang integration
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

#include "../../../sglang/srt/cuda_kernels/broker/broker_queue_v2.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Baseline: Simple atomicCAS-based queue
// ============================================================================

template<typename T, uint32_t CAPACITY>
struct AtomicQueue {
    uint64_t head;
    uint64_t tail;
    T data[CAPACITY];

    __device__ void init() {
        head = 0;
        tail = 0;
    }

    __device__ bool enqueue(const T& item) {
        uint64_t current_head = head;
        uint64_t current_tail = tail;

        // Check if full
        if (current_head - current_tail >= CAPACITY) {
            return false;
        }

        // Atomic increment head
        uint64_t pos = atomicAdd((unsigned long long*)&head, 1ULL);
        data[pos % CAPACITY] = item;
        return true;
    }

    __device__ bool dequeue(T& item) {
        uint64_t current_head = head;
        uint64_t current_tail = tail;

        // Check if empty
        if (current_tail >= current_head) {
            return false;
        }

        // Atomic increment tail
        uint64_t pos = atomicAdd((unsigned long long*)&tail, 1ULL);
        item = data[pos % CAPACITY];
        return true;
    }
};

// ============================================================================
// Benchmark kernels
// ============================================================================

// Test 1: Single block, measure enqueue/dequeue throughput
// Interleave enqueue/dequeue to avoid filling queue
template<typename QueueType>
__global__ void benchmark_single_block(QueueType* queue, int* items, int num_ops, uint64_t* elapsed_ns) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->init();

        // Start timer
        clock_t start = clock64();

        // Interleave: enqueue batch, dequeue batch to stay within capacity
        const int BATCH_SIZE = 512;  // Half of capacity
        for (int batch = 0; batch < num_ops / BATCH_SIZE; batch++) {
            // Enqueue batch
            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = batch * BATCH_SIZE + i;
                queue->enqueue(items[idx]);
            }

            // Dequeue batch
            int retrieved;
            for (int i = 0; i < BATCH_SIZE; i++) {
                queue->dequeue(retrieved);
            }
        }

        // End timer
        clock_t end = clock64();
        *elapsed_ns = end - start;
    }
}

// Test 2: Multiple producers, single consumer
template<typename QueueType>
__global__ void benchmark_multi_producer(QueueType* queue, int* items, int num_ops_per_block,
                                          int* success_count) {
    __shared__ int local_success;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->init();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        local_success = 0;
    }
    __syncthreads();

    // Each block produces items
    if (threadIdx.x == 0) {
        int base = blockIdx.x * num_ops_per_block;
        for (int i = 0; i < num_ops_per_block; i++) {
            if (queue->enqueue(items[base + i])) {
                atomicAdd(&local_success, 1);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success_count = local_success;
    }
}

// ============================================================================
// Host benchmark runner
// ============================================================================

template<typename QueueType>
double run_single_block_benchmark(const char* name, int num_ops) {
    QueueType* d_queue;
    int* d_items;
    uint64_t* d_elapsed;

    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(QueueType)));
    CUDA_CHECK(cudaMalloc(&d_items, num_ops * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_elapsed, sizeof(uint64_t)));

    // Initialize items
    std::vector<int> h_items(num_ops);
    for (int i = 0; i < num_ops; i++) {
        h_items[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_items, h_items.data(), num_ops * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    benchmark_single_block<<<1, 32>>>(d_queue, d_items, num_ops / 10, d_elapsed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    benchmark_single_block<<<1, 32>>>(d_queue, d_items, num_ops, d_elapsed);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ops_per_sec = (num_ops * 2.0) / (elapsed_ms / 1000.0);  // *2 for enqueue+dequeue

    printf("%-20s: %d ops in %.2f ms = %.2f M ops/sec\n",
           name, num_ops * 2, elapsed_ms, ops_per_sec / 1e6);

    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_items));
    CUDA_CHECK(cudaFree(d_elapsed));

    return ops_per_sec;
}

int main() {
    printf("===========================================\n");
    printf("Broker Queue Quick Throughput Sanity Check\n");
    printf("===========================================\n\n");

    const int NUM_OPS = 10240;  // Must be multiple of BATCH_SIZE (512)

    printf("Test Configuration:\n");
    printf("  Operations: %d enqueue + %d dequeue = %d total\n", NUM_OPS, NUM_OPS, NUM_OPS * 2);
    printf("  Queue capacity: 1024\n");
    printf("  Element type: int (4 bytes)\n\n");

    // Test 1: AtomicCAS baseline
    printf("[1/2] AtomicCAS Baseline\n");
    double atomic_ops_sec = run_single_block_benchmark<AtomicQueue<int, 1024>>(
        "AtomicCAS Queue", NUM_OPS);

    printf("\n");

    // Test 2: Broker Queue V2
    printf("[2/2] Broker Queue V2\n");
    double broker_ops_sec = run_single_block_benchmark<BrokerQueueV2<int, 1024>>(
        "Broker Queue V2", NUM_OPS);

    printf("\n");
    printf("===========================================\n");
    printf("Results Summary\n");
    printf("===========================================\n");
    printf("AtomicCAS:    %.2f M ops/sec\n", atomic_ops_sec / 1e6);
    printf("Broker Queue: %.2f M ops/sec\n", broker_ops_sec / 1e6);
    printf("Speedup:      %.2fx\n", broker_ops_sec / atomic_ops_sec);
    printf("\n");

    if (broker_ops_sec > atomic_ops_sec * 2.0) {
        printf("✓ PASS: >2x faster than baseline\n");
        printf("✓ Ready for SGLang integration\n");
        return 0;
    } else if (broker_ops_sec > atomic_ops_sec * 1.2) {
        printf("⚠ WARNING: Only %.2fx faster (expected >2x)\n", broker_ops_sec / atomic_ops_sec);
        printf("⚠ Still proceed, but may need optimization\n");
        return 0;
    } else {
        printf("✗ FAIL: Not faster than baseline (%.2fx)\n", broker_ops_sec / atomic_ops_sec);
        printf("✗ Check implementation before SGLang integration\n");
        return 1;
    }
}
