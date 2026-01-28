/**
 * Test to understand count advancement logic
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void test_count_logic() {
    __shared__ uint64_t head, tail, count;

    if (threadIdx.x == 0) {
        head = 0;   // Even: 0, 2, 4, 6, ...
        tail = 1;   // Odd: 1, 3, 5, 7, ...
        count = 0;

        printf("=== Single Item Test ===\n");

        // Enqueue 1 item
        uint64_t enq_ticket = atomicAdd((unsigned long long*)&head, 2ULL);
        printf("Enqueue: ticket=%llu, count=%llu\n", enq_ticket, count);
        while (count != enq_ticket) {}  // Wait
        printf("Enqueue: got turn\n");

        // Advance count - for single item, we advance by 1
        atomicAdd((unsigned long long*)&count, 1ULL);
        printf("Enqueue done: count=%llu\n", count);

        // Dequeue 1 item
        uint64_t deq_ticket = atomicAdd((unsigned long long*)&tail, 2ULL);
        printf("Dequeue: ticket=%llu, count=%llu\n", deq_ticket, count);
        while (count != deq_ticket) {}  // Wait
        printf("Dequeue: got turn\n");

        atomicAdd((unsigned long long*)&count, 1ULL);
        printf("Dequeue done: count=%llu\n", count);

        printf("\n=== Batch Test (8 items) ===\n");

        // Enqueue 8 items
        uint64_t enq_ticket2 = atomicAdd((unsigned long long*)&head, 16ULL);  // 8 * 2
        printf("Enqueue batch: ticket=%llu, count=%llu\n", enq_ticket2, count);
        while (count != enq_ticket2) {}
        printf("Enqueue batch: got turn\n");

        // For 8 items advancing ticket by 16, count should advance by...?
        // Option 1: Advance by 1 (same as single item)
        // Option 2: Advance by 8 (num_items)
        // Option 3: Advance by 16 (ticket increment)

        // Let's try option 2: advance by num_items
        atomicAdd((unsigned long long*)&count, 8ULL);
        printf("Enqueue batch done: count=%llu (advanced by 8)\n", count);

        // Dequeue 8 items
        uint64_t deq_ticket2 = atomicAdd((unsigned long long*)&tail, 16ULL);  // 8 * 2
        printf("Dequeue batch: ticket=%llu, count=%llu\n", deq_ticket2, count);

        if (count == deq_ticket2) {
            printf("Dequeue batch: READY (count matches ticket)\n");
        } else {
            printf("Dequeue batch: WAITING (count=%llu != ticket=%llu)\n", count, deq_ticket2);
        }
    }
}

int main() {
    test_count_logic<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
