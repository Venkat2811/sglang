#ifndef BROKER_QUEUE_CUH
#define BROKER_QUEUE_CUH

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Broker Queue: Lock-free concurrent queue for GPU
 *
 * Reference: "Broker Queue" (TU Graz 2019)
 * https://arbook.icg.tugraz.at/schmalstieg/Schmalstieg_354.pdf
 *
 * Key concepts:
 * - Ticketing system: even tickets = enqueue, odd tickets = dequeue
 * - Count variable: the "broker" ensures FIFO ordering
 * - Work-group level operations: amortize atomic costs across 32 threads
 *
 * Algorithm (from paper Section 3.2):
 *
 * Enqueue:
 *   1. ticket = atomicAdd(&head, num_items)
 *   2. wait until (count == ticket)
 *   3. memcpy items to data[ticket % CAPACITY]
 *   4. atomicAdd(&count, num_items)
 *
 * Dequeue:
 *   1. ticket = atomicAdd(&tail, num_items)
 *   2. wait until (count == ticket)
 *   3. memcpy data[ticket % CAPACITY] to items
 *   4. atomicAdd(&count, num_items)
 */

template<typename T, uint32_t CAPACITY>
struct BrokerQueue {
    // Ticketing system (64-bit to avoid overflow)
    uint64_t head;  // Even tickets for enqueue
    uint64_t tail;  // Odd tickets for dequeue
    uint64_t count; // The "broker" - critical for correctness

    // Ring buffer
    T data[CAPACITY];

    // Initialize queue
    __device__ __host__ void init() {
        head = 0;  // Even tickets for enqueue: 0, 2, 4, 6, ...
        tail = 1;  // Odd tickets for dequeue: 1, 3, 5, 7, ...
        count = 0; // Count alternates: 0(enq), 1(deq), 2(enq), 3(deq), ...
    }

    // =================================================================
    // Core atomic helpers with memory ordering
    // =================================================================

    __device__ __forceinline__ uint64_t claim_ticket(uint64_t* ticket_ptr, uint32_t num_items) {
        // Claim num_items slots in the queue
        // Returns the starting ticket number
        // CRITICAL: Tickets advance by 2 because:
        // - Even tickets (0,2,4,6,...) are for enqueue
        // - Odd tickets (1,3,5,7,...) are for dequeue
        // For batch operations, we claim 2*num_items to skip over the other type's tickets
        return atomicAdd((unsigned long long*)ticket_ptr, (unsigned long long)(num_items * 2));
    }

    __device__ __forceinline__ void wait_for_turn(uint64_t ticket, volatile uint64_t* count_ptr) {
        // Busy-wait until it's our turn (count reaches our ticket)
        // This ensures FIFO ordering and linearizability
        while (*count_ptr != ticket) {
            __threadfence_system(); // Ensure visibility across SMs
        }
    }

    __device__ __forceinline__ void advance_count(uint64_t* count_ptr, uint32_t num_items) {
        // Signal completion by advancing count
        // Count advances by num_items, allowing next operation to proceed
        atomicAdd((unsigned long long*)count_ptr, (unsigned long long)num_items);
        __threadfence_system(); // Ensure visibility
    }

    // =================================================================
    // Work-group level enqueue (Algorithm 1 from paper)
    // =================================================================

    __device__ bool enqueue_wg(const T* items, uint32_t num_items,
                                uint32_t lane_id, uint32_t warp_size) {
        // Check capacity
        if (num_items > CAPACITY) return false;

        uint64_t ticket;

        // Leader thread claims tickets for the work-group
        if (lane_id == 0) {
            ticket = claim_ticket(&head, num_items);
        }

        // Broadcast ticket to all threads in work-group
        ticket = __shfl_sync(0xffffffff, ticket, 0);

        // Leader waits for turn
        if (lane_id == 0) {
            wait_for_turn(ticket, &count);
        }
        __syncwarp();

        // All threads cooperate to copy data
        // Each thread copies a subset of items
        // IMPORTANT: Divide ticket by 2 because tickets are even (0,2,4...)
        // but array indices are sequential (0,1,2...)
        for (uint32_t i = lane_id; i < num_items; i += warp_size) {
            uint32_t slot = ((ticket / 2) + i) % CAPACITY;
            data[slot] = items[i];
        }
        __syncwarp();

        // Leader signals completion
        // CRITICAL: Advance count by ticket increment (2*num_items)
        // to allow the next ticket to proceed
        // Each operation gets a unique ticket range, count tracks which ticket can proceed
        if (lane_id == 0) {
            advance_count(&count, 2 * num_items);
        }

        return true;
    }

    // =================================================================
    // Work-group level dequeue (Algorithm 2 from paper)
    // =================================================================

    __device__ bool dequeue_wg(T* items, uint32_t num_items,
                                uint32_t lane_id, uint32_t warp_size) {
        // Check capacity
        if (num_items > CAPACITY) return false;

        uint64_t ticket;

        // Leader thread claims tickets
        if (lane_id == 0) {
            ticket = claim_ticket(&tail, num_items);
        }

        // Broadcast ticket
        ticket = __shfl_sync(0xffffffff, ticket, 0);

        // Leader waits for turn
        if (lane_id == 0) {
            wait_for_turn(ticket, &count);
        }
        __syncwarp();

        // All threads cooperate to copy data out
        // IMPORTANT: Divide ticket by 2 because tickets are odd (1,3,5...)
        // but array indices are sequential (0,1,2...)
        for (uint32_t i = lane_id; i < num_items; i += warp_size) {
            uint32_t slot = ((ticket / 2) + i) % CAPACITY;
            items[i] = data[slot];
        }
        __syncwarp();

        // Leader signals completion
        // CRITICAL: Advance count by ticket increment (2*num_items)
        // to allow the next ticket to proceed
        // Each operation gets a unique ticket range, count tracks which ticket can proceed
        if (lane_id == 0) {
            advance_count(&count, 2 * num_items);
        }

        return true;
    }

    // =================================================================
    // Single-thread convenience wrappers
    // =================================================================

    __device__ bool enqueue(const T& item) {
        return enqueue_wg(&item, 1, 0, 1);
    }

    __device__ bool dequeue(T& item) {
        return dequeue_wg(&item, 1, 0, 1);
    }

    // =================================================================
    // Non-blocking try operations (for polling)
    // =================================================================

    __device__ bool try_dequeue_wg(T* items, uint32_t num_items,
                                     uint32_t lane_id, uint32_t warp_size) {
        // Non-blocking variant: check if data is available
        uint64_t current_tail = tail;
        uint64_t current_count = count;

        // Check if enough items are available
        if (current_count < current_tail + num_items) {
            return false; // Queue empty or not enough items
        }

        // Proceed with normal dequeue
        return dequeue_wg(items, num_items, lane_id, warp_size);
    }

    // =================================================================
    // Queue status queries
    // =================================================================

    __device__ uint64_t size() const {
        // Approximate size (may be stale due to concurrent ops)
        uint64_t h = head;
        uint64_t t = tail;
        return (h > t) ? (h - t) : 0;
    }

    __device__ bool is_empty() const {
        return head == tail;
    }

    __device__ bool is_full() const {
        return size() >= CAPACITY;
    }
};

// =================================================================
// Token structure for MoE routing
// =================================================================

struct MoEToken {
    // Pointer to hidden state in global memory
    float* hidden_state_ptr;

    // Token metadata
    int32_t token_id;
    int32_t expert_id;
    float weight;

    // Padding for alignment
    int32_t _padding;
};

// Specialization for MoE token routing
using MoETokenQueue = BrokerQueue<MoEToken, 256>;

#endif // BROKER_QUEUE_CUH
