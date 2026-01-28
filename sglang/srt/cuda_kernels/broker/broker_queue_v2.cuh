#ifndef BROKER_QUEUE_V2_CUH
#define BROKER_QUEUE_V2_CUH

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Broker Queue V2: Simplified sequential ticket approach
 *
 * All operations (enqueue and dequeue) use sequential tickets: 0, 1, 2, 3, ...
 * The count variable tracks which ticket is currently allowed to proceed
 */

template<typename T, uint32_t CAPACITY>
struct BrokerQueueV2 {
    uint64_t ticket;  // Next ticket number for any operation
    uint64_t count;   // Which ticket is currently allowed
    T data[CAPACITY];
    uint64_t write_pos;  // Write position in ring buffer
    uint64_t read_pos;   // Read position in ring buffer

    __device__ __host__ void init() {
        ticket = 0;
        count = 0;
        write_pos = 0;
        read_pos = 0;
    }

    // Claim a ticket for any operation
    __device__ __forceinline__ uint64_t claim_ticket() {
        return atomicAdd((unsigned long long*)&ticket, 1ULL);
    }

    // Wait for our turn
    __device__ __forceinline__ void wait_for_turn(uint64_t my_ticket) {
        while (count != my_ticket) {
            __threadfence_system();
        }
    }

    // Signal we're done
    __device__ __forceinline__ void release_ticket() {
        atomicAdd((unsigned long long*)&count, 1ULL);
        __threadfence_system();
    }

    // Single-thread enqueue
    __device__ bool enqueue(const T& item) {
        // Claim ticket
        uint64_t my_ticket = claim_ticket();

        // Wait for turn
        wait_for_turn(my_ticket);

        // Write data
        uint64_t pos = atomicAdd((unsigned long long*)&write_pos, 1ULL);
        data[pos % CAPACITY] = item;

        // Release
        release_ticket();
        return true;
    }

    // Single-thread dequeue
    __device__ bool dequeue(T& item) {
        // Claim ticket
        uint64_t my_ticket = claim_ticket();

        // Wait for turn
        wait_for_turn(my_ticket);

        // Check if queue has data
        if (read_pos >= write_pos) {
            // Queue empty - still release ticket
            release_ticket();
            return false;
        }

        // Read data
        uint64_t pos = atomicAdd((unsigned long long*)&read_pos, 1ULL);
        item = data[pos % CAPACITY];

        // Release
        release_ticket();
        return true;
    }

    // Work-group enqueue
    __device__ bool enqueue_wg(const T* items, uint32_t num_items,
                                uint32_t lane_id, uint32_t warp_size) {
        uint64_t my_ticket;

        // Leader claims ticket
        if (lane_id == 0) {
            my_ticket = claim_ticket();
        }
        my_ticket = __shfl_sync(0xffffffff, my_ticket, 0);

        // Leader waits
        if (lane_id == 0) {
            wait_for_turn(my_ticket);
        }
        __syncwarp();

        // All threads cooperate to write
        uint64_t base_pos;
        if (lane_id == 0) {
            base_pos = atomicAdd((unsigned long long*)&write_pos, (unsigned long long)num_items);
        }
        base_pos = __shfl_sync(0xffffffff, base_pos, 0);

        for (uint32_t i = lane_id; i < num_items; i += warp_size) {
            data[(base_pos + i) % CAPACITY] = items[i];
        }
        __syncwarp();

        // Leader releases
        if (lane_id == 0) {
            release_ticket();
        }

        return true;
    }

    // Work-group dequeue
    __device__ bool dequeue_wg(T* items, uint32_t num_items,
                                uint32_t lane_id, uint32_t warp_size) {
        uint64_t my_ticket;

        // Leader claims ticket
        if (lane_id == 0) {
            my_ticket = claim_ticket();
        }
        my_ticket = __shfl_sync(0xffffffff, my_ticket, 0);

        // Leader waits and checks
        if (lane_id == 0) {
            wait_for_turn(my_ticket);

            // Check if enough data
            if (read_pos + num_items > write_pos) {
                release_ticket();
                return false;
            }
        }
        __syncwarp();

        // All threads cooperate to read
        uint64_t base_pos;
        if (lane_id == 0) {
            base_pos = atomicAdd((unsigned long long*)&read_pos, (unsigned long long)num_items);
        }
        base_pos = __shfl_sync(0xffffffff, base_pos, 0);

        for (uint32_t i = lane_id; i < num_items; i += warp_size) {
            items[i] = data[(base_pos + i) % CAPACITY];
        }
        __syncwarp();

        // Leader releases
        if (lane_id == 0) {
            release_ticket();
        }

        return true;
    }

    __device__ bool is_empty() const {
        return read_pos >= write_pos;
    }

    __device__ uint64_t size() const {
        return (write_pos > read_pos) ? (write_pos - read_pos) : 0;
    }
};

// MoE Token type
struct MoEToken {
    float* hidden_state_ptr;
    int32_t token_id;
    int32_t expert_id;
    float weight;
    int32_t _padding;
};

using MoETokenQueueV2 = BrokerQueueV2<MoEToken, 256>;

#endif // BROKER_QUEUE_V2_CUH
