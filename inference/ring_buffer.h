/*
 * GPU Packet Processing - Improved Inference Ring Buffer Header
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Lock-free ring buffer for CPU-GPU inference data exchange
 */

#ifndef IMPROVED_INFERENCE_UVM_H
#define IMPROVED_INFERENCE_UVM_H

#include <stdint.h>

/* Ring Buffer size - 128 performs better than 256 in testing */
#define INFERENCE_RING_SIZE 256

#define IQ_CAPACITY 512
#define IQ_MASK     (IQ_CAPACITY - 1)
#define IQ_EMPTY    (-1)   /* Sentinel: slot entry not yet written */

/* UVM buffer states - Producer-Consumer state machine */
#define UVM_STATUS_FREE         0  /* Buffer is idle */
#define UVM_STATUS_PARAM_READY  1  /* GPU has written parameters */
#define UVM_STATUS_PROCESSING   2  /* CPU is processing (atomic claim) */
#define UVM_STATUS_RESULT_READY 3  /* CPU has written inference result */
#define UVM_STATUS_CONSUMED     4  /* GPU has read result */

/*
 * Record Directory entry — describes one request within a packed slot
 */
#define MAX_RECORDS_PER_SLOT 8

struct record_entry {
    uint32_t offset;    /* byte offset within data[] buffer */
    uint32_t length;    /* byte length of this record's payload */
};

/*
 * Ring Buffer slot structure
 * Cache-line aligned: Control Header + Record Directory in first cache line
 */
struct inference_ring_slot {
    /* Cache Line 1: Control Header + Record Directory (128 bytes) */
    volatile uint32_t ready;           /* 4B: Ownership state */
    volatile uint32_t record_count;    /* 4B: Number of records packed in slot */
    volatile uint64_t request_id;      /* 8B: Request ID (first record) */
    volatile uint32_t len;             /* 4B: Total payload bytes across all records */
    uint32_t _pad_ctrl;                /* 4B: alignment */
    struct record_entry directory[MAX_RECORDS_PER_SLOT]; /* 64B: Record Directory */
    char _pad_dir[40];                 /* 40B: pad to 128 bytes */

    /* TCP Routing Context (40 bytes) — shared by all records (same connection) */
    uint8_t eth_src_addr_bytes[6];     /* Ethernet source address */
    uint8_t eth_dst_addr_bytes[6];     /* Ethernet destination address */
    uint32_t ip_src_addr;              /* IP source address */
    uint32_t ip_dst_addr;              /* IP destination address */
    uint16_t ip_total_length;          /* IP total length */
    uint16_t tcp_src_port;             /* TCP source port */
    uint16_t tcp_dst_port;             /* TCP destination port */
    uint8_t tcp_dt_off;                /* TCP data offset */
    uint8_t http_page_type;            /* HTTP page type (enum http_page_get) */
    uint32_t tcp_sent_seq;             /* TCP sent sequence number */
    uint32_t tcp_recv_ack;             /* TCP receive acknowledgment */
    char padding_tcp[2];               /* Alignment padding */

    /* Profiling Timestamps (80 bytes) — moved out of hot cache line */
    volatile uint64_t start_timestamp; /* GPU processing start timestamp */
    volatile uint64_t t0_gpu_received;   /* T0: HTTP request arrived at GPU */
    volatile uint64_t t1_slot_allocated; /* T1: GPU allocated ring slot */
    volatile uint64_t t2_gpu_wrote_uvm;  /* T2: GPU finished writing to UVM */
    volatile uint64_t t3_cpu_read;       /* T3: CPU detected data */
    volatile uint64_t t4_tensorrt_start; /* T4: CPU started TensorRT inference */
    volatile uint64_t t5_tensorrt_end;   /* T5: TensorRT inference complete */
    volatile uint64_t t6_cpu_wrote_uvm;  /* T6: CPU wrote result to UVM */
    volatile uint64_t t7_gpu_read;       /* T7: GPU detected result */
    volatile uint64_t t8_gpu_sent;       /* T8: GPU sent HTTP response */

    /* Data Buffer — holds packed payloads for all records */
    char data[856];                    /* Payload buffer */
    char padding2[48];                 /* Pad to 1152 bytes (9 cache lines) */
} __attribute__((aligned(128)));

/*
 * Lock-free SPSC/MPSC index queue (Q2 optimization)
 *
 * Replaces O(N) slot scanning with O(1) FIFO notification.
 * Three actors push/pop slot indices through dedicated queues
 * instead of scanning all 128 slots to find target state.
 *
 * Memory ordering:
 *   GPU side: cuda::atomic_ref<T, cuda::thread_scope_system>
 *   CPU side: __atomic_* builtins with __ATOMIC_ACQUIRE / __ATOMIC_RELEASE
 *
 * Push: FAA(tail) → get position → CAS(entry, IQ_EMPTY, value)
 * Pop:  FAA(head) → get position → spin until entry != IQ_EMPTY → exchange to IQ_EMPTY
 */
struct index_queue {
    volatile uint32_t head __attribute__((aligned(128)));
    char pad_h[124];
    volatile uint32_t tail __attribute__((aligned(128)));
    char pad_t[124];
    volatile int32_t entries[IQ_CAPACITY] __attribute__((aligned(128)));
};

/*
 * Clock sync info for GPU/CPU clock conversion
 */
struct clock_sync_info {
    uint64_t gpu_clock_base;      /* GPU clock64() baseline */
    uint64_t cpu_time_ns_base;    /* CPU clock_gettime() baseline (ns) */
    double   gpu_clock_freq_ghz;  /* GPU clock frequency (GHz) */
};

/*
 * Ring Buffer structure
 * Lock-free producer-consumer ring buffer
 */
struct inference_ring_buffer {
    struct inference_ring_slot slots[INFERENCE_RING_SIZE];
    volatile uint32_t head;    /* Producer (GPU) write position */
    volatile uint64_t next_request_id;

    volatile uint32_t batch_epoch;
    char batch_padding[108];              /* Pad head group to full 128-byte cache line */

    /* Isolated cache line: GPU atomicAdd, CPU atomic_load/fetch_sub */
    volatile uint32_t pending_count __attribute__((aligned(128)));
    char pending_padding[124];

    /*
     * Q2: O(1) FIFO notification queues (replace O(N) slot scanning)
     *
     * free_pool:      GPU TX pushes after send → GPU RX pops for allocation
     * request_queue:  GPU RX pushes after data write → CPU pops for inference
     * response_queue: CPU pushes after result write → GPU TX pops for response
     */
    struct index_queue free_pool      __attribute__((aligned(128)));
    struct index_queue request_queue   __attribute__((aligned(128)));
    struct index_queue response_queue  __attribute__((aligned(128)));

    /* Clock sync - set once at init, read-only thereafter */
    struct clock_sync_info clock_sync;
};

/* Forward declaration for DOCA types */
struct doca_gpu_semaphore;
struct doca_gpu_semaphore_gpu;

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Ring Buffer. Returns pointer, NULL on failure */
struct inference_ring_buffer* init_inference_ring_buffer(int doca_gpu_id);

/* Set inference result notification semaphore (CPU-side, for notifying GPU after writing result) */
void set_inference_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Set request notification semaphore (GPU -> CPU notification) */
void set_request_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Get request notification semaphore CPU handle (for semaphore-guided polling) */
struct doca_gpu_semaphore* get_request_semaphore_cpu(void);

/* Free Ring Buffer */
void free_inference_ring_buffer(struct inference_ring_buffer *ring);

/* CPU-side write processing result to GPU ring. Returns 1 on success, 0 on failure */
int cpu_write_inference_result_to_gpu_ring(struct inference_ring_buffer *ring_gpu, uint32_t slot_index, const char *result);

/* Host-side slot data accessors (for cudaMemcpyAsync source) */
const char *get_slot_data_host(uint32_t slot_index);
uint32_t get_slot_len_host(uint32_t slot_index);

/* GPU-side slot allocation. Returns slot index, -1 if ring buffer is full */
#ifdef __CUDACC__
__device__ int gpu_alloc_ring_slot(struct inference_ring_buffer *ring, uint64_t *request_id);

/* GPU-side store inference data to slot */
__device__ void gpu_store_inference_data_to_slot(struct inference_ring_buffer *ring, int slot_index, const char *data);

/* GPU-side FIFO queue operations (thread_scope_system atomics) */
__device__ int  gpu_iq_pop(struct index_queue *q);
__device__ void gpu_iq_push(struct index_queue *q, int value);

#endif

/* CPU-side FIFO queue operations (__atomic_* builtins) */
int  cpu_iq_pop(struct index_queue *q);
void cpu_iq_push(struct index_queue *q, int value);

#ifdef __cplusplus
}
#endif

#endif /* IMPROVED_INFERENCE_UVM_H */
