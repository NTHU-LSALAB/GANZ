/*
 * GPU Packet Processing - Improved Inference Ring Buffer Header
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Split-plane ring buffer for CPU-GPU inference coordination.
 *
 * TWO PLANES, ONE SLOT INDEX
 * ==========================
 *
 *   Control plane (struct inference_ring_slot)
 *       Host-mapped pinned memory (cudaHostAllocMapped). Holds ONLY metadata:
 *       ownership state, record directory (offsets/lengths), TCP routing
 *       context and profiling timestamps. The host reads and writes this
 *       plane because it must know which slots to launch work for.
 *
 *   Data plane (struct inference_payload_slot)
 *       Device-only memory (cudaMalloc). Holds request bytes parsed out of
 *       the NIC packet and the response bytes assembled for transmission.
 *       ONLY GPU kernels ever dereference this plane.
 *
 * INVARIANT: no request or result payload byte is ever read, written, copied,
 * tokenized or formatted by the CPU. There is deliberately no host-side
 * accessor for the data plane — the absence of one is what enforces this.
 * Payload lives in device memory from NIC ingress through HTTP parsing,
 * tokenization, TensorRT inference, result formatting and response assembly.
 *
 * PUBLICATION ORDERING (system scope)
 * ===================================
 * Because payload and control now live in different memories, every handoff
 * publishes the data plane BEFORE the control plane, and every consumer
 * acquires the control plane before touching the data plane:
 *
 *   GPU RX  : store request bytes (device) -> release-store ready=PARAM_READY
 *             (thread_scope_system) -> release-push request_queue
 *   Host    : acquire-CAS ready PARAM_READY->PROCESSING. The acquire
 *             synchronises-with the GPU release, so the payload writes
 *             happen-before the host's subsequent kernel launches, and
 *             therefore happen-before those kernels' reads.
 *   GPU     : tokenize -> enqueueV3 -> format, all ordered on one CUDA stream.
 *   Host    : cudaStreamSynchronize (the format kernel's device writes are
 *             now visible device-wide) -> release-store ready=RESULT_READY
 *             -> release-push response_queue. The sync MUST precede the
 *             release-store, otherwise TX could read a half-written response.
 *   GPU TX  : acquire-pop response_queue -> acquire-CAS ready -> read response
 *             bytes (device) -> assemble packet.
 */

#ifndef IMPROVED_INFERENCE_UVM_H
#define IMPROVED_INFERENCE_UVM_H

#include <stdint.h>

#if defined(__cplusplus)
#define GAZSI_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define GAZSI_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

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
 * Record Directory entry — describes one request within a packed slot.
 *
 * offset/length address the request bytes inside the slot's device-resident
 * request[] buffer. resp_length is the formatted response length written by
 * the GPU format kernel; the response offset is not stored because it is
 * always rec * GAZSI_RESPONSE_STRIDE — a fixed stride removes the
 * "where does record N's response start" special case entirely, and stops
 * record N's response from ever overwriting record N+1's request text.
 */
#define MAX_RECORDS_PER_SLOT 4

struct record_entry {
    uint32_t offset;
    uint32_t length;
    uint32_t resp_length;
    uint32_t tcp_sent_seq;
    uint32_t tcp_recv_ack;
    uint32_t _pad;
};

GAZSI_STATIC_ASSERT(sizeof(struct record_entry) == 24, "record_entry must stay 24B");

/*
 * Data plane geometry.
 *
 * GAZSI_REQUEST_BYTES  : request text arena per slot (all packed records)
 * GAZSI_RESPONSE_STRIDE: fixed per-record response arena
 * GAZSI_RESPONSE_MAX   : longest response the GPU formatter may emit. Must
 *                        stay <= HTTP_BODY_MAX_LEN in the TX kernel, else TX
 *                        would replace the body with its error JSON.
 * GAZSI_TOKENIZE_MAX   : request bytes fed to the tokenizer. Matches the
 *                        512-byte clamp the CPU path used, so token streams
 *                        (and therefore model outputs) are unchanged.
 */
#define GAZSI_REQUEST_BYTES    896
#define GAZSI_RESPONSE_STRIDE  1024
#define GAZSI_RESPONSE_MAX     900
#define GAZSI_TOKENIZE_MAX     512

/* Device-only payload slot. Never dereferenced by host code. */
struct inference_payload_slot {
    char request[GAZSI_REQUEST_BYTES];
    char response[MAX_RECORDS_PER_SLOT * GAZSI_RESPONSE_STRIDE];
};

struct inference_payload_plane {
    struct inference_payload_slot slots[INFERENCE_RING_SIZE];
};

#define GPU_CONN_TABLE_SIZE 256

struct gpu_conn_state {
    volatile uint32_t active;
    volatile uint32_t conn_hash;
    volatile uint32_t sent_seq;
    volatile uint32_t recv_ack;
    volatile uint32_t cwnd;
    volatile uint32_t ssthresh;
    volatile uint32_t expected_seq;
    volatile uint32_t dup_ack_count;
    volatile uint32_t reorder_seq[4];
    volatile uint32_t reorder_len[4];
    volatile uint32_t reorder_count;
};

/*
 * Ring Buffer slot structure — CONTROL PLANE ONLY (host-mapped).
 *
 * Deliberately carries no payload bytes. Shrinking the slot from 1152B to
 * 256B also shrinks every host-mapped metadata access the GPU makes over
 * PCIe by the same factor.
 *
 * Cache-line aligned: Control Header + Record Directory in first cache line
 */
struct inference_ring_slot {
    /* Cache Line 1: Control Header + Record Directory (128 bytes) */
    volatile uint32_t ready;           /* 4B: Ownership state */
    volatile uint32_t record_count;    /* 4B: Number of records packed in slot */
    volatile uint64_t request_id;      /* 8B: Request ID (first record) */
    volatile uint32_t len;             /* 4B: Total request bytes across all records */
    uint32_t _pad_ctrl;                /* 4B: alignment */
    struct record_entry directory[MAX_RECORDS_PER_SLOT]; /* 96B: Record Directory */
    char _pad_dir[8];                  /* 8B: pad to 128 bytes */

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

    /*
     * No payload buffer here by design. Request and response bytes live in
     * the device-resident inference_payload_plane at the same slot index.
     */
    char _pad_tail[8];                 /* Pad to 256 bytes (2 cache lines) */
} __attribute__((aligned(128)));

GAZSI_STATIC_ASSERT(sizeof(struct inference_ring_slot) == 256,
                    "control slot must stay 256B (2 cache lines)");

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

    struct gpu_conn_state conn_table[GPU_CONN_TABLE_SIZE] __attribute__((aligned(128)));

    struct clock_sync_info clock_sync;
};

/*
 * Host-side queue primitives. Header-only so that the slot ownership rules in
 * slot_claim.h can be built without CUDA or DOCA.
 */
#include "index_queue.h"

/* Forward declaration for DOCA types */
struct doca_gpu_semaphore;
struct doca_gpu_semaphore_gpu;

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Ring Buffer. Returns pointer, NULL on failure */
struct inference_ring_buffer* init_inference_ring_buffer(int doca_gpu_id);

void gpu_conn_table_init_entry(struct inference_ring_buffer *ring,
                               uint32_t conn_hash, uint32_t sent_seq, uint32_t recv_ack);
void gpu_conn_table_clear_entry(struct inference_ring_buffer *ring, uint32_t conn_hash);

/* Set inference result notification semaphore (CPU-side, for notifying GPU after writing result) */
void set_inference_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Set request notification semaphore (GPU -> CPU notification) */
void set_request_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Get request notification semaphore CPU handle (for semaphore-guided polling) */
struct doca_gpu_semaphore* get_request_semaphore_cpu(void);

/* Free Ring Buffer */
void free_inference_ring_buffer(struct inference_ring_buffer *ring);

/*
 * Device pointer to the payload data plane, for passing to CUDA kernels.
 *
 * There is intentionally NO host-mapped counterpart: this pointer is only
 * ever valid inside device code. Dereferencing it on the host is a
 * segmentation fault, which is exactly the guard rail we want.
 */
struct inference_payload_plane *get_payload_plane_device(void);

/*
 * Publish a device-formatted result for a slot: mark RESULT_READY and hand
 * the slot to the GPU TX kernel.
 *
 * Caller MUST have synchronised the stream that ran the format kernel first,
 * so the response bytes are visible device-wide before TX can observe the
 * state change. Touches control metadata only.
 *
 * Returns 1 on success, 0 on failure.
 */
int publish_inference_result(struct inference_ring_buffer *ring_gpu, uint32_t slot_index);

/* GPU-side slot allocation. Returns slot index, -1 if ring buffer is full */
#ifdef __CUDACC__
__device__ int gpu_alloc_ring_slot(struct inference_ring_buffer *ring, uint64_t *request_id);

/* GPU-side FIFO queue operations (thread_scope_system atomics) */
__device__ int  gpu_iq_pop(struct index_queue *q);
__device__ void gpu_iq_push(struct index_queue *q, int value);

#endif

/* CPU-side FIFO queue operations live in index_queue.h (static inline). */

#ifdef __cplusplus
}
#endif

#endif /* IMPROVED_INFERENCE_UVM_H */
