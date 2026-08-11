/*
 * GPU Packet Processing - Improved Inference Ring Buffer
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Zero-Copy Ring Buffer Implementation
 */

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ring_buffer.h"
#include <doca_gpunetio.h>

/* ------------------------------------------------------------------ */
/* FIFO index queue operations                                        */
/* ------------------------------------------------------------------ */

/*
 * GPU push: FAA(tail) for position, then store value into entry.
 * Multiple GPU threads may push concurrently (MPSC pattern for free_pool).
 */
__device__ void gpu_iq_push(struct index_queue *q, int value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        tail_ref(*(uint32_t *)&q->tail);
    uint32_t pos = tail_ref.fetch_add(1, cuda::memory_order_relaxed) & IQ_MASK;

    cuda::atomic_ref<int32_t, cuda::thread_scope_system>
        entry_ref(*(int32_t *)&q->entries[pos]);

    /* Spin until slot is consumed (IQ_EMPTY) — bounded by queue capacity */
    while (entry_ref.load(cuda::memory_order_acquire) != IQ_EMPTY)
        ;
    entry_ref.store(value, cuda::memory_order_release);
}

/*
 * GPU pop: CAS-loop on head to safely claim a position.
 * Returns slot index (>= 0) or -1 if empty.
 */
__device__ int gpu_iq_pop(struct index_queue *q)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        head_ref(*(uint32_t *)&q->head);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        tail_ref(*(uint32_t *)&q->tail);

    static __shared__ uint32_t cached_tail;
    uint32_t h, t;
    for (;;) {
        h = head_ref.load(cuda::memory_order_acquire);
        t = cached_tail;
        if (h >= t) {
            t = tail_ref.load(cuda::memory_order_acquire);
            cached_tail = t;
            if (h >= t)
                return -1;
        }
        if (head_ref.compare_exchange_weak(h, h + 1,
                cuda::memory_order_acq_rel, cuda::memory_order_relaxed))
            break;
    }

    uint32_t pos = h & IQ_MASK;
    cuda::atomic_ref<int32_t, cuda::thread_scope_system>
        entry_ref(*(int32_t *)&q->entries[pos]);

    int32_t val;
    while ((val = entry_ref.load(cuda::memory_order_acquire)) == IQ_EMPTY)
        ;
    entry_ref.store(IQ_EMPTY, cuda::memory_order_release);
    return (int)val;
}

/* The CPU-side counterparts are static inline in inference/index_queue.h. */

/* ------------------------------------------------------------------ */

/* Control plane: host-mapped, one allocation aliased by host and device pointers */
static struct inference_ring_buffer *g_ring_host = NULL;
static struct inference_ring_buffer *g_ring_device = NULL;

/*
 * Data plane: device-only. There is no host alias on purpose — payload bytes
 * must never be reachable from CPU code.
 */
static struct inference_payload_plane *g_payload_device = NULL;

/* Semaphore handle for CPU to GPU notification */
static struct doca_gpu_semaphore *g_sem_inference_cpu = NULL;

/* Semaphore handle for GPU to CPU request notification */
static struct doca_gpu_semaphore *g_sem_request_cpu = NULL;

/* GPU kernel to get current GPU clock64() value */
__global__ void get_gpu_clock_kernel(uint64_t *output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = clock64();
    }
}

extern "C" void gpu_conn_table_init_entry(struct inference_ring_buffer *ring,
                                          uint32_t conn_hash, uint32_t sent_seq, uint32_t recv_ack)
{
    uint32_t idx = conn_hash & (GPU_CONN_TABLE_SIZE - 1);
    struct gpu_conn_state *cs = &ring->conn_table[idx];
    cs->conn_hash = conn_hash;
    cs->sent_seq = sent_seq;
    cs->recv_ack = recv_ack;
    cs->cwnd = 10 * 1460;
    cs->ssthresh = 65535;
    cs->expected_seq = recv_ack;
    cs->dup_ack_count = 0;
    cs->reorder_count = 0;
    for (int i = 0; i < 4; i++) {
        cs->reorder_seq[i] = 0;
        cs->reorder_len[i] = 0;
    }
    __atomic_store_n((uint32_t *)&cs->active, 1, __ATOMIC_RELEASE);
}

extern "C" void gpu_conn_table_clear_entry(struct inference_ring_buffer *ring, uint32_t conn_hash)
{
    uint32_t idx = conn_hash & (GPU_CONN_TABLE_SIZE - 1);
    __atomic_store_n((uint32_t *)&ring->conn_table[idx].active, 0, __ATOMIC_RELEASE);
    ring->conn_table[idx].conn_hash = 0;
    ring->conn_table[idx].sent_seq = 0;
    ring->conn_table[idx].recv_ack = 0;
}

struct inference_ring_buffer* init_inference_ring_buffer(int gpu_id)
{
    cudaSetDevice(gpu_id);

    /* cudaHostAllocMapped: CPU and GPU share the same memory */
    if (cudaHostAlloc(&g_ring_host, sizeof(*g_ring_host),
                      cudaHostAllocMapped) != cudaSuccess) {
        return NULL;
    }

    if (cudaHostGetDevicePointer(&g_ring_device, g_ring_host, 0) != cudaSuccess) {
        cudaFreeHost(g_ring_host);
        return NULL;
    }

    /*
     * Device-resident payload arena. cudaMalloc (not cudaHostAlloc) is the
     * whole point: these bytes are never mapped into the host address space,
     * so no CPU path can read, tokenize, format or copy them.
     */
    if (cudaMalloc(&g_payload_device, sizeof(*g_payload_device)) != cudaSuccess) {
        fprintf(stderr, "[RING] Failed to allocate %zu B device payload plane\n",
                sizeof(*g_payload_device));
        cudaFreeHost(g_ring_host);
        g_ring_host = NULL;
        return NULL;
    }
    if (cudaMemset(g_payload_device, 0, sizeof(*g_payload_device)) != cudaSuccess) {
        cudaFree(g_payload_device);
        cudaFreeHost(g_ring_host);
        g_payload_device = NULL;
        g_ring_host = NULL;
        return NULL;
    }
    fprintf(stderr, "[RING] Device payload plane: %zu B at %p (control plane %zu B)\n",
            sizeof(*g_payload_device), (void *)g_payload_device, sizeof(*g_ring_host));

    /* Initialize control slots (metadata only — payload lives on the device) */
    for (int i = 0; i < INFERENCE_RING_SIZE; i++) {
        g_ring_host->slots[i].len = 0;
        g_ring_host->slots[i].ready = UVM_STATUS_FREE;
        g_ring_host->slots[i].request_id = 0;
        g_ring_host->slots[i].record_count = 0;
    }

    g_ring_host->head = 0;
    g_ring_host->next_request_id = 1;
    g_ring_host->batch_epoch = 0;
    g_ring_host->pending_count = 0;

    /* Initialize FIFO index queues */
    g_ring_host->free_pool.head = 0;
    g_ring_host->free_pool.tail = INFERENCE_RING_SIZE;
    for (int i = 0; i < IQ_CAPACITY; i++)
        g_ring_host->free_pool.entries[i] = (i < INFERENCE_RING_SIZE) ? i : IQ_EMPTY;

    g_ring_host->request_queue.head = 0;
    g_ring_host->request_queue.tail = 0;
    for (int i = 0; i < IQ_CAPACITY; i++)
        g_ring_host->request_queue.entries[i] = IQ_EMPTY;

    g_ring_host->response_queue.head = 0;
    g_ring_host->response_queue.tail = 0;
    for (int i = 0; i < IQ_CAPACITY; i++)
        g_ring_host->response_queue.entries[i] = IQ_EMPTY;

    for (int i = 0; i < GPU_CONN_TABLE_SIZE; i++) {
        memset((void *)&g_ring_host->conn_table[i], 0, sizeof(struct gpu_conn_state));
    }

    /* Clock synchronization - measure actual GPU frequency */
    uint64_t *d_gpu_clock;
    cudaMalloc(&d_gpu_clock, sizeof(uint64_t));

    struct timespec ts1, ts2;
    uint64_t gpu_clock1, gpu_clock2;

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    get_gpu_clock_kernel<<<1, 1>>>(d_gpu_clock);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_clock1, d_gpu_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    usleep(100000);  /* Wait 100ms to measure frequency */

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    get_gpu_clock_kernel<<<1, 1>>>(d_gpu_clock);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_clock2, d_gpu_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    /* Calculate actual frequency */
    uint64_t cpu_ns1 = (uint64_t)ts1.tv_sec * 1000000000ULL + (uint64_t)ts1.tv_nsec;
    uint64_t cpu_ns2 = (uint64_t)ts2.tv_sec * 1000000000ULL + (uint64_t)ts2.tv_nsec;
    uint64_t gpu_cycles_elapsed = gpu_clock2 - gpu_clock1;
    uint64_t cpu_ns_elapsed = cpu_ns2 - cpu_ns1;
    double gpu_freq_ghz = (double)gpu_cycles_elapsed / (double)cpu_ns_elapsed;

    /* Use first measurement as baseline */
    uint64_t gpu_clock_base = gpu_clock1;
    uint64_t cpu_time_ns_base = cpu_ns1;

    cudaFree(d_gpu_clock);

    /* Save to ring buffer */
    g_ring_host->clock_sync.gpu_clock_base = gpu_clock_base;
    g_ring_host->clock_sync.cpu_time_ns_base = cpu_time_ns_base;
    g_ring_host->clock_sync.gpu_clock_freq_ghz = gpu_freq_ghz;

    /* Ensure CPU writes are visible to GPU */
    cudaDeviceSynchronize();

    return g_ring_device;  /* GPU kernels use this */
}

void free_inference_ring_buffer(struct inference_ring_buffer *ring)
{
    if (g_payload_device) {
        cudaFree(g_payload_device);
        g_payload_device = NULL;
    }
    if (g_ring_host) {
        cudaFreeHost(g_ring_host);
        g_ring_host = NULL;
        g_ring_device = NULL;
    }
    g_sem_inference_cpu = NULL;
    g_sem_request_cpu = NULL;
}

struct inference_payload_plane *get_payload_plane_device(void)
{
    return g_payload_device;
}

/* Set inference result notification semaphore (called after semaphore creation) */
void set_inference_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu)
{
    g_sem_inference_cpu = sem_cpu;
}

/* Set request notification semaphore (GPU to CPU) for semaphore-guided polling */
void set_request_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu)
{
    g_sem_request_cpu = sem_cpu;
}

/* Get request notification semaphore CPU handle */
struct doca_gpu_semaphore* get_request_semaphore_cpu(void)
{
    return g_sem_request_cpu;
}

/*
 * Hand a slot whose response bytes were already written by a GPU kernel over
 * to the GPU TX kernel. Control metadata only — no payload byte is touched.
 *
 * Ordering contract: the caller must have synchronised the CUDA stream that
 * ran the format kernel BEFORE calling this. Kernel completion makes the
 * response bytes visible device-wide; the release-store below then guarantees
 * TX cannot observe RESULT_READY ahead of those bytes.
 */
int publish_inference_result(struct inference_ring_buffer *ring_gpu, uint32_t slot_index)
{
    if (slot_index >= INFERENCE_RING_SIZE || !g_ring_host)
        return 0;

    struct inference_ring_slot *slot = &g_ring_host->slots[slot_index];

    __atomic_store_n(&slot->ready, UVM_STATUS_RESULT_READY, __ATOMIC_RELEASE);

    /* Push to response_queue so GPU TX can find it in O(1) */
    cpu_iq_push(&g_ring_host->response_queue, (int)slot_index);

    /* Also notify via DOCA semaphore (backward compat) */
    if (g_sem_inference_cpu) {
        doca_gpu_semaphore_set_status(g_sem_inference_cpu, slot_index,
                                      DOCA_GPU_SEMAPHORE_STATUS_READY);
    }

    return 1;
}

/* GPU-side function: allocate ring slot via O(1) free_pool pop */
__device__ int gpu_alloc_ring_slot(struct inference_ring_buffer *ring, uint64_t *request_id)
{
    int index = gpu_iq_pop(&ring->free_pool);
    if (index < 0)
        return -1;

    struct inference_ring_slot *slot = &ring->slots[index];

    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        ready_ref(*(uint32_t *)&slot->ready);
    ready_ref.store(UVM_STATUS_PROCESSING, cuda::memory_order_release);

    *request_id = atomicAdd((unsigned long long *)&ring->next_request_id, 1);
    slot->request_id = *request_id;
    slot->len = 0;
    slot->record_count = 0;
    for (int i = 0; i < MAX_RECORDS_PER_SLOT; i++) {
        slot->directory[i].offset = 0;
        slot->directory[i].length = 0;
        slot->directory[i].resp_length = 0;
        slot->directory[i].tcp_sent_seq = 0;
        slot->directory[i].tcp_recv_ack = 0;
    }
    slot->t1_slot_allocated = clock64();

    return index;
}

