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

/*
 * CPU push: __atomic FAA(tail), then store value.
 */
void cpu_iq_push(struct index_queue *q, int value)
{
    uint32_t pos = __atomic_fetch_add((uint32_t *)&q->tail, 1, __ATOMIC_RELAXED) & IQ_MASK;
    /* Spin until slot is consumed */
    while (__atomic_load_n((int32_t *)&q->entries[pos], __ATOMIC_ACQUIRE) != IQ_EMPTY)
        ;
    __atomic_store_n((int32_t *)&q->entries[pos], value, __ATOMIC_RELEASE);
}

/*
 * CPU pop: CAS-loop on head to safely claim a position.
 * Returns slot index (>= 0) or -1 if empty.
 */
int cpu_iq_pop(struct index_queue *q)
{
    uint32_t h, t;
    for (;;) {
        h = __atomic_load_n((uint32_t *)&q->head, __ATOMIC_ACQUIRE);
        t = __atomic_load_n((uint32_t *)&q->tail, __ATOMIC_ACQUIRE);
        if (h >= t)
            return -1;
        if (__atomic_compare_exchange_n((uint32_t *)&q->head, &h, h + 1,
                                         false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            break;
    }

    uint32_t pos = h & IQ_MASK;
    int32_t val;
    while ((val = __atomic_load_n((int32_t *)&q->entries[pos], __ATOMIC_ACQUIRE)) == IQ_EMPTY)
        ;
    __atomic_store_n((int32_t *)&q->entries[pos], (int32_t)IQ_EMPTY, __ATOMIC_RELEASE);
    return (int)val;
}

/* ------------------------------------------------------------------ */

/* Global variables: host and device pointer mapping */
static struct inference_ring_buffer *g_ring_host = NULL;
static struct inference_ring_buffer *g_ring_device = NULL;

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

    /* Initialize slots */
    for (int i = 0; i < INFERENCE_RING_SIZE; i++) {
        g_ring_host->slots[i].len = 0;
        g_ring_host->slots[i].ready = UVM_STATUS_FREE;
        g_ring_host->slots[i].request_id = 0;
        memset(g_ring_host->slots[i].data, 0, sizeof(g_ring_host->slots[i].data));
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
    if (g_ring_host) {
        cudaFreeHost(g_ring_host);
        g_ring_host = NULL;
        g_ring_device = NULL;
    }
    g_sem_inference_cpu = NULL;
    g_sem_request_cpu = NULL;
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

/* CPU write - UVM with explicit CUDA sync for concurrency */
int cpu_write_inference_result_to_gpu_ring(struct inference_ring_buffer *ring_gpu,
                                            uint32_t slot_index,
                                            const char *result)
{
    if (slot_index >= INFERENCE_RING_SIZE) {
        return 0;
    }

    struct inference_ring_slot *slot = &g_ring_host->slots[slot_index];

    uint32_t max_len = sizeof(slot->data) - 1;
    uint32_t len = (uint32_t)strlen(result);
    if (len > max_len)
        len = max_len;
    memcpy(slot->data, result, len);
    slot->data[len] = '\0';
    slot->len = len;

    slot->record_count = 1;
    slot->directory[0].offset = 0;
    slot->directory[0].length = len;
    slot->directory[0].tcp_sent_seq = slot->tcp_sent_seq;
    slot->directory[0].tcp_recv_ack = slot->tcp_recv_ack;

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

/* Get host-side pointer to slot data (for cudaMemcpyAsync source) */
const char *get_slot_data_host(uint32_t slot_index)
{
    if (slot_index >= INFERENCE_RING_SIZE || !g_ring_host) return NULL;
    return g_ring_host->slots[slot_index].data;
}

uint32_t get_slot_len_host(uint32_t slot_index)
{
    if (slot_index >= INFERENCE_RING_SIZE || !g_ring_host) return 0;
    return g_ring_host->slots[slot_index].len;
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
        slot->directory[i].tcp_sent_seq = 0;
        slot->directory[i].tcp_recv_ack = 0;
    }
    slot->t1_slot_allocated = clock64();

    return index;
}

__device__ void gpu_store_inference_data_to_slot(struct inference_ring_buffer *ring, int slot_index, const char *data)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pending_ref(*(uint32_t*)&ring->pending_count);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ready_ref(*(uint32_t*)&ring->slots[slot_index].ready);

    struct inference_ring_slot *slot = &ring->slots[slot_index];

    slot->start_timestamp = clock64();

    uint32_t max_len = sizeof(slot->data) - 1;
    uint32_t offset = slot->len;
    uint32_t rec = slot->record_count;

    uint32_t len = 0;
    while (data[len] != '\0' && (offset + len) < max_len)
        len++;

    for (uint32_t i = 0; i < len; i++)
        slot->data[offset + i] = data[i];
    slot->data[offset + len] = '\0';

    if (rec < MAX_RECORDS_PER_SLOT) {
        slot->directory[rec].offset = offset;
        slot->directory[rec].length = len;
        slot->record_count = rec + 1;
    }
    slot->len = offset + len;

    slot->t2_gpu_wrote_uvm = clock64();

    pending_ref.fetch_add(1, cuda::memory_order_release);
    ready_ref.store(UVM_STATUS_PARAM_READY, cuda::memory_order_release);

    gpu_iq_push(&ring->request_queue, slot_index);
}


