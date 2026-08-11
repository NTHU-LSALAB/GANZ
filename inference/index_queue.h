/*
 * GPU Packet Processing - Host-side lock-free index queue primitives
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * The host half of the FIFO index queues. It used to live in ring_buffer.cu,
 * which pulls in CUDA, DOCA and the whole control plane; it is header-only here
 * so the ownership logic built on top of it (slot_claim.h) can be compiled and
 * exercised on its own, with nothing but a C compiler.
 *
 * Behaviour is unchanged from the out-of-line versions — same atomics, same
 * memory orders, same spin conditions. The device-side counterparts stay in
 * ring_buffer.cu: nothing outside device code needs them.
 */

#ifndef GAZSI_INDEX_QUEUE_H
#define GAZSI_INDEX_QUEUE_H

#include <stdint.h>

/*
 * CPU push: __atomic FAA(tail), then store value.
 */
static inline void cpu_iq_push(struct index_queue *q, int value)
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
static inline int cpu_iq_pop(struct index_queue *q)
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

#endif /* GAZSI_INDEX_QUEUE_H */
