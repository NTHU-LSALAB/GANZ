/*
 * GPU Packet Processing - Slot ownership claims
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * A queue entry is a hint, not a title deed.
 *
 * request_queue and response_queue carry slot indices, but the thing that
 * actually confers ownership of a slot is winning the CAS on slot->ready. The
 * two can disagree: the host runs a scan mode that claims PARAM_READY slots
 * without consulting request_queue, and either queue can still hold an index
 * for a slot that has since moved on. Such an entry is stale.
 *
 * A stale entry must be DROPPED and nothing else. The tempting alternative —
 * "the slot came from a queue, so hand it back to free_pool" — publishes a slot
 * this actor does not own. Whoever does own it will publish it again when they
 * finish, so free_pool ends up holding the index twice, two future requests are
 * handed the same slot, and their payloads and TCP routing context overwrite
 * each other. Dropping the entry costs nothing: the owner is responsible for
 * releasing the slot, exactly as it would have been had the entry never
 * existed.
 *
 * Both helpers therefore have the same contract: on failure they mutate no
 * shared state at all.
 */

#ifndef GAZSI_SLOT_CLAIM_H
#define GAZSI_SLOT_CLAIM_H

#include "ring_buffer.h"
#include "index_queue.h"

/*
 * Host: pop request_queue until a slot is genuinely claimed.
 *
 * Transitions the winning slot PARAM_READY -> PROCESSING and returns its index.
 * Returns -1 once the queue is drained. Stale and out-of-range entries are
 * discarded; pending_count is left alone because whoever wins the CAS for a
 * given slot is the one that accounts for it.
 */
static inline int gazsi_claim_request_slot(struct inference_ring_buffer *ring)
{
	for (;;) {
		int idx = cpu_iq_pop(&ring->request_queue);
		uint32_t expected;

		if (idx < 0)
			return -1;
		if (idx >= INFERENCE_RING_SIZE)
			continue; /* corrupt entry: drop it, own nothing */

		expected = UVM_STATUS_PARAM_READY;
		if (__atomic_compare_exchange_n(&ring->slots[idx].ready,
						&expected, UVM_STATUS_PROCESSING,
						false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
			return idx;
		/* Stale entry. Drop it — do NOT touch free_pool. */
	}
}

#ifdef __CUDACC__

/*
 * Device (TX): claim a slot popped from response_queue.
 *
 * Transitions RESULT_READY -> CONSUMED. Returns true if this caller now owns
 * the slot, false if the entry was stale or out of range — in which case the
 * caller must simply move on to the next entry.
 *
 * Single-thread call: invoke from one lane and broadcast the result.
 */
__device__ static inline bool gazsi_tx_claim_slot(struct inference_ring_buffer *ring,
						  int slot_idx)
{
	if (slot_idx < 0 || slot_idx >= INFERENCE_RING_SIZE)
		return false;

	unsigned int old = atomicCAS((unsigned int *)&ring->slots[slot_idx].ready,
				     (unsigned int)UVM_STATUS_RESULT_READY,
				     (unsigned int)UVM_STATUS_CONSUMED);

	return old == (unsigned int)UVM_STATUS_RESULT_READY;
}

#endif /* __CUDACC__ */

#endif /* GAZSI_SLOT_CLAIM_H */
