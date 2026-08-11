/*
 * GAZSI - Slot ownership regression
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Both queue consumers used to treat a queue entry as proof of ownership: if
 * the state CAS failed they handed the slot to free_pool ("it came from a
 * queue, so it must belong to the pool"). That is exactly backwards. A failed
 * CAS means somebody else owns the slot, and that owner will release it when
 * they are done — so the index lands in free_pool twice and two live requests
 * are eventually given the same slot, overwriting each other's payload and TCP
 * routing context.
 *
 * This pins the corrected contract for both consumers:
 *   host  (main.c collect_ready_slots -> gazsi_claim_request_slot)
 *   device(kernels/http_server.cu TX  -> gazsi_tx_claim_slot)
 *
 * On failure the helpers must mutate NO shared state — in particular free_pool
 * must be untouched.
 *
 * Needs a CUDA device for the TX half. No DOCA, no TensorRT engine.
 *
 * Build and run:
 *   nvcc -std=c++17 -I../inference -o test_slot_claim test_slot_claim.cu
 *   ./test_slot_claim
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slot_claim.h"
#include "tensorrt.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                       \
	do {                                                                   \
		g_checks++;                                                    \
		if (!(cond)) {                                                 \
			g_failures++;                                          \
			printf("  FAIL %s:%d: ", __FILE__, __LINE__);          \
			printf(__VA_ARGS__);                                   \
			printf("\n");                                          \
		}                                                              \
	} while (0)

#define CUDA_OK(call)                                                          \
	do {                                                                   \
		cudaError_t e = (call);                                        \
		if (e != cudaSuccess) {                                        \
			printf("FATAL %s:%d: %s\n", __FILE__, __LINE__,        \
			       cudaGetErrorString(e));                         \
			exit(1);                                               \
		}                                                              \
	} while (0)

/* ------------------------------------------------------------------ */

static void queue_reset(struct index_queue *q)
{
	q->head = 0;
	q->tail = 0;
	for (int i = 0; i < IQ_CAPACITY; i++)
		q->entries[i] = IQ_EMPTY;
}

static uint32_t queue_depth(const struct index_queue *q)
{
	return q->tail - q->head;
}

static void ring_reset(struct inference_ring_buffer *ring)
{
	memset(ring, 0, sizeof(*ring));
	queue_reset(&ring->free_pool);
	queue_reset(&ring->request_queue);
	queue_reset(&ring->response_queue);
}

/* ------------------------------------------------------------------ */
/* Host: request_queue -> PARAM_READY -> PROCESSING                    */
/* ------------------------------------------------------------------ */

static void test_host_claim(struct inference_ring_buffer *ring)
{
	printf("host claim (request_queue)\n");

	/* A live entry is claimed and flips to PROCESSING. */
	ring_reset(ring);
	ring->slots[5].ready = UVM_STATUS_PARAM_READY;
	cpu_iq_push(&ring->request_queue, 5);

	CHECK(gazsi_claim_request_slot(ring) == 5, "live entry not claimed");
	CHECK(ring->slots[5].ready == UVM_STATUS_PROCESSING,
	      "claimed slot state %u, want PROCESSING", ring->slots[5].ready);
	CHECK(queue_depth(&ring->free_pool) == 0,
	      "free_pool grew to %u on a successful claim", queue_depth(&ring->free_pool));

	/*
	 * A stale entry (slot already owned elsewhere — here the scan-mode
	 * claimer got there first, so it is PROCESSING) is dropped silently.
	 */
	ring_reset(ring);
	ring->slots[7].ready = UVM_STATUS_PROCESSING;
	cpu_iq_push(&ring->request_queue, 7);

	CHECK(gazsi_claim_request_slot(ring) == -1, "stale entry was claimed");
	CHECK(ring->slots[7].ready == UVM_STATUS_PROCESSING,
	      "stale slot state changed to %u", ring->slots[7].ready);
	CHECK(queue_depth(&ring->free_pool) == 0,
	      "REGRESSION: stale entry pushed %u index(es) to free_pool",
	      queue_depth(&ring->free_pool));

	/* The bug's actual shape: one slot queued twice. */
	ring_reset(ring);
	ring->slots[11].ready = UVM_STATUS_PARAM_READY;
	cpu_iq_push(&ring->request_queue, 11);
	cpu_iq_push(&ring->request_queue, 11);

	CHECK(gazsi_claim_request_slot(ring) == 11, "first of a duplicate pair not claimed");
	CHECK(gazsi_claim_request_slot(ring) == -1, "duplicate entry claimed twice");
	CHECK(queue_depth(&ring->free_pool) == 0,
	      "REGRESSION: duplicate entry published slot 11 to free_pool (depth %u); "
	      "the owner will publish it too, so two requests would share it",
	      queue_depth(&ring->free_pool));

	/* A stale entry must not hide a live one behind it. */
	ring_reset(ring);
	ring->slots[7].ready = UVM_STATUS_FREE;
	ring->slots[9].ready = UVM_STATUS_PARAM_READY;
	cpu_iq_push(&ring->request_queue, 7);
	cpu_iq_push(&ring->request_queue, 9);

	CHECK(gazsi_claim_request_slot(ring) == 9, "live entry behind a stale one was lost");
	CHECK(queue_depth(&ring->free_pool) == 0,
	      "REGRESSION: free_pool depth %u after skipping a stale entry",
	      queue_depth(&ring->free_pool));

	/* Out-of-range index: dropped, never indexed into slots[]. */
	ring_reset(ring);
	cpu_iq_push(&ring->request_queue, INFERENCE_RING_SIZE + 3);

	CHECK(gazsi_claim_request_slot(ring) == -1, "out-of-range entry was claimed");
	CHECK(queue_depth(&ring->free_pool) == 0,
	      "out-of-range entry pushed to free_pool (depth %u)",
	      queue_depth(&ring->free_pool));

	/* Empty queue is not an error. */
	ring_reset(ring);
	CHECK(gazsi_claim_request_slot(ring) == -1, "empty queue did not return -1");
}

/* ------------------------------------------------------------------ */
/* Device: response_queue -> RESULT_READY -> CONSUMED                  */
/* ------------------------------------------------------------------ */

/*
 * One warp, lane 0 claims and broadcasts — the same shape as the TX kernel.
 * results[i] is 1 if the i-th attempt took ownership.
 */
__global__ void tx_claim_kernel(struct inference_ring_buffer *ring,
				const int *slot_indices, int n, int *results)
{
	unsigned lane_id = threadIdx.x % 32;

	for (int i = 0; i < n; i++) {
		unsigned claimed = 0;

		if (lane_id == 0)
			claimed = gazsi_tx_claim_slot(ring, slot_indices[i]) ? 1u : 0u;
		claimed = __shfl_sync(0xffffffff, claimed, 0);

		if (lane_id == 0)
			results[i] = (int)claimed;
		__syncwarp();
	}
}

static void test_device_claim(void)
{
	printf("device claim (response_queue, TX warp)\n");

	struct inference_ring_buffer *ring = NULL;
	int *idx = NULL;
	int *res = NULL;
	const int n = 4;

	CUDA_OK(cudaMallocManaged(&ring, sizeof(*ring)));
	CUDA_OK(cudaMallocManaged(&idx, n * sizeof(int)));
	CUDA_OK(cudaMallocManaged(&res, n * sizeof(int)));

	ring_reset(ring);
	ring->slots[3].ready = UVM_STATUS_RESULT_READY;  /* claimable once */
	ring->slots[4].ready = UVM_STATUS_FREE;          /* stale entry */

	idx[0] = 3;                          /* claim succeeds */
	idx[1] = 3;                          /* same slot again: stale */
	idx[2] = 4;                          /* never was RESULT_READY */
	idx[3] = INFERENCE_RING_SIZE + 1;    /* out of range */

	memset(res, 0, n * sizeof(int));

	tx_claim_kernel<<<1, 32>>>(ring, idx, n, res);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	CHECK(res[0] == 1, "RESULT_READY slot was not claimed");
	CHECK(ring->slots[3].ready == UVM_STATUS_CONSUMED,
	      "claimed slot state %u, want CONSUMED", ring->slots[3].ready);

	CHECK(res[1] == 0, "already-consumed slot was claimed a second time");
	CHECK(ring->slots[3].ready == UVM_STATUS_CONSUMED,
	      "second claim changed state to %u", ring->slots[3].ready);

	CHECK(res[2] == 0, "FREE slot was claimed");
	CHECK(ring->slots[4].ready == UVM_STATUS_FREE,
	      "stale slot state changed to %u", ring->slots[4].ready);

	CHECK(res[3] == 0, "out-of-range index was claimed");

	CHECK(queue_depth(&ring->free_pool) == 0,
	      "REGRESSION: stale TX entries pushed %u index(es) to free_pool",
	      queue_depth(&ring->free_pool));

	CUDA_OK(cudaFree(res));
	CUDA_OK(cudaFree(idx));
	CUDA_OK(cudaFree(ring));
}

/* ------------------------------------------------------------------ */
/* Inference failure classification                                    */
/* ------------------------------------------------------------------ */

/*
 * Only the mapping is pinned here. Provoking a real CUDA/TensorRT/binding
 * failure needs an engine and a second GPU, so batch_infer_device_resident's
 * own classification is covered by inspection, not by this test.
 */
static void test_infer_status(void)
{
	printf("inference status classification\n");

	CHECK(GAZSI_INFER_OK == 0, "OK must be 0");
	CHECK(GAZSI_INFER_ERR_BATCH != 0 && GAZSI_INFER_ERR_FATAL != 0,
	      "errors must stay non-zero for legacy rc != 0 callers");
	CHECK(GAZSI_INFER_ERR_BATCH != GAZSI_INFER_ERR_FATAL,
	      "batch and fatal must be distinguishable");
	CHECK(!gazsi_infer_is_fatal(GAZSI_INFER_OK), "OK classified fatal");
	CHECK(!gazsi_infer_is_fatal(GAZSI_INFER_ERR_BATCH),
	      "batch failure classified fatal — serving would stop on a bad descriptor");
	CHECK(gazsi_infer_is_fatal(GAZSI_INFER_ERR_FATAL),
	      "fatal failure not classified fatal — the server would answer every "
	      "request with inference_failed forever");
}

int main(void)
{
	printf("GAZSI slot ownership regression\n");

	struct inference_ring_buffer *ring =
		(struct inference_ring_buffer *)malloc(sizeof(*ring));
	if (!ring) {
		printf("FATAL: out of memory\n");
		return 1;
	}

	test_host_claim(ring);
	free(ring);

	test_device_claim();
	test_infer_status();

	printf("%s: %d checks, %d failures\n",
	       g_failures ? "FAILED" : "PASSED", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
