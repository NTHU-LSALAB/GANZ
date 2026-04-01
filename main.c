/*
 * GPU Packet Processing - Main Application
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <rte_ethdev.h>

#include "common.h"
#include "tcp/session.h"
#include "tcp/cpu_rss.h"
#include "inference/tensorrt.h"
#include "inference/ring_buffer.h"  /* Ring Buffer UVM - supports concurrency */
#include <pthread.h>
#include <ctype.h>
#define SLEEP_IN_NANOS (10 * 1000) /* Sample the PE every 10 microseconds  */

/* Buffer and inference constants */
#define DATA_BUFFER_SIZE     1024    /* Max size for inference data buffer */
#define MAX_EMBEDDING_DIM    4096    /* Max output embedding dimension (alloc size) */
#define MAX_DATA_LEN         855     /* Maximum valid data length (slot->data is 856 bytes, -1 for '\0') */
#define SPIN_ITERATIONS      500

/*
 * URL decode function - converts %XX encoding back to original characters
 * Example: %20 -> space, %2B -> +
 */
static void url_decode(char *str) {
    char *src = str;
    char *dst = str;

    while (*src) {
        if (*src == '%' && isxdigit((unsigned char)src[1]) && isxdigit((unsigned char)src[2])) {
            /* Decode %XX */
            char hex[3] = {src[1], src[2], '\0'};
            *dst++ = (char)strtol(hex, NULL, 16);
            src += 3;
        } else if (*src == '+') {
            /* '+' also represents space */
            *dst++ = ' ';
            src++;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

DOCA_LOG_REGISTER(GPUNET);

bool force_quit;
static struct doca_gpu *gpu_dev;
static struct app_gpu_cfg app_cfg = {0};
static struct doca_dev *ddev;
static uint16_t dpdk_dev_port_id;
static struct rxq_tcp_queues tcp_queues;
static struct txq_http_queues http_queues;
static struct doca_flow_port *df_port;

/* Simple UVM inference buffer - Linus-style global variable */
static struct inference_ring_buffer *g_inference_ring_buf = NULL;  /* Ring Buffer - supports concurrent requests */

/* Context Pool: concurrent inference threads, each bound to an execution context */
#define NUM_INFERENCE_THREADS 1  /* Single thread + Batching is more efficient: 1 batch inference is faster than 4 separate inferences */
static pthread_t inference_reader_threads[NUM_INFERENCE_THREADS];
static bool inference_reader_running = false;

struct reader_thread_args {
	int thread_id;
	int context_id;
};

static struct doca_pe *pe;

/* UVM inference processing */
static TensorRT_Model_t *tensorrt_model = NULL;
static int g_cuda_id = 0;  /* CUDA device ID */
/* Context Pool removes global mutex - each thread uses independent context for lock-free concurrency */

/*
 * Get CPU timestamp in nanoseconds - for comparison with GPU clock64()
 */
static inline uint64_t get_timestamp_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/*
 * ============================================================================
 * Dynamic Batching Inference Reader
 * ============================================================================
 *
 * Collect multiple PARAM_READY slots and process in one batch inference.
 *
 * Parameters:
 * - BATCH_MAX_SIZE: Maximum batch size supported by TensorRT engine (dynamic batch 1~8)
 * - BATCH_COLLECT_WINDOW_US: Time window to collect concurrent requests
 */
#define BATCH_MAX_SIZE 8               /* Max concurrent batch size */
static int g_fixed_batch = 0;          /* 0=EWMA adaptive (default), >0=fixed batch */

#define EWMA_ALPHA             0.1
#define ADAPTIVE_MAX_WAIT_US   2000
#define ADAPTIVE_COLD_ROUNDS   3

/*
 * Collect PARAM_READY slots (non-blocking, using CAS atomic claim)
 * Returns the number of slots collected
 *
 * Zero-copy optimization: URL-decodes directly on slot->data (slot is in
 * PROCESSING state, CPU-owned) and returns pointers into the ring buffer.
 * Eliminates the per-request memcpy to a stack buffer.
 */
static int g_use_scan = 0;
static uint64_t g_collect_time_sum = 0;
static uint64_t g_collect_count = 0;

static int collect_ready_slots(struct inference_ring_buffer *ring,
                               uint32_t *batch_slots,
                               uint32_t *batch_record_idx,
                               const char **batch_texts,
                               int max_batch) {
	if (__atomic_load_n(&ring->pending_count, __ATOMIC_ACQUIRE) == 0)
		return 0;

	uint64_t t_start = get_timestamp_ns();
	int count = 0;

	if (g_use_scan) {
		while (cpu_iq_pop(&ring->request_queue) >= 0) ;

		for (int i = 0; i < INFERENCE_RING_SIZE && count < max_batch; i++) {
			uint32_t expected = UVM_STATUS_PARAM_READY;
			if (__atomic_compare_exchange_n(&ring->slots[i].ready,
			                                &expected, UVM_STATUS_PROCESSING,
			                                false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
				__atomic_fetch_sub(&ring->pending_count, 1, __ATOMIC_RELEASE);
				int idx = i;
				struct inference_ring_slot *slot = &ring->slots[idx];
				uint32_t data_len = slot->len;
				if (data_len > MAX_DATA_LEN) {
					__atomic_store_n(&slot->ready, UVM_STATUS_FREE, __ATOMIC_RELEASE);
					cpu_iq_push(&ring->free_pool, idx);
					continue;
				}
				slot->data[data_len] = '\0';
				url_decode(slot->data);
				if (strlen(slot->data) > 512) slot->data[512] = '\0';
				batch_slots[count] = (uint32_t)idx;
				batch_record_idx[count] = 0;
				batch_texts[count] = slot->data;
				slot->t3_cpu_read = get_timestamp_ns();
				count++;
			}
		}
		uint64_t elapsed = get_timestamp_ns() - t_start;
		g_collect_time_sum += elapsed;
		g_collect_count++;
		return count;
	}

	while (count < max_batch) {
		int idx = cpu_iq_pop(&ring->request_queue);
		if (idx < 0)
			break;

		struct inference_ring_slot *slot = &ring->slots[idx];

		uint32_t expected = UVM_STATUS_PARAM_READY;
		if (!__atomic_compare_exchange_n(&slot->ready,
		                                  &expected,
		                                  UVM_STATUS_PROCESSING,
		                                  false,
		                                  __ATOMIC_ACQ_REL,
		                                  __ATOMIC_ACQUIRE)) {
			/* Shouldn't happen — slot came from request_queue.
			 * Return to free_pool as safety fallback. */
			cpu_iq_push(&ring->free_pool, idx);
			continue;
		}

		__atomic_fetch_sub(&ring->pending_count, 1, __ATOMIC_RELEASE);

		uint32_t data_len = slot->len;
		if (data_len > MAX_DATA_LEN) {
			DOCA_LOG_WARN("Slot %u has invalid len=%u, resetting to FREE", (uint32_t)idx, data_len);
			__atomic_store_n(&slot->ready, UVM_STATUS_FREE, __ATOMIC_RELEASE);
			cpu_iq_push(&ring->free_pool, idx);
			continue;
		}

		slot->t3_cpu_read = get_timestamp_ns();

		uint32_t nrec = slot->record_count;
		if (nrec == 0) nrec = 1;

		for (uint32_t r = 0; r < nrec && count < max_batch; r++) {
			uint32_t off = slot->directory[r].offset;
			uint32_t rlen = slot->directory[r].length;
			if (rlen == 0 && r == 0) {
				rlen = data_len;
				off = 0;
			}
			if (off + rlen > sizeof(slot->data)) continue;
			slot->data[off + rlen] = '\0';
			if (rlen > 512) slot->data[off + 512] = '\0';

			batch_slots[count] = (uint32_t)idx;
			batch_record_idx[count] = r;
			batch_texts[count] = slot->data + off;
			count++;
		}
	}

	uint64_t elapsed = get_timestamp_ns() - t_start;
	g_collect_time_sum += elapsed;
	g_collect_count++;
	return count;
}

/*
 * Inference-Driven Adaptive Batching
 *
 * Core insight: each inference round acts as a free traffic probe.
 * During cudaStreamSynchronize (~1.6-4.9ms), GPU RX kernel keeps
 * incrementing pending_count. When CPU wakes:
 *   λ_now = Δpending_count / t_inference
 * This is a zero-cost instantaneous arrival rate, unique to GPU-native
 * network architectures where NIC delivers directly to GPU memory.
 *
 * Dispatch rule: if (1/λ) > c₀/(n*(n+1)), dispatch immediately.
 *   - 1/λ = expected wait for next arrival
 *   - c₀/(n*(n+1)) = marginal cost saving from batching one more
 *   - c₀, c_m calibrated online via EWMA regression (zero manual setup)
 */
void* simple_inference_reader(void* arg) {
	struct reader_thread_args *args = (struct reader_thread_args*)arg;
	int thread_id = args->thread_id;
	(void)args->context_id;

	DOCA_LOG_INFO("[ADAPTIVE] Inference reader thread %d started (max_batch=%d)",
	              thread_id, BATCH_MAX_SIZE);

	cudaSetDevice(g_cuda_id);
	DOCA_LOG_INFO("[ADAPTIVE] Thread %d: Using GPU %d for TensorRT", thread_id, g_cuda_id);

	uint32_t batch_slots[BATCH_MAX_SIZE];
	uint32_t batch_record_idx[BATCH_MAX_SIZE];
	const char *batch_texts[BATCH_MAX_SIZE];
	float batch_embeddings[BATCH_MAX_SIZE * MAX_EMBEDDING_DIM];
	int batch_token_counts[BATCH_MAX_SIZE];
	char result_buffer[DATA_BUFFER_SIZE];

	struct timespec start_time, end_time;

	/* Adaptive batching state — all derived online, zero manual config */
	double lambda_now = 0.0;
	double c0_est = 0.0;
	double cm_est = 0.0;
	int cal_count = 0;

	/* EWMA regression: tracks E[n], E[t], E[n²], E[n×t] */
	double ew_n = 0.0, ew_t = 0.0, ew_n2 = 0.0, ew_nt = 0.0;

	double lambda_ema = 0.0, lambda_var_ema = 0.0;
	int probe_count = 0;

	long stat_batches = 0, stat_requests = 0, stat_waits = 0;

	while (inference_reader_running && !force_quit) {
		if (g_inference_ring_buf == NULL) {
			usleep(100);
			continue;
		}

		int batch_size = collect_ready_slots(g_inference_ring_buf, batch_slots,
		                                      batch_record_idx, batch_texts, BATCH_MAX_SIZE);

		if (batch_size == 0) {
			for (int spin = 0; spin < 2000; spin++) {
				if (__atomic_load_n(&g_inference_ring_buf->pending_count,
				                    __ATOMIC_ACQUIRE) > 0)
					break;
				__asm__ __volatile__("pause" ::: "memory");
			}
			continue;
		}

		if (g_fixed_batch > 0) {
			/* Fixed batch mode: wait until we have exactly g_fixed_batch requests */
			int target = (g_fixed_batch > BATCH_MAX_SIZE) ? BATCH_MAX_SIZE : g_fixed_batch;
			int wait_rounds = 0;
			while (batch_size < target && wait_rounds < 1000) {
				usleep(100);
				int more = collect_ready_slots(g_inference_ring_buf,
				                               &batch_slots[batch_size],
				                               &batch_record_idx[batch_size],
				                               &batch_texts[batch_size],
				                               target - batch_size);
				if (more > 0) batch_size += more;
				wait_rounds++;
			}
		} else {
			/* Adaptive dispatch: wait for more only if marginal saving > wait cost */
			if (cal_count >= ADAPTIVE_COLD_ROUNDS
			    && lambda_now > 0
			    && c0_est > 0
			    && batch_size < BATCH_MAX_SIZE) {

				double save_us = c0_est / ((double)batch_size * (batch_size + 1));

				while ((1.0 / lambda_now) <= save_us && batch_size < BATCH_MAX_SIZE) {
					double wait_us = 1.0 / lambda_now;
					if (wait_us > ADAPTIVE_MAX_WAIT_US) wait_us = ADAPTIVE_MAX_WAIT_US;
					uint64_t spin_until = get_timestamp_ns()
					                    + (uint64_t)(wait_us * 1000.0);
					while (get_timestamp_ns() < spin_until)
						__asm__ __volatile__("pause" ::: "memory");

				int more = collect_ready_slots(g_inference_ring_buf,
				                               &batch_slots[batch_size],
				                               &batch_record_idx[batch_size],
				                               &batch_texts[batch_size],
				                               BATCH_MAX_SIZE - batch_size);
					if (more == 0)
						break;
					batch_size += more;
					stat_waits++;
					save_us = c0_est / ((double)batch_size * (batch_size + 1));
				}
			}
		}

		if ((stat_batches & 0x1FF) == 0)
			DOCA_LOG_INFO("[ADAPTIVE] batch=%d λ=%.0f/s c₀=%.0f c_m=%.0f α=%.2f probe_cv=%.2f",
			              batch_size, lambda_now * 1e6, c0_est, cm_est,
			              (probe_count > 2 && lambda_ema > 1e-9)
			              ? 0.05 + 0.45 * fmin(sqrt(lambda_var_ema) / lambda_ema, 1.0)
			              : EWMA_ALPHA,
			              (lambda_ema > 1e-9) ? sqrt(lambda_var_ema) / lambda_ema : 0.0);

		uint32_t pending_before = __atomic_load_n(
			&g_inference_ring_buf->pending_count, __ATOMIC_ACQUIRE);

		if (tensorrt_model != NULL) {
			clock_gettime(CLOCK_MONOTONIC, &start_time);

			uint64_t t4 = get_timestamp_ns();
			for (int i = 0; i < batch_size; i++)
				g_inference_ring_buf->slots[batch_slots[i]].t4_tensorrt_start = t4;

			batch_tokenize_and_infer(tensorrt_model, batch_texts, batch_size,
			                          batch_embeddings, batch_token_counts);

			uint64_t t5 = get_timestamp_ns();
			for (int i = 0; i < batch_size; i++)
				g_inference_ring_buf->slots[batch_slots[i]].t5_tensorrt_end = t5;

			clock_gettime(CLOCK_MONOTONIC, &end_time);
			long elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
			                  (end_time.tv_nsec - start_time.tv_nsec) / 1000;

			uint32_t pending_after = __atomic_load_n(
				&g_inference_ring_buf->pending_count, __ATOMIC_ACQUIRE);
			int n_arrived = (int)pending_after - (int)pending_before;
			if (n_arrived < 0) n_arrived = 0;

			lambda_now = (elapsed_us > 0)
				? (double)n_arrived / (double)elapsed_us
				: 0.0;

			if (probe_count == 0) {
				lambda_ema = lambda_now;
				lambda_var_ema = 0.0;
			} else {
				double diff = lambda_now - lambda_ema;
				lambda_var_ema = 0.1 * diff * diff + 0.9 * lambda_var_ema;
				lambda_ema = 0.1 * lambda_now + 0.9 * lambda_ema;
			}
			probe_count++;

			/* EWMA online regression: T(n) = c₀ + n × c_m */
			double nd = (double)batch_size;
			double td = (double)elapsed_us;

			double alpha = EWMA_ALPHA;
			if (probe_count > 2 && lambda_ema > 1e-9) {
				double cv = sqrt(lambda_var_ema) / lambda_ema;
				alpha = 0.05 + 0.45 * (cv > 1.0 ? 1.0 : cv);
			}

			if (cal_count == 0) {
				ew_n = nd; ew_t = td;
				ew_n2 = nd * nd; ew_nt = nd * td;
			} else {
				ew_n  = alpha * nd      + (1.0 - alpha) * ew_n;
				ew_t  = alpha * td      + (1.0 - alpha) * ew_t;
				ew_n2 = alpha * nd * nd + (1.0 - alpha) * ew_n2;
				ew_nt = alpha * nd * td + (1.0 - alpha) * ew_nt;
			}
			cal_count++;

			double var_n = ew_n2 - ew_n * ew_n;
			if (var_n < 1e-6) var_n = 1e-6;
			if (cal_count >= ADAPTIVE_COLD_ROUNDS && var_n > 0.01) {
				cm_est = (ew_nt - ew_n * ew_t) / var_n;
				c0_est = ew_t - cm_est * ew_n;
				if (!isfinite(cm_est) || cm_est < 5.0) cm_est = 5.0;
				if (!isfinite(c0_est) || c0_est < 50.0) c0_est = 50.0;
			}

			stat_batches++;
			stat_requests += batch_size;

			/* Per-stage timing accumulation for profiling */
			{
				static uint64_t sum_t3t4 = 0, sum_t4t5 = 0, sum_t5t6 = 0;
				static uint64_t prof_count = 0;
				uint64_t t6_now = get_timestamp_ns();
				for (int i = 0; i < batch_size; i++) {
					struct inference_ring_slot *s = &g_inference_ring_buf->slots[batch_slots[i]];
					uint64_t dt34 = t4 - s->t3_cpu_read;
					uint64_t dt45 = t5 - t4;
					uint64_t dt56 = t6_now - t5;
					sum_t3t4 += dt34;
					sum_t4t5 += dt45;
					sum_t5t6 += dt56;
					prof_count++;
				}
				static uint64_t last_print = 0;
				static double baseline_infer_us = 0;
				if (prof_count >= 100 && prof_count - last_print >= 5000) {
					last_print = prof_count;
					double avg_infer = (double)sum_t4t5 / prof_count / 1000.0;
					if (baseline_infer_us < 1.0)
						baseline_infer_us = avg_infer;
					double drift_pct = (avg_infer - baseline_infer_us) / baseline_infer_us * 100.0;
					fprintf(stderr, "[PROFILE] n=%lu queue+tok=%.1fµs infer=%.1fµs result=%.1fµs total=%.1fµs drift=%.1f%%\n",
						(unsigned long)prof_count,
						(double)sum_t3t4 / prof_count / 1000.0,
						avg_infer,
						(double)sum_t5t6 / prof_count / 1000.0,
						(double)(sum_t3t4 + sum_t4t5 + sum_t5t6) / prof_count / 1000.0,
						drift_pct);
					if (drift_pct > 15.0) {
						fprintf(stderr, "[DRIFT] inference +%.0f%%, resetting profiler baseline\n", drift_pct);
						baseline_infer_us = avg_infer;
						sum_t3t4 = sum_t4t5 = sum_t5t6 = 0;
						prof_count = 0;
					}
					if (g_collect_count > 0)
						fprintf(stderr, "[COLLECT] mode=%s avg=%.1fns count=%lu\n",
							g_use_scan ? "O(N)-scan" : "O(1)-FIFO",
							(double)g_collect_time_sum / g_collect_count,
							(unsigned long)g_collect_count);
				}
			}

			DOCA_LOG_INFO("[ADAPTIVE] done batch=%d %ldµs (%.1f/req) arrived=%d λ=%.0f/s",
			              batch_size, elapsed_us, (float)elapsed_us / batch_size,
			              n_arrived, lambda_now * 1e6);

			for (int i = 0; i < batch_size; i++) {
				uint32_t slot_idx = batch_slots[i];
				uint32_t rec = batch_record_idx[i];
				float *emb = &batch_embeddings[i * MAX_EMBEDDING_DIM];
				int tokens = batch_token_counts[i];
				struct inference_ring_slot *s = &g_inference_ring_buf->slots[slot_idx];

				int rlen = snprintf(result_buffer, sizeof(result_buffer),
					"{\"input\":\"%s\",\"tokens\":%d,\"embedding_sample\":[%.6f,%.6f,%.6f],\"inference_time_us\":%ld,\"batch_size\":%d}",
					batch_texts[i], tokens,
					emb[0], emb[1], emb[2],
					elapsed_us, batch_size);

				uint32_t off = s->directory[rec].offset;
				uint32_t max_len = sizeof(s->data) - off - 1;
				if ((uint32_t)rlen > max_len) rlen = max_len;
				memcpy(s->data + off, result_buffer, rlen);
				s->data[off + rlen] = '\0';
				s->directory[rec].length = rlen;

				bool is_last_record = (i + 1 >= batch_size || batch_slots[i + 1] != slot_idx);
				if (is_last_record) {
					s->len = off + rlen;
					s->t6_cpu_wrote_uvm = get_timestamp_ns();
					__atomic_store_n(&s->ready, UVM_STATUS_RESULT_READY, __ATOMIC_RELEASE);
					cpu_iq_push(&g_inference_ring_buf->response_queue, (int)slot_idx);
				}
			}
		} else {
			DOCA_LOG_WARN("[ADAPTIVE] TensorRT not loaded, fallback");
			for (int i = 0; i < batch_size; i++) {
				snprintf(result_buffer, sizeof(result_buffer),
					"{\"input\":\"%.900s\",\"status\":\"no_model\"}", batch_texts[i]);
				cpu_write_inference_result_to_gpu_ring(g_inference_ring_buf, batch_slots[i], result_buffer);
			}
		}
	}

	if (stat_batches > 0) {
		DOCA_LOG_INFO("[ADAPTIVE] === Final Stats ===");
		DOCA_LOG_INFO("[ADAPTIVE] batches=%ld reqs=%ld avg_batch=%.2f waits=%ld",
		              stat_batches, stat_requests,
		              (double)stat_requests / stat_batches, stat_waits);
		DOCA_LOG_INFO("[ADAPTIVE] c₀=%.0fµs c_m=%.0fµs λ=%.0f/s (cal=%d)",
		              c0_est, cm_est, lambda_now * 1e6, cal_count);
	}

	DOCA_LOG_INFO("[ADAPTIVE] Thread %d stopped", thread_id);
	return NULL;
}

/*
 * DOCA PE callback to be invoked if any Eth Txq get an error
 * sending packets.
 *
 * @event_error [in]: DOCA PE event error handler
 * @event_user_data [in]: custom user data set at registration time
 */
void error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
	uint16_t packet_index;

	doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
	DOCA_LOG_INFO("Error in send queue %ld, packet %d. Gracefully killing the app",
		      event_user_data.u64,
		      packet_index);
	DOCA_GPUNETIO_VOLATILE(force_quit) = true;
}

/*
 * Get timestamp in nanoseconds
 *
 * @sec [out]: seconds
 * @return: UTC timestamp
 */
static uint64_t get_ns(uint64_t *sec)
{
	struct timespec t;
	int ret;

	ret = clock_gettime(CLOCK_REALTIME, &t);
	if (ret != 0)
		exit(EXIT_FAILURE);

	(*sec) = (uint64_t)t.tv_sec;

	return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * CPU thread to print statistics from GPU filtering on the console
 *
 * @args [in]: thread input args
 */
static void stats_core(void *args)
{
	(void)args;

	doca_error_t result = DOCA_SUCCESS;
	enum doca_gpu_semaphore_status status;
	struct stats_tcp tcp_st[MAX_QUEUES] = {0};
	uint32_t sem_idx_tcp[MAX_QUEUES] = {0};
	uint64_t start_time_sec = 0;
	uint64_t interval_print = 0;
	uint64_t interval_sec = 0;
	struct stats_tcp *custom_tcp_st;

	DOCA_LOG_INFO("Core %u is reporting filter stats", rte_lcore_id());
	get_ns(&start_time_sec);
	interval_print = get_ns(&interval_sec);
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		/* Check TCP packets */
		for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
			result = doca_gpu_semaphore_get_status(tcp_queues.sem_cpu[idxq], sem_idx_tcp[idxq], &status);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("TCP semaphore error");
				DOCA_GPUNETIO_VOLATILE(force_quit) = true;
				return;
			}

			if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
				result = doca_gpu_semaphore_get_custom_info_addr(tcp_queues.sem_cpu[idxq],
										 sem_idx_tcp[idxq],
										 (void **)&(custom_tcp_st));
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore get address error");
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				tcp_st[idxq].http += custom_tcp_st->http;
				tcp_st[idxq].http_head += custom_tcp_st->http_head;
				tcp_st[idxq].http_get += custom_tcp_st->http_get;
				tcp_st[idxq].http_post += custom_tcp_st->http_post;
				tcp_st[idxq].tcp_syn += custom_tcp_st->tcp_syn;
				tcp_st[idxq].tcp_fin += custom_tcp_st->tcp_fin;
				tcp_st[idxq].tcp_ack += custom_tcp_st->tcp_ack;
				tcp_st[idxq].others += custom_tcp_st->others;
				tcp_st[idxq].total += custom_tcp_st->total;

				result = doca_gpu_semaphore_set_status(tcp_queues.sem_cpu[idxq],
								       sem_idx_tcp[idxq],
								       DOCA_GPU_SEMAPHORE_STATUS_FREE);
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore %d error", sem_idx_tcp[idxq]);
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				sem_idx_tcp[idxq] = (sem_idx_tcp[idxq] + 1) % tcp_queues.nums;
			}
		}

		if ((get_ns(&interval_sec) - interval_print) > 5000000000) {
			printf("\nSeconds %ld\n", interval_sec - start_time_sec);

			for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
				printf("[TCP] QUEUE: %d HTTP: %d GET: %d POST: %d SYN: %d FIN: %d ACK: %d OTHER: %d TOTAL: %d\n",
				       idxq,
				       tcp_st[idxq].http,
				       tcp_st[idxq].http_get,
				       tcp_st[idxq].http_post,
				       tcp_st[idxq].tcp_syn,
				       tcp_st[idxq].tcp_fin,
				       tcp_st[idxq].tcp_ack,
				       tcp_st[idxq].others,
				       tcp_st[idxq].total);
			}

			interval_print = get_ns(&interval_sec);
		}
	}
}

/*
 * Signal handler to quit application gracefully
 *
 * @signum [in]: signal received
 */
static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
	}
}

/*
 * GPU packet processing application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	doca_error_t result;
	int current_lcore = 0;
	int cuda_id;
	cudaError_t cuda_ret;
	struct doca_log_backend *sdk_log;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	DOCA_LOG_INFO("===========================================================");
	DOCA_LOG_INFO("DOCA version: %s", doca_version());
	DOCA_LOG_INFO("===========================================================");

	/* Basic DPDK initialization */
	result = doca_argp_init("gpunet", &app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = register_application_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Options enabled:\n\tGPU %s\n\tNIC %s\n\tGPU Rx queues %d\n\tGPU HTTP server enabled %s",
		      app_cfg.gpu_pcie_addr,
		      app_cfg.nic_pcie_addr,
		      app_cfg.queue_num,
		      (app_cfg.http_server == true ? "Yes" : "No"));

	/* In a multi-GPU system, ensure CUDA refers to the right GPU device */
	cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, app_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id provided %s", app_cfg.gpu_pcie_addr);
		return DOCA_ERROR_INVALID_VALUE;
	}

	if (getenv("GAZSI_GREEDY")) {
		g_fixed_batch = 1;
		DOCA_LOG_INFO("Greedy dispatch (no EWMA wait)");
	}
	const char *fb_env = getenv("GAZSI_FIXED_BATCH");
	if (fb_env) {
		g_fixed_batch = atoi(fb_env);
		DOCA_LOG_INFO("Fixed batch mode: batch=%d", g_fixed_batch);
	}
	if (getenv("GAZSI_USE_SCAN")) {
		g_use_scan = 1;
		DOCA_LOG_INFO("Using O(N) scan instead of O(1) FIFO");
	}
	int packing_enabled = getenv("GAZSI_ENABLE_PACKING") != NULL;
	if (packing_enabled)
		DOCA_LOG_INFO("Multi-record packing enabled");

	g_cuda_id = cuda_id;
	cudaFree(0);
	cudaSetDevice(cuda_id);

	result = init_doca_device(app_cfg.nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize DOCA GPU instance */
	result = doca_gpu_create(app_cfg.gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize TensorRT on GPU 0 (separate from DOCA kernels on GPU 1) */
	cudaSetDevice(0);
	DOCA_LOG_INFO("Using GPU 0 for TensorRT model");
	if (init_tensorrt_gpu_buffers(0) != 0) {
		DOCA_LOG_ERR("Failed to initialize TensorRT GPU buffers");
		return EXIT_FAILURE;
	}


	/* Use custom engine path if provided, otherwise use default */
	const char *engine_path = (app_cfg.engine_path[0] != '\0') ?
		app_cfg.engine_path :
		"/opt/mellanox/doca/applications/gpu_packet_processing_v2/models/minilm.engine";
	DOCA_LOG_INFO("Loading TensorRT engine: %s", engine_path);
	tensorrt_model = load_tensorrt_engine(engine_path);
	if (tensorrt_model) {
		DOCA_LOG_INFO("TensorRT model loaded (GPU 0)");

		/* Warmup TensorRT to eliminate cold start latency */
		DOCA_LOG_INFO("[WARMUP] Warming up TensorRT contexts...");
		float temp_embeddings[32768];
		int temp_token_count;
		#define WARMUP_ITERATIONS 10

		const char *warmup_long_text =
			"The quick brown fox jumps over the lazy dog. "
			"This is a comprehensive warmup text designed to exercise "
			"the full tokenization and inference pipeline. "
			"Machine learning models require proper initialization "
			"to achieve optimal performance. GPU memory access patterns "
			"and CUDA kernel execution paths should be exercised "
			"before serving real user requests. This ensures consistent "
			"low latency for all subsequent inference operations.";

		const char *warmup_short_text = "hello world test";

		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			DOCA_LOG_INFO("[WARMUP] Warming up context %d...", i);

			for (int j = 0; j < 5; j++) {
				simple_tokenize_and_infer_with_context(tensorrt_model, i,
					warmup_long_text, temp_embeddings, &temp_token_count);
			}

			for (int j = 0; j < 5; j++) {
				simple_tokenize_and_infer_with_context(tensorrt_model, i,
					warmup_short_text, temp_embeddings, &temp_token_count);
			}

			DOCA_LOG_INFO("[WARMUP] Context %d done (%d iterations)", i, WARMUP_ITERATIONS);
		}
		DOCA_LOG_INFO("[WARMUP] All %d contexts warmed up (%d total inferences)", NUM_INFERENCE_THREADS, NUM_INFERENCE_THREADS * WARMUP_ITERATIONS);

		if (init_cuda_graphs(tensorrt_model) == 0) {
			DOCA_LOG_INFO("CUDA Graphs initialized for batch sizes 1-8");
		} else {
			DOCA_LOG_WARN("CUDA Graph init failed, using legacy enqueueV3 path");
		}
	} else {
		DOCA_LOG_WARN("TensorRT model loading failed, inference will be skipped");
	}

	/* Switch back to DOCA GPU */
	cudaSetDevice(cuda_id);
	DOCA_LOG_INFO("Switching back to GPU %d for DOCA initialization", cuda_id);


	df_port = init_doca_flow(dpdk_dev_port_id, app_cfg.queue_num);
	if (df_port == NULL) {
		DOCA_LOG_ERR("FAILED: init_doca_flow");
		return EXIT_FAILURE;
	}

	result = doca_pe_create(&pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create pe queue: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = create_tcp_queues(&tcp_queues,
				   df_port,
				   gpu_dev,
				   ddev,
				   app_cfg.queue_num,
				   app_cfg.queue_num,  /* sem_num equals queue_num, each warp handles one semaphore */
				   app_cfg.http_server,
				   &http_queues,
				   pe,
				   &error_send_packet_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_tcp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Create root control pipe to route tcp packets */
	result = create_root_pipe(&tcp_queues, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_root_pipe returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Gracefully terminate app if ctrlc */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	cudaStream_t rx_tcp_stream, tx_http_server;
	cudaError_t res_rt = cudaSuccess;
	uint32_t *cpu_exit_condition;
	uint32_t *gpu_exit_condition;

	res_rt = cudaStreamCreateWithFlags(&rx_tcp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}

	if (app_cfg.http_server) {
		res_rt = cudaStreamCreateWithFlags(&tx_http_server, cudaStreamNonBlocking);
		if (res_rt != cudaSuccess) {
			DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
			return EXIT_FAILURE;
		}
	}

	result = doca_gpu_mem_alloc(gpu_dev,
				    sizeof(uint32_t),
				    4096,
				    DOCA_GPU_MEM_TYPE_GPU_CPU,
				    (void **)&gpu_exit_condition,
				    (void **)&cpu_exit_condition);
	if (result != DOCA_SUCCESS || gpu_exit_condition == NULL || cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	cpu_exit_condition[0] = 0;

	/*
	 * Some GPUs may require an initial warmup without doing any real operation.
	 */
	if (packing_enabled)
		set_packing_mode(1);

	DOCA_LOG_INFO("Warm up CUDA kernels");
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	cudaStreamSynchronize(rx_tcp_stream);
	if (app_cfg.http_server)
		cudaStreamSynchronize(tx_http_server);
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

	/* Initialize UVM inference buffer */
	if (app_cfg.http_server) {
		DOCA_LOG_INFO("Initializing UVM inference buffer");
		g_inference_ring_buf = init_inference_ring_buffer(cuda_id);
		if (g_inference_ring_buf == NULL) {
			DOCA_LOG_ERR("Failed to initialize UVM inference buffer");
			return EXIT_FAILURE;
		}
		tcp_queues.ring = g_inference_ring_buf;

		/* Set GPU-side UVM buffer pointer */
		result = set_inference_ring_buffer_kernel(tx_http_server, g_inference_ring_buf);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to set UVM buffer in GPU: %s", doca_error_get_descr(result));
			return EXIT_FAILURE;
		}

		/* Set CPU-side inference semaphore handle for GPU notification */
		if (tcp_queues.sem_inference_cpu) {
			set_inference_semaphore_cpu(tcp_queues.sem_inference_cpu);
			DOCA_LOG_INFO("Inference semaphore passed to CPU write function");
		}

		/* Set GPU-side inference semaphore handle (GPU waits for CPU notification) */
		if (tcp_queues.sem_inference_gpu) {
			result = set_inference_semaphore_kernel(tx_http_server, tcp_queues.sem_inference_gpu);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to set inference semaphore in GPU: %s",
				             doca_error_get_descr(result));
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("Inference semaphore GPU handle set successfully");
		}

		/* Set GPU-side request notification semaphore handle */
		if (tcp_queues.sem_request_gpu) {
			result = set_request_semaphore_kernel(tx_http_server, tcp_queues.sem_request_gpu);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to set request semaphore in GPU: %s",
				             doca_error_get_descr(result));
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("Request semaphore GPU handle set successfully");

			set_request_semaphore_cpu(tcp_queues.sem_request_cpu);
		}

		/* Start CPU-side inference reader threads */
		inference_reader_running = true;
		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			struct reader_thread_args *args = malloc(sizeof(struct reader_thread_args));
			if (!args) {
				DOCA_LOG_ERR("Failed to allocate thread args for thread %d", i);
				inference_reader_running = false;
				return EXIT_FAILURE;
			}
			args->thread_id = i;
			args->context_id = i;

			if (pthread_create(&inference_reader_threads[i], NULL, simple_inference_reader, args) != 0) {
				DOCA_LOG_ERR("Failed to create inference reader thread %d", i);
				free(args);
				inference_reader_running = false;
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("[CONTEXT_POOL] Started inference thread %d (context %d)", i, i);
		}
		DOCA_LOG_INFO("[CONTEXT_POOL] All %d inference threads started", NUM_INFERENCE_THREADS);
	}

	DOCA_LOG_INFO("Launching CUDA kernels");

	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	/* Launch stats proxy thread to report pipeline status */
	current_lcore = rte_get_next_lcore(current_lcore, true, false);
	if (rte_eal_remote_launch((void *)stats_core, NULL, current_lcore) != 0) {
		DOCA_LOG_ERR("Remote launch failed");
		goto exit;
	}

	if (app_cfg.http_server) {
		tcp_queues.tcp_ack_pkt_pool = rte_pktmbuf_pool_create("tcp_ack_pkt_pool",
								      1023,
								      0,
								      0,
								      RTE_MBUF_DEFAULT_BUF_SIZE,
								      rte_socket_id());
		if (!tcp_queues.tcp_ack_pkt_pool) {
			DOCA_LOG_ERR("%s: failed to allocate tcp-ack packet pool", __func__);
			goto exit;
		}

		/* Start the CPU RSS threads to address new TCP connections */
		tcp_queues.lcore_idx_start = rte_get_next_lcore(current_lcore, true, false);
		for (int i = 0; i < tcp_queues.numq_cpu_rss; i++) {
			current_lcore = rte_get_next_lcore(current_lcore, true, false);
			if (rte_eal_remote_launch(tcp_cpu_rss_func, &tcp_queues, current_lcore) != 0) {
				DOCA_LOG_ERR("Remote launch failed");
				goto exit;
			}
		}
	}

	DOCA_LOG_INFO("Waiting for termination");

	/* Network layer warmup - fork child process to send warmup requests
	 * Environment variables:
	 * - DOCA_WARMUP_NS: Network namespace (default: test_ns)
	 * - DOCA_WARMUP_IP: Server IP (default: 10.0.0.6)
	 * - DOCA_WARMUP_PORT: Server port (default: 8089)
	 * - DOCA_NO_WARMUP: Set to 1 to skip network warmup
	 */
	if (app_cfg.http_server) {
		const char *no_warmup = getenv("DOCA_NO_WARMUP");
		if (no_warmup == NULL || strcmp(no_warmup, "1") != 0) {
			pid_t warmup_pid = fork();
			if (warmup_pid == 0) {
				/* Child process: send warmup requests */
				const char *net_ns = getenv("DOCA_WARMUP_NS");
				const char *server_ip = getenv("DOCA_WARMUP_IP");
				const char *server_port = getenv("DOCA_WARMUP_PORT");

				if (net_ns == NULL) net_ns = "test_ns";
				if (server_ip == NULL) server_ip = "10.0.0.6";
				if (server_port == NULL) server_port = "8089";

				usleep(500000);  /* Wait for parent to be ready */

				char cmd[512];
				DOCA_LOG_INFO("[WARMUP] Starting network layer warmup (5 requests)...");

				for (int i = 0; i < 5; i++) {
					snprintf(cmd, sizeof(cmd),
						"ip netns exec %s curl -s --max-time 3 "
						"'http://%s:%s/inference?d=warmup_%d' > /dev/null 2>&1",
						net_ns, server_ip, server_port, i);
					system(cmd);
					usleep(200000);
				}

				DOCA_LOG_INFO("[WARMUP] Network layer warmup complete!");
				_exit(0);
			} else if (warmup_pid > 0) {
				/* Parent: continue main loop without waiting */
				DOCA_LOG_INFO("[WARMUP] Background network warmup started (PID: %d)", warmup_pid);
			} else {
				DOCA_LOG_WARN("[WARMUP] fork() failed, skipping network warmup");
			}
		} else {
			DOCA_LOG_INFO("[WARMUP] DOCA_NO_WARMUP=1, skipping network warmup");
		}
	}

	/* This loop keeps busy main thread until force_quit is set to 1 (e.g. typing ctrl+c) */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		doca_pe_progress(pe);
		nanosleep(&ts, &ts);
	}

	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;

	cudaStreamSynchronize(rx_tcp_stream);
	cudaStreamDestroy(rx_tcp_stream);
	if (app_cfg.http_server) {
		cudaStreamSynchronize(tx_http_server);
		cudaStreamDestroy(tx_http_server);
	}

	doca_gpu_mem_free(gpu_dev, gpu_exit_condition);

	DOCA_LOG_INFO("GPU work ended");

	current_lcore = 0;
	RTE_LCORE_FOREACH_WORKER(current_lcore)
	{
		if (rte_eal_wait_lcore(current_lcore) < 0) {
			DOCA_LOG_ERR("Bad exit for coreid: %d", current_lcore);
			break;
		}
	}

exit:

	result = destroy_flow_queue(df_port, &tcp_queues, app_cfg.http_server, &http_queues);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function finalize_doca_flow returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_gpu_destroy(gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy GPU: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_pe_destroy(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_pe_destroy returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	doca_dev_close(ddev);

	/* Stop all inference reader threads - Context Pool */
	if (inference_reader_running) {
		DOCA_LOG_INFO("[CONTEXT_POOL] Stopping all inference threads...");
		inference_reader_running = false;
		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			pthread_join(inference_reader_threads[i], NULL);
			DOCA_LOG_INFO("[CONTEXT_POOL] Thread %d stopped", i);
		}
		DOCA_LOG_INFO("[CONTEXT_POOL] All inference threads stopped");
	}

	/* Clean up inference ring buffer */
	if (g_inference_ring_buf) {
		free_inference_ring_buffer(g_inference_ring_buf);
		g_inference_ring_buf = NULL;
		DOCA_LOG_INFO("Inference ring buffer cleaned up");
	}

	if (tensorrt_model) {
		tensorrt_model = NULL;
	}

	DOCA_LOG_INFO("Application finished successfully");

	return EXIT_SUCCESS;
}
