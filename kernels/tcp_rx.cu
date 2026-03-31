/*
 * GPU Packet Processing - TCP Receiver Kernel
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdlib.h>
#include <string.h>

#include <cuda/atomic>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>

#include "common.h"
#include "packets.h"
#include "filters.cuh"
#include "ring_buffer.h"

DOCA_LOG_REGISTER(GPUNET::KernelReceiveTcp);

/* Global ring buffer and semaphore pointers (extern from http_server.cu) */
extern __device__ struct inference_ring_buffer *g_inference_ring_buf;
extern __device__ struct doca_gpu_semaphore_gpu *g_sem_request_gpu;

static
__device__ enum http_page_get get_http_page_type(const uint8_t *payload)
{
	/* inference (hot path first) */
	if (payload[5] == 'i' && payload[6] == 'n' && payload[7] == 'f' && payload[8] == 'e' &&
	    payload[9] == 'r' && payload[10] == 'e' && payload[11] == 'n' && payload[12] == 'c' &&
	    payload[13] == 'e' && (payload[14] == ' ' || payload[14] == '?'))
		return HTTP_GET_INFERENCE;
	/* index */
	if (payload[5] == 'i' && payload[6] == 'n' && payload[7] == 'd' && payload[8] == 'e' &&
	    payload[9] == 'x' && payload[10] == '.')
		return HTTP_GET_INDEX;
	/* contacts */
	if (payload[5] == 'c' && payload[6] == 'o' && payload[7] == 'n' && payload[8] == 't' &&
	    payload[9] == 'a' && payload[10] == 'c' && payload[11] == 't' && payload[12] == 's' &&
	    payload[13] == '.')
		return HTTP_GET_CONTACTS;
	return HTTP_GET_NOT_FOUND;
}


__global__ void cuda_kernel_receive_tcp(uint32_t *exit_cond,
					struct doca_gpu_eth_rxq *rxq0, struct doca_gpu_eth_rxq *rxq1, struct doca_gpu_eth_rxq *rxq2, struct doca_gpu_eth_rxq *rxq3,
					int sem_num,
					struct doca_gpu_semaphore_gpu *sem_stats0, struct doca_gpu_semaphore_gpu *sem_stats1, struct doca_gpu_semaphore_gpu *sem_stats2, struct doca_gpu_semaphore_gpu *sem_stats3,
					struct doca_gpu_semaphore_gpu *sem_http0, struct doca_gpu_semaphore_gpu *sem_http1, struct doca_gpu_semaphore_gpu *sem_http2, struct doca_gpu_semaphore_gpu *sem_http3,
					bool http_server)
{
	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;
	__shared__ struct stats_tcp stats_sh;

	doca_error_t ret;
	struct doca_gpu_eth_rxq *rxq = NULL;
	struct doca_gpu_semaphore_gpu *sem_stats = NULL;
	struct doca_gpu_buf *buf_ptr;
	struct stats_tcp stats_thread;
	struct stats_tcp *stats_global;
	struct eth_ip_tcp_hdr *hdr;
	uintptr_t buf_addr;
	uint64_t buf_idx = 0;
	uint32_t laneId = threadIdx.x % WARP_SIZE;
	uint32_t sem_stats_idx = 0;
	uint8_t *payload;
	uint32_t max_pkts;
	uint64_t timeout_ns;

	max_pkts = MAX_RX_NUM_PKTS_HTTP;
	timeout_ns = MAX_RX_TIMEOUT_NS_HTTP;

	if (blockIdx.x == 0) {
		rxq = rxq0;
		sem_stats = sem_stats0;
	} else if (blockIdx.x == 1) {
		rxq = rxq1;
		sem_stats = sem_stats1;
	} else if (blockIdx.x == 2) {
		rxq = rxq2;
		sem_stats = sem_stats2;
	} else if (blockIdx.x == 3) {
		rxq = rxq3;
		sem_stats = sem_stats3;
	} else {
		return;
	}

	if (threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(stats_sh.http) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.http_head) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.http_get) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.http_post) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_syn) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_fin) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_ack) = 0;
		DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
	}
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		stats_thread.http = 0;
		stats_thread.http_head = 0;
		stats_thread.http_get = 0;
		stats_thread.http_post = 0;
		stats_thread.tcp_syn = 0;
		stats_thread.tcp_fin = 0;
		stats_thread.tcp_ack = 0;
		stats_thread.others = 0;

		ret = doca_gpu_dev_eth_rxq_receive_block(rxq, max_pkts, timeout_ns, &rx_pkt_num, &rx_buf_idx);
		/* If any thread returns receive error, the whole execution stops */
		if (ret != DOCA_SUCCESS) {
			if (threadIdx.x == 0) {
				/*
				 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
				 * If application prints this message on the console, something bad happened and
				 * applications needs to exit
				 */
				printf("Receive TCP kernel error %d Block %d rxpkts %d error %d\n", ret, blockIdx.x, rx_pkt_num, ret);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
			}
			break;
		}

		if (rx_pkt_num == 0)
			continue;

		buf_idx = threadIdx.x;
		while (buf_idx < rx_pkt_num) {
			ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + buf_idx, &buf_ptr);
			if (ret != DOCA_SUCCESS) {
				printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
			if (ret != DOCA_SUCCESS) {
				printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			raw_to_tcp(buf_addr, &hdr, &payload);

			/* Priority to GET for the HTTP server mode */
			bool is_http_get = filter_is_http_get(payload);
			if (is_http_get) {
				if (http_server && g_inference_ring_buf != nullptr) {
					uint64_t request_id = 0;
					int slot_idx = gpu_alloc_ring_slot(g_inference_ring_buf, &request_id);

					if (slot_idx >= 0) {
						struct inference_ring_slot *slot = &g_inference_ring_buf->slots[slot_idx];

							/* T0: HTTP request received */
							slot->t0_gpu_received = clock64();

							/* Pre-swap TCP info: store as response-ready (src↔dst swapped) */
							((uint16_t *)slot->eth_src_addr_bytes)[0] = ((uint16_t *)hdr->l2_hdr.d_addr_bytes)[0];
							((uint16_t *)slot->eth_src_addr_bytes)[1] = ((uint16_t *)hdr->l2_hdr.d_addr_bytes)[1];
							((uint16_t *)slot->eth_src_addr_bytes)[2] = ((uint16_t *)hdr->l2_hdr.d_addr_bytes)[2];
							((uint16_t *)slot->eth_dst_addr_bytes)[0] = ((uint16_t *)hdr->l2_hdr.s_addr_bytes)[0];
							((uint16_t *)slot->eth_dst_addr_bytes)[1] = ((uint16_t *)hdr->l2_hdr.s_addr_bytes)[1];
							((uint16_t *)slot->eth_dst_addr_bytes)[2] = ((uint16_t *)hdr->l2_hdr.s_addr_bytes)[2];
							slot->ip_src_addr = hdr->l3_hdr.dst_addr;
							slot->ip_dst_addr = hdr->l3_hdr.src_addr;
							slot->ip_total_length = hdr->l3_hdr.total_length;
							slot->tcp_src_port = hdr->l4_hdr.dst_port;
							slot->tcp_dst_port = hdr->l4_hdr.src_port;
							slot->tcp_dt_off = hdr->l4_hdr.dt_off;

							uint32_t client_data_len = BYTE_SWAP16(hdr->l3_hdr.total_length)
							                         - sizeof(struct ipv4_hdr) - ((hdr->l4_hdr.dt_off >> 4) * 4);
							slot->tcp_sent_seq = hdr->l4_hdr.recv_ack;
							slot->tcp_recv_ack = BYTE_SWAP32(BYTE_SWAP32(hdr->l4_hdr.sent_seq) + client_data_len);

							slot->http_page_type = (uint8_t)get_http_page_type(payload);

							/* Calculate TCP payload length */
							uint16_t ip_total_len = BYTE_SWAP16(hdr->l3_hdr.total_length);
							uint32_t tcp_header_len = (hdr->l4_hdr.dt_off >> 4) * 4;
							uint32_t payload_len = ip_total_len - sizeof(struct ipv4_hdr) - tcp_header_len;

							slot->len = 0;
						slot->record_count = 0;
						slot->data[0] = '\0';

						/* Fast path: /inference?d= has "?d=" at known offset */
							int param_start = -1;
							if (payload_len > 18 &&
							    payload[15] == '?' && payload[16] == 'd' && payload[17] == '=') {
								param_start = 18;
							} else {
								for (int i = 5; i < (int)payload_len && i < 1024; i++) {
									if (payload[i] == '?' && payload[i+1] == 'd' && payload[i+2] == '=') {
										param_start = i + 3;
										break;
									}
								}
							}

							if (param_start >= 0) {
								int param_len = 0;
								int max_search = payload_len < (uint32_t)(param_start + 511) ? payload_len : (param_start + 511);
								for (int j = param_start; j < max_search; j++) {
									uint8_t c = payload[j];
									if (c == ' ' || c == '&' || c == '\r' || c == '\n')
										break;
									if (c == '+') {
										slot->data[param_len++] = ' ';
									} else if (c == '%' && j + 2 < max_search) {
										uint8_t hi = payload[j+1], lo = payload[j+2];
										uint8_t h = (hi >= '0' && hi <= '9') ? hi - '0' :
										            (hi >= 'a' && hi <= 'f') ? hi - 'a' + 10 :
										            (hi >= 'A' && hi <= 'F') ? hi - 'A' + 10 : 0;
										uint8_t l = (lo >= '0' && lo <= '9') ? lo - '0' :
										            (lo >= 'a' && lo <= 'f') ? lo - 'a' + 10 :
										            (lo >= 'A' && lo <= 'F') ? lo - 'A' + 10 : 0;
										slot->data[param_len++] = (char)((h << 4) | l);
										j += 2;
									} else {
										slot->data[param_len++] = (char)c;
									}
								}
								slot->data[param_len] = '\0';
								slot->len = param_len;
							}

				/* Only set PARAM_READY if valid parameter was extracted */
			if (slot->len > 0) {
					slot->record_count = 1;
					slot->directory[0].offset = 0;
					slot->directory[0].length = slot->len;
					cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
						pending_ref(*(uint32_t*)&g_inference_ring_buf->pending_count);
					cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
						ready_ref(*(uint32_t*)&slot->ready);

					pending_ref.fetch_add(1, cuda::memory_order_release);
					ready_ref.store(UVM_STATUS_PARAM_READY, cuda::memory_order_release);

					/* Q2: push to request_queue for O(1) CPU pickup */
					gpu_iq_push(&g_inference_ring_buf->request_queue, slot_idx);
					} else {
						cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
							ready_ref(*(uint32_t*)&slot->ready);
						ready_ref.store(UVM_STATUS_FREE, cuda::memory_order_release);
						/* Return unused slot to free_pool */
						gpu_iq_push(&g_inference_ring_buf->free_pool, slot_idx);
					}
						}
				}

				stats_thread.http_get++;
			}
			else if (filter_is_http_head(payload))
				stats_thread.http_head++;
			else if (filter_is_http_post(payload))
				stats_thread.http_post++;
			else if (filter_is_http(payload))
				stats_thread.http++;
			else if(filter_is_tcp_fin(&(hdr->l4_hdr)))
				stats_thread.tcp_fin++;
			else if(filter_is_tcp_syn(&(hdr->l4_hdr)))
				stats_thread.tcp_syn++;
			else if(filter_is_tcp_ack(&(hdr->l4_hdr))) {
				stats_thread.tcp_ack++;
			}
			else
				stats_thread.others++;

			/* TCP header may vary so a newer smaller packet may not overvwrite old longer packet */
			wipe_packet_32b(payload);
			buf_idx += blockDim.x;
		}

#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2) {
			stats_thread.http += __shfl_down_sync(WARP_FULL_MASK, stats_thread.http, offset);
			stats_thread.http_head += __shfl_down_sync(WARP_FULL_MASK, stats_thread.http_head, offset);
			stats_thread.http_get += __shfl_down_sync(WARP_FULL_MASK, stats_thread.http_get, offset);
			stats_thread.http_post += __shfl_down_sync(WARP_FULL_MASK, stats_thread.http_post, offset);
			stats_thread.tcp_syn += __shfl_down_sync(WARP_FULL_MASK, stats_thread.tcp_syn, offset);
			stats_thread.tcp_fin += __shfl_down_sync(WARP_FULL_MASK, stats_thread.tcp_fin, offset);
			stats_thread.tcp_ack += __shfl_down_sync(WARP_FULL_MASK, stats_thread.tcp_ack, offset);
			stats_thread.others += __shfl_down_sync(WARP_FULL_MASK, stats_thread.others, offset);

			__syncwarp();
		}

		if (laneId == 0) {
			atomicAdd(&(stats_sh.http), stats_thread.http);
			atomicAdd(&(stats_sh.http_head), stats_thread.http_head);
			atomicAdd(&(stats_sh.http_get), stats_thread.http_get);
			atomicAdd(&(stats_sh.http_post), stats_thread.http_post);
			atomicAdd(&(stats_sh.tcp_syn), stats_thread.tcp_syn);
			atomicAdd(&(stats_sh.tcp_fin), stats_thread.tcp_fin);
			atomicAdd(&(stats_sh.tcp_ack), stats_thread.tcp_ack);
			atomicAdd(&(stats_sh.others), stats_thread.others);
		}
		__syncthreads();

		if (threadIdx.x == 0 && rx_pkt_num > 0) {
			ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_stats, sem_stats_idx, (void **)&stats_global);
			if (ret != DOCA_SUCCESS) {
				printf("TCP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			DOCA_GPUNETIO_VOLATILE(stats_global->http) = DOCA_GPUNETIO_VOLATILE(stats_sh.http);
			DOCA_GPUNETIO_VOLATILE(stats_global->http_head) = DOCA_GPUNETIO_VOLATILE(stats_sh.http_head);
			DOCA_GPUNETIO_VOLATILE(stats_global->http_get) = DOCA_GPUNETIO_VOLATILE(stats_sh.http_get);
			DOCA_GPUNETIO_VOLATILE(stats_global->http_post) = DOCA_GPUNETIO_VOLATILE(stats_sh.http_post);
			DOCA_GPUNETIO_VOLATILE(stats_global->tcp_syn) = DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_syn);
			DOCA_GPUNETIO_VOLATILE(stats_global->tcp_fin) = DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_fin);
			DOCA_GPUNETIO_VOLATILE(stats_global->tcp_ack) = DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_ack);
			DOCA_GPUNETIO_VOLATILE(stats_global->others) = DOCA_GPUNETIO_VOLATILE(stats_sh.others);
			DOCA_GPUNETIO_VOLATILE(stats_global->total) = rx_pkt_num;

			doca_gpu_dev_semaphore_set_status(sem_stats, sem_stats_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
			__threadfence_system();

			sem_stats_idx = (sem_stats_idx + 1) % sem_num;

			DOCA_GPUNETIO_VOLATILE(stats_sh.http) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.http_head) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.http_get) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.http_post) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_syn) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_fin) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.tcp_ack) = 0;
			DOCA_GPUNETIO_VOLATILE(stats_sh.others) = 0;
		}

		__syncthreads();
	}
}

extern "C" {

doca_error_t kernel_receive_tcp(cudaStream_t stream, uint32_t *exit_cond, struct rxq_tcp_queues *tcp_queues, bool http_server)
{
	cudaError_t result = cudaSuccess;

	if (exit_cond == 0 || tcp_queues == NULL || tcp_queues->numq == 0) {
		DOCA_LOG_ERR("kernel_receive_tcp invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Assume MAX_QUEUES == 4 */
	cuda_kernel_receive_tcp<<<tcp_queues->numq, CUDA_THREADS, 0, stream>>>(exit_cond,
										tcp_queues->eth_rxq_gpu[0], tcp_queues->eth_rxq_gpu[1], tcp_queues->eth_rxq_gpu[2], tcp_queues->eth_rxq_gpu[3],
										tcp_queues->nums,
										tcp_queues->sem_gpu[0], tcp_queues->sem_gpu[1], tcp_queues->sem_gpu[2], tcp_queues->sem_gpu[3],
										tcp_queues->sem_http_gpu[0], tcp_queues->sem_http_gpu[1], tcp_queues->sem_http_gpu[2], tcp_queues->sem_http_gpu[3],
										http_server);

	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

} /* extern C */
