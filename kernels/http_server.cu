/*
 * GPU Packet Processing - HTTP Server Kernel
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdlib.h>
#include <string.h>

#include <cuda/atomic>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include "doca33_compat.cuh"  /* DOCA33-COMPAT: build only */

/* Global Atomic Counter for TX Buffer Allocation
 * Used for atomic allocation of TX buffer index across all warps to avoid race conditions
 */
__device__ unsigned long long g_tx_buf_counter = 0;

#include "common.h"
#include "packets.h"
#include "filters.cuh"
#include "ring_buffer.h"  /* Ring Buffer UVM */
#include "slot_claim.h"   /* Queue-entry -> slot ownership rules */

/* HTTP Response Buffer Layout Constants */
#define HTTP_BODY_TEMP_OFFSET  256  /* Temporary offset for body construction */
#define HTTP_BODY_MAX_LEN      900  /* Max body length before truncation */

#define HTTP_RESP_PREFIX "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
#define HTTP_RESP_PREFIX_LEN 65
#define HTTP_RESP_SUFFIX "\r\nConnection: keep-alive\r\n\r\n"
#define HTTP_RESP_SUFFIX_LEN 28
#define HTTP_ERR_BODY "{\"error\":\"no result\"}"
#define HTTP_ERR_BODY_LEN 21

/* Keep the compat layer's send bound in step with the real TX buffer size */
static_assert(GAZSI_TX_SEND_MAX == TX_BUF_MAX_SZ,
	      "GAZSI_TX_SEND_MAX must match TX_BUF_MAX_SZ");

/* Warp-level parallel memcpy - each lane copies dst[lane_id], dst[lane_id+32], ... */
__device__ static inline void warp_memcpy(char *dst, const char *src, int len, int lane_id)
{
    for (int i = lane_id; i < len; i += 32) {
        dst[i] = src[i];
    }
}

/* Helper: Copy string to buffer, return bytes written */
__device__ static inline int strcpy_to_buf(char *dst, const char *src, int max_len)
{
	int len = 0;
	while (src[len] != '\0' && len < max_len)
		dst[len] = src[len], len++;
	return len;
}

/* Helper: Convert integer to string, return bytes written */
__device__ static inline int int_to_str(char *buf, int value)
{
	if (value == 0) {
		buf[0] = '0';
		return 1;
	}

	char digits[16];
	int num_digits = 0;
	while (value > 0) {
		digits[num_digits++] = '0' + (value % 10);
		value /= 10;
	}

	/* Reverse digits */
	for (int i = 0; i < num_digits; i++)
		buf[i] = digits[num_digits - 1 - i];

	return num_digits;
}

/* Global control-plane pointer (host-mapped) */
__device__ struct inference_ring_buffer *g_inference_ring_buf = nullptr;

/* Global data-plane pointer (device memory only) */
__device__ struct inference_payload_plane *g_payload_plane = nullptr;

/* Global semaphore handle for CPU -> GPU notification */
__device__ struct doca_gpu_semaphore_gpu *g_sem_inference_gpu = nullptr;

/* Global semaphore handle for GPU -> CPU notification */
__device__ struct doca_gpu_semaphore_gpu *g_sem_request_gpu = nullptr;

/* Set Ring Buffer pointer - called from CPU */
__global__ void set_inference_ring_buffer(struct inference_ring_buffer *ring)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_inference_ring_buf = ring;
        printf("[RING_INIT] Ring Buffer set: %p\n", ring);
    }
}

/* Set device payload plane pointer - called from CPU */
__global__ void set_payload_plane(struct inference_payload_plane *plane)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_payload_plane = plane;
        printf("[RING_INIT] Device payload plane set: %p\n", plane);
    }
}

/* Set inference semaphore GPU handle */
__global__ void set_inference_semaphore(struct doca_gpu_semaphore_gpu *sem)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_sem_inference_gpu = sem;
        printf("[SEM] Inference semaphore GPU handle set: %p\n", sem);
    }
}

/* Set request notification semaphore GPU handle */
__global__ void set_request_semaphore(struct doca_gpu_semaphore_gpu *sem)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_sem_request_gpu = sem;
        printf("[SEM] Request semaphore GPU handle set: %p\n", sem);
    }
}

DOCA_LOG_REGISTER(GPUNET::KernelHttpServer);

static
__device__ void http_set_mac_addr(struct eth_ip_tcp_hdr *hdr, const uint16_t *src_bytes, const uint16_t *dst_bytes)
{
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[0] = src_bytes[0];
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[1] = src_bytes[1];
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[2] = src_bytes[2];

	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[0] = dst_bytes[0];
	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[1] = dst_bytes[1];
	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[2] = dst_bytes[2];
}

__global__ void cuda_kernel_http_server(uint32_t *exit_cond,
					struct doca_gpu_eth_txq *txq0, struct doca_gpu_eth_txq *txq1, struct doca_gpu_eth_txq *txq2, struct doca_gpu_eth_txq *txq3,
					int sem_num,
					struct doca_gpu_semaphore_gpu *sem_http0, struct doca_gpu_semaphore_gpu *sem_http1, struct doca_gpu_semaphore_gpu *sem_http2, struct doca_gpu_semaphore_gpu *sem_http3,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_index, uint32_t nbytes_page_index,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_contacts, uint32_t nbytes_page_contacts,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_not_found, uint32_t nbytes_page_not_found)
{
	doca_error_t ret;
	struct doca_gpu_eth_txq *txq = NULL;
	struct doca_gpu_buf *buf = NULL;
	uintptr_t buf_addr;
	struct eth_ip_tcp_hdr *hdr;
	uint8_t *payload;
	uint32_t base_pkt_len = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr);
	uint16_t send_pkts = 0;
	uint32_t nbytes_page = 0;
	uint32_t lane_id = threadIdx.x % WARP_SIZE;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	/* Use global atomic counter for TX buffer index allocation */
	uint64_t doca_gpu_buf_idx = 0;

	if (warp_id == 0) {
		txq = txq0;
	} else if (warp_id == 1) {
		txq = txq1;
	} else if (warp_id == 2) {
		txq = txq2;
	} else if (warp_id == 3) {
		txq = txq3;
	} else {
		return;
	}

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		send_pkts = 0;
		if (txq && g_inference_ring_buf != nullptr) {
			int slot_idx = -1;
			struct inference_ring_slot *current_slot = nullptr;

			/* Q2: O(1) pop from response_queue (only lane 0 pops) */
			if (lane_id == 0)
				slot_idx = gpu_iq_pop(&g_inference_ring_buf->response_queue);
			slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);

			if (slot_idx < 0) {
				__syncwarp();
				continue;
			}

			current_slot = &g_inference_ring_buf->slots[slot_idx];

			/* Claim slot: CAS RESULT_READY → CONSUMED (safety check) */
			{
				uint32_t claimed = 0;
				if (lane_id == 0)
					claimed = gazsi_tx_claim_slot(g_inference_ring_buf, slot_idx) ? 1u : 0u;
				claimed = __shfl_sync(0xffffffff, claimed, 0);

				if (!claimed) {
					/*
					 * Stale response_queue entry: the slot is owned
					 * by someone else now. Drop the entry and take
					 * the next one. Pushing it to free_pool would
					 * publish a slot this warp does not own, and the
					 * real owner will publish it too — the same index
					 * would then be handed to two requests.
					 */
					__syncwarp();
					continue;
				}

				if (g_sem_inference_gpu != nullptr) {
					doca_gpu_dev_semaphore_set_status(g_sem_inference_gpu, slot_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
				}

				/* Atomic TX buffer index allocation */
				if (lane_id == 0)
					doca_gpu_buf_idx = atomicAdd(&g_tx_buf_counter, 1ULL) % TX_BUF_NUM;
				doca_gpu_buf_idx = __shfl_sync(0xffffffff, doca_gpu_buf_idx, 0);

				enum http_page_get page_type = (enum http_page_get)current_slot->http_page_type;

				/*
				 * Any buffer or send failure must be propagated, not swallowed:
				 * the slot must NOT be recycled after a failed transmission, or it
				 * could be reused while a send referring to it is still outstanding.
				 */
				bool tx_failed = false;

				if (page_type == HTTP_GET_INDEX) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_index, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_index;
				} else if (page_type == HTTP_GET_CONTACTS) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_contacts, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_contacts;
				} else if (page_type == HTTP_GET_INFERENCE) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_index, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_index; /* Will be updated with JSON response */
				} else {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_not_found, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_not_found;
				}

				ret = doca_gpu_dev_buf_get_addr(buf, &buf_addr);
				if (ret != DOCA_SUCCESS) {
					printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
					DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
					break;
				}

				if (page_type == HTTP_GET_INFERENCE) {
					uint32_t nrec = current_slot->record_count;
					if (nrec == 0) nrec = 1;

					/*
					 * Response bytes live in device memory. resp_length is
					 * written by the GPU format kernel for every record of a
					 * claimed slot, so no fallback to slot->len is needed.
					 */
					struct inference_payload_slot *pay =
						(g_payload_plane != nullptr) ? &g_payload_plane->slots[slot_idx] : nullptr;

					for (uint32_t rec = 0; rec < nrec; rec++) {
						uint32_t rec_len = (pay != nullptr)
							? current_slot->directory[rec].resp_length : 0;

						if (rec > 0) {
							if (lane_id == 0)
								doca_gpu_buf_idx = atomicAdd(&g_tx_buf_counter, 1ULL) % TX_BUF_NUM;
							doca_gpu_buf_idx = __shfl_sync(0xffffffff, doca_gpu_buf_idx, 0);
							ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_index, doca_gpu_buf_idx, &buf);
							if (ret != DOCA_SUCCESS) { tx_failed = true; break; }
							ret = doca_gpu_dev_buf_get_addr(buf, &buf_addr);
							if (ret != DOCA_SUCCESS) { tx_failed = true; break; }
						}

						char *response_buf = (char *)(buf_addr + base_pkt_len);
						int body_len = 0;
						int header_len = 0;
						int have_body = 0;

						if (lane_id == 0) {
							if (rec == 0) current_slot->t8_gpu_sent = clock64();
							header_len = HTTP_RESP_PREFIX_LEN;
							body_len = (int)rec_len;
							have_body = (body_len > 0 && body_len <= HTTP_BODY_MAX_LEN);
							if (!have_body)
								body_len = HTTP_ERR_BODY_LEN;
							header_len += int_to_str(response_buf + header_len, body_len);
							nbytes_page = header_len + HTTP_RESP_SUFFIX_LEN + body_len;
						}

						header_len = __shfl_sync(0xffffffff, header_len, 0);
						body_len = __shfl_sync(0xffffffff, body_len, 0);
						have_body = __shfl_sync(0xffffffff, have_body, 0);
						nbytes_page = __shfl_sync(0xffffffff, nbytes_page, 0);

						warp_memcpy(response_buf, HTTP_RESP_PREFIX, HTTP_RESP_PREFIX_LEN, lane_id);
						__syncwarp();
						warp_memcpy(response_buf + header_len, HTTP_RESP_SUFFIX, HTTP_RESP_SUFFIX_LEN, lane_id);
						int bo = header_len + HTTP_RESP_SUFFIX_LEN;
						if (have_body) {
							/* Device-to-device: response arena -> TX packet buffer */
							warp_memcpy(response_buf + bo,
								    pay->response + rec * GAZSI_RESPONSE_STRIDE,
								    body_len, lane_id);
						} else {
							/*
							 * Write the error body last. The previous ordering
							 * emitted it before the body copy, which then
							 * overwrote it with unformatted bytes.
							 */
							warp_memcpy(response_buf + bo, HTTP_ERR_BODY, HTTP_ERR_BODY_LEN, lane_id);
						}

						raw_to_tcp(buf_addr, &hdr, &payload);
						http_set_mac_addr(hdr, (uint16_t *)current_slot->eth_src_addr_bytes, (uint16_t *)current_slot->eth_dst_addr_bytes);
						hdr->l3_hdr.src_addr = current_slot->ip_src_addr;
						hdr->l3_hdr.dst_addr = current_slot->ip_dst_addr;
						hdr->l4_hdr.src_port = current_slot->tcp_src_port;
						hdr->l4_hdr.dst_port = current_slot->tcp_dst_port;
						uint32_t my_seq_n = 0, my_ack_n = 0;
						if (lane_id == 0) {
							uint32_t conn_h = current_slot->ip_src_addr ^ current_slot->ip_dst_addr;
							conn_h ^= (uint32_t)current_slot->tcp_src_port ^ (uint32_t)current_slot->tcp_dst_port;
							conn_h ^= (conn_h >> 16);
							uint32_t ct_idx = conn_h & (GPU_CONN_TABLE_SIZE - 1);
							struct gpu_conn_state *cs = &g_inference_ring_buf->conn_table[ct_idx];

							if (cs->active && cs->conn_hash == conn_h) {
								uint32_t tcp_payload_size = nbytes_page;
								uint32_t cur_seq = cs->sent_seq;
								uint32_t last_ack = cs->recv_ack;
								uint32_t cwnd = cs->cwnd;
								uint32_t bytes_in_flight = cur_seq - last_ack;
								if (bytes_in_flight > 0x80000000u) bytes_in_flight = 0;

								if (bytes_in_flight + tcp_payload_size <= cwnd) {
									cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
										seq_ref(*(uint32_t *)&cs->sent_seq);
									uint32_t my_seq = seq_ref.fetch_add(tcp_payload_size, cuda::memory_order_relaxed);
									my_seq_n = BYTE_SWAP32(my_seq);
									my_ack_n = BYTE_SWAP32(last_ack);
								} else {
									my_seq_n = current_slot->tcp_sent_seq;
									my_ack_n = current_slot->tcp_recv_ack;
								}
							} else {
								my_seq_n = current_slot->tcp_sent_seq;
								my_ack_n = current_slot->tcp_recv_ack;
							}
						}
					hdr->l4_hdr.sent_seq = __shfl_sync(0xffffffff, my_seq_n, 0);
						hdr->l4_hdr.recv_ack = __shfl_sync(0xffffffff, my_ack_n, 0);

						hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK | TCP_FLAG_PSH;
						hdr->l4_hdr.cksum = 0;
						uint32_t free_estimate = g_inference_ring_buf->free_pool.tail - g_inference_ring_buf->free_pool.head;
						if (free_estimate > INFERENCE_RING_SIZE) free_estimate = 0;
						uint16_t rx_win = (uint16_t)(free_estimate * 1460);
						if (rx_win < 1460) rx_win = 1460;
						hdr->l4_hdr.rx_win = BYTE_SWAP16(rx_win);
						hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + nbytes_page);
						hdr->l3_hdr.hdr_checksum = 0;

						uint32_t send_bad = 0;
						if (lane_id == 0) {
							ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, base_pkt_len + nbytes_page, 0);
							if (ret != DOCA_SUCCESS) {
								printf("Error %d send_enqueue (inference) warp %d\n", ret, warp_id);
								send_bad = 1;
							} else {
								send_pkts++;
							}
						}
						send_bad = __shfl_sync(0xffffffff, send_bad, 0);
						if (send_bad) { tx_failed = true; break; }
						__syncwarp();

						if (rec + 1 < nrec) {
							if (lane_id == 0) {
								doca_gpu_dev_eth_txq_commit_strong(txq);
								doca_gpu_dev_eth_txq_push(txq);
							}
							__syncwarp();
						}
					}
				} else {
				raw_to_tcp(buf_addr, &hdr, &payload);
				http_set_mac_addr(hdr, (uint16_t *)current_slot->eth_src_addr_bytes, (uint16_t *)current_slot->eth_dst_addr_bytes);
				hdr->l3_hdr.src_addr = current_slot->ip_src_addr;
				hdr->l3_hdr.dst_addr = current_slot->ip_dst_addr;
				hdr->l4_hdr.src_port = current_slot->tcp_src_port;
				hdr->l4_hdr.dst_port = current_slot->tcp_dst_port;
				hdr->l4_hdr.sent_seq = current_slot->tcp_sent_seq;
				hdr->l4_hdr.recv_ack = current_slot->tcp_recv_ack;
				
				hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK | TCP_FLAG_PSH;
				hdr->l4_hdr.cksum = 0;
				hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + nbytes_page);
				hdr->l3_hdr.hdr_checksum = 0;

				uint32_t send_bad2 = 0;
				if (lane_id == 0) {
					ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, base_pkt_len + nbytes_page, 0);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						send_bad2 = 1;
					} else {
						send_pkts++;
					}
				}
				send_bad2 = __shfl_sync(0xffffffff, send_bad2, 0);
				if (send_bad2)
					tx_failed = true;
				__syncwarp();
				}

				/*
				 * Recycle only on success. After a failed transmission the slot
				 * stays CONSUMED and is deliberately NOT returned to the free
				 * pool, and the kernel asks the app to exit: reuse could race a
				 * send that may still be in flight. Losing a bounded number of
				 * slots beats corrupting one.
				 */
				if (tx_failed) {
					if (lane_id == 0) {
						printf("TX failed slot %d: not recycling, requesting exit\n", slot_idx);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
					}
					__syncwarp();
				} else {
					{
						cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
							ready_ref(*(uint32_t*)&current_slot->ready);
						ready_ref.store(UVM_STATUS_FREE, cuda::memory_order_release);
					}
					if (lane_id == 0)
						gpu_iq_push(&g_inference_ring_buf->free_pool, slot_idx);
				}
			}
		}
		__syncwarp();

		/* Send only if needed */
#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2)
			send_pkts += __shfl_down_sync(WARP_FULL_MASK, send_pkts, offset);
		__syncwarp();

		if (lane_id == 0 && send_pkts > 0) {
			doca_gpu_dev_eth_txq_commit_strong(txq);
			doca_gpu_dev_eth_txq_push(txq);
		}
		__syncwarp();
	}
}

extern "C" {

doca_error_t kernel_http_server(cudaStream_t stream, uint32_t *exit_cond, struct rxq_tcp_queues *tcp_queues, struct txq_http_queues *http_queues)
{
	cudaError_t result = cudaSuccess;

	if (tcp_queues == NULL || tcp_queues->numq == 0 || exit_cond == 0) {
		DOCA_LOG_ERR("kernel_http_server invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/*
	 * Assume no more than MAX_QUEUE (4) receive queues
	 */
	cuda_kernel_http_server<<<1, tcp_queues->numq * WARP_SIZE, 0, stream>>>(exit_cond,
								http_queues->eth_txq_gpu[0], http_queues->eth_txq_gpu[1], http_queues->eth_txq_gpu[2], http_queues->eth_txq_gpu[3],
								tcp_queues->nums,
								tcp_queues->sem_http_gpu[0], tcp_queues->sem_http_gpu[1], tcp_queues->sem_http_gpu[2], tcp_queues->sem_http_gpu[3],
								http_queues->buf_page_index.buf_arr_gpu, http_queues->buf_page_index.pkt_nbytes,
								http_queues->buf_page_contacts.buf_arr_gpu, http_queues->buf_page_contacts.pkt_nbytes,
								http_queues->buf_page_not_found.buf_arr_gpu, http_queues->buf_page_not_found.pkt_nbytes
								);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/* Set UVM buffer for inference data storage (CPU-side) */
doca_error_t set_inference_ring_buffer_kernel(cudaStream_t stream, struct inference_ring_buffer *ring)
{
	cudaError_t result = cudaSuccess;

	if (ring == NULL) {
		DOCA_LOG_ERR("set_inference_ring_buffer_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_inference_ring_buffer<<<1, 1, 0, stream>>>(ring);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/* Publish the device payload plane pointer to the GPU kernels */
doca_error_t set_payload_plane_kernel(cudaStream_t stream, struct inference_payload_plane *plane)
{
	cudaError_t result = cudaSuccess;

	if (plane == NULL) {
		DOCA_LOG_ERR("set_payload_plane_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	set_payload_plane<<<1, 1, 0, stream>>>(plane);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/* Set inference semaphore GPU handle */
doca_error_t set_inference_semaphore_kernel(cudaStream_t stream, struct doca_gpu_semaphore_gpu *sem)
{
	cudaError_t result = cudaSuccess;

	if (sem == NULL) {
		DOCA_LOG_ERR("set_inference_semaphore_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_inference_semaphore<<<1, 1, 0, stream>>>(sem);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Inference semaphore GPU handle set successfully");
	return DOCA_SUCCESS;
}

/* Set request notification semaphore GPU handle */
doca_error_t set_request_semaphore_kernel(cudaStream_t stream, struct doca_gpu_semaphore_gpu *sem)
{
	cudaError_t result = cudaSuccess;

	if (sem == NULL) {
		DOCA_LOG_ERR("set_request_semaphore_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_request_semaphore<<<1, 1, 0, stream>>>(sem);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Request semaphore GPU handle set successfully");
	return DOCA_SUCCESS;
}

} /* extern C */
