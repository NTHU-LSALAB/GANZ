/*
 * DOCA 3.3 Build Compatibility — GPUNetIO device side
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SCOPE: build compatibility ONLY. Nothing here is part of the paper-alignment
 * work. It exists so the RX and TX kernels can be compiled against the DOCA
 * 3.3.x SDK installed on this machine while the code targets the DOCA 2.x
 * GPUNetIO device API.
 *
 * Applied IDENTICALLY to the baseline tree and the modified tree, so it cannot
 * bias an A/B comparison between them.
 *
 *   DOCA 2.x                                DOCA 3.3
 *   -------------------------------------   ---------------------------------
 *   rxq_receive_block(rxq,n,t,&num,&idx)    rxq_recv_block(rxq,n,t,&idx,&num,attr)
 *                                           (note: idx and num swapped)
 *   rxq_get_buf + buf_get_addr              rxq_get_pkt_addr (returns address)
 *   buf_get_buf  -> doca_error_t            -> void
 *   buf_get_addr -> doca_error_t            -> void
 *   txq_send_enqueue_strong                 txq_send_thread (WQE based)
 *   txq_commit_strong + txq_push            folded into txq_send_thread's submit
 *
 * BEHAVIOURAL NOTE, stated plainly: DOCA 2.x separated enqueue from the
 * doorbell, so N packets could be enqueued and rung once. In 3.3,
 * txq_send_thread reserves, prepares and submits in one call, so the doorbell
 * is rung per packet and commit/push become no-ops. That is a real difference
 * from the 2.x TX batching the paper measured, and it will shift ABSOLUTE
 * throughput numbers. It applies to both sides of the A/B equally, so relative
 * comparisons remain sound — but absolute figures from this build must not be
 * copied into the manuscript.
 *
 * txq_send_warp is not used: it requires __CUDA_ARCH__ >= 800 and the local
 * V100 is sm_70. The original enqueued from lane 0 only, so the per-thread
 * variant is the faithful mapping anyway.
 */

#ifndef GAZSI_DOCA33_COMPAT_CUH
#define GAZSI_DOCA33_COMPAT_CUH

/*
 * Real SDK headers first, unmodified. Their include guards then stop the
 * function-like macros below from rewriting the SDK's own declarations.
 */
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_error.h>

/* ---------------- RX ---------------- */

/*
 * Argument order changed: 2.x returned (pkt_num, first_pkt_idx), 3.3 returns
 * (first_pkt_idx, pkt_num) and takes a per-packet attribute array we do not use.
 */
#define doca_gpu_dev_eth_rxq_receive_block(rxq, max_pkts, timeout_ns, pkt_num, first_idx) \
	doca_gpu_dev_eth_rxq_recv_block((rxq), (max_pkts), (timeout_ns), (first_idx), (pkt_num), nullptr)

/*
 * 3.3 hands back a packet address directly instead of an opaque buf handle.
 * Call sites that used the get_buf + buf_get_addr pair use this instead; each is
 * marked with a DOCA33-COMPAT comment.
 */
#define GAZSI_RXQ_PKT_ADDR(rxq, pkt_idx) \
	((uintptr_t)doca_gpu_dev_eth_rxq_get_pkt_addr((rxq), (pkt_idx)))

/* ---------------- buffers ---------------- */

__device__ static inline doca_error_t gazsi_buf_get_buf(const struct doca_gpu_buf_arr *arr,
							uint64_t idx,
							struct doca_gpu_buf **buf)
{
	doca_gpu_dev_buf_get_buf(arr, idx, buf);
	return DOCA_SUCCESS;
}

__device__ static inline doca_error_t gazsi_buf_get_addr(const struct doca_gpu_buf *buf,
							 uintptr_t *addr)
{
	doca_gpu_dev_buf_get_addr(buf, addr);
	return DOCA_SUCCESS;
}

#define doca_gpu_dev_buf_get_buf(arr, idx, buf) gazsi_buf_get_buf((arr), (idx), (buf))
#define doca_gpu_dev_buf_get_addr(buf, addr) gazsi_buf_get_addr((buf), (addr))

/* ---------------- TX ---------------- */

/* Must match TX_BUF_MAX_SZ in doca/defines.h (included later at both sites) */
#define GAZSI_TX_SEND_MAX 2048

/*
 * txq_send_thread returns void in 3.3, so there is no post-submit status to
 * forward. What CAN be validated is everything the send depends on, and those
 * failures are propagated rather than reported as success: a caller that sees
 * DOCA_SUCCESS must be able to trust that a descriptor was actually posted.
 */
__device__ static inline doca_error_t gazsi_txq_send(struct doca_gpu_eth_txq *txq,
						     const struct doca_gpu_buf *buf,
						     uint32_t nbytes)
{
	uintptr_t addr = 0;
	uint32_t mkey = 0;
	doca_gpu_dev_eth_ticket_t ticket = 0;

	if (txq == nullptr || buf == nullptr || nbytes == 0)
		return DOCA_ERROR_INVALID_VALUE;

	doca_gpu_dev_buf_get_addr(buf, &addr);
	doca_gpu_dev_buf_get_mkey(buf, &mkey);

	/*
	 * A zero address or mkey means the buffer handle did not resolve. Posting
	 * it would hand the NIC an invalid descriptor.
	 */
	if (addr == 0 || mkey == 0)
		return DOCA_ERROR_BAD_STATE;

	/*
	 * Upper bound on a single Ethernet send from a TX buffer. Kept local
	 * rather than using TX_BUF_MAX_SZ: this header is included before
	 * defines.h at both call sites.
	 */
	if (nbytes > GAZSI_TX_SEND_MAX)
		return DOCA_ERROR_TOO_BIG;

	doca_gpu_dev_eth_txq_send_thread(txq,
					 (uint64_t)addr,
					 mkey,
					 (size_t)nbytes,
					 DOCA_GPUNETIO_ETH_SEND_FLAG_NONE,
					 &ticket);
	return DOCA_SUCCESS;
}

/* The submit inside send_thread already rings the doorbell */
__device__ static inline void gazsi_txq_noop(struct doca_gpu_eth_txq *txq) { (void)txq; }

#define doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, nbytes, flags) \
	gazsi_txq_send((txq), (buf), (nbytes))
#define doca_gpu_dev_eth_txq_commit_strong(txq) gazsi_txq_noop(txq)
#define doca_gpu_dev_eth_txq_push(txq) gazsi_txq_noop(txq)

#endif /* GAZSI_DOCA33_COMPAT_CUH */
