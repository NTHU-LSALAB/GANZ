/*
 * DOCA 3.3 Build Compatibility — host side
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SCOPE: build compatibility ONLY. Nothing here is part of the paper-alignment
 * work; it exists so the application can be compiled against the DOCA 3.3.x
 * SDK installed on this machine while the code targets the DOCA 2.x API.
 *
 * This header is applied IDENTICALLY to the baseline tree and the modified
 * tree, so it cannot bias an A/B comparison between them.
 *
 * Renames and signature changes covered here:
 *
 *   DOCA 2.x                                DOCA 3.3
 *   -------------------------------------   ---------------------------------
 *   DOCA_FLOW_NO_WAIT                       DOCA_FLOW_ENTRY_FLAGS_NO_WAIT
 *   estimate_packet_buf_size(7 args)        (9 args: + head_size, tail_size)
 *
 * Two further changes could not be expressed as macros and are edited directly
 * at their call sites, each marked with a DOCA33-COMPAT comment:
 *   - struct doca_flow_fwd RSS fields moved into a nested rss sub-struct
 *     (doca/flow.c)
 *   - struct rte_tcp_hdr lost its per-flag bitfields (tcp/cpu_rss.c)
 */

#ifndef GAZSI_DOCA33_COMPAT_H
#define GAZSI_DOCA33_COMPAT_H

#include <doca_flow.h>
#include <doca_eth_rxq.h>
#include <doca_error.h>

/* Entry flags were renamed and moved into an enum */
#ifndef DOCA_FLOW_NO_WAIT
#define DOCA_FLOW_NO_WAIT DOCA_FLOW_ENTRY_FLAGS_NO_WAIT
#endif

/*
 * 3.3 inserted head_size and tail_size before the out parameter. Both are 0 for
 * the cyclic-buffer configuration this application uses, which is what the 2.x
 * call implied.
 *
 * Defined as an inline BEFORE the macro so the macro does not rewrite the
 * forwarding call inside it.
 */
static inline doca_error_t gazsi_estimate_packet_buf_size(enum doca_eth_rxq_type type,
							  uint32_t rate,
							  uint16_t pkt_max_time,
							  uint32_t max_packet_size,
							  uint32_t max_burst_size,
							  uint8_t log_max_lro_pkt_sz,
							  uint32_t *buf_size)
{
	return doca_eth_rxq_estimate_packet_buf_size(type,
						     rate,
						     pkt_max_time,
						     max_packet_size,
						     max_burst_size,
						     log_max_lro_pkt_sz,
						     0, /* head_size */
						     0, /* tail_size */
						     buf_size);
}

#define doca_eth_rxq_estimate_packet_buf_size(t, r, pmt, mps, mbs, lro, out) \
	gazsi_estimate_packet_buf_size((t), (r), (pmt), (mps), (mbs), (lro), (out))

/*
 * DPDK 25.11 dropped the TCP option-kind constants (it keeps only flag masks).
 * These are the standard IANA kind numbers the code was using; the matching
 * *_nbytes length values are defined locally in tcp/cpu_rss.c.
 */
#ifndef RTE_TCP_OPT_END
#define RTE_TCP_OPT_END            0  /* End of option list      */
#define RTE_TCP_OPT_NOP            1  /* No-Operation            */
#define RTE_TCP_OPT_MSS            2  /* Maximum Segment Size    */
#define RTE_TCP_OPT_WND_SCALE      3  /* Window Scale (RFC 1323) */
#define RTE_TCP_OPT_SACK_PERMITTED 4  /* SACK Permitted (RFC 2018) */
#define RTE_TCP_OPT_SACK           5  /* SACK (RFC 2018)         */
#define RTE_TCP_OPT_TIMESTAMP      8  /* Timestamps (RFC 1323)   */
#endif

#endif /* GAZSI_DOCA33_COMPAT_H */
