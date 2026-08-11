/*
 * GAZSI - TCP header encoding regression for the DOCA33-COMPAT edits
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * tcp/cpu_rss.c lost the rte_tcp_hdr per-flag bitfields in DPDK 25.11 and had
 * to switch to raw byte fields. Two encodings are easy to get wrong and were
 * wrong in an intermediate version of that change:
 *
 *   data_off   carries the header length in words in its HIGH nibble. Writing
 *              the word count unshifted yields a header length of 0 and a
 *              nonsense reserved nibble, so peers drop the segment.
 *   tcp_flags  is a full byte. OR-ing ACK onto a recycled mbuf's existing byte
 *              can leak stale flags (e.g. RST) into the handshake reply.
 *
 * This pins both encodings against the real DPDK constants.
 */

#include <rte_tcp.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

static int failures, checks;

#define CHECK(cond, ...)                                                       \
	do {                                                                   \
		checks++;                                                      \
		if (!(cond)) {                                                 \
			failures++;                                            \
			printf("  FAIL %s:%d: ", __FILE__, __LINE__);          \
			printf(__VA_ARGS__);                                   \
			printf("\n");                                          \
		}                                                              \
	} while (0)

/* Mirrors the expression in tcp/cpu_rss.c */
static inline uint8_t encode_data_off(size_t tcp_option_array_len)
{
	return (uint8_t)((5 + tcp_option_array_len / 4) << 4);
}

/* Mirrors the flag construction in tcp/cpu_rss.c */
static inline uint8_t encode_flags(uint8_t src_flags)
{
	uint8_t flags = RTE_TCP_ACK_FLAG;

	if (!(src_flags & RTE_TCP_ACK_FLAG))
		flags |= src_flags & (RTE_TCP_SYN_FLAG | RTE_TCP_FIN_FLAG);

	return flags;
}

int main(void)
{
	printf("GAZSI TCP header encoding regression\n");

	/* --- data_off: header length must land in the high nibble --- */
	struct { size_t opts; unsigned words; } dcases[] = {
		{ 0, 5 }, { 4, 6 }, { 8, 7 }, { 12, 8 }, { 20, 10 }, { 40, 15 },
	};
	for (unsigned i = 0; i < sizeof(dcases) / sizeof(dcases[0]); i++) {
		uint8_t off = encode_data_off(dcases[i].opts);
		CHECK((off >> 4) == dcases[i].words,
		      "opts=%zu: header words %u, expected %u",
		      dcases[i].opts, off >> 4, dcases[i].words);
		CHECK((off & 0x0F) == 0,
		      "opts=%zu: reserved low nibble must be 0, got 0x%x",
		      dcases[i].opts, off & 0x0F);
	}
	/* The real call site uses a 19-byte option array -> 5 + 4 = 9 words */
	{
		size_t real_opts = 4 /*MSS*/ + 2 /*SACK_PERM*/ + 10 /*TS*/ + 1 /*NOP*/ + 3 /*WS*/;
		uint8_t off = encode_data_off(real_opts);
		CHECK(real_opts == 20, "option array is %zu bytes, expected 20", real_opts);
		CHECK((off >> 4) == 10, "real options: %u words, expected 10", off >> 4);
		CHECK(off == 0xA0, "real options: data_off 0x%02x, expected 0xA0", off);
	}
	/* The unshifted form the bug produced must be distinguishable */
	CHECK(encode_data_off(20) != (uint8_t)(5 + 20 / 4),
	      "shifted and unshifted encodings must differ");

	/* --- tcp_flags: built from scratch, no stale bits --- */
	/* SYN in, no ACK: reply is SYN|ACK */
	CHECK(encode_flags(RTE_TCP_SYN_FLAG) == (RTE_TCP_SYN_FLAG | RTE_TCP_ACK_FLAG),
	      "SYN -> SYN|ACK, got 0x%02x", encode_flags(RTE_TCP_SYN_FLAG));
	/* FIN in, no ACK: reply is FIN|ACK */
	CHECK(encode_flags(RTE_TCP_FIN_FLAG) == (RTE_TCP_FIN_FLAG | RTE_TCP_ACK_FLAG),
	      "FIN -> FIN|ACK, got 0x%02x", encode_flags(RTE_TCP_FIN_FLAG));
	/* Already ACKed: reply is a bare ACK, SYN/FIN not copied */
	CHECK(encode_flags(RTE_TCP_ACK_FLAG | RTE_TCP_SYN_FLAG) == RTE_TCP_ACK_FLAG,
	      "ACK|SYN -> ACK, got 0x%02x", encode_flags(RTE_TCP_ACK_FLAG | RTE_TCP_SYN_FLAG));
	/* Stale/unrelated input flags must never propagate */
	uint8_t noisy = RTE_TCP_RST_FLAG | RTE_TCP_URG_FLAG | RTE_TCP_PSH_FLAG |
			RTE_TCP_CWR_FLAG | RTE_TCP_ECE_FLAG;
	CHECK((encode_flags(noisy) & noisy) == 0,
	      "unrelated flags leaked: 0x%02x", encode_flags(noisy) & noisy);
	CHECK(encode_flags(noisy) == RTE_TCP_ACK_FLAG,
	      "noisy input -> bare ACK, got 0x%02x", encode_flags(noisy));
	/* Output is always at least ACK */
	for (unsigned v = 0; v < 256; v++)
		CHECK(encode_flags((uint8_t)v) & RTE_TCP_ACK_FLAG,
		      "input 0x%02x: ACK must always be set", v);

	printf("%d checks, %d failures\n", checks, failures);
	return failures == 0 ? 0 : 1;
}
