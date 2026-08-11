/*
 * GAZSI - Device-Resident Pipeline Equivalence Test
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Asserts that moving tokenization and response formatting onto the GPU did not
 * change a single byte of behaviour:
 *
 *   1. tokenizer   GPU kernel output == host reference, over inputs chosen to
 *                  hit every branch (empty, clamping, truncation, separators)
 *   2. formatter   GPU-assembled JSON body == the exact snprintf format string
 *                  the host used to produce
 *   3. fallback    device-side status body is well formed
 *   4. isolation   request/response bytes are unreachable from the host: the
 *                  payload plane is plain cudaMalloc memory
 *
 * Requires a CUDA device. Needs no DOCA and no TensorRT engine, so it runs
 * anywhere with a GPU.
 *
 * Build and run:
 *   nvcc -std=c++17 -I../inference -o test_gpu_pipeline \
 *        test_gpu_pipeline.cu ../inference/gpu_pipeline.cu
 *   ./test_gpu_pipeline
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gpu_pipeline.h"

#define SEQ_LEN 128

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
		if (e != cudaSuccess) {                                       \
			printf("FATAL %s:%d: %s\n", __FILE__, __LINE__,       \
			       cudaGetErrorString(e));                        \
			exit(1);                                              \
		}                                                              \
	} while (0)

/* ------------------------------------------------------------------ */

struct harness {
	struct inference_payload_plane *d_plane;
	struct inference_ring_buffer *h_ring;
	struct inference_ring_buffer *d_ring;
	struct gazsi_batch_item *h_items;
	struct gazsi_batch_item *d_items;
	int64_t *d_ids;
	int64_t *d_mask;
	int32_t *d_tok;
	float *d_out;
};

static void harness_init(struct harness *h)
{
	CUDA_OK(cudaMalloc(&h->d_plane, sizeof(*h->d_plane)));
	CUDA_OK(cudaMemset(h->d_plane, 0, sizeof(*h->d_plane)));

	CUDA_OK(cudaHostAlloc(&h->h_ring, sizeof(*h->h_ring), cudaHostAllocMapped));
	memset(h->h_ring, 0, sizeof(*h->h_ring));
	CUDA_OK(cudaHostGetDevicePointer(&h->d_ring, h->h_ring, 0));

	CUDA_OK(cudaHostAlloc(&h->h_items, 8 * sizeof(*h->h_items), cudaHostAllocMapped));
	memset(h->h_items, 0, 8 * sizeof(*h->h_items));
	CUDA_OK(cudaHostGetDevicePointer(&h->d_items, h->h_items, 0));

	CUDA_OK(cudaMalloc(&h->d_ids, 8 * SEQ_LEN * sizeof(int64_t)));
	CUDA_OK(cudaMalloc(&h->d_mask, 8 * SEQ_LEN * sizeof(int64_t)));
	CUDA_OK(cudaMalloc(&h->d_tok, 8 * sizeof(int32_t)));
	CUDA_OK(cudaMalloc(&h->d_out, 8 * SEQ_LEN * 768 * sizeof(float)));
}

/*
 * Stage request text into the device payload plane.
 *
 * On the serving path the GPU RX kernel writes these bytes straight from the
 * NIC packet; the host has no way in. Here the test acts as a stand-in for the
 * RX kernel, which is why an explicit H2D copy appears in a test and nowhere
 * in the data path.
 */
static void stage_request(struct harness *h, uint32_t slot, uint32_t off,
			  const char *text, uint32_t len)
{
	if (slot >= INFERENCE_RING_SIZE || off + len > GAZSI_REQUEST_BYTES) {
		printf("FATAL bad staging: slot=%u off=%u len=%u\n", slot, off, len);
		exit(1);
	}
	char *dst = (char *)h->d_plane + offsetof(struct inference_payload_plane, slots) +
		    (size_t)slot * sizeof(struct inference_payload_slot) +
		    offsetof(struct inference_payload_slot, request) + off;
	CUDA_OK(cudaMemcpy(dst, text, len, cudaMemcpyHostToDevice));
}

static void read_response(struct harness *h, uint32_t slot, uint32_t rec,
			  char *dst, size_t n)
{
	const char *src = (const char *)h->d_plane +
			  offsetof(struct inference_payload_plane, slots) +
			  (size_t)slot * sizeof(struct inference_payload_slot) +
			  offsetof(struct inference_payload_slot, response) +
			  (size_t)rec * GAZSI_RESPONSE_STRIDE;
	CUDA_OK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

/* ------------------------------------------------------------------ */
/* 1. Tokenizer equivalence                                           */
/* ------------------------------------------------------------------ */

static void test_tokenizer(struct harness *h)
{
	/* Inputs chosen to cover each branch of the tokenizer */
	static char long_words[1200];
	static char many_tokens[1024];
	static char clamp_probe[700];

	/* > GAZSI_TOKENIZE_MAX bytes: exercises the 512-byte clamp */
	for (int i = 0; i < (int)sizeof(long_words) - 1; i++)
		long_words[i] = (char)('a' + (i % 26));
	long_words[sizeof(long_words) - 1] = '\0';

	/* > SEQ_LEN words: exercises mid-text truncation at token_pos == 127 */
	{
		int p = 0;
		for (int w = 0; w < 300 && p < (int)sizeof(many_tokens) - 8; w++)
			p += snprintf(many_tokens + p, sizeof(many_tokens) - p, "w%d ", w);
	}

	/* Word boundary sitting exactly on the clamp */
	memset(clamp_probe, 'x', sizeof(clamp_probe));
	clamp_probe[GAZSI_TOKENIZE_MAX - 1] = ' ';
	clamp_probe[GAZSI_TOKENIZE_MAX] = ' ';
	clamp_probe[sizeof(clamp_probe) - 1] = '\0';

	const char *cases[] = {
		"",
		" ",
		"    ",
		"hello",
		"hello world",
		"hello  world",		    /* consecutive separators */
		" leading",
		"trailing ",
		" both ",
		"a b c d e f g h i j",
		"warmup_0",
		"the quick brown fox jumps over the lazy dog",
		"tabs\tand\nnewlines hash into the word",
		"punctuation!@#$%^&*()_+-=[]{}|;:',.<>?/",
		"UTF8 \xc3\xa9\xc3\xa8\xc3\xaa \xe4\xb8\xad\xe6\x96\x87 bytes",
		"high\x80\xff\xfe bytes",
		"single_very_long_token_with_no_spaces_at_all_0123456789",
		long_words,
		many_tokens,
		clamp_probe,
	};
	const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

	int64_t h_ids[SEQ_LEN], h_mask[SEQ_LEN];
	int64_t ref_ids[SEQ_LEN], ref_mask[SEQ_LEN];
	int32_t h_tok = 0;

	printf("[1] tokenizer equivalence (%d cases)\n", ncases);

	for (int c = 0; c < ncases; c++) {
		const char *text = cases[c];
		uint32_t len = (uint32_t)strlen(text);

		/* Fresh slot each case so leftover bytes cannot mask a bug */
		uint32_t slot = (uint32_t)(c % INFERENCE_RING_SIZE);
		CUDA_OK(cudaMemset((char *)h->d_plane +
					   offsetof(struct inference_payload_plane, slots) +
					   (size_t)slot * sizeof(struct inference_payload_slot),
				   0, sizeof(struct inference_payload_slot)));

		uint32_t stage_len = len < GAZSI_REQUEST_BYTES - 1 ? len : GAZSI_REQUEST_BYTES - 1;
		stage_request(h, slot, 0, text, stage_len);

		h->h_items[0].slot_index = slot;
		h->h_items[0].record_index = 0;
		h->h_items[0].req_offset = 0;
		h->h_items[0].req_length = stage_len;

		CUDA_OK(cudaMemset(h->d_ids, 0xAB, 8 * SEQ_LEN * sizeof(int64_t)));
		CUDA_OK(cudaMemset(h->d_mask, 0xAB, 8 * SEQ_LEN * sizeof(int64_t)));

		int rc = gazsi_launch_tokenize(h->d_plane, h->d_items, 1,
					       h->d_ids, h->d_mask, h->d_tok, SEQ_LEN, NULL);
		CHECK(rc == 0, "case %d: launch failed", c);
		CUDA_OK(cudaDeviceSynchronize());

		CUDA_OK(cudaMemcpy(h_ids, h->d_ids, SEQ_LEN * sizeof(int64_t), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaMemcpy(h_mask, h->d_mask, SEQ_LEN * sizeof(int64_t), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaMemcpy(&h_tok, h->d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost));

		/*
		 * Reference operates on the same clamped byte range the GPU sees,
		 * mirroring the 512-byte clamp the old host path applied before
		 * tokenizing.
		 */
		char clamped[GAZSI_TOKENIZE_MAX + 1];
		uint32_t clamp = stage_len < GAZSI_TOKENIZE_MAX ? stage_len : GAZSI_TOKENIZE_MAX;
		memcpy(clamped, text, clamp);
		clamped[clamp] = '\0';

		int ref_tok = gazsi_reference_tokenize(clamped, ref_ids, ref_mask, SEQ_LEN);

		CHECK(h_tok == ref_tok, "case %d (%.28s): token_count gpu=%d ref=%d",
		      c, text, (int)h_tok, ref_tok);

		int mism_ids = -1, mism_mask = -1;
		for (int i = 0; i < SEQ_LEN; i++) {
			if (h_ids[i] != ref_ids[i] && mism_ids < 0) mism_ids = i;
			if (h_mask[i] != ref_mask[i] && mism_mask < 0) mism_mask = i;
		}
		CHECK(mism_ids < 0, "case %d (%.28s): input_ids differ at %d gpu=%lld ref=%lld",
		      c, text, mism_ids,
		      mism_ids >= 0 ? (long long)h_ids[mism_ids] : 0,
		      mism_ids >= 0 ? (long long)ref_ids[mism_ids] : 0);
		CHECK(mism_mask < 0, "case %d (%.28s): attention_mask differs at %d",
		      c, text, mism_mask);
	}
}

/* Batched tokenization must equal per-item tokenization */
static void test_tokenizer_batch(struct harness *h)
{
	const char *texts[8] = {
		"alpha beta", "", "gamma", "delta epsilon zeta",
		"eta", "theta iota kappa lambda", "mu", "nu xi omicron",
	};
	int64_t h_ids[8 * SEQ_LEN], h_mask[8 * SEQ_LEN];
	int64_t ref_ids[SEQ_LEN], ref_mask[SEQ_LEN];
	int32_t h_tok[8];

	printf("[2] batched tokenization (batch=8, distinct slots)\n");

	for (int b = 0; b < 8; b++) {
		uint32_t slot = (uint32_t)(100 + b);
		CUDA_OK(cudaMemset((char *)h->d_plane +
					   offsetof(struct inference_payload_plane, slots) +
					   (size_t)slot * sizeof(struct inference_payload_slot),
				   0, sizeof(struct inference_payload_slot)));
		stage_request(h, slot, 0, texts[b], (uint32_t)strlen(texts[b]));
		h->h_items[b].slot_index = slot;
		h->h_items[b].record_index = 0;
		h->h_items[b].req_offset = 0;
		h->h_items[b].req_length = (uint32_t)strlen(texts[b]);
	}

	int rc = gazsi_launch_tokenize(h->d_plane, h->d_items, 8,
				       h->d_ids, h->d_mask, h->d_tok, SEQ_LEN, NULL);
	CHECK(rc == 0, "batch launch failed");
	CUDA_OK(cudaDeviceSynchronize());

	CUDA_OK(cudaMemcpy(h_ids, h->d_ids, 8 * SEQ_LEN * sizeof(int64_t), cudaMemcpyDeviceToHost));
	CUDA_OK(cudaMemcpy(h_mask, h->d_mask, 8 * SEQ_LEN * sizeof(int64_t), cudaMemcpyDeviceToHost));
	CUDA_OK(cudaMemcpy(h_tok, h->d_tok, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost));

	for (int b = 0; b < 8; b++) {
		int ref_tok = gazsi_reference_tokenize(texts[b], ref_ids, ref_mask, SEQ_LEN);
		CHECK(h_tok[b] == ref_tok, "batch elem %d: token_count gpu=%d ref=%d",
		      b, (int)h_tok[b], ref_tok);
		int bad = -1;
		for (int i = 0; i < SEQ_LEN; i++) {
			if (h_ids[b * SEQ_LEN + i] != ref_ids[i] ||
			    h_mask[b * SEQ_LEN + i] != ref_mask[i]) { bad = i; break; }
		}
		CHECK(bad < 0, "batch elem %d: row differs at %d", b, bad);
	}
}

/* ------------------------------------------------------------------ */
/* 3. Formatter equivalence                                           */
/* ------------------------------------------------------------------ */

static void test_formatter(struct harness *h)
{
	/*
	 * The exact format string the host path used. Any divergence here is a
	 * change visible on the socket.
	 */
	static const char *FMT =
		"{\"input\":\"%s\",\"tokens\":%d,\"embedding_sample\":[%.6f,%.6f,%.6f],"
		"\"inference_time_us\":%ld,\"batch_size\":%d}";

	struct fcase { const char *text; float e[3]; int tokens; long us; };
	static const struct fcase cases[] = {
		{ "hello world",       { 0.123456f, -0.654321f, 0.0f },        4,  1234 },
		{ "warmup_0",          { 0.0f, 0.0f, 0.0f },                   3,     0 },
		{ "negatives",         { -1.5f, -0.000001f, -123.456789f },    3,   999 },
		{ "rounding",          { 0.9999995f, 1.9999999f, 0.0000005f }, 3,     1 },
		{ "large",             { 12345.678f, -98765.4321f, 1e6f },     3, 50000 },
		{ "tiny",              { 1e-7f, -1e-7f, 5e-7f },               3,     7 },
		{ "integral",          { 1.0f, -2.0f, 3.0f },                  3,    42 },
		{ "empty_embed",       { 0.5f, 0.25f, 0.125f },                2,     3 },
	};
	const int n = (int)(sizeof(cases) / sizeof(cases[0]));

	printf("[3] formatter equivalence (%d cases, batch=%d)\n", n, n);

	/* Stage requests, token counts and embeddings for the whole batch */
	int32_t tok[8];
	float out_host[8 * 768];
	memset(out_host, 0, sizeof(out_host));

	for (int b = 0; b < n; b++) {
		uint32_t slot = (uint32_t)(200 + b);
		CUDA_OK(cudaMemset((char *)h->d_plane +
					   offsetof(struct inference_payload_plane, slots) +
					   (size_t)slot * sizeof(struct inference_payload_slot),
				   0, sizeof(struct inference_payload_slot)));
		stage_request(h, slot, 0, cases[b].text, (uint32_t)strlen(cases[b].text));
		h->h_items[b].slot_index = slot;
		h->h_items[b].record_index = 0;
		h->h_items[b].req_offset = 0;
		h->h_items[b].req_length = (uint32_t)strlen(cases[b].text);
		tok[b] = cases[b].tokens;
		for (int k = 0; k < 3; k++)
			out_host[b * 768 + k] = cases[b].e[k];
	}
	CUDA_OK(cudaMemcpy(h->d_tok, tok, n * sizeof(int32_t), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(h->d_out, out_host, (size_t)n * 768 * sizeof(float),
			   cudaMemcpyHostToDevice));

	/*
	 * elapsed_us is a single scalar for the whole batch, exactly as the host
	 * path emitted it, so drive the batch with each case's value in turn and
	 * compare only that case.
	 */
	for (int b = 0; b < n; b++) {
		int rc = gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items, n,
						    h->d_out, h->d_tok, 768, 768,
						    cases[b].us, NULL);
		CHECK(rc == 0, "case %d: format launch failed", b);
		CUDA_OK(cudaDeviceSynchronize());

		char got[GAZSI_RESPONSE_STRIDE];
		read_response(h, (uint32_t)(200 + b), 0, got, sizeof(got));

		char want[GAZSI_RESPONSE_STRIDE];
		int want_len = snprintf(want, sizeof(want), FMT,
					cases[b].text, cases[b].tokens,
					cases[b].e[0], cases[b].e[1], cases[b].e[2],
					cases[b].us, n);

		uint32_t got_len = h->h_ring->slots[200 + b].directory[0].resp_length;

		CHECK((int)got_len == want_len, "case %d (%s): length gpu=%u host=%d",
		      b, cases[b].text, got_len, want_len);
		CHECK(memcmp(got, want, want_len) == 0,
		      "case %d (%s):\n       gpu =%.*s\n       host=%s",
		      b, cases[b].text, (int)got_len, got, want);
	}
}

/* Multi-record packing: each record's response must land at its own stride */
static void test_formatter_packed(struct harness *h)
{
	const char *recs[MAX_RECORDS_PER_SLOT] = { "rec_zero", "rec_one", "rec_two", "rec_three" };
	uint32_t slot = 250;
	uint32_t offs[MAX_RECORDS_PER_SLOT];

	printf("[4] packed multi-record formatting (%d records, one slot)\n",
	       MAX_RECORDS_PER_SLOT);

	CUDA_OK(cudaMemset((char *)h->d_plane +
				   offsetof(struct inference_payload_plane, slots) +
				   (size_t)slot * sizeof(struct inference_payload_slot),
			   0, sizeof(struct inference_payload_slot)));

	uint32_t off = 0;
	for (int r = 0; r < MAX_RECORDS_PER_SLOT; r++) {
		uint32_t len = (uint32_t)strlen(recs[r]);
		stage_request(h, slot, off, recs[r], len);
		offs[r] = off;
		h->h_items[r].slot_index = slot;
		h->h_items[r].record_index = (uint32_t)r;
		h->h_items[r].req_offset = off;
		h->h_items[r].req_length = len;
		off += len + 1;
	}

	int32_t tok[MAX_RECORDS_PER_SLOT] = { 3, 3, 3, 3 };
	float out_host[MAX_RECORDS_PER_SLOT * 768];
	memset(out_host, 0, sizeof(out_host));
	for (int r = 0; r < MAX_RECORDS_PER_SLOT; r++)
		out_host[r * 768] = 0.5f * (float)(r + 1);

	CUDA_OK(cudaMemcpy(h->d_tok, tok, sizeof(tok), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(h->d_out, out_host, sizeof(out_host), cudaMemcpyHostToDevice));

	int rc = gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items,
					    MAX_RECORDS_PER_SLOT, h->d_out, h->d_tok,
					    768, 768, 77, NULL);
	CHECK(rc == 0, "packed format launch failed");
	CUDA_OK(cudaDeviceSynchronize());

	for (int r = 0; r < MAX_RECORDS_PER_SLOT; r++) {
		char got[GAZSI_RESPONSE_STRIDE];
		read_response(h, slot, (uint32_t)r, got, sizeof(got));
		uint32_t len = h->h_ring->slots[slot].directory[r].resp_length;

		CHECK(len > 0 && len <= GAZSI_RESPONSE_MAX, "record %d: bad length %u", r, len);

		/* Each record must echo its OWN request text, not a neighbour's */
		char needle[64];
		snprintf(needle, sizeof(needle), "{\"input\":\"%s\"", recs[r]);
		CHECK(len >= strlen(needle) && memcmp(got, needle, strlen(needle)) == 0,
		      "record %d: expected prefix %s, got %.60s", r, needle, got);
	}

	/* The request text must survive formatting: responses live elsewhere */
	for (int r = 0; r < MAX_RECORDS_PER_SLOT; r++) {
		char req[64] = {0};
		const char *src = (const char *)h->d_plane +
				  offsetof(struct inference_payload_plane, slots) +
				  (size_t)slot * sizeof(struct inference_payload_slot) +
				  offsetof(struct inference_payload_slot, request) + offs[r];
		CUDA_OK(cudaMemcpy(req, src, strlen(recs[r]), cudaMemcpyDeviceToHost));
		CHECK(strcmp(req, recs[r]) == 0,
		      "record %d: request text was clobbered (%s != %s)", r, req, recs[r]);
	}
}

/* ------------------------------------------------------------------ */
/* 5. Device-side fallback body                                       */
/* ------------------------------------------------------------------ */

static void test_static_response(struct harness *h)
{
	uint32_t slot = 40;
	const char *text = "fallback_probe";

	printf("[5] device-side status body\n");

	CUDA_OK(cudaMemset((char *)h->d_plane +
				   offsetof(struct inference_payload_plane, slots) +
				   (size_t)slot * sizeof(struct inference_payload_slot),
			   0, sizeof(struct inference_payload_slot)));
	stage_request(h, slot, 0, text, (uint32_t)strlen(text));

	h->h_items[0].slot_index = slot;
	h->h_items[0].record_index = 0;
	h->h_items[0].req_offset = 0;
	h->h_items[0].req_length = (uint32_t)strlen(text);

	int rc = gazsi_launch_static_response(h->d_ring, h->d_plane, h->d_items, 1,
					      "no_model", NULL);
	CHECK(rc == 0, "static response launch failed");
	CUDA_OK(cudaDeviceSynchronize());

	char got[GAZSI_RESPONSE_STRIDE];
	read_response(h, slot, 0, got, sizeof(got));
	uint32_t len = h->h_ring->slots[slot].directory[0].resp_length;

	char want[256];
	int want_len = snprintf(want, sizeof(want), "{\"input\":\"%s\",\"status\":\"no_model\"}", text);

	CHECK((int)len == want_len, "status length gpu=%u host=%d", len, want_len);
	CHECK(memcmp(got, want, want_len) == 0, "status body:\n       gpu =%.*s\n       host=%s",
	      (int)len, got, want);
}

/* ------------------------------------------------------------------ */
/* 6. Host isolation of the data plane                                */
/* ------------------------------------------------------------------ */

static void test_host_isolation(struct harness *h)
{
	printf("[6] payload plane is device-only\n");

	cudaPointerAttributes attr;
	memset(&attr, 0, sizeof(attr));
	CUDA_OK(cudaPointerGetAttributes(&attr, h->d_plane));

	CHECK(attr.type == cudaMemoryTypeDevice,
	      "payload plane memory type is %d, expected device (%d)",
	      (int)attr.type, (int)cudaMemoryTypeDevice);
	CHECK(attr.hostPointer == NULL,
	      "payload plane has a host alias (%p) — CPU could read payload",
	      attr.hostPointer);

	/* Control plane, by contrast, is meant to be host-visible metadata */
	memset(&attr, 0, sizeof(attr));
	CUDA_OK(cudaPointerGetAttributes(&attr, h->h_ring));
	CHECK(attr.hostPointer != NULL, "control plane should be host-mapped");
}

/* ------------------------------------------------------------------ */
/* 7. Malformed descriptors must never be dereferenced                */
/* ------------------------------------------------------------------ */

static void test_invalid_descriptors(struct harness *h)
{
	printf("[7] malformed descriptor rejection\n");

	/*
	 * Each of these addresses memory outside the payload plane. Under
	 * compute-sanitizer any dereference shows up as an out-of-bounds access,
	 * so this test doubles as the sanitizer probe for descriptor validation.
	 */
	struct bad { const char *name; uint32_t slot, rec, off, len; };
	static const struct bad cases[] = {
		{ "slot_index == RING_SIZE",   INFERENCE_RING_SIZE,     0, 0, 8 },
		{ "slot_index huge",           0xFFFFFFFFu,             0, 0, 8 },
		{ "record_index == MAX",       1, MAX_RECORDS_PER_SLOT,    0, 8 },
		{ "record_index huge",         1, 0xFFFFFFFFu,            0, 8 },
		{ "req_offset == REQUEST_BYTES", 1, 0, GAZSI_REQUEST_BYTES, 8 },
		{ "req_offset huge",           1, 0, 0xFFFFFFFFu,          8 },
		{ "req_length overruns",       1, 0, 0, GAZSI_REQUEST_BYTES + 1 },
		{ "offset+length overflows",   1, 0, GAZSI_REQUEST_BYTES - 4, 0xFFFFFFFFu },
		{ "length huge, offset mid",   1, 0, 64, 0xFFFFFF00u },
	};
	const int n = (int)(sizeof(cases) / sizeof(cases[0]));

	for (int c = 0; c < n; c++) {
		h->h_items[0].slot_index   = cases[c].slot;
		h->h_items[0].record_index = cases[c].rec;
		h->h_items[0].req_offset   = cases[c].off;
		h->h_items[0].req_length   = cases[c].len;

		CUDA_OK(cudaMemset(h->d_tok, 0, sizeof(int32_t)));

		int rc = gazsi_launch_tokenize(h->d_plane, h->d_items, 1,
					       h->d_ids, h->d_mask, h->d_tok, SEQ_LEN, NULL);
		CHECK(rc == 0, "%s: tokenize launch failed", cases[c].name);
		CUDA_OK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError() == cudaSuccess, "%s: tokenize faulted", cases[c].name);

		/* Malformed input must yield the empty-input token stream */
		int32_t tok = 0;
		int64_t ids[2] = {0, 0};
		CUDA_OK(cudaMemcpy(&tok, h->d_tok, sizeof(tok), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaMemcpy(ids, h->d_ids, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
		CHECK(tok == 2, "%s: expected token_count 2, got %d", cases[c].name, (int)tok);
		CHECK(ids[0] == GAZSI_TOKEN_CLS && ids[1] == GAZSI_TOKEN_SEP,
		      "%s: expected [CLS][SEP]", cases[c].name);

		rc = gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items, 1,
						h->d_out, h->d_tok, 768, 768, 1, NULL);
		CHECK(rc == 0, "%s: format launch failed", cases[c].name);
		CUDA_OK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError() == cudaSuccess, "%s: format faulted", cases[c].name);

		rc = gazsi_launch_static_response(h->d_ring, h->d_plane, h->d_items, 1,
						  "bad_descriptor", NULL);
		CHECK(rc == 0, "%s: static launch failed", cases[c].name);
		CUDA_OK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError() == cudaSuccess, "%s: static faulted", cases[c].name);
	}

	/* A valid descriptor must still work after all that */
	uint32_t slot = 7;
	CUDA_OK(cudaMemset((char *)h->d_plane +
				   offsetof(struct inference_payload_plane, slots) +
				   (size_t)slot * sizeof(struct inference_payload_slot),
			   0, sizeof(struct inference_payload_slot)));
	stage_request(h, slot, 0, "still works", 11);
	h->h_items[0].slot_index = slot;
	h->h_items[0].record_index = 0;
	h->h_items[0].req_offset = 0;
	h->h_items[0].req_length = 11;
	gazsi_launch_tokenize(h->d_plane, h->d_items, 1, h->d_ids, h->d_mask,
			      h->d_tok, SEQ_LEN, NULL);
	CUDA_OK(cudaDeviceSynchronize());
	int32_t tok = 0;
	CUDA_OK(cudaMemcpy(&tok, h->d_tok, sizeof(tok), cudaMemcpyDeviceToHost));
	CHECK(tok == 4, "valid descriptor after invalid ones: token_count %d != 4", (int)tok);
}

/* ------------------------------------------------------------------ */
/* 8. embedding_dim guard and status handling                         */
/* ------------------------------------------------------------------ */

static void test_guards_and_status(struct harness *h)
{
	printf("[8] embedding_dim guard + status length handling\n");

	uint32_t slot = 9;
	CUDA_OK(cudaMemset((char *)h->d_plane +
				   offsetof(struct inference_payload_plane, slots) +
				   (size_t)slot * sizeof(struct inference_payload_slot),
			   0, sizeof(struct inference_payload_slot)));
	stage_request(h, slot, 0, "guard", 5);
	h->h_items[0].slot_index = slot;
	h->h_items[0].record_index = 0;
	h->h_items[0].req_offset = 0;
	h->h_items[0].req_length = 5;

	/* The response samples 3 floats, so a smaller output dim must be refused */
	for (int ed = 0; ed <= 2; ed++) {
		int rc = gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items, 1,
						    h->d_out, h->d_tok, ed, 768, 1, NULL);
		CHECK(rc == -1, "embedding_dim=%d should be refused, got rc=%d", ed, rc);
	}
	/* out_stride smaller than embedding_dim is incoherent */
	CHECK(gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items, 1,
					  h->d_out, h->d_tok, 768, 16, 1, NULL) == -1,
	      "out_stride < embedding_dim should be refused");
	/* And the coherent case still succeeds */
	CHECK(gazsi_launch_format_results(h->d_ring, h->d_plane, h->d_items, 1,
					  h->d_out, h->d_tok, 3, 3, 1, NULL) == 0,
	      "embedding_dim=3 should be accepted");
	CUDA_OK(cudaDeviceSynchronize());

	/*
	 * Status strings: empty, normal, and longer than the internal buffer.
	 * Consecutive differing statuses must each be reflected — an earlier
	 * caching bug could have left a stale keyword on the device.
	 */
	static char longstatus[256];
	for (int i = 0; i < (int)sizeof(longstatus) - 1; i++)
		longstatus[i] = 'S';
	longstatus[sizeof(longstatus) - 1] = '\0';

	const char *statuses[] = { "no_model", "x", "", "inference_failed", longstatus, "no_model" };
	const int ns = (int)(sizeof(statuses) / sizeof(statuses[0]));

	for (int i = 0; i < ns; i++) {
		CUDA_OK(cudaMemset((char *)h->d_plane +
					   offsetof(struct inference_payload_plane, slots) +
					   (size_t)slot * sizeof(struct inference_payload_slot),
				   0, sizeof(struct inference_payload_slot)));
		stage_request(h, slot, 0, "probe", 5);

		int rc = gazsi_launch_static_response(h->d_ring, h->d_plane, h->d_items, 1,
						      statuses[i], NULL);
		CHECK(rc == 0, "status %d: launch failed", i);
		CUDA_OK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError() == cudaSuccess, "status %d: faulted", i);

		char got[GAZSI_RESPONSE_STRIDE];
		read_response(h, slot, 0, got, sizeof(got));
		uint32_t len = h->h_ring->slots[slot].directory[0].resp_length;

		/* Body must be well formed and bounded */
		CHECK(len > 0 && len <= GAZSI_RESPONSE_MAX,
		      "status %d: length %u out of range", i, len);
		CHECK(memcmp(got, "{\"input\":\"probe\",\"status\":\"", 26) == 0,
		      "status %d: unexpected prefix %.40s", i, got);
		CHECK(got[len - 1] == '}', "status %d: body not closed", i);

		/* The keyword actually present must be this call's, truncated to fit */
		size_t want = strlen(statuses[i]);
		if (want > 63) want = 63;
		char needle[80];
		snprintf(needle, sizeof(needle), "\"status\":\"%.*s\"}", (int)want, statuses[i]);
		CHECK(len >= strlen(needle) &&
		      memcmp(got + len - strlen(needle), needle, strlen(needle)) == 0,
		      "status %d: expected suffix %s, got %.*s", i, needle, (int)len, got);
	}
}

/* ------------------------------------------------------------------ */

int main(void)
{
	int dev = 0;
	cudaDeviceProp prop;
	if (cudaGetDevice(&dev) != cudaSuccess ||
	    cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
		printf("No CUDA device available\n");
		return 77; /* skip */
	}
	printf("GAZSI device-resident pipeline tests on %s (sm_%d%d)\n\n",
	       prop.name, prop.major, prop.minor);

	printf("layout: control slot %zuB, payload slot %zuB, plane %.2f MiB\n\n",
	       sizeof(struct inference_ring_slot),
	       sizeof(struct inference_payload_slot),
	       sizeof(struct inference_payload_plane) / 1048576.0);

	struct harness h;
	harness_init(&h);

	test_tokenizer(&h);
	test_tokenizer_batch(&h);
	test_formatter(&h);
	test_formatter_packed(&h);
	test_static_response(&h);
	test_host_isolation(&h);
	test_invalid_descriptors(&h);
	test_guards_and_status(&h);

	printf("\n%d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
