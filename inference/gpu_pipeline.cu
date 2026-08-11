/*
 * GPU Packet Processing - Device-Resident Preprocessing and Result Assembly
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#include "gpu_pipeline.h"

/* ------------------------------------------------------------------ */
/* Descriptor validation                                              */
/* ------------------------------------------------------------------ */

/*
 * Batch descriptors live in host-mapped memory, so a host-side bug could
 * present indices or extents that do not address the payload plane. Every
 * kernel validates a descriptor BEFORE dereferencing it.
 *
 * All comparisons are overflow-safe: req_offset is bounded first, then the
 * length is compared against the remaining space by subtraction, so
 * offset + length is never formed and cannot wrap.
 */
__device__ static inline bool gazsi_item_valid(const struct gazsi_batch_item *it)
{
	if (it->slot_index >= INFERENCE_RING_SIZE)
		return false;
	if (it->record_index >= MAX_RECORDS_PER_SLOT)
		return false;
	if (it->req_offset >= GAZSI_REQUEST_BYTES)
		return false;
	if (it->req_length > (uint32_t)GAZSI_REQUEST_BYTES - it->req_offset)
		return false;
	return true;
}

/* ------------------------------------------------------------------ */
/* Tokenization                                                       */
/* ------------------------------------------------------------------ */

/*
 * Word-boundary tokenizer, device side.
 *
 * Deliberately a byte-for-byte transliteration of the host tokenizer this
 * replaces, including its quirks, because changing the token stream would
 * change model output:
 *   - ' ' is the ONLY separator (tabs and newlines hash into the word)
 *   - the scan stops as soon as token_pos reaches seq_len - 1, mid-text
 *   - an empty request emits exactly [CLS] [SEP]
 *   - unused positions stay 0 in both ids and mask
 *
 * One block per batch element: all threads clear the row, then lane 0 walks
 * the bytes. The walk is inherently serial (rolling hash), but it is at most
 * GAZSI_TOKENIZE_MAX bytes of device-local work.
 */
__global__ static void gazsi_tokenize_kernel(const struct inference_payload_plane *plane,
					     const struct gazsi_batch_item *items,
					     int batch_size,
					     int64_t *input_ids,
					     int64_t *attention_mask,
					     int32_t *token_counts,
					     int seq_len)
{
	int b = blockIdx.x;
	if (b >= batch_size)
		return;

	int64_t *ids = input_ids + (size_t)b * seq_len;
	int64_t *mask = attention_mask + (size_t)b * seq_len;

	/* Clear the row first — the host tokenizer memset both arrays. */
	for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
		ids[i] = 0;
		mask[i] = 0;
	}
	__syncthreads();

	if (threadIdx.x != 0)
		return;

	const struct gazsi_batch_item it = items[b];

	ids[0] = GAZSI_TOKEN_CLS;
	mask[0] = 1;

	/*
	 * Malformed descriptor: emit the empty-input token stream rather than
	 * dereference it. The formatter reports the same descriptor as an error,
	 * so the request still gets a well-formed response.
	 */
	if (!gazsi_item_valid(&it)) {
		ids[1] = GAZSI_TOKEN_SEP;
		mask[1] = 1;
		token_counts[b] = 2;
		return;
	}

	const char *text = plane->slots[it.slot_index].request + it.req_offset;

	uint32_t limit = it.req_length;
	if (limit > GAZSI_TOKENIZE_MAX)
		limit = GAZSI_TOKENIZE_MAX;

	if (limit == 0 || text[0] == '\0') {
		ids[1] = GAZSI_TOKEN_SEP;
		mask[1] = 1;
		token_counts[b] = 2;
		return;
	}

	int token_pos = 1;
	uint32_t hash = 0;
	bool in_word = false;

	for (uint32_t i = 0; i < limit && token_pos < seq_len - 1; i++) {
		char c = text[i];
		if (c == '\0')
			break;
		if (c == ' ') {
			if (in_word) {
				ids[token_pos] = GAZSI_TOKEN_BASE + (hash % GAZSI_TOKEN_MODULO);
				mask[token_pos] = 1;
				token_pos++;
				hash = 0;
				in_word = false;
			}
		} else {
			hash = hash * 31u + (uint32_t)(uint8_t)c;
			in_word = true;
		}
	}

	if (in_word && token_pos < seq_len - 1) {
		ids[token_pos] = GAZSI_TOKEN_BASE + (hash % GAZSI_TOKEN_MODULO);
		mask[token_pos] = 1;
		token_pos++;
	}

	ids[token_pos] = GAZSI_TOKEN_SEP;
	mask[token_pos] = 1;
	token_pos++;

	token_counts[b] = token_pos;
}

/* ------------------------------------------------------------------ */
/* Device-side number formatting                                      */
/* ------------------------------------------------------------------ */

__device__ static int dev_put_str(char *dst, const char *src, int max)
{
	int n = 0;
	while (src[n] != '\0' && n < max) {
		dst[n] = src[n];
		n++;
	}
	return n;
}

/* Unsigned decimal, no padding. Returns bytes written. */
__device__ static int dev_put_u64(char *dst, unsigned long long v, int max)
{
	char tmp[24];
	int n = 0;

	if (v == 0) {
		if (max < 1)
			return 0;
		dst[0] = '0';
		return 1;
	}
	while (v > 0 && n < (int)sizeof(tmp)) {
		tmp[n++] = (char)('0' + (v % 10ULL));
		v /= 10ULL;
	}
	if (n > max)
		n = max;
	for (int i = 0; i < n; i++)
		dst[i] = tmp[n - 1 - i];
	return n;
}

__device__ static int dev_put_i64(char *dst, long long v, int max)
{
	int n = 0;
	if (v < 0) {
		if (max < 1)
			return 0;
		dst[n++] = '-';
		return n + dev_put_u64(dst + n, (unsigned long long)(-v), max - n);
	}
	return dev_put_u64(dst, (unsigned long long)v, max);
}

/*
 * Equivalent of printf("%.6f", v).
 *
 * Integer/fraction split in double precision: the fractional scale stays below
 * 1e6 so it is exact, and float->double widening is lossless. Ties round half
 * away from zero rather than glibc's half-to-even; for float inputs an exact
 * tie at the 6th decimal is not representable, so the two agree in practice.
 */
__device__ static int dev_put_f6(char *dst, float value, int max)
{
	int n = 0;

	if (isnan(value)) {
		if (signbit(value) && max > 0)
			dst[n++] = '-';
		return n + dev_put_str(dst + n, "nan", max - n);
	}
	if (isinf(value)) {
		if (value < 0.0f && max > 0)
			dst[n++] = '-';
		return n + dev_put_str(dst + n, "inf", max - n);
	}

	double d = (double)value;
	if (signbit(value)) {
		if (max < 1)
			return 0;
		dst[n++] = '-';
		d = -d;
	}

	unsigned long long ip = (unsigned long long)d;
	double frac = d - (double)ip;
	unsigned long long f6 = (unsigned long long)(frac * 1000000.0 + 0.5);
	if (f6 >= 1000000ULL) {
		f6 -= 1000000ULL;
		ip += 1ULL;
	}

	n += dev_put_u64(dst + n, ip, max - n);
	if (n < max)
		dst[n++] = '.';

	/* Six digits, zero padded */
	for (int div = 100000; div >= 1 && n < max; div /= 10) {
		dst[n++] = (char)('0' + (int)((f6 / (unsigned long long)div) % 10ULL));
	}
	return n;
}

/* ------------------------------------------------------------------ */
/* Result formatting                                                  */
/* ------------------------------------------------------------------ */

/*
 * Assemble the JSON response body for each batch element, in place, in the
 * device-resident response arena.
 *
 * Byte-for-byte the same body the host used to snprintf, so clients see no
 * change. Note the "input" field is emitted raw and unescaped — preserved
 * from the host implementation deliberately: altering it would change what
 * goes out on the socket.
 *
 * Each record writes to response[record_index * GAZSI_RESPONSE_STRIDE], a
 * fixed stride, so no record can clobber another record's request text or
 * response — a hazard the single shared buffer used to have.
 */
__global__ static void gazsi_format_kernel(struct inference_ring_buffer *ring,
					   struct inference_payload_plane *plane,
					   const struct gazsi_batch_item *items,
					   int batch_size,
					   const float *d_output,
					   const int32_t *token_counts,
					   int out_stride,
					   long elapsed_us)
{
	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b >= batch_size)
		return;

	const struct gazsi_batch_item it = items[b];

	/*
	 * A malformed descriptor cannot be used to locate a response buffer at
	 * all, so there is nowhere safe to write. Drop it: the host treats a
	 * zero resp_length as a failed record and TX emits its error body.
	 */
	if (!gazsi_item_valid(&it))
		return;

	struct inference_payload_slot *pay = &plane->slots[it.slot_index];
	const char *req = pay->request + it.req_offset;
	char *out = pay->response + (size_t)it.record_index * GAZSI_RESPONSE_STRIDE;

	const float *emb = d_output + (size_t)b * out_stride;

	const int max = GAZSI_RESPONSE_MAX;
	int n = 0;

	n += dev_put_str(out + n, "{\"input\":\"", max - n);

	uint32_t rlen = it.req_length;
	if (rlen > GAZSI_TOKENIZE_MAX)
		rlen = GAZSI_TOKENIZE_MAX;
	for (uint32_t i = 0; i < rlen && n < max; i++) {
		char c = req[i];
		if (c == '\0')
			break;
		out[n++] = c;
	}

	n += dev_put_str(out + n, "\",\"tokens\":", max - n);
	n += dev_put_i64(out + n, (long long)token_counts[b], max - n);
	n += dev_put_str(out + n, ",\"embedding_sample\":[", max - n);
	n += dev_put_f6(out + n, emb[0], max - n);
	if (n < max)
		out[n++] = ',';
	n += dev_put_f6(out + n, emb[1], max - n);
	if (n < max)
		out[n++] = ',';
	n += dev_put_f6(out + n, emb[2], max - n);
	n += dev_put_str(out + n, "],\"inference_time_us\":", max - n);
	n += dev_put_i64(out + n, (long long)elapsed_us, max - n);
	n += dev_put_str(out + n, ",\"batch_size\":", max - n);
	n += dev_put_i64(out + n, (long long)batch_size, max - n);
	n += dev_put_str(out + n, "}", max - n);

	if (n < GAZSI_RESPONSE_STRIDE)
		out[n] = '\0';

	/* Publish length through the control plane so TX knows the byte count */
	/*
	 * Per-record length is the single source of truth: each thread writes
	 * only its own record's entry, so there is no cross-record aggregate to
	 * race on. Every record of a claimed slot is always in the same batch
	 * (see collect_ready_slots), so TX always finds a valid length.
	 */
	ring->slots[it.slot_index].directory[it.record_index].resp_length = (uint32_t)n;
}

/*
 * Fixed status body, written from the device.
 *
 * Keeps the degraded path (no TensorRT engine loaded) honest: even here the
 * CPU never writes a payload byte.
 */
__global__ static void gazsi_static_response_kernel(struct inference_ring_buffer *ring,
						    struct inference_payload_plane *plane,
						    const struct gazsi_batch_item *items,
						    int batch_size,
						    const char *status,
						    int status_len)
{
	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b >= batch_size)
		return;

	const struct gazsi_batch_item it = items[b];

	if (!gazsi_item_valid(&it))
		return;

	struct inference_payload_slot *pay = &plane->slots[it.slot_index];
	const char *req = pay->request + it.req_offset;
	char *out = pay->response + (size_t)it.record_index * GAZSI_RESPONSE_STRIDE;

	const int max = GAZSI_RESPONSE_MAX;
	int n = 0;

	n += dev_put_str(out + n, "{\"input\":\"", max - n);

	uint32_t rlen = it.req_length;
	if (rlen > GAZSI_TOKENIZE_MAX)
		rlen = GAZSI_TOKENIZE_MAX;
	for (uint32_t i = 0; i < rlen && n < max; i++) {
		char c = req[i];
		if (c == '\0')
			break;
		out[n++] = c;
	}

	n += dev_put_str(out + n, "\",\"status\":\"", max - n);
	for (int i = 0; i < status_len && n < max; i++)
		out[n++] = status[i];
	n += dev_put_str(out + n, "\"}", max - n);

	if (n < GAZSI_RESPONSE_STRIDE)
		out[n] = '\0';

	/*
	 * Per-record length is the single source of truth: each thread writes
	 * only its own record's entry, so there is no cross-record aggregate to
	 * race on. Every record of a claimed slot is always in the same batch
	 * (see collect_ready_slots), so TX always finds a valid length.
	 */
	ring->slots[it.slot_index].directory[it.record_index].resp_length = (uint32_t)n;
}

/* ------------------------------------------------------------------ */
/* Host launch wrappers                                               */
/* ------------------------------------------------------------------ */

/*
 * Device-resident copy of the fallback status string.
 *
 * A string literal lives in host memory, so it cannot be handed to a kernel.
 * One small device buffer is allocated on first use and reused; it holds a
 * status keyword, never request or result payload.
 */
#define GAZSI_STATUS_MAX 64
static char *g_d_status = NULL;
static char *g_h_status = NULL;

/*
 * Upload the status keyword for this call, unconditionally.
 *
 * An earlier version cached the last-uploaded string to skip the copy. That
 * was wrong across streams: the cache said "already present" while the copy
 * could still be pending on a different stream, so a kernel on this stream
 * could read stale bytes. The copy is 64 bytes on a path that only runs when
 * no engine is loaded, so always issuing it on the caller's stream is both
 * cheaper to reason about and correct by construction.
 *
 * The keyword is a fixed status token, never request or result payload.
 */
static int gazsi_stage_status(const char *status, int *out_len, cudaStream_t stream)
{
	int len = 0;
	while (status[len] != '\0' && len < GAZSI_STATUS_MAX - 1)
		len++;

	if (!g_d_status) {
		if (cudaMalloc(&g_d_status, GAZSI_STATUS_MAX) != cudaSuccess) {
			fprintf(stderr, "[GPU_PIPE] cudaMalloc status buffer failed\n");
			return -1;
		}
		if (cudaHostAlloc(&g_h_status, GAZSI_STATUS_MAX, cudaHostAllocDefault) != cudaSuccess) {
			fprintf(stderr, "[GPU_PIPE] cudaHostAlloc status staging failed\n");
			cudaFree(g_d_status);
			g_d_status = NULL;
			return -1;
		}
	}

	/*
	 * Stage through a pinned buffer we own. An async copy requires its source
	 * to stay valid until the copy completes, which a caller-supplied pointer
	 * cannot promise.
	 */
	for (int i = 0; i < len; i++)
		g_h_status[i] = status[i];
	g_h_status[len] = '\0';

	/* Copy len+1 bytes so the device buffer is NUL terminated */
	if (cudaMemcpyAsync(g_d_status, g_h_status, (size_t)len + 1,
			    cudaMemcpyHostToDevice, stream) != cudaSuccess) {
		fprintf(stderr, "[GPU_PIPE] status H2D failed\n");
		return -1;
	}

	*out_len = len;
	return 0;
}

extern "C" int gazsi_launch_tokenize(const struct inference_payload_plane *plane,
				     const struct gazsi_batch_item *items,
				     int batch_size,
				     void *d_input_ids,
				     void *d_attention_mask,
				     int32_t *d_token_counts,
				     int seq_len,
				     void *stream)
{
	if (!plane || !items || !d_input_ids || !d_attention_mask || !d_token_counts ||
	    batch_size <= 0 || seq_len < 2)
		return -1;

	int threads = seq_len < 128 ? seq_len : 128;

	gazsi_tokenize_kernel<<<batch_size, threads, 0, (cudaStream_t)stream>>>(
		plane, items, batch_size,
		(int64_t *)d_input_ids, (int64_t *)d_attention_mask,
		d_token_counts, seq_len);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "[GPU_PIPE] tokenize launch failed: %s\n", cudaGetErrorString(err));
		return -1;
	}
	return 0;
}

extern "C" int gazsi_launch_format_results(struct inference_ring_buffer *ring,
					   struct inference_payload_plane *plane,
					   const struct gazsi_batch_item *items,
					   int batch_size,
					   const float *d_output,
					   const int32_t *d_token_counts,
					   int embed_dim,
					   int out_stride,
					   long elapsed_us,
					   void *stream)
{
	if (!ring || !plane || !items || !d_output || !d_token_counts ||
	    batch_size <= 0 || out_stride <= 0)
		return -1;

	/*
	 * The response samples emb[0..2]. A model whose output dimension is
	 * smaller than that would make the kernel read past the tensor, so
	 * refuse rather than read out of bounds.
	 */
	if (embed_dim < 3) {
		fprintf(stderr,
			"[GPU_PIPE] embedding_dim %d < 3: response samples 3 components, refusing\n",
			embed_dim);
		return -1;
	}
	if (out_stride < embed_dim) {
		fprintf(stderr, "[GPU_PIPE] out_stride %d < embedding_dim %d\n",
			out_stride, embed_dim);
		return -1;
	}

	gazsi_format_kernel<<<1, batch_size, 0, (cudaStream_t)stream>>>(
		ring, plane, items, batch_size,
		d_output, d_token_counts, out_stride, elapsed_us);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "[GPU_PIPE] format launch failed: %s\n", cudaGetErrorString(err));
		return -1;
	}
	return 0;
}

extern "C" int gazsi_launch_static_response(struct inference_ring_buffer *ring,
					    struct inference_payload_plane *plane,
					    const struct gazsi_batch_item *items,
					    int batch_size,
					    const char *status,
					    void *stream)
{
	if (!ring || !plane || !items || !status || batch_size <= 0)
		return -1;

	int status_len = 0;
	if (gazsi_stage_status(status, &status_len, (cudaStream_t)stream) != 0)
		return -1;

	gazsi_static_response_kernel<<<1, batch_size, 0, (cudaStream_t)stream>>>(
		ring, plane, items, batch_size, g_d_status, status_len);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "[GPU_PIPE] static response launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}
	return 0;
}

/*
 * Host reference tokenizer — test oracle only, never on a request path.
 */
extern "C" int gazsi_reference_tokenize(const char *text, int64_t *input_ids,
					int64_t *attention_mask, int seq_len)
{
	for (int i = 0; i < seq_len; i++) {
		input_ids[i] = 0;
		attention_mask[i] = 0;
	}

	input_ids[0] = GAZSI_TOKEN_CLS;
	attention_mask[0] = 1;

	if (!text || text[0] == '\0') {
		input_ids[1] = GAZSI_TOKEN_SEP;
		attention_mask[1] = 1;
		return 2;
	}

	int token_pos = 1;
	uint32_t hash = 0;
	bool in_word = false;

	for (const char *p = text; *p && token_pos < seq_len - 1; p++) {
		if (*p == ' ') {
			if (in_word) {
				input_ids[token_pos] = GAZSI_TOKEN_BASE + (hash % GAZSI_TOKEN_MODULO);
				attention_mask[token_pos] = 1;
				token_pos++;
				hash = 0;
				in_word = false;
			}
		} else {
			hash = hash * 31u + (uint32_t)(uint8_t)*p;
			in_word = true;
		}
	}

	if (in_word && token_pos < seq_len - 1) {
		input_ids[token_pos] = GAZSI_TOKEN_BASE + (hash % GAZSI_TOKEN_MODULO);
		attention_mask[token_pos] = 1;
		token_pos++;
	}

	input_ids[token_pos] = GAZSI_TOKEN_SEP;
	attention_mask[token_pos] = 1;
	token_pos++;

	return token_pos;
}
