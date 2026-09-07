/*
 * GPU Packet Processing - Device-Resident Preprocessing and Result Assembly
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Tokenization and HTTP response-body formatting executed by CUDA kernels on
 * payload that never leaves device memory.
 *
 * These are the two stages that used to force payload onto the CPU:
 *   - tokenization  read request text on the host, wrote int64 tensors to
 *                   pinned host memory, then copied H2D.
 *   - result format read embeddings copied D2H, snprintf'd JSON on the host,
 *                   then wrote it back into host-mapped memory for TX to pull.
 *
 * Both now run on the GPU against the device-resident payload plane and the
 * TensorRT device tensors directly, so the request/result bytes make zero
 * PCIe crossings. The host only launches the kernels.
 */

#ifndef GAZSI_GPU_PIPELINE_H
#define GAZSI_GPU_PIPELINE_H

#include <stdint.h>
#include "ring_buffer.h"

/* Token sentinels and vocabulary mapping — must match the model's engine */
#define GAZSI_TOKEN_CLS      101
#define GAZSI_TOKEN_SEP      102
#define GAZSI_TOKEN_BASE     1000
#define GAZSI_TOKEN_MODULO   20000
#define GAZSI_GPT2_VOCAB_SIZE 50257
#define GAZSI_GPT2_EOT_ID     50256

/*
 * One batch element. Pure metadata: identifies where in the device payload
 * plane a request lives. Built by the host from control-plane state.
 */
struct gazsi_batch_item {
	uint32_t slot_index;   /* index into inference_payload_plane.slots[] */
	uint32_t record_index; /* which packed record within that slot */
	uint32_t req_offset;   /* request byte offset within request[] */
	uint32_t req_length;   /* request byte length */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tokenize a batch straight into the TensorRT input tensors.
 *
 * Reads request bytes from the device payload plane and writes input_ids,
 * attention_mask (int64, batch_size x seq_len) and token_counts (int32) all in
 * device memory. Bit-identical to the host tokenizer it replaces, so token
 * streams and therefore model outputs are unchanged.
 *
 * Asynchronous on `stream`.
 *
 * @return 0 on success, -1 on invalid arguments or launch failure
 */
int gazsi_launch_tokenize(const struct inference_payload_plane *plane,
			  const struct gazsi_batch_item *items,
			  int batch_size,
			  void *d_input_ids,
			  void *d_attention_mask,
			  int32_t *d_token_counts,
			  int seq_len,
			  void *stream);

/*
 * Format inference results into the device-resident response arena.
 *
 * Reads the TensorRT output tensor and the request text (both device memory)
 * and writes the JSON response body plus its length. Response length lands in
 * the host-mapped control plane (metadata) so the TX kernel knows how many
 * bytes to transmit; the bytes themselves stay on the device.
 *
 * Asynchronous on `stream`.
 *
 * @param d_output      TensorRT output tensor (device)
 * @param out_stride    floats between consecutive batch elements in d_output
 * @param embed_dim     embedding dimension (only the first 3 are sampled)
 * @param elapsed_us    host-measured batch inference time, for the JSON field
 * @return 0 on success, -1 on invalid arguments or launch failure
 */
int gazsi_launch_format_results(struct inference_ring_buffer *ring,
			       struct inference_payload_plane *plane,
			       const struct gazsi_batch_item *items,
			       int batch_size,
			       const float *d_output,
			       const int32_t *d_token_counts,
			       int embed_dim,
			       int out_stride,
			       long elapsed_us,
			       void *stream);

/*
 * Format a GPT-2 language-model result into the device-resident response.
 *
 * Requests for this path use the wire form `ids:<id0>,<id1>,...`, where the
 * IDs were produced by the official GPT-2 tokenizer on the client. TensorRT
 * returns one next-token ID per batch element. The decode table contains
 * JSON-escaped token text, so the GPU can copy a readable token into the HTTP
 * body without moving request or result payload through host memory.
 */
int gazsi_launch_format_next_tokens(struct inference_ring_buffer *ring,
				    struct inference_payload_plane *plane,
				    const struct gazsi_batch_item *items,
				    int batch_size,
				    const int64_t *d_next_token_ids,
				    const int32_t *d_token_counts,
				    const uint32_t *d_decode_offsets,
				    uint32_t decode_vocab_size,
				    const char *d_decode_bytes,
				    uint32_t decode_bytes_size,
				    long elapsed_us,
				    void *stream);

/*
 * Write a fixed status body into a slot's response arena from the device.
 *
 * Used for the no-model fallback and error paths. Exists so that even the
 * degraded paths never have the CPU write payload bytes.
 *
 * Asynchronous on `stream`.
 *
 * @return 0 on success, -1 on invalid arguments or launch failure
 */
int gazsi_launch_static_response(struct inference_ring_buffer *ring,
				 struct inference_payload_plane *plane,
				 const struct gazsi_batch_item *items,
				 int batch_size,
				 const char *status,
				 void *stream);

/*
 * Host reference tokenizer, exposed for the equivalence test only.
 *
 * This is the original CPU implementation kept as the test oracle. It is NOT
 * used on any request path — nothing in the serving path may read payload on
 * the host. Operates on caller-provided buffers, never on ring payload.
 */
int gazsi_reference_tokenize(const char *text, int64_t *input_ids,
			     int64_t *attention_mask, int seq_len);

#ifdef __cplusplus
}
#endif

#endif /* GAZSI_GPU_PIPELINE_H */
