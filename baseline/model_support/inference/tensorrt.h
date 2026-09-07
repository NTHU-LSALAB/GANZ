/*
 * GPU Packet Processing - TensorRT Inference Runner Header
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#ifdef __cplusplus
#include <string>
// Use a forward declaration (void*) to hide TensorRT implementation details
// from other C++ files that include this header. This avoids forcing them
// to include heavy TensorRT headers.
using TensorRT_Model_t = void;
#else
// For C compilation, just use void*
typedef void TensorRT_Model_t;
#endif

/**
 * @brief Initializes pre-allocated GPU buffers for TensorRT inference
 * @param gpu_id CUDA device ID to use
 * @return 0 on success, -1 on failure
 */
#ifdef __cplusplus
extern "C" {
#endif

int init_tensorrt_gpu_buffers(int gpu_id);

/**
 * @brief Loads a TensorRT engine from a file and prepares it for inference.
 * @param engine_path Path to the .engine file.
 * @return A handle to the TensorRT model, or nullptr on failure.
 */
TensorRT_Model_t* load_tensorrt_engine(const char* engine_path);

/**
 * @brief Warmup-only inference over a caller-supplied literal string.
 *
 * Exercises the engine at startup so the first real request does not pay cold
 * start. Tokenizes on the host, which is acceptable here and ONLY here: the
 * input is a compile-time literal, not request payload. No request or result
 * payload byte ever reaches this function.
 *
 * @param model A handle to the loaded TensorRT model (with context pool).
 * @param context_id ID of the context to use (0 to NUM_CONTEXTS-1).
 * @param text Warmup text literal.
 * @param embeddings Output embeddings array.
 * @param token_count Output token count.
 */
void simple_tokenize_and_infer_with_context(TensorRT_Model_t* model, int context_id, const char* text, float* embeddings, int* token_count);

/* Forward declarations for the device-resident data plane */
struct inference_ring_buffer;
struct inference_payload_plane;
struct gazsi_batch_item;

/**
 * Outcome of a device-resident batch.
 *
 * The distinction is about whether the NEXT batch can possibly succeed, and it
 * decides who gets to answer the client:
 *
 *   BATCH   the descriptors for this batch were rejected. The engine, the
 *           tensors and the payload plane are all intact, so serving continues
 *           and this batch alone is answered with an error body.
 *
 *   FATAL   CUDA, TensorRT, device residency, tensor binding or CUDA graph
 *           failed. None of those recover on their own: the context is sticky
 *           after a CUDA error, a mis-bound or cross-device tensor stays
 *           mis-bound, and a broken graph exec stays broken. Continuing would
 *           hand every subsequent client an "inference_failed" body forever
 *           while the server still looked healthy. Stop serving instead.
 *
 * Values are negative so the historical `rc != 0` failure test still holds for
 * any caller that does not care about the distinction.
 */
#define GAZSI_INFER_OK        0
#define GAZSI_INFER_ERR_BATCH (-1)
#define GAZSI_INFER_ERR_FATAL (-2)

/* True if rc means the pipeline itself is broken, not just this batch. */
static inline int gazsi_infer_is_fatal(int rc) { return rc == GAZSI_INFER_ERR_FATAL; }

/**
 * @brief Device-resident batch inference: tokenize, infer and format on the GPU.
 *
 * The paper's data path. Given a batch of descriptors pointing into the
 * device-resident payload plane, this:
 *   1. launches the GPU tokenizer, writing the TensorRT input tensors in place
 *   2. triggers TensorRT execution (CUDA graph if captured, else enqueueV3)
 *   3. synchronises, so the inference output is visible device-wide
 *   4. launches the GPU formatter, writing response bytes into the payload plane
 *   5. synchronises, establishing the ordering the caller needs before it may
 *      publish RESULT_READY
 *
 * No request or result payload byte is read, written or copied by the host.
 * The only host involvement is launching work and reading the wall clock.
 *
 * CONTROL-PLANE LIMITATION: step 2 must be issued from the host. TensorRT
 * exposes no device-side execution entry point, so a persistent GPU kernel
 * cannot start an inference itself. This is the documented hybrid mode: the
 * host triggers execution while all data and tensors stay device-resident.
 *
 * @param model A handle to the loaded TensorRT model.
 * @param ring Control plane (host-mapped), for publishing response lengths.
 * @param plane Device-resident payload plane.
 * @param d_items Device-readable batch descriptors (batch_size elements).
 * @param batch_size Number of records in the batch (1-8).
 * @param out_elapsed_us Measured tokenize+inference time, microseconds.
 * @return GAZSI_INFER_OK, GAZSI_INFER_ERR_BATCH (this batch is unusable,
 *         serving continues) or GAZSI_INFER_ERR_FATAL (stop serving).
 */
int batch_infer_device_resident(TensorRT_Model_t* model,
                                struct inference_ring_buffer* ring,
                                struct inference_payload_plane* plane,
                                const struct gazsi_batch_item* d_items,
                                int batch_size,
                                long* out_elapsed_us);

/**
 * @brief Pre-capture CUDA Graphs for device-resident batch inference.
 *
 * Captures TensorRT execution alone (one graph per batch size), replacing the
 * ~50-100 internal kernel launches inside enqueueV3 with a single
 * cudaGraphLaunch. Unlike a host-path graph there are no memcpy nodes: inputs
 * are produced in place by the tokenizer kernel and outputs are consumed in
 * place by the formatter kernel, so nothing crosses PCIe.
 *
 * Must be called AFTER load_tensorrt_engine() and warmup.
 *
 * @param model A handle to the loaded TensorRT model.
 * @return 0 on success, -1 on failure (falls back to enqueueV3 automatically)
 */
int init_cuda_graphs_device_resident(TensorRT_Model_t* model);

#ifdef __cplusplus
}
#endif
