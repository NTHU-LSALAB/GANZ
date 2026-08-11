/*
 * GPU Packet Processing - TensorRT Inference Runner
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorrt.h"
#include "ring_buffer.h"
#include "gpu_pipeline.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <sstream>
#include <cstring>  // for memset, memcpy (pinned memory optimization)
#include "NvInfer.h"
#include <cuda_runtime_api.h>

// Buffer allocation uses MAX_OUTPUT_DIM (vocab_size for decoders); runtime uses model->embedding_dim
#define SEQUENCE_LENGTH 128
#define MAX_OUTPUT_DIM 32768
#define NUM_CONTEXTS 4

// Context Pool: Pre-allocated GPU buffers + Pinned Host Memory per execution context
struct ContextBuffers {
    // GPU device memory
    void *d_input_ids;
    void *d_attention_mask;
    void *d_token_type_ids;
    void *d_output;
    void *d_pooler_output;
    void *d_position_ids;

    // Pinned host memory (for fast DMA transfer)
    int64_t *h_input_ids;
    int64_t *h_attention_mask;
    float *h_output;
    int64_t *h_position_ids;

    // Dedicated CUDA stream for async operations
    cudaStream_t stream;
};

static std::vector<ContextBuffers> g_context_buffers(NUM_CONTEXTS);


extern "C" int init_tensorrt_gpu_buffers(int gpu_id) {
    cudaSetDevice(gpu_id);
    size_t input_size = SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = SEQUENCE_LENGTH * MAX_OUTPUT_DIM * sizeof(float);
    size_t pooler_size = MAX_OUTPUT_DIM * sizeof(float);

    // Allocate GPU buffers + Pinned Host Memory for Context Pool
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        // 1. Allocate GPU device memory
        if (cudaMalloc(&g_context_buffers[i].d_input_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_attention_mask, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_token_type_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_output, output_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_pooler_output, pooler_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_position_ids, sizeof(int64_t)) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate GPU buffers for context %d\n", i);
            return -1;
        }

        // 2. Allocate Pinned Host Memory (for fast DMA)
        if (cudaHostAlloc(&g_context_buffers[i].h_input_ids, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_attention_mask, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_output, MAX_OUTPUT_DIM * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_position_ids, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate pinned host memory for context %d\n", i);
            return -1;
        }

        // 3. Create dedicated CUDA stream
        if (cudaStreamCreateWithFlags(&g_context_buffers[i].stream, cudaStreamNonBlocking) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to create CUDA stream for context %d\n", i);
            return -1;
        }

        // Initialize token_type_ids to 0
        if (cudaMemset(g_context_buffers[i].d_token_type_ids, 0, input_size) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to initialize token_type_ids for context %d\n", i);
            return -1;
        }

        fprintf(stderr, "[CONTEXT_POOL] Context %d: GPU buffers + pinned memory + stream (input=%zu, output=%zu)\n",
                i, input_size, output_size);
    }

    fprintf(stderr, "[TensorRT] GPU buffers pre-allocated (%d contexts)\n", NUM_CONTEXTS);
    return 0;
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Simple logger for TensorRT messages
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

/*
 * Warmup Tokenization
 *
 * Simple word-based tokenization for MiniLM model
 * Format: [CLS] word1 word2 ... wordN [SEP] [PAD] [PAD] ...
 *
 * WARMUP PATH ONLY. Request tokenization happens on the GPU against
 * device-resident payload (see gazsi_launch_tokenize); this host copy survives
 * solely to warm the engine on startup literals. Keep the two in step: the
 * device kernel is a transliteration of this function, and
 * tests/test_gpu_pipeline.cu asserts they agree byte for byte.
 */

/*
 * tokenize_text - Convert text to token IDs and attention mask
 *
 * @text:           Input text
 * @input_ids:      Output token IDs (must be pre-allocated SEQUENCE_LENGTH)
 * @attention_mask: Output attention mask (must be pre-allocated SEQUENCE_LENGTH)
 *
 * Returns: Actual token count
 */
static int tokenize_text(const char* text,
                         int64_t* input_ids,
                         int64_t* attention_mask) {
    memset(input_ids, 0, SEQUENCE_LENGTH * sizeof(int64_t));
    memset(attention_mask, 0, SEQUENCE_LENGTH * sizeof(int64_t));

    input_ids[0] = 101;
    attention_mask[0] = 1;

    if (!text || text[0] == '\0') {
        input_ids[1] = 102;
        attention_mask[1] = 1;
        return 2;
    }

    int token_pos = 1;
    uint32_t hash = 0;
    bool in_word = false;

    for (const char *p = text; *p && token_pos < SEQUENCE_LENGTH - 1; p++) {
        if (*p == ' ') {
            if (in_word) {
                input_ids[token_pos] = 1000 + (hash % 20000);
                attention_mask[token_pos] = 1;
                token_pos++;
                hash = 0;
                in_word = false;
            }
        } else {
            hash = hash * 31 + (uint8_t)*p;
            in_word = true;
        }
    }

    if (in_word && token_pos < SEQUENCE_LENGTH - 1) {
        input_ids[token_pos] = 1000 + (hash % 20000);
        attention_mask[token_pos] = 1;
        token_pos++;
    }

    input_ids[token_pos] = 102;
    attention_mask[token_pos] = 1;
    token_pos++;

    return token_pos;
}

// Wrapper struct to hold all TensorRT objects
struct TensorRT_Context {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;  // Legacy single context
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;  // Context Pool

    // Dynamic tensor names (detected from engine)
    std::string output_tensor_name;
    std::string pooler_tensor_name;
    std::string position_ids_name;
    bool has_token_type_ids;
    bool has_pooler_output;
    bool has_position_ids;
    bool output_has_seq_dim;
    int embedding_dim;
};

extern "C" TensorRT_Model_t* load_tensorrt_engine(const char* engine_path) {
    Logger logger;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << engine_path << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_blob(size);
    file.read(engine_blob.data(), size);
    file.close();

    auto model = new TensorRT_Context();
    model->runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!model->runtime) {
        std::cerr << "Error: Failed to create TensorRT Runtime." << std::endl;
        delete model;
        return nullptr;
    }

    model->engine.reset(model->runtime->deserializeCudaEngine(engine_blob.data(), size));
    if (!model->engine) {
        std::cerr << "Error: Failed to deserialize TensorRT Engine." << std::endl;
        delete model;
        return nullptr;
    }

    // Create legacy single context (backward compatibility)
    model->context.reset(model->engine->createExecutionContext());
    if (!model->context) {
        std::cerr << "Error: Failed to create TensorRT Execution Context." << std::endl;
        delete model;
        return nullptr;
    }

    // Detect tensor names from engine
    model->has_token_type_ids = false;
    model->has_pooler_output = false;
    model->has_position_ids = false;
    model->output_has_seq_dim = true;
    model->output_tensor_name = "output";
    model->pooler_tensor_name = "";
    model->position_ids_name = "";
    model->embedding_dim = 768;

    int num_tensors = model->engine->getNbIOTensors();
    std::cout << "[TensorRT] Detecting tensors from engine (" << num_tensors << " tensors):" << std::endl;

    for (int i = 0; i < num_tensors; i++) {
        const char* name = model->engine->getIOTensorName(i);
        auto mode = model->engine->getTensorIOMode(name);
        auto shape = model->engine->getTensorShape(name);

        std::cout << "  [" << i << "] " << name << " (";
        std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "): shape=(";
        for (int d = 0; d < shape.nbDims; d++) {
            if (d > 0) std::cout << ", ";
            std::cout << shape.d[d];
        }
        std::cout << ")" << std::endl;

        // Check for token_type_ids
        if (strcmp(name, "token_type_ids") == 0) {
            model->has_token_type_ids = true;
        }

        // Check for position_ids (LLaMA rotary embedding: scalar or 1D input)
        if (mode == nvinfer1::TensorIOMode::kINPUT &&
            strcmp(name, "input_ids") != 0 &&
            strcmp(name, "attention_mask") != 0 &&
            strcmp(name, "token_type_ids") != 0) {
            model->has_position_ids = true;
            model->position_ids_name = name;
        }

        // Check for output tensor name
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (strcmp(name, "last_hidden_state") == 0) {
                model->output_tensor_name = "last_hidden_state";
            } else if (strcmp(name, "output") == 0) {
                model->output_tensor_name = "output";
            } else {
                // Secondary output (pooler output like "1895", "1001")
                model->has_pooler_output = true;
                model->pooler_tensor_name = name;
            }
            // Get output dim from shape (last dimension)
            if (shape.nbDims >= 2) {
                int dim = shape.d[shape.nbDims - 1];
                if (dim > 0) {
                    model->embedding_dim = dim;
                }
            }
            // Encoder: (batch, seq, dim) → 3D; Decoder: (batch, vocab) → 2D
            model->output_has_seq_dim = (shape.nbDims >= 3);
        }
    }

    std::cout << "[TensorRT] Model config: output_tensor=" << model->output_tensor_name
              << ", has_token_type_ids=" << (model->has_token_type_ids ? "yes" : "no")
              << ", has_pooler=" << (model->has_pooler_output ? model->pooler_tensor_name : "no")
              << ", has_position_ids=" << (model->has_position_ids ? model->position_ids_name : "no")
              << ", embedding_dim=" << model->embedding_dim << std::endl;

    // Create Context Pool
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        auto ctx = model->engine->createExecutionContext();
        if (!ctx) {
            std::cerr << "[CONTEXT_POOL] Error: Failed to create execution context " << i << std::endl;
            delete model;
            return nullptr;
        }
        model->contexts.emplace_back(ctx);
        std::cout << "[CONTEXT_POOL] Created context " << i << std::endl;
    }

    return model;
}


// Context Pool version - use specified context with dedicated buffers
extern "C" void simple_tokenize_and_infer_with_context(TensorRT_Model_t* model_ptr, int context_id,
                                                         const char* text, float* embeddings, int* token_count) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    // Validate parameters
    if (!model || !text || !embeddings || !token_count) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Invalid parameters\n");
        if (token_count) *token_count = 0;
        return;
    }

    if (context_id < 0 || context_id >= NUM_CONTEXTS) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Invalid context_id %d (must be 0-%d)\n", context_id, NUM_CONTEXTS-1);
        *token_count = 0;
        return;
    }

    if (model->contexts.empty() || !model->contexts[context_id]) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Context %d not initialized\n", context_id);
        *token_count = 0;
        return;
    }

    try {
        // Use context pool's dedicated buffers (with pinned memory)
        auto& buffers = g_context_buffers[context_id];
        auto* context = model->contexts[context_id].get();

        // Tokenize directly into pinned memory
        *token_count = tokenize_text(text, buffers.h_input_ids, buffers.h_attention_mask);

        size_t input_size = SEQUENCE_LENGTH * sizeof(int64_t);
        int embed_dim = model->embedding_dim;
        size_t output_size = embed_dim * sizeof(float);

        cudaStream_t stream = buffers.stream;

        // Async copy to GPU (pinned memory → device memory via DMA)
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_input_ids, buffers.h_input_ids, input_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_attention_mask, buffers.h_attention_mask, input_size, cudaMemcpyHostToDevice, stream));

        // Set input shapes
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = 1;  // batch size
        input_dims.d[1] = SEQUENCE_LENGTH;

        context->setInputShape("input_ids", input_dims);
        context->setInputShape("attention_mask", input_dims);
        if (model->has_token_type_ids) {
            context->setInputShape("token_type_ids", input_dims);
        }

        // Bind tensors
        context->setTensorAddress("input_ids", buffers.d_input_ids);
        context->setTensorAddress("attention_mask", buffers.d_attention_mask);
        if (model->has_token_type_ids) {
            context->setTensorAddress("token_type_ids", buffers.d_token_type_ids);
        }
        if (model->has_position_ids) {
            buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
            CUDA_CHECK(cudaMemcpyAsync(buffers.d_position_ids, buffers.h_position_ids,
                                        sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            context->setTensorAddress(model->position_ids_name.c_str(), buffers.d_position_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), buffers.d_output);
        if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
            context->setTensorAddress(model->pooler_tensor_name.c_str(), buffers.d_pooler_output);
        }

        // Execute inference (async on same stream)
        context->enqueueV3(stream);

        // Async copy result back (device memory → pinned memory via DMA)
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_output, buffers.d_output, output_size, cudaMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy from pinned memory to user's buffer (fast memcpy)
        memcpy(embeddings, buffers.h_output, output_size);

    } catch (const std::exception& e) {
        fprintf(stderr, "[CONTEXT_POOL] Error in context %d: %s\n", context_id, e.what());
        *token_count = 0;
        for (int i = 0; i < model->embedding_dim; i++) {
            embeddings[i] = 0.0f;
        }
    }
}

/*
 * Device-Resident Batch Inference
 *
 * Combines N inferences into one batch, with tokenization and result
 * formatting executed by CUDA kernels on payload that never leaves device
 * memory. The host contributes exactly three things: kernel launches, the
 * TensorRT execution trigger, and a wall-clock reading.
 *
 * There are deliberately NO pinned host staging buffers for input ids,
 * attention mask or output embeddings here. Their absence is what guarantees
 * no request or result payload crosses PCIe.
 */

#define MAX_BATCH_SIZE 8

struct BatchBuffers {
    // GPU device memory (batch_size × sequence_length)
    void *d_input_ids;
    void *d_attention_mask;
    void *d_token_type_ids;
    void *d_output;
    void *d_pooler_output;
    void *d_position_ids;

    // Token counts produced by the GPU tokenizer, consumed by the GPU formatter
    int32_t *d_token_counts;

    // Pinned host memory for position_ids only: a scalar control input
    // (sequence length), not request or result payload.
    int64_t *h_position_ids;

    // Dedicated stream
    cudaStream_t stream;

    bool initialized;

    // CUDA Graph: pre-captured TensorRT execution per batch size.
    // Contains enqueueV3 only — no memcpy nodes, by design.
    cudaGraphExec_t graph_exec[MAX_BATCH_SIZE + 1];
    bool graphs_ready;
};

static BatchBuffers g_batch_buffers = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, false,
                                        {}, false};

// Initialize Batch Buffers (called once)
static void init_batch_buffers() {
    if (g_batch_buffers.initialized) return;

    size_t input_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * MAX_OUTPUT_DIM * sizeof(float);
    size_t pooler_size = MAX_BATCH_SIZE * MAX_OUTPUT_DIM * sizeof(float);

    // GPU buffers
    if (cudaMalloc(&g_batch_buffers.d_input_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_attention_mask, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_token_type_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_output, output_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_pooler_output, pooler_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_position_ids, sizeof(int64_t)) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_token_counts, MAX_BATCH_SIZE * sizeof(int32_t)) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to allocate GPU buffers\n");
        return;
    }

    // Initialize token_type_ids to 0
    cudaMemset(g_batch_buffers.d_token_type_ids, 0, input_size);
    cudaMemset(g_batch_buffers.d_token_counts, 0, MAX_BATCH_SIZE * sizeof(int32_t));

    // Pinned host memory for the position_ids scalar only
    if (cudaHostAlloc(&g_batch_buffers.h_position_ids, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to allocate pinned position_ids\n");
        return;
    }

    // Stream
    if (cudaStreamCreateWithFlags(&g_batch_buffers.stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to create stream\n");
        return;
    }

    g_batch_buffers.initialized = true;
    fprintf(stderr, "[BATCH] Device-resident batch buffers initialized (max_batch=%d)\n", MAX_BATCH_SIZE);
}

// Bind engine tensors to the fixed device buffers for a given batch size.
// Shapes and addresses are CPU-side context state, so this is done outside
// any graph capture.
static bool bind_tensors(TensorRT_Context* model, nvinfer1::IExecutionContext* context, int bs) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 2;
    input_dims.d[0] = bs;
    input_dims.d[1] = SEQUENCE_LENGTH;

    if (!context->setInputShape("input_ids", input_dims)) return false;
    if (!context->setInputShape("attention_mask", input_dims)) return false;
    if (model->has_token_type_ids) {
        if (!context->setInputShape("token_type_ids", input_dims)) return false;
    }

    context->setTensorAddress("input_ids", g_batch_buffers.d_input_ids);
    context->setTensorAddress("attention_mask", g_batch_buffers.d_attention_mask);
    if (model->has_token_type_ids) {
        context->setTensorAddress("token_type_ids", g_batch_buffers.d_token_type_ids);
    }
    if (model->has_position_ids) {
        g_batch_buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
        if (cudaMemcpy(g_batch_buffers.d_position_ids, g_batch_buffers.h_position_ids,
                       sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        context->setTensorAddress(model->position_ids_name.c_str(), g_batch_buffers.d_position_ids);
    }
    context->setTensorAddress(model->output_tensor_name.c_str(), g_batch_buffers.d_output);
    if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
        context->setTensorAddress(model->pooler_tensor_name.c_str(), g_batch_buffers.d_pooler_output);
    }
    return true;
}

// Floats between consecutive batch elements in the output tensor.
static inline int output_stride(const TensorRT_Context* model) {
    return model->output_has_seq_dim
        ? SEQUENCE_LENGTH * model->embedding_dim
        : model->embedding_dim;
}

/*
 * CUDA Graph Capture for Device-Resident Batch Inference
 *
 * Captures TensorRT execution only, one graph per batch size. Input tensors
 * are filled in place by the tokenizer kernel and the output tensor is read in
 * place by the formatter kernel, both stream-ordered around the graph launch,
 * so the graph needs no memcpy nodes at all.
 *
 * Prerequisite: init_batch_buffers(). Buffer addresses are fixed at init time
 * and baked into each graph.
 */
extern "C" int init_cuda_graphs_device_resident(TensorRT_Model_t* model_ptr) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    if (!model || model->contexts.empty() || !model->contexts[0]) {
        fprintf(stderr, "[CUDA_GRAPH] Error: model or context not ready\n");
        return -1;
    }

    init_batch_buffers();
    if (!g_batch_buffers.initialized) {
        fprintf(stderr, "[CUDA_GRAPH] Error: batch buffers not initialized\n");
        return -1;
    }

    auto* context = model->contexts[0].get();
    cudaStream_t stream = g_batch_buffers.stream;

    // Determine max batch from engine profile
    auto max_shape = model->engine->getProfileShape("input_ids", 0, nvinfer1::OptProfileSelector::kMAX);
    int engine_max_batch = (max_shape.nbDims >= 1 && max_shape.d[0] > 0) ? max_shape.d[0] : 1;
    int graph_max_batch = std::min(MAX_BATCH_SIZE, engine_max_batch);

    fprintf(stderr, "[CUDA_GRAPH] Engine max batch=%d, capturing device-resident graphs for bs 1-%d\n",
            engine_max_batch, graph_max_batch);

    for (int bs = 1; bs <= graph_max_batch; bs++) {
        if (!bind_tensors(model, context, bs)) {
            fprintf(stderr, "[CUDA_GRAPH] Error: tensor binding failed for bs=%d\n", bs);
            return -1;
        }

        // Warm-up run: let TensorRT initialize internal cuBLAS handles etc.
        if (!context->enqueueV3(stream)) {
            fprintf(stderr, "[CUDA_GRAPH] Error: warm-up enqueueV3 failed for bs=%d\n", bs);
            return -1;
        }
        cudaStreamSynchronize(stream);

        cudaGraph_t graph;
        cudaError_t err;

        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaStreamBeginCapture failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        // Inference only — inputs and outputs are already where they belong
        context->enqueueV3(stream);

        err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaStreamEndCapture failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        err = cudaGraphInstantiate(&g_batch_buffers.graph_exec[bs], graph, NULL, NULL, 0);
        cudaGraphDestroy(graph);  // Original graph no longer needed
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaGraphInstantiate failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        fprintf(stderr, "[CUDA_GRAPH] Captured inference-only graph for batch_size=%d\n", bs);
    }

    g_batch_buffers.graphs_ready = true;
    fprintf(stderr, "[CUDA_GRAPH] All %d device-resident graphs captured successfully\n", graph_max_batch);
    return 0;
}

static inline uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/*
 * The device-resident path has one hard requirement the old host path did not:
 * the tokenizer kernel reads the payload plane and writes the engine's input
 * tensors, so both must be reachable from one device. When payload staging went
 * through host memory a cross-device split merely ran slowly; now it would read
 * an address that is not mapped on the executing device.
 *
 * Checked once and reported loudly rather than left to corrupt silently.
 */
static bool check_same_device(const struct inference_payload_plane *plane) {
    static int verdict = -1;
    if (verdict >= 0)
        return verdict == 1;

    cudaPointerAttributes pa, ta;
    if (cudaPointerGetAttributes(&pa, (const void *)plane) != cudaSuccess ||
        cudaPointerGetAttributes(&ta, g_batch_buffers.d_input_ids) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Error: cannot query device residency of payload/tensors\n");
        verdict = 0;
        return false;
    }

    if (pa.device != ta.device) {
        fprintf(stderr,
                "[BATCH] FATAL: payload plane is on CUDA device %d but TensorRT input "
                "tensors are on device %d. The device-resident path needs both on one "
                "device; run DOCA and TensorRT on the same GPU.\n",
                pa.device, ta.device);
        verdict = 0;
        return false;
    }

    fprintf(stderr, "[BATCH] Payload plane and TensorRT tensors both on CUDA device %d\n",
            pa.device);
    verdict = 1;
    return true;
}

extern "C" int batch_infer_device_resident(TensorRT_Model_t* model_ptr,
                                           struct inference_ring_buffer* ring,
                                           struct inference_payload_plane* plane,
                                           const struct gazsi_batch_item* d_items,
                                           int batch_size,
                                           long* out_elapsed_us) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    /*
     * A null plane/ring/model is a wiring mistake, not a bad request: it is
     * identical on every subsequent batch, so it is fatal. Only batch_size is
     * per-batch data and therefore recoverable.
     */
    if (!model || !ring || !plane || !d_items || !out_elapsed_us) {
        fprintf(stderr, "[BATCH] FATAL: Invalid parameters\n");
        return GAZSI_INFER_ERR_FATAL;
    }
    if (batch_size < 1 || batch_size > MAX_BATCH_SIZE) {
        fprintf(stderr, "[BATCH] Error: Invalid batch_size %d (must be 1-%d)\n", batch_size, MAX_BATCH_SIZE);
        return GAZSI_INFER_ERR_BATCH;
    }

    init_batch_buffers();
    if (!g_batch_buffers.initialized) {
        fprintf(stderr, "[BATCH] FATAL: Batch buffers not initialized (CUDA allocation)\n");
        return GAZSI_INFER_ERR_FATAL;
    }
    if (model->contexts.empty() || !model->contexts[0]) {
        fprintf(stderr, "[BATCH] FATAL: Context 0 not available\n");
        return GAZSI_INFER_ERR_FATAL;
    }
    if (!check_same_device(plane))
        return GAZSI_INFER_ERR_FATAL;

    try {
        auto* context = model->contexts[0].get();
        cudaStream_t stream = g_batch_buffers.stream;

        uint64_t t_start = monotonic_ns();

        /* 1. Tokenize device-resident request bytes straight into the engine inputs */
        if (gazsi_launch_tokenize(plane, d_items, batch_size,
                                  g_batch_buffers.d_input_ids,
                                  g_batch_buffers.d_attention_mask,
                                  g_batch_buffers.d_token_counts,
                                  SEQUENCE_LENGTH, stream) != 0) {
            fprintf(stderr, "[BATCH] FATAL: tokenizer launch failed\n");
            return GAZSI_INFER_ERR_FATAL;
        }

        /*
         * 2. Trigger TensorRT execution.
         *
         * This host call is the one unavoidable control-plane crossing:
         * TensorRT has no device-side execution API, so a persistent GPU
         * kernel cannot start an inference. Only the trigger is on the host —
         * every tensor involved stays in device memory.
         */
        if (g_batch_buffers.graphs_ready && g_batch_buffers.graph_exec[batch_size]) {
            CUDA_CHECK(cudaGraphLaunch(g_batch_buffers.graph_exec[batch_size], stream));
        } else {
            if (!bind_tensors(model, context, batch_size)) {
                fprintf(stderr, "[BATCH] FATAL: tensor binding failed\n");
                return GAZSI_INFER_ERR_FATAL;
            }
            if (!context->enqueueV3(stream)) {
                fprintf(stderr, "[BATCH] FATAL: enqueueV3 failed\n");
                return GAZSI_INFER_ERR_FATAL;
            }
        }

        /*
         * 3. Sync: makes the output tensor visible device-wide before the
         *    formatter reads it, and pins down the inference time that goes
         *    into the response body.
         */
        CUDA_CHECK(cudaStreamSynchronize(stream));

        long elapsed_us = (long)((monotonic_ns() - t_start) / 1000ULL);

        /* 4. Format the response bytes in place, in device memory */
        if (gazsi_launch_format_results(ring, plane, d_items, batch_size,
                                       (const float *)g_batch_buffers.d_output,
                                       g_batch_buffers.d_token_counts,
                                       model->embedding_dim,
                                       output_stride(model),
                                       elapsed_us, stream) != 0) {
            fprintf(stderr, "[BATCH] FATAL: result formatter launch failed\n");
            return GAZSI_INFER_ERR_FATAL;
        }

        /*
         * 5. Sync before returning. The caller publishes RESULT_READY next,
         *    and the response bytes must be visible device-wide before the TX
         *    kernel can observe that state change.
         */
        CUDA_CHECK(cudaStreamSynchronize(stream));

        *out_elapsed_us = elapsed_us;
        return GAZSI_INFER_OK;

    } catch (const std::exception& e) {
        /*
         * The only throw sites reachable from here are CUDA_CHECK (graph
         * launch, stream sync) and TensorRT itself. Both leave the context in
         * a state the next batch cannot recover from.
         */
        fprintf(stderr, "[BATCH] FATAL: Exception: %s\n", e.what());
        return GAZSI_INFER_ERR_FATAL;
    }
}
