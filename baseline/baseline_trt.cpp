/*
 * Traditional HTTP + TensorRT Baseline Server
 *
 * Uses the SAME TensorRT 10.4 engine as GAZSI, but with standard
 * CPU-based TCP/HTTP networking. This isolates the network path
 * difference: traditional CPU HTTP stack vs GAZSI's DPU zero-copy.
 *
 * Pipeline: TCP recv (CPU) → HTTP parse (CPU) → tokenize (CPU) →
 *           H2D memcpy → TRT inference (GPU) → D2H memcpy →
 *           JSON format (CPU) → TCP send (CPU)
 *
 * Build:
 *   g++ -O2 -o baseline_trt baseline_trt.cpp \
 *       -I/usr/include/x86_64-linux-gnu \
 *       -L/tmp/trt104-libs/usr/lib/x86_64-linux-gnu \
 *       -lnvinfer -lcudart -lpthread
 *
 * Run:
 *   LD_LIBRARY_PATH=/tmp/trt104-libs/usr/lib/x86_64-linux-gnu \
 *     ./baseline_trt -e /path/to/model.engine -p 8090
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <thread>
#include <atomic>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

/* ─── Constants (match GAZSI) ─── */
#define SEQUENCE_LENGTH 128
#define MAX_OUTPUT_DIM 32768
#define MAX_EVENTS 512
#define RECV_BUF_SIZE 4096

static std::atomic<bool> g_quit{false};

/* ─── TensorRT Logger ─── */
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

/* ─── Model Context ─── */
struct ModelContext {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    std::string output_tensor_name;
    std::string pooler_tensor_name;
    std::string position_ids_name;
    bool has_token_type_ids = false;
    bool has_pooler_output = false;
    bool has_position_ids = false;
    int embedding_dim = 768;

    /* GPU buffers */
    void *d_input_ids = nullptr;
    void *d_attention_mask = nullptr;
    void *d_token_type_ids = nullptr;
    void *d_output = nullptr;
    void *d_pooler_output = nullptr;
    void *d_position_ids = nullptr;

    /* Pinned host buffers */
    int64_t *h_input_ids = nullptr;
    int64_t *h_attention_mask = nullptr;
    float *h_output = nullptr;
    int64_t h_position_ids = 0;

    cudaStream_t stream = nullptr;
};

/* ─── Tokenizer (identical to GAZSI tensorrt.cu) ─── */
static int tokenize_text(const char* text, int64_t* input_ids, int64_t* attention_mask) {
    memset(input_ids, 0, SEQUENCE_LENGTH * sizeof(int64_t));
    memset(attention_mask, 0, SEQUENCE_LENGTH * sizeof(int64_t));

    input_ids[0] = 101;      /* [CLS] */
    attention_mask[0] = 1;

    if (!text || text[0] == '\0') {
        input_ids[1] = 102;  /* [SEP] */
        attention_mask[1] = 1;
        return 2;
    }

    std::istringstream iss(text);
    std::string word;
    int pos = 1;
    int count = 1;

    while (std::getline(iss, word, ' ') && pos < SEQUENCE_LENGTH - 1) {
        if (!word.empty()) {
            uint32_t hash = 0;
            for (char c : word)
                hash = hash * 31 + (uint8_t)c;
            input_ids[pos] = 1000 + (hash % 20000);
            attention_mask[pos] = 1;
            pos++;
            count++;
        }
    }

    if (pos < SEQUENCE_LENGTH) {
        input_ids[pos] = 102;  /* [SEP] */
        attention_mask[pos] = 1;
        count++;
    }
    return count;
}

/* ─── URL decode (identical to GAZSI main.c) ─── */
static void url_decode(char *str) {
    char *src = str, *dst = str;
    while (*src) {
        if (*src == '%' && isxdigit((unsigned char)src[1]) && isxdigit((unsigned char)src[2])) {
            char hex[3] = {src[1], src[2], '\0'};
            *dst++ = (char)strtol(hex, NULL, 16);
            src += 3;
        } else if (*src == '+') {
            *dst++ = ' ';
            src++;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

/* ─── Load TRT Engine ─── */
static ModelContext* load_engine(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        fprintf(stderr, "Cannot read engine: %s\n", path);
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> blob(size);
    file.read(blob.data(), size);
    file.close();

    auto* m = new ModelContext();
    Logger logger;

    m->runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!m->runtime) { delete m; return nullptr; }

    m->engine.reset(m->runtime->deserializeCudaEngine(blob.data(), size));
    if (!m->engine) { delete m; return nullptr; }

    m->context.reset(m->engine->createExecutionContext());
    if (!m->context) { delete m; return nullptr; }

    /* Detect tensor layout */
    m->output_tensor_name = "output";
    int n = m->engine->getNbIOTensors();
    for (int i = 0; i < n; i++) {
        const char* name = m->engine->getIOTensorName(i);
        auto mode = m->engine->getTensorIOMode(name);
        auto shape = m->engine->getTensorShape(name);

        if (strcmp(name, "token_type_ids") == 0) m->has_token_type_ids = true;

        if (mode == nvinfer1::TensorIOMode::kINPUT &&
            strcmp(name, "input_ids") != 0 &&
            strcmp(name, "attention_mask") != 0 &&
            strcmp(name, "token_type_ids") != 0) {
            m->has_position_ids = true;
            m->position_ids_name = name;
        }

        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (strcmp(name, "last_hidden_state") == 0) m->output_tensor_name = "last_hidden_state";
            else if (strcmp(name, "output") == 0) m->output_tensor_name = "output";
            else { m->has_pooler_output = true; m->pooler_tensor_name = name; }

            if (shape.nbDims >= 2 && shape.d[shape.nbDims - 1] > 0)
                m->embedding_dim = shape.d[shape.nbDims - 1];
        }
    }

    fprintf(stderr, "[Engine] %s: output=%s, embed_dim=%d, token_type_ids=%s, position_ids=%s\n",
            path, m->output_tensor_name.c_str(), m->embedding_dim,
            m->has_token_type_ids ? "yes" : "no",
            m->has_position_ids ? m->position_ids_name.c_str() : "no");

    /* Allocate GPU buffers */
    size_t in_sz = SEQUENCE_LENGTH * sizeof(int64_t);
    size_t out_sz = SEQUENCE_LENGTH * MAX_OUTPUT_DIM * sizeof(float);
    size_t pooler_sz = MAX_OUTPUT_DIM * sizeof(float);

    cudaMalloc(&m->d_input_ids, in_sz);
    cudaMalloc(&m->d_attention_mask, in_sz);
    cudaMalloc(&m->d_token_type_ids, in_sz);
    cudaMalloc(&m->d_output, out_sz);
    cudaMalloc(&m->d_pooler_output, pooler_sz);
    cudaMalloc(&m->d_position_ids, sizeof(int64_t));
    cudaMemset(m->d_token_type_ids, 0, in_sz);

    cudaHostAlloc((void**)&m->h_input_ids, in_sz, cudaHostAllocDefault);
    cudaHostAlloc((void**)&m->h_attention_mask, in_sz, cudaHostAllocDefault);
    cudaHostAlloc((void**)&m->h_output, m->embedding_dim * sizeof(float), cudaHostAllocDefault);

    cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking);

    return m;
}

/* ─── Run Inference ─── */
static uint64_t ts_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static struct {
    uint64_t sum_tokenize, sum_h2d, sum_infer, sum_d2h;
    uint64_t count;
} g_prof = {};

static void run_inference(ModelContext* m, const char* text,
                          float* emb_out, int* token_count, long* elapsed_us) {
    uint64_t p0 = ts_ns();

    *token_count = tokenize_text(text, m->h_input_ids, m->h_attention_mask);

    uint64_t p1 = ts_ns();

    size_t in_sz = SEQUENCE_LENGTH * sizeof(int64_t);
    int edim = m->embedding_dim;
    size_t out_sz = edim * sizeof(float);

    cudaMemcpyAsync(m->d_input_ids, m->h_input_ids, in_sz, cudaMemcpyHostToDevice, m->stream);
    cudaMemcpyAsync(m->d_attention_mask, m->h_attention_mask, in_sz, cudaMemcpyHostToDevice, m->stream);
    cudaStreamSynchronize(m->stream);

    uint64_t p2 = ts_ns();

    nvinfer1::Dims dims;
    dims.nbDims = 2;
    dims.d[0] = 1;
    dims.d[1] = SEQUENCE_LENGTH;

    m->context->setInputShape("input_ids", dims);
    m->context->setInputShape("attention_mask", dims);
    if (m->has_token_type_ids) m->context->setInputShape("token_type_ids", dims);

    m->context->setTensorAddress("input_ids", m->d_input_ids);
    m->context->setTensorAddress("attention_mask", m->d_attention_mask);
    if (m->has_token_type_ids) m->context->setTensorAddress("token_type_ids", m->d_token_type_ids);
    if (m->has_position_ids) {
        m->h_position_ids = (int64_t)SEQUENCE_LENGTH;
        cudaMemcpyAsync(m->d_position_ids, &m->h_position_ids,
                        sizeof(int64_t), cudaMemcpyHostToDevice, m->stream);
        m->context->setTensorAddress(m->position_ids_name.c_str(), m->d_position_ids);
    }
    m->context->setTensorAddress(m->output_tensor_name.c_str(), m->d_output);
    if (m->has_pooler_output && !m->pooler_tensor_name.empty())
        m->context->setTensorAddress(m->pooler_tensor_name.c_str(), m->d_pooler_output);

    m->context->enqueueV3(m->stream);
    cudaStreamSynchronize(m->stream);

    uint64_t p3 = ts_ns();

    cudaMemcpyAsync(m->h_output, m->d_output, out_sz, cudaMemcpyDeviceToHost, m->stream);
    cudaStreamSynchronize(m->stream);

    uint64_t p4 = ts_ns();

    *elapsed_us = (long)((p4 - p0) / 1000);

    g_prof.sum_tokenize += (p1 - p0);
    g_prof.sum_h2d += (p2 - p1);
    g_prof.sum_infer += (p3 - p2);
    g_prof.sum_d2h += (p4 - p3);
    g_prof.count++;
    if (g_prof.count > 0 && g_prof.count % 1000 == 0) {
        fprintf(stderr, "[PROFILE] n=%lu tok=%.1fµs h2d=%.1fµs infer=%.1fµs d2h=%.1fµs total=%.1fµs\n",
            (unsigned long)g_prof.count,
            (double)g_prof.sum_tokenize / g_prof.count / 1000.0,
            (double)g_prof.sum_h2d / g_prof.count / 1000.0,
            (double)g_prof.sum_infer / g_prof.count / 1000.0,
            (double)g_prof.sum_d2h / g_prof.count / 1000.0,
            (double)(g_prof.sum_tokenize + g_prof.sum_h2d + g_prof.sum_infer + g_prof.sum_d2h) / g_prof.count / 1000.0);
    }

    memcpy(emb_out, m->h_output, (edim < 3 ? edim : 3) * sizeof(float));
}

/* ─── Optimized inference: CUDA Graph captures H2D + inference + D2H ─── */
static bool g_graph_ready = false;
static cudaGraphExec_t g_graph_exec = nullptr;

static void init_cuda_graph(ModelContext* m) {
    size_t in_sz = SEQUENCE_LENGTH * sizeof(int64_t);
    int edim = m->embedding_dim;
    size_t out_sz = edim * sizeof(float);

    nvinfer1::Dims dims;
    dims.nbDims = 2; dims.d[0] = 1; dims.d[1] = SEQUENCE_LENGTH;

    m->context->setInputShape("input_ids", dims);
    m->context->setInputShape("attention_mask", dims);
    if (m->has_token_type_ids) m->context->setInputShape("token_type_ids", dims);
    m->context->setTensorAddress("input_ids", m->d_input_ids);
    m->context->setTensorAddress("attention_mask", m->d_attention_mask);
    if (m->has_token_type_ids)
        m->context->setTensorAddress("token_type_ids", m->d_token_type_ids);
    if (m->has_position_ids) {
        m->h_position_ids = (int64_t)SEQUENCE_LENGTH;
        cudaMemcpy(m->d_position_ids, &m->h_position_ids,
                   sizeof(int64_t), cudaMemcpyHostToDevice);
        m->context->setTensorAddress(m->position_ids_name.c_str(), m->d_position_ids);
    }
    m->context->setTensorAddress(m->output_tensor_name.c_str(), m->d_output);
    if (m->has_pooler_output && !m->pooler_tensor_name.empty())
        m->context->setTensorAddress(m->pooler_tensor_name.c_str(), m->d_pooler_output);

    m->context->enqueueV3(m->stream);
    cudaStreamSynchronize(m->stream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(m->stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(m->d_input_ids, m->h_input_ids, in_sz, cudaMemcpyHostToDevice, m->stream);
    cudaMemcpyAsync(m->d_attention_mask, m->h_attention_mask, in_sz, cudaMemcpyHostToDevice, m->stream);
    m->context->enqueueV3(m->stream);
    cudaMemcpyAsync(m->h_output, m->d_output, out_sz, cudaMemcpyDeviceToHost, m->stream);
    cudaStreamEndCapture(m->stream, &graph);
    cudaGraphInstantiate(&g_graph_exec, graph, 0);
    cudaGraphDestroy(graph);
    g_graph_ready = true;
    fprintf(stderr, "[OPT] CUDA Graph captured for batch=1\n");
}

static void run_inference_optimized(ModelContext* m, const char* text,
                                     float* emb_out, int* token_count, long* elapsed_us) {
    uint64_t p0 = ts_ns();

    *token_count = tokenize_text(text, m->h_input_ids, m->h_attention_mask);

    uint64_t p1 = ts_ns();

    cudaGraphLaunch(g_graph_exec, m->stream);
    cudaStreamSynchronize(m->stream);

    uint64_t p2 = ts_ns();

    *elapsed_us = (long)((p2 - p0) / 1000);

    g_prof.sum_tokenize += (p1 - p0);
    g_prof.sum_h2d += 0;
    g_prof.sum_infer += (p2 - p1);
    g_prof.sum_d2h += 0;
    g_prof.count++;
    if (g_prof.count > 0 && g_prof.count % 1000 == 0) {
        fprintf(stderr, "[PROFILE-OPT] n=%lu tok=%.1fµs graph=%.1fµs total=%.1fµs\n",
            (unsigned long)g_prof.count,
            (double)g_prof.sum_tokenize / g_prof.count / 1000.0,
            (double)g_prof.sum_infer / g_prof.count / 1000.0,
            (double)(g_prof.sum_tokenize + g_prof.sum_infer) / g_prof.count / 1000.0);
    }

    int edim = m->embedding_dim;
    memcpy(emb_out, m->h_output, (edim < 3 ? edim : 3) * sizeof(float));
}

static bool g_optimized = false;

/* ─── Set socket non-blocking ─── */
static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/* ─── Parse "GET /inference?d=... HTTP/1.1" → extract d value ─── */
static bool parse_request(const char* buf, int len, char* data_out, int data_max) {
    /* Find "GET /inference?d=" */
    const char* p = strstr(buf, "GET /inference?d=");
    if (!p) {
        /* Also handle /health */
        if (strstr(buf, "GET /health")) {
            data_out[0] = '\0';
            return true;  /* signal health check via empty string, caller checks path */
        }
        return false;
    }

    p += strlen("GET /inference?d=");
    const char* end = strchr(p, ' ');
    if (!end) end = strchr(p, '\r');
    if (!end) end = buf + len;

    int copy_len = (int)(end - p);
    if (copy_len >= data_max) copy_len = data_max - 1;
    memcpy(data_out, p, copy_len);
    data_out[copy_len] = '\0';

    url_decode(data_out);
    return true;
}

/* ─── Connection state for epoll ─── */
struct ConnState {
    int fd;
    char buf[RECV_BUF_SIZE];
    int buf_len;
    bool is_health;
};

/* ─── Main server loop ─── */
static void serve(ModelContext* model, int port) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); exit(1); }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); exit(1); }
    if (listen(listen_fd, 1024) < 0) { perror("listen"); exit(1); }
    set_nonblocking(listen_fd);

    int epfd = epoll_create1(0);
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.ptr = nullptr;  /* nullptr = listen socket */
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

    struct epoll_event events[MAX_EVENTS];
    float emb[3];
    char data_buf[1024];
    char resp_buf[2048];

    fprintf(stderr, "[Server] Listening on :%d (epoll, single-thread, TRT %d-dim)\n",
            port, model->embedding_dim);

    while (!g_quit.load()) {
        int n = epoll_wait(epfd, events, MAX_EVENTS, 100);

        for (int i = 0; i < n; i++) {
            if (events[i].data.ptr == nullptr) {
                /* Accept new connections */
                while (true) {
                    int cfd = accept(listen_fd, NULL, NULL);
                    if (cfd < 0) break;

                    /* TCP_NODELAY for low latency */
                    int one = 1;
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    set_nonblocking(cfd);

                    auto* conn = new ConnState{cfd, {}, 0, false};
                    struct epoll_event cev;
                    cev.events = EPOLLIN | EPOLLET;
                    cev.data.ptr = conn;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &cev);
                }
            } else {
                auto* conn = (ConnState*)events[i].data.ptr;

                /* Read request */
                while (true) {
                    int r = recv(conn->fd, conn->buf + conn->buf_len,
                                 RECV_BUF_SIZE - conn->buf_len - 1, 0);
                    if (r <= 0) break;
                    conn->buf_len += r;
                }
                conn->buf[conn->buf_len] = '\0';

                /* Check for complete HTTP request */
                if (strstr(conn->buf, "\r\n\r\n")) {
                    bool is_health = (strstr(conn->buf, "GET /health") != nullptr);

                    if (is_health) {
                        int body_len = snprintf(resp_buf + 256, 1024,
                            "{\"status\":\"ok\",\"engine\":\"TRT-10.4\",\"embed_dim\":%d}",
                            model->embedding_dim);
                        int hdr_len = snprintf(resp_buf, 256,
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Type: application/json\r\n"
                            "Content-Length: %d\r\n"
                            "Connection: keep-alive\r\n\r\n", body_len);
                        /* Move body right after header */
                        memmove(resp_buf + hdr_len, resp_buf + 256, body_len);
                        send(conn->fd, resp_buf, hdr_len + body_len, MSG_NOSIGNAL);
                    } else if (parse_request(conn->buf, conn->buf_len, data_buf, sizeof(data_buf))) {
                        int tokens;
                        long inf_us;
                        if (g_optimized)
                            run_inference_optimized(model, data_buf, emb, &tokens, &inf_us);
                        else
                            run_inference(model, data_buf, emb, &tokens, &inf_us);

                        int body_len = snprintf(resp_buf + 256, 1024,
                            "{\"input\":\"%s\",\"tokens\":%d,"
                            "\"embedding_sample\":[%.6f,%.6f,%.6f],"
                            "\"inference_time_us\":%ld,\"batch_size\":1}",
                            data_buf, tokens, emb[0], emb[1], emb[2], inf_us);

                        int hdr_len = snprintf(resp_buf, 256,
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Type: application/json\r\n"
                            "Content-Length: %d\r\n"
                            "Connection: keep-alive\r\n\r\n", body_len);
                        memmove(resp_buf + hdr_len, resp_buf + 256, body_len);
                        send(conn->fd, resp_buf, hdr_len + body_len, MSG_NOSIGNAL);
                    } else {
                        const char* r404 = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                        send(conn->fd, r404, strlen(r404), MSG_NOSIGNAL);
                        epoll_ctl(epfd, EPOLL_CTL_DEL, conn->fd, nullptr);
                        close(conn->fd);
                        delete conn;
                        continue;
                    }

                    /* Reset buffer for keep-alive */
                    conn->buf_len = 0;
                }
            }
        }
    }

    close(listen_fd);
    close(epfd);
}

/* ─── Signal handler ─── */
static void sig_handler(int) { g_quit.store(true); }

/* ─── Main ─── */
int main(int argc, char* argv[]) {
    const char* engine_path = nullptr;
    int port = 8090;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) engine_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-O") == 0) g_optimized = true;
    }

    if (!engine_path) {
        fprintf(stderr, "Usage: %s -e <engine_path> [-p port] [-O]\n", argv[0]);
        return 1;
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    cudaSetDevice(0);
    cudaFree(0);  /* Init CUDA context */

    fprintf(stderr, "Loading engine: %s\n", engine_path);
    ModelContext* model = load_engine(engine_path);
    if (!model) {
        fprintf(stderr, "Failed to load engine\n");
        return 1;
    }

    /* Warmup */
    fprintf(stderr, "Warming up (20 inferences)...\n");
    float warmup_emb[3];
    int warmup_tok;
    long warmup_us;
    for (int i = 0; i < 20; i++)
        run_inference(model, "hello world warmup test", warmup_emb, &warmup_tok, &warmup_us);
    fprintf(stderr, "Warmup done. Last inference: %ldus\n", warmup_us);

    if (g_optimized) {
        fprintf(stderr, "[OPT] Capturing CUDA Graph...\n");
        init_cuda_graph(model);
    }

    serve(model, port);

    fprintf(stderr, "Server stopped\n");
    return 0;
}
