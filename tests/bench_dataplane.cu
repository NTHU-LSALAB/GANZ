/*
 * GAZSI - Data-Plane Overhead Microbenchmark
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Measures the per-request work that the device-resident data plane removes,
 * comparing the two paths end to end from "request bytes have arrived" to
 * "response bytes are where the TX kernel reads them":
 *
 *   HOST-STAGED (baseline)          DEVICE-RESIDENT (this branch)
 *   ---------------------------     ------------------------------
 *   CPU tokenize -> pinned host     GPU tokenize kernel, in place
 *   H2D copy ids + mask             (none)
 *   [TensorRT inference]            [TensorRT inference]
 *   D2H copy output tensor          (none)
 *   CPU snprintf JSON               GPU format kernel, in place
 *   write into host-mapped slot     (none)
 *
 * SCOPE AND HONESTY
 * -----------------
 * The TensorRT inference stage is EXCLUDED from both paths and is identical in
 * both, so it cancels. It is excluded because no engine in models/ can be
 * deserialized on this host: all six were built with a newer TensorRT than the
 * installed 10.9.0, so an engine-inclusive measurement is impossible here.
 *
 * This therefore measures data-plane overhead per request, NOT end-to-end
 * server throughput or latency. It is diagnostic evidence and is NOT a
 * substitute for the socket-level A/B, which needs a runnable server binary.
 * Nothing here belongs in the manuscript.
 *
 * Build and run:  make -f Makefile bench    (or see tests/Makefile)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <algorithm>

#include "gpu_pipeline.h"

#define SEQ_LEN     128
#define EMBED_DIM   768               /* BERT base */
#define OUT_STRIDE  (SEQ_LEN * EMBED_DIM) /* 3D output: (batch, seq, dim) */
#define MAX_BATCH   8

static uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t cpu_ns(void)
{
	struct rusage ru;
	getrusage(RUSAGE_SELF, &ru);
	return ((uint64_t)ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000000ULL +
	       ((uint64_t)ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000ULL;
}

#define CU(call)                                                               \
	do {                                                                   \
		cudaError_t e = (call);                                        \
		if (e != cudaSuccess) {                                       \
			printf("FATAL %s:%d %s\n", __FILE__, __LINE__,        \
			       cudaGetErrorString(e));                        \
			exit(1);                                              \
		}                                                              \
	} while (0)

/* Representative request payloads, same set for both paths */
static const char *REQS[MAX_BATCH] = {
	"the quick brown fox jumps over the lazy dog",
	"hello world",
	"bench request payload",
	"many words here to tokenize now please",
	"alpha beta gamma delta",
	"a slightly longer request with more tokens to process",
	"short",
	"tokens with plus and percent encoding decoded already",
};

struct stats { double mean_us, p99_us, cpu_us_per_req; size_t pcie_bytes_per_req; };

static void summarize(double *samples, int n, uint64_t cpu_delta, int total_reqs,
		      size_t pcie_per_req, struct stats *out)
{
	std::sort(samples, samples + n);
	double sum = 0;
	for (int i = 0; i < n; i++) sum += samples[i];
	int p = (int)(n * 0.99); if (p >= n) p = n - 1;
	out->mean_us = sum / n;
	out->p99_us = samples[p];
	out->cpu_us_per_req = (double)cpu_delta / 1000.0 / total_reqs;
	out->pcie_bytes_per_req = pcie_per_req;
}

/* ---------------- Path A: host-staged (baseline behaviour) ---------------- */
static void bench_host_staged(int batch, int iters, struct stats *st)
{
	/* Host-mapped request/response arena, as the baseline ring had */
	char *h_slots;
	CU(cudaHostAlloc(&h_slots, (size_t)MAX_BATCH * 2048, cudaHostAllocMapped));
	for (int b = 0; b < batch; b++)
		strcpy(h_slots + (size_t)b * 2048, REQS[b]);

	int64_t *h_ids, *h_mask;
	float *h_out;
	void *d_ids, *d_mask, *d_out;
	CU(cudaHostAlloc(&h_ids, (size_t)batch * SEQ_LEN * sizeof(int64_t), cudaHostAllocDefault));
	CU(cudaHostAlloc(&h_mask, (size_t)batch * SEQ_LEN * sizeof(int64_t), cudaHostAllocDefault));
	CU(cudaHostAlloc(&h_out, (size_t)batch * OUT_STRIDE * sizeof(float), cudaHostAllocDefault));
	CU(cudaMalloc(&d_ids, (size_t)batch * SEQ_LEN * sizeof(int64_t)));
	CU(cudaMalloc(&d_mask, (size_t)batch * SEQ_LEN * sizeof(int64_t)));
	CU(cudaMalloc(&d_out, (size_t)batch * OUT_STRIDE * sizeof(float)));
	CU(cudaMemset(d_out, 0, (size_t)batch * OUT_STRIDE * sizeof(float)));

	cudaStream_t s;
	CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

	size_t in_bytes = (size_t)batch * SEQ_LEN * sizeof(int64_t) * 2;
	size_t out_bytes = (size_t)batch * OUT_STRIDE * sizeof(float);

	double *samples = (double *)malloc(sizeof(double) * iters);
	char result[1024];

	/* warm */
	for (int b = 0; b < batch; b++)
		gazsi_reference_tokenize(h_slots + (size_t)b * 2048,
					 &h_ids[b * SEQ_LEN], &h_mask[b * SEQ_LEN], SEQ_LEN);
	CU(cudaStreamSynchronize(s));

	uint64_t c0 = cpu_ns();
	for (int it = 0; it < iters; it++) {
		uint64_t t0 = now_ns();

		/* 1. CPU tokenizes request bytes into pinned host memory */
		int tok[MAX_BATCH];
		for (int b = 0; b < batch; b++)
			tok[b] = gazsi_reference_tokenize(h_slots + (size_t)b * 2048,
							  &h_ids[b * SEQ_LEN],
							  &h_mask[b * SEQ_LEN], SEQ_LEN);

		/* 2. H2D of the input tensors */
		CU(cudaMemcpyAsync(d_ids, h_ids, in_bytes / 2, cudaMemcpyHostToDevice, s));
		CU(cudaMemcpyAsync(d_mask, h_mask, in_bytes / 2, cudaMemcpyHostToDevice, s));

		/* 3. [TensorRT inference would run here — excluded, identical both paths] */

		/* 4. D2H of the output tensor. The baseline graph copied the FULL
		 *    tensor even though only 3 floats are read. */
		CU(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, s));
		CU(cudaStreamSynchronize(s));

		/* 5. CPU formats JSON and writes it into the host-mapped slot */
		for (int b = 0; b < batch; b++) {
			float *e = &h_out[b * OUT_STRIDE];
			int n = snprintf(result, sizeof(result),
				"{\"input\":\"%s\",\"tokens\":%d,\"embedding_sample\":[%.6f,%.6f,%.6f],"
				"\"inference_time_us\":%ld,\"batch_size\":%d}",
				h_slots + (size_t)b * 2048, tok[b], e[0], e[1], e[2], 1234L, batch);
			memcpy(h_slots + (size_t)b * 2048 + 1024, result, n);
		}

		samples[it] = (now_ns() - t0) / 1000.0;
	}
	uint64_t c1 = cpu_ns();

	summarize(samples, iters, c1 - c0, iters * batch,
		  (in_bytes + out_bytes) / batch, st);

	free(samples);
	cudaStreamDestroy(s);
	cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_out);
	cudaFreeHost(h_ids); cudaFreeHost(h_mask); cudaFreeHost(h_out);
	cudaFreeHost(h_slots);
}

/* ---------------- Path B: device-resident (this branch) ---------------- */
static void bench_device_resident(int batch, int iters, struct stats *st)
{
	struct inference_payload_plane *d_plane;
	struct inference_ring_buffer *h_ring, *d_ring;
	struct gazsi_batch_item *h_items, *d_items;
	void *d_ids, *d_mask, *d_out;
	int32_t *d_tok;

	CU(cudaMalloc(&d_plane, sizeof(*d_plane)));
	CU(cudaMemset(d_plane, 0, sizeof(*d_plane)));
	CU(cudaHostAlloc(&h_ring, sizeof(*h_ring), cudaHostAllocMapped));
	memset(h_ring, 0, sizeof(*h_ring));
	CU(cudaHostGetDevicePointer(&d_ring, h_ring, 0));
	CU(cudaHostAlloc(&h_items, MAX_BATCH * sizeof(*h_items), cudaHostAllocMapped));
	CU(cudaHostGetDevicePointer(&d_items, h_items, 0));

	CU(cudaMalloc(&d_ids, (size_t)batch * SEQ_LEN * sizeof(int64_t)));
	CU(cudaMalloc(&d_mask, (size_t)batch * SEQ_LEN * sizeof(int64_t)));
	CU(cudaMalloc(&d_tok, MAX_BATCH * sizeof(int32_t)));
	CU(cudaMalloc(&d_out, (size_t)batch * OUT_STRIDE * sizeof(float)));
	CU(cudaMemset(d_out, 0, (size_t)batch * OUT_STRIDE * sizeof(float)));

	/* Stage request bytes into the device plane, standing in for the GPU RX
	 * kernel. One-time setup, outside the measured loop. */
	for (int b = 0; b < batch; b++) {
		char *dst = (char *)d_plane + offsetof(struct inference_payload_plane, slots) +
			    (size_t)b * sizeof(struct inference_payload_slot) +
			    offsetof(struct inference_payload_slot, request);
		CU(cudaMemcpy(dst, REQS[b], strlen(REQS[b]) + 1, cudaMemcpyHostToDevice));
		h_items[b].slot_index = b;
		h_items[b].record_index = 0;
		h_items[b].req_offset = 0;
		h_items[b].req_length = (uint32_t)strlen(REQS[b]);
	}

	cudaStream_t s;
	CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

	/* warm */
	gazsi_launch_tokenize(d_plane, d_items, batch, d_ids, d_mask, d_tok, SEQ_LEN, s);
	gazsi_launch_format_results(d_ring, d_plane, d_items, batch, (const float *)d_out,
				    d_tok, EMBED_DIM, OUT_STRIDE, 1234L, s);
	CU(cudaStreamSynchronize(s));

	double *samples = (double *)malloc(sizeof(double) * iters);

	uint64_t c0 = cpu_ns();
	for (int it = 0; it < iters; it++) {
		uint64_t t0 = now_ns();

		/* 1. GPU tokenizes device-resident bytes into the engine inputs */
		gazsi_launch_tokenize(d_plane, d_items, batch, d_ids, d_mask, d_tok, SEQ_LEN, s);

		/* 2. [TensorRT inference would run here — excluded, identical both paths] */

		/* 3. GPU formats the response in place */
		gazsi_launch_format_results(d_ring, d_plane, d_items, batch,
					    (const float *)d_out, d_tok,
					    EMBED_DIM, OUT_STRIDE, 1234L, s);
		CU(cudaStreamSynchronize(s));

		samples[it] = (now_ns() - t0) / 1000.0;
	}
	uint64_t c1 = cpu_ns();

	/* Only the 16-byte descriptor per record is read across PCIe */
	summarize(samples, iters, c1 - c0, iters * batch,
		  sizeof(struct gazsi_batch_item), st);

	free(samples);
	cudaStreamDestroy(s);
	cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_out); cudaFree(d_tok);
	cudaFreeHost(h_items); cudaFreeHost(h_ring); cudaFree(d_plane);
}

int main(int argc, char **argv)
{
	int iters = (argc > 1) ? atoi(argv[1]) : 2000;
	int reps = (argc > 2) ? atoi(argv[2]) : 3;

	cudaDeviceProp prop;
	int dev = 0;
	if (cudaGetDevice(&dev) != cudaSuccess ||
	    cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
		printf("No CUDA device\n");
		return 77;
	}

	printf("GAZSI data-plane microbenchmark on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
	printf("seq_len=%d embed_dim=%d out_stride=%d iters=%d reps=%d\n",
	       SEQ_LEN, EMBED_DIM, OUT_STRIDE, iters, reps);
	printf("TensorRT inference EXCLUDED from both paths (identical, and no engine\n"
	       "in models/ deserializes with the installed TensorRT 10.9).\n"
	       "Data-plane overhead only — NOT end-to-end server throughput.\n\n");

	printf("%-8s %-16s %10s %10s %12s %14s\n",
	       "batch", "path", "mean_us", "p99_us", "cpu_us/req", "pcie_B/req");

	int batches[] = { 1, 8 };
	for (unsigned bi = 0; bi < sizeof(batches) / sizeof(batches[0]); bi++) {
		int batch = batches[bi];
		struct stats a[8], b[8];
		for (int r = 0; r < reps; r++) {
			bench_host_staged(batch, iters, &a[r]);
			bench_device_resident(batch, iters, &b[r]);
		}
		/* median rep by mean */
		double am[8], bm[8];
		for (int r = 0; r < reps; r++) { am[r] = a[r].mean_us; bm[r] = b[r].mean_us; }
		std::sort(am, am + reps); std::sort(bm, bm + reps);
		int mid = reps / 2;

		double ap = 0, bp = 0, ac = 0, bc = 0;
		for (int r = 0; r < reps; r++) { ap += a[r].p99_us; bp += b[r].p99_us;
						 ac += a[r].cpu_us_per_req; bc += b[r].cpu_us_per_req; }

		printf("%-8d %-16s %10.2f %10.2f %12.2f %14zu\n", batch, "host-staged",
		       am[mid], ap / reps, ac / reps, a[0].pcie_bytes_per_req);
		printf("%-8d %-16s %10.2f %10.2f %12.2f %14zu\n", batch, "device-resident",
		       bm[mid], bp / reps, bc / reps, b[0].pcie_bytes_per_req);
		printf("%-8s %-16s %9.2fx %9.2fx %11.2fx %13.1fx\n", "", "ratio (host/dev)",
		       am[mid] / bm[mid], (ap / reps) / (bp / reps), (ac / reps) / (bc / reps),
		       (double)a[0].pcie_bytes_per_req / (double)b[0].pcie_bytes_per_req);
		printf("\n");
	}

	printf("Diagnostic evidence only. Not a substitute for the socket-level A/B,\n"
	       "and not for the manuscript.\n");
	return 0;
}
