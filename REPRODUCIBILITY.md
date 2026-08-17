# Reproducing the GAZSI Evaluation

This document records the experimental settings used for the paper.

## Platform

- GPU: NVIDIA Tesla V100 PCIe, 16 GB
- Network device: NVIDIA BlueField-3 with integrated ConnectX-7
- Host: Intel Xeon E5-2648L v2, 125 GB memory
- Operating system: Ubuntu 24.04
- CUDA Toolkit: 12.5
- TensorRT: 10.4.0
- DOCA SDK: 2.9.3008

The client uses ordinary HTTP/1.1 sockets. GAZSI receives traffic through the
BlueField data path, while the kernel TCP comparison terminates the same request
workload in the host Linux TCP stack. The reported GAZSI launch uses four receive
queues.

## Service configuration

- Service slots: 256
- Allocation per slot: 5,248 bytes
- Packed records per slot: at most 4
- Model sequence length: 128
- Maximum model batch size: 8

## Workloads and measurements

The primary concurrency sweep uses FP16 BERT-base with 1, 2, 4, 8, 16, and 32
concurrent requests. Each condition runs for 10 seconds. GAZSI and the kernel TCP
comparison use the same workload-specific whitespace and hash transformation.
This transformation is not a general tokenizer.

| Study | Tested conditions | Duration |
| --- | --- | --- |
| Payload size | 5, 50, 100, 250, and 500 bytes at concurrency 1, 8, 16, and 32 | 15 s |
| Dispatch | Adaptive and fixed targets 1, 4, and 8 at concurrency 2, 8, and 32 | 15 s |
| VMA | Concurrency 2, 4, 8, 16, and 32 | 15 s |
| Burst | Concurrency changes from 4 to 32 or 64 and returns to 4 | 2 s per phase |
| Cross-model | FP16 BERT-base, GPT-2, and GPT-2-Large at concurrency 16 | 15 s |

All HTTP load tests use `wrk --latency`. ApacheBench is not used. The primary
sweep uses one client thread at concurrency 1, two threads at concurrency 2, and
four threads at higher concurrency. The 15-second payload and dispatch runs use
up to four client threads.

The cross-model comparison uses five independent runs for each model and service
and reports the arithmetic mean and sample standard deviation. Other figures use
one recorded run or trace. Response-time percentiles are taken from `wrk`.
Response time is measured at the client from request issue until the complete
response arrives. Burst results report the median of the first five samples in
each phase.

## Kernel TCP comparison

The comparison server is implemented in
[`baseline/baseline_trt.cpp`](baseline/baseline_trt.cpp). It uses Linux kernel
TCP, a nonblocking `epoll` loop, pinned host buffers, asynchronous H2D and D2H
copies, a nonblocking CUDA stream, TensorRT batch size one, and a captured CUDA
Graph spanning the transfers and model execution. Request parsing, the
workload-specific hash transformation, and response construction execute on the
CPU. DPDK, `io_uring`, and GPUDirect RDMA are not part of this kernel TCP
configuration because they change its networking, host-I/O, or data-placement
boundary. VMA is evaluated separately with the same server as the
socket-preserving kernel-bypass reference.

## Building and running

The project build and application arguments are documented in
[`README.md`](README.md). Baseline-specific instructions are in
[`baseline/README.md`](baseline/README.md). The source-tree comparison check is
[`tests/ab_perf.sh`](tests/ab_perf.sh). Its comparison checkout must be supplied
with `--base` or `GAZSI_BASE_DIR`.
