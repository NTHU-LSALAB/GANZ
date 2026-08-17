# Optimized Kernel TCP Baseline

`baseline_trt.cpp` is the host-mediated baseline used by the GAZSI evaluation.
It serves HTTP/1.1 with the Linux kernel TCP stack and a single-threaded,
nonblocking epoll loop. The CPU parses each request, applies the same
workload-specific whitespace/hash transformation used by GAZSI, and formats the
response. TensorRT executes one request at a time on the GPU.

The configuration preserved for the paper campaign uses:

- pinned host input and output buffers;
- asynchronous H2D and D2H copies;
- a nonblocking CUDA stream;
- TensorRT with batch size one; and
- explicit synchronization between the transfer and inference stages.

The source also provides an optional `-O` mode that captures a batch-one CUDA
Graph spanning the transfers and TensorRT enqueue. The preserved campaign
launcher did not pass `-O`, and the archived logs use the non-graph `[PROFILE]`
path rather than `[PROFILE-OPT]`. The paper results therefore do not claim CUDA
Graph acceleration for this baseline.

## Build

Set `CUDA_ROOT` and `TENSORRT_ROOT` when they are not installed in the default
locations. The TensorRT library directory may be `lib`, `lib64`, or
`lib/x86_64-linux-gnu` depending on the installation.

```bash
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
TENSORRT_ROOT=${TENSORRT_ROOT:-/usr}

g++ -O3 -std=c++17 -pthread \
  -I"$CUDA_ROOT/include" \
  -I"$TENSORRT_ROOT/include" \
  baseline/baseline_trt.cpp \
  -L"$CUDA_ROOT/lib64" \
  -L"$TENSORRT_ROOT/lib/x86_64-linux-gnu" \
  -lnvinfer -lcudart \
  -o baseline/baseline_trt
```

## Run

Pass `-O` to evaluate the optional captured CUDA Graph mode. Omit it to match
the preserved paper campaign.

```bash
LD_LIBRARY_PATH="$TENSORRT_ROOT/lib/x86_64-linux-gnu:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}" \
  baseline/baseline_trt \
  -e models/bert_base.engine \
  -p 8090
```

An example load command matching the primary ten-second workload is:

```bash
wrk -t4 -c16 -d10s --latency \
  'http://127.0.0.1:8090/inference?d=What+is+the+sentiment+of+this+review'
```

DPDK, io_uring, and GPUDirect RDMA are intentionally not enabled in this
baseline. They define different host-I/O or data-placement boundaries. VMA/XLIO
is evaluated separately by preloading its userspace library with this same
server.
