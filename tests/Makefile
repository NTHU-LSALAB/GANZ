# GAZSI device-resident pipeline tests
# SPDX-License-Identifier: BSD-3-Clause
#
# Standalone: needs only CUDA. No DOCA SDK and no TensorRT engine, so it runs
# on any host with a GPU, including ones without a BlueField NIC.
#
#   make            build and run
#   make sanitize   run under compute-sanitizer (memcheck, racecheck, initcheck)
#   make SM=80      target a different architecture (default 70, the local V100)

NVCC ?= /usr/local/cuda/bin/nvcc
SM   ?= 70

CUDA_ARGS = -std=c++17 -Wno-deprecated-gpu-targets \
            -gencode arch=compute_$(SM),code=sm_$(SM)
INCLUDES  = -I../inference

TARGET  = test_gpu_pipeline
SOURCES = test_gpu_pipeline.cu ../inference/gpu_pipeline.cu

BENCH        = bench_dataplane
BENCH_SRCS   = bench_dataplane.cu ../inference/gpu_pipeline.cu
HDRTEST      = test_tcp_hdr
HDRTEST_SRCS = test_tcp_hdr.c

CLAIMTEST      = test_slot_claim
CLAIMTEST_SRCS = test_slot_claim.cu

DPDK_CFLAGS = $(shell PKG_CONFIG_PATH=/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig \
                      pkg-config --cflags libdpdk 2>/dev/null)

.PHONY: all run sanitize bench hdrtest claimtest clean

all: run hdrtest claimtest

$(TARGET): $(SOURCES) ../inference/gpu_pipeline.h ../inference/ring_buffer.h
	$(NVCC) $(CUDA_ARGS) $(INCLUDES) -o $@ $(SOURCES)

run: $(TARGET)
	./$(TARGET)

# Data-plane overhead benchmark (device-resident vs host-staged).
# Excludes TensorRT inference; see the header comment in bench_dataplane.cu.
$(BENCH): $(BENCH_SRCS) ../inference/gpu_pipeline.h ../inference/ring_buffer.h
	$(NVCC) $(CUDA_ARGS) -O3 $(INCLUDES) -o $@ $(BENCH_SRCS)

bench: $(BENCH)
	./$(BENCH) $(ITERS) $(REPS)

# TCP header encoding regression (DOCA33-COMPAT edits in tcp/cpu_rss.c)
$(HDRTEST): $(HDRTEST_SRCS)
	$(CC) -std=gnu11 -O2 -Wall -Wextra $(DPDK_CFLAGS) -o $@ $(HDRTEST_SRCS)

hdrtest: $(HDRTEST)
	./$(HDRTEST)

# Slot ownership regression: a failed state CAS must not publish the slot.
# Covers both consumers (host request_queue, device response_queue).
$(CLAIMTEST): $(CLAIMTEST_SRCS) ../inference/slot_claim.h \
              ../inference/index_queue.h ../inference/ring_buffer.h \
              ../inference/tensorrt.h
	$(NVCC) $(CUDA_ARGS) $(INCLUDES) -o $@ $(CLAIMTEST_SRCS)

claimtest: $(CLAIMTEST)
	./$(CLAIMTEST)

sanitize: $(TARGET) $(CLAIMTEST)
	@for bin in $(TARGET) $(CLAIMTEST); do \
		for tool in memcheck racecheck initcheck; do \
			echo "=== compute-sanitizer --tool $$tool ./$$bin ==="; \
			$(dir $(NVCC))compute-sanitizer --tool $$tool --error-exitcode 1 ./$$bin \
				| tail -3 || exit 1; \
		done; \
	done

clean:
	rm -f $(TARGET) $(BENCH) $(HDRTEST) $(CLAIMTEST)
