# GAZSI: GPU-Accelerated Native Zero-copy for Socket-Based Inference Serving Framework

A high-performance GPU-accelerated HTTP server and packet processing framework using NVIDIA DOCA SDK and GPUNetIO.

## Overview

GAZSI enables zero-copy processing of network packets directly on the GPU for minimal latency. The system implements:

- **GPU-Direct HTTP Server**: HTTP request/response handling entirely on GPU
- **TensorRT Inference Integration**: ML inference via lock-free CPU-GPU ring buffer
- **Dynamic Batching**: Automatic request batching for improved throughput

## Architecture

The current implementation runs GPUNetIO networking and TensorRT inference on
one physical GPU. They use separate CUDA streams on the same CUDA device; the
design does not require MIG or another GPU-partitioning mechanism.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Network Interface                            │
│                     SmartNIC Rx/Tx Queues                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ GPUNetIO direct DMA
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                             GPU Memory                              │
│                                                                     │
│  Networking Stream                         Inference Stream          │
│  ┌─────────────┐  ┌─────────────┐         ┌─────────────┐          │
│  │ TCP Receive │─►│ HTTP Parser │────────►│ GPU         │          │
│  │ / Transmit  │  │             │         │ Tokenizer   │          │
│  └──────┬──────┘  └──────┬──────┘         └──────┬──────┘          │
│         │                │                       ▼                 │
│         │     ┌──────────▼─────────────────────────────┐           │
│         └────►│ Device-Resident Request/Response Ring  │◄────┐     │
│               └────────────────────────────────────────┘     │     │
│                                                ┌────────▼───┐ │     │
│                                                │ TensorRT   │ │     │
│                                                └────────┬───┘ │     │
│                                                ┌────────▼─────┴──┐  │
│                                                │ GPU Formatter   │  │
│                                                └─────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Compact control metadata only
                               ▼
                    ┌────────────────────────┐
                    │ CPU Metadata Dispatcher│
                    │ Batch Formation/Launch │
                    └────────────────────────┘
```

Request and response bytes remain in device memory throughout this path. A
small host-mapped, pinned control structure contains only slot state and record
descriptors. The CPU uses that metadata to form batches and invoke TensorRT,
but it does not copy, tokenize, or format request and response data. The
zero-copy mechanism therefore remains intact while the CPU performs the
execution control required to launch GPU work.

## Features

- **Zero-Copy Packet Processing**: Direct NIC-to-GPU data path
- **Split-Plane Lock-Free Ring Buffer**: 256 metadata slots with device-only payload storage
- **Semaphore Signaling**: Event-driven CPU-GPU synchronization
- **Dynamic Batch Inference**: Up to 8 requests per batch
- **HTTP/1.1 Support**: GET/POST request handling on GPU

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0+
- NVIDIA ConnectX-6 or newer SmartNIC
- 8GB+ System Memory

### Software
- Linux
- NVIDIA DOCA SDK with GPUNetIO support
- CUDA Toolkit
- TensorRT
- GCC

## Building

The project builds as part of the DOCA SDK application framework:

```bash
# Create symlink in DOCA applications directory
sudo ln -s /path/to/GAZSI \
    /opt/mellanox/doca/applications/GAZSI

# Build
cd /opt/mellanox/doca/applications
sudo meson build -Denable_GAZSI=true
sudo ninja -C build
```

## Usage

```bash
# Basic usage
sudo ./GAZSI -g <GPU_PCI> -n <NIC_PCI> -q <NUM_QUEUES>

# With HTTP server
sudo ./GAZSI -g E6:00.0 -n c1:00.0 -q 2 -s

# Options:
#   -g  GPU PCIe address
#   -n  NIC PCIe address
#   -q  Number of receive queues (1-4)
#   -s  Enable HTTP server mode
#   -e  Path to TensorRT engine file
```

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/index.` | GET | Static index page |
| `/inference?d=<text>` | GET | TensorRT inference endpoint |

### Inference Example

```bash
# Send inference request
curl "http://10.0.0.6/inference?d=hello%20world"

# Response
{
  "input": "hello world",
  "tokens": 3,
  "embedding_sample": [0.123, -0.456, 0.789],
  "inference_time_us": 1234,
  "batch_size": 1
}
```

## License

BSD-3-Clause. See individual source files for details.

## Acknowledgments

Built on NVIDIA DOCA SDK and GPUNetIO technology.
