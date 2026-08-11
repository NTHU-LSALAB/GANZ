#!/bin/bash
# GAZSI - Compile-only build of every translation unit against the installed DOCA
# SPDX-License-Identifier: BSD-3-Clause
#
# The full meson build cannot produce a binary on the development host: DOCA
# 3.3's libdoca_gpunetio_device.a ships cubins for sm_75 and newer only, and the
# local GPU is a Tesla V100 (sm_70), so the device link step has no code to
# link. Compilation is unaffected by that, so this compiles every TU with -c and
# checks the sources are consistent. It is a source check, not a link check —
# it deliberately does not claim a runnable binary.
#
#   ./compile_check.sh          compile everything, non-zero exit on any error
#   SM=80 ./compile_check.sh    target another architecture

set -u

cd "$(dirname "$0")/.." || exit 1

SM=${SM:-70}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/flexio/lib/pkgconfig

PKGC=$(pkg-config --cflags doca-gpunetio doca-flow doca-common libdpdk) || {
	echo "pkg-config failed — is DOCA installed?" >&2
	exit 1
}
# nvcc rejects the host-only flags pkg-config emits; hand them over via -Xcompiler.
PKGC_CU=$(echo "$PKGC" | sed 's/-march=corei7//; s/-mrtm//; s/-pthread//; s/-include rte_config.h//')

INC="-I. -Idoca -Iutils -Itcp -Iinference -Ikernels
     -I/opt/mellanox/doca/applications/common -I/usr/local/cuda/include"

C_SRCS="main.c
        doca/args.c doca/device.c doca/flow.c doca/http_tx.c doca/tcp.c
        tcp/cpu_rss.c tcp/session.c
        utils/utils.c"

CU_SRCS="kernels/tcp_rx.cu kernels/http_server.cu
         inference/gpu_pipeline.cu inference/ring_buffer.cu inference/tensorrt.cu"

fail=0

for f in $C_SRCS; do
	printf '%-24s ' "$f"
	if gcc -std=gnu11 -D_GNU_SOURCE -DDOCA_ALLOW_EXPERIMENTAL_API \
	       -c -o "$OUT/$(basename "$f" .c).o" "$f" $PKGC $INC 2>"$OUT/err"; then
		echo ok
	else
		echo FAIL
		sed -n '1,15p' "$OUT/err"
		fail=1
	fi
done

for f in $CU_SRCS; do
	printf '%-24s ' "$f"
	if $NVCC -std=c++17 -Wno-deprecated-gpu-targets \
	         -gencode "arch=compute_$SM,code=sm_$SM" -rdc=true \
	         -DDOCA_ALLOW_EXPERIMENTAL_API \
	         -Xcompiler -include -Xcompiler rte_config.h \
	         -c -o "$OUT/$(basename "$f" .cu).o" "$f" $PKGC_CU $INC 2>"$OUT/err"; then
		echo ok
	else
		echo FAIL
		sed -n '1,15p' "$OUT/err"
		fail=1
	fi
done

if [ $fail -eq 0 ]; then
	echo "compile-only build clean (no link attempted: sm_${SM} has no DOCA device code)"
fi
exit $fail
