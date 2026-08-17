#!/bin/bash
# GAZSI - Same-host A/B: device-resident data plane vs original host-staged path
# SPDX-License-Identifier: BSD-3-Clause
#
# Compares this branch against the untouched baseline tree on one host, same
# engine, same client, at concurrency 1 / 16 / 32 with >= 3 repetitions each.
#
# Collects per (variant, concurrency, rep):
#   throughput (req/s), average latency, p99 latency, server CPU time,
#   PCIe traffic (nvidia-smi dmon rx/tx counters)
# and checks response-body equivalence between variants.
#
# Regression gates, applied to medians across reps:
#   throughput  < 95%  of baseline  -> REGRESSION
#   avg latency > 105% of baseline  -> REGRESSION
#   p99 latency > 105% of baseline  -> REGRESSION
#
# THIS SCRIPT PRODUCED NO NUMBERS ON THE DEVELOPMENT HOST.
#
# Both trees now COMPILE cleanly against the installed DOCA 3.3.0109 via the
# shared compatibility layer (doca/doca33_compat.h, kernels/doca33_compat.cuh),
# which is applied identically to both so it cannot bias the comparison. What
# stops a runnable binary on that host is the GPU, not the source:
#
#   * libdoca_gpunetio_device.a ships cubins for sm_75 and newer only. The
#     development GPU is a Tesla V100 (sm_70), so there is no device code for
#     it, and no source change can create any.
#   * That archive's relocatable device code is CUDA 13 ABI (version 8) while
#     the newest toolkit installed is CUDA 12.8 (ABI 7), so nvlink refuses it
#     at any -arch.
#
# Run this on a host with an sm_75+ GPU, a CUDA toolkit matching the DOCA
# device archive ABI, hugepages, nvidia-peermem, and the client namespace.
#
# Usage:
#   ./ab_perf.sh --gpu <pci> --nic <pci> [options]
#
# Required:
#   --gpu   <pci>      GPU PCIe address passed to the server
#   --nic   <pci>      NIC PCIe address passed to the server
#
# Options:
#   --engine <path>    TensorRT engine        (default ../models/bert_base.engine)
#   --new    <dir>     this branch            (default the parent of this script)
#   --base   <dir>     comparison source tree (or set GAZSI_BASE_DIR)
#   --ip     <addr>    server IP              (default 10.0.0.6)
#   --port   <port>    server port            (default 8089)
#   --netns  <ns>      client namespace       (default test_ns)
#   --reps   <n>       repetitions            (default 3)
#   --secs   <n>       seconds per run        (default 30)
#   --out    <dir>     results directory      (default ./ab_results)

set -uo pipefail

GPU_PCI=""; NIC_PCI=""
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_DIR="$(dirname "$HERE")"
BASE_DIR="${GAZSI_BASE_DIR:-}"
ENGINE="$NEW_DIR/models/bert_base.engine"
SRV_IP="10.0.0.6"; SRV_PORT="8089"; NETNS="test_ns"
REPS=3; SECS=30; OUTDIR="./ab_results"
CONCURRENCIES=(1 16 32)

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU_PCI="$2"; shift 2 ;;
        --nic) NIC_PCI="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --new) NEW_DIR="$2"; shift 2 ;;
        --base) BASE_DIR="$2"; shift 2 ;;
        --ip) SRV_IP="$2"; shift 2 ;;
        --port) SRV_PORT="$2"; shift 2 ;;
        --netns) NETNS="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --secs) SECS="$2"; shift 2 ;;
        --out) OUTDIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$GPU_PCI" ] || [ -z "$NIC_PCI" ]; then
    echo "ERROR: --gpu and --nic are required" >&2
    exit 2
fi

if [ -z "$BASE_DIR" ]; then
    echo "ERROR: --base <comparison-tree> or GAZSI_BASE_DIR is required" >&2
    exit 2
fi

mkdir -p "$OUTDIR"
CSV="$OUTDIR/raw.csv"
echo "variant,concurrency,rep,requests,seconds,throughput_rps,avg_ms,p99_ms,non2xx,cpu_s,pcie_rx_MBs,pcie_tx_MBs" > "$CSV"

# ------------------------------------------------------------------
# Preflight: refuse to emit numbers unless every prerequisite is real
# ------------------------------------------------------------------
preflight_fail=0
say() { printf '%-58s %s\n' "$1" "$2"; }

echo "=== preflight ==="
for d in "$NEW_DIR" "$BASE_DIR"; do
    if [ -d "$d" ]; then say "tree present: $d" "ok"; else say "tree present: $d" "MISSING"; preflight_fail=1; fi
done
if [ -f "$ENGINE" ]; then say "engine: $ENGINE" "ok"; else say "engine: $ENGINE" "MISSING"; preflight_fail=1; fi
if command -v pkg-config >/dev/null && pkg-config --exists doca-gpunetio; then
    say "DOCA SDK version" "$(cat /opt/mellanox/doca/VERSION 2>/dev/null || echo unknown)"
else
    say "doca-gpunetio pkg-config" "MISSING"; preflight_fail=1
fi

# The GPU's compute capability must appear in the DOCA device archive. This is
# the blocker that stopped the development host: a V100 is sm_70 and the archive
# starts at sm_75.
DEV_ARCHIVE="$(pkg-config --variable=libdir doca-gpunetio 2>/dev/null)/libdoca_gpunetio_device.a"
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
if [ -n "$CC_RAW" ] && [ -f "$DEV_ARCHIVE" ] && command -v cuobjdump >/dev/null; then
    if cuobjdump "$DEV_ARCHIVE" 2>/dev/null | grep -qE "sm_${CC_RAW}\b"; then
        say "DOCA device archive has sm_${CC_RAW}" "ok"
    else
        have="$(cuobjdump "$DEV_ARCHIVE" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
        say "DOCA device archive has sm_${CC_RAW}" "NO (archive has: ${have})"
        preflight_fail=1
    fi
else
    say "GPU arch vs DOCA device archive" "UNVERIFIED (need nvidia-smi + cuobjdump)"
    preflight_fail=1
fi

# The toolkit must be able to device-link that archive (relocatable ABI match).
# Indicative only: an empty probe object may not force nvlink to process archive
# members, so a real build is the authority. On the development host a real link
# failed with "ABI version '8' is incompatible with target ABI version '7'".
if command -v nvcc >/dev/null; then
    probe_dir="$(mktemp -d)"
    : > "$probe_dir/p.cu"
    if nvcc -rdc=true -gencode arch=compute_${CC_RAW},code=sm_${CC_RAW} \
            -c "$probe_dir/p.cu" -o "$probe_dir/p.o" 2>/dev/null && \
       nvcc -gencode arch=compute_${CC_RAW},code=sm_${CC_RAW} -dlink \
            "$probe_dir/p.o" "$DEV_ARCHIVE" -o "$probe_dir/dl.o" 2>"$probe_dir/err"; then
        say "nvcc device-link probe (indicative only)" "ok"
    else
        say "nvcc device-link probe" "NO ($(head -1 "$probe_dir/err" | cut -c1-60))"
        preflight_fail=1
    fi
    rm -rf "$probe_dir"
else
    say "nvcc" "MISSING"; preflight_fail=1
fi

# Runtime prerequisites that need privilege
[ "$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo 0)" -gt 0 ] \
    && say "hugepages allocated" "ok" \
    || { say "hugepages allocated" "NONE (DPDK EAL needs them; root)"; preflight_fail=1; }
lsmod 2>/dev/null | grep -q nvidia_peermem \
    && say "nvidia-peermem loaded" "ok" \
    || { say "nvidia-peermem loaded" "NO (GPUDirect needs it; root)"; preflight_fail=1; }
if ip netns list 2>/dev/null | grep -qw "$NETNS"; then
    say "client namespace: $NETNS" "ok"
else
    say "client namespace: $NETNS" "MISSING"; preflight_fail=1
fi
command -v wrk >/dev/null && say "wrk (load generator)" "ok" || { say "wrk" "MISSING"; preflight_fail=1; }
command -v curl >/dev/null && say "curl (equivalence probe)" "ok" || { say "curl" "MISSING"; preflight_fail=1; }
command -v jq  >/dev/null && say "jq (response normalisation)" "ok" || { say "jq" "MISSING (falling back to sed)" ; }

# PCIe counters: prove the columns exist AND carry numbers on THIS GPU before
# relying on them. Reporting zeros from an unsupported counter is worse than
# reporting that it is unavailable.
PCIE_OK=0
if command -v nvidia-smi >/dev/null; then
    pcie_probe="$(timeout 8 nvidia-smi dmon -s t -c 2 2>/dev/null)"
    if echo "$pcie_probe" | grep -qE "rxpci" && \
       echo "$pcie_probe" | awk '/^ *[0-9]/ { if ($2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) ok=1 } END { exit !ok }'; then
        PCIE_OK=1; say "nvidia-smi dmon -s t (rxpci/txpci)" "ok"
    else
        say "nvidia-smi dmon -s t (rxpci/txpci)" "UNSUPPORTED -> PCIe reported as NA"
    fi
else
    say "nvidia-smi" "MISSING -> PCIe reported as NA"
fi

if [ "$preflight_fail" -ne 0 ]; then
    cat >&2 <<EOF

=== PREFLIGHT FAILED — no measurements taken ===
One or more prerequisites are absent, so this script will not produce numbers.
Reporting partial or synthetic figures from an incomplete environment would be
worse than reporting none. Fix the items marked MISSING/MISMATCH and re-run.

Do not transcribe anything from a failed run into the manuscript.
EOF
    exit 1
fi

# ------------------------------------------------------------------
# Build one variant
# ------------------------------------------------------------------
build_variant() { # $1=srcdir $2=tag
    local src="$1" tag="$2" bdir="/tmp/ab-build-$tag"
    echo "--- building $tag from $src ---"
    rm -rf "$bdir"
    ( cd /opt/mellanox/doca/applications \
      && sudo rm -f gpu_packet_processing \
      && sudo ln -sfn "$src" gpu_packet_processing \
      && sudo env PATH="/usr/local/cuda/bin:$PATH" \
           PKG_CONFIG_PATH="/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/flexio/lib/pkgconfig" \
           meson "$bdir" -Denable_all_applications=false -Denable_gpu_packet_processing=true \
      && sudo env PATH="/usr/local/cuda/bin:$PATH" ninja -C "$bdir" ) \
      > "$OUTDIR/build_$tag.log" 2>&1
    local rc=$?
    ( cd /opt/mellanox/doca/applications && sudo rm -f gpu_packet_processing )
    if [ $rc -ne 0 ]; then
        echo "BUILD FAILED for $tag; see $OUTDIR/build_$tag.log" >&2
        return 1
    fi
    BIN="$(find "$bdir" -maxdepth 3 -type f -name 'GAZSI' -o -maxdepth 3 -type f -name 'gpunet' | head -1)"
    [ -n "$BIN" ] || { echo "no binary produced for $tag" >&2; return 1; }
    echo "$BIN"
}

start_server() { # $1=binary $2=tag -> sets SRV_PID
    DOCA_NO_WARMUP=1 "$1" -g "$GPU_PCI" -n "$NIC_PCI" -q 4 -s -e "$ENGINE" \
        > "$OUTDIR/server_$2.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 60); do
        sleep 1
        ip netns exec "$NETNS" curl -s --max-time 2 \
            "http://$SRV_IP:$SRV_PORT/inference?d=ping" >/dev/null 2>&1 && return 0
        kill -0 "$SRV_PID" 2>/dev/null || return 1
    done
    return 1
}

stop_server() {
    [ -n "${SRV_PID:-}" ] || return 0
    kill -INT "$SRV_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    SRV_PID=""
}

# Fixed request set, identical for both variants: equivalence must be exact.
PROBES=("hello world" "warmup_0" "the quick brown fox jumps over the lazy dog"
        "a" "" "tokens+with+plus" "percent%20encoded" "many words here to tokenize now")

capture_bodies() { # $1=raw outfile  $2=normalised outfile
    : > "$1"; : > "$2"
    local p
    for p in "${PROBES[@]}"; do
        local enc; enc="$(printf '%s' "$p" | sed 's/ /+/g')"
        local body code
        # -w captures the status so success/error is compared, not just the body
        body="$(ip netns exec "$NETNS" curl -s --max-time 5 -w '\n%{http_code}' \
                "http://$SRV_IP:$SRV_PORT/inference?d=$enc" 2>/dev/null)"
        code="$(printf '%s' "$body" | tail -n1)"
        body="$(printf '%s' "$body" | sed '$d')"

        printf '%s\tHTTP=%s\t%s\n' "$p" "$code" "$body" >> "$1"

        # Normalisation: inference_time_us and batch_size are intentionally
        # dynamic (wall clock, and how many requests happened to co-arrive), so
        # they are masked. Everything that must match is kept: the echoed input,
        # the token count, the embedding values, and the HTTP status.
        local norm
        norm="$(printf '%s' "$body" \
                | sed -E 's/"inference_time_us":[0-9-]+/"inference_time_us":<DYN>/g; \
                          s/"batch_size":[0-9]+/"batch_size":<DYN>/g')"
        printf '%s\tHTTP=%s\t%s\n' "$p" "$code" "$norm" >> "$2"
    done
}

# ------------------------------------------------------------------
# Load generation
#
# Uses the repo's performance_client if present (the root Makefile builds it),
# else falls back to a fixed-duration concurrent curl driver. Whichever is used
# is used for BOTH variants, so the comparison stays apples-to-apples.
# ------------------------------------------------------------------
run_load() { # $1=concurrency $2=outfile -> "requests rps avg_ms p99_ms non2xx"
    local conc="$1" out="$2"
    # One wrk thread per 8 connections, at least 1, capped at nproc.
    local threads=$(( (conc + 7) / 8 ))
    [ "$threads" -lt 1 ] && threads=1
    local ncpu; ncpu=$(nproc 2>/dev/null || echo 4)
    [ "$threads" -gt "$ncpu" ] && threads=$ncpu

    ip netns exec "$NETNS" wrk -t"$threads" -c"$conc" -d"${SECS}s" --latency \
        "http://$SRV_IP:$SRV_PORT/inference?d=bench+request+payload" > "$out" 2>&1

    # wrk output:
    #   N requests in Xs, ...
    #   Requests/sec:  1234.56
    #   Latency   avg   stdev   max
    #   99%   12.34ms
    #   Non-2xx or 3xx responses: N        (absent when all succeeded)
    awk '
        /requests in/            { reqs = $1 }
        /^Requests\/sec:/        { rps = $2 }
        /^ *Latency/ && $2 ~ /[0-9]/ { avg = $2 }
        /^ *99%/                 { p99 = $2 }
        /Non-2xx or 3xx/         { bad = $NF }
        END {
            if (bad == "") bad = 0
            printf "%d %s %s %s %d\n", reqs+0, (rps==""?"0":rps), (avg==""?"0":avg), (p99==""?"0":p99), bad+0
        }' "$out" | sed 's/ms//g; s/us//g'
}

pcie_sample() { # $1=duration -> "rx_MBs tx_MBs" averaged, or "NA NA"
    if [ "$PCIE_OK" != "1" ]; then echo "NA NA"; return; fi
    timeout "$1" nvidia-smi dmon -s t -d 1 2>/dev/null \
      | awk '/^ *[0-9]/ { if ($2 ~ /^[0-9]+$/) { rx += $2; tx += $3; n++ } }
             END { if (n) printf "%.1f %.1f\n", rx/n, tx/n; else print "NA NA" }'
}

cpu_time_of() { # $1=pid -> cumulative CPU seconds
    awk '{ printf "%.2f\n", ($14 + $15) / '"$(getconf CLK_TCK)"' }' "/proc/$1/stat" 2>/dev/null || echo NA
}

# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
declare -A BIN
for pair in "new:$NEW_DIR" "base:$BASE_DIR"; do
    tag="${pair%%:*}"; dir="${pair#*:}"
    b="$(build_variant "$dir" "$tag")" || exit 1
    BIN[$tag]="$b"
    echo "$tag binary: $b"
done

for tag in base new; do
    for conc in "${CONCURRENCIES[@]}"; do
        for rep in $(seq 1 "$REPS"); do
            echo "=== $tag concurrency=$conc rep=$rep ==="
            start_server "${BIN[$tag]}" "${tag}_c${conc}_r${rep}" || {
                echo "server failed to come up for $tag" >&2; stop_server; exit 1; }

            [ "$rep" = 1 ] && capture_bodies "$OUTDIR/bodies_$tag.raw" "$OUTDIR/bodies_$tag.norm"

            cpu0="$(cpu_time_of "$SRV_PID")"
            pcie_sample "$SECS" > "$OUTDIR/pcie_${tag}_c${conc}_r${rep}.txt" &
            pcie_bg=$!

            read -r reqs rps avg p99 non2xx <<<"$(run_load "$conc" "$OUTDIR/load_${tag}_c${conc}_r${rep}")"

            wait $pcie_bg
            read -r prx ptx < "$OUTDIR/pcie_${tag}_c${conc}_r${rep}.txt"
            cpu1="$(cpu_time_of "$SRV_PID")"
            stop_server

            cpu="$(echo "scale=2; $cpu1 - $cpu0" | bc 2>/dev/null || echo NA)"
            echo "$tag,$conc,$rep,$reqs,$SECS,$rps,$avg,$p99,$non2xx,$cpu,$prx,$ptx" >> "$CSV"
            echo "  -> $rps req/s avg=${avg}ms p99=${p99}ms non2xx=${non2xx} cpu=${cpu}s pcie_rx=${prx} pcie_tx=${ptx}"
        done
    done
done

# ------------------------------------------------------------------
# Equivalence + regression gates
# ------------------------------------------------------------------
echo
echo "=== response equivalence (normalised) ==="
echo "compared: echoed input, tokens, embedding values, HTTP status"
echo "masked:   inference_time_us, batch_size (intentionally dynamic)"
equiv_fail=0
if diff -u "$OUTDIR/bodies_base.norm" "$OUTDIR/bodies_new.norm" > "$OUTDIR/bodies.norm.diff"; then
    echo "PASS: normalised responses identical across all probes"
else
    echo "FAIL: normalised responses differ — see $OUTDIR/bodies.norm.diff"
    head -30 "$OUTDIR/bodies.norm.diff"
    equiv_fail=1
fi

# HTTP success/error counts must match across variants at every concurrency
awk -F, 'NR>1 { bad[$1] += $9; tot[$1] += $4 }
END {
    printf "HTTP non-2xx/3xx: base=%d/%d new=%d/%d\n", bad["base"], tot["base"], bad["new"], tot["new"]
    if (bad["base"] != bad["new"]) { print "FAIL: error counts differ between variants"; exit 1 }
    print "PASS: error counts match"
}' "$CSV" || equiv_fail=1

echo
echo "=== regression gates (medians over $REPS reps) ==="
awk -F, 'NR > 1 {
    key = $1 "," $2
    rps[key] = rps[key] $6 " "; avg[key] = avg[key] $7 " "; p99[key] = p99[key] $8 " "
}
function med(s, a, n, i) {
    n = split(s, a, " "); m = 0
    for (i = 1; i <= n; i++) if (a[i] != "") b[++m] = a[i] + 0
    for (i = 1; i < m; i++) for (j = i + 1; j <= m; j++) if (b[j] < b[i]) { t = b[i]; b[i] = b[j]; b[j] = t }
    return (m % 2) ? b[(m + 1) / 2] : (b[m / 2] + b[m / 2 + 1]) / 2
}
END {
    split("1 16 32", cs, " ")
    printf "%-6s %10s %10s %8s   %-s\n", "conc", "metric", "base", "new", "ratio / verdict"
    fail = 0
    for (c in cs) {
        conc = cs[c]
        bk = "base," conc; nk = "new," conc
        if (!(bk in rps) || !(nk in rps)) continue

        br = med(rps[bk]); nr = med(rps[nk])
        ba = med(avg[bk]); na = med(avg[nk])
        bp = med(p99[bk]); np = med(p99[nk])

        r = (br > 0) ? nr / br : 0
        printf "%-6s %10s %10.2f %8.2f   %.3fx %s\n", conc, "rps", br, nr, r, (r < 0.95 ? "REGRESSION" : "ok")
        if (r < 0.95) fail = 1
        r = (ba > 0) ? na / ba : 0
        printf "%-6s %10s %10.3f %8.3f   %.3fx %s\n", conc, "avg_ms", ba, na, r, (r > 1.05 ? "REGRESSION" : "ok")
        if (r > 1.05) fail = 1
        r = (bp > 0) ? np / bp : 0
        printf "%-6s %10s %10.3f %8.3f   %.3fx %s\n", conc, "p99_ms", bp, np, r, (r > 1.05 ? "REGRESSION" : "ok")
        if (r > 1.05) fail = 1
    }
    print ""
    print fail ? "VERDICT: REGRESSION -> profile and fix before accepting" \
               : "VERDICT: within gates (rps >= 95%, avg/p99 <= 105%)"
    exit fail
}' "$CSV"
gate_rc=$?

if [ "$equiv_fail" -ne 0 ]; then
    echo
    echo "VERDICT: response equivalence FAILED — a performance verdict is meaningless"
    echo "         until the two variants produce the same answers."
    gate_rc=1
fi

echo
echo "raw data: $CSV"
echo "Targeted engineering check only. Do not transcribe these into the manuscript."
exit $gate_rc
