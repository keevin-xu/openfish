#!/bin/bash
# NPU bf16 GEMM (registry kernel) at slorado sup@v5.0.0 linear shapes, M rows per launch.
# Correctness (registry tolerance) + kernel-only mean latency in the same invocation.
# Usage (server): source ~/npu-env.sh && flock -x -w 3600 $NPU_LOCK npu/bench/gemm_sup.sh [M]
set -euo pipefail
NPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${1:-4096}"
ITERS="${ITERS:-30}"
export TMPDIR="${HOME}/.npu-tmp"; mkdir -p "$TMPDIR"
cd "$NPU_DIR/kernels/gemm_bf16"
#        name      K    N
for spec in "fc1 512 4096" "fc2 2048 512" "wqkv 512 1536" "out_proj 512 512"; do
    set -- $spec
    name=$1 K=$2 N=$3
    if (( M * K * N >= 4000000000 )); then TILE_M=64; else TILE_M=32; fi
    for c in "$((M % (TILE_M * 8)))" "$((K % 256))" "$((N % (128 * 4)))"; do
        [ "$c" -eq 0 ] || { echo "$name: illegal tiling for ${M}x${K}x${N}" >&2; exit 1; }
    done
    echo "=== $name ${M}x${K}x${N} TILE_M=$TILE_M"
    rm -rf build_peano
    make -s run M="$M" K="$K" N="$N" TILE_M="$TILE_M" PERF_ITERS="$ITERS" 2>&1 \
        | grep -E "method|Latency|precision|PASS|failed|ERROR|Error" || true
done
