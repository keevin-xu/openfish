#!/bin/bash
# Fused-cast bf16 GEMM ELFs for the sup@v5.0.0 linears, both mmul variants:
#   $ARTIFACTS_DIR/gemm_bfp16/, $ARTIFACTS_DIR/gemm_native/  (point OPENFISH_NPU_GEMM_ARTIFACTS at one)
# M rows per launch is chosen per shape so that M*K*N >= 4e9 (the registry's fused-cast regime).
#   fc1 + crf 512->4096 (M 4096) | fc2 2048->512 (M 4096) | wqkv 512->1536 (M 8192)
#   out_proj 512->512 (M 16384)  | upsample 512->1024 (M 8192)
# Usage: [SHAPES="M K N ..."] build_artifacts.sh
set -euo pipefail
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAPES="${SHAPES:-4096 512 4096 4096 2048 512 8192 512 1536 16384 512 512 8192 512 1024}"
set -- $SHAPES
shapes=("$@")
for mmul in bfp16 native; do
    for ((i = 0; i < ${#shapes[@]}; i += 3)); do
        python3 "$KDIR/build_artifact.py" --m "${shapes[i]}" --k "${shapes[i+1]}" --n "${shapes[i+2]}" \
            --mmul "$mmul" --out-dir "$ARTIFACTS_DIR/gemm_$mmul"
    done
done
