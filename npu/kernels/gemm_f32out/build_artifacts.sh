#!/bin/bash
# bf16-in / f32-out GEMM ELFs for the same SUP linears as gemm_bf16 (same M per shape), both mmul variants:
#   $ARTIFACTS_DIR/gemm_f32out_bfp16/, $ARTIFACTS_DIR/gemm_f32out_native/ (point OPENFISH_NPU_GEMM_ARTIFACTS at one)
# Usage: [SHAPES="M K N ..."] [MMULS="bfp16 native"] build_artifacts.sh
set -euo pipefail
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAPES="${SHAPES:-4096 512 4096 4096 2048 512 8192 512 1536 16384 512 512 8192 512 1024 32768 256 512 16384 768 512 16384 1280 512}"
set -- $SHAPES
shapes=("$@")
for mmul in ${MMULS:-bfp16 native}; do
    for ((i = 0; i < ${#shapes[@]}; i += 3)); do
        python3 "$KDIR/build_artifact.py" --m "${shapes[i]}" --k "${shapes[i+1]}" --n "${shapes[i+2]}" \
            --mmul "$mmul" --out-dir "$ARTIFACTS_DIR/gemm_f32out_$mmul"
    done
done
