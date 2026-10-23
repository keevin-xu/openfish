#!/bin/bash
# bf16 x bfp16ebs8-weight GEMM ELFs for the same SUP linears (same M per shape) -> $ARTIFACTS_DIR/gemm_bfp16w/
# Usage: [SHAPES="M K N ..."] build_artifacts.sh
set -euo pipefail
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAPES="${SHAPES:-4096 512 4096 4096 2048 512 8192 512 1536 16384 512 512 8192 512 1024 32768 256 512 16384 768 512 16384 1280 512}"
set -- $SHAPES
shapes=("$@")
for ((i = 0; i < ${#shapes[@]}; i += 3)); do
    python3 "$KDIR/build_artifact.py" --m "${shapes[i]}" --k "${shapes[i+1]}" --n "${shapes[i+2]}" \
        --out-dir "$ARTIFACTS_DIR/gemm_bfp16w"
done
