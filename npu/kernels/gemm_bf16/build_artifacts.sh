#!/bin/bash
# fc1 (512->4096) and fc2 (2048->512) GEMMs at M=4096 rows, both mmul variants:
#   $ARTIFACTS_DIR/gemm_bfp16/, $ARTIFACTS_DIR/gemm_native/  (point OPENFISH_NPU_GEMM_ARTIFACTS at one)
set -euo pipefail
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for mmul in bfp16 native; do
    for kn in "512 4096" "2048 512"; do
        set -- $kn
        python3 "$KDIR/build_artifact.py" --k "$1" --n "$2" --mmul "$mmul" --out-dir "$ARTIFACTS_DIR/gemm_$mmul"
    done
done
