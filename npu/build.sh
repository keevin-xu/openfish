#!/bin/bash
# Build every NPU kernel under kernels/<name>/ into artifacts/.
# Each kernel dir provides a Makefile with an `artifact` target that writes into $ARTIFACTS_DIR.
set -euo pipefail

NPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARTIFACTS_DIR="${NPU_DIR}/artifacts"

[ -n "${MLIR_AIR_INSTALL_DIR:-}" ] || { echo "MLIR_AIR_INSTALL_DIR unset: source ~/npu-env.sh first" >&2; exit 1; }
[ -n "${PEANO_INSTALL_DIR:-}" ] || { echo "PEANO_INSTALL_DIR unset: source ~/npu-env.sh first" >&2; exit 1; }

mkdir -p "${ARTIFACTS_DIR}"
built=0
for mk in "${NPU_DIR}"/kernels/*/Makefile; do
    [ -e "$mk" ] || continue
    dir="$(dirname "$mk")"
    echo "[build.sh] $(basename "$dir")"
    make -C "$dir" artifact
    built=$((built + 1))
done
echo "[build.sh] built ${built} kernel(s) into ${ARTIFACTS_DIR}"
