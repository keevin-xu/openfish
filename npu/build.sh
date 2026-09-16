#!/bin/bash
# Build every NPU kernel under kernels/<name>/ into artifacts/.
# Each kernel dir provides build_artifacts.sh, or a Makefile with an `artifact` target, writing into $ARTIFACTS_DIR.
set -euo pipefail

NPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARTIFACTS_DIR="${NPU_DIR}/artifacts"

[ -n "${MLIR_AIR_INSTALL_DIR:-}" ] || { echo "MLIR_AIR_INSTALL_DIR unset: source ~/npu-env.sh first" >&2; exit 1; }
[ -n "${PEANO_INSTALL_DIR:-}" ] || { echo "PEANO_INSTALL_DIR unset: source ~/npu-env.sh first" >&2; exit 1; }

mkdir -p "${ARTIFACTS_DIR}"
built=0
for dir in "${NPU_DIR}"/kernels/*/; do
    dir="${dir%/}"
    if [ -x "$dir/build_artifacts.sh" ]; then
        echo "[build.sh] $(basename "$dir") (build_artifacts.sh)"
        "$dir/build_artifacts.sh"
    elif [ -e "$dir/Makefile" ]; then
        echo "[build.sh] $(basename "$dir")"
        make -C "$dir" artifact
    else
        continue
    fi
    built=$((built + 1))
done
echo "[build.sh] built ${built} kernel(s) into ${ARTIFACTS_DIR}"
