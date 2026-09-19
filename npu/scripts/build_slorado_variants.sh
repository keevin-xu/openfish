#!/bin/bash
# Build slorado binaries side by side into $OUT (default ~/p0/bin):
#   slorado-npu-dump, slorado-npu, slorado-cpu-dump, slorado-cpu,
#   slorado-rocm, slorado-rocm-dump (iGPU), slorado-rocm-npu, slorado-rocm-npu-dump (iGPU + NPU)
# The plain CPU build is built last, so ~/slorado/slorado stays the CPU oracle.
# openfish's lib and slorado's objects depend on HAVE_NPU/OPENFISH_DUMP but make does not
# track flags, so every variant is a clean rebuild. Run binaries with cwd = slorado root
# (libtorch rpath is relative). Usage: VARIANTS="npu cpu" build_slorado_variants.sh
# rocm: iGPU (gfx1151) build against a separate ROCm-7.2 libtorch ($LIBTORCH_ROCM, default
# ~/torch-rocm72/libtorch; thirdparty/torch is rocm6.4 without gfx1151 BLAS). Build and run it WITHOUT
# ~/npu-env.sh (its LD_LIBRARY_PATH loads distro HSA 5.7.1); run with LD_LIBRARY_PATH=/opt/rocm/lib.
set -euo pipefail

SLORADO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$HOME/p0/bin}"
JOBS="${JOBS:-16}"
VARIANTS="${VARIANTS:-npu-dump npu cpu-dump cpu}"
LIBTORCH_ROCM="${LIBTORCH_ROCM:-$HOME/torch-rocm72/libtorch}"

cd "$SLORADO"
mkdir -p "$OUT"
for v in $VARIANTS; do
    flags=""
    npu=()
    ldpath="${LD_LIBRARY_PATH:-}"
    case "$v" in
        npu) npu=(npu=1) ;;
        npu-dump) npu=(npu=1); flags="-DOPENFISH_DUMP" ;;
        cpu) ;;
        cpu-dump) flags="-DOPENFISH_DUMP" ;;
        rocm|rocm-dump|rocm-npu|rocm-npu-dump)
              npu=(rocm=1 "ROCM_ARCH=--offload-arch=gfx1151" "LIBTORCH_DIR=$LIBTORCH_ROCM")
              [[ "$v" == *npu* ]] && npu+=(npu=1)
              [[ "$v" == *dump ]] && flags="-DOPENFISH_DUMP"
              ldpath="/opt/rocm/lib" ;;  # libtorch's ROCm deps (hipblas, miopen, ...) resolve at link time too
        *) echo "unknown variant $v" >&2; exit 1 ;;
    esac
    echo "[variants] building $v (${npu[*]:-cpu} CPPFLAGS=$flags)"
    make -s -C openfish clean
    rm -f build/*.o build/*.d slorado
    CPPFLAGS="$flags" LD_LIBRARY_PATH="$ldpath" make -s -j"$JOBS" cxx11_abi=1 "${npu[@]}"
    cp slorado "$OUT/slorado-$v"
    echo "[variants] $OUT/slorado-$v $(sha256sum slorado | cut -c1-16)"
done
