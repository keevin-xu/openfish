#!/bin/bash
# Build slorado binaries side by side into $OUT (default ~/p0/bin):
#   slorado-npu-dump, slorado-npu, slorado-cpu-dump, slorado-cpu
# The plain CPU build is built last, so ~/slorado/slorado stays the CPU oracle.
# openfish's lib and slorado's objects depend on HAVE_NPU/OPENFISH_DUMP but make does not
# track flags, so every variant is a clean rebuild. Run binaries with cwd = slorado root
# (libtorch rpath is relative). Usage: VARIANTS="npu cpu" build_slorado_variants.sh
set -euo pipefail

SLORADO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$HOME/p0/bin}"
JOBS="${JOBS:-16}"
VARIANTS="${VARIANTS:-npu-dump npu cpu-dump cpu}"

cd "$SLORADO"
mkdir -p "$OUT"
for v in $VARIANTS; do
    flags=""
    npu=()
    case "$v" in
        npu) npu=(npu=1) ;;
        npu-dump) npu=(npu=1); flags="-DOPENFISH_DUMP" ;;
        cpu) ;;
        cpu-dump) flags="-DOPENFISH_DUMP" ;;
        *) echo "unknown variant $v" >&2; exit 1 ;;
    esac
    echo "[variants] building $v (${npu[*]:-cpu} CPPFLAGS=$flags)"
    make -s -C openfish clean
    rm -f build/*.o build/*.d slorado
    CPPFLAGS="$flags" make -s -j"$JOBS" cxx11_abi=1 "${npu[@]}"
    cp slorado "$OUT/slorado-$v"
    echo "[variants] $OUT/slorado-$v $(sha256sum slorado | cut -c1-16)"
done
