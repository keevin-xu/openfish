#!/bin/bash
# Build the standalone zero-copy iGPU <-> NPU test (zero_copy_test.cpp, porting_targets.md B1 step 1).
# Run in a shell WITHOUT ~/npu-env.sh (its LD_LIBRARY_PATH breaks ROCm); XRT 2.21 is the distro one under /usr.
#   openfish/npu/test/build_zero_copy_test.sh ~/p0/bin/zero_copy_test
set -euo pipefail
out=${1:-zero_copy_test}
/opt/rocm/bin/hipcc -x hip --offload-arch=gfx1151 -O2 -std=c++17 "$(dirname "$0")/zero_copy_test.cpp" -lxrt_coreutil -o "$out"
echo "built $out"
