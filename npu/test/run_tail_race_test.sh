#!/bin/bash
# Build tail_race_test.cpp and run it on every NPU kernel kind slorado uses (shapes as in the 10-23 runs).
#   openfish/npu/test/run_tail_race_test.sh [launches=200]   (artifacts from build_artifacts.sh; takes the NPU lock)
set -uo pipefail
L=${1:-200}
here=$(cd "$(dirname "$0")" && pwd)
A=${OPENFISH_NPU_ARTIFACTS:-$here/../artifacts}
bin=${TAIL_RACE_BIN:-$HOME/p0/bin/tail_race_test}
g++ -O2 -std=c++17 "$here/tail_race_test.cpp" -lxrt_coreutil -o "$bin" || exit 2
run() { flock -x -w 600 "${NPU_LOCK:-$HOME/.npu.lock}" "$bin" "$@"; echo; }
M=1048576
# eltwise: axpy (x, y, out), silu (x, out), silu_mul (gate, up, out), rmsnorm (a, w, c), rope (a, lut, c)
run "$A/axpy.elf" main:axpy "$L" 2 2 $((8*M)) $((8*M)) $((8*M))
run "$A/silu.elf" main:silu "$L" 1 2 $((8*M)) $((8*M))
run "$A/silu_mul_n16777216_t4096_h8x1.elf" main:silu_and_mul "$((L/2))" 2 2 $((32*M)) $((32*M)) $((32*M))
run "$A/rmsnorm_n512.elf" main:weighted_rms_norm "$L" 2 2 $((8*M)) 1024 $((8*M))
run "$A/rope_d64.elf" main:rope_halfsplit "$L" 2 2 $M $M $M
# fused norm2 + wqkv + RoPE: 13 args, observe k_roped (12) and q_roped (11)
R=$((512*1024)); W=$((512*512*2))
for o in 12 11; do
  run "$A/rms_qkv_rope.elf" main:rms_gemms_rope "$L" $o 2 $R 1024 $R $W $R $W $R $W $R $R $R $R $R
done
# GEMMs at fc1 (4096x512x4096) and fc2 (4096x2048x512): f32-out native (comb12) and fused-cast bfp16 (C_f32 + C_bf16)
run "$A/gemm_f32out_native/gemm_f32out_m4096_k512_n4096_native.elf" main:matmul_bf16 "$((L/4))" 2 4 $((4*M)) $((4*M)) $((64*M))
run "$A/gemm_f32out_native/gemm_f32out_m4096_k2048_n512_native.elf" main:matmul_bf16 "$((L/4))" 2 4 $((16*M)) $((2*M)) $((8*M))
run "$A/gemm_bfp16/gemm_m4096_k512_n4096_bfp16.elf" main:gemm_cast_bf16 "$((L/4))" 3 2 $((4*M)) $((4*M)) $((64*M)) $((32*M))
