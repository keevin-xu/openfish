# gemm_bfp16w — provenance (target A6, Kind: EXAMPLE)
Copied unmodified from mlir-air `programming_examples/matrix_multiplication/bf16_x_bfp16/` on the server clone
(6746658f): matmul_bf16_x_bfp16.py 396a8156b4f8328e7435cbe679665a6b, mm_bf16_x_bfp16.cc 44096b6c383b024f84cc701e9c8a2e64,
Makefile 8265c89f742a2051c5185b2f4277ff64 (test.cpp not needed).
Ours: build_artifact.py / build_artifacts.sh (the example's build_module, llama q-proj lit tiles TILE_M 64, TILE_N 64,
TILE_K_L1 128, TILE_K_L2 = K, herd 8x4; output_format elf instead of xclbin) and the host-side weight packer
`src/npu_bfp16_pack.h`, a C++ port of the example's `pack_b_bfp16ebs8()` checked byte-identical by
`test/check_bfp16_pack.py`. Output is bf16 only (no f32 accumulator leaves the core).
Numerics: A stays bf16; only the weights are block-quantized (8 K-contiguous values share an exponent, 8-bit
signed mantissas). The registry `bfp16` GEMM build emulates bf16 MMUL with BFP16 for both operands.
