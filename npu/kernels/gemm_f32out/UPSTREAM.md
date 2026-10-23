# gemm_f32out — provenance (target A5, Kind: EXAMPLE)
Copied unmodified from mlir-air `programming_examples/matrix_multiplication/bf16_in_fp32_out/` on the server clone
(6746658f): run.py e38aa83e5eaa676ad41b6cc9cd1b1084, mm_aie2p.cc 475c98f64974bc594db3a4a44f461548 (identical to
bf16_in_bf16_out's, used by gemm_bf16), mm.cc 3cfd2984b58908388989e0326e56ab7f, zero.cc 964587d7bfdeda6ba9de0c50228c6106,
Makefile 90930c158bc0e997e7f6d9b47e06fdff, README.md 3d94af7f4edea2ba8584d65527be0148.
Ours: build_artifact.py / build_artifacts.sh only (`build_module(..., emit_external_call=True)`, the example's
external codegen path and tiles). `--mmul native` drops the BFP16-emulation define, as gemm_bf16 does.
Single launch (A, B, C_f32): replaces the fused-cast GEMM's GEMM+cast pair when the caller wants an f32 result.
