# gemm_bf16 — provenance

Copied unmodified from `mlir-air/programming_examples/matrix_multiplication/bf16_in_bf16_out/`
(MIT). md5 identical on Mac `12e6f9f5` and server `6746658f` (= installed wheel):
run.py 0fb3f468ab81d5d2db5996a77be6c2de, mm_aie2p.cc 475c98f64974bc594db3a4a44f461548,
mm.cc 3cfd2984b58908388989e0326e56ab7f, zero.cc 964587d7bfdeda6ba9de0c50228c6106,
Makefile d5db54b3adf82ef3dca054e3a30095c1.

Registry: `kernel_registry/details/GEMM_bf16_in_bf16_out.md`. `C[M,N] = A[M,K] @ B[K,N]`, high precision
(FP32 accumulate, single bf16 cast), tolerance rtol 1.6e-2 / atol 1.5e-3, herd 8x4 (32 tiles).
`--method auto`: fused-cast (TILE_M=64) when M*K*N >= 4e9, else drain (needs TILE_M=32).
Legality: M % (tile_m*herd_m) == 0, K % tile_k_l2 == 0, N % (tile_n*herd_n) == 0 (NOT asserted: silent corruption).

Run with a private runner lock: `TMPDIR=~/.npu-tmp flock -x -w 1800 $NPU_LOCK make run M=.. K=.. N=.. TILE_M=..`.
