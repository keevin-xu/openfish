# rms_qkv_rope — provenance (target A2, Kind: EXAMPLE)
`rms_gemms_rope_multi.py` copied unmodified from mlir-air `programming_examples/llms/shared/builders/` on the server
clone (6746658f), md5 4adb38f8b00b963d4400751c9e210c17. It imports mlir-air's own `shared.infra` (stitching, cache,
external_kernels, backend_presets), `shared.builders.gemm_builder` and `kernel_registry` from the mlir-air tree
(read-only, `MLIR_AIR_SRC`). External objects are built exactly as llama32_1b_int4_prefill does (mm_m32.o, mm_m64.o
via compile_gemm_mm; rope.o from rope_halfsplit.cc via prepare_air_project).
Ours: build_artifact.py only. SUP dims: emb 512, 8 heads x 64, kv_dim 512 (MHA), full half-split rotary
(SUP rotary_half 32 = head_dim/2, same LUT row [cos(32) | sin(32)] as kernels/rope).
Rows per launch = 512: the builder looks every GEMM shape up in kernel_registry and 8192x512x512 is not measured
there (KeyError); adding it would mean editing mlir-air. 512x512x512 resolves to the drain method (tile_m 32).
GEMM microkernel is BFP16-emulated (compile_gemm_mm default; mlir-air notes its non-BFP16 path is not usable
at these tiles), so this op has no native variant.
SUP use (TxModel.cpp): layer i's norm2 (deepnorm residual folded first) + layer i+1's wqkv + RoPE in one call;
layer 0's attention input is the conv stack output (no norm), so layer 0 keeps the unfused path.
