# silu — provenance (target A8, Kind: EXAMPLE)
Copied unmodified from mlir-air `programming_examples/silu/` on the server clone (6746658f): silu.py
f1eac423ecb79f50a96648bd765c6e13, Makefile abd254ad888e3f0b24f653eeb7aa5343. Ours: build_artifact.py only
(`build_silu(n, tile_n)`, the example's runtime_loop_tiling_sizes [4, 4]). npu2 / bf16 only (example docstring).
SUP use: swish after each of the 5 conv layers (ConvStack, activation "swish").
