# axpy — provenance (target A7, Kind: EXAMPLE)
Copied unmodified from mlir-air `programming_examples/axpy/` on the server clone (6746658f): axpy.py
61a84f13d9a7d665ef4faf17baf76a8b, Makefile 9aa42577fed5356c4fc3eda9212ce34a. Ours: build_artifact.py only
(`build_axpy(n, tile_n, alpha, bf16)`, the example's runtime_loop_tiling_sizes [4, 4]).
SUP use: deepnorm residual k = alpha * x + in (x = residual stream, in = attention / FFN output), alpha = 2.4494897
compiled in; nn_npu checks the runtime alpha against the manifest.
