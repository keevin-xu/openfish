# rmsnorm — provenance
Copied unmodified from mlir-air `programming_examples/weighted_rms_norm/` (md5 weighted_rms_norm.py 81192774674fde3fed947e957d459155, Makefile bb9003b53b7a17906a4cf18a257b1550; identical on server 6746658f). Direct codegen, args (X[M,N], W[N], Y[M,N]), eps 1e-5. SUP deepnorm residual (in + residual*alpha) is computed on the host side before the call.
