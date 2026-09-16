# openfish/npu — lessons (append-only)

## 2026-09-16
- The CPU oracle for `silu_mul` and `rmsnorm` is **torch** in `TxModel.cpp` (non-`USE_GPU` branch), not
  `nn_cpu.c` — `nn_cpu.c` only implements `rotary_emb`. openfish's GPU kernels are only reachable under `USE_GPU`.
- openfish `silu_mul` input is `[y ‖ gate]` per row; mlir-air `silu_and_mul` takes `(gate, up, out)` as
  separate tensors. A negative control (swapped layout, max_abs 4.18) proves the check can fail.
- SUP transformer T is chunk/12 = 1024 (5 conv layers), not chunk/6; the upsample x2 restores 2048 afterwards.
- Encoder input activations are non-contiguous (strides 524288,1,1024) — an NPU wrapper must `contiguous()` or
  honour strides.
- Regenerating the rotary sin/cos tables in numpy differs from torch's by ~6e-5 (float32 `pow`/`outer` at
  positions up to 2047). Treat the tables as model inputs (use torch's), not as something to recompute.
- `slorado -C` (gpu batch size) also governs CPU memory: sup at the default 512 exceeds 30 GiB.
- Build-flag trick without editing the Makefile: `CPPFLAGS=-DFOO make` (env var; the Makefile's `+=` appends).
  `make CPPFLAGS+=...` on the command line would *replace* the Makefile's include paths.
