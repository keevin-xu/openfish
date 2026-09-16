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

## 2026-09-16 (Phase 1, silu_mul)
- The installed `XRTRunner.run_test` locks `tempfile.gettempdir()/npu.lock`, i.e. `/tmp/npu.lock`, which another user
  owns on zhang-ryzen2. Set `TMPDIR=~/.npu-tmp` for mlir-air example runs instead of patching mlir-air.
- Don't take `$NPU_LOCK` inside a Python harness that is launched under `flock $NPU_LOCK`: the child inherits the
  locked fd and a second open+flock would deadlock. Lock from outside only.
- `XRTBackend.last_latency_us` is a **mean** over `n_perf_iters` (kernel-only, syncs excluded) — fine for registry
  comparability; our own G2 timing must report median/p95.
- The Python invoker allocates and maps new BOs on every call: 73 ms per call vs 4.3 ms of kernel time at N=16.8M.
  Real SUP activations are *easier* than randn for bf16 SiLU-Mul (8.9e-3 vs 1.0e-2) because |gate| is smaller.
- aircc compile of this 8-tile elementwise design takes ~0.2 s on the server (verified from a clean build dir).
