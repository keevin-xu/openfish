# Phase 1 — kernel × shape tracking (append-only)

| kernel | shape (N / layout) | source | herd (tiles in flight / target) | rtol / atol | mean_rel_L1 | abs_err max | rel_err max | verdict | latency (kernel-only) | artifact sha256 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| silu_mul | 16,777,216 randn (gate, up) | `kernels/silu_mul` `make run` / `make profile` | 8×1, tile_n 4096 (8 / 8) | 1.6e-2 / 8e-2 | 1.024e-2 | 1.250e-1 | 4.112e-1 | PASS | 4060.1 µs mean ×30 → 24.8 GB/s | xclbin 4cb6352d…d976, insts 4af5c725…73c1, .o 7db1e50e…774f | matches registry 16777216 row (4016 µs, 1.024e-2, 0.125) |
| silu_mul | 16,777,216 real SUP L0 `[8,1024,4096]` `[y‖gate]` → split | `test/test_silu_mul_npu.py` case 2 | 8×1 (8 / 8) | 1.6e-2 / 8e-2 | 8.858e-3 | 4.688e-2 | 2.000e-1 | PASS | — (invoke wall incl. BO alloc/sync 73 ms) | same | vs CPU fp32 dump: mean_rel_L1 9.289e-3, max_abs 6.112e-2, 0 outside tol; host cast 11–12 ms, split 7 ms |
| silu_mul | negative control: gate/up swapped | case 3 | 8×1 | 1.6e-2 / 8e-2 | — | — | — | FAILS as required (408,757 elements outside tol) | — | same | proves the check has teeth and the split order matters |

Harness run (case 1 timing): 4267.1 µs mean ×30 (23.6 GB/s). Clean rebuild (`rm -rf build_peano`) reproduced all numbers.
