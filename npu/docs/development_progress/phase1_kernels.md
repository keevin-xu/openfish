# Phase 1 — kernel × shape tracking (append-only)

| kernel | shape (N / layout) | source | herd (tiles in flight / target) | rtol / atol | mean_rel_L1 | abs_err max | rel_err max | verdict | latency (kernel-only) | artifact sha256 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| silu_mul | 16,777,216 randn (gate, up) | `kernels/silu_mul` `make run` / `make profile` | 8×1, tile_n 4096 (8 / 8) | 1.6e-2 / 8e-2 | 1.024e-2 | 1.250e-1 | 4.112e-1 | PASS | 4060.1 µs mean ×30 → 24.8 GB/s | xclbin 4cb6352d…d976, insts 4af5c725…73c1, .o 7db1e50e…774f | matches registry 16777216 row (4016 µs, 1.024e-2, 0.125) |
| silu_mul | 16,777,216 real SUP L0 `[8,1024,4096]` `[y‖gate]` → split | `test/test_silu_mul_npu.py` case 2 | 8×1 (8 / 8) | 1.6e-2 / 8e-2 | 8.858e-3 | 4.688e-2 | 2.000e-1 | PASS | — (invoke wall incl. BO alloc/sync 73 ms) | same | vs CPU fp32 dump: mean_rel_L1 9.289e-3, max_abs 6.112e-2, 0 outside tol; host cast 11–12 ms, split 7 ms |
| silu_mul | negative control: gate/up swapped | case 3 | 8×1 | 1.6e-2 / 8e-2 | — | — | — | FAILS as required (408,757 elements outside tol) | — | same | proves the check has teeth and the split order matters |

Harness run (case 1 timing): 4267.1 µs mean ×30 (23.6 GB/s). Clean rebuild (`rm -rf build_peano`) reproduced all numbers.

## 2026-10-23 — direct mlir-air example ports (A5, A7, A8, A2; A6 below when run)
Real SUP tensors (CPU dumps `~/p0/dump_sup16`, `~/p0/dump_cpu18`), the example's own tolerance, artifacts built by each
kernel's `build_artifact.py` (openfish `166d881`); server mlir-air 6746658f.

| kernel (target) | shape | source | rtol / atol | result | mean_rel_L1 | notes |
|---|---|---|---|---|---|---|
| gemm_f32out native (A5) | fc1 4096×512×4096, fc2 4096×2048×512, wqkv 8192×512×1536, out_proj 16384×512×512, upsample 8192×512×1024, crf 4096×512×4096 | `test_gemm_npu.py --kernel f32out --mmul native` | 1.6e-2 / 1.5e-3 vs fp32 product | **PASS 6/6** (0 outside tol) | 1.2e-7 – 4.3e-7 | vs CPU dump cosine ≥ 0.9999983; kernel 9.4–18.9 ms (1.2–1.3× torch fp32 8 thr) |
| gemm_f32out bfp16 (A5) | same 6 | same, `--mmul bfp16` | same | FAIL 6/6 (1,238 – 5,008,840 outside) | 3.1e-3 – 8.8e-3 | same failure as the fused-cast bfp16 build (R15): BFP16 block emulation; vs dump cosine ≥ 0.99996; kernel 1.4–3.0 ms (5–8× torch) |
| axpy (A7) | n 4,194,304, alpha 2.4494897 | `test_examples_npu.py` (x = L0 norm1_residual, y = L0 norm1_in) | 1e-2 / 1e-8 | **FAIL** 287,848 / 4,194,304 outside | 2.86e-3 | correlation 0.999995. vs a reference using bf16(alpha) = 2.453125: **0 outside** — the air.api expression holds alpha as a bf16 constant (−0.15% on alpha·x); the kernel arithmetic is exact |
| silu (A8) | n 4,194,304 | same (x = L0 ff_fc1_out) | 1e-1 / 5e-2 | **PASS** 0 outside | 6.66e-3 | correlation 0.999913, max abs 3.1e-2 |
| rms_qkv_rope (A2) | 512 rows × 512, 8 heads × 64 | same (x = L0 norm2_in + α·norm2_residual, L0 norm2_weight, L1 wqkv, L1 rotary tables) | builder's own: 0.2 / 0.5, correlation ≥ 0.99 | **PASS** q_roped, k_roped (0 outside, corr 0.999972 / 0.999968) | 1.0e-2 | info normed corr 0.999985, v 0.999935; vs CPU dump cosine normed 0.999979, q 0.999970, k 0.999966, v 0.999934 |
| bfp16 weight packer (A6 host side) | 6 SUP weights | `test/check_bfp16_pack.py` (C++ `src/npu_bfp16_pack.h` vs example `pack_b_bfp16ebs8`) | byte-exact | **PASS** 7.67 MB identical | — | |
| gemm_bfp16w (A6) | fc1, fc2, wqkv, out_proj, upsample, crf (production M) | `test_gemm_bfp16w_npu.py` (dump_sup16) | example gate vs quantized-weight ref: 0.1 / 0.05, ≤ 0.05% mismatch, corr ≥ 0.999 | **PASS 6/6** | vs exact product 4.3e-3 – 1.1e-2 (weight quantization alone 3.2e-3 – 9.0e-3) | vs dump cosine ≥ 0.99994; kernel 1.9–4.6 ms; conv shapes 32768×256, 16384×768/1280/512 ×512 on random data corr 0.99998 |
