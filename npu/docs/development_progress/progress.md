# openfish/npu — progress (append-only)

Detailed session ledger (decisions, provenance, open questions):
`npu-adapting/9-16-npu-adapting-session/` (outside this repo).

## 2026-09-16 — Phase 0: CPU oracle (silu_mul template; covers all SUP kernels)

Env: zhang-ryzen2 (Ryzen AI MAX+ 395, 32 CPU, 30 GiB), slorado `ce74220`→`ba21b8e`, openfish `2e1a8cc`
(branch `npu-adapting` of keevin-xu/openfish), libtorch 2.9.0+rocm6.4 (CPU path),
input `test/PGXXXX230339/reads_1k.blow5` (md5 aec36d28647ad20cdf4b8d633f49fe54),
minimap2 2.24 + datamash 1.8 vs hg38noAlt.

1. Identity (gate item 3), `scripts/calculate_basecalling_accuracy.sh`:
   - fast@v5.0.0: median **0.9405915** (README 0.940374), n=1028 — PASS
   - hac@v5.0.0: median **0.9780595** (README 0.977594), n=1104 — PASS
   - sup@v5.0.0 `-C 128`: median **0.988506** (README 0.988561), n=1155 — PASS
     (default `-C 512` was SIGKILLed at 26.9 GB RSS; see debug_log)
2. Tensor dumps (gate item 4): slorado `ba21b8e` guarded `OPENFISH_DUMP` hooks; SUP layer 0, 4 of 128 rows,
   `~/p0/dump_sup` (28 files, 266 MB). Runtime shapes recorded in `../../TODO.md`.
3. References (gate item 5): `test/reference.py` (`silu_mul_ref`, `rmsnorm_ref`, `rotary_qkv_ref`, `gemm_ref`,
   plus `split_silu_mul_input` for the mlir-air (gate, up) order). `test/run_all.py` vs the dump:

   | check | shape | max_abs |
   |---|---|---|
   | silu_mul [y‖gate] | 4x1024x2048 | 3.576e-07 |
   | rmsnorm norm1 / norm2 | 4x1024x512 | 4.768e-07 / 7.153e-07 |
   | rotary q,k | 4x1024x3x8x64 | 0 |
   | gemm fc1 / fc2 | 4x1024x4096 / 4x1024x512 | 2.861e-06 / 6.199e-06 |
   | gemm wqkv / out_proj | 4x1024x3x8x64 / 4x1024x512 | 3.338e-06 / 1.013e-06 |
   | silu_mul swapped (negative control) | — | 4.183 (must differ) |

   All < 1e-5 → PASS.

**Phase 0: PASS** (all five gate items of `bc-phase-0-cpu-oracle`).
