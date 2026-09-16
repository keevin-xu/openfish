# silu_mul kernel — provenance

Copied from `mlir-air/programming_examples/silu_and_mul/` (Xilinx/mlir-air, MIT).
Identical in the server clone `6746658f` (matches the installed wheel `0.0.1.2026090504+6746658`)
and the Mac clone `12e6f9f5`:

| file | md5 | modified here |
|---|---|---|
| silu_and_mul.py | 626209006696c729ebc2fbec7d1e8bfd | no |
| silu_and_mul.cc | 6c06126ff0b68181a5f24467e933d97d | no |
| Makefile | 10e441698d06d9722ad3662b7cd41551 (upstream) | yes: `TMPDIR` for the runner lock, `artifact` target |

Registry: `kernel_registry/details/SiLU_Mul_bf16.md` — best config `tile_n=4096, herd 8x1`, bf16,
`rtol=1.6e-2, atol=8e-2`, argument order `(gate, up, out)`.

openfish layout adapter (host side, Tier 1): `x[..., 2K] = [y ‖ gate]` -> `gate = x[..., K:]`, `up = x[..., :K]`
(`test/reference.py:split_silu_mul_input`). Costs one copy of each half per call. TODO(Tier 2): a strided
variant that reads `[y ‖ gate]` rows directly.
