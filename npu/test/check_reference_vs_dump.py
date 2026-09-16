#!/usr/bin/env python3
"""Phase-0 gate item 5: reference.py must reproduce slorado CPU tensor dumps.

Usage: check_reference_vs_dump.py [--dump-dir DIR] [--layer 0] [--tol 1e-5]
Exit 0 = every check max_abs < tol; 1 = a check failed; 2 = dump missing.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=os.environ.get("OPENFISH_DUMP_DIR"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    if not args.dump_dir or not Path(args.dump_dir, "manifest.tsv").exists():
        print(f"FAIL: no dump at {args.dump_dir!r} (set OPENFISH_DUMP_DIR or --dump-dir)")
        return 2
    d = Path(args.dump_dir)

    def L(site):
        return np.load(d / f"L{args.layer}_{site}.npy")

    alpha = float(L("norm_deepnorm_alpha"))
    sin, cos = L("attn_rotary_sin"), L("attn_rotary_cos")
    wqkv_bias = L("attn_wqkv_bias") if (d / f"L{args.layer}_attn_wqkv_bias.npy").exists() else None
    qkv_lin = L("attn_qkv_linear_out")

    checks = [
        ("silu_mul [y|gate]", lambda: ref.silu_mul_ref(L("ff_fc1_out")), L("ff_silu_mul_out")),
        ("rmsnorm norm1", lambda: ref.rmsnorm_ref(L("norm1_in"), L("norm1_residual"), L("norm1_weight"), alpha), L("norm1_out")),
        ("rmsnorm norm2", lambda: ref.rmsnorm_ref(L("norm2_in"), L("norm2_residual"), L("norm2_weight"), alpha), L("norm2_out")),
        ("rotary q,k", lambda: ref.rotary_qkv_ref(qkv_lin, sin, cos, 32), L("attn_qkv_rotary_out")),
        ("gemm fc1", lambda: ref.gemm_ref(L("ff_in"), L("ff_fc1_weight")), L("ff_fc1_out")),
        ("gemm fc2", lambda: ref.gemm_ref(L("ff_silu_mul_out"), L("ff_fc2_weight")), L("ff_fc2_out")),
        ("gemm wqkv", lambda: ref.gemm_ref(L("attn_in"), L("attn_wqkv_weight"), wqkv_bias).reshape(qkv_lin.shape), qkv_lin),
        ("gemm out_proj", lambda: ref.gemm_ref(L("attn_sdpa_out"), L("attn_out_proj_weight"), L("attn_out_proj_bias")), L("attn_out_proj_out")),
    ]

    ok = True
    print(f"dump: {d}  layer {args.layer}  gate max_abs < {args.tol:g}")
    print(f"{'check':<20} {'shape':<22} {'max_abs':>10} {'mean_rel_L1':>12}  verdict")
    for name, fn, expected in checks:
        m = ref.compare(fn(), expected)
        passed = m["max_abs"] < args.tol
        ok &= passed
        print(f"{name:<20} {str(tuple(expected.shape)):<22} {m['max_abs']:>10.3e} {m['mean_rel_L1']:>12.3e}  {'PASS' if passed else 'FAIL'}")

    # Teeth: the swapped silu_mul layout must NOT match.
    x = L("ff_fc1_out")
    k = x.shape[-1] // 2
    swapped = ref.compare(ref.silu(x[..., :k]) * x[..., k:], L("ff_silu_mul_out"))["max_abs"]
    teeth = swapped > 1e-2
    ok &= teeth
    print(f"{'silu_mul swapped':<20} {'(must differ)':<22} {swapped:>10.3e} {'':>12}  {'PASS' if teeth else 'FAIL'}")

    # Informational: tables regenerated in numpy vs the ones torch built.
    s2, c2 = ref.rotary_tables(64, sin.shape[0])
    print(f"info: rotary tables numpy vs dump max_abs sin {np.abs(s2 - sin).max():.3e} cos {np.abs(c2 - cos).max():.3e}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
