#!/usr/bin/env python3
"""Phase-2 gate: compare an NPU-path slorado tensor dump against the CPU dump of the same batch.

Every activation present in both dumps (same file, same shape) must be finite and have
whole-tensor cosine >= --min-cos; parameters/buffers must be bit-identical. Per-position
cosine (over the last dim) is reported. Exit 0 = PASS.
Usage: compare_dumps.py --ref ~/p0/dump_cpu18 --test ~/p0/dump_npu18 [--min-cos 0.99]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PARAMS = ("_weight", "_bias", "_alpha", "_sin", "_cos", "_mask")


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def per_position_min_cos(a, b):
    a = a.astype(np.float64).reshape(-1, a.shape[-1])
    b = b.astype(np.float64).reshape(-1, b.shape[-1])
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    return float((num[ok] / den[ok]).min()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()
    ref, test = Path(args.ref), Path(args.test)

    rows = list(csv.reader(open(ref / "manifest.tsv"), delimiter="\t"))
    ok = True
    n_act = n_param = 0
    per_layer = defaultdict(dict)
    print(f"{'layer':>5} {'site':<24} {'shape':<20} {'cosine':>12} {'pos_min_cos':>12} {'max_abs':>10} {'rel_L1':>10}  verdict")
    for layer, site, _full, _saved, _stride, _dtype, fname in rows:
        tf = test / fname
        if not tf.exists():
            print(f"{layer:>5} {site:<24} MISSING in test dump  FAIL")
            ok = False
            continue
        a, b = np.load(ref / fname), np.load(tf)
        if a.shape != b.shape:
            print(f"{layer:>5} {site:<24} shape {a.shape} vs {b.shape}  FAIL")
            ok = False
            continue
        if site.endswith(PARAMS):
            same = np.array_equal(a, b)
            n_param += 1
            if not same:
                print(f"{layer:>5} {site:<24} {str(a.shape):<20} parameter differs  FAIL")
                ok = False
            continue
        n_act += 1
        finite = bool(np.isfinite(b).all())
        c = cos(a, b)
        pmin = per_position_min_cos(a, b) if a.ndim >= 2 else float("nan")
        err = np.abs(a.astype(np.float64) - b)
        rel = float(err.mean() / (np.abs(a).mean() + 1e-300))
        passed = finite and c >= args.min_cos
        ok &= passed
        per_layer[int(layer)][site] = (c, pmin)
        print(f"{layer:>5} {site:<24} {str(a.shape):<20} {c:>12.8f} {pmin:>12.6f} {err.max():>10.3e} {rel:>10.3e}  "
              f"{'PASS' if passed else 'FAIL'}{'' if finite else ' (non-finite)'}")

    print(f"\n{n_act} activations compared (gate cosine >= {args.min_cos}, finite), {n_param} parameters bit-identical checked")
    print("per-layer summary (enc_out):")
    for layer in sorted(per_layer):
        if "enc_out" in per_layer[layer]:
            c, p = per_layer[layer]["enc_out"]
            print(f"  L{layer:<3} cosine {c:.8f}  per-position min {p:.6f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
