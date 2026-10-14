#!/usr/bin/env python3
"""RMSNorm (registry weighted_rms_norm, direct codegen) at SUP's width: Y[M,N] = X·rsqrt(mean(X²)+eps)·W.
Args (X[M,N], W[N], Y[M,N]); manifest rmsnorm_n<N>.json with a = M*N, b = N, c = M*N elements."""
import argparse
import os
import sys
from pathlib import Path

from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from weighted_rms_norm import build_module  # noqa: E402
from elf_artifact import compile_elf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=8192)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--herd-x", type=int, default=8)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
a = ap.parse_args()
build = HERE / "build_peano"
build.mkdir(exist_ok=True)
os.chdir(build)
module = build_module(a.m, a.n, bfloat16, 16, herd_x=a.herd_x)
compile_elf(module, f"rmsnorm_n{a.n}", a.out_dir,
            {"m": a.m, "n": a.n, "a": a.m * a.n, "b": a.n, "c": a.m * a.n, "herd": f"{a.herd_x}x1", "eps": "1e-5"})
