#!/usr/bin/env python3
"""Half-split RoPE (registry rope_halfsplit + rope.o) for rows of head_dim: out = rot(x, LUT[row] = [cos..., sin...]).
Args (src[rows*D], lut[rows*D], dst[rows*D]); manifest rope_d<D>.json."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=8192)
ap.add_argument("--head-dim", type=int, default=64)
ap.add_argument("--herd-x", type=int, default=8)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
a = ap.parse_args()
subprocess.run(["make", "-s", "-C", str(HERE), "compile-kernel"], check=True)
os.chdir(HERE / "build_peano")
from rope_halfsplit import build_module  # noqa: E402
from elf_artifact import compile_elf  # noqa: E402

module = build_module(a.rows, a.head_dim, herd_x=a.herd_x).build(target="npu2")
n = a.rows * a.head_dim
compile_elf(module, f"rope_d{a.head_dim}", a.out_dir,
            {"m": a.rows, "n": a.head_dim, "a": n, "b": n, "c": n, "herd": f"{a.herd_x}x1"})
