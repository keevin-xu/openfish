#!/usr/bin/env python3
"""AXPY (mlir-air programming_examples/axpy, air.api, bf16, fused fma) for SUP's deepnorm residual:
out = alpha * x + y, alpha baked in at compile time (SUP deepnorm_alpha = 2.4494897).
Args (x[n], y[n], out[n]); manifest axpy.json with n elements per launch, inputs 2, alpha."""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from axpy import build_axpy  # noqa: E402
from air.api.types import bf16  # noqa: E402
from elf_artifact import compile_elf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4194304)
ap.add_argument("--tile-n", type=int, default=1024)
ap.add_argument("--alpha", type=float, default=2.4494897)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
a = ap.parse_args()
build = HERE / "build_peano"
build.mkdir(exist_ok=True)
os.chdir(build)
launch = build_axpy(n=a.n, tile_n=a.tile_n, alpha=a.alpha, dtype=bf16)
module = launch.build(target="npu2")
compile_elf(module, "axpy", a.out_dir, {"n": a.n, "tile_n": a.tile_n, "inputs": 2, "alpha": a.alpha, "args": "x,y,out"},
            instance_name="axpy", runtime_loop_tiling_sizes=[4, 4])
