#!/usr/bin/env python3
"""SiLU (mlir-air programming_examples/silu, air.api, bf16 via tanh) for the conv stack's swish activation:
out = x * sigmoid(x). Args (x[n], out[n]); manifest silu.json with n elements per launch, inputs 1."""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from silu import build_silu  # noqa: E402
from elf_artifact import compile_elf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4194304)
ap.add_argument("--tile-n", type=int, default=1024)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
a = ap.parse_args()
build = HERE / "build_peano"
build.mkdir(exist_ok=True)
os.chdir(build)
module = build_silu(n=a.n, tile_n=a.tile_n).build(target="npu2")
compile_elf(module, "silu", a.out_dir, {"n": a.n, "tile_n": a.tile_n, "inputs": 1, "args": "x,out"},
            instance_name="silu", runtime_loop_tiling_sizes=[4, 4])
