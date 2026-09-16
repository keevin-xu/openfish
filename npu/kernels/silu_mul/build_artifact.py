#!/usr/bin/env python3
"""Compile silu_mul (no NPU needed) into ARTIFACTS_DIR as an ELF + silu_mul.json manifest for nn_npu.cpp.

Uses the same compile settings as test/test_silu_mul_npu.py --output-format elf (the Phase-1 validated path).
Run from kernels/silu_mul/build_peano (silu_and_mul.o must be in cwd); `make artifact` does this.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from air.backend.xrt import XRTBackend  # noqa: E402
from silu_and_mul import build_module  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16777216)
ap.add_argument("--tile-n", type=int, default=4096)
ap.add_argument("--herd-x", type=int, default=8)
ap.add_argument("--herd-y", type=int, default=1)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
args = ap.parse_args()

module = build_module(args.n, args.tile_n, bfloat16, herd_x=args.herd_x, herd_y=args.herd_y)
backend = XRTBackend(target_device="npu2", omit_while_true_loop=False, output_format="elf",
                     instance_name="silu_and_mul")
artifact = backend.compile(module)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
name = f"silu_mul_n{args.n}_t{args.tile_n}_h{args.herd_x}x{args.herd_y}.elf"
shutil.copyfile(artifact.output_binary, out / name)
sha = hashlib.sha256((out / name).read_bytes()).hexdigest()
manifest = {
    "kernel": "silu_mul",
    "file": name,
    "format": "elf",
    "kernel_name": artifact.kernel,
    "n": args.n,
    "tile_n": args.tile_n,
    "herd": f"{args.herd_x}x{args.herd_y}",
    "dtype": "bf16",
    "args": "gate,up,out",
    "sha256": sha,
}
(out / "silu_mul.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
