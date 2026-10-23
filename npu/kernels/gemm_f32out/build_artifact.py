#!/usr/bin/env python3
"""Compile one bf16-in / f32-out GEMM (mlir-air matrix_multiplication/bf16_in_fp32_out, external mm.o codegen)
into an ELF + gemm_k<K>_n<N>.json for nn_npu.cpp. One launch, args (A_bf16[M,K], B_bf16[K,N], C_f32[M,N]):
the f32 accumulator is the output, so there is no cast launch and no bf16 output buffer.

--mmul bfp16: the example's mm.o (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16; fast, lossy on heavy-tailed data)
--mmul native: the same mm_aie2p.cc without that define (identical source to gemm_bf16's, validated in R15).
Tiles and backend settings are the example's defaults (README "Tile config", run.py __main__).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

KDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KDIR))

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=4096)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--mmul", choices=["bfp16", "native"], required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

TILE = dict(tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4)
M, K, N = args.m, args.k, args.n
for c in (M % (TILE["tile_m"] * TILE["herd_m"]), K % TILE["tile_k_l2"], N % (TILE["tile_n"] * TILE["herd_n"])):
    assert c == 0, f"illegal tiling for {M}x{K}x{N}"

build = KDIR / f"build_{args.mmul}_m{M}_k{K}_n{N}"
mk = subprocess.run(["make", "-s", "-n", "-C", str(KDIR), "compile-kernel", f"TILE_M={TILE['tile_m']}"],
                    check=True, capture_output=True, text=True).stdout
flag = "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"
assert flag in mk
if args.mmul == "native":
    mk = mk.replace(flag, "")
mk = mk.replace("build_peano", build.name)
subprocess.run(mk, shell=True, check=True, cwd=KDIR, executable="/bin/bash")
os.chdir(build)

from air.backend.xrt import XRTBackend  # noqa: E402
import run as gemm  # noqa: E402

module = gemm.build_module(M, K, N, TILE["tile_m"], TILE["tile_k_l2"], TILE["tile_k_l1"], TILE["tile_n"],
                           TILE["herd_m"], TILE["herd_n"], bfloat16, np.float32, arch="aie2p", emit_external_call=True)
backend = XRTBackend(target_device="npu2", omit_while_true_loop=False, runtime_loop_tiling_sizes=[2, 2],
                     stack_size=2048, output_format="elf", instance_name="matmul_bf16")
artifact = backend.compile(module)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
name = f"gemm_f32out_m{M}_k{K}_n{N}_{args.mmul}.elf"
shutil.copyfile(artifact.output_binary, out / name)
manifest = {
    "kernel": "gemm_bf16_in_fp32_out", "method": "f32-out", "mmul": args.mmul, "file": name, "format": "elf",
    "kernel_name": artifact.kernel, "m": M, "k": K, "n": N, **{k: v for k, v in TILE.items() if k.startswith("tile")},
    "herd": f"{TILE['herd_m']}x{TILE['herd_n']}", "args": "A_bf16,B_bf16,C_f32",
    "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
}
(out / f"gemm_k{K}_n{N}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
