#!/usr/bin/env python3
"""Compile one bf16 A x bfp16ebs8 B -> bf16 C GEMM (mlir-air matrix_multiplication/bf16_x_bfp16, copied unmodified)
into an ELF + gemm_k<K>_n<N>.json for nn_npu.cpp. Args (A_bf16[M,K], B_packed u8[N/tile_n, K/tile_k_l1, tile_bytes],
C_bf16[M,N]); B is the weight block-quantized offline (8 K-contiguous values share one exponent, 8-bit mantissas),
A stays bf16, accumulation is f32 in L1 and C is narrowed to bf16.
Tiles: the example's llama q-proj lit (TILE_M 64, TILE_N 64, TILE_K_L1 128, TILE_K_L2 = K, herd 8x4).
Backend settings: the example's common_kwargs, with output_format elf.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(KDIR))

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=4096)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

TILE = dict(tile_m=64, tile_k_l2=args.k, tile_k_l1=128, tile_n=64, herd_m=8, herd_n=4)
M, K, N = args.m, args.k, args.n
assert K <= 2048, "tile_k_l2 = K is only exercised upstream up to K = 2048"

build = KDIR / f"build_m{M}_k{K}_n{N}"
subprocess.run(["make", "-s", "-C", str(KDIR), "compile-kernel", f"BUILD_DIR={build.name}",
                f"TILE_M={TILE['tile_m']}", f"TILE_N={TILE['tile_n']}", f"TILE_K_L1={TILE['tile_k_l1']}"], check=True)
os.chdir(build)

from air.backend.xrt import XRTBackend  # noqa: E402
import matmul_bf16_x_bfp16 as mm  # noqa: E402

module = mm.build_module(M, K, N, TILE["tile_m"], TILE["tile_k_l2"], TILE["tile_k_l1"], TILE["tile_n"],
                         TILE["herd_m"], TILE["herd_n"])
backend = XRTBackend(omit_while_true_loop=False, output_format="elf", stack_size=2048, runtime_loop_tiling_sizes=[2, 2],
                     instance_name="matmul_bf16_x_bfp16")
artifact = backend.compile(module)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
name = f"gemm_bfp16w_m{M}_k{K}_n{N}.elf"
shutil.copyfile(artifact.output_binary, out / name)
manifest = {
    "kernel": "gemm_bf16_x_bfp16", "method": "bfp16w", "mmul": "bfp16ebs8-weights", "file": name, "format": "elf",
    "kernel_name": artifact.kernel, "m": M, "k": K, "n": N, **{k: v for k, v in TILE.items() if k.startswith("tile")},
    "tile_bytes": mm.bfp_tile_bytes(TILE["tile_n"], TILE["tile_k_l1"]),
    "herd": f"{TILE['herd_m']}x{TILE['herd_n']}", "args": "A_bf16,B_bfp16ebs8_packed,C_bf16",
    "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
}
(out / f"gemm_k{K}_n{N}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
