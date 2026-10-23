#!/usr/bin/env python3
"""Byte-compare the C++ bfp16ebs8 packer (src/npu_bfp16_pack.h, used by nn_npu for kernels/gemm_bfp16w) with the
example's pack_b_bfp16ebs8() on real SUP weights. Exit 0 iff identical for every weight.
  cd ~/slorado/openfish/npu && python3 test/check_bfp16_pack.py --dump-dir ~/p0/dump_sup16"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernels" / "gemm_bfp16w"))
from matmul_bf16_x_bfp16 import pack_b_bfp16ebs8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dump-dir", required=True)
ap.add_argument("--tile-n", type=int, default=64)
ap.add_argument("--tile-k-l1", type=int, default=128)
a = ap.parse_args()
d = Path(a.dump_dir)
tmp = Path(tempfile.mkdtemp())
exe = tmp / "pack"
subprocess.run(["g++", "-O2", "-std=c++17", str(HERE / "check_bfp16_pack.cpp"), "-o", str(exe)], check=True)
ok = True
for name in ["L0_ff_fc1_weight", "L0_ff_fc2_weight", "L0_attn_wqkv_weight", "L0_attn_out_proj_weight",
             "L99_tail_up_weight", "L99_tail_crf_weight"]:
    w = np.load(d / f"{name}.npy").astype(bfloat16).astype(np.float32)  # [N, K]
    N, K = w.shape
    py = pack_b_bfp16ebs8(np.ascontiguousarray(w.T).astype(bfloat16), a.tile_n, a.tile_k_l1).ravel()
    w.tofile(tmp / "w.f32")
    subprocess.run([str(exe), str(tmp / "w.f32"), str(N), str(K), str(a.tile_n), str(a.tile_k_l1), str(tmp / "o.u8")], check=True)
    cc = np.fromfile(tmp / "o.u8", np.uint8)
    same = cc.shape == py.shape and bool((cc == py).all())
    ok &= same
    print(f"{name} [{N}x{K}]: {py.size} bytes, {'identical' if same else f'DIFFER at {int((cc != py).sum())} bytes'}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
