#!/usr/bin/env python3
"""Fused RMSNorm -> Q/K/V GEMMs -> RoPE(Q, K) as one 6-launch ELF (mlir-air
llms/shared/builders/rms_gemms_rope_multi.py, copied unmodified), at SUP's attention dims:
emb 512, 8 heads x 64 (MHA, so kv_dim 512), full half-split rotary over head_dim 64 (= SUP rotary_half 32).

seq_len = rows per launch. The registry-driven builder only accepts GEMM shapes measured in mlir-air's
kernel_registry; 512x512x512 is (drain method, tile_m 32), 8192x512x512 is not, so rows per launch = 512.
13 args: x_in, norm_w, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped (manifest rms_qkv_rope.json).
Needs MLIR_AIR_SRC (default ~/mlir-air): the builder imports mlir-air's shared stitching/registry modules and
the external kernels (mm_m32.o, mm_m64.o, rope.o from rope_halfsplit.cc) are compiled from that tree.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("MLIR_AIR_SRC", Path.home() / "mlir-air")) / "programming_examples"
sys.path[:0] = [str(HERE), str(HERE.parent), str(SRC / "llms"), str(SRC)]

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--emb", type=int, default=512)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--head-dim", type=int, default=64)
ap.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR"))
a = ap.parse_args()
build = HERE / "build_peano"
build.mkdir(exist_ok=True)
os.chdir(build)

from shared.infra.external_kernels import compile_gemm_mm  # noqa: E402
from shared.infra.cache import prepare_air_project  # noqa: E402
from shared.infra.backend_presets import RMS_GEMMS_ROPE_BACKEND  # noqa: E402
from shared.builders.gemm_builder import gemm_registry_config  # noqa: E402
from rms_gemms_rope_multi import build_rms_gemms_rope_module  # noqa: E402
from elf_artifact import compile_elf  # noqa: E402

# the same two microkernel variants llama32_1b_prefill links into its rms_gemms_rope ELF
compile_gemm_mm(tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o")
compile_gemm_mm(tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o")
prepare_air_project()  # compiles rope.o etc. and stages every .o into air_project/

spec = gemm_registry_config(a.rows, a.emb, a.emb, "bf16", "high")
assert not spec["needs_f32_scratch"], "nn_npu expects the 13-arg (drain) signature"
module = build_rms_gemms_rope_module(seq_len=a.rows, emb_dim=a.emb, kv_dim=a.emb, n_heads=a.heads,
                                     n_kv_heads=a.heads, head_dim=a.head_dim)
compile_elf(module, "rms_qkv_rope", a.out_dir,
            {"m": a.rows, "n": a.emb, "heads": a.heads, "head_dim": a.head_dim, "nargs": 13,
             "gemm_method": spec["method"], "gemm_tile_m": spec["tile_m"], "eps": "1e-5",
             "args": "x_in,norm_w,normed,wq,q,wk,k,wv,v,lut_q,lut_k,q_roped,k_roped"},
            **RMS_GEMMS_ROPE_BACKEND)
