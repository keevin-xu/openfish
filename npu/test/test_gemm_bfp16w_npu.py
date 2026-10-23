#!/usr/bin/env python3
"""Phase-1 harness for kernels/gemm_bfp16w (mlir-air bf16_x_bfp16: bf16 A x bfp16ebs8-packed weights -> bf16 C) on
real slorado sup@v5.0.0 weights and activations, at the production M rows per launch (same shapes as gemm_bf16).
  gate: the example's own check vs A @ dequant(B_packed) (rtol 0.1, atol 0.05, <= 0.05% mismatches, correlation
        >= 0.999), i.e. the kernel is correct for the weights it was given
  info: vs the exact fp32 product of the bf16 inputs and vs slorado's CPU fp32 dump (cost of the weight quantization)
Exit 0 only if every op passes the gate.
  source ~/npu-env.sh && cd ~/slorado/openfish/npu
  OPENFISH_DUMP_DIR=~/p0/dump_sup16 flock -x -w 1800 $NPU_LOCK python3 test/test_gemm_bfp16w_npu.py [--ops fc1 fc2]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "kernels" / "gemm_bfp16w"
sys.path[:0] = [str(HERE), str(KDIR)]

DEFAULT_M = {"fc1": 4096, "fc2": 4096, "wqkv": 8192, "out_proj": 16384, "upsample": 8192, "crf": 4096}
TILE = dict(tile_m=64, tile_k_l1=128, tile_n=64, herd_m=8, herd_n=4)


def dequant(packed, K, N, tn, tk):
    """[N/tn, K/tk, tile_bytes] bfp16ebs8 records -> B[K, N] float32 (vectorized cpu_reference_from_bfp_packed)."""
    rec = packed.reshape(N // tn, K // tk, tn // 8, tk // 8, 8, 9)  # nb, kb, nbi, kbi, n_i, [exp, m0..m7]
    e = rec[..., 0].astype(np.int32)
    m = rec[..., 1:].view(np.int8).astype(np.float32)
    v = m * np.ldexp(1.0, e - 127)[..., None] / 64.0  # [nb, kb, nbi, kbi, n_i, k_i]
    return np.transpose(v, (1, 3, 5, 0, 2, 4)).reshape(K, N)


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_M))
    ap.add_argument("--dump-dir", default=os.environ.get("OPENFISH_DUMP_DIR"))
    args = ap.parse_args()
    d = Path(args.dump_dir or "")
    L = lambda s: np.load(d / f"L0_{s}.npy")  # noqa: E731
    T = lambda s: np.load(d / f"L99_{s}.npy")  # noqa: E731
    specs = {
        "fc1": (lambda: L("ff_in"), lambda: L("ff_fc1_weight"), lambda: L("ff_fc1_out")),
        "fc2": (lambda: L("ff_silu_mul_out"), lambda: L("ff_fc2_weight"), lambda: L("ff_fc2_out")),
        "wqkv": (lambda: L("attn_in"), lambda: L("attn_wqkv_weight"), lambda: L("attn_qkv_linear_out").reshape(-1, 1536)),
        "out_proj": (lambda: L("attn_sdpa_out"), lambda: L("attn_out_proj_weight"),
                     lambda: L("attn_out_proj_out") - L("attn_out_proj_bias")),
        "upsample": (lambda: T("tail_up_in"), lambda: T("tail_up_weight"),
                     lambda: T("tail_up_out").reshape(-1, 1024) - T("tail_up_bias")),
        "crf": (lambda: T("tail_up_out"), lambda: T("tail_crf_weight"), lambda: T("tail_crf_out")),
    }
    from air.backend.xrt import XRTBackend
    from air.backend.xrt_runner import XRTRunner
    import matmul_bf16_x_bfp16 as mm

    rows = lambda x, m: np.ascontiguousarray(x.reshape(-1, x.shape[-1])[:m], dtype=np.float32)  # noqa: E731
    all_ok = True
    table = []
    for name in args.ops:
        M = DEFAULT_M[name]
        src, wl, el = specs[name]
        a32, w, cpu = rows(src(), M), wl().astype(np.float32), rows(el(), M)
        K, N = a32.shape[1], w.shape[0]
        build = KDIR / f"build_m{M}_k{K}_n{N}"
        subprocess.run(["make", "-s", "-C", str(KDIR), "compile-kernel", f"BUILD_DIR={build.name}",
                        f"TILE_M={TILE['tile_m']}", f"TILE_N={TILE['tile_n']}", f"TILE_K_L1={TILE['tile_k_l1']}"], check=True)
        os.chdir(build)
        module = mm.build_module(M, K, N, TILE["tile_m"], K, TILE["tile_k_l1"], TILE["tile_n"], TILE["herd_m"], TILE["herd_n"])
        a_bf = a32.astype(bfloat16)
        b_bf = np.ascontiguousarray(w.T).astype(bfloat16)
        packed = mm.pack_b_bfp16ebs8(b_bf, TILE["tile_n"], TILE["tile_k_l1"])
        bq = dequant(packed, K, N, TILE["tile_n"], TILE["tile_k_l1"])
        expected = (a_bf.astype(np.float32) @ bq).astype(bfloat16)
        exact = a_bf.astype(np.float32) @ b_bf.astype(np.float32)
        be = XRTBackend(omit_while_true_loop=False, output_format="elf", stack_size=2048, runtime_loop_tiling_sizes=[2, 2],
                        instance_name="matmul_bf16_x_bfp16", n_perf_iters=10)
        artifact = be.compile(module)
        invoke = be.load(artifact)
        try:
            out = np.asarray(invoke(a_bf, packed, np.zeros((M, N), bfloat16))[2]).reshape(M, N)
        finally:
            be.unload()
        lat = be.last_latency_us / 1e3
        print(f"== {name} {M}x{K}x{N} bfp16w tiles m/kl2/kl1/n 64/{K}/128/64 herd 8x4")
        ok = XRTRunner(report_precision=True)._check_outputs([out], [expected], rtol=0.1, atol=0.05,
                                                             max_mismatch_percentage=0.05, min_correlation=0.999)
        all_ok &= bool(ok)
        o = out.astype(np.float32)
        r = dict(op=name, shape=f"{M}x{K}x{N}", gate="PASS" if ok else "FAIL",
                 mrl1_exact=float(np.abs(o - exact).mean() / np.abs(exact).mean()),
                 mrl1_dump=float(np.abs(o - cpu).mean() / np.abs(cpu).mean()), cos_dump=cosine(o, cpu),
                 mrl1_quant=float(np.abs(a_bf.astype(np.float32) @ bq - exact).mean() / np.abs(exact).mean()), ms=lat)
        table.append(r)
    print("\n| op | M×K×N | gate (vs quantized-weight ref) | mean_rel_L1 vs exact bf16-input product | of which weight quantization | vs CPU dump mean_rel_L1 | vs dump cosine | NPU ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table:
        print(f"| {r['op']} | {r['shape']} | {r['gate']} | {r['mrl1_exact']:.3e} | {r['mrl1_quant']:.3e} | {r['mrl1_dump']:.3e} | "
              f"{r['cos_dump']:.8f} | {r['ms']:.2f} |")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
