#!/usr/bin/env python3
"""Phase-1 harness for the 10-23 direct example ports, on real slorado sup@v5.0.0 tensors (CPU dump):

  axpy          kernels/axpy (mlir-air axpy)     deepnorm residual alpha*x + in   gate: example tol rtol 1e-2, atol 1e-8
  silu          kernels/silu (mlir-air silu)     swish on fc1 outputs              gate: example tol rtol 1e-1, atol 5e-2
  rms_qkv_rope  kernels/rms_qkv_rope (mlir-air llms rms_gemms_rope_multi)
                L0 norm2 (residual folded) -> L1 wqkv -> RoPE, first 512 rows
                gate: the builder's own test tol (rtol 0.2, atol 0.5, correlation >= 0.99) on q_roped and k_roped
Each reference is computed in fp32 from the same bf16-cast inputs; the CPU dump comparison is info.
The module is built exactly as the kernel's build_artifact.py builds it (same function, args, backend settings).
Exit 0 only if every selected kernel passes.

  source ~/npu-env.sh && cd ~/slorado/openfish/npu
  flock -x -w 1800 $NPU_LOCK python3 test/test_examples_npu.py --dump-dir ~/p0/dump_cpu18 [--kernels axpy silu rms_qkv_rope]
The rms_qkv_rope kernel needs >= 2 dumped layers (L1 wqkv) and kernels/rms_qkv_rope/build_peano from build_artifact.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
KROOT = HERE.parent / "kernels"
SRC = Path(os.environ.get("MLIR_AIR_SRC", Path.home() / "mlir-air")) / "programming_examples"


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def n_outside(out, exp, rtol, atol):
    out, exp = np.asarray(out, np.float32), np.asarray(exp, np.float32)
    return int((np.abs(out - exp) > atol + rtol * np.abs(exp)).sum())


def first(x, n):
    x = np.asarray(x, np.float32).ravel()
    return np.resize(x, n) if x.size < n else x[:n].copy()


def run(backend, module, args):
    artifact = backend.compile(module)
    invoke = backend.load(artifact)
    try:
        return invoke(*args)
    finally:
        backend.unload()


def gate(name, out, exp, rtol, atol, min_corr=None):
    bad = n_outside(out, exp, rtol, atol)
    corr = float(np.corrcoef(np.asarray(out, np.float64).ravel(), np.asarray(exp, np.float64).ravel())[0, 1])
    ok = bad == 0 if min_corr is None else (corr >= min_corr)
    # the builder's own check (XRTRunner min_correlation) gates on correlation; report both either way
    e = np.abs(np.asarray(out, np.float32) - np.asarray(exp, np.float32))
    print(f"  {name}: outside tol (rtol {rtol}, atol {atol}) {bad}/{e.size}, correlation {corr:.6f}, "
          f"mean_rel_L1 {e.mean() / np.abs(np.asarray(exp, np.float32)).mean():.3e}, max abs err {e.max():.3e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_axpy(L, alpha):
    sys.path[:0] = [str(KROOT / "axpy")]
    from axpy import build_axpy
    from air.api.types import bf16
    from air.backend.xrt import XRTBackend

    n = 4194304
    x = first(L(0, "norm1_residual"), n).astype(bfloat16)
    y = first(L(0, "norm1_in"), n).astype(bfloat16)
    ref = (alpha * x.astype(np.float32) + y.astype(np.float32)).astype(bfloat16)
    os.chdir(KROOT / "axpy" / "build_peano")
    module = build_axpy(n=n, tile_n=1024, alpha=alpha, dtype=bf16).build(target="npu2")
    be = XRTBackend(target_device="npu2", omit_while_true_loop=False, output_format="elf", instance_name="axpy",
                    runtime_loop_tiling_sizes=[4, 4])
    out = run(be, module, [x, y, np.zeros(n, bfloat16)])[2]
    print(f"== axpy n={n} alpha={alpha} (x = L0 norm1_residual, y = L0 norm1_in)")
    ok = gate("out", out, ref, 1e-2, 1e-8)
    # the air.api expression carries alpha as a bf16 constant: isolate that rounding from the kernel's arithmetic
    a16 = float(np.float32(bfloat16(alpha)))
    gate(f"info vs ref with bf16(alpha) = {a16}", out,
         (a16 * x.astype(np.float32) + y.astype(np.float32)).astype(bfloat16), 1e-2, 1e-8)
    return ok


def test_silu(L):
    sys.path[:0] = [str(KROOT / "silu")]
    from silu import build_silu, silu_reference
    from air.backend.xrt import XRTBackend

    n = 4194304
    x = first(L(0, "ff_fc1_out"), n).astype(bfloat16)
    ref = silu_reference(x)
    os.chdir(KROOT / "silu" / "build_peano")
    module = build_silu(n=n, tile_n=1024).build(target="npu2")
    be = XRTBackend(target_device="npu2", omit_while_true_loop=False, output_format="elf", instance_name="silu",
                    runtime_loop_tiling_sizes=[4, 4])
    out = run(be, module, [x, np.zeros(n, bfloat16)])[1]
    print(f"== silu n={n} (x = L0 ff_fc1_out)")
    return gate("out", out, ref, 1e-1, 5e-2)


def rope_halfsplit_ref(x, lut):
    """x, lut: [rows, D]; lut row = [cos(D/2) | sin(D/2)] (rope_halfsplit.cc)."""
    h = x.shape[1] // 2
    c, s = lut[:, :h], lut[:, h:]
    x1, x2 = x[:, :h], x[:, h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)


def test_rms_qkv_rope(L, alpha):
    sys.path[:0] = [str(KROOT / "rms_qkv_rope"), str(SRC / "llms"), str(SRC)]
    from shared.infra.cache import prepare_air_project
    from shared.infra.backend_presets import RMS_GEMMS_ROPE_BACKEND
    from rms_gemms_rope_multi import build_rms_gemms_rope_module
    from air.backend.xrt import XRTBackend

    M, D, H, HD = 512, 512, 8, 64
    k = (L(0, "norm2_in").astype(np.float32) + alpha * L(0, "norm2_residual").astype(np.float32)).reshape(-1, D)[:M]
    x = k.astype(bfloat16)
    nw = L(0, "norm2_weight").astype(bfloat16)
    W = L(1, "attn_wqkv_weight").astype(np.float32)  # [3D, D]
    wq, wk, wv = (np.ascontiguousarray(W[i * D:(i + 1) * D].T).astype(bfloat16) for i in range(3))
    cos, sin = L(1, "attn_rotary_cos").astype(np.float32), L(1, "attn_rotary_sin").astype(np.float32)  # [2048, 32]
    lut_pos = np.concatenate([cos[:M], sin[:M]], axis=1)  # first M positions of chunk 0 (rows 0..511)
    lut = np.repeat(lut_pos, H, axis=0).astype(bfloat16)  # (position, head) rows, seq-first

    xf = x.astype(np.float32)
    normed = (xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1e-5) * nw.astype(np.float32)).astype(bfloat16)
    nf = normed.astype(np.float32)
    q = (nf @ wq.astype(np.float32)).astype(bfloat16).astype(np.float32)
    kk = (nf @ wk.astype(np.float32)).astype(bfloat16).astype(np.float32)
    v = (nf @ wv.astype(np.float32)).astype(bfloat16)
    lf = lut.astype(np.float32)
    q_ref = rope_halfsplit_ref(q.reshape(-1, HD), lf).reshape(M, D).astype(bfloat16)
    k_ref = rope_halfsplit_ref(kk.reshape(-1, HD), lf).reshape(M, D).astype(bfloat16)

    os.chdir(KROOT / "rms_qkv_rope" / "build_peano")
    prepare_air_project()
    module = build_rms_gemms_rope_module(seq_len=M, emb_dim=D, kv_dim=D, n_heads=H, n_kv_heads=H, head_dim=HD)
    be = XRTBackend(**RMS_GEMMS_ROPE_BACKEND)
    z = lambda: np.zeros((M, D), bfloat16)  # noqa: E731
    res = run(be, module, [x, nw, z(), wq, z(), wk, z(), wv, z(), lut.ravel(), lut.ravel(), z(), z()])
    out_normed, out_v, out_q, out_k = (np.asarray(res[i]).reshape(M, D) for i in (2, 8, 11, 12))
    print(f"== rms_qkv_rope M={M} D={D} heads {H}x{HD} (x = L0 norm2_in + alpha*norm2_residual, L0 norm2_weight, "
          f"L1 wqkv, L1 rotary tables)")
    ok = gate("q_roped", out_q, q_ref, 0.2, 0.5, min_corr=0.99)
    ok &= gate("k_roped", out_k, k_ref, 0.2, 0.5, min_corr=0.99)
    gate("info normed", out_normed, normed, 0.2, 0.5, min_corr=0.99)
    gate("info v", out_v, v, 0.2, 0.5, min_corr=0.99)
    rot = L(1, "attn_qkv_rotary_out").astype(np.float32).reshape(-1, 3, D)[:M]
    print(f"  info vs CPU dump: normed cos {cosine(out_normed.astype(np.float32), L(0, 'norm2_out').reshape(-1, D)[:M]):.6f}, "
          f"q cos {cosine(out_q.astype(np.float32), rot[:, 0]):.6f}, k cos {cosine(out_k.astype(np.float32), rot[:, 1]):.6f}, "
          f"v cos {cosine(out_v.astype(np.float32), rot[:, 2]):.6f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=["axpy", "silu", "rms_qkv_rope"])
    ap.add_argument("--dump-dir", default=os.environ.get("OPENFISH_DUMP_DIR"))
    args = ap.parse_args()
    d = Path(args.dump_dir or "")
    if not (d / "manifest.tsv").exists():
        print(f"FAIL: no dump at {d}")
        return 2
    L = lambda layer, site: np.load(d / f"L{layer}_{site}.npy")  # noqa: E731
    alpha = float(np.asarray(L(0, "norm_deepnorm_alpha")).ravel()[0])
    tests = {"axpy": lambda: test_axpy(L, alpha), "silu": lambda: test_silu(L),
             "rms_qkv_rope": lambda: test_rms_qkv_rope(L, alpha)}
    results = {}
    for k in args.kernels:
        results[k] = tests[k]()
    print("\n| kernel | gate |\n|---|---|")
    for k, ok in results.items():
        print(f"| {k} | {'PASS' if ok else 'FAIL'} |")
    ok = all(results.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
