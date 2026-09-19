#!/usr/bin/env python3
"""Phase-1 harness: registry bf16 GEMM on NPU2 with real slorado sup@v5.0.0 weights and activations.

For each linear (fc1, fc2, wqkv, out_proj, upsample, crf) at its production M rows per launch
(DEFAULT_M; same shapes as kernels/gemm_bf16/build_artifacts.sh): A = dumped layer input (first M rows),
B = weight.T, one compile, one launch; checks
  gate: element-wise vs fp32 A@B from the bf16-cast inputs (registry protocol, rtol 1.6e-2 / atol 1.5e-3)
  info: vs slorado's CPU fp32 dump output (bias added host-side for out_proj), cosine, n outside tol
  info: CPU floor = the same fp32 reference from bf16 inputs vs the dump (input-rounding error alone)
and kernel-only latency (XRTBackend perf mode, mean) next to torch fp32 linear on the same A (median).
Exit 0 only if every op passes the gate.

  source ~/npu-env.sh && cd ~/slorado/openfish/npu
  OPENFISH_DUMP_DIR=~/p0/dump_sup16 flock -x -w 1800 $NPU_LOCK python3 test/test_gemm_npu.py [--ops fc1 fc2]
The dump needs >= 16 rows (out_proj M=16384) and the tail (OPENFISH_DUMP_TAIL=1) for upsample/crf.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "kernels" / "gemm_bf16"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KDIR))
import reference as ref  # noqa: E402

RTOL, ATOL = 1.6e-2, 1.5e-3
DEFAULT_M = {"fc1": 4096, "fc2": 4096, "wqkv": 8192, "out_proj": 16384, "upsample": 8192, "crf": 4096}
TILES = dict(tile_k_l2=256, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4)


def n_fail(out, expected):
    return int((np.abs(out - expected) > ATOL + RTOL * np.abs(expected)).sum())


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_M))
    ap.add_argument("--m", type=int, default=None, help="override rows per launch for every op")
    ap.add_argument("--perf-iters", type=int, default=20)
    ap.add_argument("--dump-dir", default=os.environ.get("OPENFISH_DUMP_DIR"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--mmul", choices=["bfp16", "native"], default="bfp16",
                    help="bfp16: registry build (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, 8-elem shared-exponent "
                         "8-bit-mantissa blocks); native: same mm_aie2p.cc without that define")
    args = ap.parse_args()
    d = Path(args.dump_dir or "")
    if not (d / "manifest.tsv").exists():
        print(f"FAIL: no dump at {d}")
        return 2
    L = lambda s: np.load(d / f"L{args.layer}_{s}.npy")  # noqa: E731
    T = lambda s: np.load(d / f"L99_{s}.npy")  # noqa: E731  (tail: OPENFISH_DUMP_TAIL)

    def rows(x, m):  # [B, T, C] -> first m rows as [m, C]
        x = x.reshape(-1, x.shape[-1])
        assert x.shape[0] >= m, f"dump has {x.shape[0]} rows < M={m}"
        return np.ascontiguousarray(x[:m], dtype=np.float32)

    # name -> (input site loader, weight, bias, expected loader); rows are cut once M is known
    specs = {
        "fc1": (lambda: L("ff_in"), lambda: L("ff_fc1_weight"), lambda: None, lambda: L("ff_fc1_out")),
        "fc2": (lambda: L("ff_silu_mul_out"), lambda: L("ff_fc2_weight"), lambda: None, lambda: L("ff_fc2_out")),
        "wqkv": (lambda: L("attn_in"), lambda: L("attn_wqkv_weight"), lambda: None,
                 lambda: L("attn_qkv_linear_out").reshape(-1, 1536)),
        "out_proj": (lambda: L("attn_sdpa_out"), lambda: L("attn_out_proj_weight"), lambda: L("attn_out_proj_bias"),
                     lambda: L("attn_out_proj_out")),
        # upsample: linear(x) [.., 1024] reshaped to [N, 2T, 512]; undo the (pure view) reshape
        "upsample": (lambda: T("tail_up_in"), lambda: T("tail_up_weight"), lambda: T("tail_up_bias"),
                     lambda: T("tail_up_out").reshape(-1, 1024)),
        "crf": (lambda: T("tail_up_out"), lambda: T("tail_crf_weight"), lambda: None, lambda: T("tail_crf_out")),
    }

    import torch
    from air.backend.xrt import XRTBackend
    from air.backend.xrt_runner import XRTRunner
    import run as gemm

    torch.set_num_threads(8)
    checker = XRTRunner(report_precision=True)
    results = []
    all_ok = True
    for name in args.ops:
        M = args.m or DEFAULT_M[name]
        src, wl, bl, el = specs[name]
        a32, w, bias, cpu_out = rows(src(), M), wl(), bl(), rows(el(), M)
        K, N = a32.shape[1], w.shape[0]
        method = "fused-cast" if M * K * N >= 4e9 else "drain"
        tile_m = 64 if method == "fused-cast" else 32
        for c in (M % (tile_m * 8), K % 256, N % 512):
            assert c == 0, f"{name}: illegal tiling {M}x{K}x{N}"
        mk = subprocess.run(["make", "-s", "-n", "-C", str(KDIR), "compile-kernel", f"TILE_M={tile_m}"],
                            check=True, capture_output=True, text=True).stdout
        if args.mmul == "native":
            assert "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" in mk
            mk = mk.replace("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16", "")
        subprocess.run(mk, shell=True, check=True, cwd=KDIR, executable="/bin/bash")
        os.chdir(KDIR / "build_peano")
        if method == "fused-cast":
            module = gemm.build_module_gemm_cast(M, K, N, tile_m, **TILES, arch="aie2p")
            fmt, inst = "elf", "gemm_cast_bf16"
        else:
            module = gemm.build_module(M, K, N, tile_m, TILES["tile_k_l2"], TILES["tile_k_l1"], TILES["tile_n"],
                                       TILES["herd_m"], TILES["herd_n"], bfloat16, bfloat16, arch="aie2p",
                                       emit_external_call=True, drain_chunks=1)
            fmt, inst = "xclbin", "matmul_bf16"
        backend = XRTBackend(target_device="npu2", omit_while_true_loop=False, runtime_loop_tiling_sizes=[2, 2],
                             stack_size=2048, output_format=fmt, instance_name=inst, n_perf_iters=args.perf_iters)
        artifact = backend.compile(module)

        a_bf = a32.astype(bfloat16)
        b_bf = np.ascontiguousarray(w.T).astype(bfloat16)
        expected = (a_bf.astype(np.float32) @ b_bf.astype(np.float32)).astype(bfloat16)
        inputs = [a_bf, b_bf] + ([np.zeros((M, N), np.float32)] if method == "fused-cast" else [])
        invoke = backend.load(artifact)
        try:
            res = invoke(*inputs, np.zeros((M, N), bfloat16))
            out = res[len(inputs)].reshape(M, N)
            f32_out = res[2].reshape(M, N) if method == "fused-cast" else None
        finally:
            backend.unload()
        lat_ms = backend.last_latency_us / 1e3

        print(f"== {name} {M}x{K}x{N} {method} tile_m={tile_m} herd 8x4 ({fmt}) mmul={args.mmul}")
        ok = checker._check_outputs([out], [expected], rtol=RTOL, atol=ATOL)
        all_ok &= ok
        o32 = out.astype(np.float32) + (bias if bias is not None else 0)
        floor = expected.astype(np.float32) + (bias if bias is not None else 0)
        err = np.abs(o32 - cpu_out)
        x, wt = torch.from_numpy(a32), torch.from_numpy(w)
        ts = []
        for i in range(25):
            t0 = time.perf_counter()
            torch.nn.functional.linear(x, wt)
            if i >= 5:
                ts.append(time.perf_counter() - t0)
        cpu_ms = float(np.median(ts)) * 1e3
        r = dict(op=name, shape=f"{M}x{K}x{N}", method=method, gate="PASS" if ok else "FAIL",
                 n_fail=n_fail(out.astype(np.float32), expected.astype(np.float32)),
                 mrl1=ref.compare(out, expected)["mean_rel_L1"],
                 dump_mrl1=float(err.mean() / np.abs(cpu_out).mean()), dump_cos=cosine(o32, cpu_out),
                 dump_nfail=n_fail(o32, cpu_out), floor_mrl1=ref.compare(floor, cpu_out)["mean_rel_L1"],
                 npu_ms=lat_ms, cpu_ms=cpu_ms)
        r["mmul"] = args.mmul
        results.append(r)
        if f32_out is not None:
            e32 = a_bf.astype(np.float32) @ b_bf.astype(np.float32)
            fe = np.abs(f32_out - e32)
            print(f"info fp32 intermediate (= bf16_in_fp32_out datapath): mean_rel_L1 {fe.mean() / np.abs(e32).mean():.3e} "
                  f"abs_err max {fe.max():.3e} outside tol {n_fail(f32_out, e32)} "
                  f"| vs CPU dump mean_rel_L1 {np.abs(f32_out + (bias if bias is not None else 0) - cpu_out).mean() / np.abs(cpu_out).mean():.3e}")
        print(f"info vs CPU dump: mean_rel_L1 {r['dump_mrl1']:.3e} cosine {r['dump_cos']:.8f} outside tol {r['dump_nfail']} "
              f"| bf16-input floor mean_rel_L1 {r['floor_mrl1']:.3e}")
        print(f"perf: NPU kernel-only mean {lat_ms:.2f} ms ({args.perf_iters} iters) vs torch fp32 8 thr median {cpu_ms:.2f} ms "
              f"-> {cpu_ms / lat_ms:.1f}x")

    print("\n| op | M×K×N | method | gate | n outside tol | mean_rel_L1 | vs dump mean_rel_L1 | vs dump cosine | NPU ms | CPU 8thr ms | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['op']} | {r['shape']} | {r['method']} | {r['gate']} | {r['n_fail']} | {r['mrl1']:.3e} | {r['dump_mrl1']:.3e} | "
              f"{r['dump_cos']:.8f} | {r['npu_ms']:.2f} | {r['cpu_ms']:.2f} | {r['cpu_ms'] / r['npu_ms']:.1f}x |")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
