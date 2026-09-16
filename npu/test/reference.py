"""FP32 numpy references for openfish NPU kernels, in openfish's own tensor layouts.

These are the Phase-1 oracles. Each function mirrors what slorado's CPU path
computes (torch in thirdparty/dorado/TxModel.cpp, or openfish nn_cpu.c) and is
cross-checked against slorado tensor dumps by check_reference_vs_dump.py.

Layouts (SUP v5.0.0, batch B, transformer length T, see ../TODO.md):
  silu_mul : x[..., 2K] = [y ‖ gate]  ->  out[..., K] = SiLU(gate) * y
  rmsnorm  : out = RMSNorm(input + residual * alpha) * weight   (eps 1e-5, fused residual)
  rotary   : qkv[B, T, 3, H, D]; q and k rotated in place over the first 2*rotary_half dims,
             half-split: x0 = x[..., :r], x1 = x[..., r:2r], tables sin/cos [max_seq_len, r]
  gemm     : out = x @ weight.T (+ bias)
"""

import numpy as np

F32 = np.float32


def _f32(a):
    return np.asarray(a, dtype=F32)


def silu(x):
    x = _f32(x)
    with np.errstate(over="ignore"):
        return x / (F32(1.0) + np.exp(-x))


def silu_mul_ref(x):
    """openfish silu_mul: x[..., 2K] = [y ‖ gate] -> SiLU(gate) * y, shape [..., K]."""
    x = _f32(x)
    k = x.shape[-1] // 2
    assert x.shape[-1] == 2 * k, f"last dim must be even, got {x.shape}"
    return silu(x[..., k:]) * x[..., :k]


def split_silu_mul_input(x):
    """openfish [y ‖ gate] -> contiguous (gate, up) in mlir-air silu_and_mul argument order."""
    x = _f32(x)
    k = x.shape[-1] // 2
    return np.ascontiguousarray(x[..., k:]), np.ascontiguousarray(x[..., :k])


def rmsnorm_ref(inp, residual, weight, alpha, eps=1e-5):
    """openfish rmsnorm (deepnorm epilogue): RMSNorm(inp + residual * alpha) * weight."""
    k = _f32(inp) + _f32(residual) * F32(alpha)
    rstd = F32(1.0) / np.sqrt((k * k).mean(axis=-1, keepdims=True) + F32(eps))
    return k * rstd * _f32(weight)


def rotary_tables(head_dim=64, max_seq_len=2048, theta=10000.0):
    """sin, cos [max_seq_len, head_dim // 2] as RotaryEmbeddingImpl builds them (float32)."""
    inv_freq = F32(1.0) / np.power(F32(theta), np.arange(0, head_dim, 2, dtype=F32) / F32(head_dim))
    freqs = np.outer(np.arange(max_seq_len, dtype=F32), inv_freq).astype(F32)
    return np.sin(freqs).astype(F32), np.cos(freqs).astype(F32)


def rotary_ref(x, sin, cos, rotary_half=32):
    """openfish rotary_emb on one chunk x[B, T, H, D]; returns a rotated copy."""
    x = _f32(x)
    r = rotary_half
    t = x.shape[1]
    s = _f32(sin)[:t, :r][None, :, None, :]
    c = _f32(cos)[:t, :r][None, :, None, :]
    out = x.copy()
    x0 = x[..., :r]
    x1 = x[..., r:2 * r]
    out[..., :r] = x0 * c - x1 * s
    out[..., r:2 * r] = x0 * s + x1 * c
    return out


def rotary_qkv_ref(qkv, sin, cos, rotary_half=32):
    """Apply rotary_ref to q and k of qkv[B, T, 3, H, D] (v untouched), as TxModel does."""
    out = _f32(qkv).copy()
    for i in (0, 1):
        out[:, :, i] = rotary_ref(out[:, :, i], sin, cos, rotary_half)
    return out


def gemm_ref(x, weight, bias=None):
    """Linear: x[..., in] @ weight[out, in].T (+ bias[out])."""
    out = _f32(x) @ _f32(weight).T
    if bias is not None:
        out = out + _f32(bias)
    return out


def compare(out, ref, rtol=None, atol=None):
    """Registry-style metrics. Returns dict; 'pass' only if rtol/atol given."""
    out = _f32(out)
    ref = _f32(ref)
    err = np.abs(out - ref)
    res = {
        "max_abs": float(err.max()),
        "mean_rel_L1": float(err.mean() / max(np.abs(ref).mean(), 1e-30)),
        "max_rel": float((err / np.maximum(np.abs(ref), 1e-30)).max()),
    }
    if rtol is not None and atol is not None:
        bad = err > atol + rtol * np.abs(ref)
        res["n_fail"] = int(bad.sum())
        res["pass"] = res["n_fail"] == 0
    return res
