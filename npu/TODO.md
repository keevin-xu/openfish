# NPU port: openfish / slorado

## Phase status — silu_mul (Tier 1 #1, template kernel)
- [x] 0: CPU Oracle (`bc-phase-0-cpu-oracle`) — PASS 2026-09-16 (identity fast/hac/sup, dumps, reference.py; see docs/development_progress/progress.md)
- [x] 1: Kernel Validation (`bc-phase-1-kernel-validation`) — PASS 2026-09-16 (1/1 shapes: N=16,777,216 randn + real SUP activations; 8x1 tiles; `docs/development_progress/phase1_kernels.md`)
- [ ] 2: Single-Block / integration (`openfish-npu-backend`, per-layer cosine >= 0.99)
- [ ] 3: Full basecaller (identity-score gate)
- [ ] 4/5: Optimization (Tier 2/3)
- [ ] 6: Finalize
- [ ] 7: Independent Evaluation (`phase-7-independent-evaluator`)

Later kernels (only after silu_mul reaches Phase 3): rmsnorm, rotary_emb, gemm_fc1, gemm_fc2, gemm_wqkv, gemm_oproj, lstm_layer.

## Active blockers
(none)

## Resolved config — dna_r10.4.1_e8.2_400bps_sup@v5.0.0 (config.toml + tensor dump)
conv: 1->64->64->128->128->512, strides 1,1,3,2,2 (total 12); chunk 12288, overlap 600
transformer: depth 18, d_model 512, nhead 8, head_dim 64, dim_feedforward 2048,
attn_window (127, 128) bidirectional, deepnorm_alpha 2.4494897, rotary_dim 32 (half-split over head_dim 64), theta 1e4
runtime (batch -C 128): encoder activations [128, 1024, 512]; qkv [128, 1024, 3, 8, 64];
fc1 out [128, 1024, 4096] = [y ‖ gate]; silu_mul out [128, 1024, 2048] (N = 268,435,456 per call, 18 calls/batch)
weights: fc1 [4096, 512], fc2 [512, 2048], wqkv [1536, 512] (no bias), out_proj [512, 512] + bias

## Baselines (CPU, reads_1k.blow5, slorado ce74220 / openfish 2e1a8cc)
| model | median identity | README | wall |
|---|---|---|---|
| fast@v5.0.0 | 0.9405915 | 0.940374 | — |
| hac@v5.0.0 | 0.9780595 | 0.977594 | 1:58.6 |
| sup@v5.0.0 (-C 128) | 0.988506 | 0.988561 | 20:11.9 |
