# openfish/npu (silu_mul template) — per-phase wall-clock log

## Baselines

- Deployment session start: 2026-09-16 ~12:50 EDT (context read + skills setup; epoch not captured)
- Scaffold complete:        2026-09-16 14:05 EDT (approx.; see commit time of the scaffold)

## Phase log

### Phase 0 — CPU Oracle  (PASS, 2026-09-16)

- start_ts:           ~1789579000  (~13:16 EDT, first slorado CPU run; approximate — not captured at the time)
- end_ts:             1789581900  (~14:05 EDT, reference checks PASS; approximate)
- wall_min:           **~50** (excl. the ~25 min ssh-key blocker before it)
- npu_compile_min:    0
- npu_runtime_s:      0
- dev_min:            **~20** (CPU runs: fast ~2, hac 2, sup 20 + 2 failed, dumps 0.05)
- notable_events:
  - ssh key not authorized; human ran ssh-copy-id (blocker before Phase 0)
  - accuracy tooling missing on server; installed to ~/tools
  - sup OOM at -C 512; rerun -C 128
  - T=1024 correction (initially assumed 2048 from context pack)

### Phase 1 — Kernel Validation, silu_mul  (PASS, 2026-09-16)

- start_ts:           ~1789581500  (~13:58 EDT, after go-ahead; approximate)
- end_ts:             1789582080  (~14:08 EDT, results recorded)
- wall_min:           **~10**
- npu_compile_min:    ~0.03 (4 aircc compiles × ~0.2 s + Peano .o)
- npu_runtime_s:      ~15 (make run 3.7 s, make profile, harness 3.3 s ×2)
- dev_min:            **~9**
- notable_events:
  - `/tmp/npu.lock` owned by another user blocks the installed XRTRunner → `TMPDIR=~/.npu-tmp` in the Makefile
  - re-dumped SUP L0 with 8 rows so real activations fill N exactly
  - suspiciously fast compile (0.2 s) re-verified from a clean build dir

### Phase 2+3 — Integration + full basecaller, silu_mul  (PASS, 2026-09-16)

- start_ts:           ~1789582100  (~14:08 EDT, after go-ahead)
- end_ts:             1789584464  (18:47:44 UTC, Phase 3 accuracy done)
- wall_min:           **~40**
- npu_compile_min:    ~0.01 (ELF artifact)
- npu_runtime_s:      ~1650 (smoke 38 s, npu dump 36 s, full run 1538 s of which kernel 56 s)
- dev_min:            **~12** (backend code, 2 bug fixes: set_arg overload, ignored header)
- notable_events:
  - `patch_value() only supports 64-bit values` from `run.set_arg(xrt::ext::bo)`
  - openfish header ignored by `.gitignore` → slorado briefly pushed without pointer bump
  - full SUP run ~25.6 min on a shared machine

## Summary table  (filled at deployment end)

| Phase | wall_min | npu_compile_min | npu_runtime_s | dev_min | notes |
|---|---:|---:|---:|---:|---|
| Scaffold + Step 0-3 | | | | | |
| 0: CPU Oracle | ~50 | 0 | 0 | ~20 | timestamps approximate |
| 1: silu_mul kernel | ~10 | ~0.03 | ~15 | ~9 | timestamps approximate |
| 2+3: silu_mul integration + full run | ~40 | ~0.01 | ~1650 | ~12 | full run dominates wall |
| **Total** | | | | | |
