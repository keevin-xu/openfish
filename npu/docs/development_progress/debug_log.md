# openfish/npu — debug log (append-only)

## 2026-09-16
- **sup CPU run SIGKILLed** at default `-C 512`: max RSS 26.9 GB on a 30 GiB host, probable OOM (dmesg not
  readable). Rerun with `-C 128` (README sup batch): PASS at 12.2 GB peak. Not fully confirmed whether the kill
  was the OOM killer or a manual `kill -9`.
- **Accuracy script** needs `<ref.fa> <fastq>` and minimap2 + datamash; none were on the server. Installed
  per-user in `~/tools` (no sudo).
- **`git submodule sync` reverted a `git config submodule.openfish.url` override** — sync copies from
  `.gitmodules`. Fixed by pointing `.gitmodules` at the fork.

## 2026-09-16 (Phase 2)
- `patch_value() only supports 64-bit values or less` on the first launch: `xrt::run::set_arg`'s forwarding
  template matched `xrt::ext::bo` better than the `const xrt::bo&` overload, so the BO was patched as a 32-byte
  scalar. Fix: hold BOs as `xrt::bo` built from `xrt::ext::bo(...)`.
- `git add include/openfish/openfish.h` refused (openfish `.gitignore` has `include`, tracked file) → needed `-f`.
