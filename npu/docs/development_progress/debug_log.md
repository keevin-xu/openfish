# openfish/npu — debug log (append-only)

## 2026-09-16
- **sup CPU run SIGKILLed** at default `-C 512`: max RSS 26.9 GB on a 30 GiB host, probable OOM (dmesg not
  readable). Rerun with `-C 128` (README sup batch): PASS at 12.2 GB peak. Not fully confirmed whether the kill
  was the OOM killer or a manual `kill -9`.
- **Accuracy script** needs `<ref.fa> <fastq>` and minimap2 + datamash; none were on the server. Installed
  per-user in `~/tools` (no sudo).
- **`git submodule sync` reverted a `git config submodule.openfish.url` override** — sync copies from
  `.gitmodules`. Fixed by pointing `.gitmodules` at the fork.
