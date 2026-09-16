#!/usr/bin/env python3
"""Run every NPU check under test/ in one command. Exit code = worst child exit code."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scripts = [HERE / "check_reference_vs_dump.py"] + sorted(HERE.glob("test_*.py"))

worst = 0
for s in scripts:
    print(f"==== {s.name}", flush=True)
    rc = subprocess.call([sys.executable, str(s)])
    print(f"==== {s.name}: exit {rc}", flush=True)
    worst = max(worst, rc)
print("ALL PASS" if worst == 0 else f"FAILURES (worst exit {worst})")
sys.exit(worst)
