#!/bin/bash
set -euo pipefail
python3 "$(dirname "$0")/build_artifact.py" --out-dir "$ARTIFACTS_DIR"
