#!/usr/bin/env bash
set -euo pipefail
DESIGNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESIGNER_UV="${UV:-uv}"
if ! command -v "$DESIGNER_UV" >/dev/null 2>&1; then
  echo "uv is required. Install it as described in README.md, then rerun ./INSTALL.sh." >&2
  exit 1
fi
cd "$DESIGNER_ROOT"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$DESIGNER_ROOT/.python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DESIGNER_ROOT/.uv-cache}"
if [[ ! -x .venv/bin/python ]]; then
  "$DESIGNER_UV" venv --python 3.12.15 --managed-python .venv
fi
"$DESIGNER_UV" pip sync --python .venv/bin/python requirements-locked.txt
.venv/bin/python -s -c 'import sys, newton, warp, numpy, scipy, trimesh, imgui_bundle; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"; print("EC66 Release Designer environment ready")'
