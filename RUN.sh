#!/usr/bin/env bash
set -euo pipefail
DESIGNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$DESIGNER_ROOT"
if [[ ! -x "$DESIGNER_ROOT/.venv/bin/python" ]]; then
  echo "Missing project environment. Run ./INSTALL.sh first." >&2
  exit 1
fi
exec "$DESIGNER_ROOT/.venv/bin/python" -s cli.py "$@"
