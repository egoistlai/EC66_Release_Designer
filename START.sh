#!/usr/bin/env bash
set -euo pipefail
DESIGNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESIGNER_PY="$DESIGNER_ROOT/.venv/bin/python"
if [[ ! -x "$DESIGNER_PY" ]]; then
  echo "Missing project environment. Run ./INSTALL.sh first." >&2
  exit 1
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
# Desktop shell sessions do not always inherit the WSLg display variables.
if [[ -z "${DISPLAY:-}" && -S /mnt/wslg/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi
cd "$DESIGNER_ROOT"
exec "$DESIGNER_PY" -s gui.py --viewer gl --render-fps 60 "$@"
