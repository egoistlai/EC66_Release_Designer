#!/usr/bin/env bash
set -euo pipefail
DESIGNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$DESIGNER_ROOT"
exec "$DESIGNER_ROOT/../EC66_Newton_Pose_Controller/.venv/bin/python" -s cli.py "$@"
