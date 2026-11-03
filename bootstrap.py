"""Locate bundled EC66 modules and assets within this checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
