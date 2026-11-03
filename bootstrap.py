"""Resolve the existing, read-only EC66 dependencies without copying scenes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EC66 = ROOT.parent
LEGACY = EC66 / "EC66_Newton_Pose_Controller"
DYNAMIC = EC66 / "EC66_Type08_Dynamic_Planner"
NEAR = EC66 / "EC66_Newton_2cm_Near_Constrained"
for directory in (DYNAMIC, LEGACY / "app"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
