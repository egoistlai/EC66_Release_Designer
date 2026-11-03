"""Background jobs: each consumes an immutable design and writes its own directory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import traceback
import numpy as np

from bootstrap import ROOT, RUNTIME
from design import read_json, write_json, fingerprint
from motion import build_release


def run(directory, task):
    directory = Path(directory).resolve()
    spec = read_json(directory / "design.json")
    def status(message, state="running", **extra):
        print(message, flush=True)
        write_json(directory / "status.json", {"state": state, "message": message, "task": task,
                                              "design_hash": fingerprint(spec), **extra})
    try:
        write_json(directory / "provenance.json", {
            "design_hash": fingerprint(spec),
            "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [*ROOT.glob("*.py"), *ROOT.glob("requirements*.txt"),
                                        *sorted(RUNTIME.rglob("*"))]
                              if p.is_file() and "__pycache__" not in p.parts}})
        context, release, audit = build_release(spec, directory, status)
        result = {"release_audit": str(directory / "release_audit.json"), "task": task}
        if task in ("simulate", "approach"):
            from physics import simulate
            result["ideal"] = simulate(directory / "release.npz", directory / "ideal", spec, context, ideal=True, status=status)
            # Preserve a useful release preview even if the subsequent approach search fails.
            write_json(directory / "result.json", result)
        if task == "approach":
            from approach import plan_approach
            result["approach"] = plan_approach(spec, context, release, directory, status)
            with np.load(directory / "ideal/distribution.npz") as a, np.load(Path(result["approach"]["directory"]) / "distribution.npz") as b:
                ga, gb = a["mass_grid"], b["mass_grid"]
                result["comparison"] = {"mass_grid_tv_including_missing_mass":
                    .5 * float(np.sum(abs(ga - gb)) + abs(ga.sum() - gb.sum()))}
        write_json(directory / "result.json", result)
        message = "Complete"
        if task == "approach" and not result["approach"]["retention_pass"]:
            message = "Complete: best approach is below the requested retention threshold; inspect the measured result"
        status(message, "complete")
        return 0
    except Exception as error:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status(str(error), "failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("build", "simulate", "approach"), required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.run, args.task))
