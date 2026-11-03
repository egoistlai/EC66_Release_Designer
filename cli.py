"""Reproduce a GUI experiment without a display."""
import argparse
from pathlib import Path
from design import default_design, read_json, create_run
from worker import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("build", "simulate", "approach"))
    parser.add_argument("--scheme", type=Path)
    args = parser.parse_args()
    spec = read_json(args.scheme) if args.scheme else default_design()
    directory = create_run(spec, args.task)
    print(directory, flush=True)
    raise SystemExit(run(directory, args.task))
