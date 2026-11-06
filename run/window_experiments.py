"""Materialize configurations; never starts a GPU job or downloads assets."""

import argparse
import json
from pathlib import Path

from egorecover.window.experiments import experiment_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, choices=(1, 3, 5, 10), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = experiment_matrix(args.window)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for row in matrix:
            path = args.output / (row["name"] + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row["protocol"], indent=2) + "\n")
        (args.output / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(matrix, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
