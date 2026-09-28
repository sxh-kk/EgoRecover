"""Select 12/72-take P/G budgets on common development SMPL22 scores."""

import argparse
import json
from pathlib import Path


def parse_budget(value):
    budget, separator, label = value.partition("=")
    parts = budget.split("_")
    if not separator or not label or len(parts) != 2 or parts[0] not in ("12take", "72take"):
        raise ValueError("Budgets must be 12take_STEPS=EXPERIMENT or 72take_STEPS=EXPERIMENT.")
    steps = int(parts[1])
    if steps not in (400, 1200, 2400):
        raise ValueError("Expected the predeclared 400/1200/2400 budget grid.")
    return parts[0], steps, label


def select(summary, values):
    if summary["identity"]["group"] != "dev" or len(summary["takes"]) != 12:
        raise ValueError("Budget selection requires all 12 common development takes.")
    if summary["identity"]["variants"] != ["clean", "freeze_3s", "drift_0p03mps"]:
        raise ValueError("Budget selection requires the predeclared three conditions.")
    cases = {}
    for value in values:
        group, steps, label = parse_budget(value)
        key = f"{group}_{steps}"
        if key in cases:
            raise ValueError("Duplicate budget.")
        scores = [summary["scores"][f"{label}/{mode}"]["macro_smpl22_mm"] for mode in ("gaussian", "history")]
        cases[key] = {"label": label, "group": group, "steps": steps,
                      "gaussian_mm": scores[0], "history_mm": scores[1],
                      "two_source_mean_mm": sum(scores) / 2}
    if set(cases) != {f"{group}_{steps}" for group in ("12take", "72take") for steps in (400, 1200, 2400)}:
        raise ValueError("Budget grid must contain all six cells.")
    winners = {
        group: min((case for case in cases.values() if case["group"] == group),
                   key=lambda case: (case["two_source_mean_mm"], case["steps"]))
        for group in ("12take", "72take")
    }
    return {"criterion": "mean of Gaussian and History take-macro SMPL22 MPJPE over three dev variants; lower wins",
            "cases": cases, "winners": winners, "holdout_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--budget", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new selection output.")
    result = select(json.loads(args.summary.read_text()), args.budget)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for group, case in result["winners"].items():
        print(f"{group}: {case['steps']} steps, {case['two_source_mean_mm']:.3f} mm")


if __name__ == "__main__":
    main()
