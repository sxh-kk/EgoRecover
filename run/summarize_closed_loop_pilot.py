"""Summarize paired pilot SMPL-X reports by take, without frame pseudoreplication."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def row_summary(take, method, row):
    paired = row.get("paired_fault_delta") or {}
    return {
        "take": take,
        "method": method,
        "variant": row["variant"],
        "smpl22_mpjpe_mm": row["metrics"]["mpjpe_body_m"] * 1000,
        "fault_signed_mean_delta_mm": paired.get("fault_signed_mean_mm"),
        "fault_positive_auc_mm_s": paired.get("fault_positive_auc_mm_s"),
        "recovery_time_status": paired.get("recovery_time_status"),
    }


def summarize(reports, frozen_reports):
    if not reports:
        raise ValueError("At least one trained-model report is required.")
    parsed = [json.loads(Path(path).read_text()) for path in reports]
    if any(not report.get("completed") for report in parsed):
        raise ValueError("All trained-model reports must be complete.")
    groups = {report["group"] for report in parsed}
    if len(groups) != 1:
        raise ValueError("Cannot mix dev and holdout in one summary.")
    by_take = {report["take"]: report for report in parsed}
    if len(by_take) != len(parsed):
        raise ValueError("Duplicate take report.")
    rows = [row_summary(report["take"], row["source_mode"], row)
            for report in parsed for row in report["results"]]
    for path in frozen_reports:
        report = json.loads(Path(path).read_text())
        if not report.get("completed") or report["take"] not in by_take:
            raise ValueError("Frozen E7 report is incomplete or outside the selected take group.")
        rows.extend(row_summary(report["take"], "frozen_e7", row)
                    for row in report["results"] if row["action"] == "a11")
    keys = defaultdict(list)
    for row in rows:
        keys[row["method"], row["variant"]].append(row)
    take_set = set(by_take)
    if any({row["take"] for row in values} != take_set for values in keys.values()):
        raise ValueError("Every method/variant must cover the same take set.")
    macro = []
    for (method, variant), values in sorted(keys.items()):
        macro.append({
            "method": method,
            "variant": variant,
            "takes": len(values),
            "mean_smpl22_mpjpe_mm": statistics.mean(row["smpl22_mpjpe_mm"] for row in values),
            "mean_fault_signed_delta_mm": (
                statistics.mean(row["fault_signed_mean_delta_mm"] for row in values)
                if variant != "clean" else None
            ),
            "mean_fault_positive_auc_mm_s": (
                statistics.mean(row["fault_positive_auc_mm_s"] for row in values)
                if variant != "clean" else None
            ),
        })
    return {"group": groups.pop(), "take_count": len(parsed), "takes": sorted(take_set),
            "per_take": rows, "macro_by_take": macro,
            "scope": "small_pilot_development_only; no independent official-val claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-reports", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    result = summarize(args.reports, args.frozen_reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["macro_by_take"], indent=2))


if __name__ == "__main__":
    main()
