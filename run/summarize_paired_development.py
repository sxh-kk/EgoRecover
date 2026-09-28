"""Summarize complete paired development evaluations by take and condition."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def interval(values, seed=62, repeats=2000):
    if not values:
        raise ValueError("A paired interval needs at least one take.")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats)]]


def summarize(paths):
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports or not all(report["completed"] for report in reports):
        raise ValueError("All paired evaluation reports must be complete.")
    identity_keys = ("group", "variants", "sampling_seed", "dataset_spec_sha256",
                     "evaluation_split_manifest_sha256",
                     "bootstrap_cache_sha256", "e7_checkpoint_sha256", "stats_sha256")
    identity = {key: reports[0][key] for key in identity_keys}
    for report in reports[1:]:
        if any(report[key] != identity[key] for key in identity_keys):
            raise ValueError("Evaluation protocols differ.")
    experiment_paths = {}
    for report in reports:
        for label, path in report["experiments"].items():
            if label in experiment_paths and experiment_paths[label] != path:
                raise ValueError(f"Experiment label {label} points to different checkpoints.")
            experiment_paths[label] = path
    rows = [row for report in reports for row in report["results"]]
    keys = [(row["experiment"], row["mode"], row["take"], row["variant"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate experiment/mode/take/variant results.")
    take_sets = defaultdict(set)
    by_case = defaultdict(dict)
    for row in rows:
        case = (row["experiment"], row["mode"])
        take_sets[case].add(row["take"])
        by_case[case][(row["take"], row["variant"])] = row
    all_takes = set().union(*take_sets.values())
    if any(takes != all_takes for takes in take_sets.values()):
        raise ValueError("Cases use different take sets.")
    if any(set(variant for take, variant in cases) != set(identity["variants"]) or
           len(cases) != len(all_takes) * len(identity["variants"]) for cases in by_case.values()):
        raise ValueError("Cases have missing variants or takes.")
    scores = {}
    for (label, mode), cases in sorted(by_case.items()):
        per_take = {
            take: sum(cases[take, variant]["smpl22_mpjpe_mm"] for variant in identity["variants"])
            / len(identity["variants"])
            for take in sorted(all_takes)
        }
        values = list(per_take.values())
        scores[f"{label}/{mode}"] = {
            "macro_smpl22_mm": sum(values) / len(values),
            "take_bootstrap_95ci_mm": interval(values),
            "per_take_macro_smpl22_mm": per_take,
            "per_variant_smpl22_mm": {
                variant: sum(cases[take, variant]["smpl22_mpjpe_mm"] for take in all_takes) / len(all_takes)
                for variant in identity["variants"]
            },
            "fault_deltas": {
                variant: {
                    metric: sum(cases[take, variant]["paired_fault_delta"][metric] for take in all_takes)
                    / len(all_takes)
                    for metric in ("fault_signed_mean_mm", "fault_positive_auc_mm_s", "fault_peak_increase_mm")
                }
                for variant in identity["variants"] if variant != "clean"
            },
        }
    paired = {}
    for left, left_score in scores.items():
        for right, right_score in scores.items():
            if left >= right:
                continue
            differences = [
                left_score["per_take_macro_smpl22_mm"][take] - right_score["per_take_macro_smpl22_mm"][take]
                for take in sorted(all_takes)
            ]
            paired[f"{left} minus {right}"] = {
                "mean_mm": sum(differences) / len(differences),
                "take_bootstrap_95ci_mm": interval(differences),
                "take_differences_mm": dict(zip(sorted(all_takes), differences)),
            }
    return {"identity": identity, "takes": sorted(all_takes), "experiments": experiment_paths,
            "scores": scores, "paired": paired,
            "reports": [str(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new summary output.")
    result = summarize(args.report)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case, score in result["scores"].items():
        print(f"{case}: {score['macro_smpl22_mm']:.3f} mm")


if __name__ == "__main__":
    main()
