#!/usr/bin/env python3
"""Summarize paired key-joint metrics with deterministic bootstrap intervals.

The metric arrays produced by ``eval/compute_3d_metrics.py`` follow the
insertion order of the prediction dictionary.  This script verifies that the
prediction keys have the same order before treating two metric arrays as
paired observations.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


TASKS = ("recon", "gen", "fore")
METRICS = ("mpjpe_key", "mpjpe_key_pa", "root_trans_error")
METRIC_LABELS = {
    "mpjpe_key": "MPJPE-Key",
    "mpjpe_key_pa": "PA-MPJPE-Key",
    "root_trans_error": "Root translation",
}


@dataclass(frozen=True)
class Experiment:
    label: str
    path: Path
    suffix: str

    def metric_path(self, task):
        return self.path / f"metrics_3d_ee4d_{task}{self.suffix}_keyjoints.pkl"

    def prediction_path(self, task):
        return self.path / f"preds_ee4d_{task}{self.suffix}.pkl"


def normalize_suffix(suffix):
    if suffix in {"", "-"}:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def parse_experiments(items):
    experiments = {}
    for label, path, suffix in items:
        if label in experiments:
            raise ValueError(f"Duplicate experiment label: {label}")
        experiments[label] = Experiment(label, Path(path), normalize_suffix(suffix))
    return experiments


def ordered_key_hash(keys):
    payload = "\n".join(str(key) for key in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_prediction_keys(experiment, task):
    path = experiment.prediction_path(task)
    predictions = joblib.load(path)
    if not isinstance(predictions, dict):
        raise TypeError(f"Expected a prediction dict at {path}, got {type(predictions).__name__}")
    keys = tuple(predictions.keys())
    return keys, ordered_key_hash(keys)


def load_metrics(experiment, task):
    path = experiment.metric_path(task)
    values = joblib.load(path)
    missing = [name for name in METRICS if name not in values]
    if missing:
        raise KeyError(f"Missing metrics {missing} in {path}")

    result = {}
    for name in METRICS:
        array = np.asarray(values[name], dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"Expected a 1-D {name} array in {path}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values found in {name} at {path}")
        result[name] = array
    return result


def bootstrap_interval(delta_mm, indices, confidence):
    bootstrap_means = delta_mm[indices].mean(axis=1)
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(bootstrap_means, [tail, 100.0 - tail])
    return float(low), float(high)


def resolve_comparisons(experiments, baseline, requested):
    if requested:
        comparisons = [tuple(pair) for pair in requested]
    elif baseline:
        comparisons = [(baseline, label) for label in experiments if label != baseline]
    else:
        raise ValueError("Provide --baseline or at least one --compare BASELINE CANDIDATE")

    for baseline_label, candidate_label in comparisons:
        if baseline_label not in experiments:
            raise ValueError(f"Unknown baseline label: {baseline_label}")
        if candidate_label not in experiments:
            raise ValueError(f"Unknown candidate label: {candidate_label}")
        if baseline_label == candidate_label:
            raise ValueError(f"A comparison must use two different experiments: {baseline_label}")
    return comparisons


def summarize(experiments, comparisons, resamples, seed, confidence):
    metric_data = {
        label: {task: load_metrics(experiment, task) for task in TASKS}
        for label, experiment in experiments.items()
    }

    key_hashes = {}
    reference_keys = {}
    for task in TASKS:
        for label, experiment in experiments.items():
            keys, digest = load_prediction_keys(experiment, task)
            if task not in reference_keys:
                reference_keys[task] = keys
            elif keys != reference_keys[task]:
                raise ValueError(
                    f"Prediction key order differs for {label}/{task}; paired statistics are invalid"
                )
            key_hashes.setdefault(label, {})[task] = digest
            for metric_name in METRICS:
                size = metric_data[label][task][metric_name].size
                if size != len(keys):
                    raise ValueError(
                        f"{label}/{task}/{metric_name} has {size} values but {len(keys)} prediction keys"
                    )

    # A fresh generator with the documented seed creates one shared resampling
    # matrix per sample count. Reusing it across comparisons reduces Monte Carlo
    # noise when two model differences are compared.
    bootstrap_indices = {}
    results = []
    for baseline_label, candidate_label in comparisons:
        for task in TASKS:
            for metric_name in METRICS:
                baseline_values = metric_data[baseline_label][task][metric_name]
                candidate_values = metric_data[candidate_label][task][metric_name]
                if baseline_values.shape != candidate_values.shape:
                    raise ValueError(
                        f"Shape mismatch for {candidate_label} - {baseline_label}, "
                        f"{task}/{metric_name}: {candidate_values.shape} vs {baseline_values.shape}"
                    )
                sample_count = baseline_values.size
                if sample_count not in bootstrap_indices:
                    rng = np.random.default_rng(seed)
                    bootstrap_indices[sample_count] = rng.integers(
                        0, sample_count, size=(resamples, sample_count)
                    )

                baseline_mm = baseline_values * 1000.0
                candidate_mm = candidate_values * 1000.0
                delta_mm = candidate_mm - baseline_mm
                ci_low, ci_high = bootstrap_interval(
                    delta_mm, bootstrap_indices[sample_count], confidence
                )
                results.append(
                    {
                        "baseline": baseline_label,
                        "candidate": candidate_label,
                        "task": task,
                        "metric": metric_name,
                        "samples": sample_count,
                        "baseline_mean_mm": float(baseline_mm.mean()),
                        "candidate_mean_mm": float(candidate_mm.mean()),
                        "delta_mean_mm": float(delta_mm.mean()),
                        "ci_low_mm": ci_low,
                        "ci_high_mm": ci_high,
                        "significant": bool(ci_low > 0.0 or ci_high < 0.0),
                    }
                )

    return {
        "bootstrap": {
            "method": "paired percentile bootstrap",
            "resamples": resamples,
            "seed": seed,
            "confidence_percent": confidence,
            "delta_definition": "candidate - baseline",
            "unit": "mm",
        },
        "prediction_key_sha256": key_hashes,
        "results": results,
    }


def print_markdown(summary):
    print("| Comparison | Task | Metric | Baseline | Candidate | Delta [CI] | Significant |")
    print("|---|---|---|---:|---:|---:|:---:|")
    for row in summary["results"]:
        comparison = f"{row['candidate']} - {row['baseline']}"
        interval = (
            f"{row['delta_mean_mm']:+.3f} "
            f"[{row['ci_low_mm']:+.3f}, {row['ci_high_mm']:+.3f}]"
        )
        print(
            f"| {comparison} | {row['task']} | {METRIC_LABELS[row['metric']]} | "
            f"{row['baseline_mean_mm']:.3f} | {row['candidate_mean_mm']:.3f} | "
            f"{interval} | {'yes' if row['significant'] else 'no'} |"
        )


def main(args):
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if not 0.0 < args.confidence < 100.0:
        raise ValueError("--confidence must be between 0 and 100")

    experiments = parse_experiments(args.experiment)
    comparisons = resolve_comparisons(experiments, args.baseline, args.compare)
    summary = summarize(
        experiments,
        comparisons,
        args.bootstrap_resamples,
        args.bootstrap_seed,
        args.confidence,
    )
    print_markdown(summary)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved JSON summary to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap intervals from key-joint metric files."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("LABEL", "EXP_PATH", "EVAL_SUFFIX"),
        required=True,
        help="Register an experiment. Use '-' when the evaluation suffix is empty.",
    )
    parser.add_argument(
        "--baseline",
        help="Compare every other registered experiment with this baseline.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Select a comparison; may be repeated and takes precedence over --baseline.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=62)
    parser.add_argument("--confidence", type=float, default=95.0)
    parser.add_argument("--output", help="Optional JSON output path.")
    main(parser.parse_args())
