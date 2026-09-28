"""Validate a shared P for controlled generator-loss comparisons."""

import json
from pathlib import Path

import torch

from egorecover.evaluation_protocol import file_sha256


def load_fixed_prior(experiment, target):
    experiment = Path(experiment)
    report = json.loads((experiment / "report.json").read_text())
    if not report.get("completed") or "prior" not in report:
        raise ValueError("Fixed prior requires a completed experiment.")
    keys = ("stats_sha256", "split_manifest_sha256", "bootstrap_cache_sha256", "reference_mode",
            "e7_checkpoint_sha256", "e7_weight_source", "splits", "selection_split_manifest_sha256",
            "selection_dataset_spec_sha256", "selection_bootstrap_cache_sha256", "prior_selection")
    for key in keys:
        if key not in report or key not in target or report[key] != target[key]:
            raise ValueError(f"Fixed prior experiment has a different {key}.")
    path = experiment / "prior.pt"
    checksum = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("stats_sha256", "split_manifest_sha256", "bootstrap_cache_sha256", "reference_mode"):
        if checkpoint.get(key) != target[key]:
            raise ValueError(f"Fixed prior checkpoint has a different {key}.")
    if checkpoint.get("kind") != "history_prior" or checkpoint.get("selected_step") != report["prior"]["selected_step"]:
        raise ValueError("Fixed prior checkpoint does not match its selection report.")
    if not all(bool(torch.isfinite(value).all()) for value in checkpoint["state_dict"].values()):
        raise ValueError("Fixed prior contains nonfinite weights.")
    return checkpoint, report, {
        "experiment": str(experiment), "checkpoint_sha256": checksum,
        "report_sha256": file_sha256(experiment / "report.json"),
        "original_prior_steps": report["prior_steps"],
        "original_geometry_weight": report["geometry_weight"], "original_fk_weight": report["fk_weight"],
    }
