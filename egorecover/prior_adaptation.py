"""Validated fixed-history data and same-state next-frame FK evaluation."""

import json
from collections import defaultdict
from pathlib import Path

import torch

from egorecover.evaluation_protocol import file_sha256
from egorecover.losses import fk_position_errors


SCOPES = {"train": "training_only_predicted_histories",
          "dev": "development_diagnostic_predicted_histories"}
SHAPES = {"history_motion": (20, 243), "base_mu": (1, 243), "velocity_mu": (1, 243),
          "target": (1, 243), "previous_reference": (4, 4), "target_joints": (22, 3),
          "beta_boot": (10,), "beta_boot_is_model": ()}
TEACHER_KEYS = ("reference_mode", "stats_sha256", "dataset_spec_sha256", "split_manifest_sha256",
                "bootstrap_cache_sha256", "bootstrap_identity", "p_sha256", "g_sha256",
                "variants", "seed", "sigma", "nfe", "policy")


class FixedHistoryFrames:
    def __init__(self, path, *, group, split_path, stats_path, records):
        if group not in SCOPES:
            raise ValueError("Only train and dev caches are accepted; holdout is excluded.")
        path = Path(path)
        report = json.loads(path.with_name("report.json").read_text())
        self.sha256 = file_sha256(path)
        if not report.get("completed") or report.get("cache_sha256") != self.sha256:
            raise ValueError("Incomplete or altered history cache.")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.identity = identity = payload["identity"]
        manifest = json.loads(Path(split_path).read_text())
        splits = manifest["splits"]
        all_takes = [take for names in splits.values() for take in names]
        if len(all_takes) != len(set(all_takes)):
            raise ValueError("Train/dev/holdout splits overlap.")
        if identity != report["identity"] or identity.get("scope") != SCOPES[group] or identity.get("group") != group:
            raise ValueError("History cache scope/group mismatch.")
        for key, expected in (("stats_sha256", file_sha256(stats_path)),
                              ("split_manifest_sha256", file_sha256(split_path)),
                              ("dataset_spec_sha256", manifest["dataset_spec_sha256"])):
            if identity.get(key) != expected:
                raise ValueError(f"History identity mismatch: {key}.")
        if not identity.get("bootstrap_is_model") or any(key not in identity for key in TEACHER_KEYS):
            raise ValueError("Missing frozen teacher or model-bootstrap provenance.")
        if set(identity["selected_takes"]) != set(splits[group]) or set(identity["group_takes"]) != set(splits[group]):
            raise ValueError("Cache does not cover the entire requested split.")
        self.tensors = {key: payload["tensors"][key] for key in SHAPES}
        self.metadata = {key: payload[key] for key in ("record_ids", "time_indices", "take_names", "generator_modes")}
        self.count = report["frames"]
        if not self.count or any(len(values) != self.count for values in self.metadata.values()):
            raise ValueError("History metadata length mismatch.")
        self.variants = []
        actual = []
        for rid, t, take, mode in zip(*self.metadata.values()):
            record = records[rid]
            if take != record["base_take_name"] or take not in splits[group]:
                raise ValueError("History row belongs to a different take/split.")
            self.variants.append(record["variant_name"])
            actual.append((rid, t, mode))
        expected = {(rid, t, mode) for rid, record in records.items()
                    if record["base_take_name"] in splits[group] and record["variant_name"] in identity["variants"]
                    for t in range(record["bootstrap_frames"], record["num_frames"])
                    for mode in ("gaussian", "history")}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Missing, duplicated, or unexpected history frame keys.")
        for key, shape in SHAPES.items():
            value = self.tensors[key]
            if value.shape != (self.count, *shape) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid cached tensor: {key}.")
        flags = self.tensors["beta_boot_is_model"]
        if flags.dtype != torch.bool or not bool(flags.all()):
            raise ValueError("FK evaluation requires model-generated bootstrap shape.")

    def __len__(self):
        return self.count

    def batch(self, indices, device):
        return {key: value[indices].to(device) for key, value in self.tensors.items()}


def validate_teachers(train, dev):
    if train.identity["group"] != "train" or dev.identity["group"] != "dev":
        raise ValueError("Training and selection caches were swapped.")
    if any(train.identity[key] != dev.identity[key] for key in TEACHER_KEYS):
        raise ValueError("Train and dev histories must use the same frozen P/G and protocol.")


def predict(prior, batch):
    history = batch["history_motion"]
    valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
    return prior(history, valid, batch["base_mu"])


def summarize_errors(errors, frames):
    if errors.shape != (len(frames),) or not bool(torch.isfinite(errors).all()):
        raise ValueError("Expected finite FK errors for every fixed dev state.")
    cells = defaultdict(list)
    for error, take, mode, variant in zip(errors.tolist(), frames.metadata["take_names"],
                                         frames.metadata["generator_modes"], frames.variants):
        cells[take, mode, variant].append(error)
    cell_means = {key: sum(values) / len(values) for key, values in cells.items()}
    takes = sorted(set(frames.metadata["take_names"]))
    per_take = {take: sum(value for key, value in cell_means.items() if key[0] == take)
                / sum(key[0] == take for key in cell_means) for take in takes}
    groups = sorted({(key[1], key[2]) for key in cell_means})
    return {"macro_fk22_mm": sum(per_take.values()) / len(per_take), "per_take_mm": per_take,
            "by_generator_variant_mm": {f"{mode}/{variant}":
                sum(cell_means[take, mode, variant] for take in takes) / len(takes)
                for mode, variant in groups}}


@torch.no_grad()
def evaluate(prior, frames, codec, smpl, device, batch_size, *, baselines=False):
    prior.eval()
    errors = {name: [] for name in (("prior", "hold", "constant_velocity") if baselines else ("prior",))}
    for start in range(0, len(frames), batch_size):
        batch = frames.batch(slice(start, start + batch_size), device)
        predictions = {"prior": predict(prior, batch)}
        if baselines:
            predictions.update(hold=batch["base_mu"], constant_velocity=batch["velocity_mu"])
        for name, prediction in predictions.items():
            errors[name].append(fk_position_errors(codec, prediction, batch, smpl).norm(dim=-1).mean(-1).cpu() * 1000)
    arrays = {name: torch.cat(values) for name, values in errors.items()}
    return {name: summarize_errors(values, frames) for name, values in arrays.items()}, arrays
