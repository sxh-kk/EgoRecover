"""OFFLINE training cache of reachable predicted body histories.

Only train takes are accepted. This is never imported by online rollout.
"""

from pathlib import Path

import torch


class PredictedHistoryFrames:
    def __init__(self, path, *, train_takes, reference_mode, stats_sha256=None,
                 split_manifest_sha256=None, dataset_spec_sha256=None, require_model_bootstrap=False):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
        self.identity = payload["identity"]
        if self.identity.get("scope") != "training_only_predicted_histories":
            raise ValueError("Expected a training-only predicted-history cache.")
        if self.identity.get("reference_mode") != reference_mode:
            raise ValueError("Replay and training must use the same reference convention.")
        if stats_sha256 is not None and self.identity.get("stats_sha256") != stats_sha256:
            raise ValueError("Replay normalization statistics do not match training.")
        if split_manifest_sha256 is not None and self.identity.get("split_manifest_sha256") != split_manifest_sha256:
            raise ValueError("Replay split manifest does not match training.")
        if dataset_spec_sha256 is not None and self.identity.get("dataset_spec_sha256") != dataset_spec_sha256:
            raise ValueError("Replay dataset spec does not match training.")
        if not set(payload["take_names"]).issubset(set(train_takes)):
            raise ValueError("Replay contains takes outside the training split.")
        self.tensors = payload["tensors"]
        self.record_ids, self.time_indices = payload["record_ids"], payload["time_indices"]
        self.count = len(self.record_ids)
        if not self.count or len(payload["take_names"]) != self.count or len(self.time_indices) != self.count:
            raise ValueError("Replay metadata lengths do not match.")
        # Older GT-start engineering caches predate shape provenance. They may
        # still train the old dense objective, but can never pass formal FK loss.
        if "beta_boot" not in self.tensors:
            self.tensors = {
                **self.tensors,
                "beta_boot": torch.zeros(self.count, 10),
                "beta_boot_is_model": torch.zeros(self.count, dtype=torch.bool),
                "floor_estimate_m": torch.zeros(self.count),
            }
            self.identity = {**self.identity, "legacy_missing_bootstrap_shape": True}
        if self.tensors["beta_boot"].shape != (self.count, 10) or self.tensors["beta_boot_is_model"].shape != (
            self.count,
        ):
            raise ValueError("Replay bootstrap shape fields have wrong dimensions.")
        if self.tensors["beta_boot_is_model"].dtype != torch.bool:
            raise ValueError("Replay bootstrap provenance must be Boolean.")
        model_bootstrap = bool(self.tensors["beta_boot_is_model"].all())
        if model_bootstrap != bool(self.identity.get("bootstrap_is_model", False)):
            raise ValueError("Replay bootstrap provenance disagrees with its per-frame flags.")
        if require_model_bootstrap and not model_bootstrap:
            raise ValueError("Formal FK training rejects GT-start or legacy predicted-history caches.")
        for key, value in self.tensors.items():
            if value.shape[0] != self.count or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid/nonfinite cached field {key}.")

    def __len__(self):
        return self.count

    def batch(self, indices, device):
        return {key: value[indices].to(device) for key, value in self.tensors.items()}


def training_batch(teacher, replay, count, generator, device, probability=0.5):
    batch = teacher.batch(torch.randint(len(teacher), (count,), generator=generator), device)
    if replay is None:
        return batch
    if not 0 <= probability <= 1:
        raise ValueError("Replay probability must be in [0,1].")
    predicted = replay.batch(torch.randint(len(replay), (count,), generator=generator), device)
    if set(batch) != set(predicted):
        raise ValueError("Replay and teacher batches have different fields.")
    choose = (torch.rand(count, generator=generator) < probability).to(device)
    return {
        key: torch.where(choose.reshape(count, *([1] * (value.ndim - 1))), predicted[key], value)
        for key, value in batch.items()
    }
