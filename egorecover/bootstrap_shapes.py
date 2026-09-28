"""Verified model-generated startup shapes for offline training/evaluation."""

import math
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from model.uniegomotion import UniEgoMotion
from module.ema import apply_ema_weights_from_checkpoint


class ModelBootstrapShapes:
    def __init__(self, path, *, allowed_takes, stats_sha256, split_manifest_sha256, dataset_spec_sha256=None):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        identity = payload["identity"]
        if identity.get("scope") != "model_generated_clean_prefix_bootstraps":
            raise ValueError("Expected a model-generated clean-prefix bootstrap cache.")
        if identity.get("stats_sha256") != stats_sha256:
            raise ValueError("Bootstrap cache normalization statistics differ from training.")
        if identity.get("split_manifest_sha256") != split_manifest_sha256:
            raise ValueError("Bootstrap cache take split differs from training.")
        if dataset_spec_sha256 is not None and identity.get("dataset_spec_sha256") != dataset_spec_sha256:
            raise ValueError("Bootstrap cache dataset spec differs from training.")
        if not identity.get("e7_checkpoint_sha256"):
            raise ValueError("Bootstrap cache must identify trained E7 weights.")
        self.identity = identity
        self.episodes = payload["episodes"]
        if not isinstance(self.episodes, dict) or not self.episodes:
            raise ValueError("Bootstrap cache contains no episodes.")
        if not {entry["take"] for entry in self.episodes.values()}.issubset(set(allowed_takes)):
            raise ValueError("Bootstrap cache includes takes outside the permitted split.")
        for episode_id, entry in self.episodes.items():
            beta = entry["beta_boot"]
            if (
                not isinstance(episode_id, str)
                or not isinstance(beta, torch.Tensor)
                or beta.shape != (10,)
                or not bool(torch.isfinite(beta).all())
            ):
                raise ValueError(f"Invalid bootstrap shape for episode {episode_id}.")
            motion = entry.get("normalized_motion")
            reference = entry.get("initial_reference")
            references = entry.get("references")
            world_joints = entry.get("world_joints")
            if (
                not isinstance(motion, torch.Tensor)
                or motion.shape != (20, 243)
                or not isinstance(reference, torch.Tensor)
                or reference.shape != (4, 4)
                or not isinstance(references, torch.Tensor)
                or references.shape != (20, 4, 4)
                or not isinstance(world_joints, torch.Tensor)
                or world_joints.shape != (20, 22, 3)
                or not all(bool(torch.isfinite(value).all()) for value in (motion, reference, references, world_joints))
                or not math.isfinite(float(entry.get("floor_estimate_m", float("nan"))))
            ):
                raise ValueError(f"Invalid clean-prefix startup for episode {episode_id}.")

    def for_record(self, record):
        entry = self.episodes[record["episode_id"]]
        if entry["take"] != record["base_take_name"]:
            raise ValueError("Bootstrap cache episode/take identity mismatch.")
        return entry["beta_boot"].clone()

    def startup_for_record(self, record):
        self.for_record(record)
        return self.episodes[record["episode_id"]]


def load_e7_initializer(checkpoint_path, *, device, weight_source="model"):
    """Load the original E7 network for causal 20-frame initialization."""
    if weight_source not in ("model", "ema"):
        raise ValueError("E7 startup weight source must be model or ema.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("model."): value for key, value in state.items()}
    model = UniEgoMotion(get_cfg_defaults()).to(device).eval()
    model.load_state_dict(state, strict=True)
    if weight_source == "ema" and not apply_ema_weights_from_checkpoint(model, checkpoint):
        raise ValueError("Requested bootstrap EMA format is unavailable.")
    return model.requires_grad_(False)
