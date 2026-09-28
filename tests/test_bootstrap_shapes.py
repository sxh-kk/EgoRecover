"""Bootstrap cache admission tests; no trained checkpoint is required."""

import pytest
import torch

from egorecover.bootstrap_shapes import ModelBootstrapShapes


def test_common_startup_cache_binds_split_stats_take_and_body(tmp_path):
    path = tmp_path / "startup.pt"
    identity = {
        "scope": "model_generated_clean_prefix_bootstraps",
        "stats_sha256": "stats",
        "split_manifest_sha256": "split",
        "e7_checkpoint_sha256": "trained-e7",
    }
    episode = {
        "take": "dev_take",
        "beta_boot": torch.zeros(10),
        "normalized_motion": torch.zeros(20, 243),
        "initial_reference": torch.eye(4),
        "references": torch.eye(4).expand(20, 4, 4).clone(),
        "world_joints": torch.zeros(20, 22, 3),
        "floor_estimate_m": 0.0,
    }
    torch.save({"identity": identity, "episodes": {"episode": episode}}, path)
    cache = ModelBootstrapShapes(path, allowed_takes=["dev_take"], stats_sha256="stats", split_manifest_sha256="split")
    assert cache.for_record({"episode_id": "episode", "base_take_name": "dev_take"}).shape == (10,)
    with pytest.raises(ValueError, match="outside the permitted split"):
        ModelBootstrapShapes(path, allowed_takes=["train_take"], stats_sha256="stats", split_manifest_sha256="split")
    with pytest.raises(ValueError, match="statistics differ"):
        ModelBootstrapShapes(path, allowed_takes=["dev_take"], stats_sha256="wrong", split_manifest_sha256="split")
    episode["normalized_motion"][0, 0] = float("nan")
    torch.save({"identity": identity, "episodes": {"episode": episode}}, path)
    with pytest.raises(ValueError, match="Invalid clean-prefix startup"):
        ModelBootstrapShapes(path, allowed_takes=["dev_take"], stats_sha256="stats", split_manifest_sha256="split")
