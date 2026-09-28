import pytest
import torch

from egorecover.replay import PredictedHistoryFrames, training_batch


def test_replay_rejects_dev_takes_and_incompatible_coordinate_conventions(tmp_path):
    path = tmp_path / "frames.pt"
    payload = {
        "identity": {"scope": "training_only_predicted_histories", "reference_mode": "planar"},
        "tensors": {"target": torch.zeros(2, 1, 243)},
        "record_ids": ["a", "b"],
        "time_indices": [20, 21],
        "take_names": ["train_a", "dev_a"],
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="outside the training split"):
        PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar")
    payload["take_names"] = ["train_a"] * 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="same reference"):
        PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="legacy_se3")
    valid = PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar")
    assert len(valid) == 2
    with pytest.raises(ValueError, match="rejects GT-start"):
        PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar", require_model_bootstrap=True)
    payload["tensors"]["target"][0, 0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonfinite"):
        PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar")


def test_replay_requires_truthful_per_frame_bootstrap_provenance(tmp_path):
    path = tmp_path / "frames.pt"
    payload = {
        "identity": {
            "scope": "training_only_predicted_histories",
            "reference_mode": "planar",
            "bootstrap_is_model": True,
        },
        "tensors": {
            "target": torch.zeros(2, 1, 243),
            "beta_boot": torch.zeros(2, 10),
            "beta_boot_is_model": torch.tensor([True, False]),
            "floor_estimate_m": torch.zeros(2),
        },
        "record_ids": ["a", "b"],
        "time_indices": [20, 21],
        "take_names": ["train_a", "train_a"],
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance disagrees"):
        PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar")
    payload["tensors"]["beta_boot_is_model"][:] = True
    torch.save(payload, path)
    assert (
        len(
            PredictedHistoryFrames(path, train_takes=["train_a"], reference_mode="planar", require_model_bootstrap=True)
        )
        == 2
    )


def test_replay_mixes_complete_rows_and_is_paired_by_explicit_generator():
    class Frames:
        def __init__(self, value):
            self.value = value

        def __len__(self):
            return 10

        def batch(self, indices, device):
            return {
                "history_motion": torch.full((len(indices), 20, 243), self.value, device=device),
                "target": torch.full((len(indices), 1, 243), self.value, device=device),
            }

    first = training_batch(Frames(1.0), Frames(2.0), 32, torch.Generator().manual_seed(62), "cpu")
    second = training_batch(Frames(1.0), Frames(2.0), 32, torch.Generator().manual_seed(62), "cpu")
    assert torch.equal(first["target"], second["target"])
    assert torch.equal(first["history_motion"][:, 0, 0], first["target"][:, 0, 0])
    assert set(first["target"][:, 0, 0].tolist()) == {1.0, 2.0}
