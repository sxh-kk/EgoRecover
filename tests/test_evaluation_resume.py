import copy

import pytest
import torch

from egorecover.evaluation_resume import EvaluationProgress, validate_trace


def trace():
    return {
        "frame_indices": [20, 21], "reference_mode": "planar",
        "body_history_source": "model_predictions_only", "bootstrap_frames": 20,
        "committed_motion": torch.zeros(2, 243), "normalized_motion": torch.zeros(2, 243),
        "references": torch.eye(4).repeat(2, 1, 1), "dense_world_joints": torch.zeros(2, 22, 3),
        "bootstrap_motion": torch.zeros(20, 243), "bootstrap_references": torch.eye(4).repeat(20, 1, 1),
        "bootstrap_world_joints": torch.zeros(20, 22, 3), "bootstrap_initial_reference": torch.eye(4),
        "beta_boot": torch.zeros(10), "actions": ["a11", "a11"], "latency_ms": [1., 1.],
        "floor_estimate_m": 0.,
    }


def test_resume_skips_complete_and_recomputes_missing_or_corrupt(tmp_path):
    directory = tmp_path / "evaluation"
    tasks = {name: [20, 21] for name in ("clean.pt", "fault.pt", "missing.pt")}
    identity = {"seed": 62, "weights": "abc"}
    store = EvaluationProgress(directory, identity, tasks)
    expected = trace()
    store.save("clean.pt", expected)
    store.save("fault.pt", expected)
    (directory / "fault.pt").write_bytes(b"interrupted write")
    resumed = EvaluationProgress(directory, identity, tasks, resume=True)
    generated = []
    for name in tasks:
        if not resumed.reusable(name):
            generated.append(name)
            resumed.save(name, expected)
    assert generated == ["fault.pt", "missing.pt"]
    for name in tasks:
        assert resumed.reusable(name)
        actual = torch.load(directory / name, weights_only=True)
        assert torch.equal(actual["committed_motion"], expected["committed_motion"])
    (directory / "clean.pt").unlink()
    assert not resumed.reusable("clean.pt")


def test_resume_rejects_identity_change_and_legacy_directory(tmp_path):
    directory = tmp_path / "evaluation"
    EvaluationProgress(directory, {"weights": "a"}, {"clean.pt": [20, 21]})
    with pytest.raises(ValueError, match="identity"):
        EvaluationProgress(directory, {"weights": "b"}, {"clean.pt": [20, 21]}, resume=True)
    with pytest.raises(ValueError, match="task list"):
        EvaluationProgress(directory, {"weights": "a"}, {"clean.pt": [20]}, resume=True)
    with pytest.raises(ValueError, match="without an identity"):
        EvaluationProgress(tmp_path, {}, {}, resume=True)


def test_trace_requires_complete_finite_committed_history():
    saved = trace()
    validate_trace(saved, [20, 21])
    for key, value in (("frame_indices", [20]), ("committed_motion", torch.zeros(1, 243)),
                       ("beta_boot", torch.full((10,), float("nan")))):
        broken = copy.deepcopy(saved)
        broken[key] = value
        with pytest.raises(ValueError):
            validate_trace(broken, [20, 21])
