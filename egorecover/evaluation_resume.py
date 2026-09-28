"""Atomic, identity-bound progress for causal evaluation trajectories."""

import json
import math
import os
from pathlib import Path

import torch

from egorecover.evaluation_protocol import file_sha256


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_trace(saved, frames):
    """Check the full committed suffix and the fields used by FK decoding."""
    if saved["frame_indices"] != frames or not frames:
        raise ValueError("Trajectory frames differ from the expected suffix.")
    if saved["reference_mode"] != "planar" or saved["body_history_source"] != "model_predictions_only":
        raise ValueError("Trajectory uses a different causal protocol.")
    startup = saved["bootstrap_frames"]
    if startup != frames[0]:
        raise ValueError("Trajectory startup length differs.")
    shapes = {
        "committed_motion": (len(frames), 243), "normalized_motion": (len(frames), 243),
        "references": (len(frames), 4, 4), "dense_world_joints": (len(frames), 22, 3),
        "bootstrap_motion": (startup, 243), "bootstrap_references": (startup, 4, 4),
        "bootstrap_world_joints": (startup, 22, 3), "bootstrap_initial_reference": (4, 4),
        "beta_boot": (10,),
    }
    for key, shape in shapes.items():
        if not torch.is_tensor(saved[key]) or tuple(saved[key].shape) != shape:
            raise ValueError(f"Invalid trajectory shape: {key}.")
    for value in saved.values():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite trajectory tensor.")
    if len(saved["actions"]) != len(frames) or len(saved["latency_ms"]) != len(frames):
        raise ValueError("Incomplete trajectory actions/timing.")
    if not all(math.isfinite(value) for value in saved["latency_ms"] + [saved["floor_estimate_m"]]):
        raise ValueError("Nonfinite trajectory metadata.")


class EvaluationProgress:
    def __init__(self, output, identity, tasks, *, resume=False):
        self.output = Path(output)
        self.path = self.output / "progress.json"
        expected = {"version": 1, "identity": identity, "tasks": tasks}
        if resume:
            if not self.path.is_file():
                raise ValueError("Cannot resume an old directory without an identity manifest.")
            self.data = json.loads(self.path.read_text())
            if any(self.data.get(key) != value for key, value in expected.items()):
                raise ValueError("Evaluation resume identity or task list differs.")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.data = {**expected, "traces": {}}
            atomic_json(self.path, self.data)

    def reusable(self, name):
        entry = self.data["traces"].get(name)
        if entry is None:
            return False
        path = self.output / name
        try:
            if file_sha256(path) != entry["sha256"]:
                return False
            saved = torch.load(path, map_location="cpu", weights_only=True)
            validate_trace(saved, self.data["tasks"][name])
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, EOFError):
            return False
        return True

    def save(self, name, saved):
        validate_trace(saved, self.data["tasks"][name])
        path = self.output / name
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        self.data["traces"][name] = {"sha256": file_sha256(path)}
        atomic_json(self.path, self.data)
