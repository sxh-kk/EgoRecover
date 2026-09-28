"""Body-only hold/CV from the SAME saved model bootstrap as online G.

World positions follow the codec's physical hold/constant-velocity definition.
No subsequent observation, GT state or per-frame realignment enters a baseline.
Only the dense position metric is evaluated; no FK/rotation result is claimed.
"""

import argparse
import json
from pathlib import Path

import torch

from egorecover.data import DEFAULT_SIGNAL, open_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    args = parser.parse_args()
    report = json.loads((args.rollout / "report.json").read_text())
    dataset, _ = open_dataset(signal=args.signal, spec_sha256=report["dataset_spec_sha256"])
    record = next(r for r in dataset.records if r["base_take_name"] == report["take"] and r["variant_name"] == "clean")
    saved = torch.load(args.rollout / "gaussian_clean.pt", map_location="cpu", weights_only=True)
    bootstrap = saved["bootstrap_world_joints"]
    indices = saved["frame_indices"]
    steps = torch.tensor(indices, dtype=bootstrap.dtype) - (saved["bootstrap_frames"] - 1)
    predictions = {
        "physical_hold": bootstrap[-1:].expand(len(indices), -1, -1),
        "constant_velocity": bootstrap[-1:] + steps[:, None, None] * (bootstrap[-1:] - bootstrap[-2:-1]),
    }
    # Read targets only after constructing predictions.
    target = torch.from_numpy(dataset.supervision(record["variant_id"])["kp3d"])[indices, :22]
    result = {
        "scope": "closed_loop_dense_positions_only",
        "initialization": report["initialization"],
        "take": report["take"],
        "frames": len(indices),
        "bootstrap": "same saved model outputs as G; no GT startup",
        "reference_mode": saved["reference_mode"],
        "results": {},
    }
    for name, prediction in predictions.items():
        error = (prediction - target).double().norm(dim=-1).mean(-1) * 1000
        result["results"][name] = {
            "mean_dense22_mm": float(error.mean()),
            "last_dense22_mm": float(error[-1]),
            "peak_dense22_mm": float(error.max()),
        }
    (args.rollout / "body_baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
