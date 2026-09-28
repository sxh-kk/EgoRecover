"""Compare P and physical hold against G on exactly the saved dev frames."""

import argparse
import json
from pathlib import Path

import torch

from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.engineering import TeacherForcedFrames
from egorecover.prior import HistoryPrior
from run.engineering_pilot import weighted_mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-mode", choices=("planar", "legacy_se3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = json.loads((args.experiment / "report.json").read_text())
    dataset, _ = open_dataset()
    stats = torch.load(
        dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt", weights_only=False, map_location="cpu"
    )
    if args.reference_mode and not args.output:
        parser.error("A decoder override requires --output to preserve the original baseline report.")
    reference_mode = args.reference_mode or report.get("reference_mode", "legacy_se3")
    codec = MotionCodec(stats, reference_mode=reference_mode)
    frames = TeacherForcedFrames(
        dataset,
        codec,
        report["splits"]["dev"],
        contact_floor_source=report.get("contact_floor_source", "startup_estimate"),
    )
    indices = torch.linspace(0, len(frames) - 1, 64).long()
    data = frames.batch(indices, args.device)
    codec.to(args.device)
    prior = HistoryPrior().to(args.device)
    prior.load_state_dict(
        torch.load(args.experiment / "prior.pt", map_location=args.device, weights_only=True)["state_dict"]
    )
    prior.freeze()
    valid = torch.ones(data["history_motion"].shape[:2], device=args.device, dtype=torch.bool)
    with torch.no_grad():
        predictions = {
            "physical_hold": data["base_mu"],
            "constant_velocity": data["velocity_mu"],
            "P": prior(data["history_motion"], valid, data["base_mu"]),
        }
        metrics = {}
        for name, prediction in predictions.items():
            decoded = codec.decode_current(prediction[:, 0], data["previous_reference"])
            error = (decoded.joints[..., :3, 3] - data["target_joints"]).norm(dim=-1).mean(-1) * 1000
            metrics[name] = {
                "weighted_mse": float(weighted_mse(prediction, data["target"])),
                "dense22_mm": float(error.mean()),
            }
    if reference_mode == report.get("reference_mode", "legacy_se3"):
        for mode, values in report["sources"].items():
            metrics[f"G_{mode}_a11"] = {"dense22_mm": values["dense22_mm_per_action"]["a11"]}
    output = {
        "scope": report["scope"],
        "metric": report["metric"],
        "reference_mode": reference_mode,
        "num_frames": 64,
        "results": metrics,
    }
    (args.output or args.experiment / "baselines.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
