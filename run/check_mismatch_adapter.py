"""Check every handed-off variant through the model's 9D -> 18D codec."""

import argparse
import json
from pathlib import Path

import torch

from egorecover.codec import MotionCodec, transform_from_9d
from egorecover.data import open_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verification/mismatch_adapter.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"scope": "real-observation adapter geometry; no body reconstruction accuracy", "datasets": []}
    for profile in ("extended", "pilot"):
        dataset, _ = open_dataset(profile=profile)
        stats = torch.load(
            dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt", weights_only=False, map_location="cpu"
        )
        codec = MotionCodec(stats)
        max_error = 0.0
        missing_img = missing_traj = frames = 0
        for record in dataset.records:
            count = record["num_frames"]
            observed = dataset.observations(record["variant_id"], as_of=count - 1, history_frames=count)
            available = observed["traj_available"]
            reference = torch.eye(4).expand(count, 4, 4)
            encoded = codec.encode_observation(observed["aria_traj_obs"], reference, available)
            assert encoded.shape == (count, 18) and torch.isfinite(encoded).all()
            raw = codec.denormalize(encoded[available], "traj")
            rebuilt = reference[available] @ transform_from_9d(raw[:, 9:]) @ transform_from_9d(raw[:, :9])
            target = transform_from_9d(observed["aria_traj_obs"][available])
            if len(target):
                max_error = max(max_error, float((rebuilt - target).abs().max()))
                torch.testing.assert_close(rebuilt, target, atol=1e-4, rtol=1e-5)
            assert torch.isfinite(observed["img_feats"]).all()
            assert not encoded[~available].any()
            missing_img += int((~observed["img_available"]).sum())
            missing_traj += int((~available).sum())
            frames += count
        report["datasets"].append(
            {
                "profile": profile,
                "variants": len(dataset),
                "frames": frames,
                "max_transform_absolute_error": max_error,
                "missing_image_frames": missing_img,
                "missing_trajectory_frames": missing_traj,
            }
        )
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
