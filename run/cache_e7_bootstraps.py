"""Cache one causal, model-generated clean startup per pilot episode.

Requires a trained E7 Flow checkpoint. No supervision enters initialization.
"""

import argparse
import json
from pathlib import Path

import torch

from egorecover.bootstrap_shapes import load_e7_initializer
from egorecover.codec import MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.evaluation_protocol import DEFAULT_SPLIT_MANIFEST, file_sha256, load_fixed_split
from egorecover.rollout import initialize_history
from mydiffusion.flow_matching import FlowMatching


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-seed", type=int, default=1062)
    parser.add_argument("--groups", nargs="+", choices=("train", "dev", "holdout"), default=["train", "dev"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, _ = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    splits = load_fixed_split(args.split_manifest, dataset)
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False), reference_mode="planar").to(
        device
    )
    initializer = load_e7_initializer(args.checkpoint, device=device, weight_source=args.weight_source)
    allowed = {take for group in args.groups for take in splits[group]}
    records = [r for r in dataset.records if r["variant_name"] == "clean" and r["base_take_name"] in allowed]
    episodes = {}
    for index, record in enumerate(records):
        length = record["bootstrap_frames"]
        if length != 20:
            raise ValueError("This cache protocol requires exactly 20 clean startup frames.")
        prefix = {
            key: value.to(device)
            for key, value in dataset.observations(
                record["variant_id"], as_of=length - 1, history_frames=length
            ).items()
        }
        buffer, floor, _ = initialize_history(
            initializer, FlowMatching(), codec, prefix, seed=args.sampling_seed + index, length=length
        )
        world_joints = torch.stack([state.joints[..., :3, 3] for state in buffer.states])
        world_joints[..., 2] += floor
        episodes[record["episode_id"]] = {
            "take": record["base_take_name"],
            "beta_boot": buffer.beta_boot.cpu(),
            "normalized_motion": buffer.bootstrap_motion.cpu(),
            "initial_reference": buffer.initial_reference.cpu(),
            "references": torch.stack([state.reference for state in buffer.states]).cpu(),
            "world_joints": world_joints.cpu(),
            "floor_estimate_m": float(floor),
            "sampling_seed": args.sampling_seed + index,
        }
        print(f"{record['base_take_name']}: beta={buffer.beta_boot.cpu().tolist()}", flush=True)
    identity = {
        "scope": "model_generated_clean_prefix_bootstraps",
        "groups": args.groups,
        "reference_mode": codec.reference_mode,
        "e7_checkpoint_sha256": file_sha256(args.checkpoint),
        "e7_weight_source": args.weight_source,
        "stats_sha256": file_sha256(stats_path),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "dataset_spec_sha256": file_sha256(dataset.root / "spec.json"),
        "sampling_seed": args.sampling_seed,
        "prefix_frames": 20,
        "sampler": "FlowMatching Euler10",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "episodes": episodes}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {"identity": identity, "episode_count": len(episodes), "cache_sha256": file_sha256(args.output)},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
