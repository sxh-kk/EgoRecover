import argparse
import json
from pathlib import Path

from .data import Episodes, online_inputs, save_json, save_torch, sha256
from .evaluation import evaluate
from .streaming import ACTION_BITS, rollout
from .train import load_bundle, resolve_device


def main():
    parser = argparse.ArgumentParser(
        description="Independent closed-loop prediction/evaluation; never feeds GT into inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--action", choices=tuple(ACTION_BITS), default="a11")
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--episode", help="If supplied, save predictions for this episode instead of evaluation JSON.")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Inference requires a new output file; existing results are not overwritten.")
    cfg, norm, tae, flow, checkpoint = load_bundle(args.checkpoint)
    device = resolve_device(args.device, cfg.precision)
    if flow is None:
        raise ValueError("Inference requires a Flow checkpoint.")
    flow, tae, norm = flow.to(device), tae.to(device), norm.to(device)
    episodes = Episodes(args.manifest, cfg)
    if episodes.digest != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the checkpoint's frozen dataset.")
    if args.episode:
        record = next((r for r in episodes.split(args.split) if r["id"] == args.episode), None)
        if record is None:
            raise ValueError("Episode is absent from requested split.")
        packet = episodes.load(record)
        prediction = rollout(flow, tae, norm, online_inputs(packet), action=args.action, seed=args.seed)
        save_torch(
            args.output,
            {
                "prediction": prediction,
                "first_frame": cfg.bootstrap,
                "stop_frame_exclusive": len(packet["pose"]),
                "fps": cfg.fps,
                "timestamps_s": packet["timestamps_s"][cfg.bootstrap :],
                "timing_provenance": packet.get(
                    "timing_provenance", {"timestamp_source": "provided_uniform_timestamps"}
                ),
                "episode": record["id"],
                "action": args.action,
                "seed": args.seed,
                "checkpoint_sha256": sha256(args.checkpoint),
                "coordinate_transforms": {
                    "floor_world_from_anchor": packet["anchor"],
                    "original_world_from_anchor": packet["original_world_from_anchor"],
                },
                "coordinate_spaces": {
                    "root": "anchor",
                    "joints": "anchor",
                    "root_floor_world": "floor-normalized world",
                    "joints_floor_world": "floor-normalized world",
                    "root_world": "original annotation world",
                    "joints_world": "original annotation world",
                },
            },
        )
    else:
        result = evaluate(flow, tae, norm, episodes, args.split, args.action, args.seed, args.start, args.stop)
        result["checkpoint_sha256"] = sha256(args.checkpoint)
        result["manifest_sha256"] = episodes.digest
        save_json(args.output, result)
        print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
