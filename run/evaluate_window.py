"""Evaluate solver cost, matched-cache effects, rollouts and paired horizons."""

import argparse
import json
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from egorecover.codec import MotionCodec
from egorecover.window import WindowConfig, WindowReconstructor
from egorecover.window.data import PreparedWindowData, file_sha256, synthetic_episode
from egorecover.window.evaluation import evaluate_episode, paired_horizons, summarize
from model.window_uniegomotion import WindowUniEgoMotion
from run.train_window import (
    FORMAT,
    code_identity,
    joint_decoder,
    load_predictor,
    load_smpl,
    save_checkpoint,
    save_json,
    smpl_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--steps", type=int, nargs="+", choices=(1, 2, 4, 10), default=[1, 2, 4, 10])
    parser.add_argument("--mode", choices=("both", "rollout", "matched"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draw", type=int, default=1062)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved.get("format") != FORMAT or saved["identity"]["synthetic"] != args.synthetic:
        raise ValueError("Checkpoint format or synthetic/production mode mismatch.")
    config = WindowConfig(**saved["protocol"])
    if args.synthetic:
        if args.data_root or args.prior_checkpoint:
            raise ValueError("Synthetic checks do not read production assets.")
        fixture = synthetic_episode(device=device)
        data, count = None, 1
    else:
        if not args.data_root or not args.prior_checkpoint:
            raise ValueError("Provide the recorded dataset and calibrated predictor.")
        data = PreparedWindowData(args.data_root, args.split, device=device)
        if data.digest != saved["identity"]["manifest"]:
            raise ValueError("Evaluation manifest differs from training identity.")
        if file_sha256(args.prior_checkpoint) != saved["identity"]["prior"]:
            raise ValueError("Evaluation predictor differs from training.")
        if smpl_identity() != saved["identity"]["smpl_asset"]:
            raise ValueError("Evaluation SMPL-X asset differs from training.")
        if any(not torch.equal(data.stats[k].cpu(), saved["stats"][k].cpu()) for k in data.stats):
            raise ValueError("Evaluation normalization statistics differ from training.")
        count = len(data.records)
    if args.limit:
        count = min(count, args.limit)
    codec = MotionCodec(saved["stats"]).to(device)
    predictor = load_predictor(args.prior_checkpoint, device, None if args.synthetic else data.digest)
    system = WindowReconstructor(codec, predictor, config)
    model = (
        WindowUniEgoMotion(get_cfg_defaults(), protocol=config, dropout=0, compact=saved["identity"]["compact"])
        .to(device)
        .eval()
    )
    model.load_state_dict(saved["ema"], strict=True)
    smpl = None if args.synthetic else load_smpl(device)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": file_sha256(args.checkpoint),
        "protocol": config.to_dict(),
        "manifest": saved["identity"]["manifest"],
        "split": args.split,
        "synthetic": args.synthetic,
        "mode": args.mode,
        "steps": args.steps,
        "draw": args.draw,
        "limit": args.limit,
        "selection": "no model selection; explicit checkpoint",
        "timebase": "verified source frame grid",
        "code": {**code_identity(), "run/evaluate_window.py": file_sha256(__file__)},
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != metadata:
        raise ValueError("Evaluation output identity changed; choose another output directory.")
    save_json(identity_path, metadata)
    report = {"identity": metadata, "results": {}}
    for steps in args.steps:
        groups, diagnostics = {}, []
        for index in range(count):
            episode = fixture if args.synthetic else data.load(index)
            joints = joint_decoder(episode, args.synthetic, smpl)
            for matched in ((False, True) if args.mode == "both" else (args.mode == "matched",)):
                row = evaluate_episode(
                    system, model, episode, joints, seed=args.draw, steps=steps, matched_cache=matched
                )
                prediction = row.pop("predictions")
                label = "matched" if matched else "rollout"
                groups.setdefault(label, []).append(row)
                if args.save_predictions:
                    save_checkpoint(
                        args.output / f"{episode.identifier}_{label}_nfe{steps}.pt",
                        {
                            "joints_floor_world": prediction,
                            "joints_world": episode.restore_world(prediction),
                            "original_world_from_floor_world": episode.original_world_from_floor_world.cpu(),
                            "identity": metadata,
                        },
                    )
            diagnostics.append(
                {
                    "episode": episode.identifier,
                    "take": episode.take,
                    "horizons": paired_horizons(system, model, episode, joints, seed=args.draw, steps=steps),
                }
            )
        report["results"][str(steps)] = {
            "summary": {k: summarize(v) for k, v in groups.items()},
            "rows": groups,
            "paired_horizons": diagnostics,
        }
        save_json(args.output / "report.json", report)
        print(json.dumps({"steps": steps, "summary": report["results"][str(steps)]["summary"]}), flush=True)


if __name__ == "__main__":
    main()
