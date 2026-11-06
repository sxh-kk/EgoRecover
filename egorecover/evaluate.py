"""Evaluate solver cost, matched-cache effects, rollouts and paired horizons."""

import json

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.checkpoint import FORMAT, load_predictor
from egorecover.codec import MotionCodec
from egorecover.data import PreparedWindowData
from egorecover.diagnostics import synthetic_episode
from egorecover.evaluation import evaluate_episode, paired_horizons, summarize
from egorecover.fk import joint_decoder, load_smpl, smpl_identity
from egorecover.io import code_identity, file_sha256, save_checkpoint, save_json
from egorecover.models import MotionModel


def evaluate(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
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
    model = MotionModel(config, ModelConfig(**saved["model_config"])).to(device).eval()
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
        "code": code_identity(),
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
