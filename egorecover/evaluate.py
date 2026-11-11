"""Evaluate solver cost, matched-cache effects, rollouts and paired horizons."""

import csv
import json
from dataclasses import dataclass

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.checkpoint import FORMAT, load_predictor, resolve_prior_mode
from egorecover.codec import MotionCodec
from egorecover.data import open_window_data, resolve_data_format
from egorecover.diagnostics import synthetic_episode
from egorecover.evaluation import evaluate_episode, paired_anchor_diagnostics, summarize
from egorecover.fk import joint_decoder, load_smpl, smpl_identity
from egorecover.io import code_identity, file_sha256, save_checkpoint, save_json
from egorecover.metrics import comparison_protocol
from egorecover.models import MotionModel


def save_metrics_csv(path, report):
    """One comparable row per solver/mode/aggregation, with no hidden units."""
    rows = []
    for steps, result in report["results"].items():
        for mode, summary in result["summary"].items():
            metrics = {k: v for k, v in summary.items() if k != "comparison"}
            metrics.update(summary.get("comparison", {}))
            for aggregation, key in (("take_mean", "mean"), ("episode_mean", "episode_mean")):
                rows.append(
                    {
                        "nfe": steps,
                        "mode": mode,
                        "aggregation": aggregation,
                        "timing_aggregation": "pooled_steady_frames",
                        "synthetic": report["identity"]["synthetic"],
                        "episodes": len(result["rows"][mode]),
                        "takes": len({r["take"] for r in result["rows"][mode]}),
                        "floor_calibrated_episodes": sum(r.get("floor_calibrated", True) for r in result["rows"][mode]),
                        **{name: value[key] for name, value in metrics.items()},
                    }
                )
    prefix = [
        "nfe",
        "mode",
        "aggregation",
        "timing_aggregation",
        "synthetic",
        "episodes",
        "takes",
        "floor_calibrated_episodes",
    ]
    fields = prefix + sorted({key for row in rows for key in row} - set(prefix))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class EvaluationContext:
    saved: dict
    system: object
    model: object
    data: object
    fixture: object
    smpl: object
    count: int
    dataset_format: str
    dataset_protocol: dict
    prior_mode: str


def load_evaluation_context(args):
    """Load and validate the same checkpoint/assets for all evaluation tools."""
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved.get("format") != FORMAT or saved["identity"]["synthetic"] != args.synthetic:
        raise ValueError("Checkpoint format or synthetic/production mode mismatch.")
    config = WindowConfig(**saved["protocol"])
    smpl, fixture = None, None
    dataset_format = "synthetic"
    dataset_protocol = {"scope": "synthetic_diagnostic"}
    if args.synthetic:
        if args.data_root or args.prior_checkpoint:
            raise ValueError("Synthetic checks do not read production assets.")
        fixture = synthetic_episode(device=device)
        data, count = None, 1
    else:
        if not args.data_root:
            raise ValueError("Provide the recorded --data-root.")
        dataset_format = resolve_data_format(args.data_root, getattr(args, "data_format", "auto"))
        if dataset_format != saved["identity"].get("data_format", "paired"):
            raise ValueError("Evaluation data format differs from training.")
        smpl = load_smpl(device)
        bootstrap = None
        if dataset_format == "official":
            from .bootstrap import HeadAlignedBootstrap

            bootstrap = HeadAlignedBootstrap.from_smpl(smpl)
        data = open_window_data(
            args.data_root, args.split, device=device, data_format=dataset_format, bootstrap=bootstrap
        )
        dataset_protocol = getattr(data, "protocol", {"scope": "legacy_paired_cache"})
        if data.digest != saved["identity"]["manifest"]:
            raise ValueError("Evaluation manifest differs from training identity.")
        supplied_prior = file_sha256(args.prior_checkpoint) if args.prior_checkpoint else "constant_velocity"
        if supplied_prior != saved["identity"]["prior"]:
            raise ValueError("Evaluation predictor differs from training.")
        if smpl_identity() != saved["identity"]["smpl_asset"]:
            raise ValueError("Evaluation SMPL-X asset differs from training.")
        if any(not torch.equal(data.stats[k].cpu(), saved["stats"][k].cpu()) for k in data.stats):
            raise ValueError("Evaluation normalization statistics differ from training.")
        count = len(data.records)
    if args.limit:
        count = min(count, args.limit)
    codec = MotionCodec(saved["stats"]).to(device)
    prior_mode = resolve_prior_mode(getattr(args, "prior_mode", "auto"), args.prior_checkpoint)
    predictor = load_predictor(args.prior_checkpoint, device, None if args.synthetic else data.digest)
    system = WindowReconstructor(codec, predictor, config)
    model = MotionModel(config, ModelConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["ema"], strict=True)
    return EvaluationContext(
        saved, system, model, data, fixture, smpl, count, dataset_format, dataset_protocol, prior_mode
    )


def evaluate(args):
    context = load_evaluation_context(args)
    saved, system, model = context.saved, context.system, context.model
    data, fixture, smpl, count = context.data, context.fixture, context.smpl, context.count
    config = system.config
    skip_paired = bool(getattr(args, "skip_paired", False))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": file_sha256(args.checkpoint),
        "protocol": config.to_dict(),
        "manifest": saved["identity"]["manifest"],
        "data_format": context.dataset_format,
        "data_protocol": context.dataset_protocol,
        "prior_mode": context.prior_mode,
        "split": args.split,
        "synthetic": args.synthetic,
        "mode": args.mode,
        "steps": args.steps,
        "anchors": args.anchors,
        "skip_paired": skip_paired,
        "draw": args.draw,
        "limit": args.limit,
        "selection": "no model selection; explicit checkpoint",
        "timebase": "verified source frame grid",
        "comparison_protocol": comparison_protocol(),
        "latency_scope": "P + source + Flow + cache update + joint decoding; excludes SLAM, image encoding, observation packaging, noise sampling and metrics",
        "latency_population": "exclude first query; report first query separately; P95 nearest-rank",
        "latency_aggregation": "pool steady frame timings; FPS = 1000 / pooled mean latency; no averaging FPS or P95",
        "csv_aggregation": "take_mean/episode_mean apply to quality; timings always use pooled steady frames",
        "first_query_scope": "first evaluated query at frame 20; excludes startup history construction and model/asset loading; not time to first pose",
        "frame_budget_exceed_ratio": "fraction of measured core calls exceeding source frame period; excludes queuing; not end-to-end deadline misses",
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
            joints = joint_decoder(episode, args.synthetic, smpl, shape_mode=config.shape_mode)
            for matched in ((False, True) if args.mode == "both" else (args.mode == "matched",)):
                row = evaluate_episode(
                    system, model, episode, joints, seed=args.draw, steps=steps, matched_cache=matched, comparison=True
                )
                prediction = row.pop("predictions")
                rotations = row.pop("body_rotations")
                label = "matched" if matched else "rollout"
                groups.setdefault(label, []).append(row)
                if args.save_predictions:
                    save_checkpoint(
                        args.output / f"{episode.identifier}_{label}_nfe{steps}.pt",
                        {
                            "joints_floor_world": prediction,
                            "joints_world": episode.restore_world(prediction),
                            "body_rotations_world": rotations,
                            "timestamps": episode.timestamps[20:].cpu(),
                            "original_world_from_floor_world": episode.original_world_from_floor_world.cpu(),
                            "identity": metadata,
                        },
                    )
            if not skip_paired:
                available = any(frame >= 20 and frame + 10 < len(episode) for frame in args.anchors)
                diagnostics.append(
                    {
                        "episode": episode.identifier,
                        "take": episode.take,
                        "cache_source": "own_continuous_rollout",
                        "anchors": (
                            paired_anchor_diagnostics(
                                system, model, episode, joints, anchors=args.anchors, seed=args.draw, steps=steps
                            )
                            if available
                            else {}
                        ),
                        "status": "evaluated" if available else "not_available_short_clip",
                    }
                )
        report["results"][str(steps)] = {
            "summary": {k: summarize(v) for k, v in groups.items()},
            "rows": groups,
            "paired_horizons": diagnostics,
        }
        save_json(args.output / "report.json", report)
        save_metrics_csv(args.output / "metrics.csv", report)
        console = {}
        for mode, scores in report["results"][str(steps)]["summary"].items():
            compact = {key: scores[key]["mean"] for key in ("latency_ms", "p95_ms", "frame_budget_exceed_ratio")}
            for key in ("mpjpe_world_mm", "pa_mpjpe_mm", "wrist_pe_mm", "mpjre_f", "foot_slide_uniegomotion_mm"):
                if key in scores.get("comparison", {}):
                    compact[key] = scores["comparison"][key]["mean"]
            console[mode] = compact
        print(json.dumps({"steps": steps, "summary": console}), flush=True)
