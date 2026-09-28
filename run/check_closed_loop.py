"""Predicted-history stability check with an explicitly identified initializer.

Without pretrained E7, --allow-random-initializer is required and all outputs
are labelled as an untrained-startup diagnostic, never method performance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config.defaults import get_cfg_defaults
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.codec import MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.evaluation_protocol import (
    DEFAULT_SPLIT_MANIFEST,
    event_window,
    file_sha256,
    load_fixed_split,
    phase_indices,
)
from egorecover.prior import HistoryPrior
from egorecover.rollout import run_episode
from egorecover.history_flow import HistoryFlow
from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import UniEgoMotion
from module.ema import apply_ema_weights_from_checkpoint
from mydiffusion.flow_matching import FlowMatching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--bootstrap-weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--bootstrap-cache", type=Path)
    parser.add_argument("--allow-random-initializer", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--reference-mode", choices=("planar", "legacy_se3"), default="planar")
    parser.add_argument("--take-index", type=int, default=0)
    parser.add_argument("--take-name")
    parser.add_argument("--source-modes", nargs="+", choices=("gaussian", "history", "prior_only"),
                        default=["gaussian", "history"])
    parser.add_argument("--variants", nargs="+", default=["clean", "freeze_3s", "drift_0p03mps"])
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--group", choices=("dev", "holdout"), default="dev")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    if args.bootstrap_checkpoint is None and not args.allow_random_initializer:
        parser.error("Supply E7 weights or explicitly permit an untrained-startup diagnostic.")
    if args.bootstrap_cache is not None and args.bootstrap_checkpoint is None:
        parser.error("A model-generated startup cache requires its trained E7 --bootstrap-checkpoint.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    experiment = json.loads((args.experiment / "report.json").read_text())
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, handoff = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    frozen_splits = load_fixed_split(args.split_manifest, dataset)
    if experiment["splits"] != frozen_splits and set(frozen_splits[args.group]) & set(experiment["splits"]["train"]):
        raise ValueError("Evaluation takes overlap the model training split.")
    take = args.take_name or frozen_splits[args.group][args.take_index]
    if take not in frozen_splits[args.group]:
        raise ValueError("Requested take is outside the evaluation group.")
    records = [
        record
        for record in dataset.records
        if record["base_take_name"] == take and record["variant_name"] in args.variants
    ]
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats_sha = file_sha256(stats_path)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    codec = MotionCodec(stats, reference_mode=args.reference_mode).to(device)
    bootstraps = (
        ModelBootstrapShapes(
            args.bootstrap_cache,
            allowed_takes=frozen_splits["train"] + frozen_splits["dev"] + frozen_splits["holdout"],
            stats_sha256=stats_sha,
            split_manifest_sha256=file_sha256(args.split_manifest),
            dataset_spec_sha256=file_sha256(dataset.root / "spec.json"),
        )
        if args.bootstrap_cache
        else None
    )
    if bootstraps and (
        bootstraps.identity["e7_checkpoint_sha256"] != file_sha256(args.bootstrap_checkpoint)
        or bootstraps.identity["e7_weight_source"] != args.bootstrap_weight_source
        or bootstraps.identity["reference_mode"] != codec.reference_mode
    ):
        raise ValueError("Cached common startup differs from this E7 initializer configuration.")
    prior = HistoryPrior().to(device)
    prior_path = args.experiment / "prior.pt"
    prior_checkpoint = torch.load(prior_path, map_location=device, weights_only=True)
    if prior_checkpoint.get("reference_mode") != codec.reference_mode:
        raise ValueError("Prior checkpoint reference convention differs from this rollout.")
    if prior_checkpoint.get("stats_sha256", stats_sha) != stats_sha:
        raise ValueError("Prior checkpoint normalization statistics differ from this rollout.")
    prior.load_state_dict(prior_checkpoint["state_dict"])
    prior.freeze()
    torch.manual_seed(args.seed)
    initializer = UniEgoMotion(get_cfg_defaults()).to(device).eval()
    if args.bootstrap_checkpoint:
        checkpoint = torch.load(args.bootstrap_checkpoint, map_location=device, weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        initializer.load_state_dict(state, strict=True)
        if args.bootstrap_weight_source == "ema" and not apply_ema_weights_from_checkpoint(initializer, checkpoint):
            raise ValueError("Requested bootstrap EMA format is unavailable.")
    initializer.requires_grad_(False)
    report = {
        "scope": "closed_loop_stability_only",
        "reference_mode": codec.reference_mode,
        "completion_semantics": "execution and finite metrics only; inspect magnitude separately",
        "initialization": str(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else "random E7; not pretrained",
        "bootstrap_is_trained": args.bootstrap_checkpoint is not None,
        "history_source": "model_predictions_only; no resets",
        "take": take,
        "group": args.group,
        "training_split_manifest_sha256": experiment["split_manifest_sha256"],
        "evaluation_split_manifest_sha256": file_sha256(args.split_manifest),
        "base_split": dataset.spec["base_split"],
        "metric": "dense22_world_position_error_mm; not SMPL FK",
        "sampling_seed": args.seed,
        "generator_sampling_seed": args.seed,
        "bootstrap_sampling_seed": bootstraps.identity["sampling_seed"] if bootstraps else args.seed,
        "stats_sha256": stats_sha,
        "dataset_spec_sha256": file_sha256(dataset.root / "spec.json"),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "dataset_handoff_status": handoff["status"],
        "bootstrap_checkpoint_sha256": file_sha256(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else None,
        "bootstrap_weight_source": args.bootstrap_weight_source if args.bootstrap_checkpoint else None,
        "bootstrap_cache_sha256": file_sha256(args.bootstrap_cache) if bootstraps else None,
        "bootstrap_cache_identity": bootstraps.identity if bootstraps else None,
        "prior_checkpoint_sha256": file_sha256(prior_path),
        "prior_checkpoint_step": prior_checkpoint.get("selected_step"),
        "generator_checkpoints": {},
        "results": [],
    }
    for mode in args.source_modes:
        generator_mode = "gaussian" if mode == "prior_only" else mode
        checkpoint_path = args.experiment / f"g_{generator_mode}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("source_mode") != generator_mode or checkpoint.get("reference_mode") != codec.reference_mode:
            raise ValueError(f"{mode} checkpoint mode/reference convention differs from this rollout.")
        if checkpoint.get("stats_sha256", stats_sha) != stats_sha:
            raise ValueError(f"{mode} checkpoint normalization statistics differ from this rollout.")
        if bootstraps and checkpoint.get("e7_checkpoint_sha256") != bootstraps.identity["e7_checkpoint_sha256"]:
            raise ValueError(f"{mode} G and cached startup use different trained E7 weights.")
        report["generator_checkpoints"][mode] = {
            "sha256": file_sha256(checkpoint_path),
            "selected_step": checkpoint.get("selected_step"),
            "sigma": checkpoint["sigma"],
            "weight_source": checkpoint.get("e7_weight_source"),
            "e7_checkpoint_sha256": checkpoint.get("e7_checkpoint_sha256"),
            "reference_mode": checkpoint["reference_mode"],
            "stats_identity_verified": "stats_sha256" in checkpoint,
            "generator_used": mode != "prior_only",
        }
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=generator_mode, sigma=checkpoint["sigma"])
        for record in records:
            last_requested = [-1]

            def observations(**kwargs):
                last_requested[0] = kwargs["as_of"]
                return dataset.observations(record["variant_id"], **kwargs)

            print(f"Rolling out {mode}/{record['variant_name']} with predicted history", flush=True)
            try:
                result = run_episode(
                    model,
                    flow,
                    initializer,
                    FlowMatching(),
                    codec,
                    observations,
                    num_frames=min(args.frames, record["num_frames"]),
                    prior=prior,
                    prior_only=mode == "prior_only",
                    seed=args.seed,
                    startup=bootstraps.startup_for_record(record) if bootstraps else None,
                )
                trace_path = args.output / f"{mode}_{record['variant_name']}.pt"
                torch.save(result, trace_path)
                # Supervision and fault windows are read ONLY after inference.
                target = torch.from_numpy(dataset.supervision(record["variant_id"])["kp3d"][:, :22])
                indices = result["frame_indices"]
                error = (result["dense_world_joints"] - target[indices]).norm(dim=-1).mean(-1) * 1000
                if (
                    not bool(torch.isfinite(target).all())
                    or not bool(torch.isfinite(error).all())
                    or not bool(torch.isfinite(result["bootstrap_world_joints"]).all())
                    or not bool(np.isfinite(result["latency_ms"]).all())
                ):
                    raise ValueError("Nonfinite GT, rollout, startup or latency metric.")
                window = event_window(record, records)
                sections = phase_indices(indices, window)
                row = {
                    "source_mode": mode,
                    "variant": record["variant_name"],
                    "finite": True,
                    "trace_sha256": file_sha256(trace_path),
                    "fault_window": list(window) if window is not None else None,
                    "mean_dense22_mm": float(error.mean()),
                    "last_frame_dense22_mm": float(error[-1]),
                    "first_frame_dense22_mm": float(error[0]),
                    "peak_dense22_mm": float(error.max()),
                    "phase_dense22_mm": {
                        name: float(error[items].mean()) if items else None for name, items in sections.items()
                    },
                    "latency_median_ms": float(np.median(result["latency_ms"])),
                    "latency_p95_ms": float(np.percentile(result["latency_ms"], 95)),
                    "floor_estimate_m": result["floor_estimate_m"],
                    "frames": len(indices),
                    "bootstrap_dense22_mm": float(
                        (result["bootstrap_world_joints"] - target[: result["bootstrap_frames"]]).norm(dim=-1).mean()
                        * 1000
                    ),
                }
            except (ValueError, RuntimeError) as error:
                row = {
                    "source_mode": mode,
                    "variant": record["variant_name"],
                    "completed": False,
                    "error": str(error),
                    "last_requested_frame": last_requested[0],
                }
            report["results"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row, indent=2), flush=True)
        del model, checkpoint
    report["completed"] = all(row.get("finite", False) for row in report["results"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
