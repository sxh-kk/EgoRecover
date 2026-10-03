"""Evaluate trained models on a common, take-disjoint development split."""

import argparse
import json
import os
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults, validate_e7
from dataset.smpl_utils import get_smpl
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import BodyState, MotionCodec
from egorecover.data import open_dataset
from egorecover.evaluation_protocol import event_window, file_sha256, load_fixed_split, paired_fault_delta
from egorecover.evaluation_resume import EvaluationProgress, atomic_json
from egorecover.fk import FixedShapeFK
from egorecover.history_flow import HistoryFlow
from egorecover.prior import HistoryPrior
from egorecover.rollout import run_episode
from egorecover.smpl_evaluation import decode_committed_rollout, prepare_ground_truth
from model.history_uniegomotion import HistoryUniEgoMotion


def shard_takes(takes, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid take shard.")
    return takes[index::count]


def parse_experiments(values):
    experiments = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in experiments or "/" in label:
            raise ValueError("Experiments must be unique LABEL=PATH arguments.")
        experiments[label] = Path(path)
    return experiments


def experiment_weight_hashes(experiments, source_modes):
    """Hash exactly the checkpoints load_model uses, including prior-only's G."""
    sources = {"gaussian" if mode == "prior_only" else mode for mode in source_modes}
    names = ("prior.pt", *(f"g_{source}.pt" for source in sorted(sources)), "report.json")
    return {label: {name: file_sha256(path / name) for name in names}
            for label, path in experiments.items()}


def load_model(experiment, mode, *, device, stats_sha256, e7_sha256, reference_mode):
    report = json.loads((experiment / "report.json").read_text())
    if not report.get("completed") or report["stats_sha256"] != stats_sha256 or report["e7_checkpoint_sha256"] != e7_sha256:
        raise ValueError(f"Incompatible training report: {experiment}")
    if report["reference_mode"] != reference_mode or report["e7_weight_source"] != "ema":
        raise ValueError("Model and common evaluation protocol differ.")
    prior_data = torch.load(experiment / "prior.pt", map_location="cpu", weights_only=True)
    if prior_data["stats_sha256"] != stats_sha256 or prior_data["split_manifest_sha256"] != report["split_manifest_sha256"]:
        raise ValueError("Prior and training report identities differ.")
    prior = HistoryPrior().to(device).eval()
    prior.load_state_dict(prior_data["state_dict"], strict=True)
    prior.freeze()
    source = "gaussian" if mode == "prior_only" else mode
    flow_data = torch.load(experiment / f"g_{source}.pt", map_location="cpu", weights_only=True)
    for key, expected in (("stats_sha256", stats_sha256), ("e7_checkpoint_sha256", e7_sha256),
                          ("split_manifest_sha256", report["split_manifest_sha256"]),
                          ("reference_mode", reference_mode), ("source_mode", source)):
        if flow_data[key] != expected:
            raise ValueError(f"{experiment}/{source} has a different {key}.")
    model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval()
    model.load_state_dict(flow_data["state_dict"], strict=True)
    model.requires_grad_(False)
    return model, HistoryFlow(source_mode=source, sigma=flow_data["sigma"]), prior, report


@torch.no_grad()
def smpl22_error(smpl, codec, saved, ground_truth, *, batch_size=32):
    state, beta_boot, _ = decode_committed_rollout(codec, saved)
    if saved["frame_indices"] != ground_truth["frame_indices"]:
        raise ValueError("Predicted and audited GT frame indices differ.")
    fk = FixedShapeFK(smpl, beta_boot)
    errors = []
    for start in range(0, len(saved["frame_indices"]), batch_size):
        end = start + batch_size
        part = BodyState(state.joints[start:end], state.reference[start:end], state.auxiliary[start:end])
        joints, _ = fk.project(part, joint_count=22, return_verts=False)
        target = ground_truth["joints"][start:end, :22].to(joints.device)
        errors.append((joints - target).norm(dim=-1).mean(dim=-1).cpu())
    per_frame = torch.cat(errors) * 1000
    if not bool(torch.isfinite(per_frame).all()):
        raise ValueError("Nonfinite SMPL22 FK error.")
    return per_frame.tolist()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", action="append", default=[], help="LABEL=PATH; repeat for each model")
    parser.add_argument("--include-frozen-e7", action="store_true")
    parser.add_argument("--group", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--take-name", action="append")
    parser.add_argument("--variants", nargs="+", default=["clean", "freeze_3s", "drift_0p03mps"])
    parser.add_argument("--source-modes", nargs="+", choices=("gaussian", "history", "prior_only"),
                        default=["gaussian", "history", "prior_only"])
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume a directory with a verified progress manifest")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Choose a new output directory.")
    experiments = parse_experiments(args.experiment)
    if len(set(args.source_modes)) != len(args.source_modes):
        parser.error("Source modes must be unique.")
    if args.include_frozen_e7 and "frozen_e7" in experiments:
        parser.error("The frozen_e7 label is reserved for the baseline.")
    if not experiments and not args.include_frozen_e7:
        parser.error("At least one experiment or frozen E7 is required.")
    if "clean" not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("Variants must be unique and include clean.")
    args.variants = ["clean"] + [variant for variant in args.variants if variant != "clean"]
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, _ = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    splits = load_fixed_split(args.split_manifest, dataset)
    selected = args.take_name or splits[args.group]
    if len(selected) != len(set(selected)) or not set(selected).issubset(splits[args.group]):
        parser.error("Selected takes must be unique and inside the evaluation group.")
    takes = shard_takes(selected, args.shard_index, args.shard_count)
    if not takes:
        parser.error("This shard contains no takes.")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats_sha256 = file_sha256(stats_path)
    e7_sha256 = file_sha256(args.checkpoint)
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False), reference_mode="planar").to(device)
    bootstraps = ModelBootstrapShapes(
        args.bootstrap_cache,
        allowed_takes=splits["train"] + splits["dev"] + splits["holdout"],
        stats_sha256=stats_sha256,
        split_manifest_sha256=file_sha256(args.split_manifest),
        dataset_spec_sha256=file_sha256(dataset.root / "spec.json"),
    )
    if (bootstraps.identity["e7_checkpoint_sha256"] != e7_sha256
            or bootstraps.identity["e7_weight_source"] != "ema"
            or bootstraps.identity["reference_mode"] != codec.reference_mode):
        raise ValueError("Shared startup and E7 evaluation protocol differ.")
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    records = {
        take: {record["variant_name"]: record for record in dataset.records if record["base_take_name"] == take}
        for take in takes
    }
    for take, variants in records.items():
        if not set(args.variants).issubset(variants):
            raise ValueError(f"Missing requested variant for {take}.")
    report = {
        "completed": False,
        "group": args.group,
        "takes": takes,
        "variants": args.variants,
        "source_modes": args.source_modes,
        "sampling_seed": args.seed,
        "dataset_spec_sha256": expected_spec,
        "evaluation_split_manifest_sha256": file_sha256(args.split_manifest),
        "bootstrap_cache_sha256": file_sha256(args.bootstrap_cache),
        "e7_checkpoint_sha256": e7_sha256,
        "stats_sha256": stats_sha256,
        "experiments": {label: str(path) for label, path in experiments.items()},
        "results": [],
    }
    for label, path in experiments.items():
        training = json.loads((path / "report.json").read_text())
        if not training.get("completed"):
            raise ValueError(f"Incomplete training experiment: {label}.")
        if set(takes) & set(training["splits"]["train"]):
            raise ValueError(f"Evaluation take occurs in {label}'s training split.")
    model_cases = [(label, path, mode) for label, path in experiments.items() for mode in args.source_modes]
    if args.include_frozen_e7:
        model_cases.append(("frozen_e7", None, "gaussian"))
    identity = {key: value for key, value in report.items() if key not in ("completed", "results")}
    identity["weights"] = experiment_weight_hashes(experiments, args.source_modes)
    root = Path(__file__).resolve().parents[1]
    identity["smplx_asset_sha256"] = file_sha256(
        Path(os.environ.get("SMPLX_MODEL_PATH", root / "body_models/smplx")) / "SMPLX_NEUTRAL.npz"
    )
    identity["e7_config_sha256"] = file_sha256(root / "config/e7.yaml")
    identity["code"] = {
        str(path.relative_to(root)): file_sha256(path)
        for directory in ("egorecover", "model", "diffusion", "utils", "dataset", "config")
        for path in sorted((root / directory).rglob("*.py"))
    }
    identity["code"]["run/evaluate_paired_development.py"] = file_sha256(Path(__file__))
    tasks = {
        f"{label}_{mode}_{take}_{variant}.pt": list(range(records[take][variant]["bootstrap_frames"],
                                                       records[take][variant]["num_frames"]))
        for label, _, mode in model_cases for take in takes for variant in args.variants
    }
    progress = EvaluationProgress(args.output, identity, tasks, resume=args.resume)
    atomic_json(args.output / "report.json", report)
    pending = {}
    for label, path, mode in model_cases:
        if path is None:
            cfg = get_cfg_defaults()
            cfg.merge_from_file("config/e7.yaml")
            validate_e7(cfg)
            model = HistoryUniEgoMotion(cfg).to(device).eval().requires_grad_(False)
            load_e7_weights(model, args.checkpoint, weight_source="ema")
            flow, prior = HistoryFlow(source_mode="gaussian", sigma=1.0), None
        else:
            model, flow, prior, _ = load_model(
                path, mode, device=device, stats_sha256=stats_sha256,
                e7_sha256=e7_sha256, reference_mode=codec.reference_mode,
            )
        for take in takes:
            siblings = [records[take][variant] for variant in args.variants]
            for record in siblings:
                trace = args.output / f"{label}_{mode}_{take}_{record['variant_name']}.pt"
                pending[label, mode, take, record["variant_name"]] = trace
                if progress.reusable(trace.name):
                    print(f"Reused {label}/{mode}/{take}/{record['variant_name']}", flush=True)
                    continue
                def observations(**kwargs):
                    return dataset.observations(record["variant_id"], **kwargs)

                saved = run_episode(
                    model, flow, None, None, codec, observations,
                    num_frames=record["num_frames"], prior=prior,
                    prior_only=mode == "prior_only", seed=args.seed,
                    startup=bootstraps.startup_for_record(record),
                )
                progress.save(trace.name, saved)
                print(f"Saved {label}/{mode}/{take}/{record['variant_name']}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # The entire online inference pass finishes before any supervision is read.
    for take in takes:
        clean_record = records[take]["clean"]
        first = torch.load(pending[model_cases[0][0], model_cases[0][2], take, "clean"],
                           map_location="cpu", weights_only=True)
        ground_truth = prepare_ground_truth(
            smpl, dataset.supervision(clean_record["variant_id"]), first["frame_indices"]
        )
        for label, path, mode in model_cases:
            siblings = [records[take][variant] for variant in args.variants]
            evaluated = {}
            for record in siblings:
                variant = record["variant_name"]
                trace = pending[label, mode, take, variant]
                saved = torch.load(trace, map_location="cpu", weights_only=True)
                per_frame = smpl22_error(smpl, codec, saved, ground_truth)
                evaluated[variant] = per_frame
                row = {
                    "experiment": label,
                    "mode": "frozen_e7" if path is None else mode,
                    "take": take,
                    "variant": variant,
                    "smpl22_mpjpe_mm": sum(per_frame) / len(per_frame),
                    "trace": str(trace),
                    "trace_sha256": file_sha256(trace),
                    "per_frame_smpl22_mm": per_frame,
                    "fault_window": list(event_window(record, siblings)) if variant != "clean" else None,
                }
                if variant != "clean":
                    row["paired_fault_delta"] = paired_fault_delta(
                        per_frame, evaluated["clean"],
                        saved["frame_indices"], event_window(record, siblings), fps=dataset.spec["motion_fps"],
                    )
                report["results"].append(row)
                print(f"{label}/{mode}/{take}/{variant}: {row['smpl22_mpjpe_mm']:.3f} mm", flush=True)
            atomic_json(args.output / "report.json", report)
    report["completed"] = True
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
