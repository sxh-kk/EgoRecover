"""Pilot P/G training with fixed take splits and optional model-start FK loss.

The default remains an engineering diagnostic. Formal closed-loop FK selection
requires a trained E7 initializer and a frozen common startup cache.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from config.defaults import get_cfg_defaults
from dataset.smpl_utils import get_smpl
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.engineering import TeacherForcedFrames, conditions_from_batch
from egorecover.evaluation_protocol import DEFAULT_SPLIT_MANIFEST, load_fixed_split
from egorecover.history_flow import HistoryFlow
from egorecover.losses import physical_objective, dense_position_mm, fk_position_mm, weighted_representation_mse
from egorecover.prior import HistoryPrior
from egorecover.replay import PredictedHistoryFrames, training_batch
from egorecover.rollout import run_episode
from egorecover.smpl_evaluation import evaluate_saved_case, prepare_ground_truth
from egorecover.utility import generate_utility_labels
from egorecover.fixed_prior import load_fixed_prior
from model.history_uniegomotion import HistoryUniEgoMotion
from mydiffusion.flow_matching import FlowMatching


def weighted_mse(prediction, target):
    return weighted_representation_mse(prediction, target)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def closed_loop_dev_fk_mm(model, flow, prior, codec, smpl, dataset, records, bootstraps, gt_cache, *, seed):
    """Continuous predicted-history FK model selection on frozen clean dev takes."""
    errors = []
    for record in records:

        def observations(**kwargs):
            return dataset.observations(record["variant_id"], **kwargs)

        saved = run_episode(
            model,
            flow,
            None,
            None,
            codec,
            observations,
            num_frames=record["num_frames"],
            prior=prior,
            action="a11",
            seed=seed,
            startup=bootstraps.startup_for_record(record),
        )
        evaluated = evaluate_saved_case(smpl, codec, saved, gt_cache[record["episode_id"]], fault_window=None)
        errors.append(evaluated["metrics"]["mpjpe_body_m"] * 1000)
    return sum(errors) / len(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", "--training-seed", dest="training_seed", type=int, default=62)
    parser.add_argument("--sampling-seed", type=int, default=1062)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--selection-signal", type=Path)
    parser.add_argument("--selection-split-manifest", type=Path)
    parser.add_argument("--selection-bootstrap-cache", type=Path)
    parser.add_argument("--prior-steps", type=int, default=200)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--source-modes", nargs="+", choices=("gaussian", "history"), default=["gaussian", "history"])
    parser.add_argument("--fixed-prior-experiment", type=Path,
                        help="Reuse a completed experiment's exact P; loss weights then apply only to G.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--contact-floor-source", choices=("annotation", "startup_estimate"), default="annotation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weight-source", choices=("model", "ema"), default="model")
    parser.add_argument("--reference-mode", choices=("planar", "legacy_se3"), default="planar")
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--fk-weight", type=float, default=0.0)
    parser.add_argument("--record-loss-components", action="store_true")
    parser.add_argument("--bootstrap-cache", type=Path)
    parser.add_argument("--prior-selection", choices=("dense", "representation", "fk"), default="dense")
    parser.add_argument("--flow-selection", choices=("dense", "fk", "closed_loop_fk", "final"), default="dense")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--init-experiment", type=Path)
    parser.add_argument("--resume-incomplete", type=Path)
    parser.add_argument("--replay-probability", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve previous experiments.")
    if (args.prior_steps < 0 or (args.prior_steps == 0 and not args.fixed_prior_experiment)
            or args.flow_steps < 1 or args.batch_size < 1):
        parser.error("Step counts and batch size must be positive.")
    if len(set(args.source_modes)) != len(args.source_modes):
        parser.error("Source modes must be unique.")
    if args.fixed_prior_experiment and (args.prior_steps != 0 or args.init_experiment or args.resume_incomplete):
        parser.error("A fixed P requires --prior-steps 0 and cannot be combined with continuation/resume.")
    if args.resume_incomplete and args.source_modes != ["gaussian", "history"]:
        parser.error("Incomplete resume requires both source modes in their original order.")
    if args.geometry_weight < 0 or args.fk_weight < 0 or args.eval_every < 1:
        parser.error("Geometry/FK weights must be nonnegative and eval interval positive.")
    if not 0 <= args.replay_probability <= 1:
        parser.error("Replay probability must be in [0,1].")
    if (args.fk_weight or args.prior_selection == "fk" or args.flow_selection in ("fk", "closed_loop_fk")) and (
        args.checkpoint is None or args.bootstrap_cache is None
    ):
        parser.error("FK training/selection requires a trained E7 --checkpoint and --bootstrap-cache.")
    if args.bootstrap_cache is not None and args.checkpoint is None:
        parser.error("A model-generated bootstrap cache requires its trained E7 --checkpoint.")
    if args.history_cache is not None and args.bootstrap_cache is None:
        parser.error("Predicted-history adaptation requires a model-generated E7 bootstrap cache.")
    if args.init_experiment is not None and args.history_cache is None:
        parser.error("Continuing a pilot requires a predicted-history cache.")
    if args.resume_incomplete and (args.init_experiment or args.history_cache):
        parser.error("An incomplete budget resume cannot be mixed with replay continuation.")
    selection_options = (args.selection_signal, args.selection_split_manifest, args.selection_bootstrap_cache)
    if any(value is not None for value in selection_options) and not all(value is not None for value in selection_options):
        parser.error("External dev selection requires signal, split manifest, and bootstrap cache together.")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.training_seed)
    device = torch.device(args.device)
    expected_spec = json.loads(args.split_manifest.read_text())["dataset_spec_sha256"]
    dataset, handoff = open_dataset(signal=args.signal, spec_sha256=expected_spec)
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    codec = MotionCodec(stats, reference_mode=args.reference_mode)
    geometry_codec = MotionCodec(stats, reference_mode=args.reference_mode).to(device)
    splits = load_fixed_split(args.split_manifest, dataset)
    bootstraps = (
        ModelBootstrapShapes(
            args.bootstrap_cache,
            allowed_takes=splits["train"] + splits["dev"] + splits["holdout"],
            stats_sha256=sha256(stats_path),
            split_manifest_sha256=sha256(args.split_manifest),
            dataset_spec_sha256=sha256(dataset.root / "spec.json"),
        )
        if args.bootstrap_cache
        else None
    )
    if bootstraps and (
        bootstraps.identity["e7_checkpoint_sha256"] != sha256(args.checkpoint)
        or bootstraps.identity["e7_weight_source"] != args.weight_source
        or bootstraps.identity["reference_mode"] != codec.reference_mode
    ):
        raise ValueError("E7 training initializer and cached model startup identities differ.")
    selection_dataset, selection_splits, selection_bootstraps = dataset, splits, bootstraps
    if args.selection_signal:
        selection_spec = json.loads(args.selection_split_manifest.read_text())["dataset_spec_sha256"]
        selection_dataset, _ = open_dataset(signal=args.selection_signal, spec_sha256=selection_spec)
        selection_splits = load_fixed_split(args.selection_split_manifest, selection_dataset)
        if set(selection_splits["dev"]) & set(splits["train"]):
            raise ValueError("External dev selection overlaps training takes.")
        selection_bootstraps = ModelBootstrapShapes(
            args.selection_bootstrap_cache,
            allowed_takes=selection_splits["train"] + selection_splits["dev"] + selection_splits["holdout"],
            stats_sha256=sha256(stats_path),
            split_manifest_sha256=sha256(args.selection_split_manifest),
            dataset_spec_sha256=sha256(selection_dataset.root / "spec.json"),
        )
        for key, expected in (("e7_checkpoint_sha256", sha256(args.checkpoint)),
                              ("e7_weight_source", args.weight_source),
                              ("reference_mode", codec.reference_mode)):
            if selection_bootstraps.identity[key] != expected:
                raise ValueError(f"External dev bootstrap has a different {key}.")
    smpl = (
        get_smpl().to(device).eval().requires_grad_(False)
        if args.fk_weight or args.prior_selection == "fk" or args.flow_selection in ("fk", "closed_loop_fk")
        else None
    )
    smplx_asset = (
        Path(os.environ.get("SMPLX_MODEL_PATH", Path(__file__).resolve().parents[1] / "body_models/smplx"))
        / "SMPLX_NEUTRAL.npz"
    )
    replay = (
        PredictedHistoryFrames(
            args.history_cache,
            train_takes=splits["train"],
            reference_mode=codec.reference_mode,
            stats_sha256=sha256(stats_path),
            split_manifest_sha256=sha256(args.split_manifest),
            dataset_spec_sha256=sha256(dataset.root / "spec.json"),
            require_model_bootstrap=True,
        )
        if args.history_cache
        else None
    )
    if replay and replay.identity.get("bootstrap_cache_sha256") != sha256(args.bootstrap_cache):
        raise ValueError("Predicted-history cache and teacher batches use different model startups.")
    if replay and args.init_experiment:
        if (replay.identity.get("p_sha256") != sha256(args.init_experiment / "prior.pt")
                or replay.identity.get("g_sha256") != {
                    mode: sha256(args.init_experiment / f"g_{mode}.pt") for mode in ("gaussian", "history")
                }):
            raise ValueError("Continuation weights differ from the P/G that generated the replay cache.")
    report = {
        "scope": ("teacher_forced_development_pilot" if dataset.spec["base_split"] == "train"
                  else "teacher_forced_engineering_only"),
        "base_split": dataset.spec["base_split"],
        "history_source": "GT for development diagnostic; not online evaluation",
        "floor_source": "startup_head_z_median_minus_training_mean_head_height",
        "contact_floor_source": args.contact_floor_source,
        "metric": "single-step dense22 diagnostic; closed-loop FK selection only when explicitly requested",
        "initialization": str(args.checkpoint) if args.checkpoint else "random; no pretrained E7",
        "seed": args.training_seed,
        "training_seed": args.training_seed,
        "sampling_seed": args.sampling_seed,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "selection_split_manifest_sha256": sha256(args.selection_split_manifest or args.split_manifest),
        "selection_dataset_spec_sha256": sha256(selection_dataset.root / "spec.json"),
        "selection_bootstrap_cache_sha256": sha256(args.selection_bootstrap_cache or args.bootstrap_cache) if selection_bootstraps else None,
        "selection_dev_takes": selection_splits["dev"],
        "e7_checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "e7_weight_source": args.weight_source if args.checkpoint else None,
        "splits": splits,
        "sigma": args.sigma,
        "nfe": 10,
        "stats_sha256": sha256(stats_path),
        "dataset_handoff": handoff,
        "prior_steps": args.prior_steps,
        "flow_steps": args.flow_steps,
        "batch_size": args.batch_size,
        "reference_mode": args.reference_mode,
        "geometry_weight": args.geometry_weight,
        "fk_weight": args.fk_weight,
        "bootstrap_cache_sha256": sha256(args.bootstrap_cache) if bootstraps else None,
        "bootstrap_identity": bootstraps.identity if bootstraps else None,
        "smplx_asset_sha256": sha256(smplx_asset) if smpl is not None else None,
        "geometry_scale_m": 0.1,
        "prior_selection": args.prior_selection,
        "flow_selection": args.flow_selection,
        "predicted_history_cache": (
            {
                "path": str(args.history_cache),
                "sha256": sha256(args.history_cache),
                "identity": replay.identity,
                "frames": len(replay),
                "sample_probability": args.replay_probability,
            }
            if replay
            else None
        ),
        "init_experiment": str(args.init_experiment) if args.init_experiment else None,
        "sources": {},
        "source_modes": args.source_modes,
        "fixed_prior": None,
        "record_loss_components": args.record_loss_components,
    }
    fixed_prior = None
    if args.fixed_prior_experiment:
        fixed_prior, fixed_report, fixed_identity = load_fixed_prior(args.fixed_prior_experiment, report)
        report["fixed_prior"] = fixed_identity
        report["loss_weight_scope"] = "G only; P reused unchanged"
    resumed = None
    if args.resume_incomplete:
        resumed = json.loads((args.resume_incomplete / "report.json").read_text())
        same = (
            "split_manifest_sha256", "selection_split_manifest_sha256", "selection_dataset_spec_sha256",
            "selection_bootstrap_cache_sha256", "e7_checkpoint_sha256", "e7_weight_source",
            "stats_sha256", "bootstrap_cache_sha256", "prior_steps", "flow_steps", "batch_size",
            "training_seed", "sampling_seed", "reference_mode", "geometry_weight", "fk_weight",
            "prior_selection", "flow_selection", "sigma", "nfe", "splits", "selection_dev_takes",
        )
        if (resumed.get("completed") or "prior" not in resumed
                or set(resumed.get("sources", {})) != {"gaussian"}
                or any(resumed.get(key) != report[key] for key in same)):
            raise ValueError("Incomplete experiment differs from this budget run or lacks P/Gaussian.")
        if sha256(args.resume_incomplete / "g_gaussian.pt") != resumed["sources"]["gaussian"]["checkpoint_sha256"]:
            raise ValueError("Incomplete Gaussian checkpoint differs from its report.")
        prior_checkpoint = torch.load(args.resume_incomplete / "prior.pt", map_location="cpu", weights_only=True)
        if any(prior_checkpoint.get(key) != report[key] for key in
               ("split_manifest_sha256", "stats_sha256", "bootstrap_cache_sha256", "reference_mode")):
            raise ValueError("Incomplete prior checkpoint differs from this budget run.")
        report["resumed_from"] = str(args.resume_incomplete)
    def continuation_checkpoint(name):
        path = args.init_experiment / name
        value = torch.load(path, map_location="cpu", weights_only=False)
        for key, expected in (
            ("split_manifest_sha256", report["split_manifest_sha256"]),
            ("stats_sha256", report["stats_sha256"]),
            ("bootstrap_cache_sha256", report["bootstrap_cache_sha256"]),
            ("reference_mode", args.reference_mode),
        ):
            if value.get(key) != expected:
                raise ValueError(f"Continuation checkpoint {path} has a different {key}.")
        return value
    if replay:
        report["scope"] = ("mixed_history_development_pilot" if dataset.spec["base_split"] == "train"
                           else "mixed_history_engineering_only")
        report["history_source"] = (
            "training mixture of GT and frozen-model predicted histories; model bootstrap only on train takes; dev diagnostic remains GT-history"
        )
    report["evaluation_history_source"] = "GT for the single-step dev diagnostic; evaluate closed loop separately"
    if args.flow_selection == "closed_loop_fk":
        report["evaluation_history_source"] = (
            "predicted history for G checkpoint selection; GT single-step metrics remain diagnostics"
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Preparing take-disjoint teacher-forced engineering batches", flush=True)
    prior_train = TeacherForcedFrames(
        dataset,
        codec,
        splits["train"],
        clean_only=True,
        contact_floor_source=args.contact_floor_source,
        bootstrap_shapes=bootstraps,
    )
    prior_dev = TeacherForcedFrames(
        selection_dataset,
        codec,
        selection_splits["dev"],
        clean_only=True,
        contact_floor_source=args.contact_floor_source,
        bootstrap_shapes=selection_bootstraps,
    )
    report["floor_diagnostics"] = prior_train.floor_diagnostics | prior_dev.floor_diagnostics
    prior = HistoryPrior().to(device)
    if fixed_prior is not None:
        prior.load_state_dict(fixed_prior["state_dict"], strict=True)
    if args.init_experiment:
        prior.load_state_dict(continuation_checkpoint("prior.pt")["state_dict"], strict=True)
    if resumed:
        prior.load_state_dict(prior_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=0.01)
    random = torch.Generator().manual_seed(args.training_seed)
    dev = prior_dev.batch(torch.arange(len(prior_dev)), device)
    hv = torch.ones(dev["history_motion"].shape[:2], device=device, dtype=torch.bool)
    baseline = float(weighted_mse(dev["base_mu"], dev["target"]))
    baseline_dense = dense_position_mm(geometry_codec, dev["base_mu"], dev)
    baseline_fk = fk_position_mm(geometry_codec, dev["base_mu"], dev, smpl) if smpl is not None else None
    prior.eval()
    with torch.no_grad():
        initial_prediction = prior(dev["history_motion"], hv, dev["base_mu"])
        selected_representation = float(weighted_mse(initial_prediction, dev["target"]))
        selected_dense = dense_position_mm(geometry_codec, initial_prediction, dev)
        selected_fk = fk_position_mm(geometry_codec, initial_prediction, dev, smpl) if smpl is not None else None
    best = {"dense": selected_dense, "representation": selected_representation, "fk": selected_fk}[args.prior_selection]
    best_step, best_state = 0, cpu_state(prior)
    curve = []
    for step in range(1, (0 if resumed else args.prior_steps) + 1):
        prior.train()
        batch = training_batch(prior_train, replay, args.batch_size, random, device, args.replay_probability)
        valid = torch.ones(batch["history_motion"].shape[:2], device=device, dtype=torch.bool)
        prediction = prior(batch["history_motion"], valid, batch["base_mu"])
        loss = physical_objective(
            geometry_codec, prediction, batch, geometry_weight=args.geometry_weight, fk_weight=args.fk_weight, smpl=smpl
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == args.prior_steps:
            prior.eval()
            with torch.no_grad():
                dev_prediction = prior(dev["history_motion"], hv, dev["base_mu"])
                value = float(weighted_mse(dev_prediction, dev["target"]))
                dev_dense = dense_position_mm(geometry_codec, dev_prediction, dev)
                dev_fk = fk_position_mm(geometry_codec, dev_prediction, dev, smpl) if smpl is not None else None
            score = {"dense": dev_dense, "representation": value, "fk": dev_fk}[args.prior_selection]
            if score < best:
                best, best_step, best_state = score, step, cpu_state(prior)
                selected_representation, selected_dense = value, dev_dense
                selected_fk = dev_fk
            curve.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "dev_loss": value,
                    "dev_dense22_mm": dev_dense,
                    "dev_fk22_mm": dev_fk,
                }
            )
            print(
                f"P {step}/{args.prior_steps}: train={float(loss.detach()):.6f} dev_mse={value:.6f} dev_mm={dev_dense:.3f}",
                flush=True,
            )
    prior.load_state_dict(best_state, strict=True)
    prior.freeze()
    torch.save(
        {
            "state_dict": best_state,
            "kind": "history_prior",
            "scope": report["scope"],
            "selected_step": best_step,
            "reference_mode": codec.reference_mode,
            "selection": args.prior_selection,
            "stats_sha256": report["stats_sha256"],
            "split_manifest_sha256": report["split_manifest_sha256"],
            "training_seed": args.training_seed,
            "fk_weight": args.fk_weight,
            "bootstrap_cache_sha256": report["bootstrap_cache_sha256"],
        },
        args.output / "prior.pt",
    )
    report["prior"] = {
        "baseline_dev_loss": baseline,
        "selected_dev_loss": selected_representation,
        "baseline_dev_dense22_mm": baseline_dense,
        "selected_dev_dense22_mm": selected_dense,
        "baseline_dev_fk22_mm": baseline_fk,
        "selected_dev_fk22_mm": selected_fk,
        "selected_step": best_step,
        "curve": curve,
    }
    if resumed:
        shutil.copy2(args.resume_incomplete / "prior.pt", args.output / "prior.pt")
        report["prior"] = resumed["prior"]
    if fixed_prior is not None:
        shutil.copy2(args.fixed_prior_experiment / "prior.pt", args.output / "prior.pt")
        if sha256(args.output / "prior.pt") != report["fixed_prior"]["checkpoint_sha256"]:
            raise ValueError("Fixed prior changed while copying.")
        report["prior"] = fixed_report["prior"]
    del optimizer, best_state, prior_train, prior_dev, dev
    train = TeacherForcedFrames(
        dataset, codec, splits["train"], contact_floor_source=args.contact_floor_source, bootstrap_shapes=bootstraps
    )
    development = TeacherForcedFrames(
        selection_dataset, codec, selection_splits["dev"], contact_floor_source=args.contact_floor_source,
        bootstrap_shapes=selection_bootstraps
    )
    # Holdout takes remain unused for all model selection and this diagnostic.
    indices = torch.linspace(0, len(development) - 1, 64).long()
    evaluation = development.batch(indices, device)
    y_eval = conditions_from_batch(evaluation, prior=prior)
    codec = codec.to(device)
    clean_dev_records = [
        record
        for record in selection_dataset.records
        if record["base_take_name"] in selection_splits["dev"] and record["variant_name"] == "clean"
    ]
    gt_cache = (
        {
            record["episode_id"]: prepare_ground_truth(
                smpl,
                selection_dataset.supervision(record["variant_id"]),
                list(range(record["bootstrap_frames"], record["num_frames"])),
            )
            for record in clean_dev_records
        }
        if args.flow_selection == "closed_loop_fk"
        else None
    )
    for source_mode in args.source_modes:
        if resumed and source_mode == "gaussian":
            for name in ("g_gaussian.pt", "utility_diagnostic_gaussian.pt"):
                shutil.copy2(args.resume_incomplete / name, args.output / name)
            report["sources"]["gaussian"] = resumed["sources"]["gaussian"]
            (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            continue
        torch.manual_seed(args.training_seed)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device)
        if args.checkpoint:
            load_e7_weights(model, args.checkpoint, weight_source=args.weight_source)
        if args.init_experiment:
            previous = continuation_checkpoint(f"g_{source_mode}.pt")
            if previous.get("source_mode") != source_mode or previous.get("sigma") != args.sigma:
                raise ValueError(f"Continuation G checkpoint has different flow settings: {source_mode}.")
            if previous.get("e7_checkpoint_sha256") != report["e7_checkpoint_sha256"] or previous.get("e7_weight_source") != report["e7_weight_source"]:
                raise ValueError("Continuation G checkpoint has a different E7 initializer.")
            model.load_state_dict(previous["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=source_mode, sigma=args.sigma)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8.5e-5, weight_decay=0.01)
        random = torch.Generator().manual_seed(args.training_seed + 1)
        noise = torch.Generator(device=device).manual_seed(args.training_seed + 2)
        evaluation_noise = torch.Generator(device=device).manual_seed(args.sampling_seed)
        fixed_epsilon = torch.randn(evaluation["target"].shape, device=device, generator=evaluation_noise)
        model.eval()
        with torch.no_grad():
            before = float(
                flow.training_losses(model, evaluation["target"], y_eval, epsilon=fixed_epsilon, t=0.5)["loss"].mean()
            )
        curve = []
        best_flow_mm, selected_flow_step, best_flow_state = float("inf"), 0, None
        for step in range(1, args.flow_steps + 1):
            model.train()
            batch = training_batch(train, replay, args.batch_size, random, device, args.replay_probability)
            action = torch.randint(4, (args.batch_size,), generator=random).to(device)
            y = conditions_from_batch(batch, prior=prior, action=action)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = flow.training_losses(
                    model,
                    batch["target"],
                    y,
                    generator=noise,
                    return_diagnostics=args.geometry_weight > 0 or args.fk_weight > 0 or args.record_loss_components,
                )
            loss = (
                physical_objective(
                    codec,
                    result["model_output"],
                    batch,
                    geometry_weight=args.geometry_weight,
                    fk_weight=args.fk_weight,
                    smpl=smpl,
                    return_components=args.record_loss_components,
                )
                if args.geometry_weight or args.fk_weight or args.record_loss_components
                else result["loss"].mean()
            )
            components = loss if args.record_loss_components else None
            loss = components["loss"] if components is not None else loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step == args.flow_steps:
                entry = {"step": step, "train_loss": float(loss.detach())}
                if components is not None:
                    entry["loss_components"] = {key: float(value.detach()) for key, value in components.items()}
                curve.append(entry)
                print(f"G/{source_mode} {step}/{args.flow_steps}: train={float(loss.detach()):.6f}", flush=True)
                if components is not None:
                    print(f"G/{source_mode} components: {entry['loss_components']}", flush=True)
            if args.flow_selection in ("dense", "fk", "closed_loop_fk") and (
                step % args.eval_every == 0 or step == args.flow_steps
            ):
                model.eval()
                with torch.no_grad():
                    if args.flow_selection == "closed_loop_fk":
                        value = closed_loop_dev_fk_mm(
                            model,
                            flow,
                            prior,
                            codec,
                            smpl,
                            selection_dataset,
                            clean_dev_records,
                            selection_bootstraps,
                            gt_cache,
                            seed=args.sampling_seed,
                        )
                    else:
                        sample = flow.sample(model, y_eval, epsilon=fixed_epsilon)
                        value = (
                            dense_position_mm(codec, sample, evaluation)
                            if args.flow_selection == "dense"
                            else fk_position_mm(codec, sample, evaluation, smpl)
                        )
                if value < best_flow_mm:
                    best_flow_mm, selected_flow_step, best_flow_state = value, step, cpu_state(model)
                curve.append(
                    {"step": step, "train_loss": float(loss.detach()), f"dev_sample_{args.flow_selection}22_mm": value}
                )
                print(f"G/{source_mode} dev sample {step}: {value:.3f} mm ({args.flow_selection})", flush=True)
        if best_flow_state is not None:
            model.load_state_dict(best_flow_state, strict=True)
            del best_flow_state
        else:
            selected_flow_step = args.flow_steps
        model.eval()
        with torch.no_grad():
            after = float(
                flow.training_losses(model, evaluation["target"], y_eval, epsilon=fixed_epsilon, t=0.5)["loss"].mean()
            )
        checkpoint_path = args.output / f"g_{source_mode}.pt"
        torch.save(
            {
                "state_dict": cpu_state(model),
                "source_mode": source_mode,
                "sigma": args.sigma,
                "scope": report["scope"],
                "history_length": 20,
                "steps": args.flow_steps,
                "selected_step": selected_flow_step,
                "reference_mode": codec.reference_mode,
                "geometry_weight": args.geometry_weight,
                "fk_weight": args.fk_weight,
                "flow_selection": args.flow_selection,
                "bootstrap_cache_sha256": report["bootstrap_cache_sha256"],
                "stats_sha256": report["stats_sha256"],
                "split_manifest_sha256": report["split_manifest_sha256"],
                "e7_checkpoint_sha256": report["e7_checkpoint_sha256"],
                "e7_weight_source": report["e7_weight_source"],
                "training_seed": args.training_seed,
                "sampling_seed": args.sampling_seed,
            },
            checkpoint_path,
        )

        def dense_error(prediction):
            body = codec.decode_current(prediction[:, 0], evaluation["previous_reference"])
            return (body.joints[..., :3, 3] - evaluation["target_joints"]).norm(dim=-1).mean(-1) * 1000

        labels = generate_utility_labels(
            model, flow, y_eval, fixed_epsilon, dense_error, checkpoint_id=sha256(checkpoint_path)
        )
        labels["identity"].update(
            metric=report["metric"],
            scope=report["scope"],
            prior_sha256=sha256(args.output / "prior.pt"),
            stats_sha256=report["stats_sha256"],
            reference_mode=codec.reference_mode,
        )
        torch.save(
            {
                **labels,
                "record_ids": [development.record_ids[i] for i in indices],
                "time_indices": [development.time_indices[i] for i in indices],
            },
            args.output / f"utility_diagnostic_{source_mode}.pt",
        )
        errors = labels["errors"]
        report["sources"][source_mode] = {
            "dev_loss_before": before,
            "dev_loss_after": after,
            "selected_step": selected_flow_step,
            "curve": curve,
            "dense22_mm_per_action": dict(zip(("a11", "a10", "a01", "a00"), errors.mean(0).tolist())),
            "dense22_oracle_gap_mm": float((errors[:, 0] - errors.min(1).values).mean()),
            "oracle_action_counts": torch.bincount(errors.argmin(1), minlength=4).tolist(),
            "checkpoint_sha256": labels["identity"]["checkpoint_id"],
        }
        print(json.dumps(report["sources"][source_mode], indent=2), flush=True)
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del model, optimizer, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["completed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved pilot to {args.output}; G selection={args.flow_selection}; independent validation remains separate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
