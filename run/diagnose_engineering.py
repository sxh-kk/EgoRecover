"""Frozen-checkpoint root-cause probes; includes explicitly oracle GT bootstrap.

All GT substitutions are OFFLINE diagnostics, never production rollout inputs.
No parameter is trained or changed by these probes.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from config.defaults import get_cfg_defaults
from egorecover.annotations import body_states_from_supervision
from egorecover.codec import BodyState, MotionCodec, planar_reference, transform_from_9d, transform_to_9d
from egorecover.data import open_dataset
from egorecover.engineering import TeacherForcedFrames, conditions_from_batch
from egorecover.history import HistoryBuffer
from egorecover.history_flow import HistoryFlow
from egorecover.prior import HistoryPrior
from egorecover.rollout import initialize_history
from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import UniEgoMotion
from mydiffusion.flow_matching import FlowMatching
from run.engineering_pilot import weighted_mse


def body_slice(state, index):
    return BodyState(*(getattr(state, key)[index] for key in ("joints", "reference", "auxiliary")))


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("exp/egorecover_engineering_v1"))
    parser.add_argument("--output", type=Path, default=Path("exp/egorecover_diagnosis_v1"))
    parser.add_argument("--probe-set", choices=("initial", "geometry", "planar_pair", "none"), default="initial")
    parser.add_argument("--skip-single-step", action="store_true")
    parser.add_argument("--eval-split", choices=("train", "dev"), default="dev")
    parser.add_argument("--take-index", type=int, default=0)
    parser.add_argument("--rollout-noise-seed", type=int, default=62)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    experiment = json.loads((args.experiment / "report.json").read_text())
    dataset, _ = open_dataset()
    stats = torch.load(dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt", weights_only=False)
    # This script reproduces the ORIGINAL decoder and applies its own explicit
    # intervention. Do not silently change old causal comparisons as defaults evolve.
    codec = MotionCodec(stats, reference_mode="legacy_se3")
    frames = TeacherForcedFrames(dataset, codec, experiment["splits"][args.eval_split])
    data = frames.batch(torch.linspace(0, len(frames) - 1, 64).long(), device)
    codec.to(device)
    prior = HistoryPrior().to(device)
    prior.load_state_dict(
        torch.load(args.experiment / "prior.pt", map_location=device, weights_only=True)["state_dict"]
    )
    prior.freeze()
    y = conditions_from_batch(data, prior=prior)
    epsilon = torch.randn(data["target"].shape, device=device, generator=torch.Generator(device=device).manual_seed(64))
    channels = {
        "joint_rotation": [9 * j + k for j in range(22) for k in range(6)],
        "joint_translation": [9 * j + k for j in range(22) for k in (6, 7, 8)],
        "reference_rotation": list(range(198, 204)),
        "reference_translation": list(range(204, 207)),
        "hands": list(range(207, 231)),
        "contact": list(range(231, 233)),
        "beta": list(range(233, 243)),
    }

    def project_reference(prediction):
        raw = codec.denormalize(prediction)
        projected = planar_reference(transform_from_9d(raw[..., 198:207]))
        raw[..., 198:207] = transform_to_9d(projected)
        return codec.normalize(raw)

    def metrics(prediction):
        squared = (prediction - data["target"]).square()

        def error(candidate):
            body = codec.decode_current(candidate[:, 0], data["previous_reference"])
            pos = body.joints[..., :3, 3]
            return float((pos - data["target_joints"]).norm(dim=-1).mean() * 1000)

        oracle_reference = prediction.clone()
        oracle_reference[..., 198:207] = data["target"][..., 198:207]
        oracle_local = prediction.clone()
        indices = channels["joint_translation"]
        oracle_local[..., indices] = data["target"][..., indices]
        body = codec.decode_current(prediction[:, 0], data["previous_reference"])
        pos = body.joints[..., :3, 3]
        target = data["target_joints"]
        return {
            "weighted_mse": float(weighted_mse(prediction, data["target"])),
            "channel_mse": {name: float(squared[..., idx].mean()) for name, idx in channels.items()},
            "dense22_mm": error(prediction),
            "planar_reference_dense22_mm": error(project_reference(prediction)),
            "oracle_reference_dense22_mm": error(oracle_reference),
            "oracle_local_translation_dense22_mm": error(oracle_local),
            "root_relative_dense22_mm": float(
                ((pos - pos[:, :1]) - (target - target[:, :1])).norm(dim=-1).mean() * 1000
            ),
        }

    report = {
        "scope": "OFFLINE root-cause diagnostics; oracle substitutions are not deployable results",
        "experiment": str(args.experiment),
        "eval_split": args.eval_split,
        "probe_set": args.probe_set,
        "random_bootstrap_copy": "physical_state_exact_copy_no_reencoding",
        "rollout_noise_seed": args.rollout_noise_seed,
        "single_step": {
            "hold": metrics(data["base_mu"]),
            "constant_velocity": metrics(data["velocity_mu"]),
            "P": metrics(y["prior_mu"]),
            "GT_roundtrip": metrics(data["target"]),
        },
        "training_frame_counts": {
            "unique_base_next_frame_targets": len(experiment["splits"]["train"]) * 180,
            "variants_per_take": 7,
            "note": "7 variants share the same body labels; overlapping histories are not independent sequences",
        },
        "closed_loop": [],
    }
    models = {}
    for mode in ("gaussian", "history"):
        checkpoint = torch.load(args.experiment / f"g_{mode}.pt", map_location=device, weights_only=True)
        model = HistoryUniEgoMotion(get_cfg_defaults()).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        flow = HistoryFlow(source_mode=mode, sigma=checkpoint["sigma"])
        models[mode] = (model, flow)
        report["parameter_count_G"] = sum(p.numel() for p in model.parameters())
        if args.skip_single_step:
            continue
        entry = {"teacher_path": {}, "sampling": {}}
        for t in (0.1, 0.5, 1.0):
            result = flow.training_losses(model, data["target"], y, epsilon=epsilon, t=t, return_diagnostics=True)
            entry["teacher_path"][str(t)] = metrics(result["model_output"])
        for nfe in (1, 10, 50):
            entry["sampling"][str(nfe)] = metrics(flow.sample(model, y, epsilon=epsilon, num_steps=nfe))
        report["single_step"][mode] = entry
        print(mode, json.dumps({n: r["dense22_mm"] for n, r in entry["sampling"].items()}), flush=True)

    (args.output / "report.json").write_text(json.dumps(finite_json(report), indent=2) + "\n")
    if args.probe_set == "none":
        return

    record = next(
        r
        for r in dataset.records
        if r["base_take_name"] == experiment["splits"]["dev"][args.take_index] and r["variant_name"] == "clean"
    )

    def observed(t, length):
        return {
            k: v.to(device)
            for k, v in dataset.observations(record["variant_id"], as_of=t, history_frames=length).items()
        }

    prefix = observed(19, 20)
    torch.manual_seed(62)
    initializer = UniEgoMotion(get_cfg_defaults()).to(device).eval().requires_grad_(False)
    random_buffer, floor, _ = initialize_history(initializer, FlowMatching(), codec, prefix)
    labels = dataset.supervision(record["variant_id"])
    clean = observed(199, 200)
    truth = body_states_from_supervision(
        labels, clean["aria_traj_obs"], floor_height=floor, contact_floor_height=float(labels["floor_height"])
    )
    oracle_start = body_slice(truth, slice(0, 20))
    oracle_codes = codec.encode_history(
        BodyState(*(getattr(oracle_start, k)[None] for k in ("joints", "reference", "auxiliary"))), truth.reference[0:1]
    )[0]
    report["bootstrap"] = {
        "take": record["base_take_name"],
        "floor_error_m": floor - float(labels["floor_height"]),
        "random_dense22_mm": float(
            (torch.stack([s.joints[..., :3, 3] for s in random_buffer.states]) - truth.joints[:20, :, :3, 3])
            .norm(dim=-1)
            .mean()
            * 1000
        ),
        "random_history_absmax": float(random_buffer.encoded().abs().max()),
        "GT_history_absmax": float(data["history_motion"].abs().max()),
    }
    # A GT bootstrap is an artificial upper-bound diagnostic, never substituted
    # into production run_episode. After frame 19 every variant uses predictions.
    initial_probes = [
        ("history", "random", "P"),
        ("history", "GT_oracle", "P"),
        ("history", "GT_oracle", "hold"),
        ("gaussian", "GT_oracle", "P"),
        ("history", "random", "hold"),
    ]
    probes = (
        [(m, s, p, "a11", False) for m, s, p in initial_probes]
        if args.probe_set == "initial"
        else [
            ("history", "GT_oracle", "P", "a10", False),
            ("history", "GT_oracle", "P", "a00", False),
            ("history", "GT_oracle", "P", "a11", True),
        ]
    )
    if args.probe_set == "planar_pair":
        probes = [("history", "GT_oracle", "P", "a11", planar) for planar in (False, True)]
    for mode, start, prior_kind, action, planar in probes:
        buffer = (
            HistoryBuffer.from_bootstrap(codec, oracle_codes, truth.reference[0])
            if start == "GT_oracle"
            else copy.deepcopy(random_buffer)
        )
        model, flow = models[mode]
        row = {
            "source": mode,
            "bootstrap": start,
            "prior": prior_kind,
            "action": action,
            "planar_reference_projection": planar,
            "trace": [],
            "completed": False,
        }
        for t in range(20, 200):
            trace = {"frame": t}
            try:
                condition = buffer.conditions(observed(t, 1), action=action, floor_height=floor)
                trace["history_absmax"] = float(condition["history_motion"].abs().max())
                trace["history_channel_absmax"] = {
                    k: float(condition["history_motion"][..., idx].abs().max()) for k, idx in channels.items()
                }
                trace["base_mu_absmax"] = float(condition["prior_mu"].abs().max())
                trace["traj_absmax"] = float(condition["traj"].abs().max())
                if prior_kind == "P":
                    condition["prior_mu"] = prior(
                        condition["history_motion"], condition["history_valid"], condition["prior_mu"]
                    )
                trace["mu_absmax"] = float(condition["prior_mu"].abs().max())
                noise = torch.randn(
                    1,
                    1,
                    243,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(args.rollout_noise_seed + t + 1),
                )
                prediction = flow.sample(model, condition, epsilon=noise)[0, 0]
                if planar:
                    prediction = project_reference(prediction)
                trace["prediction_absmax"] = float(prediction.abs().max())
                state = buffer.commit(prediction, t)
                trace["dense22_mm"] = float(
                    (state.joints[..., :3, 3] - truth.joints[t, :, :3, 3]).double().norm(dim=-1).mean() * 1000
                )
                trace["reference_translation_error_m"] = float(
                    (state.reference[:3, 3] - truth.reference[t, :3, 3]).double().norm()
                )
                trace["reference_tilt"] = float(state.reference[2, :2].norm())
                row["trace"].append(trace)
            except (ValueError, RuntimeError) as error:
                trace["error"] = str(error)
                row["trace"].append(trace)
                row["failure_frame"] = t
                break
        else:
            row["completed"] = True
        errors = [point["dense22_mm"] for point in row["trace"] if "dense22_mm" in point]
        row["mean_error_mm_over_executed_frames"] = sum(errors) / len(errors) if errors else None
        row["max_error_mm_over_executed_frames"] = max(errors) if errors else None
        row["first_error_over_1m_frame"] = next(
            (point["frame"] for point in row["trace"] if point.get("dense22_mm", 0) > 1000), None
        )
        row["execution_complete_is_not_accuracy_pass"] = True
        report["closed_loop"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)

        (args.output / "report.json").write_text(json.dumps(finite_json(report), indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
