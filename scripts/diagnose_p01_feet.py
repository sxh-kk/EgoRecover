"""Post-inference foot-height/slide diagnostics for frozen P01 and matched CV."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

FEET = (7, 10, 8, 11)
FOOT_NAMES = ("left_ankle", "left_toe", "right_ankle", "right_toe")
THRESHOLDS_M = (0.08, 0.04, 0.08, 0.04)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def contact_weights(feet):
    """Exactly the current UniEgoMotion slide metric's height soft weights."""
    height = feet[:-1, :, 2]
    threshold = height.new_tensor(THRESHOLDS_M)
    contact = height < threshold
    return torch.where(contact, 2 - torch.exp2(torch.where(contact, height / threshold, 0)), 0)


def weighted_slide(weights, displacement, frames):
    return float((weights * displacement).abs().sum(0).mean() * (1000 / frames))


def foot_statistics(joints, truth, timestamps, auxiliary_contact, target_contact):
    joints, truth, timestamps = (value.detach().cpu().double() for value in (joints, truth, timestamps))
    feet, target = joints[:, FEET], truth[:, FEET]
    height, target_height = feet[..., 2], target[..., 2]
    horizontal = torch.diff(feet[..., :2], dim=0)
    target_horizontal = torch.diff(target[..., :2], dim=0)
    dt = torch.diff(timestamps)
    if len(joints) < 3 or not bool((dt > 0).all()):
        raise ValueError("A diagnostic phase needs three consecutive frames.")
    if not torch.allclose(dt, dt.mean().expand_as(dt), rtol=2e-5, atol=1e-7):
        raise ValueError("Diagnostic temporal derivatives require uniform timestamps.")
    displacement = horizontal.norm(dim=-1)
    relative_height_error = (height - joints[:, 0, 2:3]) - (target_height - truth[:, 0, 2:3])
    result = {
        "official_slide_mm": weighted_slide(contact_weights(feet), displacement, len(joints)),
        "xy_step_mean_mm": float(displacement.mean() * 1000),
        "xy_velocity_magnitude_mm_s": float((horizontal / dt[:, None, None]).norm(dim=-1).mean() * 1000),
        "xy_velocity_error_mm_s": float(
            ((horizontal - target_horizontal) / dt[:, None, None]).norm(dim=-1).mean() * 1000
        ),
        "xy_acceleration_magnitude_mm_s2": float(
            (torch.diff(feet[..., :2], n=2, dim=0) / dt.mean().square()).norm(dim=-1).mean() * 1000
        ),
        "xy_acceleration_error_mm_s2": float(
            (torch.diff(feet[..., :2] - target[..., :2], n=2, dim=0) / dt.mean().square()).norm(dim=-1).mean() * 1000
        ),
        "foot_z_mean_mm": float(height.mean() * 1000),
        "foot_z_error_bias_mm": float((height - target_height).mean() * 1000),
        "foot_z_error_MAE_mm": float((height - target_height).abs().mean() * 1000),
        "root_z_mean_mm": float(joints[:, 0, 2].mean() * 1000),
        "root_z_error_bias_mm": float((joints[:, 0, 2] - truth[:, 0, 2]).mean() * 1000),
        "root_z_error_MAE_mm": float((joints[:, 0, 2] - truth[:, 0, 2]).abs().mean() * 1000),
        "root_relative_foot_z_mean_mm": float((height - joints[:, 0, 2:3]).mean() * 1000),
        "root_relative_foot_z_error_MAE_mm": float(relative_height_error.abs().mean() * 1000),
        "penetration_fraction": float((height < 0).double().mean()),
        "penetration_unconditional_depth_mm": float((-height).clamp_min(0).mean() * 1000),
        "below_metric_threshold_fraction": float((height < height.new_tensor(THRESHOLDS_M)).double().mean()),
        "transition_mean_height_weight": float(contact_weights(feet).mean()),
        "foot_position_error_mm": float((feet - target).norm(dim=-1).mean() * 1000),
    }
    gt_contact = target_height < target_height.new_tensor(THRESHOLDS_M)
    predicted_contact = height < height.new_tensor(THRESHOLDS_M)
    for suffix, mask in (("GT_height_contact", gt_contact), ("GT_height_noncontact", ~gt_contact)):
        result[f"selected_fraction_{suffix}"] = float(mask.double().mean())
        for name, values in (
            ("foot_z_mean_mm", height * 1000),
            ("foot_z_error_bias_mm", (height - target_height) * 1000),
            ("foot_z_error_MAE_mm", (height - target_height).abs() * 1000),
            ("penetration_fraction", (height < 0).double()),
            ("predicted_height_contact_fraction", predicted_contact.double()),
        ):
            result[f"{name}_on_{suffix}"] = float(values[mask].mean()) if bool(mask.any()) else None
    for name, index in zip(FOOT_NAMES, range(4)):
        result[f"{name}_z_mean_mm"] = float(height[:, index].mean() * 1000)
        result[f"{name}_z_MAE_mm"] = float((height[:, index] - target_height[:, index]).abs().mean() * 1000)
        result[f"{name}_penetration_fraction"] = float((height[:, index] < 0).double().mean())
        result[f"{name}_below_metric_threshold_fraction"] = float(
            (height[:, index] < THRESHOLDS_M[index]).double().mean()
        )
        result[f"{name}_xy_step_mean_mm"] = float(displacement[:, index].mean() * 1000)
        result[f"{name}_mean_height_weight"] = float(contact_weights(feet)[:, index].mean())
    for threshold in (0.01, 0.02, 0.04, 0.08, 0.10):
        result[f"foot_fraction_z_below_{threshold:g}m"] = float((height < threshold).double().mean())
    if auxiliary_contact is not None:
        auxiliary_contact, target_contact = [
            value.detach().cpu().double() for value in (auxiliary_contact, target_contact)
        ]
        result.update(
            auxiliary_contact_mean=float(auxiliary_contact.mean()),
            auxiliary_left_contact_mean=float(auxiliary_contact[:, 0].mean()),
            auxiliary_right_contact_mean=float(auxiliary_contact[:, 1].mean()),
            auxiliary_contact_outside_0_1_fraction=float(
                ((auxiliary_contact < 0) | (auxiliary_contact > 1)).double().mean()
            ),
            auxiliary_contact_MAE_vs_GT=float((auxiliary_contact - target_contact).abs().mean()),
            auxiliary_contact_threshold05_agreement=float(
                ((auxiliary_contact >= 0.5) == (target_contact >= 0.5)).double().mean()
            ),
        )
    return result


def cross_diagnostic(cv, prediction, truth, target_contact):
    cv, prediction, truth = (value.detach().cpu().double() for value in (cv, prediction, truth))
    cv_feet, p_feet, gt_feet = [value[:, FEET] for value in (cv, prediction, truth)]
    wcv, wp, wgt = [contact_weights(value) for value in (cv_feet, p_feet, gt_feet)]
    dcv, dp, dgt = [torch.diff(value[..., :2], dim=0).norm(dim=-1) for value in (cv_feet, p_feet, gt_feet)]
    frames = len(cv)
    values = {
        "wCV_dCV_mm": weighted_slide(wcv, dcv, frames),
        "wCV_dP_mm": weighted_slide(wcv, dp, frames),
        "wP_dCV_mm": weighted_slide(wp, dcv, frames),
        "wP_dP_mm": weighted_slide(wp, dp, frames),
        "wGT_dCV_mm": weighted_slide(wgt, dcv, frames),
        "wGT_dP_mm": weighted_slide(wgt, dp, frames),
        "wGT_dGT_mm": weighted_slide(wgt, dgt, frames),
    }
    if target_contact is not None:
        # Official auxiliary contact labels are left/right toe contacts. Hold
        # each side's label fixed for its ankle and toe, never infer from P.
        wgta = target_contact.detach().cpu().double()[:-1, [0, 0, 1, 1]]
        values.update(
            wGTaux_dCV_mm=weighted_slide(wgta, dcv, frames),
            wGTaux_dP_mm=weighted_slide(wgta, dp, frames),
            wGTaux_dGT_mm=weighted_slide(wgta, dgt, frames),
        )
    delta = values["wP_dP_mm"] - values["wCV_dCV_mm"]
    values.update(
        total_slide_delta_mm=delta,
        displacement_at_fixed_CV_weight_delta_mm=values["wCV_dP_mm"] - values["wCV_dCV_mm"],
        weight_at_fixed_CV_displacement_delta_mm=values["wP_dCV_mm"] - values["wCV_dCV_mm"],
        interaction_mm=values["wP_dP_mm"] - values["wCV_dP_mm"] - values["wP_dCV_mm"] + values["wCV_dCV_mm"],
        displacement_Shapley_mm=0.5
        * (values["wCV_dP_mm"] - values["wCV_dCV_mm"] + values["wP_dP_mm"] - values["wP_dCV_mm"]),
        height_weight_Shapley_mm=0.5
        * (values["wP_dCV_mm"] - values["wCV_dCV_mm"] + values["wP_dP_mm"] - values["wCV_dP_mm"]),
        fixed_GT_height_weight_delta_mm=values["wGT_dP_mm"] - values["wGT_dCV_mm"],
    )
    if target_contact is not None:
        values["fixed_GT_auxiliary_contact_delta_mm"] = values["wGTaux_dP_mm"] - values["wGTaux_dCV_mm"]
    if abs(values["displacement_Shapley_mm"] + values["height_weight_Shapley_mm"] - delta) > 1e-10:
        raise AssertionError("The two-factor symmetric decomposition must exactly sum to the metric difference.")
    return values


def diagnose_clip(metadata, cv, prediction, truth, timestamps, cv_contacts=None, p_contacts=None, gt_contacts=None):
    frames = torch.tensor([row["frame"] for row in metadata])
    if not bool((torch.diff(frames) == 1).all()):
        raise ValueError("Foot diagnostics require consecutive, nonduplicated frames.")
    rows = []
    for phase, selected in (
        ("all", frames >= 20),
        ("startup", (frames >= 20) & (frames < 40)),
        ("mature", frames >= 40),
    ):
        if int(selected.sum()) < 3:
            continue
        values = {"CV": cv[selected], "P": prediction[selected], "GT": truth[selected]}
        times = timestamps[selected]
        contacts = {
            "CV": cv_contacts[selected] if cv_contacts is not None else None,
            "P": p_contacts[selected] if p_contacts is not None else None,
            "GT": gt_contacts[selected] if gt_contacts is not None else None,
        }
        row = {
            "episode": metadata[0]["episode"],
            "take": metadata[0]["take"],
            "phase": phase,
            "frames": int(selected.sum()),
            "floor_calibrated": metadata[0]["floor_calibrated"],
            "CV": foot_statistics(values["CV"], values["GT"], times, contacts["CV"], contacts["GT"]),
            "P": foot_statistics(values["P"], values["GT"], times, contacts["P"], contacts["GT"]),
            "GT": foot_statistics(values["GT"], values["GT"], times, contacts["GT"], contacts["GT"]),
            "cross": cross_diagnostic(values["CV"], values["P"], values["GT"], contacts["GT"]),
        }
        # Unknown-floor episodes retain XY/relative-height diagnostics, but no
        # calibrated absolute-height interpretation or official-slide effect.
        rows.append(row)
    return rows


def paired_take_bootstrap(rows, field, samples=2000, seed=1062):
    selected = [(row["take"], float(field(row))) for row in rows]
    grouped = defaultdict(list)
    for take, value in selected:
        grouped[take].append(value)
    sums = np.asarray([sum(grouped[take]) for take in sorted(grouped)])
    counts = np.asarray([len(grouped[take]) for take in sorted(grouped)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(sums), size=(samples, len(sums)))
    bootstrap = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "delta": float(sums.sum() / counts.sum()),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "clips": len(rows),
        "takes": len(grouped),
        "positive_clip_fraction": sum(value > 0 for _, value in selected) / len(selected),
    }


def summarize(rows):
    result = {}
    for phase in ("all", "startup", "mature"):
        selected = [row for row in rows if row["phase"] == phase and row["floor_calibrated"]]
        if not selected:
            result[phase] = None
            continue
        means = {
            name: {
                key: (
                    float(np.mean([row[name][key] for row in selected if row[name][key] is not None]))
                    if any(row[name][key] is not None for row in selected)
                    else None
                )
                for key in selected[0][name]
            }
            for name in ("P", "CV", "GT", "cross")
        }
        effects = {
            key: paired_take_bootstrap(selected, lambda row, key=key: row["cross"][key])
            for key in (
                "total_slide_delta_mm",
                "displacement_Shapley_mm",
                "height_weight_Shapley_mm",
                "fixed_GT_height_weight_delta_mm",
                "fixed_GT_auxiliary_contact_delta_mm",
                "displacement_at_fixed_CV_weight_delta_mm",
                "weight_at_fixed_CV_displacement_delta_mm",
                "interaction_mm",
            )
            if key in selected[0]["cross"]
        }
        for key in (
            "xy_step_mean_mm",
            "xy_velocity_error_mm_s",
            "xy_acceleration_error_mm_s2",
            "foot_z_mean_mm",
            "root_z_mean_mm",
            "root_relative_foot_z_mean_mm",
            "foot_z_error_MAE_mm",
            "root_z_error_MAE_mm",
            "root_relative_foot_z_error_MAE_mm",
            "penetration_fraction",
            "penetration_unconditional_depth_mm",
            "below_metric_threshold_fraction",
            "transition_mean_height_weight",
        ):
            effects["P_minus_CV_" + key] = paired_take_bootstrap(
                selected, lambda row, key=key: row["P"][key] - row["CV"][key]
            )
        for suffix in ("GT_height_contact", "GT_height_noncontact"):
            for name in (
                "foot_z_error_MAE_mm",
                "foot_z_error_bias_mm",
                "penetration_fraction",
                "predicted_height_contact_fraction",
            ):
                key = f"{name}_on_{suffix}"
                available = [row for row in selected if row["P"][key] is not None and row["CV"][key] is not None]
                if available:
                    effects["P_minus_CV_" + key] = paired_take_bootstrap(
                        available, lambda row, key=key: row["P"][key] - row["CV"][key]
                    )
        result[phase] = {
            "clips": len(selected),
            "takes": len({row["take"] for row in selected}),
            "frames": sum(row["frames"] for row in selected),
            "means": means,
            "metric_clip_counts": {
                name: {key: sum(row[name][key] is not None for row in selected) for key in selected[0][name]}
                for name in ("P", "CV", "GT", "cross")
            },
            "paired_effects": effects,
        }
    return result


@torch.no_grad()
def main(args):
    # Import the producer-compatible P/FK implementation after placing the
    # immutable experiment snapshot first on sys.path.
    sys.path.insert(0, str(args.source_snapshot.resolve()))
    from egorecover.codec import MotionCodec
    from egorecover.fk import FixedShapeFK, load_smpl
    from egorecover.io import save_checkpoint, save_json
    from egorecover.models import HistoryPrior
    from egorecover.prior_data import predictor_forward, select_examples

    torch.set_num_threads(2)
    device = torch.device(args.device)
    bank = torch.load(args.bank, map_location="cpu", weights_only=True, mmap=True)
    if bank["cohort"] != "D128" or bank["limited_preflight"] or len(bank["rows"]) != 7680:
        raise ValueError("This diagnosis requires the unchanged full D128 bank.")
    codec = MotionCodec(bank["stats"]).to(device)
    fk = FixedShapeFK(load_smpl(device), torch.zeros(10, device=device), shape_mode="predicted")
    exports = {
        "P_EMA500": torch.load(args.run_directory / "best.pt", map_location="cpu", weights_only=True, mmap=True),
        "P_EMA2000": torch.load(args.run_directory / "last.pt", map_location="cpu", weights_only=True, mmap=True),
    }
    raw = torch.load(args.run_directory / "resume.pt", map_location="cpu", weights_only=True, mmap=True)
    if exports["P_EMA500"]["step"] != 500 or exports["P_EMA2000"]["step"] != 2000 or raw["step"] != 2000:
        raise ValueError("The diagnosis version assumes best EMA500 and last/raw2000.")
    model = HistoryPrior(**exports["P_EMA500"]["model_config"]).to(device).eval()
    groups = defaultdict(list)
    for index, row in enumerate(bank["rows"]):
        groups[row["episode"]].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: bank["rows"][index]["frame"])
    groups = sorted(groups.items())
    arrays = {"CV": [], "GT": [], **{name: [] for name in (*exports, "P_raw2000")}}
    contact_arrays = {name: [] for name in arrays}
    # Preserve the same per-clip32 batching used for native D128 validation.
    for _, indices in groups:
        cv, gt, cvc, gtc = [], [], [], []
        for offset in range(0, len(indices), 32):
            example = select_examples(bank["tensors"], indices[offset : offset + 32], device)
            state = codec.decode_current(example["base"][:, 0], example["reference"])
            target = codec.decode_current(example["target"][:, 0], example["reference"])
            cv.append(fk(state).cpu())
            gt.append(example["truth_xyz"].cpu())
            cvc.append(state.auxiliary[:, 24:26].cpu())
            gtc.append(target.auxiliary[:, 24:26].cpu())
        arrays["CV"].append(torch.cat(cv))
        arrays["GT"].append(torch.cat(gt))
        contact_arrays["CV"].append(torch.cat(cvc))
        contact_arrays["GT"].append(torch.cat(gtc))
    states = {name: value["state_dict"] for name, value in exports.items()}
    states["P_raw2000"] = raw["model"]
    checkpoint_files = {
        "P_EMA500": args.run_directory / "best.pt",
        "P_EMA2000": args.run_directory / "last.pt",
        "P_raw2000": args.run_directory / "resume.pt",
    }
    report = {
        "scope": "post-inference mechanism diagnosis on identical frozen G histories, no training, no GT input to P, no state writes",
        "bank_sha256": sha256(args.bank),
        "bank_selection_sha256": bank["selection_sha256"],
        "source_snapshot": str(args.source_snapshot),
        "metric": {
            "feet": list(FEET),
            "thresholds_m": list(THRESHOLDS_M),
            "height_used": "initial frame of each in-phase forward difference",
            "normalization": "1000 * sum over T-1 transitions, mean4 feet, divided by T frames",
            "soft_weight": "if z<threshold: 2-2^(z/threshold), else0; negative z increases weight up to2",
            "auxiliary_contact": "two raw decoded channels are diagnostic only, not used in official metric",
            "penetration_fields": "joint z<0 proxies, not mesh-surface penetration; fitted GT joints also have negative z",
            "GT_auxiliary_contact": "official decoded left/right toe slow-and-low labels at transition start, copied to side ankle/toe",
            "decomposition": "exact2x2 and symmetric two-factor Shapley, algebraic attribution, not causal attribution of network channels",
        },
        "aggregation": "within clip phase, then equal clips; paired take-cluster bootstrap2000 seed1062; fixed-checkpoint development diagnostic, not an independent test",
        "models": {},
    }
    report["implementation"] = {
        "diagnostic_script_sha256": sha256(__file__),
        "codec_file": str(Path(sys.modules[MotionCodec.__module__].__file__).resolve()),
        "fk_file": str(Path(sys.modules[FixedShapeFK.__module__].__file__).resolve()),
    }
    for name, state in states.items():
        model.load_state_dict(state, strict=True)
        all_rows = []
        for ordinal, (episode, indices) in enumerate(groups):
            position, contacts = [], []
            for offset in range(0, len(indices), 32):
                example = select_examples(bank["tensors"], indices[offset : offset + 32], device)
                prediction = predictor_forward(model, example)
                body = codec.decode_current(prediction[:, 0], example["reference"])
                position.append(fk(body).cpu())
                contacts.append(body.auxiliary[:, 24:26].cpu())
            position, contacts = torch.cat(position), torch.cat(contacts)
            arrays[name].append(position)
            contact_arrays[name].append(contacts)
            metadata = [bank["rows"][index] for index in indices]
            timestamps = torch.tensor([row["timestamp_s"] for row in metadata], dtype=torch.float64)
            all_rows += diagnose_clip(
                metadata,
                arrays["CV"][ordinal],
                position,
                arrays["GT"][ordinal],
                timestamps,
                contact_arrays["CV"][ordinal],
                contacts,
                contact_arrays["GT"][ordinal],
            )
        report["models"][name] = {
            "checkpoint_sha256": sha256(checkpoint_files[name]),
            "weight": "raw2000" if name == "P_raw2000" else "EMA",
            "step": 500 if name == "P_EMA500" else 2000,
            "summary": summarize(all_rows),
            "rows": all_rows,
        }
        if name != "P_raw2000":
            native = json.loads((args.run_directory / f"validation-step{exports[name]['step']}.json").read_text())
            maximum = max(
                abs(
                    report["models"][name]["summary"][phase]["means"][source]["official_slide_mm"]
                    - native["summary"][phase]["metrics"]["foot_slide_mm"][source]
                )
                for phase in ("all", "startup", "mature")
                for source in ("CV", "P")
            )
            report["models"][name]["native_validation_foot_slide_max_abs_error_mm"] = maximum
            if maximum > 1e-4:
                raise AssertionError("Foot diagnostics do not reproduce native validation within numerical precision.")
        save_json(
            args.output / "progress.json",
            {"phase": "diagnosing", "completed_models": len(report["models"]), "model": name},
        )
    if args.save_positions:
        ordered_rows = [bank["rows"][index] for _, indices in groups for index in indices]
        save_checkpoint(
            args.output / "positions.pt",
            {
                "format": "P01_feet_diagnostic_positions_v1",
                "row_order": ordered_rows,
                "positions_floor_world_m": {name: torch.cat(values) for name, values in arrays.items()},
                "auxiliary_contacts_raw": {name: torch.cat(values) for name, values in contact_arrays.items()},
                "bank_sha256": report["bank_sha256"],
                "schema": "all tensors first dimension matches row_order; positions [7680,22,3] float32 meters; contacts [7680,2] raw decoded left/right toe aux; rows sorted by episode then local frame, not original bank order",
            },
        )
    save_json(args.output / "feet.json", report)
    save_json(
        args.output / "progress.json", {"phase": "complete", "models": list(report["models"]), "clips": len(groups)}
    )
    print(
        json.dumps(
            {
                name: {
                    phase: {
                        "CV_slide_mm": value["means"]["CV"]["official_slide_mm"],
                        "P_slide_mm": value["means"]["P"]["official_slide_mm"],
                        "motion_Sh_mm": value["means"]["cross"]["displacement_Shapley_mm"],
                        "height_Sh_mm": value["means"]["cross"]["height_weight_Shapley_mm"],
                        "GT_fixed_delta_mm": value["means"]["cross"]["fixed_GT_height_weight_delta_mm"],
                    }
                    for phase, value in model_report["summary"].items()
                }
                for name, model_report in report["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("/root/workspace/songxuhua/exp/formal-p-w3-g10k-20261113")
    parser.add_argument("--bank", type=Path, default=root / "banks/D128/bank.pt")
    parser.add_argument("--source-snapshot", type=Path, default=root / "source")
    parser.add_argument("--run-directory", type=Path, default=root / "P01_W3G10k_s62")
    parser.add_argument("--output", type=Path, default=root / "diagnosis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-positions", action="store_true")
    main(parser.parse_args())
