"""Postfit P01 interventions on frozen caches; no training or feedback."""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.fk import FixedShapeFK, load_smpl, smpl_identity
from egorecover.io import file_sha256, save_checkpoint, save_json
from egorecover.models import HistoryPrior
from egorecover.prior_data import predictor_forward, select_examples

FACTORS = ("root_position", "root_orientation", "articulation", "shape")
PARTS = {
    "pelvis": [0],
    "torso_head": [3, 6, 9, 12, 15],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21],
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "wrists": [20, 21],
    "feet": [10, 11],
}


def physical_mix(cv, prediction, bits):
    """Four-factor physical intervention, never a normalized-coordinate swap."""
    root, heading, articulation, shape = bits
    joints = cv.joints.clone()
    orientation = prediction if heading else cv
    pose = prediction if articulation else cv
    if heading == articulation:
        rotation = orientation.joints[..., :3, :3]
    else:
        relative = pose.joints[..., 0:1, :3, :3].transpose(-1, -2) @ pose.joints[..., :3, :3]
        rotation = orientation.joints[..., 0:1, :3, :3] @ relative
    joints[..., :3, :3] = rotation
    joints[..., 0, :3, 3] = (prediction if root else cv).joints[..., 0, :3, 3]
    auxiliary = cv.auxiliary.clone()
    auxiliary[..., -10:] = (prediction if shape else cv).auxiliary[..., -10:]
    return BodyState(joints, cv.reference, auxiliary)


def shapley(values):
    """Exact four-factor attribution including all interaction orders."""
    n = len(FACTORS)
    result = {}
    for factor in range(n):
        total = 0.0
        for bits in itertools.product((0, 1), repeat=n):
            if bits[factor]:
                continue
            after = list(bits)
            after[factor] = 1
            k = sum(bits)
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            total += weight * (values[tuple(after)] - values[bits])
        result[FACTORS[factor]] = total
    return result


def correction_energy(cv, prediction, truth):
    error, delta = cv.double() - truth.double(), prediction.double() - cv.double()
    prior = float(error.square().sum(-1).mean())
    linear = float(2 * (error * delta).sum(-1).mean())
    quadratic = float(delta.square().sum(-1).mean())
    after = float((error + delta).square().sum(-1).mean())
    dot = linear / 2
    return {
        "CV_position_MSE_mm2": prior * 1e6,
        "P_position_MSE_mm2": after * 1e6,
        "delta_MSE_mm2": (after - prior) * 1e6,
        "linear_alignment_term_mm2": linear * 1e6,
        "squared_correction_term_mm2": quadratic * 1e6,
        "position_linear_oracle_alpha": -dot / quadratic if quadratic else None,
        "correction_RMS_mm": math.sqrt(quadratic) * 1000,
        "cosine_error_correction": dot / math.sqrt(prior * quadratic) if prior * quadratic else None,
    }


def center_metrics(value, truth):
    value, truth = value.double(), truth.double()
    distance = (value - truth).norm(dim=-1)
    centroid = value.mean(-2) - truth.mean(-2)
    relative = (value - value.mean(-2, keepdim=True)) - (truth - truth.mean(-2, keepdim=True))
    return {
        "mpjpe_mm": float(distance.mean() * 1000),
        "position_MSE_mm2": float(distance.square().mean() * 1e6),
        "centroid_MSE_mm2": float(centroid.square().sum(-1).mean() * 1e6),
        "relative_MSE_mm2": float(relative.square().sum(-1).mean() * 1e6),
        "root_z_bias_mm": float((value[:, 0, 2] - truth[:, 0, 2]).mean() * 1000),
        "joint_position_p50_mm": float(distance.quantile(0.5) * 1000),
        "joint_position_p95_mm": float(distance.quantile(0.95) * 1000),
        **{f"{part}_mm": float(distance[:, indices].mean() * 1000) for part, indices in PARTS.items()},
    }


def paired_summary(rows, variant, baseline="CV", seed=1062):
    take_names = sorted({r["take"] for r in rows})
    sums = np.array(
        [
            sum(
                r["metrics"][variant]["mpjpe_mm"] - r["metrics"][baseline]["mpjpe_mm"]
                for r in rows
                if r["take"] == take
            )
            for take in take_names
        ]
    )
    counts = np.array([sum(r["take"] == take for r in rows) for take in take_names])
    draw = np.random.default_rng(seed).integers(len(take_names), size=(2000, len(take_names)))
    boot = sums[draw].sum(1) / counts[draw].sum(1)
    return {
        "delta_mm": float(sums.sum() / counts.sum()),
        "delta_ci95_mm": np.quantile(boot, [0.025, 0.975]).tolist(),
        "clips_improved": sum(r["metrics"][variant]["mpjpe_mm"] < r["metrics"][baseline]["mpjpe_mm"] for r in rows),
    }


@torch.no_grad()
def diagnose(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    bank = torch.load(args.root / "banks/D128/bank.pt", map_location="cpu", weights_only=True, mmap=True)
    codec = MotionCodec(bank["stats"]).to(args.device)
    smpl = load_smpl(args.device)
    if smpl_identity() != bank["provenance"]["producer_smpl_asset"]:
        raise ValueError("SMPL asset differs from actual bank producer.")
    fk = FixedShapeFK(smpl, torch.zeros(10, device=args.device), shape_mode="predicted")
    grouped = {}
    for index, row in enumerate(bank["rows"]):
        grouped.setdefault(row["episode"], []).append(index)
    grouped = {k: sorted(v, key=lambda i: bank["rows"][i]["frame"]) for k, v in sorted(grouped.items())}
    examples = bank["tensors"]
    baseline, truth_positions, target_positions = [], [], []
    for start in range(0, len(bank["rows"]), args.batch):
        e = select_examples(examples, list(range(start, min(start + args.batch, len(bank["rows"])))), args.device)
        baseline.append(fk(codec.decode_current(e["base"][:, 0], e["reference"])).cpu())
        target_positions.append(fk(codec.decode_current(e["target"][:, 0], e["reference"])).cpu())
        truth_positions.append(e["truth_xyz"].cpu())
    baseline, target_positions, truth = [torch.cat(x) for x in (baseline, target_positions, truth_positions)]
    checkpoints = [
        ("EMA500", args.root / "P01_W3G10k_s62/best.pt", "state_dict"),
        ("EMA2000", args.root / "P01_W3G10k_s62/last.pt", "state_dict"),
        ("raw2000", args.root / "P01_W3G10k_s62/resume.pt", "model"),
    ]
    result = {
        "scope": "Postfit D128 matching-cache diagnostic. No parameter updates, P feedback, gate revision or held-out-test claim.",
        "bank_sha256": file_sha256(args.root / "banks/D128/bank.pt"),
        "factors": list(FACTORS),
        "models": {},
        "physical_intervention_warning": "Mixed factors are counterfactual state-component attributions, not deployable methods or new trained models.",
        "alpha_warning": "Fixed alpha curves use validation labels only for diagnosis; oracle position alpha is not a learned/deployable gate.",
        "aggregation": "frames within each clip/phase, then clip equal; paired CI clusters takes",
    }
    traces = {"CV": baseline, "GT": truth, "GT_code_FK": target_positions, "rows": bank["rows"]}
    for name, path, key in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config = saved["model_config"] if "model_config" in saved else saved["identity"]["model_config"]
        model = HistoryPrior(**config).to(args.device).eval().requires_grad_(False)
        model.load_state_dict(saved[key], strict=True)
        variants = {}
        bits_list = list(itertools.product((0, 1), repeat=4)) if name == "EMA2000" else [(0, 0, 0, 0), (1, 1, 1, 1)]
        for bits in bits_list:
            variants["mix" + "".join(map(str, bits))] = []
        for alpha in (-0.25, 0.25, 0.5, 0.75):
            variants[f"alpha{alpha:g}"] = []
        normalized_correction = []
        for start in range(0, len(bank["rows"]), args.batch):
            indices = list(range(start, min(start + args.batch, len(bank["rows"]))))
            e = select_examples(examples, indices, args.device)
            prediction = predictor_forward(model, e)
            normalized_correction.append((prediction - e["base"]).cpu())
            cv = codec.decode_current(e["base"][:, 0], e["reference"])
            state = codec.decode_current(prediction[:, 0], e["reference"])
            for bits in bits_list:
                variants["mix" + "".join(map(str, bits))].append(fk(physical_mix(cv, state, bits)).cpu())
            for alpha in (-0.25, 0.25, 0.5, 0.75):
                codes = e["base"] + alpha * (prediction - e["base"])
                variants[f"alpha{alpha:g}"].append(fk(codec.decode_current(codes[:, 0], e["reference"])).cpu())
        variants = {k: torch.cat(v) for k, v in variants.items()}
        torch.testing.assert_close(variants["mix0000"], baseline, rtol=0, atol=0)
        variants["P"] = variants["mix1111"]
        variants["CV"] = baseline
        variants["GT_code_FK"] = target_positions
        correction = torch.cat(normalized_correction)
        rows = []
        for episode, indices in grouped.items():
            frames = torch.tensor([bank["rows"][i]["frame"] for i in indices])
            for phase, mask in [("all", frames >= 20), ("mature", frames >= 40), ("startup", frames < 40)]:
                selected = torch.tensor(indices)[mask]
                actual = truth[selected]
                metrics = {k: center_metrics(v[selected], actual) for k, v in variants.items()}
                vals = {bits: metrics["mix" + "".join(map(str, bits))]["mpjpe_mm"] for bits in bits_list}
                row = {
                    "episode": episode,
                    "take": bank["rows"][indices[0]]["take"],
                    "phase": phase,
                    "metrics": metrics,
                    "correction": correction_energy(baseline[selected], variants["P"][selected], actual),
                    "normalized_code_correction_RMS": float(correction[selected].square().mean().sqrt()),
                }
                if len(bits_list) == 16:
                    row["shapley_MPJPE_mm"] = shapley(vals)
                    assert abs(sum(row["shapley_MPJPE_mm"].values()) - (vals[(1, 1, 1, 1)] - vals[(0, 0, 0, 0)])) < 1e-8
                rows.append(row)
        summary = {}
        for phase in ("all", "mature", "startup"):
            selected = [x for x in rows if x["phase"] == phase]

            def mean(seq):
                return sum(seq) / len(seq)

            summary[phase] = {
                "metrics": {
                    variant: {
                        metric: mean([row["metrics"][variant][metric] for row in selected])
                        for metric in selected[0]["metrics"][variant]
                    }
                    for variant in variants
                },
                "paired": {variant: paired_summary(selected, variant) for variant in variants if variant != "CV"},
                "correction": {
                    metric: mean(
                        [row["correction"][metric] for row in selected if row["correction"][metric] is not None]
                    )
                    for metric in selected[0]["correction"]
                },
                "normalized_code_correction_RMS": mean([row["normalized_code_correction_RMS"] for row in selected]),
            }
            if len(bits_list) == 16:
                summary[phase]["shapley_MPJPE_mm"] = {
                    factor: mean([row["shapley_MPJPE_mm"][factor] for row in selected]) for factor in FACTORS
                }
            energy = summary[phase]["correction"]
            for key in ("position_linear_oracle_alpha", "cosine_error_correction", "correction_RMS_mm"):
                energy["mean_per_clip_" + key] = energy.pop(key)
            energy["clip_equal_MSE_single_XYZ_alpha"] = (
                -0.5 * energy["linear_alignment_term_mm2"] / energy["squared_correction_term_mm2"]
                if energy["squared_correction_term_mm2"]
                else None
            )
            energy["alpha_scope"] = "XYZ interpolation postfit oracle, not normalized-code alpha and not deployable."
        result["models"][name] = {
            "step": saved["step"],
            "checkpoint_sha256": file_sha256(path),
            "summary": summary,
            "rows": rows,
        }
        traces[name] = variants["P"]
        save_json(args.output, result)
        print(
            json.dumps(
                {
                    "model": name,
                    "mature_P": summary["mature"]["metrics"]["P"]["mpjpe_mm"],
                    "mature_CV": summary["mature"]["metrics"]["CV"]["mpjpe_mm"],
                    "mature_shapley": summary["mature"].get("shapley_MPJPE_mm"),
                    "mature_energy": summary["mature"]["correction"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del model, saved
    save_checkpoint(args.output.with_name("correction_positions.pt"), traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("exp/formal-p-w3-g10k-20261113"))
    parser.add_argument("--output", type=Path, default=Path("exp/formal-p-w3-g10k-20261113/diagnosis/corrections.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
