"""Read-only P01 phase/target conflict diagnosis on fixed real history banks."""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from egorecover.codec import MotionCodec
from egorecover.fk import FixedShapeFK, load_smpl, smpl_identity
from egorecover.formal_prior_training import FormalPriorSampler
from egorecover.io import code_identity, file_sha256, save_json, state_sha256
from egorecover.models import HistoryPrior
from egorecover.prior_data import predictor_forward, select_examples

GROUPS = {
    "body_rotation6d": tuple(9 * joint + channel for joint in range(22) for channel in range(6)),
    "root_xyz": (6, 7, 8),
    "other_dense_xyz": tuple(9 * joint + channel for joint in range(1, 22) for channel in range(6, 9)),
    "reference9": tuple(range(198, 207)),
    "hand_PCA24": tuple(range(207, 231)),
    "contact2": (231, 232),
    "beta10": tuple(range(233, 243)),
}
PHASE_WEIGHTS = {"mature": 0.75, "startup": 0.25}


def sample_train_indices(bank, *, samples=1024, seed=9063):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 8 or samples % 8:
        raise ValueError("Sample count must be a positive multiple of8.")
    sampler = FormalPriorSampler(bank["rows"])
    generator = torch.Generator().manual_seed(seed)
    return [index for _ in range(samples // 8) for index in sampler.sample(generator, 8)]


def split_phases(bank, indices):
    return {
        phase: [index for index in indices if (bank["rows"][index]["frame"] >= 40) == (phase == "mature")]
        for phase in PHASE_WEIGHTS
    }


def code_vectors(prediction, target):
    error = (prediction[:, 0] - target[:, 0]).square()
    weights = error.new_ones(243)
    weights[198:207] = 8
    result = {"code_loss": (error * weights).sum(-1) / weights.sum()}
    for name, indices in GROUPS.items():
        mse = error[:, indices].mean(-1)
        result[f"code_mse/{name}"] = mse
        result[f"code_contribution/{name}"] = mse * (len(indices) * (8 if name == "reference9" else 1) / 306)
    return result


def paired_loss_vectors(prediction, example, codec, fk, geometry_weight):
    result = {}
    for name, codes in (("P", prediction), ("CV", example["base"])):
        values = code_vectors(codes, example["target"])
        points = fk(codec.decode_current(codes[:, 0], example["reference"]))
        squared = (points - example["truth_xyz"]).square()
        values["joint_squared_error_m2"] = squared.sum(-1).mean(-1)
        values["coordinate_mse_m2"] = squared.mean((-2, -1))
        values["geometry_loss"] = values["joint_squared_error_m2"] / 0.1**2
        values["mpjpe_mm"] = squared.sum(-1).sqrt().mean(-1) * 1000
        values["total_loss"] = values["code_loss"] + geometry_weight * values["geometry_loss"]
        result[name] = values
    return result


def metric_rows(bank, indices, vectors):
    detached = {
        name: {key: value.detach().double().cpu().tolist() for key, value in data.items()}
        for name, data in vectors.items()
    }
    return [
        {
            "index": index,
            "episode": bank["rows"][index]["episode"],
            "take": bank["rows"][index]["take"],
            "source": bank["rows"][index].get("source", bank["rows"][index]["episode"]),
            "frame": bank["rows"][index]["frame"],
            **{name: {key: values[position] for key, values in data.items()} for name, data in detached.items()},
        }
        for position, index in enumerate(indices)
    ]


def _flatten(gradients, parameters):
    return torch.cat(
        [
            (torch.zeros_like(p) if gradient is None else gradient).reshape(-1).double()
            for p, gradient in zip(parameters, gradients)
        ]
    )


def phase_gradient_audit(model, bank, indices, codec, fk, geometry_weight, *, device="cpu", microbatch=8):
    """Mean parameter gradients of phase losses, with eval-mode dropout disabled."""
    if not indices:
        raise ValueError("A phase gradient requires at least one sample.")
    model.eval()
    model.zero_grad(set_to_none=True)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    count = sum(p.numel() for p in parameters)
    gradients = {name: torch.zeros(count, dtype=torch.float64, device=device) for name in ("code", "geometry")}
    output_means = {name: torch.zeros(243, dtype=torch.float64, device=device) for name in gradients}
    output_norm_sums = {name: 0.0 for name in gradients}
    rows = []
    for start in range(0, len(indices), microbatch):
        selected = indices[start : start + microbatch]
        example = select_examples(bank["tensors"], selected, device)
        prediction = predictor_forward(model, example)
        vectors = paired_loss_vectors(prediction, example, codec, fk, geometry_weight)
        fraction = len(selected) / len(indices)
        for name in gradients:
            values = torch.autograd.grad(
                vectors["P"][f"{name}_loss"].mean(),
                (*parameters, prediction),
                retain_graph=name == "code",
                allow_unused=True,
            )
            gradients[name].add_(_flatten(values[:-1], parameters), alpha=fraction)
            # Loss.mean differentiates each output with1/B. Undo that factor
            # before averaging per-example norms; sum output rows for the mean.
            output_gradient = values[-1].reshape(len(selected), 243).detach().double()
            output_means[name].add_(output_gradient.sum(0), alpha=fraction)
            output_norm_sums[name] += float((output_gradient * len(selected)).norm(dim=-1).sum()) / len(indices)
        rows.extend(metric_rows(bank, selected, vectors))
        del vectors, prediction, example, values
    if any(not bool(torch.isfinite(value).all()) for value in gradients.values()):
        raise ValueError("Nonfinite diagnostic parameter gradient.")
    if any(p.grad is not None for p in parameters):
        raise AssertionError("Read-only autograd.grad unexpectedly accumulated optimizer gradients.")
    return {"gradients": gradients, "output_means": output_means, "output_norm_means": output_norm_sums, "rows": rows}


@torch.no_grad()
def inference_rows(model, bank, indices, codec, fk, geometry_weight, *, device="cpu", batch_size=64):
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        example = select_examples(bank["tensors"], selected, device)
        vectors = paired_loss_vectors(predictor_forward(model, example), example, codec, fk, geometry_weight)
        rows.extend(metric_rows(bank, selected, vectors))
    return rows


def _norm(value):
    return float(value.norm())


def _cosine(first, second):
    denominator = _norm(first) * _norm(second)
    return float(torch.dot(first, second)) / denominator if denominator else None


def gradient_relationships(audits, geometry_weight):
    vectors = {}
    phases = {}
    for phase, audit in audits.items():
        code, geometry = audit["gradients"]["code"], audit["gradients"]["geometry"]
        total = code + geometry_weight * geometry
        vectors[phase] = {"code": code, "geometry": geometry, "weighted_total": total}
        phases[phase] = {
            "samples": len(audit["rows"]),
            "norms": {
                "code": _norm(code),
                "geometry": _norm(geometry),
                "weighted_geometry": geometry_weight * _norm(geometry),
                "weighted_total": _norm(total),
            },
            "code_geometry_cosine": _cosine(code, geometry),
            "weighted_geometry_to_code_norm": geometry_weight * _norm(geometry) / _norm(code) if _norm(code) else None,
            "output_gradients": {
                name: {
                    "norm_of_mean": _norm(value),
                    "mean_per_example_norm": audit["output_norm_means"][name],
                    "group_norm_of_mean": {group: _norm(value[list(indices)]) for group, indices in GROUPS.items()},
                    "mean_vector": value.cpu().tolist(),
                }
                for name, value in audit["output_means"].items()
            },
        }
    relationship = {}
    for name in ("code", "geometry", "weighted_total"):
        mature, startup = vectors["mature"][name], vectors["startup"][name]
        weighted_mature, weighted_startup = 0.75 * mature, 0.25 * startup
        mixed = weighted_mature + weighted_startup
        norm_mature, norm_startup = _norm(weighted_mature), _norm(weighted_startup)
        denominator = norm_mature + norm_startup
        relationship[name] = {
            "mature_norm": _norm(mature),
            "startup_norm": _norm(startup),
            "weighted_mature_norm": norm_mature,
            "weighted_startup_norm": norm_startup,
            "weighted_startup_over_mature_norm": norm_startup / norm_mature if norm_mature else None,
            "phase_cosine": _cosine(mature, startup),
            "sum_norm": _norm(mixed),
            "cos_sum_mature": _cosine(mixed, mature),
            "cos_sum_startup": _cosine(mixed, startup),
            "sum_opposes_mature": bool(torch.dot(mixed, mature) < 0),
            "sum_opposes_startup": bool(torch.dot(mixed, startup) < 0),
            "cancellation_ratio": _norm(mixed) / denominator if denominator else None,
        }
    components = {
        f"{phase}_{term}": value
        for phase, terms in vectors.items()
        for term, value in terms.items()
        if term != "weighted_total"
    }
    mixed_total = sum(PHASE_WEIGHTS[phase] * terms["weighted_total"] for phase, terms in vectors.items())
    directional = {}
    for name, value in components.items():
        dot = float(torch.dot(mixed_total, value))
        directional[name] = {
            "cosine_with_mixed_total": _cosine(mixed_total, value),
            "dot_with_mixed_total": dot,
            "first_order_loss_change_per_unit_lr_under_negative_mixed_gradient": -dot,
            "mixed_descent_increases_this_loss_locally": dot < 0,
        }
    return {
        "phase": phases,
        "quota_weighted_relationships": relationship,
        "component_cosines": {
            name: {other: _cosine(value, second) for other, second in components.items()}
            for name, value in components.items()
        },
        "mixed_objective_direction": directional,
        "directional_scope": "First-order unpreconditioned gradient descent on eval-mode mean losses; not an AdamW update or a guarantee about finite-step MPJPE.",
    }


def aggregate_metrics(rows, *, clip_equal=False):
    result = {}
    for phase in ("all", "mature", "startup"):
        selected = [row for row in rows if phase == "all" or (row["frame"] >= 40) == (phase == "mature")]
        if not selected:
            result[phase] = None
            continue
        names = sorted(selected[0]["P"])
        grouped = defaultdict(list)
        for position, row in enumerate(selected):
            grouped[row["episode"] if clip_equal else position].append(row)
        means = {
            source: {
                name: math.fsum(
                    math.fsum(row[source][name] for row in group) / len(group) for group in grouped.values()
                )
                / len(grouped)
                for name in names
            }
            for source in ("P", "CV")
        }
        result[phase] = {
            "samples": len(selected),
            "unique_anchors": len({row["index"] for row in selected}),
            "episodes": len({row["episode"] for row in selected}),
            "takes": len({row["take"] for row in selected}),
            "sources": len({row["source"] for row in selected}),
            "aggregation": "clip_equal" if clip_equal else "sample_row_equal",
            "P": means["P"],
            "CV": means["CV"],
            "delta_P_minus_CV": {name: means["P"][name] - means["CV"][name] for name in names},
        }
    if result["mature"] is not None and result["startup"] is not None:
        result["standardized_75_25"] = {
            source: {
                name: sum(weight * result[phase][source][name] for phase, weight in PHASE_WEIGHTS.items())
                for name in result["mature"][source]
            }
            for source in ("P", "CV", "delta_P_minus_CV")
        }
    return result


def _bank(path):
    bank = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if bank.get("format") != "egorecover_prior_bank_v2" or bank.get("producer_window") != 3:
        raise ValueError("Diagnosis requires the frozen W3 v2 bank.")
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("exp/formal-p-w3-g10k-20261113"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=9063)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.microbatch < 1 or args.batch_size < 1:
        parser.error("Batch sizes must be positive.")
    if args.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        parser.error("This diagnosis is allocated physical GPU1; set CUDA_VISIBLE_DEVICES=1.")
    torch.set_num_threads(2)
    root = args.root.resolve()
    run = root / "P01_W3G10k_s62"
    identity = json.loads((run / "identity.json").read_text())
    if code_identity() != identity["code"]:
        raise ValueError("Import the original frozen P01 package through PYTHONPATH; do not diagnose substituted code.")
    output = args.output or root / "diagnosis/gradients.json"
    train_path, val_path = root / "banks/train/bank.pt", root / "banks/D128/bank.pt"
    train, val = _bank(train_path), _bank(val_path)
    if train["producer"] != val["producer"] or train["producer"] != identity["producer"]:
        raise ValueError("Bank producer differs from the actual P01 training identity.")
    for key in train["stats"]:
        if not torch.equal(train["stats"][key], val["stats"][key]):
            raise ValueError("Train and validation statistics disagree.")
    device = torch.device(args.device)
    codec = MotionCodec(train["stats"]).to(device)
    smpl = load_smpl(device)
    if smpl_identity() != identity["smpl_asset_sha256"]:
        raise ValueError("Diagnosis SMPL-X asset differs from P01 training.")
    fk = FixedShapeFK(smpl, torch.zeros(10, device=device), shape_mode="predicted")
    indices = sample_train_indices(train, samples=args.samples, seed=args.seed)
    phases = split_phases(train, indices)
    if [len(phases[name]) for name in PHASE_WEIGHTS] != [args.samples * 3 // 4, args.samples // 4]:
        raise AssertionError("Diagnostic phase quota differs from the unchanged P01 sampler.")
    weight = identity["geometry_audit"]["weight"]
    recovery = torch.load(run / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
    if recovery["step"] != 2000 or recovery["identity"] != identity:
        raise ValueError("Training-visit membership must come from the matching raw2000 checkpoint.")
    visited = set().union(*(set(state["visited_indices"]) for state in recovery["rank_rng_states"]))
    del recovery
    result = {
        "status": "running",
        "protocol": {
            "scope": "read-only loss/gradient diagnosis; no training, no optimizer or quota/weight changes",
            "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gradient_mode": "eval mode, dropout disabled; mean parameter gradients, not exact stochastic clipped AdamW updates",
            "geometry_weight": weight,
            "phase_weights": PHASE_WEIGHTS,
            "gradient_phase_counts": {phase: len(selected) for phase, selected in phases.items()},
            "sample_seed": args.seed,
            "train_indices": indices,
            "training_visits": {
                "membership_checkpoint_step": 2000,
                "unique_bank_anchors_visited": len(visited),
                "sampled_rows_visited": sum(index in visited for index in indices),
                "interpretation": "For init/EMA500 this is future membership by2000, not proof those rows had been seen at the earlier checkpoint.",
            },
            "sampler": "unchanged FormalPriorSampler: take -> source -> anchor, each micro8 is6 mature +2 startup",
            "code_groups": {
                name: {"indices": list(selected), "weight": 8 if name == "reference9" else 1}
                for name, selected in GROUPS.items()
            },
            "code_denominator": 306,
            "geometry": "own predicted-beta FK vs official joints; mean joint squared Euclidean distance /0.1m^2",
            "aggregation_boundary": "train: fixed hierarchy-sampled rows; val: dense rows and equal clips. Also report train query20..79 and standardized75:25; these do not erase task/source differences.",
            "frozen_code": identity["code"],
            "script_sha256": file_sha256(__file__),
            "asset_sha256": identity["smpl_asset_sha256"],
            "expected_training_bank_sha256": identity["bank_sha256"],
            "bank_file_fingerprints": {
                name: {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for name, path in (("train", train_path), ("val", val_path))
            },
            "bank_rehashed": False,
        },
        "models": {},
    }
    save_json(output, result)
    cases = (
        ("init_CV", None, None),
        ("EMA500", run / "best.pt", "state_dict"),
        ("EMA2000", run / "last.pt", "state_dict"),
        ("raw2000", run / "resume.pt", "model"),
    )
    for label, path, state_key in cases:
        started = time.monotonic()
        torch.manual_seed(62)
        model = HistoryPrior(**identity["model_config"]).to(device).eval()
        checkpoint = None if path is None else torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        if checkpoint is not None:
            expected_step = 500 if label == "EMA500" else 2000
            if checkpoint["step"] != expected_step or checkpoint["identity"] != identity:
                raise ValueError(f"Unexpected {label} checkpoint identity/step.")
            model.load_state_dict(checkpoint[state_key], strict=True)
        before = state_sha256(model.state_dict())
        if path is None and before != identity["random_initial_model_sha256"]:
            raise ValueError("Reconstructed initial P differs from seed62 training initialization.")
        audits = {}
        for phase, selected in phases.items():
            audits[phase] = phase_gradient_audit(
                model, train, selected, codec, fk, weight, device=device, microbatch=args.microbatch
            )
            print(json.dumps({"model": label, "phase": phase, "gradient_samples": len(selected)}), flush=True)
        train_rows = [row for audit in audits.values() for row in audit["rows"]]
        val_rows = inference_rows(
            model, val, list(range(len(val["rows"]))), codec, fk, weight, device=device, batch_size=args.batch_size
        )
        after = state_sha256(model.state_dict())
        if before != after:
            raise AssertionError("Read-only diagnosis changed P weights.")
        result["models"][label] = {
            "checkpoint": str(path) if path is not None else None,
            "checkpoint_sha256": file_sha256(path) if path is not None else None,
            "step": checkpoint["step"] if checkpoint is not None else 0,
            "weights_sha256": before,
            "weights_unchanged": before == after,
            "gradient_analysis": gradient_relationships(audits, weight),
            "train": aggregate_metrics(train_rows),
            "train_visited_by2000": aggregate_metrics([row for row in train_rows if row["index"] in visited]),
            "train_not_visited_by2000": aggregate_metrics([row for row in train_rows if row["index"] not in visited]),
            "train_matched_query20_79": aggregate_metrics([row for row in train_rows if row["frame"] < 80]),
            "val_row_equal": aggregate_metrics(val_rows),
            "val_clip_equal": aggregate_metrics(val_rows, clip_equal=True),
            "elapsed_seconds": time.monotonic() - started,
        }
        save_json(output, result)
        print(
            json.dumps(
                {
                    "model": label,
                    "elapsed_seconds": result["models"][label]["elapsed_seconds"],
                    "weighted_total": result["models"][label]["gradient_analysis"]["quota_weighted_relationships"][
                        "weighted_total"
                    ],
                }
            ),
            flush=True,
        )
        del model, checkpoint, audits, train_rows, val_rows
    result["status"] = "complete"
    save_json(output, result)


if __name__ == "__main__":
    main()
