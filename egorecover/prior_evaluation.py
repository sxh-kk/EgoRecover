"""Evaluate P centers on consecutive frozen-G histories, including mature motion."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .codec import MotionCodec
from .distributed import DistributedContext
from .fk import FixedShapeFK, load_smpl
from .io import file_sha256, save_json
from .metrics import foot_slide_uniegomotion
from .models import HistoryPrior
from .prior_data import predictor_forward, select_examples
from .prior_training import load_bank

METRICS = ("mpjpe_mm", "velocity_mm_s", "acceleration_mm_s2", "foot_slide_mm")


def center_metrics(positions, truth, timestamps, floor_calibrated):
    """Physical derivatives of consecutive centers; feet use a known floor only."""
    positions, truth, timestamps = [value.double() for value in (positions, truth, timestamps)]
    intervals = torch.diff(timestamps)
    if len(positions) < 3 or not bool((intervals > 0).all()):
        raise ValueError("Center trajectories need at least three consecutive frames.")
    if not torch.allclose(intervals, intervals.mean().expand_as(intervals), atol=1e-7, rtol=2e-5):
        raise ValueError("Center derivatives require uniform timestamps.")
    dt = float(intervals.mean())
    error = positions - truth
    return {
        "mpjpe_mm": float(error.norm(dim=-1).mean() * 1000),
        "velocity_mm_s": float((torch.diff(error, dim=0) / dt).norm(dim=-1).mean() * 1000),
        "acceleration_mm_s2": float((torch.diff(error, n=2, dim=0) / dt**2).norm(dim=-1).mean() * 1000),
        "foot_slide_mm": foot_slide_uniegomotion(positions) if floor_calibrated else None,
    }


def summarize_centers(rows):
    """Average clips within take before averaging takes; bootstrap paired takes."""
    groups = {}
    for phase in ("all", "mature"):
        for window in (1, 3):
            selected = [row for row in rows if row["phase"] == phase and row["window"] == window]
            takes = sorted({row["take"] for row in selected})
            result = {"clips": len(selected), "takes": len(takes), "metrics": {}}
            for metric in METRICS:
                per_take = {}
                for take in takes:
                    values = [row for row in selected if row["take"] == take and row["P"][metric] is not None]
                    if values:
                        per_take[take] = {
                            source: sum(row[source][metric] for row in values) / len(values) for source in ("P", "CV")
                        }
                if not per_take:
                    result["metrics"][metric] = None
                    continue
                delta = np.array([value["P"] - value["CV"] for value in per_take.values()])
                rng = np.random.default_rng(1062)
                boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
                result["metrics"][metric] = {
                    source: sum(value[source] for value in per_take.values()) / len(per_take) for source in ("P", "CV")
                }
                result["metrics"][metric].update(
                    delta=float(delta.mean()),
                    delta_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                    takes=len(per_take),
                )
            groups[f"{phase}_W{window}"] = result
    # Improvement in position alone is insufficient for this mature-history trial.
    available = all(groups[f"mature_W{window}"]["takes"] for window in (1, 3))
    position = available and all(
        groups[f"mature_W{window}"]["metrics"]["mpjpe_mm"]["delta_ci95"][1] < 0 for window in (1, 3)
    )
    temporal = available and all(
        groups[f"mature_W{window}"]["metrics"][metric] is not None
        and groups[f"mature_W{window}"]["metrics"][metric]["delta"] <= 0
        for window in (1, 3)
        for metric in METRICS[1:]
    )
    return {
        "groups": groups,
        "position_gate_passed": bool(position),
        "temporal_gate_passed": bool(temporal),
        "gate_passed": bool(position and temporal),
        "gate_rule": "Both mature W1/W3: MPJPE paired-take upper95CI<0; mean velocity, acceleration and calibrated-foot slide no worse than matched CV",
        "uncertainty_scope": "paired take bootstrap at a fixed checkpoint; not training-seed uncertainty",
    }


@torch.no_grad()
def evaluate_centers(model, bank, codec, fk, context, *, mature_frame=40, all_floor_calibrated=False):
    model.eval()
    grouped = defaultdict(list)
    for index, row in enumerate(bank["rows"]):
        if row["kind"] == "predicted":
            grouped[(row["episode"], row["window"])].append(index)
    rows = []
    for group_index, ((episode, window), indices) in enumerate(sorted(grouped.items())):
        if group_index % context.world_size != context.rank:
            continue
        indices.sort(key=lambda index: bank["rows"][index]["frame"])
        metadata = [bank["rows"][index] for index in indices]
        frames = torch.tensor([row["frame"] for row in metadata])
        if not bool((torch.diff(frames) == 1).all()):
            raise ValueError("Temporal center evaluation requires consecutive, nonduplicated history queries.")
        timestamps = torch.tensor([row.get("timestamp_s", row["frame"] / 10) for row in metadata], dtype=torch.float64)
        floor = all(row.get("floor_calibrated", all_floor_calibrated) for row in metadata)
        outputs = {key: [] for key in ("P", "CV", "truth")}
        for start in range(0, len(indices), 32):
            example = select_examples(bank["tensors"], indices[start : start + 32], context.device)
            prediction = predictor_forward(model, example)
            for source, codes in (("P", prediction), ("CV", example["base"])):
                outputs[source].append(fk(codec.decode_current(codes[:, 0], example["reference"])).cpu())
            outputs["truth"].append(example["truth_xyz"].cpu())
        outputs = {key: torch.cat(value) for key, value in outputs.items()}
        for phase, selected in (("all", frames >= 20), ("mature", frames >= mature_frame)):
            if int(selected.sum()) < 3:
                continue
            rows.append(
                {
                    "episode": episode,
                    "take": metadata[0]["take"],
                    "window": window,
                    "phase": phase,
                    "frames": int(selected.sum()),
                    "floor_calibrated": floor,
                    **{
                        source: center_metrics(
                            outputs[source][selected], outputs["truth"][selected], timestamps[selected], floor
                        )
                        for source in ("P", "CV")
                    },
                }
            )
    rows = context.gather_rows(rows)
    return {
        "scope": "P/CV center prediction on matched frozen-G histories, not final G reconstructions",
        "aggregation": "clip means within take, then take-equal; mature starts after20 generated frames",
        "mature_frame": mature_frame,
        "summary": summarize_centers(rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mature-frame", type=int, default=40)
    parser.add_argument("--all-floor-calibrated", action="store_true")
    args = parser.parse_args()
    with DistributedContext.from_environment(args.device) as context:
        torch.set_num_threads(2)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        codec = MotionCodec(saved["stats"]).to(context.device)
        fk = FixedShapeFK(load_smpl(context.device), torch.zeros(10, device=context.device))
        bank = load_bank(args.bank, "val", saved["manifest_sha256"])
        model = HistoryPrior(**saved["model_config"]).to(context.device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        report = evaluate_centers(
            model,
            bank,
            codec,
            fk,
            context,
            mature_frame=args.mature_frame,
            all_floor_calibrated=args.all_floor_calibrated,
        )
        if context.is_primary:
            save_json(
                args.output,
                {
                    "checkpoint_step": saved["step"],
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "bank_sha256": file_sha256(args.bank),
                    **report,
                },
            )


if __name__ == "__main__":
    main()
