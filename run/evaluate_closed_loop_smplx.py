"""Evaluate saved causal rollouts using an authorized SMPL-X neutral asset.

This is an OFFLINE evaluator. The original inference already ended before this
script reads supervision. It does not substitute GT shape, pose, or floor into
the saved predictions. An incompatible SMPL-X asset fails the GT roundtrip
audit before any metric report is written.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from dataset.smpl_utils import get_smpl
from egorecover.codec import MotionCodec
from egorecover.data import DEFAULT_SIGNAL, open_dataset
from egorecover.evaluation_protocol import event_window, paired_fault_delta
from egorecover.smpl_evaluation import evaluate_saved_case, prepare_ground_truth


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True, help="A completed run.check_closed_loop directory.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path.")
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument("--smplx-dir", type=Path, help="Directory containing SMPLX_NEUTRAL.npz.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gt-audit-tolerance-mm", type=float, default=5.0)
    parser.add_argument(
        "--recovery-tolerance-mm", type=float, help="Predeclared development-set tolerance for paired fault recovery."
    )
    parser.add_argument("--source-modes", nargs="+", choices=("gaussian", "history", "prior_only"))
    parser.add_argument("--variants", nargs="+", choices=("clean", "freeze_3s", "drift_0p03mps"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file to preserve prior evidence.")
    if args.batch_size < 1 or args.gt_audit_tolerance_mm <= 0:
        parser.error("Batch size and GT audit tolerance must be positive.")
    if args.recovery_tolerance_mm is not None and (
        not math.isfinite(args.recovery_tolerance_mm) or args.recovery_tolerance_mm < 0
    ):
        parser.error("Recovery tolerance must be finite and nonnegative.")
    model_dir = (args.smplx_dir or Path(os.environ.get("SMPLX_MODEL_PATH", "body_models/smplx"))).expanduser()
    model_file = model_dir / "SMPLX_NEUTRAL.npz"
    if not model_file.is_file():
        parser.error(
            f"Authorized SMPL-X neutral asset missing: {model_file}. "
            "Download it from the registered SMPL-X website; do not generate metrics without it."
        )
    # Keep the factory, asset fingerprint, and GT audit on the same file.
    os.environ["SMPLX_MODEL_PATH"] = str(model_dir.resolve())
    torch.set_num_threads(4)
    device = torch.device(args.device)
    original = json.loads((args.rollout / "report.json").read_text())
    if not original.get("completed") or original.get("history_source") != "model_predictions_only; no resets":
        raise ValueError("Expected a completed prediction-only closed-loop report.")
    dataset, _ = open_dataset(signal=args.signal, spec_sha256=original["dataset_spec_sha256"])
    records = {
        record["variant_name"]: record for record in dataset.records if record["base_take_name"] == original["take"]
    }
    if "clean" not in records:
        raise ValueError(f"No clean source for {original['take']}.")
    stats_path = dataset.source.root / "uniegomotion/v4_beta_ee_train_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    codec = MotionCodec(stats, reference_mode=original["reference_mode"]).to(device)
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    rows = [
        row
        for row in original["results"]
        if (args.source_modes is None or row["source_mode"] in args.source_modes)
        and (args.variants is None or row["variant"] in args.variants)
    ]
    if not rows or not all(row.get("finite") for row in rows):
        raise ValueError("Select at least one completed finite rollout case.")
    selected_records = [records[row["variant"]] for row in rows]
    first_path = args.rollout / f"{rows[0]['source_mode']}_{rows[0]['variant']}.pt"
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    supervision = dataset.supervision(records["clean"]["variant_id"])
    ground_truth = prepare_ground_truth(
        smpl,
        supervision,
        first["frame_indices"],
        batch_size=args.batch_size,
        audit_tolerance_mm=args.gt_audit_tolerance_mm,
    )
    results = []
    for row in rows:
        mode, variant = row["source_mode"], row["variant"]
        record = records[variant]
        window = event_window(record, selected_records)
        trace_path = args.rollout / f"{mode}_{variant}.pt"
        saved = torch.load(trace_path, map_location="cpu", weights_only=True)
        evaluated = evaluate_saved_case(
            smpl,
            codec,
            saved,
            ground_truth,
            fault_window=window,
            batch_size=args.batch_size,
        )
        delta = abs(evaluated["diagnostics"]["dense22_mm_recomputed"] - row["mean_dense22_mm"])
        if delta > 0.02:
            raise ValueError(f"{mode}/{variant} does not reproduce its earlier dense22 metric: {delta:.4f} mm.")
        results.append(
            {
                "source_mode": mode,
                "variant": variant,
                "trace_sha256": file_sha256(trace_path),
                "fault_onset": record["fault_onset"],
                "fault_window": list(window) if window is not None else None,
                **evaluated,
            }
        )
        print(
            f"{mode}/{variant}: SMPL22 MPJPE={evaluated['metrics']['mpjpe_body_m']:.4f} m, "
            f"PA={evaluated['metrics']['mpjpe_body_pa_m']:.4f} m",
            flush=True,
        )
    by_mode = {(result["source_mode"], result["variant"]): result for result in results}
    for result in results:
        if result["variant"] == "clean":
            continue
        clean = by_mode.get((result["source_mode"], "clean"))
        if clean is not None and result["fault_window"] is not None:
            result["paired_fault_delta"] = paired_fault_delta(
                result["per_frame_smpl22_mm"],
                clean["per_frame_smpl22_mm"],
                first["frame_indices"],
                result["fault_window"],
                fps=dataset.spec["motion_fps"],
                recovery_tolerance_mm=args.recovery_tolerance_mm,
            )
    report = {
        "completed": True,
        "scope": f"offline_Smplx_geometry_on_saved_predicted_history; official_{dataset.spec['base_split']}_pilot",
        "group": original.get("group", "dev"),
        "protocol": "200-frame take, frames after the 20-frame model bootstrap; not the paper's 80-frame strided validation",
        "metric_definitions": "UniEgoMotion eval.metrics geometric reconstruction subset; meters except foot_slide_mm",
        "paper": "https://chaitanya100100.github.io/UniEgoMotion/Patel2025UniEgoMotion.pdf",
        "unavailable_metrics": "TMR semantic similarity and FID require separate pretrained motion encoder and official sampling",
        "take": original["take"],
        "reference_mode": original["reference_mode"],
        "prediction_source": "saved model commits only; no GT body/shape/floor used in inference or FK",
        "fixed_shape_source": "beta_boot from the shared model-generated startup",
        "ground_truth_usage": "offline SMPL-X asset audit and metric calculation after inference",
        "floor_for_foot_metrics": "source annotation, offline evaluation only",
        "smplx_asset_sha256": file_sha256(model_file),
        "model_bundle_version": (
            (model_dir / "version.txt").read_text().strip() if (model_dir / "version.txt").is_file() else None
        ),
        "stats_sha256": file_sha256(stats_path),
        "source_rollout_report_sha256": file_sha256(args.rollout / "report.json"),
        "gt_asset_audit": ground_truth["asset_audit"],
        "gt_audit_tolerance_mm": args.gt_audit_tolerance_mm,
        "recovery_tolerance_mm": args.recovery_tolerance_mm,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved SMPL-X geometry report to {args.output}")


if __name__ == "__main__":
    main()
