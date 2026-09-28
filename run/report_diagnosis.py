"""Export frozen-checkpoint cause-analysis probes, including their limitations."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    reports = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted((root / "exp").glob("egorecover_diagnosis*/report.json"))
    }
    base = reports["egorecover_diagnosis_v1"]
    geometry = reports["egorecover_diagnosis_geometry_v1"]
    train = reports["egorecover_diagnosis_train_v1"]["single_step"]
    dev = reports["egorecover_diagnosis_dev_geometry_v1"]["single_step"]
    rows = []
    for name, report in reports.items():
        for row in report["closed_loop"]:
            errors = [point["dense22_mm"] for point in row["trace"] if point.get("dense22_mm") is not None]
            metrics_complete = len(errors) == 180 and row["completed"]
            rows.append(
                {
                    "report": name,
                    "take": report["bootstrap"]["take"],
                    "noise_seed": report.get("rollout_noise_seed", 62),
                    **{key: value for key, value in row.items() if key != "trace"},
                    "full_mean_mm": float(np.mean(errors)) if metrics_complete else None,
                    "full_max_mm": max(errors) if metrics_complete else None,
                    "first_error_over_1m_frame": next(
                        (point["frame"] for point in row["trace"] if (point.get("dense22_mm") or 0) > 1000), None
                    ),
                    "finite_metric_frames": len(errors),
                }
            )
    experiment = json.loads((root / "exp/egorecover_engineering_v1/report.json").read_text())
    summary = {
        "scope": "frozen-checkpoint cause analysis; GT-bootstrap probes are offline only; dense22, not SMPL FK",
        "rollouts": rows,
        "reports": reports,
        "checkpoint_ids": {mode: value["checkpoint_sha256"] for mode, value in experiment["sources"].items()},
        "limitations": [
            "Only two engineering dev takes and limited noise seeds; no significance/generalization claim",
            "No valid deployed initializer or actual SMPL-X FK result",
            "Initial random-bootstrap probes used an extra encode/decode roundtrip; later probes preserve physical states",
            "Completed execution does not establish physical stability; large errors must also be inspected",
            "Planar projection is only an experimental postprocess in the diagnostic script",
        ],
        "script_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("run/diagnose_engineering.py", "run/report_diagnosis.py")
        },
    }
    output = root / "verification/egorecover_cause_analysis.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = ["joint_translation", "joint_rotation", "reference_translation", "reference_rotation"]
    x = np.arange(len(names))
    for offset, method, color in ((-0.18, "hold", "#777777"), (0.18, "P", "#0072b2")):
        values = [base["single_step"][method]["channel_mse"][name] for name in names]
        axes[0, 0].bar(x + offset, values, 0.34, label=method, color=color)
    axes[0, 0].set_xticks(x, ["Joint pos.", "Joint rot.", "Ref. pos.", "Ref. rot."])
    axes[0, 0].set(title="P: dev loss components move in different directions", ylabel="Normalized channel MSE")
    axes[0, 0].legend()
    x = np.arange(4)
    raw = []
    planar = []
    for split in (train, dev):
        for mode in ("gaussian", "history"):
            item = split[mode]["sampling"]["10"]
            raw.append(item["dense22_mm"])
            planar.append(item["planar_reference_dense22_mm"])
    for offset, values, label, color in (
        (-0.18, raw, "Original", "#d55e00"),
        (0.18, planar, "Planar reference", "#009e73"),
    ):
        bars = axes[0, 1].bar(x + offset, values, 0.34, label=label, color=color)
        axes[0, 1].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0, 1].set_xticks(x, ["Train Gaus.", "Train Hist.", "Dev Gaus.", "Dev Hist."])
    axes[0, 1].set(
        title="Same frozen G: reference constraint ablation", ylabel="Single-step dense22 error (mm)", ylim=(0, 210)
    )
    axes[0, 1].legend()
    originals = next(
        row
        for row in base["closed_loop"]
        if row["source"] == "history" and row["bootstrap"] == "GT_oracle" and row["prior"] == "P"
    )
    trajectories = [(originals, "a11 original", "#d55e00")]
    for row in geometry["closed_loop"]:
        if row["action"] == "a10":
            trajectories.append((row, "a10 original", "#0072b2"))
        if row["planar_reference_projection"]:
            trajectories.append((row, "a11 planar ref.", "#009e73"))
    for row, label, color in trajectories:
        points = [point for point in row["trace"] if point["frame"] <= 60 and point.get("dense22_mm") is not None]
        axes[1, 0].plot([p["frame"] for p in points], [p["dense22_mm"] for p in points], label=label, color=color)
        points = [p for p in row["trace"] if p.get("reference_tilt") is not None]
        axes[1, 1].plot([p["frame"] for p in points], [p["reference_tilt"] for p in points], label=label, color=color)
    axes[1, 0].set(
        title="GT startup, then predictions only: early error growth",
        xlabel="Frame index (0-based)",
        ylabel="Dense22 error (mm, log scale)",
        yscale="log",
    )
    axes[1, 1].set(
        title="Internal reference leaves its planar definition",
        xlabel="Frame index (0-based)",
        ylabel="Off-plane rotation component norm",
    )
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.suptitle("EgoRecover: frozen-checkpoint failure diagnosis", fontsize=16)
    fig.text(
        0.5,
        0.025,
        "GT bootstrap / GT substitutions are OFFLINE diagnostics. Learned weights are unchanged.\n"
        "These results identify an implementation issue; they do not prove deployed robustness or superiority over baselines.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"output": str(output), "rollouts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
