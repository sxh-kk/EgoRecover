"""Snapshot corrected-code verification and engineering experiments v2/v3."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    experiments = {}
    for version in ("v1", "v2", "v3"):
        folder = root / "exp" / f"egorecover_engineering_{version}"
        report = json.loads((folder / "report.json").read_text())
        if not report.get("completed"):
            raise ValueError(f"{version} has not finished.")
        experiments[version] = {"report": report, "baselines": json.loads((folder / "baselines.json").read_text())}
    names = (
        "egorecover_closed_loop_v1_planar_production",
        "egorecover_closed_loop_v2_clean",
        "egorecover_closed_loop_v3_take0",
        "egorecover_closed_loop_v3_take1",
    )
    closed = {name: json.loads((root / "exp" / name / "report.json").read_text()) for name in names}
    body_baselines = {name: json.loads((root / "exp" / name / "body_baselines.json").read_text()) for name in names[1:]}
    planar_diagnostic = json.loads((root / "exp/egorecover_diagnosis_dev_geometry_v1/report.json").read_text())[
        "single_step"
    ]
    v1_planar = {
        mode: planar_diagnostic[mode]["sampling"]["10"]["planar_reference_dense22_mm"]
        for mode in ("gaussian", "history")
    }
    suites = list(ET.parse(root / "verification/egorecover_tests.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    sources = sorted(
        list((root / "egorecover").glob("*.py"))
        + [root / "model/history_uniegomotion.py"]
        + [
            root / "run" / name
            for name in (
                "engineering_pilot.py",
                "check_closed_loop.py",
                "collect_predicted_histories.py",
                "compare_closed_loop_baselines.py",
                "compare_engineering_baselines.py",
                "report_repairs.py",
            )
        ]
    )
    summary = {
        "scope": "engineering repair verification; official-val development data; dense22 rather than SMPL FK",
        "tests": tests,
        "adapter": json.loads((root / "verification/mismatch_adapter_planar.json").read_text()),
        "experiments": experiments,
        "closed_loop": closed,
        "closed_loop_body_baselines": body_baselines,
        "v1_fixed_weights_planar_single_step_mm": v1_planar,
        "replay": json.loads((root / "exp/egorecover_predicted_train_v1/report.json").read_text()),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "limitations": [
            "random common E7 initializer, not pretrained",
            "no actual SMPL-X FK verification",
            "Q not formally trained",
            "single-step selection uses GT-history dev frames",
            "replay is one offline collection round from frozen v1 policies, with GT startup on train takes only",
            "concurrent GPU jobs: recorded latency is not a fair speed benchmark",
            "official-val pilot engineering splits; no final independent benchmark claim",
            "hold/CV closed-loop baselines use noisy random bootstrap positions and no subsequent observations",
        ],
    }
    output = root / "verification/egorecover_repair_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(4)
    colors = {"gaussian": "#d55e00", "history": "#0072b2"}
    for offset, mode in ((-0.18, "gaussian"), (0.18, "history")):
        values = [experiments["v1"]["report"]["sources"][mode]["dense22_mm_per_action"]["a11"], v1_planar[mode]]
        values += [experiments[v]["report"]["sources"][mode]["dense22_mm_per_action"]["a11"] for v in ("v2", "v3")]
        bars = axes[0, 0].bar(x + offset, values, 0.34, label=mode, color=colors[mode])
        axes[0, 0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0, 0].axhline(
        experiments["v3"]["baselines"]["results"]["constant_velocity"]["dense22_mm"],
        color="#009e73",
        linestyle="--",
        label="constant velocity",
    )
    axes[0, 0].set_xticks(x, ["v1 original", "v1 planar", "v2 geometry", "v3 + replay"], fontsize=8)
    axes[0, 0].set(title="Same 64 dev frames, GT history", ylabel="Dense22 error (mm)")
    axes[0, 0].legend(fontsize=8)
    for version, color in (("v2", "#d55e00"), ("v3", "#0072b2")):
        prior = experiments[version]["report"]["prior"]
        axes[0, 1].plot(
            [0] + [p["step"] for p in prior["curve"]],
            [prior["baseline_dev_dense22_mm"]] + [p["dev_dense22_mm"] for p in prior["curve"]],
            label=version,
            color=color,
        )
    axes[0, 1].axhline(prior["baseline_dev_dense22_mm"], linestyle="--", color="#777777", label="hold / step 0")
    axes[0, 1].set(
        title="P selection uses physical error", xlabel="Training step", ylabel="Clean-dev dense22 error (mm)"
    )
    axes[0, 1].legend()
    clean_reports = [closed[names[0]], closed[names[1]], closed[names[2]]]
    x = np.arange(3)
    for offset, mode in ((-0.18, "gaussian"), (0.18, "history")):
        values = [
            next(
                r.get("mean_dense22_mm", float("nan"))
                for r in report["results"]
                if r["source_mode"] == mode and r["variant"] == "clean"
            )
            for report in clean_reports
        ]
        bars = axes[1, 0].bar(x + offset, values, 0.34, label=mode, color=colors[mode])
        axes[1, 0].bar_label(bars, fmt="%.1f", padding=3)
    axes[1, 0].set_xticks(x, ["v1 + planar codec", "v2", "v3"])
    axes[1, 0].set(
        title="Soccer clean: random E7 startup, predictions only", ylabel="180-frame mean dense22 error (mm)"
    )
    axes[1, 0].legend()
    labels = []
    values = {mode: [] for mode in colors}
    for take_index in (0, 1):
        report = closed[f"egorecover_closed_loop_v3_take{take_index}"]
        for variant in ("clean", "freeze_3s", "drift_0p03mps"):
            labels.append(
                f"T{take_index+1} " + {"clean": "clean", "freeze_3s": "freeze", "drift_0p03mps": "drift"}[variant]
            )
            for mode in colors:
                row = next(r for r in report["results"] if r["source_mode"] == mode and r["variant"] == variant)
                values[mode].append(row.get("mean_dense22_mm", float("nan")))
    for mode, color in colors.items():
        axes[1, 1].plot(range(6), values[mode], marker="o", label=mode, color=color)
    axes[1, 1].set_xticks(range(6), labels, rotation=20)
    axes[1, 1].set(title="v3 continuous replay: two dev takes", ylabel="180-frame mean dense22 error (mm)")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("EgoRecover: repairs and verification", fontsize=16)
    fig.text(
        0.5,
        0.022,
        "v2/v3 include planar reference, geometry loss and physical checkpoint selection; v3 adds predicted training histories.\n"
        "Engineering pilot only. Upper panels use GT history; lower panels use random MODEL startup with no GT reset. No SMPL FK.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(
        json.dumps(
            {
                "summary": str(output),
                "tests": tests,
                "closed_loop_cases": sum(len(r["results"]) for r in closed.values()),
                "closed_loop_finite_cases": sum(
                    row.get("finite", False) for report in closed.values() for row in report["results"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
