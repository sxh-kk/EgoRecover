"""Export the completed diagnostics as a compact JSON and standalone figure."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verification/egorecover_engineering_summary.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiments = {}
    for version in ("v0", "v1"):
        folder = root / "exp" / f"egorecover_engineering_{version}"
        experiments[version] = {
            "report": json.loads((folder / "report.json").read_text()),
            "baselines": json.loads((folder / "baselines.json").read_text()),
        }
    closed = {}
    for path in sorted((root / "exp").glob("egorecover_closed_loop_*/report.json")):
        closed[path.parent.name] = json.loads(path.read_text())
    suites = list(ET.parse(root / "verification/egorecover_tests.xml").getroot().iter("testsuite"))
    checks = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    sources = sorted(
        list((root / "egorecover").glob("*.py"))
        + [
            root / "model/history_uniegomotion.py",
            root / "environment.yml",
            *(root / "run").glob("*engineering*.py"),
            root / "run/check_closed_loop.py",
            root / "run/check_mismatch_adapter.py",
        ]
    )
    summary = {
        "scope": "engineering feasibility checks; not a positive method-performance result",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "conda_env": "egorecover",
        },
        "tests": checks,
        "dataset_adapter": json.loads((root / "verification/mismatch_adapter.json").read_text()),
        "experiments": experiments,
        "closed_loop": closed,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "unresolved": [
            "no pretrained E7 initializer located",
            "no SMPL-X asset / formal FK metrics",
            "teacher-forced to predicted-history distribution shift",
            "P/G worse than constant velocity in current dense diagnostic",
            "startup floor estimation error",
            "formal Q training requires reachable-state FK labels",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    latest = experiments["v1"]["report"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    prior = latest["prior"]
    axes[0, 0].plot(
        [0] + [r["step"] for r in prior["curve"]],
        [prior["baseline_dev_loss"]] + [r["dev_loss"] for r in prior["curve"]],
        label="P",
        color="#0072b2",
    )
    axes[0, 0].axhline(prior["baseline_dev_loss"], color="#555555", linestyle="--", label="Physical hold")
    axes[0, 0].set(title="P: clean-dev representation loss", xlabel="Training steps", ylabel="Weighted MSE")
    axes[0, 0].legend()
    colors = {"gaussian": "#d55e00", "history": "#0072b2"}
    for mode in ("gaussian", "history"):
        curve = latest["sources"][mode]["curve"]
        axes[0, 1].plot(
            [r["step"] for r in curve], [r["train_loss"] for r in curve], label=mode, color=colors[mode], alpha=0.85
        )
    axes[0, 1].set(title="G: training loss (v1)", xlabel="Training steps", ylabel="Weighted MSE", yscale="log")
    axes[0, 1].legend()
    names = ["constant_velocity", "physical_hold", "P", "G_gaussian_a11", "G_history_a11"]
    results = experiments["v1"]["baselines"]["results"]
    values = [results[name]["dense22_mm"] for name in names]
    bars = axes[1, 0].bar(
        range(5), values, color=["#009e73", "#888888", "#cc79a7", colors["gaussian"], colors["history"]]
    )
    axes[1, 0].bar_label(bars, fmt="%.1f", padding=3)
    axes[1, 0].set_xticks(range(5), ["Const. vel.", "Hold", "P", "G Gaussian", "G History"])
    axes[1, 0].set(
        title="Same 64 dev frames: simple priors remain stronger",
        ylabel="Dense22 position error (mm)",
        ylim=(0, max(values) * 1.2),
    )
    x = np.arange(2)
    for offset, mode in ((-0.18, "gaussian"), (0.18, "history")):
        values = [
            experiments[version]["report"]["sources"][mode]["dense22_mm_per_action"]["a11"] for version in ("v0", "v1")
        ]
        bars = axes[1, 1].bar(x + offset, values, width=0.34, label=mode, color=colors[mode])
        axes[1, 1].bar_label(bars, fmt="%.1f", padding=3)
    axes[1, 1].set_xticks(x, ["v0: 100 steps, sigma=1", "v1: 1000 steps, sigma=0.3"])
    axes[1, 1].set(
        title="Development iterations (multiple settings changed)", ylabel="Dense22 position error (mm)", ylim=(0, 470)
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("EgoRecover engineering diagnostics", fontsize=16)
    fig.text(
        0.5,
        0.025,
        "Teacher-forced official-val engineering samples; random G initialization; dense joints, not SMPL FK.\n"
        "Contact supervision also changed in v1. These plots do not establish closed-loop reconstruction quality.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    figure = args.output.with_suffix(".png")
    fig.savefig(figure, dpi=160)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"summary": str(args.output), "figure": str(figure), "tests": checks}, indent=2))


if __name__ == "__main__":
    main()
