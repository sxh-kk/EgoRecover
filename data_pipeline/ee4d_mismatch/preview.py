"""Plot injected visual age and trajectory change for one paired episode."""

import argparse
from pathlib import Path

import numpy as np

from .dataset import MismatchDataset


def preview(root, output, source=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = MismatchDataset(root, source=source)
    first = dataset.record(0)
    paired = [r for r in dataset.records if r["pair_id"] == first["pair_id"]]
    clean = dataset.arrays(first["clean_variant_id"])
    time = np.arange(first["num_frames"])/10
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for record in paired:
        name = record["variant_name"]
        if name not in {"freeze_1s", "freeze_3s", "delay_0p2s", "delay_0p4s", "drift_0p01mps", "drift_0p03mps"}:
            continue
        arrays = dataset.arrays(record["variant_id"])
        if name.startswith("drift"):
            difference = np.linalg.norm(arrays["aria_traj_obs"][:, 6:]-clean["aria_traj_obs"][:, 6:], axis=-1)
            axes[1].plot(time, difference*100, label=name)
        else:
            age = (clean["img_source_idx"]-arrays["img_source_idx"])/5
            axes[0].step(time, age, where="post", label=name)
    for ax in axes:
        ax.axvspan(0, first["bootstrap_frames"]/10, color="gray", alpha=0.12, label="clean bootstrap")
        ax.axvline(first["fault_onset"]/10, color="black", alpha=0.4, linestyle="--")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Extra visual source age (s)")
    axes[1].set_ylabel("Injected position change (cm)")
    axes[1].set_xlabel("Episode time (s)")
    figure.suptitle(f"EE4D mismatch input audit | {first['base_take_name']}\nInput changes only; this is not a model accuracy plot")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(preview(args.dataset, args.output))


if __name__ == "__main__":
    main()
