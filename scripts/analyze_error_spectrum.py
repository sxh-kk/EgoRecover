"""CPU-only frequency diagnostics of existing, matched EE4D rollout outputs."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

CASES = {
    "baseline": "history-policy-ablation-20261110/evaluation400/C_mixed_on",
    "two_step": "two-step-vs-paired-C-20261110/evaluation400/two_step",
    "detached": "ac-detach-velocity-C-20261110/evaluation400/A_two_step_detached",
    "velocity": "ac-detach-velocity-C-20261110/evaluation400/C_two_step_velocity",
}
BANDS = ("DC", "0_to_0p5Hz", "0p5_to_1Hz", "1_to_2p5Hz", "2p5_to_5Hz")
GROUPS = {
    "pelvis": [0],
    "torso_head": [3, 6, 9, 12, 15],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21],
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "wrists": [20, 21],
    "feet": [10, 11],
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(frequencies):
    f = frequencies
    return [f == 0, (f > 0) & (f <= 0.5), (f > 0.5) & (f <= 1), (f > 1) & (f <= 2.5), f > 2.5]


def spectrum(errors, *, hann=False):
    """Return per-clip/joint/bin mean-square vector error, preserving rFFT energy."""
    length = errors.shape[1]
    window = np.hanning(length) if hann else np.ones(length)
    coefficients = np.fft.rfft(errors * window[None, :, None, None], axis=1, norm="ortho")
    multiplicity = np.full(coefficients.shape[1], 2.0)
    multiplicity[0] = 1
    if length % 2 == 0:
        multiplicity[-1] = 1
    power = (np.abs(coefficients) ** 2).sum(-1)
    power *= multiplicity[None, :, None] / length / np.mean(window**2)
    return np.fft.rfftfreq(length, d=0.1), power


def band_values(frequencies, power):
    return np.stack([power[:, mask].sum(1).mean(-1) for mask in masks(frequencies)], axis=-1)


def summary(errors):
    f, power = spectrum(errors)
    energy = band_values(f, power)
    mse = np.square(errors).sum(-1).mean((1, 2))
    identity_error = float(np.max(np.abs(energy.sum(-1) - mse)))
    assert identity_error < 1e-12
    mean_energy = energy.mean(0)
    return {
        "mpjpe_mm": float(np.linalg.norm(errors, axis=-1).mean() * 1000),
        "total_vector_mse": float(mse.mean()),
        "vector_rms": float(np.sqrt(mse.mean())),
        "band_mse": dict(zip(BANDS, mean_energy.tolist())),
        "band_share_percent": dict(zip(BANDS, (mean_energy / mean_energy.sum() * 100).tolist())),
        "frequencies_hz": f.tolist(),
        "bin_mse": power.mean((0, 2)).tolist(),
        "max_parseval_error": identity_error,
    }, energy


def clustered_draws(values, takes, bootstrap_indices):
    unique = sorted(set(takes))
    sums = np.stack([values[np.array(takes) == take].sum(0) for take in unique])
    counts = np.array([takes.count(take) for take in unique])
    selected = sums[bootstrap_indices]
    denominator = counts[bootstrap_indices].sum(1)
    return selected.sum(1) / denominator.reshape((-1,) + (1,) * (values.ndim - 1))


def inspect(errors):
    result, energy = summary(errors)
    centered = errors - errors.mean(2, keepdims=True)
    translation = errors.mean(2, keepdims=True)
    root_relative = errors - errors[:, :, :1]
    result["centroid_relative"], _ = summary(centered)
    result["pelvis_relative"], _ = summary(root_relative)
    translation_mse = float(np.square(translation).sum(-1).mean())
    centered_mse = float(np.square(centered).sum(-1).mean())
    assert abs(translation_mse + centered_mse - result["total_vector_mse"]) < 1e-12
    result["translation_decomposition"] = {
        "centroid_translation_mse_m2": translation_mse,
        "centroid_relative_pose_mse_m2": centered_mse,
        "centroid_translation_share_percent": translation_mse / result["total_vector_mse"] * 100,
        "centroid_relative_pose_share_percent": centered_mse / result["total_vector_mse"] * 100,
        "note": "Centroid decomposition is orthogonal algebra. Pelvis-relative diagnostics are separate, not additive components.",
    }
    result["groups"] = {}
    for name, indices in GROUPS.items():
        item, _ = summary(errors[:, :, indices])
        item["contribution_to_all_joint_mse_percent"] = (
            item["total_vector_mse"] * len(indices) / 22 / result["total_vector_mse"] * 100
        )
        result["groups"][name] = item
    result["per_joint"] = [summary(errors[:, :, [j]])[0] for j in range(22)]
    result["per_axis"] = [summary(errors[..., [axis]])[0] for axis in range(3)]
    result["velocity_error"], _ = summary(np.diff(errors, axis=1) * 10)
    result["acceleration_error"], _ = summary(np.diff(errors, n=2, axis=1) * 100)
    for derivative in ("velocity_error", "acceleration_error"):
        result[derivative]["mean_vector_norm"] = result[derivative].pop("mpjpe_mm") / 1000
    ac = errors - errors.mean(1, keepdims=True)
    f, tapered = spectrum(ac, hann=True)
    ac_energy = band_values(f, tapered).mean(0)
    result["hann_demeaned_ac_check"] = {
        "nonzero_band_share_percent": dict(zip(BANDS[1:], (ac_energy[1:] / ac_energy[1:].sum() * 100).tolist())),
        "note": "Demean each clip/joint, apply Hann, exclude reintroduced DC; robustness check of nonzero-frequency distribution only.",
    }
    time = np.arange(errors.shape[1], dtype=float)
    time -= time.mean()
    trend = (ac * time[None, :, None, None]).sum(1, keepdims=True) / np.square(time).sum() * time[None, :, None, None]
    residual = ac - trend
    trend_mse = float(np.square(trend).sum(-1).mean())
    residual_mse = float(np.square(residual).sum(-1).mean())
    dc = result["band_mse"]["DC"]
    assert abs(dc + trend_mse + residual_mse - result["total_vector_mse"]) < 1e-12
    result["linear_trend_decomposition"] = {
        "constant_share_percent": dc / result["total_vector_mse"] * 100,
        "linear_trend_share_percent": trend_mse / result["total_vector_mse"] * 100,
        "residual_share_percent": residual_mse / result["total_vector_mse"] * 100,
        "note": "Per-joint least-squares projection within each six-second clip, not proof of long-term drift.",
    }
    result["pooled_signed_error_mse_m2"] = {}
    for width in (2, 4, 8, 16, 32, errors.shape[1]):
        pooled = np.lib.stride_tricks.sliding_window_view(errors, width, axis=1).mean(-1)
        result["pooled_signed_error_mse_m2"][str(width)] = float(np.square(pooled).sum(-1).mean())
    result["mature_last40"], _ = summary(errors[:, -40:])
    return result, energy


def figures(output, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "axes.unicode_minus": False})
    labels = ["片段平均偏移", "0–0.5 Hz", "0.5–1 Hz", "1–2.5 Hz", "2.5–5 Hz"]
    colors = ["#9c6c20", "#287a56", "#66ad85", "#3c80b6", "#8952a4"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    names = list(cases)
    bottom = np.zeros(len(names))
    for band, label, color in zip(BANDS, labels, colors):
        values = [cases[name]["band_share_percent"][band] for name in names]
        axes[0, 0].bar(names, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0, 0].set(ylabel="位置平方误差占比（%）", title="四组现有模型：位置误差集中在哪些频段？")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    for name, color in (("baseline", "#627689"), ("velocity", "#cf593c")):
        case = cases[name]
        axes[0, 1].semilogy(
            case["frequencies_hz"][1:], np.array(case["bin_mse"])[1:] * 1e6, "o-", label=name, color=color, ms=3
        )
    axes[0, 1].axvline(1, color="#777777", ls="--", lw=1)
    axes[0, 1].set(
        xlabel="频率（Hz），不含零频", ylabel="每个频率格的位置误差能量（mm²）", title="非零频率：基线与速度监督模型"
    )
    axes[0, 1].legend()
    group_names = ["torso_head", "arms", "legs", "wrists", "feet"]
    groups = cases["velocity"]["groups"]
    bottom = np.zeros(len(group_names))
    for band, label, color in zip(BANDS, labels, colors):
        values = [groups[name]["band_share_percent"][band] for name in group_names]
        axes[1, 0].bar(["躯干与头", "上肢", "下肢", "手腕", "脚尖"], values, bottom=bottom, color=color)
        bottom += values
    axes[1, 0].set(ylabel="各部位自身的平方误差占比（%）", title="速度监督模型：不同部位的误差")
    case = cases["velocity"]
    derivative_cases = [case, case["velocity_error"], case["acceleration_error"]]
    bottom = np.zeros(3)
    for band, label, color in zip(BANDS, labels, colors):
        values = [item["band_share_percent"][band] for item in derivative_cases]
        axes[1, 1].bar(["位置误差", "速度误差", "加速度误差"], values, bottom=bottom, color=color)
        bottom += values
    axes[1, 1].set(ylabel="各自平方误差占比（%）", title="高频位置误差小，仍可能影响速度与加速度")
    fig.savefig(output / "spectrum_overview.png", dpi=160)
    plt.close(fig)
    frequency = np.linspace(0, 5, 501)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    responses = []
    for width in (2, 4, 8):
        response = np.abs(np.exp(-2j * np.pi * frequency[:, None] / 10 * np.arange(width)).mean(-1)) ** 2
        responses.append(response)
        ax.plot(frequency, response, label=f"S={width}", alpha=0.65)
    ax.plot(frequency, np.mean(responses, axis=0), color="black", lw=2.5, label="三个尺度等权平均")
    ax.set(
        xlabel="频率（Hz）",
        ylabel="额外损失对该频率的相对权重",
        title="现有 2/4/8 帧平均误差损失的频率响应",
        ylim=(-0.03, 1.05),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "loss_weight_response.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("exp/error-spectrum-20261111"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    evaluation = workspace / "exp/ac-detach-velocity-C-20261110/evaluation400"
    selection_path = evaluation / "selection.json"
    selection = read(selection_path)
    records = selection["records"]
    assert len(records) == 400 and len({record["take"] for record in records}) == 212
    plan = read(evaluation / "plan.json")
    data_file = Path(plan["data_root"]) / "uniegomotion/ee_val.pt"
    motion = torch.load(data_file, map_location="cpu", weights_only=False, mmap=True)
    targets = []
    for record in records:
        row = motion[record["source"]]
        start = record["start"] + 20
        target = row["kp3d"][start : start + 60, :22].float().clone()
        target[:, :, 2] -= float(row["floor_height"])
        targets.append(target.double().numpy())
    targets = np.stack(targets)
    takes = [record["take"] for record in records]
    rng = np.random.default_rng(1062)
    bootstrap_indices = rng.integers(212, size=(2000, 212))
    result = {
        "created_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": {
            "scope": "Existing saved predictions only; CPU analysis; no new model inference or training changes.",
            "clips": 400,
            "takes": 212,
            "frames_per_clip": 60,
            "motion_fps": 10,
            "score_indices": "20..79 inclusive in each official 80-frame clip",
            "selection_sha256": selection["sha256"],
            "selection_file_sha256": sha(selection_path),
            "script_sha256": sha(Path(__file__)),
            "position_units": "m; vector MSE sums xyz, averages joints/frames/clips",
            "spectrum": "Per-clip orthonormal rFFT, correct doubled interior-bin energy; no clipping, filtering, detrending or tapering in the primary analysis.",
            "bands": "DC; (0,0.5]; (0.5,1]; (1,2.5]; (2.5,5] Hz",
            "frequency_resolution_hz": 10 / 60,
            "aggregation": "Equal clips, with energy ratios computed from pooled mean energies; paired take-cluster bootstrap 2000 draws.",
            "boundaries": "DC is per-clip/joint six-second mean error, not a universal sensor bias or evidence of long-term drift. Energy shares are not additive MPJPE shares or achievable improvements. Fixed-beta predictions can contain shape mismatch.",
            "derivatives": "Actual finite differences of error at 10Hz, transformed separately (59 velocity and 58 acceleration samples); their band fractions are not direct transforms of the 60-frame position spectrum.",
            "hann_check": "Only a nonzero-frequency leakage sensitivity check, not an exact decomposition of original MSE.",
            "pooled_loss_response": "Ideal box-average response; finite overlapping training blocks also have boundary effects.",
        },
        "cases": {},
        "comparisons": {},
        "verification": {},
    }
    all_energy, all_mpjpe, rows = {}, {}, []
    old = read(evaluation / "error-frequency-diagnosis.json")
    old_labels = dict(zip(CASES, ("C_baseline", "B_two_step", "A_two_step_detached", "C_two_step_velocity")))
    max_old_error = 0.0
    for name, relative in CASES.items():
        directory = workspace / "exp" / relative
        report = read(directory / "report.json")
        assert report["identity"]["selection_sha256"] == selection["sha256"]
        errors = []
        for record in records:
            identifier = record["episode_id"]
            filename = hashlib.sha256(identifier.encode()).hexdigest() + ".pt"
            saved = torch.load(
                directory / f"shard{record['shard']}" / "predictions" / filename, map_location="cpu", weights_only=True
            )
            assert saved["episode_id"] == identifier and saved["take"] == record["take"]
            assert saved["selection_sha256"] == selection["sha256"]
            assert torch.equal(saved["score_frame_indices"].cpu(), torch.arange(20, 80))
            timestamps = saved["timestamps"].double().cpu().numpy()
            assert np.max(np.abs(np.diff(timestamps) - 0.1)) < 1e-9
            source = motion[record["source"]]
            expected = (int(source["start_idx"]) + 3 * (record["start"] + np.arange(20, 80))) / 30
            assert np.max(np.abs(timestamps - expected)) < 1e-9
            prediction = saved["joints_floor_world"].double().cpu().numpy()
            assert prediction.shape == (60, 22, 3) and np.isfinite(prediction).all()
            errors.append(prediction)
        errors = np.stack(errors) - targets
        case, energy = inspect(errors)
        case["report"] = str(directory / "report.json")
        case["report_sha256"] = sha(directory / "report.json")
        old_case = old["cases"][old_labels[name]]
        assert abs(case["mpjpe_mm"] - old_case["mpjpe_mm"]) < 2e-4
        coarse = [
            case["band_mse"]["DC"],
            case["band_mse"]["0_to_0p5Hz"] + case["band_mse"]["0p5_to_1Hz"],
            case["band_mse"]["1_to_2p5Hz"],
            case["band_mse"]["2p5_to_5Hz"],
        ]
        old_coarse = [
            old_case[k]
            for k in ("error_DC_mse_m2", "error_0_to_1Hz_mse_m2", "error_1_to_2p5Hz_mse_m2", "error_2p5_to_5Hz_mse_m2")
        ]
        difference = float(np.max(np.abs(np.array(coarse) - old_coarse)))
        assert difference < 1e-9
        max_old_error = max(max_old_error, difference)
        draws = clustered_draws(energy, takes, bootstrap_indices)
        case["band_share_ci95_percent"] = dict(
            zip(BANDS, np.quantile(draws / draws.sum(1, keepdims=True) * 100, [0.025, 0.975], axis=0).T.tolist())
        )
        low_share = energy[:, :3].sum(1) / energy.sum(1) * 100
        take_energy = np.stack([energy[np.array(takes) == take].mean(0) for take in sorted(set(takes))]).mean(0)
        case["cohort_robustness"] = {
            "per_clip_0_to_1Hz_including_DC_share_mean_percent": float(low_share.mean()),
            "per_clip_share_p05_median_p95_percent": np.quantile(low_share, [0.05, 0.5, 0.95]).tolist(),
            "clips_with_low_share_above90_percent": int((low_share > 90).sum()),
            "clips_with_low_share_above80_percent": int((low_share > 80).sum()),
            "take_equal_band_share_percent": dict(zip(BANDS, (take_energy / take_energy.sum() * 100).tolist())),
        }
        all_energy[name] = energy
        mpjpe = np.linalg.norm(errors, axis=-1).mean((1, 2)) * 1000
        all_mpjpe[name] = mpjpe
        for index, record in enumerate(records):
            rows.append(
                {
                    "case": name,
                    "episode_id": record["episode_id"],
                    "take": record["take"],
                    "mpjpe_mm": float(mpjpe[index]),
                    **dict(zip(BANDS, energy[index].tolist())),
                }
            )
        result["cases"][name] = case
        print(
            json.dumps({"case": name, "mpjpe_mm": case["mpjpe_mm"], "band_share_percent": case["band_share_percent"]}),
            flush=True,
        )
    pairs = [(name, "baseline") for name in list(CASES)[1:]]
    pairs += [("velocity", "two_step"), ("velocity", "detached")]
    for name, reference in pairs:
        current = clustered_draws(all_energy[name], takes, bootstrap_indices)
        baseline = clustered_draws(all_energy[reference], takes, bootstrap_indices)
        reduction = (1 - current / baseline) * 100
        point = (1 - all_energy[name].mean(0) / all_energy[reference].mean(0)) * 100
        delta = all_mpjpe[name] - all_mpjpe[reference]
        result["comparisons"][name + "_vs_" + reference] = {
            "band_energy_reduction_percent": dict(zip(BANDS, point.tolist())),
            "band_energy_reduction_ci95_percent": dict(
                zip(BANDS, np.quantile(reduction, [0.025, 0.975], axis=0).T.tolist())
            ),
            "mpjpe_delta_mm": float(delta.mean()),
            "mpjpe_delta_ci95_mm": np.quantile(
                clustered_draws(delta, takes, bootstrap_indices), [0.025, 0.975]
            ).tolist(),
        }
    result["verification"] = {
        "existing_coarse_spectrum_max_difference_m2": max_old_error,
        "all_1600_prediction_ids_and_timestamps_verified": True,
        "parseval_checked_for_all_summary_views": True,
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "per_clip.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(output, result["cases"])
    print(json.dumps({"status": "complete", "output": str(output), "verification": result["verification"]}), flush=True)


if __name__ == "__main__":
    main()
