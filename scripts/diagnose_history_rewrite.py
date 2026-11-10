"""Frozen-G W=3 rewrite diagnostics on a preselected official EE4D cohort.

The worker changes only the inference revise_recent flag. P's provenance flags
follow the actual cache policy, including its effect on startup body coverage.
Training and original experiment artifacts are never modified.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
from history_rewrite_metrics import association, diagnostic_metrics, paired_cluster

QUALITY = (
    "mpjpe_world_mm",
    "pa_mpjpe_mm",
    "velocity_error_mm_s",
    "acceleration_error_mm_s2",
    "foot_slide_uniegomotion_mm",
    "root_final_initial_translation_mm",
)
TEMPORAL = QUALITY[2:5]
POLICIES = ("on", "off")
SOURCES = ("gaussian", "prediction")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identifier_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Recorder:
    def __init__(self, system):
        self.system = system
        self.windows, self.incoming = [], []

    def __getattr__(self, name):
        return getattr(self.system, name)

    def online(self, model, cache, observations, epsilon, *, steps=None):
        from egorecover.state import state_map

        result = self.system.online(model, cache, observations, epsilon, steps=steps)
        self.windows.append(state_map(result.body, lambda x: x[0].detach().cpu()))
        self.incoming.append(state_map(cache.body, lambda x: x[0, -2:].detach().cpu()))
        return result


def decode_recorded(states, smpl, beta, width):
    import torch

    from egorecover.fk import FixedShapeFK
    from egorecover.state import state_cat, state_map

    joined = state_cat(states, dim=0)
    decoder = FixedShapeFK(smpl, beta.reshape(10))
    device = beta.device
    decoded = []
    with torch.no_grad():
        for start in range(0, len(joined.reference), 32):
            batch = state_map(joined, lambda x: x[start : start + 32].to(device))
            decoded.append(decoder(batch).cpu())
    return torch.cat(decoded).reshape(len(states), width, 22, 3)


def validate_cache_positions(emitted, windows, incoming, rewrite):
    """Physical lag1 must be last emitted; lag2 is rewritten only under ON."""
    if not np.allclose(incoming[1:, 1], emitted[:-1], atol=2e-5, rtol=0):
        raise RuntimeError("Previous emitted current was not preserved in incoming cache.")
    expected = windows[1:-1, 1] if rewrite else emitted[:-2]
    if not np.allclose(incoming[2:, 0], expected, atol=2e-5, rtol=0):
        raise RuntimeError("Lag2 cache does not follow the requested rewrite policy.")


def worker(args, plan):
    sys.path.insert(0, plan["source_snapshot"])
    import torch

    from egorecover.evaluate import load_evaluation_context
    from egorecover.evaluation import evaluate_episode
    from egorecover.fk import joint_decoder
    from egorecover.io import code_identity, file_sha256
    from egorecover.metrics import comparison_metrics
    from egorecover.system import WindowReconstructor

    torch.set_num_threads(2)
    selection = read(plan["selection_file"])
    records = selection["records"][: args.limit] if args.limit else selection["records"]
    records = [record for index, record in enumerate(records) if index % args.shards == args.shard]
    output = args.output / args.source / args.policy / f"shard{args.shard}"
    progress = {
        "phase": "initializing",
        "source": args.source,
        "policy": args.policy,
        "shard": args.shard,
        "completed_clips": 0,
        "target_clips": len(records),
        "at_beijing": now(),
    }
    write(output / "progress.json", progress)
    job = plan["jobs"][args.source]
    options = SimpleNamespace(
        checkpoint=Path(job["checkpoint"]),
        data_root=Path(plan["data_root"]),
        prior_checkpoint=Path(plan["prior_checkpoint"]),
        split="val",
        device="cuda",
        synthetic=False,
        data_format="official",
        prior_mode="learned",
        limit=None,
    )
    if code_identity() != plan["source_code_identity"]:
        raise RuntimeError("Frozen evaluation source changed.")
    if file_sha256(options.checkpoint) != job["checkpoint_sha256"]:
        raise RuntimeError("G checkpoint changed.")
    if file_sha256(options.prior_checkpoint) != plan["prior_checkpoint_sha256"]:
        raise RuntimeError("Frozen P checkpoint changed.")
    context = load_evaluation_context(options)
    original = context.system.config
    if (
        original.window != 3
        or original.source != args.source
        or original.current_weight != 0.03
        or original.steps != 1
        or not original.revise_recent
        or context.saved["step"] != 12000
    ):
        raise RuntimeError("Unexpected checkpoint controls.")
    if not context.data.inventory["official_reference_verified"]:
        raise RuntimeError("Official cohort verification failed.")
    if context.data.inventory["selected_clip_candidates"] != selection["official_candidate_count"]:
        raise RuntimeError("Official candidate ordering changed.")
    for record in selection["records"]:
        actual = context.data.records[record["record_index"]]
        if actual["episode_id"] != record["episode_id"] or actual["num_frames"] != 80:
            raise RuntimeError("Selected clip changed.")
    rewrite = args.policy == "on"
    cfg = replace(original, revise_recent=rewrite)
    context.system = WindowReconstructor(context.system.codec, context.system.predictor, cfg)
    context.model.requires_grad_(False).eval()
    if context.system.predictor.training or any(p.requires_grad for p in context.system.predictor.parameters()):
        raise RuntimeError("P must remain frozen and deterministic.")
    baseline = read(job["baseline_report"])
    if baseline["identity"]["checkpoint_sha256"] != job["checkpoint_sha256"]:
        raise RuntimeError("Baseline uses a different G checkpoint.")
    if baseline["identity"]["prior_checkpoint_sha256"] != plan["prior_checkpoint_sha256"]:
        raise RuntimeError("Baseline uses a different P.")
    if baseline["identity"]["selection_sha256"] != selection["sha256"]:
        raise RuntimeError("Baseline uses a different cohort.")
    baseline_rows = {r["episode"]: r for r in baseline["results"]["1"]["rows"]["rollout"]}
    write(
        output / "identity.json",
        {
            "source": args.source,
            "rewrite": rewrite,
            "checkpoint_sha256": job["checkpoint_sha256"],
            "prior_checkpoint_sha256": plan["prior_checkpoint_sha256"],
            "selection_sha256": selection["sha256"],
            "original_protocol": original.to_dict(),
            "inference_protocol": cfg.to_dict(),
            "source_code_identity": code_identity(),
            "diagnostic_script_sha256": file_sha256(__file__),
            "diagnostic_metrics_sha256": file_sha256(Path(__file__).with_name("history_rewrite_metrics.py")),
            "seed": plan["seed"],
            "note": plan["interpretation"],
        },
    )
    for record in records:
        progress.update(phase="evaluating", current_clip=record["episode_id"], at_beijing=now())
        write(output / "progress.json", progress)
        episode = context.data.load(record["record_index"])
        if len(episode) != 80:
            raise RuntimeError("Only full80-frame clips are allowed.")
        recorder = Recorder(context.system)
        row = evaluate_episode(
            recorder,
            context.model,
            episode,
            joint_decoder(episode, False, context.smpl),
            seed=plan["seed"],
            steps=1,
            matched_cache=False,
            comparison=True,
        )
        emitted = row.pop("predictions")[0]
        rotations = row.pop("body_rotations")
        windows = decode_recorded(recorder.windows, context.smpl, episode.beta, 3)
        incoming = decode_recorded(recorder.incoming, context.smpl, episode.beta, 2)
        truth = episode.truth.joints[20:, :, :3, 3].detach().cpu()
        timestamps = episode.timestamps[20:].detach().cpu()
        terminal_roundoff = float((windows[:, -1] - emitted).abs().max())
        if terminal_roundoff > 2e-5:
            raise RuntimeError("Recorded window and native emitted FK differ.")
        windows[:, -1] = emitted
        validate_cache_positions(emitted.numpy(), windows.numpy(), incoming.numpy(), rewrite)
        summary, traces = diagnostic_metrics(
            emitted.numpy(), truth.numpy(), windows.numpy(), incoming.numpy(), timestamps.numpy(), rewrite=rewrite
        )
        native_pairs = (
            ("first_emission_velocity_error_mm_s", "velocity_error_mm_s"),
            ("first_emission_acceleration_error_mm_s2", "acceleration_error_mm_s2"),
            ("first_emission_foot_slide_mm", "foot_slide_uniegomotion_mm"),
        )
        for diagnostic, native in native_pairs:
            if abs(summary[diagnostic] - row["comparison"][native]) > 1e-6:
                raise RuntimeError(f"Diagnostic metric differs from native {native}.")
        old = baseline_rows[episode.identifier]
        old_file = Path(job["baseline_report"]).parent.parent / old["prediction_file"]
        old_emitted = torch.load(old_file, map_location="cpu", weights_only=True)["joints_floor_world"]
        first_difference = float((emitted[0] - old_emitted[0]).abs().max())
        if first_difference > 2e-6:
            raise RuntimeError("Changing history writes changed the shared first emission.")
        audit = {
            "first_frame_max_abs_position_difference_m": first_difference,
            "batched_FK_terminal_roundoff_m": terminal_roundoff,
            "cache_policy_verified": True,
        }
        if rewrite:
            audit["baseline_max_abs_position_difference_m"] = float((emitted - old_emitted).abs().max())
            audit["baseline_metric_absolute_differences"] = {
                key: abs(row["comparison"][key] - old["comparison"][key]) for key in QUALITY
            }
            if audit["baseline_max_abs_position_difference_m"] > 2e-6 or any(
                value > 0.05 for value in audit["baseline_metric_absolute_differences"].values()
            ):
                raise RuntimeError("Rewrite-ON failed to reproduce the original400 evaluation.")
        row["rewrite_diagnostics"] = summary
        row["mature_comparison"] = comparison_metrics(emitted[20:], truth[20:], timestamps[20:])
        row["source"], row["rewrite_policy"] = args.source, args.policy
        row["audit"] = audit
        target = output / "traces" / (identifier_hash(episode.identifier) + ".pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episode_id": episode.identifier,
            "take": episode.take,
            "source": args.source,
            "rewrite": rewrite,
            "frame_indices": torch.arange(20, 80),
            "past_slot_frame_indices": torch.arange(20, 80)[:, None] - torch.tensor([2, 1]),
            "first_emission_joints": emitted,
            "candidate_window_joints": windows,
            "incoming_cached_past_joints": incoming,
            "truth_joints": truth,
            "body_rotations": rotations,
            "timestamps": timestamps,
            "traces": {key: torch.from_numpy(value) for key, value in traces.items()},
            "selection_sha256": selection["sha256"],
        }
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        row["trace_file"] = str(target.relative_to(args.output))
        write(output / "rows" / (identifier_hash(episode.identifier) + ".json"), row)
        progress.update(completed_clips=progress["completed_clips"] + 1, at_beijing=now())
        write(output / "progress.json", progress)
        print(json.dumps({key: progress[key] for key in ("source", "policy", "completed_clips", "target_clips")}))
    progress.update(phase="complete", current_clip=None, at_beijing=now())
    write(output / "progress.json", progress)


def rows_for(output, source, policy, records, shards):
    rows = []
    for index, record in enumerate(records):
        path = output / source / policy / f"shard{index % shards}" / "rows"
        row = read(path / (identifier_hash(record["episode_id"]) + ".json"))
        if row is None or row["episode"] != record["episode_id"] or row["frames"] != 60:
            raise RuntimeError("Incomplete or mismatched clip coverage.")
        if not row["audit"]["cache_policy_verified"]:
            raise RuntimeError("Unverified cache behavior.")
        rows.append(row)
    return rows


def collect(plan, output, *, shards=2, limit=None):
    records = read(plan["selection_file"])["records"]
    records = records[:limit] if limit else records
    grouped = {
        f"{source}_{policy}": rows_for(output, source, policy, records, shards)
        for source in SOURCES
        for policy in POLICIES
    }
    table, paired, associations, quartiles = [], [], {}, {}
    for label, rows in grouped.items():
        take_names = sorted({row["take"] for row in rows})
        table.append(
            {
                "case": label,
                "clips": len(rows),
                "takes": len(take_names),
                **{key: float(np.mean([row["comparison"][key] for row in rows])) for key in QUALITY},
                "mature_metrics": {
                    key: float(np.mean([row["mature_comparison"][key] for row in rows])) for key in QUALITY
                },
                "take_equal_metrics": {
                    key: float(
                        np.mean(
                            [
                                np.mean([row["comparison"][key] for row in rows if row["take"] == take])
                                for take in take_names
                            ]
                        )
                    )
                    for key in QUALITY
                },
                "rewrite_diagnostics": {
                    key: float(np.mean([row["rewrite_diagnostics"][key] for row in rows]))
                    for key in rows[0]["rewrite_diagnostics"]
                },
            }
        )
        write(output / label / "report.json", {"rows": rows, "result": table[-1]})
    for source in SOURCES:
        on, off = grouped[f"{source}_on"], grouped[f"{source}_off"]
        takes = [row["take"] for row in on]
        if any(a["episode"] != b["episode"] for a, b in zip(on, off)):
            raise RuntimeError("Unpaired ON/OFF cohorts.")
        for subset, key in (("all60", "comparison"), ("mature40", "mature_comparison")):
            for metric in QUALITY:
                values = [b[key][metric] - a[key][metric] for a, b in zip(on, off)]
                paired.append({"source": source, "subset": subset, "metric": metric, **paired_cluster(values, takes)})
        if limit:
            continue
        controls = np.array(
            [
                [
                    row["rewrite_diagnostics"][key]
                    for key in ("gt_motion_speed_mm_s", "gt_motion_acceleration_mm_s2", "gt_contact_fraction")
                ]
                for row in on
            ]
        )
        associations[source] = {}
        x = np.array([r["rewrite_diagnostics"]["candidate_cache_update_lag1_body_mm"] for r in on])
        feet = np.array([r["rewrite_diagnostics"]["candidate_cache_update_feet_xy_mm"] for r in on])
        for metric in TEMPORAL:
            predictor = feet if "foot_slide" in metric else x
            values = np.array([r["comparison"][metric] for r in on])
            associations[source][metric] = association(predictor, values, takes, controls)
            associations[source][metric]["correction_measure"] = (
                "candidate_cache_update_feet_xy_mm" if "foot_slide" in metric else "candidate_cache_update_lag1_body_mm"
            )
            off_delta = np.array([b["comparison"][metric] - a["comparison"][metric] for a, b in zip(on, off)])
            associations[source][metric]["correlation_with_OFF_minus_ON"] = association(
                predictor, off_delta, takes, draws=2000
            )
        ordered = np.argsort(x, kind="stable")
        quartiles[source] = []
        for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
            quartiles[source].append(
                {
                    "quartile": quartile,
                    "clips": len(indices),
                    "mean_lag1_rewrite_mm": float(x[indices].mean()),
                    "metrics": {
                        metric: {
                            "on": float(np.mean([on[i]["comparison"][metric] for i in indices])),
                            "off": float(np.mean([off[i]["comparison"][metric] for i in indices])),
                            **paired_cluster(
                                [off[i]["comparison"][metric] - on[i]["comparison"][metric] for i in indices],
                                [takes[i] for i in indices],
                            ),
                        }
                        for metric in TEMPORAL
                    },
                }
            )
    report = {
        "status": "complete",
        "generated_at_beijing": now(),
        "protocol": plan["protocol"],
        "interpretation": plan["interpretation"],
        "selection_sha256": read(plan["selection_file"])["sha256"],
        "results": table,
        "paired_OFF_minus_ON": paired,
        "correlations_rewrite_ON": associations,
        "ON_rewrite_magnitude_quartiles": quartiles,
        "uncertainty": "paired take-cluster bootstrap2000; fixed checkpoints; no training-seed uncertainty",
        "baseline_reproduction": {
            source: {
                "max_abs_position_difference_m": max(
                    row["audit"]["baseline_max_abs_position_difference_m"] for row in grouped[f"{source}_on"]
                ),
                "max_metric_differences": {
                    key: max(
                        row["audit"]["baseline_metric_absolute_differences"][key] for row in grouped[f"{source}_on"]
                    )
                    for key in QUALITY
                },
            }
            for source in SOURCES
        },
        "timing_note": "Instrumentation and parallel GPU workers; no latency claim from this pass.",
    }
    write(output / "comparison.json", report)
    fields = ["case", "clips", "takes", *QUALITY]
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    if not limit:
        plot_results(output, grouped, report)
    return report


def plot_results(output, grouped, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gaussian": "#1f77b4", "prediction": "#d95f02"}
    labels = ["Velocity error (mm/s)", "Acceleration error (mm/s²)", "Foot slide (mm)"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for axis, metric, label in zip(axes, TEMPORAL, labels):
        for source in SOURCES:
            rows = grouped[f"{source}_on"]
            key = (
                "candidate_cache_update_feet_xy_mm" if "foot_slide" in metric else "candidate_cache_update_lag1_body_mm"
            )
            x = [r["rewrite_diagnostics"][key] for r in rows]
            y = [r["comparison"][metric] for r in rows]
            rho = report["correlations_rewrite_ON"][source][metric]["spearman"]
            axis.scatter(x, y, s=13, alpha=0.35, color=colors[source], label=f"{source}: Spearman {rho:.2f}")
        axis.set_xlabel("Mean foot XY correction (mm)" if "foot_slide" in metric else "Mean lag1 correction (mm)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Rewrite ON: corrections vs first-emission errors (400 clips; association, not causality)")
    figure.savefig(output / "rewrite-correlations.png", dpi=180)
    figure.savefig(output / "rewrite-correlations.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, metric, label in zip(axes, TEMPORAL, labels):
        for index, source in enumerate(SOURCES):
            entry = next(
                x
                for x in report["paired_OFF_minus_ON"]
                if x["source"] == source and x["subset"] == "all60" and x["metric"] == metric
            )
            lo, hi = entry["ci95"]
            axis.bar(index, entry["delta"], color=colors[source], alpha=0.8)
            axis.errorbar(
                index, entry["delta"], yerr=[[entry["delta"] - lo], [hi - entry["delta"]]], color="black", capsize=5
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks([0, 1], SOURCES)
        axis.set_ylabel("OFF minus ON: " + label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Disable history writes: paired changes (negative = better; 95% take-cluster CI)")
    figure.savefig(output / "rewrite-ablation.png", dpi=180)
    figure.savefig(output / "rewrite-ablation.pdf")
    plt.close(figure)


def run_stage(plan, plan_path, output, *, preflight):
    shards, limit = (1, 2) if preflight else (2, None)
    jobs = []
    try:
        for source in SOURCES:
            for policy in POLICIES:
                for shard in range(shards):
                    gpu = len(jobs)
                    directory = output / source / policy / f"shard{shard}"
                    directory.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--plan",
                        str(plan_path),
                        "--output",
                        str(output),
                        "--source",
                        source,
                        "--policy",
                        policy,
                        "--shard",
                        str(shard),
                        "--shards",
                        str(shards),
                    ]
                    if limit:
                        command += ["--limit", str(limit)]
                    env = os.environ.copy()
                    env.update(
                        CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS="2",
                        PYTHONUNBUFFERED="1",
                        SMPLX_MODEL_PATH=plan["smpl_directory"],
                    )
                    log = (directory / "worker.log").open("x")
                    process = subprocess.Popen(
                        command,
                        cwd=plan["workspace"],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    jobs.append(
                        {
                            "source": source,
                            "policy": policy,
                            "shard": shard,
                            "gpu": gpu,
                            "process": process,
                            "log": log,
                            "directory": directory,
                        }
                    )
        while True:
            states = [
                {
                    "source": j["source"],
                    "policy": j["policy"],
                    "shard": j["shard"],
                    "gpu": j["gpu"],
                    "pid": j["process"].pid,
                    "exit_code": j["process"].poll(),
                    "progress": read(j["directory"] / "progress.json"),
                }
                for j in jobs
            ]
            write(
                plan_path.parent / "status.json",
                {
                    "phase": "preflight" if preflight else "evaluation400",
                    "at_beijing": now(),
                    "jobs": states,
                    "completed_clips": sum((s["progress"] or {}).get("completed_clips", 0) for s in states),
                    "target_clips": 8 if preflight else 1600,
                },
            )
            if any(s["exit_code"] not in (None, 0) for s in states):
                raise RuntimeError("A diagnostic worker failed; inspect its preserved worker.log.")
            if all(s["exit_code"] == 0 for s in states):
                break
            time.sleep(10)
        report = collect(plan, output, shards=shards, limit=limit)
        if preflight:
            write(
                plan_path.parent / "preflight-verification.json",
                {
                    "status": "passed",
                    "at_beijing": now(),
                    "cases": 4,
                    "clips_per_case": 2,
                    "baseline_reproduction": report["baseline_reproduction"],
                    "same_first_emission_ON_OFF": True,
                    "native_temporal_metric_parity": True,
                    "actual_cache_policy_and_physical_lag_alignment": True,
                },
            )
    finally:
        for job in jobs:
            if job["process"].poll() is None:
                job["process"].terminate()
            job["log"].close()


def monitor(root):
    status = read(root / "status.json")
    print(json.dumps(status, ensure_ascii=False, indent=2) if status else "No status yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supervise", "worker", "collect", "monitor"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    plan = read(args.plan)
    args.output = args.output or args.plan.parent
    if args.command == "worker":
        worker(args, plan)
    elif args.command == "collect":
        collect(plan, args.output, shards=args.shards, limit=args.limit)
    elif args.command == "monitor":
        monitor(args.plan.parent)
    else:
        try:
            if not (args.plan.parent / "preflight-verification.json").exists():
                run_stage(plan, args.plan, args.plan.parent / "preflight", preflight=True)
            run_stage(plan, args.plan, args.plan.parent, preflight=False)
            write(
                args.plan.parent / "status.json",
                {
                    "phase": "complete",
                    "at_beijing": now(),
                    "completed_clips": 1600,
                    "target_clips": 1600,
                    "comparison": str(args.plan.parent / "comparison.json"),
                },
            )
        except Exception as exc:
            write(args.plan.parent / "status.json", {"phase": "failed", "at_beijing": now(), "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
