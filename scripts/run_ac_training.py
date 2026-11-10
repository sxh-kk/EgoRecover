"""Matched two-step detach and GT-velocity controls, with the existing B run."""

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

snapshot = Path(__file__).resolve().parent / "source"
if snapshot.is_dir():
    sys.path.insert(0, str(snapshot))

from scripts.run_history_ablation import (  # noqa: E402
    environment,
    evaluation_worker,
    load_module,
    monitor,
    now,
    read,
    run_processes,
    write,
)
from scripts.run_two_route_training import route_command, verify_training  # noqa: E402


def ac_command(plan, job, output, *, preflight=False, resume=False):
    command = route_command(plan, job, output, preflight=preflight, resume=resume)
    command += ["--velocity-weight", str(job["velocity_weight"]), "--velocity-scale-m-s", "1.0"]
    if job["detach_between"]:
        command.append("--detach-between-steps")
    return command


def check_space(plan, *, preflight=False):
    root = Path(plan["output"])
    checkpoint, light = plan["expected_resume_bytes"], plan["expected_light_bytes"]
    directories = [root / "preflight" / job["name"] if preflight else root / job["name"] for job in plan["jobs"]]
    missing = sum(
        max(0, size - (path.stat().st_size if path.exists() else 0))
        for directory in directories
        for path, size in ((directory / "resume.pt", checkpoint), (directory / "best.pt", light))
    )
    # Both DDP jobs may replace their full checkpoint simultaneously.
    needed = missing + 2 * checkpoint + 512 * 1024**2
    free = shutil.disk_usage(root).free
    write(root / "space-check.json", {"at_beijing": now(), "free_bytes": free, "required_additional_bytes": needed})
    if free < needed:
        raise RuntimeError(
            "Insufficient project-disk space for simultaneous atomic checkpoints; previous artifacts preserved."
        )


def verify_against_b(plan, *, preflight=False):
    root = Path(plan["output"])
    baseline = Path(plan["B_training_root"])
    budget = 2 if preflight else 2000
    b_rows = [json.loads(line) for line in (baseline / "train.jsonl").read_text().splitlines()][:budget]
    evidence = {}
    for job in plan["jobs"]:
        directory = root / "preflight" / job["name"] if preflight else root / job["name"]
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        supervision = read(directory / "identity.json")["supervision"]
        assert supervision["detach_between_steps"] == job["detach_between"]
        assert supervision["velocity_weight"] == job["velocity_weight"]
        assert [r["sampling_sha256"] for r in rows] == [r["sampling_sha256"] for r in b_rows]
        first = rows[0]
        for component in ("flow", "current", "hand_pca", "feedback"):
            assert abs(first[component] - b_rows[0][component]) < 1e-8
        expected = b_rows[0]["loss"] + job["velocity_weight"] * first["velocity"]
        assert abs(first["loss"] - expected) < 1e-7
        assert read(directory / "sampling-first-step.json") == read(baseline / "sampling-first-step.json")
        evidence[job["name"]] = {
            "matched_B_sampling_steps": budget,
            "first_base_losses_match_B": True,
            "first_current_codes_match_B": True,
            "first_velocity_loss": first["velocity"],
            "detach_between_steps": job["detach_between"],
            "velocity_weight": job["velocity_weight"],
        }
    native = root / ("preflight-verification.json" if preflight else "training-verification.json")
    path = root / ("matched-preflight-verification.json" if preflight else "matched-training-verification.json")
    write(path, {**read(native), "matched_existing_B": evidence})


def collect_evaluation(plan):
    from egorecover.evaluation import summarize
    from egorecover.io import file_sha256

    root = Path(plan["output"]) / "evaluation400"
    shared = load_module(root / "shared_worker.py")
    selection = read(root / "selection.json")
    assert len(selection["records"]) == 400 and selection["takes"] == 212
    references = [
        ("C_baseline", Path(plan["baseline_root"]), 0),
        ("B_two_step", Path(plan["B_evaluation_root"]), 2000),
    ]
    cases = references + [(job["name"], root / job["name"], 2000) for job in plan["jobs"]]
    groups, results = {}, []
    for label, directory, updates in cases:
        rows = [
            read(directory / f"shard{r['shard']}/rows" / (shared.name(r["episode_id"]) + ".json"))
            for r in selection["records"]
        ]
        assert all(row is not None and row["frames"] == 60 for row in rows)
        assert [r["episode"] for r in rows] == [r["episode_id"] for r in selection["records"]]
        assert [r["take"] for r in rows] == [r["take"] for r in selection["records"]]
        identity = read(directory / "shard0/identity.json")
        assert identity["selection_sha256"] == selection["sha256"]
        assert identity["prior_checkpoint_sha256"] == plan["prior_checkpoint_sha256"]
        assert identity["dataset_digest"] == plan["manifest"]
        groups[label] = rows
        results.append(
            {
                "case": label,
                "additional_updates": updates,
                "clips": 400,
                "takes": 212,
                **{key: sum(row["comparison"][key] for row in rows) / 400 for key in shared.KEYS},
                "mature_metrics": {
                    key: sum(row["mature_comparison"][key] for row in rows) / 400 for key in shared.KEYS[:10]
                },
            }
        )
        if label not in {r[0] for r in references}:
            write(
                directory / "report.json",
                {
                    "identity": identity,
                    "results": {"1": {"rows": {"rollout": rows}, "summary": {"rollout": summarize(rows)}}},
                },
            )
    effects = []
    for subset, key in (("all60", "comparison"), ("mature40", "mature_comparison")):
        for metric in (shared.KEYS if key == "comparison" else shared.KEYS[:10]):
            for right, left in itertools.combinations(groups, 2):
                values = [(a["take"], b[key][metric] - a[key][metric]) for a, b in zip(groups[right], groups[left])]
                effects.append(
                    {
                        "contrast": f"{left}_minus_{right}",
                        "subset": subset,
                        "metric": metric,
                        **shared.paired_cluster(values),
                    }
                )
    assert file_sha256(Path(plan["B_training_root"]) / "final-ema.pt") == plan["B_checkpoint_sha256"]
    assert file_sha256(plan["common_G_checkpoint"]) == plan["common_G_checkpoint_sha256"]
    assert file_sha256(plan["prior_checkpoint"]) == plan["prior_checkpoint_sha256"]
    write(
        root / "comparison.json",
        {
            "status": "complete",
            "at_beijing": now(),
            "results": results,
            "paired_effects": effects,
            "selection_sha256": selection["sha256"],
            "protocol": plan["evaluation_protocol"],
            "comparison_boundary": "A/B/C: identical256000 directly supervised frames and2000 updates; B-A isolates cache gradients; C-B isolates emitted velocity supervision. C_baseline has no extra updates. Single-seed development comparison.",
            "timing": "parallel quality evaluation; no isolated latency benchmark",
        },
    )
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case", "additional_updates", "clips", "takes", *shared.KEYS], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(results)


def supervise(plan):
    root = Path(plan["output"])
    try:
        if not (root / "velocity-calibration.json").exists():
            write(root / "status.json", {"phase": "velocity_gradient_calibration", "at_beijing": now()})
            with (root / "calibration.log").open("a") as log:
                subprocess.run(
                    [plan["python"], "-m", "scripts.calibrate_velocity", "--plan", str(root / "plan.json")],
                    cwd=plan["source_snapshot"],
                    env=environment(plan, "0"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        weight = read(root / "velocity-calibration.json")["weight"]
        for job in plan["jobs"]:
            job["velocity_weight"] = weight if job["velocity_supervision"] else 0.0
        write(root / "plan.json", plan)
        if not (root / "matched-preflight-verification.json").exists():
            check_space(plan, preflight=True)
            for resume in (False, True):
                specs = [
                    (
                        job["name"],
                        ac_command(plan, job, root / "preflight" / job["name"], preflight=True, resume=resume),
                        root / "preflight" / job["name"],
                        job["gpus"],
                    )
                    for job in plan["jobs"]
                ]
                run_processes(plan, specs, phase="preflight-resume" if resume else "preflight-start")
            verify_training(plan, preflight=True)
            verify_against_b(plan, preflight=True)
            removed = []
            for job in plan["jobs"]:
                for filename in ("resume.pt", "best.pt"):
                    path = root / "preflight" / job["name"] / filename
                    removed.append({"path": str(path), "bytes": path.stat().st_size, "verified_step": 2})
                    path.unlink()
            write(
                root / "preflight-temporary-cleanup.json",
                {
                    "removed": removed,
                    "kept": "all logs, initialization, identity, sampling and native resume verification; formal and old checkpoints preserved",
                },
            )
        check_space(plan)
        specs = [
            (
                job["name"],
                ac_command(plan, job, root / job["name"], resume=(root / job["name"] / "resume.pt").exists()),
                root / job["name"],
                job["gpus"],
            )
            for job in plan["jobs"]
        ]
        run_processes(plan, specs, phase="training")
        verify_training(plan)
        verify_against_b(plan)
        specs = []
        for job in plan["jobs"]:
            for shard, gpu in enumerate(job["gpus"].split(",")):
                directory = root / "evaluation400" / job["name"] / f"shard{shard}"
                command = [
                    plan["python"],
                    str(Path(__file__).resolve()),
                    "evaluation-worker",
                    "--plan",
                    str(root / "plan.json"),
                    "--case",
                    job["name"],
                    "--shard",
                    str(shard),
                ]
                specs.append((f"{job['name']}/shard{shard}", command, directory, gpu))
        run_processes(plan, specs, phase="evaluation400")
        collect_evaluation(plan)
        write(root / "status.json", {"phase": "complete", "at_beijing": now()})
    except Exception as error:
        write(root / "status.json", {"phase": "failed", "at_beijing": now(), "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("supervise", "monitor", "evaluation-worker", "collect"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    plan = read(args.plan)
    if args.action == "evaluation-worker":
        evaluation_worker(plan, args.case, args.shard)
    else:
        {"supervise": supervise, "monitor": monitor, "collect": collect_evaluation}[args.action](plan)


if __name__ == "__main__":
    main()
