"""Queue two matched shape-aware controls after the existing SIMPLE queue."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from scripts.run_history_ablation import evaluation_worker, load_module, monitor, now, read, run_processes, write


def training_command(plan, job, output, *, preflight=False, resume=False):
    command = [
        plan["python"],
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port",
        str(job["port"]),
        "--module",
        "egorecover",
        "train",
        "--config",
        job["config"],
        "--data-root",
        plan["data_root"],
        "--data-format",
        "official",
        "--prior-mode",
        "learned",
        "--prior-checkpoint",
        plan["prior_checkpoint"],
        "--device",
        "cuda",
        "--seed",
        "62",
        "--train-steps",
        "2" if preflight else str(plan["training_steps"]),
        "--batch-size",
        "16",
        "--microbatch-size",
        "2",
        "--sample-reference-world-size",
        "2",
        "--sampling-rollin-frames",
        "20",
        "--rollin-frames",
        "20",
        "--history-mode",
        "mixed",
        "--history-startup-fraction",
        "0.25",
        "--learning-rate",
        "3e-5",
        "--lr-schedule",
        "uniegomotion",
        "--eval-every",
        "1" if preflight else "500",
        "--dev-limit",
        "2" if preflight else "128",
        "--dev-seed",
        "1062",
        "--unroll-steps",
        "8",
        "--detach-between-steps",
        "--isolate-forward-rng",
        "--velocity-weight",
        str(plan["velocity_weight"]),
        "--slow-weight",
        str(job["slow_weight"]),
        "--output",
        str(output),
    ]
    command += ["--resume"] if resume else ["--initialize", plan["common_G_checkpoint"]]
    if preflight and not resume:
        command += ["--stop-after", "1"]
    return command


def queue_evidence(plan):
    """Do not claim brief gaps between SIMPLE's three jobs as availability."""
    simple = read(plan["simple_queue_status"]) or {}
    queue_done = simple.get("status") == "all_three_models_completed" and simple.get("completed_tasks") == 3
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    devices = [[int(field.strip()) for field in line.split(",")] for line in result.stdout.splitlines() if line]
    idle = len(devices) == 8 and all(memory < 1024 and utilization < 10 for _, memory, utilization in devices)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    return {
        "simple_status": simple.get("status"),
        "simple_active_profile": simple.get("active_profile"),
        "simple_completed_tasks": len(simple.get("tasks", {})),
        "simple_queue_complete": queue_done,
        "gpus": devices,
        "compute_processes": apps,
        "ready": queue_done and idle and not apps,
    }


def verify_assets(plan):
    from egorecover.io import code_identity, file_sha256

    if code_identity() != plan["source_code_identity"]:
        raise ValueError("Frozen training source changed.")
    for key in ("common_G_checkpoint", "prior_checkpoint", "selection_file", "smpl_asset"):
        if file_sha256(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Recorded asset changed: {key}")
    for job in plan["jobs"]:
        if file_sha256(job["config"]) != job["config_sha256"]:
            raise ValueError("A controlled experiment configuration changed.")


def space_check(plan):
    # Two complete resumable checkpoints, two light best/final pairs, and two
    # simultaneous atomic replacements; evaluation trajectories have headroom.
    needed = 9 * 1024**3
    free = shutil.disk_usage(plan["output"]).free
    evidence = {"free_bytes": free, "required_free_bytes": needed, "passed": free >= needed}
    write(Path(plan["output"]) / "space-check.json", evidence)
    if free < needed:
        raise RuntimeError(
            "Insufficient project-disk headroom for two resumable runs; old formal checkpoints preserved."
        )


def verify_training(plan, *, preflight=False):
    import torch

    from egorecover.io import file_sha256, save_checkpoint, state_sha256

    root = Path(plan["output"])
    budget = 2 if preflight else plan["training_steps"]
    signatures, first_rows, evidence = [], [], {}
    for job in plan["jobs"]:
        directory = root / "preflight" / job["name"] if preflight else root / job["name"]
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(1, budget + 1)):
            raise ValueError("Incomplete native training log.")
        initial, identity = read(directory / "initialization.json"), read(directory / "identity.json")
        assert initial["checkpoint_sha256"] == plan["common_G_checkpoint_sha256"]
        assert initial["loaded_model_sha256"] == plan["common_G_model_sha256"]
        assert identity["prior"] == plan["prior_checkpoint_sha256"]
        assert identity["protocol"]["shape_mode"] == "predicted"
        assert identity["protocol"]["geometry_target"] == "official"
        assert identity["supervision"]["persistent_error"]["weight"] == job["slow_weight"]
        assert all(row["world_size"] == 4 and row["effective_batch_size"] == 16 for row in rows)
        assert all(row["supervised_current_frames_seen"] == row["step"] * 128 for row in rows)
        assert all(row["rollin_frames_counts"]["20"] == 12 for row in rows)
        saved = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True, mmap=True)
        assert saved["step"] == budget and len(saved["rank_rng_states"]) == 4
        assert len({state["model_sha256"] for state in saved["rank_rng_states"]}) == 1
        assert len({state["ema_sha256"] for state in saved["rank_rng_states"]}) == 1
        assert all(int(value["step"]) == budget for value in saved["optimizer"]["state"].values())
        assert state_sha256(saved["feedback_model"]) == plan["common_G_model_sha256"]
        if not preflight:
            keys = ("format", "model_config", "identity", "protocol", "stats", "ema", "step", "best")
            save_checkpoint(
                directory / "final-ema.pt",
                {**{key: saved[key] for key in keys}, "purpose": "fixed2000_step_evaluation"},
            )
        signatures.append([row["sampling_sha256"] for row in rows])
        first_rows.append(rows[0])
        evidence[job["name"]] = {
            "updates": budget,
            "supervised_frames": budget * 128,
            "ema_sha256": state_sha256(saved["ema"]),
        }
        del saved
    assert signatures[0] == signatures[1]
    for key in ("flow", "current", "hand_pca", "velocity", "slow"):
        assert abs(first_rows[0][key] - first_rows[1][key]) < 1e-6
    assert (
        abs(first_rows[1]["loss"] - first_rows[0]["loss"] - plan["jobs"][1]["slow_weight"] * first_rows[1]["slow"])
        < 1e-6
    )
    a, b = (root / "preflight" / job["name"] if preflight else root / job["name"] for job in plan["jobs"])
    assert read(a / "sampling-first-step.json") == read(b / "sampling-first-step.json")
    assert file_sha256(plan["prior_checkpoint"]) == plan["prior_checkpoint_sha256"]
    write(
        root / ("preflight-verification.json" if preflight else "training-verification.json"),
        {
            "status": "passed",
            "at_beijing": now(),
            "all_queries_and_noise_matched": True,
            "first_base_losses_match": True,
            "P_unchanged": True,
            "jobs": evidence,
        },
    )


def collect(plan):
    root = Path(plan["output"]) / "evaluation400"
    shared = load_module(root / "shared_worker.py")
    selection = read(root / "selection.json")
    groups, results, effects = {}, [], []
    for job in plan["jobs"]:
        rows = [
            read(
                root / job["name"] / f"shard{record['shard']}" / "rows" / (shared.name(record["episode_id"]) + ".json")
            )
            for record in selection["records"]
        ]
        assert len(rows) == 400 and all(row is not None and row["frames"] == 60 for row in rows)
        assert [row["episode"] for row in rows] == [record["episode_id"] for record in selection["records"]]
        identities = [read(root / job["name"] / f"shard{shard}" / "identity.json") for shard in range(4)]
        assert all(value["selection_sha256"] == selection["sha256"] for value in identities)
        groups[job["name"]] = rows
        results.append(
            {
                "case": job["name"],
                "clips": 400,
                "takes": 212,
                **{metric: sum(row["comparison"][metric] for row in rows) / 400 for metric in shared.KEYS},
            }
        )
    first, second = (job["name"] for job in plan["jobs"])
    for metric in shared.KEYS:
        values = [
            (a["take"], b["comparison"][metric] - a["comparison"][metric])
            for a, b in zip(groups[first], groups[second])
        ]
        effects.append({"contrast": f"{second}_minus_{first}", "metric": metric, **shared.paired_cluster(values)})
    write(
        root / "comparison.json",
        {
            "status": "complete",
            "at_beijing": now(),
            "results": results,
            "paired_effects": effects,
            "selection_sha256": selection["sha256"],
            "protocol": plan["evaluation_protocol"],
            "boundary": "Single-seed development comparison; both groups have identical256000 supervised bodies; only persistent-error weight differs.",
        },
    )


def supervise(plan):
    root = Path(plan["output"])
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_assets(plan)
            stable = 0
            while stable < 3:
                evidence = queue_evidence(plan)
                stable = stable + 1 if evidence["ready"] else 0
                write(
                    root / "status.json",
                    {
                        "phase": "waiting_for_SIMPLE_and_eight_idle_gpus",
                        "at_beijing": now(),
                        "queue_pid": os.getpid(),
                        "stable_idle_polls": stable,
                        **evidence,
                    },
                )
                if stable < 3:
                    time.sleep(30)
            verify_assets(plan)
            space_check(plan)
            if not (root / "preflight-verification.json").exists():
                for resume in (False, True):
                    specs = [
                        (
                            job["name"],
                            training_command(
                                plan, job, root / "preflight" / job["name"], preflight=True, resume=resume
                            ),
                            root / "preflight" / job["name"],
                            job["gpus"],
                        )
                        for job in plan["jobs"]
                    ]
                    run_processes(plan, specs, phase="preflight-resume" if resume else "preflight-start")
                verify_training(plan, preflight=True)
                for job in plan["jobs"]:
                    for name in ("resume.pt", "best.pt"):
                        (root / "preflight" / job["name"] / name).unlink()
            space_check(plan)
            specs = [
                (
                    job["name"],
                    training_command(plan, job, root / job["name"], resume=(root / job["name"] / "resume.pt").exists()),
                    root / job["name"],
                    job["gpus"],
                )
                for job in plan["jobs"]
            ]
            run_processes(plan, specs, phase="training")
            verify_training(plan)
            specs = []
            for job in plan["jobs"]:
                for shard, gpu in enumerate(job["gpus"].split(",")):
                    directory = root / "evaluation400" / job["name"] / f"shard{shard}"
                    command = [
                        plan["python"],
                        "-m",
                        "scripts.run_shape_persistent",
                        "--plan",
                        str(root / "plan.json"),
                        "--worker",
                        job["name"],
                        "--shard",
                        str(shard),
                    ]
                    specs.append((f"{job['name']}-{shard}", command, directory, gpu))
            run_processes(plan, specs, phase="evaluation400")
            collect(plan)
            write(root / "status.json", {"phase": "complete", "at_beijing": now()})
        except BaseException as error:
            write(
                root / "status.json",
                {
                    "phase": "failed_preserved",
                    "at_beijing": now(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    if args.worker:
        evaluation_worker(plan, args.worker, args.shard)
    elif args.monitor:
        monitor(plan)
    else:
        supervise(plan)


if __name__ == "__main__":
    main()
