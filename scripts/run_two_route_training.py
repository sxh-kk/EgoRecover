"""Frozen-P two-step BPTT versus one-step paired supervision, then fixed400."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

snapshot = Path(__file__).resolve().parent / "source"
if snapshot.is_dir():
    sys.path.insert(0, str(snapshot))

from scripts.run_history_ablation import (  # noqa: E402 -- frozen runner adds its snapshot before importing helpers
    environment,
    evaluation_worker,
    load_module,
    monitor,
    now,
    read,
    run_processes,
    train_command,
    write,
)


def route_command(plan, job, output, *, preflight=False, resume=False):
    return train_command(plan, job, output, preflight=preflight, resume=resume) + [
        "--unroll-steps",
        str(job["unroll_steps"]),
        "--feedback-intervention",
        "body_pose",
        "--isolate-forward-rng",
    ]


def verify_training(plan, *, preflight=False):
    import torch

    from egorecover.io import code_identity, file_sha256, save_checkpoint, state_sha256

    root = Path(plan["output"])
    original = torch.load(plan["common_G_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    signatures, results = [], {}
    for job in plan["jobs"]:
        directory = root / "preflight" / job["name"] if preflight else root / job["name"]
        identity, initial = read(directory / "identity.json"), read(directory / "initialization.json")
        assert initial["checkpoint_sha256"] == plan["common_G_checkpoint_sha256"]
        assert initial["loaded_model_sha256"] == plan["common_G_model_sha256"]
        assert initial["checkpoint_step"] == 2000 and initial["optimizer"] == "fresh"
        assert identity["prior"] == plan["prior_checkpoint_sha256"] and identity["code"] == code_identity()
        assert identity["supervision"]["unroll_steps"] == job["unroll_steps"]
        assert identity["supervision"]["feedback_intervention"] == "body_pose"
        assert identity["supervision"]["isolate_forward_rng"]
        budget = 2 if preflight else 2000
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert [row["step"] for row in rows] == list(range(1, budget + 1))
        signatures.append([row["sampling_sha256"] for row in rows])
        for row in rows:
            assert row["effective_batch_size"] == 64 and row["world_size"] == 4
            assert row["supervised_current_frames_seen"] == row["step"] * 64 * job["unroll_steps"]
            assert row["rollin_frames_counts"]["20"] == 48
        saved = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True, mmap=True)
        assert saved["step"] == budget and len(saved["rank_rng_states"]) == 4
        assert len({r["model_sha256"] for r in saved["rank_rng_states"]}) == 1
        assert all(int(value["step"]) == budget for value in saved["optimizer"]["state"].values())
        assert state_sha256(saved["feedback_model"]) == plan["common_G_model_sha256"]
        for key, value in original["ema"].items():
            torch.testing.assert_close(saved["feedback_model"][key], value, atol=0, rtol=0)
        if not preflight:
            keys = ("format", "model_config", "identity", "protocol", "stats", "ema", "step", "best")
            save_checkpoint(
                directory / "final-ema.pt", {**{k: saved[k] for k in keys}, "purpose": "fixed2000_step_evaluation"}
            )
        results[job["name"]] = {
            "updates": budget,
            "supervised_current_frames": budget * 64 * job["unroll_steps"],
            "sampling_sha256": hashlib.sha256(json.dumps(signatures[-1]).encode()).hexdigest(),
            "teacher_unchanged": True,
            "native_rank_weights_synchronized": True,
        }
        del saved
    assert signatures[0] == signatures[1]
    assert file_sha256(plan["prior_checkpoint"]) == plan["prior_checkpoint_sha256"]
    write(
        root / ("preflight-verification.json" if preflight else "training-verification.json"),
        {
            "status": "passed",
            "at_beijing": now(),
            "same_query_tau_epsilon_and_initial_current_outputs": True,
            "jobs": results,
        },
    )


def collect_evaluation(plan):
    from egorecover.evaluation import summarize

    root = Path(plan["output"]) / "evaluation400"
    shared = load_module(root / "shared_worker.py")
    selection = read(root / "selection.json")
    assert len(selection["records"]) == 400 and selection["takes"] == 212
    groups, results = {}, []
    labels = [(job["name"], root / job["name"], 2000) for job in plan["jobs"]]
    labels.insert(0, ("C_baseline", Path(plan["baseline_root"]), 0))
    for label, directory, updates in labels:
        rows = [
            read(directory / f"shard{r['shard']}/rows" / (shared.name(r["episode_id"]) + ".json"))
            for r in selection["records"]
        ]
        if any(row is None or row["frames"] != 60 for row in rows):
            raise ValueError("Incomplete fixed400 evaluation.")
        assert [r["episode"] for r in rows] == [r["episode_id"] for r in selection["records"]]
        identity = read(directory / "shard0/identity.json")
        assert identity["selection_sha256"] == selection["sha256"]
        groups[label] = rows
        results.append(
            {
                "case": label,
                "additional_updates": updates,
                "clips": 400,
                "takes": 212,
                **{metric: sum(r["comparison"][metric] for r in rows) / 400 for metric in shared.KEYS},
                "mature_metrics": {
                    metric: sum(r["mature_comparison"][metric] for r in rows) / 400 for metric in shared.KEYS[:10]
                },
            }
        )
        if label != "C_baseline":
            write(
                directory / "report.json",
                {
                    "identity": identity,
                    "results": {"1": {"rows": {"rollout": rows}, "summary": {"rollout": summarize(rows)}}},
                },
            )
    effects = []
    a, b = (job["name"] for job in plan["jobs"])
    for subset, key in (("all60", "comparison"), ("mature40", "mature_comparison")):
        for metric in (shared.KEYS if key == "comparison" else shared.KEYS[:10]):
            for left, right in ((a, "C_baseline"), (b, "C_baseline"), (b, a)):
                values = [(x["take"], y[key][metric] - x[key][metric]) for x, y in zip(groups[right], groups[left])]
                effects.append(
                    {
                        "contrast": f"{left}_minus_{right}",
                        "subset": subset,
                        "metric": metric,
                        **shared.paired_cluster(values),
                    }
                )
    report = {
        "status": "complete",
        "at_beijing": now(),
        "results": results,
        "paired_effects": effects,
        "selection_sha256": selection["sha256"],
        "protocol": plan["evaluation_protocol"],
        "comparison_boundary": "Route screening: same128000 anchors; two-step has256000 directly supervised frames versus128000 for paired. C reference receives no additional training.",
        "timing": "Parallel quality evaluation; no isolated latency claim",
    }
    write(root / "comparison.json", report)
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case", "additional_updates", "clips", "takes", *shared.KEYS], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(results)


def supervise(plan):
    root = Path(plan["output"])
    try:
        if not (root / "feedback-calibration.json").exists():
            write(root / "status.json", {"phase": "gradient_calibration", "at_beijing": now()})
            with (root / "calibration.log").open("a") as log:
                subprocess.run(
                    [plan["python"], "-m", "scripts.calibrate_feedback", "--plan", str(root / "plan.json")],
                    cwd=plan["source_snapshot"],
                    env=environment(plan, "0"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if not (root / "preflight-verification.json").exists():
            for resume in (False, True):
                specs = [
                    (
                        job["name"],
                        route_command(plan, job, root / "preflight" / job["name"], preflight=True, resume=resume),
                        root / "preflight" / job["name"],
                        job["gpus"],
                    )
                    for job in plan["jobs"]
                ]
                run_processes(plan, specs, phase="preflight-resume" if resume else "preflight-start")
            verify_training(plan, preflight=True)
            # These two newly-created optimizer snapshots served only the native
            # 1->2-step resume check; retain their lightweight weights and logs.
            removed = []
            for job in plan["jobs"]:
                path = root / "preflight" / job["name"] / "resume.pt"
                removed.append({"path": str(path), "bytes": path.stat().st_size, "verified_step": 2})
                path.unlink()
            write(
                root / "preflight-temporary-cleanup.json",
                {"removed": removed, "kept": "best.pt, logs, identity and verification"},
            )
        specs = []
        for job in plan["jobs"]:
            directory = root / job["name"]
            specs.append(
                (
                    job["name"],
                    route_command(plan, job, directory, resume=(directory / "resume.pt").exists()),
                    directory,
                    job["gpus"],
                )
            )
        run_processes(plan, specs, phase="training")
        verify_training(plan)
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
