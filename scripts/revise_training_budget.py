"""Continue a reviewed G prefix with a smaller budget and a late LR decay.

The handoff is on the first new decay boundary. Model, EMA, teacher, Adam
moments, sampling/CPU/CUDA RNG and validation curves are retained exactly.
Only the declared budget and scheduler/LR metadata may change.
"""

import argparse
import copy
import fcntl
import json
import math
import os
import time
from pathlib import Path

import torch

from egorecover.io import file_sha256, save_checkpoint, save_json
from scripts.formal_runtime_migration import assert_same_state
from scripts.run_formal_mainline import continuation_step
from scripts.run_history_ablation import now, read
from scripts.schedule_formal_runtime import start_supervisor, stop_jobs, verified_source


def migrate_budget_checkpoint(saved, *, expected_code, total_updates, decay_step, gamma):
    """Apply the first LR decay after the preserved boundary update."""
    step = saved["step"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (total_updates, decay_step)):
        raise ValueError("Budget and decay boundary must be integer updates.")
    identity, scheduler, optimizer = saved["identity"], saved["scheduler"], saved["optimizer"]
    if identity["code"] != expected_code:
        raise ValueError("Budget revisions require the same reviewed native package code.")
    if not (0 < step == decay_step < total_updates < identity["train_steps"]):
        raise ValueError("Revise on the first new decay boundary, before the smaller total budget.")
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("The LR decay multiplier must be in (0,1).")
    previous = identity["optimization"]["scheduler"]
    if previous["mode"] != "uniegomotion" or previous["decay_step"] <= step:
        raise ValueError("The retained prefix must precede the original first LR decay.")
    if scheduler["last_epoch"] != step or scheduler["step_size"] != previous["decay_step"]:
        raise ValueError("Original scheduler state does not match its declared update boundary.")
    if scheduler["gamma"] != previous["gamma"]:
        raise ValueError("Original scheduler multiplier disagrees with its identity.")
    groups = optimizer["param_groups"]
    bases = scheduler["base_lrs"]
    if len(groups) != len(bases) or len(scheduler["_last_lr"]) != len(groups):
        raise ValueError("Optimizer groups and scheduler LR arrays disagree.")
    for group, base, last in zip(groups, bases, scheduler["_last_lr"]):
        if group["lr"] != base or last != base or group["initial_lr"] != base:
            raise ValueError("The prefix already changed LR; it cannot be reused as the new high-LR prefix.")
    revised_identity = copy.deepcopy(identity)
    revised_identity["train_steps"] = total_updates
    revised_identity["optimization"]["scheduler"] = {**previous, "decay_step": decay_step, "gamma": gamma}
    last_rates = [group["lr"] * gamma for group in groups]
    result = {
        **saved,
        "identity": revised_identity,
        "optimizer": {**optimizer, "param_groups": [{**group, "lr": lr} for group, lr in zip(groups, last_rates)]},
        "scheduler": {**scheduler, "step_size": decay_step, "gamma": gamma, "_last_lr": last_rates},
    }
    assert_retained_training_state(saved, result)
    return result


def assert_retained_training_state(original, revised):
    """Permit exactly the budget, decay configuration and current group LRs."""
    if original["identity"]["code"] != revised["identity"]["code"]:
        raise AssertionError("Native package code changed during a budget-only revision.")
    identity = copy.deepcopy(revised["identity"])
    identity["train_steps"] = original["identity"]["train_steps"]
    for key in ("decay_step", "gamma"):
        identity["optimization"]["scheduler"][key] = original["identity"]["optimization"]["scheduler"][key]
    groups = revised["optimizer"]["param_groups"]
    old_groups = original["optimizer"]["param_groups"]
    if len(groups) != len(old_groups):
        raise AssertionError("Optimizer group count changed.")
    optimizer = {
        **revised["optimizer"],
        "param_groups": [{**group, "lr": old["lr"]} for group, old in zip(groups, old_groups)],
    }
    scheduler = {**revised["scheduler"]}
    for key in ("step_size", "gamma", "_last_lr"):
        scheduler[key] = original["scheduler"][key]
    normalized = {**revised, "identity": identity, "optimizer": optimizer, "scheduler": scheduler}
    assert_same_state(original, normalized)


def complete_training_rows(path):
    """Ignore only an interrupted, uncommitted final log line."""
    raw = Path(path).read_text()
    lines, rows = raw.splitlines(), []
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or raw.endswith("\n"):
                raise
    return rows


def capture_boundary(parent, target, checkpoint_step):
    """Pin an atomically saved parent checkpoint while its peer catches up."""
    if (target / "capture-complete.json").exists():
        if continuation_step(target) != checkpoint_step:
            raise ValueError("Captured boundary checkpoint has the wrong update count.")
        return True
    if (target / "resume.pt").exists():
        raise ValueError("Incomplete boundary capture exists; inspect it before retrying.")
    selection = read(parent / "selection.json") or {}
    if not any(row["step"] == checkpoint_step for row in selection.get("curve", [])):
        return False
    if continuation_step(parent) != checkpoint_step:
        raise ValueError("Requested boundary has already been overwritten; leave parent training untouched.")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("resume.pt", "identity.json", "best.pt", "selection.json", "dev.json", "sampling-first-step.json"):
        if (parent / name).exists():
            os.link(parent / name, target / name)
    rows = [row for row in complete_training_rows(parent / "train.jsonl") if row["step"] <= checkpoint_step]
    if not rows or rows[-1]["step"] != checkpoint_step:
        raise ValueError("Boundary checkpoint is missing its completed update log.")
    (target / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if continuation_step(target) != checkpoint_step:
        raise ValueError("Parent checkpoint changed during capture; no jobs stopped.")
    save_json(target / "capture-complete.json", {"step": checkpoint_step, "at_beijing": now()})
    return True


def install_continuations(old_plan, revised_plan, checkpoint_step, *, captured_root=None):
    """Create new run directories; preserve the complete original run files."""
    old_root, new_root = Path(old_plan["output"]), Path(revised_plan["output"])
    if len(old_plan["jobs"]) != 2 or len(revised_plan["jobs"]) != 2:
        raise ValueError("The budget revision requires both matched window jobs.")
    old_code, new_code = verified_source(old_plan), verified_source(revised_plan)
    if old_code != new_code:
        raise ValueError("Native model/training code must be unchanged for this budget revision.")
    records = []
    for old_job, job in zip(old_plan["jobs"], revised_plan["jobs"]):
        if (old_job["window"], old_job["gpus"]) != (job["window"], job["gpus"]):
            raise ValueError("Window or GPU allocation changed.")
        parent = (old_root if captured_root is None else Path(captured_root)) / old_job["name"]
        output = new_root / job["name"]
        output.mkdir(parents=True, exist_ok=False)
        original = torch.load(parent / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
        if original["step"] != checkpoint_step:
            raise ValueError("Parent checkpoint does not match the common saved handoff step.")
        if original["protocol"]["window"] != job["window"]:
            raise ValueError("Parent checkpoint window disagrees with the destination job.")
        revised = migrate_budget_checkpoint(
            original,
            expected_code=old_code,
            total_updates=revised_plan["standard_updates"],
            decay_step=revised_plan["lr_decay_step"],
            gamma=revised_plan["lr_decay_gamma"],
        )
        save_checkpoint(output / "resume.pt", revised)
        restored = torch.load(output / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
        assert_same_state(revised, restored)
        assert_retained_training_state(original, restored)
        save_json(output / "identity.json", restored["identity"])
        for name in ("best.pt", "selection.json", "dev.json", "sampling-first-step.json"):
            if (parent / name).exists():
                os.link(parent / name, output / name)
        rows = complete_training_rows(parent / "train.jsonl")
        prefix = [row for row in rows if row["step"] <= checkpoint_step]
        if not prefix or prefix[-1]["step"] != checkpoint_step:
            raise ValueError("Parent log is missing the saved boundary update.")
        (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prefix))
        # Progress reports the last completed update's old LR. The migration
        # record separately states the LR to be used at the next update.
        save_json(output / "progress.json", prefix[-1])
        save_json(output / "status.json", {"status": "budget_revision_resume_ready", "step": checkpoint_step})
        record = {
            "parent_case": old_job["name"],
            "new_case": job["name"],
            "handoff_step": checkpoint_step,
            "parent_checkpoint_sha256": file_sha256(parent / "resume.pt"),
            "revised_checkpoint_sha256": file_sha256(output / "resume.pt"),
            "model_EMA_teacher_moments_RNG_curve_bit_exact": True,
            "initial_learning_rate": restored["identity"]["learning_rate"],
            "next_update_learning_rates": [group["lr"] for group in restored["optimizer"]["param_groups"]],
            "scheduler_epoch": restored["scheduler"]["last_epoch"],
            "scheduler_step_count": restored["scheduler"]["_step_count"],
            "parent_unsaved_updates_preserved_only_in_parent_logs": max(
                0, (read(old_root / old_job["name"] / "progress.json") or prefix[-1])["step"] - checkpoint_step
            ),
        }
        save_json(output / "budget-revision.json", record)
        records.append(record)
        del original, revised, restored
    save_json(
        new_root / "preflight-verification.json",
        {
            "at_beijing": now(),
            "type": "native-package-unchanged continuation; scheduler boundary and retained-state verification",
            "parent_preflight": str(old_root / "preflight-verification.json"),
            "checkpoint_migrations": records,
            "scheduler_reference": "tested against an uninterrupted AdamW/StepLR run using the new schedule",
        },
    )
    return records


def wait_and_continue(args):
    torch.set_num_threads(2)
    old, revised = read(args.parent_plan), read(args.revised_plan)
    old_root, new_root = Path(old["output"]), Path(revised["output"])
    if len(old["jobs"]) != 2 or len(revised["jobs"]) != 2:
        raise ValueError("Both window jobs are required.")
    if old_root.resolve() == new_root.resolve():
        raise ValueError("A budget revision requires a new output directory.")
    for parent, job in zip(old["jobs"], revised["jobs"]):
        if any(parent.get(key) != job.get(key) for key in ("window", "gpus", "config", "port")):
            raise ValueError("Window, GPU allocation, config and rendezvous port must be preserved.")
    for key in (
        "data_root",
        "data_digest",
        "smpl_sha256",
        "dev_selection",
        "evaluation_selection",
        "long_selection",
        "velocity_weight",
        "direct_supervised_bodies_per_update",
    ):
        if old.get(key) != revised.get(key):
            raise ValueError(f"Non-budget training/evaluation setting changed: {key}.")
    old_code = verified_source(old)
    if old_code != verified_source(revised):
        raise ValueError("Native package changed; budget revision refused.")
    if revised["training_budgets"] != [revised["standard_updates"]]:
        raise ValueError("The compressed revision must have one budget and no automatic extension.")
    if revised["standard_updates"] != revised["native_checkpoint_total_cap"]:
        raise ValueError("The native maximum must enforce the new total update budget.")
    status_path = new_root / "budget-revision-status.json"
    captured_root = new_root / "parent-boundary"
    while True:
        state = read(old_root / "status.json") or {}
        if state.get("phase") == "failed" or any(
            job.get("exit_code") not in (None, 0) for job in state.get("jobs", [])
        ):
            raise RuntimeError("Parent training failed before the common handoff checkpoint.")
        steps, ready = {}, True
        for job in old["jobs"]:
            directory = old_root / job["name"]
            progress = read(directory / "progress.json") or {}
            steps[job["name"]] = progress.get("step")
            ready &= capture_boundary(directory, captured_root / job["name"], args.checkpoint_step)
        waiting = {
            "phase": "waiting_for_parent_checkpoint",
            "at_beijing": now(),
            "checkpoint_step": args.checkpoint_step,
            "parent_steps": steps,
            "new_total_updates": revised["standard_updates"],
            "new_learning_rates": revised["learning_rates"],
        }
        save_json(status_path, waiting)
        save_json(new_root / "status.json", waiting)
        if ready:
            break
        time.sleep(5)
    if read(args.parent_plan) != old:
        raise ValueError("Parent plan changed while waiting; do not interrupt its jobs.")
    if any(continuation_step(captured_root / job["name"]) != args.checkpoint_step for job in old["jobs"]):
        raise ValueError("The exact common handoff checkpoint is unavailable; no jobs stopped.")
    verified_source(old)
    verified_source(revised)
    # Validate recoverability before interrupting the source processes.
    for job in old["jobs"]:
        saved = torch.load(captured_root / job["name"] / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
        migrate_budget_checkpoint(
            saved,
            expected_code=old_code,
            total_updates=revised["standard_updates"],
            decay_step=revised["lr_decay_step"],
            gamma=revised["lr_decay_gamma"],
        )
        del saved
    stop_jobs(old, state)
    records = install_continuations(old, revised, args.checkpoint_step, captured_root=captured_root)
    save_json(
        old_root / "status.json",
        {
            "phase": "continued_under_budget_revision",
            "at_beijing": now(),
            "new_plan": str(args.revised_plan),
            "new_output": str(new_root),
            "parent_checkpoint_step": args.checkpoint_step,
        },
    )
    process = start_supervisor(revised, args.revised_plan, "budget12k")
    save_json(
        status_path,
        {"phase": "restoring_budget12k_continuations", "at_beijing": now(), "pid": process.pid, "records": records},
    )
    deadline = time.monotonic() + 1200
    while True:
        if process.poll() is not None:
            raise RuntimeError("Compressed-budget supervisor exited; original checkpoints remain intact.")
        progress = {job["name"]: read(new_root / job["name"] / "progress.json") for job in revised["jobs"]}
        if all(row and row["step"] > args.checkpoint_step for row in progress.values()):
            for row in progress.values():
                if not math.isclose(row["learning_rate"], 3e-6, rel_tol=1e-12):
                    raise ValueError("A resumed training job did not use the intended low learning rate.")
            result = {
                "phase": "budget12k_training_running",
                "at_beijing": now(),
                "total_updates": revised["standard_updates"],
                "auto_extension": False,
                "steps": {name: row["step"] for name, row in progress.items()},
                "learning_rate": 3e-6,
                "records": records,
                "fully_converged_claim": False,
            }
            save_json(status_path, result)
            report_path = Path(revised["workspace"]) / "docs/reports/data/formal_budget12k_revision_20261112.json"
            report = read(report_path) or {}
            report["execution"] = result
            save_json(report_path, report)
            return
        if time.monotonic() > deadline:
            raise RuntimeError("Compressed-budget continuation did not resume within the startup limit.")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--revised-plan", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    args = parser.parse_args()
    status_path = Path(read(args.revised_plan)["output"]) / "budget-revision-status.json"
    with (status_path.parent / "budget-revision.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_and_continue(args)
        except BaseException as error:
            save_json(
                status_path,
                {
                    "phase": "budget_revision_needs_attention",
                    "at_beijing": now(),
                    "error": repr(error),
                    "parent_checkpoints_preserved": True,
                },
            )
            raise


if __name__ == "__main__":
    main()
