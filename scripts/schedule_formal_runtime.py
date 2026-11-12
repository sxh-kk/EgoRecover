"""Switch both formal jobs at a saved checkpoint, retaining every tensor state."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

import torch

from egorecover.io import file_sha256, save_checkpoint, save_json
from scripts.formal_runtime_migration import PACKED_PACKAGE_CHANGES, assert_same_state, migrate_checkpoint
from scripts.run_formal_mainline import continuation_step, environment
from scripts.run_history_ablation import now, read


def verified_source(plan):
    source = Path(plan["source_snapshot"])
    for relative, digest in plan["source_hashes"].items():
        if file_sha256(source / relative) != digest:
            raise ValueError(f"Frozen source changed: {relative}.")
    return {str(p.relative_to(source / "egorecover")): file_sha256(p) for p in (source / "egorecover").rglob("*.py")}


def assert_owned_process(pid, fragment):
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if fragment not in command:
        raise ValueError(f"Process {pid} is not the expected formal job.")


def timestamp_sample(directory):
    path = directory / "progress.json"
    row = read(path)
    return {"step": row["step"], "mtime_ns": path.stat().st_mtime_ns} if row else None


def rate(samples):
    if len(samples) < 2 or samples[-1]["step"] <= samples[0]["step"]:
        raise ValueError("At least two advancing update timestamps are required.")
    updates = samples[-1]["step"] - samples[0]["step"]
    return {
        "first_step": samples[0]["step"],
        "last_step": samples[-1]["step"],
        "updates": updates,
        "seconds_per_update": (samples[-1]["mtime_ns"] - samples[0]["mtime_ns"]) / 1e9 / updates,
    }


def start_supervisor(plan, path, suffix):
    root = Path(plan["output"])
    log_path = root / f"supervisor-{suffix}.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [plan["python"], "-u", "-m", "scripts.run_formal_mainline", "supervise", "--plan", str(path)],
            cwd=plan["source_snapshot"],
            env=environment(plan, "0,1,2,3,4,5,6,7"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save_json(
        root / "supervisor-process.json",
        {
            "pid": process.pid,
            "at_beijing": now(),
            "log": str(log_path),
            "source_snapshot": plan["source_snapshot"],
            "runtime_revision": suffix,
        },
    )
    return process


def stop_jobs(plan, state):
    root = Path(plan["output"])
    supervisor = read(root / "supervisor-process.json")
    assert_owned_process(supervisor["pid"], "scripts.run_formal_mainline supervise")
    assert_owned_process(supervisor["pid"], str(root / "plan.json"))
    for job in state["jobs"]:
        assert_owned_process(job["pid"], str(root / job["case"]))
    os.kill(supervisor["pid"], signal.SIGTERM)
    for job in state["jobs"]:
        os.killpg(job["pid"], signal.SIGTERM)
    for _ in range(100):
        alive = []
        for job in state["jobs"]:
            path = Path(f"/proc/{job['pid']}/stat")
            if path.exists() and path.read_text().split()[2] != "Z":
                alive.append(job["pid"])
        if not alive:
            return
        time.sleep(0.2)
    raise RuntimeError(f"Original job processes have not stopped: {alive}.")


def upgrade(plan, prepared, checkpoint_step, journal):
    """Original files remain hard-linked in the journal until explicitly cleaned."""
    root = Path(plan["output"])
    old_code, new_code = verified_source(plan), verified_source(prepared)
    changed = {key for key in old_code.keys() | new_code.keys() if old_code.get(key) != new_code.get(key)}
    if changed != PACKED_PACKAGE_CHANGES:
        raise ValueError(f"Unexpected runtime3 changes: {sorted(changed)}.")
    journal.mkdir(parents=True, exist_ok=False)
    staged, records = [], []
    for job in plan["jobs"]:
        directory, archive = root / job["name"], journal / job["name"]
        archive.mkdir()
        for name in (
            "identity.json",
            "resume.pt",
            "best.pt",
            "train.jsonl",
            "progress.json",
            "status.json",
            "selection.json",
            "dev.json",
        ):
            if (directory / name).is_file():
                os.link(directory / name, archive / name)
        original = torch.load(archive / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
        if original["step"] != checkpoint_step:
            raise ValueError("Saved optimizer step differs from the scheduled common checkpoint.")
        migrated = migrate_checkpoint(original, original_code=old_code, reviewed_code=new_code, revision="runtime3")
        stage = directory / "runtime3-staged.pt"
        save_checkpoint(stage, migrated)
        restored = torch.load(stage, weights_only=True, map_location="cpu", mmap=True)
        assert_same_state(original, restored)
        records.append(
            {
                "case": job["name"],
                "resume_step": checkpoint_step,
                "old_checkpoint_sha256": file_sha256(archive / "resume.pt"),
                "new_checkpoint_sha256": file_sha256(stage),
                "all_other_values_bit_exact": True,
            }
        )
        staged.append((directory, stage, migrated["identity"]))
        del original, migrated, restored
    for directory, stage, identity in staged:
        stage.replace(directory / "resume.pt")
        save_json(directory / "identity.json", identity)
        rows = [
            json.loads(line) for line in (journal / directory.name / "train.jsonl").read_text().splitlines() if line
        ]
        rows = [row for row in rows if row["step"] <= checkpoint_step]
        temp = directory / "train.jsonl.tmp"
        temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temp.replace(directory / "train.jsonl")
        save_json(directory / "progress.json", rows[-1])
        save_json(directory / "status.json", {"status": "runtime3_resume_ready", "step": checkpoint_step})
    save_json(
        journal / "checkpoint-migration.json",
        {"at_beijing": now(), "records": records, "changed_files": sorted(changed)},
    )
    return records


def restore_archived_jobs(plan, journal, checkpoint_step):
    """Restore the original recovery state if the new runtime cannot start."""
    root = Path(plan["output"])
    for job in plan["jobs"]:
        directory, archive = root / job["name"], journal / job["name"]
        failed = journal / (job["name"] + "-failed-runtime3")
        failed.mkdir(exist_ok=True)
        for name in (
            "identity.json",
            "resume.pt",
            "best.pt",
            "train.jsonl",
            "progress.json",
            "status.json",
            "selection.json",
            "dev.json",
        ):
            if (directory / name).exists() and not (failed / name).exists():
                os.link(directory / name, failed / name)
            if (archive / name).exists():
                staged = directory / (name + ".rollback")
                os.link(archive / name, staged)
                staged.replace(directory / name)
        saved = torch.load(directory / "resume.pt", weights_only=True, map_location="cpu", mmap=True)
        if saved["step"] != checkpoint_step:
            raise ValueError("Rollback checkpoint does not match the common saved step.")
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line]
        rows = [row for row in rows if row["step"] <= checkpoint_step]
        temp = directory / "train.jsonl.tmp"
        temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        temp.replace(directory / "train.jsonl")
        save_json(directory / "progress.json", rows[-1])
        save_json(directory / "status.json", {"status": "runtime2_rollback_ready", "step": checkpoint_step})


def run(args):
    torch.set_num_threads(2)
    old, prepared = read(args.plan), read(args.prepared_plan)
    root = Path(old["output"])
    assert old["runtime_revision"] == "runtime2" and prepared["runtime_revision"] == "runtime3"
    for key in old.keys() - {
        "source_snapshot",
        "source_hashes",
        "paired_helper",
        "runtime_revision",
        "runtime_upgrade",
    }:
        if prepared[key] != old[key]:
            raise ValueError(f"Non-runtime plan field changed: {key}.")
    verified_source(old)
    verified_source(prepared)
    traces = {job["name"]: deque(maxlen=100) for job in old["jobs"]}
    status_path = root / "runtime-upgrade-status.json"
    while True:
        state = read(root / "status.json")
        if state and any(job["exit_code"] not in (None, 0) for job in state["jobs"]):
            raise RuntimeError("A source training job failed before the scheduled switch.")
        ready = True
        for job in old["jobs"]:
            directory = root / job["name"]
            row = timestamp_sample(directory)
            trace = traces[job["name"]]
            if row and (not trace or trace[-1]["step"] != row["step"]):
                trace.append(row)
            selection = read(directory / "selection.json") or {}
            ready &= any(item["step"] == args.checkpoint_step for item in selection.get("curve", []))
        save_json(
            status_path,
            {
                "phase": "waiting_for_common_checkpoint",
                "at_beijing": now(),
                "checkpoint_step": args.checkpoint_step,
                "steps": {key: value[-1]["step"] if value else None for key, value in traces.items()},
                "prepared_source": prepared["source_snapshot"],
            },
        )
        if ready:
            break
        time.sleep(5)
    baseline = {
        name: rate([v for v in samples if args.checkpoint_step - 50 <= v["step"] < args.checkpoint_step - 5])
        for name, samples in traces.items()
    }
    if read(args.plan) != old:
        raise ValueError("Active plan changed while waiting; leave running jobs untouched.")
    if any(continuation_step(root / job["name"]) != args.checkpoint_step for job in old["jobs"]):
        raise ValueError("Expected common saved checkpoint is unavailable; leave running jobs untouched.")
    verified_source(old)
    verified_source(prepared)
    journal = root / "runtime-upgrade-20261112" / f"runtime2-at{args.checkpoint_step}"
    stop_jobs(old, state)
    # Archive controller provenance before replacing any active plan.
    controller_backup = root / f"runtime2-controller-at{args.checkpoint_step}"
    controller_backup.mkdir()
    for name in ("plan.json", "supervisor-process.json", "status.json", "supervisor-runtime2.log"):
        if (root / name).exists():
            os.link(root / name, controller_backup / name)
    process, training_verified = None, False
    try:
        records = upgrade(old, prepared, args.checkpoint_step, journal)
        save_json(args.plan, prepared)
        process = start_supervisor(prepared, args.plan, "runtime3")
        save_json(
            status_path,
            {
                "phase": "warming_up_runtime3",
                "at_beijing": now(),
                "resume_step": args.checkpoint_step,
                "pid": process.pid,
                "migration": records,
            },
        )
        deadline = time.monotonic() + 1200
        while True:
            if process.poll() is not None:
                raise RuntimeError("Runtime3 supervisor exited during startup.")
            current = {job["name"]: timestamp_sample(root / job["name"]) for job in prepared["jobs"]}
            if all(row["step"] >= args.checkpoint_step + 10 for row in current.values()):
                training_verified = True
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Runtime3 did not resume updates within the startup limit.")
            time.sleep(5)
        after = {job["name"]: [] for job in prepared["jobs"]}
        started = time.monotonic()
        while time.monotonic() - started < 90:
            for name, samples in after.items():
                row = timestamp_sample(root / name)
                if not samples or row["step"] != samples[-1]["step"]:
                    samples.append(row)
            time.sleep(0.2)
        comparisons = []
        for name, samples in after.items():
            updated = rate(samples)
            a, b = updated["seconds_per_update"], baseline[name]["seconds_per_update"]
            comparisons.append(
                {
                    "case": name,
                    "before": baseline[name],
                    "after": updated,
                    "speed_ratio": b / a,
                    "elapsed_reduction_percent": 100 * (1 - a / b),
                }
            )
        result = {
            "phase": "runtime3_training_measured",
            "at_beijing": now(),
            "resume_step": args.checkpoint_step,
            "production_timing": comparisons,
            "checkpoint_migration": records,
            "FP32": True,
            "training_attention_and_supervision_unchanged": True,
        }
        save_json(status_path, result)
        save_json(Path(old["workspace"]) / "exp/speedup-20261112/production-runtime3.json", result)
        report_path = Path(old["workspace"]) / "docs/reports/data/training_speedup_20261112.json"
        report = read(report_path) or {}
        report["runtime3"] = result
        report["status"] = "runtime3_training_measured"
        report["at_beijing"] = now()
        save_json(report_path, report)
    except BaseException as error:
        if not training_verified:
            # Stop only this task's new supervisor/workers, then restore the
            # original code identity and all checkpoint tensors. The old
            # implementation resumes instead of leaving both jobs stopped.
            new_state = read(root / "status.json") or {}
            for job in new_state.get("jobs", []):
                try:
                    assert_owned_process(job["pid"], str(root / job["case"]))
                    os.killpg(job["pid"], signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            time.sleep(2)
            restore_archived_jobs(old, journal, args.checkpoint_step)
            fallback = {**old, "runtime_revision": "runtime2-fallback"}
            save_json(args.plan, fallback)
            resumed = start_supervisor(fallback, args.plan, "runtime2-fallback")
            save_json(
                status_path,
                {
                    "phase": "runtime3_failed_restored_runtime2",
                    "at_beijing": now(),
                    "error": repr(error),
                    "checkpoint_step": args.checkpoint_step,
                    "fallback_pid": resumed.pid,
                    "checkpoint_backup": str(journal),
                },
            )
            raise
        save_json(
            status_path,
            {
                "phase": "runtime3_upgrade_failed",
                "at_beijing": now(),
                "error": repr(error),
                "checkpoint_backup": str(journal),
                "controller_backup": str(controller_backup),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared-plan", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    args = parser.parse_args()
    with (args.plan.parent / "runtime-upgrade.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
