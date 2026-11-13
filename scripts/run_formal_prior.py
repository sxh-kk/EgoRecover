"""Supervise frozen-history P preparation, four-rank preflight and gated training."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scripts.formal_prior_processes import (
    ROOT_ENV,
    TOKEN_ENV,
    SupervisorInterrupted,
    bind_job,
    cleanup_owned_jobs,
    finish_job,
    journal_job,
    termination_signals,
)
from scripts.run_history_ablation import now, read, write

HEALTH_CHECKS = {
    "initial_P_equals_CV",
    "beta_FK_gradient",
    "finite_objective_and_gradients",
    "exact_phase_quota",
    "causal_inputs_test",
    "producer_identity",
    "train_val_take_disjoint",
    "gradient_diagnostics_finite",
}

BANK_KEYS = {
    "train": "train_bank",
    "D128": "val_bank",
    "preflight_train": "preflight_train_bank",
    "preflight_D128": "preflight_val_bank",
}


def validate_protocol_family(protocol):
    run_id = protocol.get("run_id", "")
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("The P run_id must be a single output-directory name.")
    if protocol["producer"]["window"] != 3:
        raise ValueError("The runtime must bind the reviewed W3 history producer.")
    if run_id == "P01_W3G10k_s62":
        return "P01"
    if protocol.get("experiment_family") != "P02_common_cv_ab" or protocol.get("case") not in ("A", "B"):
        raise ValueError("The runtime requires a reviewed P01 or P02 A/B protocol.")
    expected_hold = protocol["case"] == "B"
    if (
        protocol["loss"].get("objective") != "common_cv_valid_state"
        or protocol["model"].get("hold_beta", False) is not expected_hold
        or protocol["loss"].get("fixed_geometry_weight") != 0.008869197811349056
    ):
        raise ValueError("P02 objective, beta policy or fixed geometry coefficient changed.")
    if not set(BANK_KEYS) <= protocol.get("existing_banks", {}).keys():
        raise ValueError("P02 must reuse the frozen formal and preflight history banks.")
    return "P02"


def verify_existing_banks(plan):
    """Bind reusable histories to content, not a mutable filename or marker."""
    for name, item in plan.get("existing_banks", {}).items():
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"Incomplete existing-bank identity: {name}")
        path = Path(item["path"])
        if not path.is_absolute():
            raise ValueError(f"Existing-bank path must be absolute: {name}")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Existing-bank identity mismatch: {name}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bank_stages(plan, cohort, directory, *, limit=None):
    """Run the reviewed producer on disjoint shards, then merge on the CPU."""
    gpu_ids = plan.get("bank_gpus", plan["training_gpus"]).split(",")
    shards, directory = len(gpu_ids), Path(directory)
    workers = []
    for shard, gpu in enumerate(gpu_ids):
        out = directory / f"shard{shard:04d}"
        argv = [
            plan["python"],
            "scripts/build_formal_prior_bank.py",
            "build",
            "--plan",
            plan["protocol"],
            "--cohort",
            cohort,
            "--output",
            str(directory),
            "--shard",
            str(shard),
            "--shards",
            str(shards),
            "--batch-contexts",
            str(plan.get("bank_batch_contexts", 8)),
            "--device",
            "cuda",
            "--resume",
        ]
        if limit is not None:
            argv += ["--limit-contexts", str(limit)]
        if plan.get("bank_compile_eval", False):
            argv.append("--compile-eval")
        workers.append(
            {
                "name": f"{directory.name}/shard{shard}",
                "argv": argv,
                "directory": str(out),
                "gpus": gpu,
                "expected_output": str(out / "complete.json"),
            }
        )
    merge = {
        "name": directory.name + "/merge",
        "argv": [plan["python"], "scripts/build_formal_prior_bank.py", "merge", "--output", str(directory)],
        "directory": str(directory),
        "gpus": "",
        "expected_output": str(directory / "bank.pt"),
    }
    return [workers, [merge]]


def configure_bank_jobs(plan, *, include_final_evaluation=True):
    """Populate durable argv specifications; nothing runs during configuration."""
    root = Path(plan["output"])
    reused = plan.get("existing_banks", {})
    plan["preflight_bank_jobs"] = []
    for name, cohort, path_key, limit in (
        ("preflight_train", "train", "preflight_train_bank", 16),
        ("preflight_D128", "D128", "preflight_val_bank", 4),
    ):
        if name not in reused:
            plan["preflight_bank_jobs"] += bank_stages(plan, cohort, Path(plan[path_key]).parent, limit=limit)
    plan["bank_jobs"] = []
    for name, path_key in (("train", "train_bank"), ("D128", "val_bank")):
        if name not in reused:
            plan["bank_jobs"] += bank_stages(plan, name, Path(plan[path_key]).parent)
    plan["evaluation_jobs"] = []
    plan["evaluation_banks"] = {}
    if include_final_evaluation:
        for cohort in ("D400", "long50"):
            bank = Path(reused[cohort]["path"]) if cohort in reused else root / "banks" / cohort / "bank.pt"
            plan["evaluation_banks"][cohort] = str(bank)
            if cohort not in reused:
                plan["evaluation_jobs"] += bank_stages(plan, cohort, bank.parent)
            output = root / "evaluation" / cohort
            argv = train_command(plan, output)
            argv[argv.index("--val-bank") + 1] = str(bank)
            argv += ["--evaluate-checkpoint", str(Path(plan["run_directory"]) / "best.pt")]
            plan["evaluation_jobs"].append(
                [
                    {
                        "name": cohort + "/evaluate",
                        "argv": argv,
                        "directory": str(output),
                        "gpus": plan["training_gpus"],
                        "expected_output": str(output / "report.json"),
                    }
                ]
            )
    plan["status"] = "configured_waiting_for_preflight_banks"
    return plan


def initialize(
    protocol_file, output, source, *, python=sys.executable, gpus="0,1,2,3", port=31071, shared_snapshot=None
):
    """Freeze a completed implementation; this does not start GPU work."""
    protocol_file, output, source = map(lambda value: Path(value).resolve(), (protocol_file, output, source))
    if (output / "plan.json").exists():
        raise FileExistsError("An initialized P runtime already exists; continue its frozen plan.")
    protocol = read(protocol_file)
    family = validate_protocol_family(protocol)
    if sha256(protocol["producer"]["checkpoint"]) != protocol["producer"]["sha256"]:
        raise ValueError("The frozen G producer checkpoint changed.")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(shared_snapshot).resolve() if shared_snapshot is not None else output / "source"
    if shared_snapshot is None:
        if snapshot.exists():
            raise FileExistsError("Refusing to overwrite an existing source snapshot.")
        snapshot.mkdir()
        for name in ("egorecover", "scripts"):
            shutil.copytree(source / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif not all((snapshot / name).is_dir() for name in ("egorecover", "scripts")):
        raise ValueError("Shared frozen source snapshot is incomplete.")
    shutil.copy2(protocol_file, output / "protocol.json")
    hashes = {str(path.relative_to(snapshot)): sha256(path) for path in sorted(snapshot.rglob("*.py"))}
    plan = {
        "format": "egorecover_formal_prior_runtime_v1",
        "experiment_family": family,
        "case": protocol.get("case"),
        "created_at_beijing": now(),
        "output": str(output),
        "protocol": str(output / "protocol.json"),
        "protocol_sha256": sha256(output / "protocol.json"),
        "source_snapshot": str(snapshot),
        "source_hashes": hashes,
        "python": str(Path(python).resolve()),
        "data_root": protocol["data"]["root"],
        "smpl_directory": str(source / "body_models/smplx"),
        "training_gpus": gpus,
        "port": port,
        "run_id": protocol["run_id"],
        "run_directory": str(output / protocol["run_id"]),
        "train_bank": str(output / "banks/train/bank.pt"),
        "val_bank": str(output / "banks/D128/bank.pt"),
        "preflight_train_bank": str(output / "banks/preflight-train/bank.pt"),
        "preflight_val_bank": str(output / "banks/preflight-D128/bank.pt"),
        "pilot_updates": 2000,
        "max_updates": 10000,
        "minimum_free_GiB": 10,
        "preflight_bank_jobs": [],
        "bank_jobs": [],
        "evaluation_jobs": [],
        "status": "initialized_waiting_for_bank_job_configuration",
        "existing_banks": protocol.get("existing_banks", {}),
    }
    for name, key in BANK_KEYS.items():
        if name in plan["existing_banks"]:
            plan[key] = str(Path(plan["existing_banks"][name]["path"]).resolve())
    verify_existing_banks(plan)
    configure_bank_jobs(plan)
    write(output / "plan.json", plan)
    write(output / "status.json", {"phase": plan["status"], "at_beijing": now(), "P_training_started": False})
    return plan


def verify_identity(plan):
    if sha256(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("P protocol changed after runtime freeze.")
    actual_files = {
        str(path.relative_to(plan["source_snapshot"])) for path in Path(plan["source_snapshot"]).rglob("*.py")
    }
    if actual_files != set(plan["source_hashes"]):
        raise ValueError("The frozen implementation file set changed.")
    for relative, expected in plan["source_hashes"].items():
        if sha256(Path(plan["source_snapshot"]) / relative) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    protocol = read(plan["protocol"])
    if sha256(protocol["producer"]["checkpoint"]) != protocol["producer"]["sha256"]:
        raise ValueError("Frozen G producer changed.")
    if plan["pilot_updates"] != 2000 or plan["max_updates"] != 10000:
        raise ValueError("P must use the reviewed 2k pilot and 10k cap.")
    if plan.get("experiment_family") == "P02":
        validate_protocol_family(protocol)
        if plan.get("existing_banks") != protocol["existing_banks"]:
            raise ValueError("Existing-bank bindings differ from the frozen protocol.")
        for name, key in BANK_KEYS.items():
            if Path(plan[key]).resolve() != Path(protocol["existing_banks"][name]["path"]).resolve():
                raise ValueError(f"Runtime bank path differs from its frozen binding: {name}")
        if plan["run_id"] != protocol["run_id"] or plan.get("case") != protocol["case"]:
            raise ValueError("Runtime P02 case differs from the frozen protocol.")
    verify_existing_banks(plan)


def environment(plan, gpus):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=gpus,
        PYTHONPATH=plan["source_snapshot"],
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        NCCL_DEBUG="WARN",
        SMPLX_MODEL_PATH=plan["smpl_directory"],
    )
    return env


def train_command(plan, output, *, preflight=False, resume=False, stop_after=None):
    if len(plan["training_gpus"].split(",")) != 4:
        raise ValueError("P and its resume preflight require exactly four GPUs.")
    argv = [
        plan["python"],
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port",
        str(plan["port"]),
        "--module",
        "egorecover.prior_training",
        "--formal-protocol",
        plan["protocol"],
        "--output",
        str(output),
        "--data-root",
        plan["data_root"],
        "--train-bank",
        plan["preflight_train_bank"] if preflight else plan["train_bank"],
        "--val-bank",
        plan["preflight_val_bank"] if preflight else plan["val_bank"],
        "--device",
        "cuda",
        "--seed",
        "62",
        "--train-steps",
        "2" if preflight else "10000",
        "--batch-size",
        "64",
        "--microbatch-size",
        "8",
        "--learning-rate",
        "3e-5",
        "--eval-every",
        "1" if preflight else "500",
        "--lr-decay-step",
        "1" if preflight else "8000",
        "--lr-gamma",
        "0.1",
    ]
    if preflight:
        argv.append("--formal-preflight")
    if resume:
        argv.append("--resume")
    if stop_after is not None:
        if stop_after < 1:
            raise ValueError("A continuation must make at least one update.")
        argv += ["--stop-after", str(stop_after)]
    return argv


def checkpoint_step(directory):
    import torch

    path = Path(directory) / "resume.pt"
    return int(torch.load(path, map_location="cpu", weights_only=True, mmap=True)["step"]) if path.exists() else 0


def recover_checkpoint_exports(directory):
    """Repair committed exports only after the owning worker has exited."""
    if not (Path(directory) / "resume.pt").exists():
        return {"status": "no_resume_checkpoint"}
    from egorecover.formal_prior_training import recover_formal_checkpoint

    return recover_formal_checkpoint(Path(directory))


def run_processes(plan, specs, phase):
    """Each explicit argv runs in a new process group; failures preserve artifacts."""
    with termination_signals():
        return _run_processes(plan, specs, phase)


def _run_processes(plan, specs, phase):
    root, processes = Path(plan["output"]), []
    try:
        for spec in specs:
            directory = Path(spec["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / f"{phase}-{time.time_ns()}.log").open("x")
            journal, record = journal_job(plan, spec)
            env = environment(plan, spec["gpus"])
            env.update({ROOT_ENV: record["runtime"], TOKEN_ENV: record["token"]})
            process = subprocess.Popen(
                spec["argv"],
                cwd=plan["source_snapshot"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((spec, process, log, journal, record))
            bind_job(journal, record, process)
        write(
            root / f"{phase}-processes.json",
            [
                {
                    "name": spec["name"],
                    "pid": process.pid,
                    "gpus": spec["gpus"],
                    "journal": str(journal),
                    "identity": record["leader"],
                }
                for spec, process, _, journal, record in processes
            ],
        )
        while True:
            jobs = [
                {
                    "name": spec["name"],
                    "pid": process.pid,
                    "gpus": spec["gpus"],
                    "exit_code": process.poll(),
                    "progress": read(Path(spec["directory"]) / "progress.json"),
                    "status": read(Path(spec["directory"]) / "status.json"),
                }
                for spec, process, _, _, _ in processes
            ]
            write(
                root / "status.json",
                {"phase": phase, "at_beijing": now(), "jobs": jobs, "pilot_updates": 2000, "max_updates": 10000},
            )
            if any(row["exit_code"] not in (None, 0) for row in jobs):
                raise RuntimeError(f"{phase} failed; see preserved worker logs.")
            free = shutil.disk_usage(root).free
            if free < plan.get("minimum_free_GiB", 10) * 1024**3:
                raise RuntimeError("Project free space fell below the declared reserve; no artifacts were deleted.")
            if all(row["exit_code"] == 0 for row in jobs):
                break
            time.sleep(10)
        for spec in specs:
            if spec.get("expected_output") and not Path(spec["expected_output"]).is_file():
                raise RuntimeError(f"Worker completed without required output: {spec['name']}")
        return [finish_job(journal, record, process.returncode) for _, process, _, journal, record in processes]
    finally:
        try:
            cleanup_owned_jobs(plan, reason="worker_scope_exit")
        finally:
            for _, process, log, _, _ in processes:
                if process.poll() is not None:
                    process.wait()
                log.close()


def run_bank_jobs(plan, key, phase):
    """Stages run in order; workers within one stage may use disjoint GPUs."""
    for index, stage in enumerate(plan.get(key, [])):
        jobs = stage if isinstance(stage, list) else [stage]
        for job in jobs:
            target = Path(job["expected_output"]) if job.get("expected_output") else None
            if target is not None and target.name == "bank.pt" and target.is_file():
                marker = read(target.parent / "complete.json")
                if not marker or marker.get("bank_sha256") != sha256(target):
                    raise ValueError(f"Existing bank completion identity does not match: {target}")
        pending = [job for job in jobs if not job.get("expected_output") or not Path(job["expected_output"]).is_file()]
        if pending:
            run_processes(plan, pending, f"{phase}-{index}")


def _assert_tree_equal(left, right, *, floating=False):
    import torch

    from scripts.run_formal_mainline import assert_recovered_tensor

    if isinstance(left, torch.Tensor):
        if floating:
            return assert_recovered_tensor(left, right, optimizer=True)
        if not torch.equal(left, right):
            raise AssertionError("Restored RNG/counter/sampler tensor differs.")
        return 0.0
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((_assert_tree_equal(left[key], right[key], floating=floating) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        return max((_assert_tree_equal(a, b, floating=floating) for a, b in zip(left, right)), default=0.0)
    assert left == right
    return 0.0


def verify_preflight_segments(plan):
    base = Path(plan["output"]) / "preflight"
    ledger_path = base / "execution-segments.json"
    ledger = read(ledger_path)
    if (
        not ledger
        or ledger.get("format") != "egorecover_preflight_segments_v1"
        or ledger.get("complete") is not True
        or ledger.get("protocol_sha256") != plan["protocol_sha256"]
        or ledger.get("source_hashes") != plan["source_hashes"]
    ):
        raise ValueError("Preflight needs complete, frozen single-process execution-segment evidence.")
    segments = ledger.get("segments", [])
    expected = [("full", 0, 2, False), ("resumed", 0, 1, False), ("resumed", 1, 2, True)]
    if len(segments) != len(expected):
        raise ValueError("Preflight must contain one continuous full run and two separate resume runs.")
    identities = []
    for segment, (branch, start, end, resume) in zip(segments, expected):
        if (segment.get("branch"), segment.get("from_step"), segment.get("to_step"), segment.get("resume")) != (
            branch,
            start,
            end,
            resume,
        ):
            raise ValueError("Preflight execution segments are not full0-to2 versus resumed0-to1-plus1-to2.")
        worker = segment.get("worker", {})
        identity = worker.get("leader") or {}
        argv = worker.get("argv", [])
        expected_argv = train_command(plan, base / branch, preflight=True, resume=resume, stop_after=end - start)
        command_hash = hashlib.sha256(b"\0".join(os.fsencode(value) for value in expected_argv) + b"\0").hexdigest()
        if (
            worker.get("exit_code") != 0
            or argv != expected_argv
            or not isinstance(identity.get("pid"), int)
            or identity["pid"] <= 0
            or identity.get("start_ticks", 0) <= 0
            or identity.get("pgid") != identity["pid"]
            or identity.get("session_id") != identity["pid"]
            or identity.get("cmdline_sha256") != command_hash
            or not worker.get("token")
        ):
            raise ValueError("Preflight worker PID/start/argv/group or successful-exit identity is incomplete.")
        identities.append((identity["pid"], identity["start_ticks"], worker["token"]))
        checkpoint = base / branch / ("resume-step1.pt" if branch == "resumed" and end == 1 else "resume.pt")
        if not checkpoint.is_file() or segment.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError("Preflight execution checkpoint identity changed.")
        import torch

        if torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["step"] != end:
            raise ValueError("Preflight execution checkpoint does not contain its declared end step.")
    if len(set(identities)) != 3:
        raise ValueError("Preflight must use three separate process executions.")
    return {"execution_segments_sha256": sha256(ledger_path), "continuous_full0_to2_verified": True}


def verify_preflight(plan):
    import torch

    from scripts.run_formal_mainline import assert_recovered_tensor

    root = Path(plan["output"])
    base = root / "preflight"
    execution = verify_preflight_segments(plan)
    for branch in ("full", "resumed"):
        recover_checkpoint_exports(base / branch)
    full = torch.load(base / "full/resume.pt", map_location="cpu", weights_only=True, mmap=True)
    resumed = torch.load(base / "resumed/resume.pt", map_location="cpu", weights_only=True, mmap=True)
    assert full["step"] == resumed["step"] == 2
    assert full["identity"] == resumed["identity"]
    differences = {}
    for group in ("model", "ema"):
        assert full[group].keys() == resumed[group].keys()
        differences[group] = max(
            assert_recovered_tensor(tensor, resumed[group][key]) for key, tensor in full[group].items()
        )
    _assert_tree_equal(full["scheduler"], resumed["scheduler"])
    _assert_tree_equal(full["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
    differences["optimizer"] = _assert_tree_equal(
        full["optimizer"]["state"], resumed["optimizer"]["state"], floating=True
    )
    assert len(full["rank_rng_states"]) == len(resumed["rank_rng_states"]) == 4
    for left, right in zip(full["rank_rng_states"], resumed["rank_rng_states"]):
        if any(
            not isinstance(row.get("cuda_rng"), torch.Tensor)
            or row["cuda_rng"].dtype != torch.uint8
            or row["cuda_rng"].numel() == 0
            for row in (left, right)
        ):
            raise AssertionError("A four-GPU preflight requires actual CUDA RNG states on all ranks.")
        _assert_tree_equal(
            {k: v for k, v in left.items() if k != "model_sha256"},
            {k: v for k, v in right.items() if k != "model_sha256"},
        )
    logs = {}
    for name in ("full", "resumed"):
        if not _health_report_passed(read(base / name / "health_checks.json"), 2):
            raise AssertionError("Preflight numerical, causality, FK or sampling health checks did not pass.")
        logs[name] = [
            json.loads(line) for line in (base / name / "train.jsonl").read_text().splitlines() if line.strip()
        ]
        assert [row["step"] for row in logs[name]] == [1, 2]
        rates = [row["learning_rate"] if "learning_rate" in row else row["lr"] for row in logs[name]]
        assert math.isclose(rates[0], 3e-5, rel_tol=1e-12)
        assert math.isclose(rates[1], 3e-6, rel_tol=1e-12)
        assert all(math.isfinite(row[key]) for row in logs[name] for key in ("loss", "gradient_norm"))
    assert [row["trace_hash"] for row in logs["full"]] == [row["trace_hash"] for row in logs["resumed"]]
    evidence = {
        "status": "passed",
        "at_beijing": now(),
        "protocol_sha256": plan["protocol_sha256"],
        "source_hashes": plan["source_hashes"],
        "updates": 2,
        "ranks": 4,
        "CUDA_RNG_state_present_for_all_ranks": True,
        "preflight_bank_hashes": {key: sha256(plan[key]) for key in ("preflight_train_bank", "preflight_val_bank")},
        "RNG_sampling_trace_scheduler_optimizer_steps_exact": True,
        "weight_max_abs_differences": differences,
        "GPU_precision_rule": "model/EMA float32 rtol1e-7 atol2e-8; optimizer tensor relative max-difference1e-5; RNG, sampling and scheduler exact",
        "validation_is_accuracy_claim": False,
        **execution,
    }
    write(root / "preflight-verification.json", evidence)
    return evidence


def preflight(plan):
    root = Path(plan["output"])
    completed = read(root / "preflight-verification.json")
    if completed:
        if (
            completed.get("status") != "passed"
            or completed.get("protocol_sha256") != plan["protocol_sha256"]
            or completed.get("source_hashes") != plan["source_hashes"]
        ):
            raise ValueError("Existing preflight evidence does not match this frozen run.")
        for key, expected in completed["preflight_bank_hashes"].items():
            if sha256(plan[key]) != expected:
                raise ValueError("Preflight bank changed after verification.")
        execution = verify_preflight_segments(plan)
        if any(completed.get(key) != value for key, value in execution.items()):
            raise ValueError("Preflight execution evidence differs from its accepted verification.")
        return completed
    for key in ("preflight_train_bank", "preflight_val_bank"):
        if not Path(plan[key]).is_file():
            raise FileNotFoundError(f"Prepare the small preflight bank first: {key}")
    base = root / "preflight"
    if base.exists():
        archive = root / "preflight-interrupted" / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        base.rename(archive)
        write(
            archive / "archive-reason.json",
            {
                "reason": "Unverified preflight; preserve evidence and rerun continuous full0-to2 from scratch.",
                "at_beijing": now(),
            },
        )
    ledger = {
        "format": "egorecover_preflight_segments_v1",
        "complete": False,
        "protocol_sha256": plan["protocol_sha256"],
        "source_hashes": plan["source_hashes"],
        "segments": [],
    }
    for kind, start, target, resume in (("full", 0, 2, False), ("resumed", 0, 1, False), ("resumed", 1, 2, True)):
        directory = base / kind
        if checkpoint_step(directory) != start:
            raise ValueError("Preflight checkpoint does not match its declared execution-segment start.")
        spec = {
            "name": f"preflight-{kind}-to{target}",
            "argv": train_command(plan, directory, preflight=True, resume=resume, stop_after=target - start),
            "directory": str(directory),
            "gpus": plan["training_gpus"],
            "expected_output": str(directory / "resume.pt"),
        }
        workers = run_processes(plan, [spec], spec["name"])
        recover_checkpoint_exports(directory)
        if checkpoint_step(directory) != target:
            raise ValueError("Preflight worker did not commit the requested execution-segment end.")
        checkpoint = directory / "resume.pt"
        if kind == "resumed" and target == 1:
            checkpoint = directory / "resume-step1.pt"
            shutil.copy2(directory / "resume.pt", checkpoint)
        ledger["segments"].append(
            {
                "branch": kind,
                "from_step": start,
                "to_step": target,
                "resume": resume,
                "worker": workers[0],
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        write(base / "execution-segments.json", ledger)
    ledger["complete"] = True
    write(base / "execution-segments.json", ledger)
    return verify_preflight(plan)


def _health_report_passed(report, step):
    return bool(
        report
        and report.get("step", -1) >= step
        and report.get("passed") is True
        and HEALTH_CHECKS <= report.get("checks", {}).keys()
        and all(value is True for value in report["checks"].values())
    )


def health_passed(plan, step):
    return _health_report_passed(read(Path(plan["run_directory"]) / "health_checks.json"), step)


def pilot_decision(plan):
    from egorecover.formal_prior_evaluation import pilot_gate

    root = Path(plan["output"])
    recover_checkpoint_exports(plan["run_directory"])
    report = read(Path(plan["run_directory"]) / "selection.json")
    if not report or not isinstance(report.get("curve"), list):
        raise ValueError("P pilot has no validation curve; cannot authorize continuation.")
    verified = read(root / "preflight-verification.json")
    checks = bool(
        verified
        and verified.get("status") == "passed"
        and verified.get("protocol_sha256") == plan["protocol_sha256"]
        and health_passed(plan, plan["pilot_updates"])
    )
    decision = pilot_gate(report["curve"], checks_passed=checks)
    decision.update(at_beijing=now(), protocol_sha256=plan["protocol_sha256"], standalone_predictor_only=True)
    write(root / "pilot-decision.json", decision)
    return decision


def final_decision(plan):
    from egorecover.formal_prior_evaluation import final_gate, predictor_plateau_status

    root = Path(plan["output"])
    recover_checkpoint_exports(plan["run_directory"])
    paths = {cohort: root / "evaluation" / cohort / "report.json" for cohort in ("D400", "long50")}
    if not all(path.is_file() for path in paths.values()):
        return None
    reports = {cohort: read(path) for cohort, path in paths.items()}
    digest = sha256(Path(plan["run_directory"]) / "best.pt")
    protocol = read(plan["protocol"])
    exclusions = {}
    for cohort, report in reports.items():
        if report.get("checkpoint_sha256") != digest:
            raise ValueError(f"{cohort} evaluation did not use the current D128-selected best checkpoint.")
        expected = protocol["evaluation"][cohort]
        bank = Path(plan.get("evaluation_banks", {}).get(cohort, root / "banks" / cohort / "bank.pt"))
        if (
            report.get("bank_cohort") != cohort
            or report.get("bank_selection_sha256") != expected["selection_id"]
            or report.get("bank_clips") != expected["clips"]
            or not bank.is_file()
            or report.get("bank_sha256") != sha256(bank)
        ):
            raise ValueError(f"{cohort} evaluation bank/cohort/selection identity differs from the frozen protocol.")
        overlap = protocol["evaluation"].get("selection_and_report_overlap", {}).get(cohort)
        if overlap is None:
            exclusions[cohort] = {"status": "legacy_protocol_without_exclusion_metadata"}
            continue
        supplementary = report.get("selection_take_excluded") or {}
        selection = protocol["evaluation"]["D128"]
        selection_path = Path(selection["file"])
        frozen = read(selection_path)
        excluded_takes = sorted({row["take"] for row in frozen["records"]})
        if (
            frozen.get("sha256") != selection["selection_id"]
            or supplementary.get("selection_id") != selection["selection_id"]
            or supplementary.get("excluded_takes") != excluded_takes
            or supplementary.get("source_clips") != expected["clips"]
            or supplementary.get("overlap_clips") != overlap["overlap_clips"]
            or supplementary.get("clips") != overlap["remaining_clips_excluding_D128_takes"]
            or supplementary.get("takes") != overlap["remaining_takes_excluding_D128"]
            or supplementary.get("selection_file_sha256") != sha256(selection_path)
            or (
                selection.get("selection_file_sha256") is not None
                and supplementary.get("selection_file_sha256") != selection["selection_file_sha256"]
            )
            or not isinstance(supplementary.get("summary"), dict)
            or supplementary.get("status") != "ok"
            or any(
                supplementary.get("summary", {}).get(phase, {}).get("clips") != supplementary.get("clips")
                or supplementary.get("summary", {}).get(phase, {}).get("takes") != supplementary.get("takes")
                for phase in ("all", "mature", "startup")
            )
        ):
            raise ValueError(f"{cohort} D128-take-excluded report coverage/identity differs from the frozen protocol.")
        exclusions[cohort] = {
            "status": "verified_supplement_main_gate_unchanged",
            "selection_id": selection["selection_id"],
            "clips": supplementary["clips"],
            "takes": supplementary["takes"],
        }
    result = final_gate(reports["D400"], reports["long50"])
    selection_curve = read(Path(plan["run_directory"]) / "selection.json") or {}
    result["training_sufficiency"] = predictor_plateau_status(selection_curve.get("curve", []))
    result["selection_take_excluded_verification"] = exclusions
    result.update(
        at_beijing=now(),
        protocol_sha256=plan["protocol_sha256"],
        checkpoint_sha256=digest,
        final_G_gain_established=False,
    )
    write(root / "final-decision.json", result)
    return result


def supervise(plan):
    if plan.get("pair_managed"):
        raise ValueError("Use run_formal_prior_ab for a matched A/B experiment.")
    verify_identity(plan)
    cleanup_owned_jobs(plan, reason="startup_recovery_before_any_launch")
    recover_checkpoint_exports(plan["run_directory"])
    run_bank_jobs(plan, "preflight_bank_jobs", "preflight-banks")
    preflight(plan)
    run_bank_jobs(plan, "bank_jobs", "formal-banks")
    for key in ("train_bank", "val_bank"):
        if not Path(plan[key]).is_file():
            raise FileNotFoundError(f"Required formal bank missing: {key}")
    root, directory = Path(plan["output"]), Path(plan["run_directory"])
    step = checkpoint_step(directory)
    if step < plan["pilot_updates"]:
        spec = {
            "name": plan["run_id"],
            "argv": train_command(plan, directory, resume=step > 0, stop_after=plan["pilot_updates"] - step),
            "directory": str(directory),
            "gpus": plan["training_gpus"],
            "expected_output": str(directory / "resume.pt"),
        }
        run_processes(plan, [spec], "training-to2000")
    recover_checkpoint_exports(directory)
    if checkpoint_step(directory) < plan["pilot_updates"]:
        raise RuntimeError("Training worker did not produce its requested recoverable pilot checkpoint.")
    decision = pilot_decision(plan)
    if not decision["passed"]:
        write(
            root / "status.json",
            {
                "phase": "stopped_gate_failed",
                "at_beijing": now(),
                "step": checkpoint_step(directory),
                "decision": decision,
                "P_training_started": True,
                "auto_restart": False,
            },
        )
        return
    step = checkpoint_step(directory)
    if step < plan["max_updates"]:
        spec = {
            "name": plan["run_id"],
            "argv": train_command(plan, directory, resume=True, stop_after=plan["max_updates"] - step),
            "directory": str(directory),
            "gpus": plan["training_gpus"],
            "expected_output": str(directory / "resume.pt"),
        }
        run_processes(plan, [spec], "training-to10000")
    recover_checkpoint_exports(directory)
    if checkpoint_step(directory) != plan["max_updates"]:
        raise RuntimeError("P did not end at its declared 10k cap.")
    run_bank_jobs(plan, "evaluation_jobs", "fixed-evaluation")
    final = final_decision(plan)
    write(
        root / "status.json",
        {
            "phase": (
                "training_complete_pending_fixed_evaluation"
                if final is None
                else (
                    "P_complete_predictor_gate_passed_awaiting_review"
                    if final["passed"]
                    else "P_complete_predictor_gate_failed_awaiting_review"
                )
            ),
            "at_beijing": now(),
            "step": plan["max_updates"],
            "P_training_started": True,
            "system_G_gain_established": False,
            "automatic_extension": False,
            "final_decision": final,
        },
    )


def monitor(plan):
    root = Path(plan["output"])
    verification = read(root / "preflight-verification.json")
    return {
        "status": read(root / "status.json"),
        "training": read(Path(plan["run_directory"]) / "progress.json"),
        "trainer_status": read(Path(plan["run_directory"]) / "status.json"),
        "pilot_decision": read(root / "pilot-decision.json"),
        "final_decision": read(root / "final-decision.json"),
        "preflight": (
            {
                key: verification[key]
                for key in (
                    "status",
                    "at_beijing",
                    "updates",
                    "ranks",
                    "RNG_sampling_trace_scheduler_optimizer_steps_exact",
                )
                if key in verification
            }
            if verification
            else None
        ),
    }


def reexec_frozen_controller(plan, action, plan_file):
    """Keep controller gates and helpers on the same frozen source as workers."""
    if action not in ("run", "preflight", "verify-preflight"):
        return False
    snapshot = Path(plan["source_snapshot"]).resolve()
    script = snapshot / "scripts/run_formal_prior.py"
    if (
        Path(__file__).resolve() == script
        and Path.cwd().resolve() == snapshot
        and os.environ.get("PYTHONPATH") == str(snapshot)
        and Path(sys.executable).resolve() == Path(plan["python"]).resolve()
    ):
        return False
    verify_identity(plan)
    argv = [plan["python"], str(script), action, "--plan", str(Path(plan_file).resolve())]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot)
    os.chdir(snapshot)
    os.execve(plan["python"], argv, env)
    raise RuntimeError("Frozen-controller process replacement unexpectedly returned.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "preflight", "verify-preflight", "monitor"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=31071)
    args = parser.parse_args()
    if args.action == "init":
        if args.protocol is None or args.output is None:
            parser.error("init requires --protocol and --output")
        plan = initialize(args.protocol, args.output, args.source, python=args.python, gpus=args.gpus, port=args.port)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.plan is None:
        parser.error("--plan is required")
    plan = read(args.plan)
    if args.action == "monitor":
        print(json.dumps(monitor(plan), ensure_ascii=False, indent=2))
        return
    reexec_frozen_controller(plan, args.action, args.plan)
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("This P runtime is already being supervised.") from error
        write(root / "controller.json", {"pid": os.getpid(), "action": args.action, "at_beijing": now()})
        with termination_signals():
            try:
                if args.action == "run":
                    supervise(plan)
                elif args.action == "preflight":
                    verify_identity(plan)
                    cleanup_owned_jobs(plan, reason="preflight_startup_recovery")
                    run_bank_jobs(plan, "preflight_bank_jobs", "preflight-banks")
                    preflight(plan)
                else:
                    verify_identity(plan)
                    cleanup_owned_jobs(plan, reason="verification_startup_recovery")
                    verify_preflight(plan)
            except BaseException as exc:
                write(
                    root / "status.json",
                    {
                        "phase": "interrupted" if isinstance(exc, SupervisorInterrupted) else "failed",
                        "at_beijing": now(),
                        "error": traceback.format_exc(),
                        "artifacts_preserved": True,
                    },
                )
                raise
            finally:
                cleanup_owned_jobs(plan, reason="supervisor_exit")


if __name__ == "__main__":
    main()
