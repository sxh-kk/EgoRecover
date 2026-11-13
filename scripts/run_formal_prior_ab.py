"""Freeze and supervise matched P02 A/B experiments, waiting for free GPUs."""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scripts import run_formal_prior as single
from scripts.run_history_ablation import now, read, write

TERMINAL = {
    "stopped_gate_failed",
    "training_complete_pending_fixed_evaluation",
    "P_complete_predictor_gate_passed_awaiting_review",
    "P_complete_predictor_gate_failed_awaiting_review",
}


def validate_pair_protocols(protocols):
    if set(protocols) != {"A", "B"}:
        raise ValueError("P02 requires both A and B protocols.")
    for case, protocol in protocols.items():
        if single.validate_protocol_family(protocol) != "P02" or protocol["case"] != case:
            raise ValueError("P02 case and protocol disagree.")
    for key in (
        "producer",
        "history_distribution",
        "data",
        "sampling",
        "optimization",
        "evaluation",
        "gates",
        "loss",
        "existing_banks",
    ):
        if protocols["A"][key] != protocols["B"][key]:
            raise ValueError(f"A/B matched protocol field differs: {key}")
    models = [{key: value for key, value in p["model"].items() if key != "hold_beta"} for p in protocols.values()]
    if models[0] != models[1] or protocols["A"]["run_id"] == protocols["B"]["run_id"]:
        raise ValueError("A/B networks must match and run IDs must differ.")


def validate_gpu_pools(pools):
    parsed = {case: str(value).split(",") for case, value in pools.items()}
    if set(parsed) != {"A", "B"} or any(len(ids) != 4 or len(set(ids)) != 4 for ids in parsed.values()):
        raise ValueError("Each A/B group requires four distinct GPUs.")
    if set(parsed["A"]) & set(parsed["B"]):
        raise ValueError("A/B GPU groups must not overlap.")
    if any(not index.isdigit() for ids in parsed.values() for index in ids):
        raise ValueError("A/B GPU pools use explicit numeric physical-device indices.")
    return parsed


def process_identity(pid):
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        return None


def command_identity(pid):
    try:
        return single.sha256(Path("/proc") / str(pid) / "cmdline")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def initialize_ab(
    protocol_files,
    output,
    source,
    *,
    python=sys.executable,
    gpu_pools=None,
    ports=None,
    blocking_pids=(),
    stable_seconds=60,
):
    output, source = Path(output).resolve(), Path(source).resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("A P02 A/B runtime already exists; resume its frozen plan.")
    protocols = {case: read(Path(path)) for case, path in protocol_files.items()}
    validate_pair_protocols(protocols)
    gpu_pools = gpu_pools or {"A": "0,1,2,3", "B": "4,5,6,7"}
    parsed = validate_gpu_pools(gpu_pools)
    ports = ports or {"A": 31081, "B": 31091}
    if ports["A"] == ports["B"]:
        raise ValueError("A/B torchrun ports must be distinct.")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source"
    if snapshot.exists():
        raise FileExistsError("Refusing to overwrite the P02 source snapshot.")
    snapshot.mkdir()
    for name in ("egorecover", "scripts"):
        shutil.copytree(source / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plans = {}
    for case in ("A", "B"):
        child = single.initialize(
            protocol_files[case],
            output / case,
            source,
            python=python,
            gpus=gpu_pools[case],
            port=ports[case],
            shared_snapshot=snapshot,
        )
        child["pair_managed"] = str(output / "plan.json")
        write(output / case / "plan.json", child)
        plans[case] = str(output / case / "plan.json")
    plan = {
        "format": "egorecover_formal_prior_ab_v1",
        "created_at_beijing": now(),
        "output": str(output),
        "source_snapshot": str(snapshot),
        "source_hashes": read(plans["A"])["source_hashes"],
        "python": str(Path(python).resolve()),
        "smpl_directory": str(source / "body_models/smplx"),
        "cases": plans,
        "gpu_pools": gpu_pools,
        "ports": ports,
        "minimum_free_GiB": 10,
        "resource_wait": {
            "enabled": True,
            "gpus": parsed["A"] + parsed["B"],
            "max_memory_MiB": 2048,
            "stable_seconds": stable_seconds,
            "poll_seconds": 15,
            "blocking_processes": [
                {
                    "pid": int(pid),
                    "start_ticks": process_identity(int(pid)),
                    "cmdline_sha256": command_identity(int(pid)),
                }
                for pid in blocking_pids
                if process_identity(int(pid)) is not None
            ],
        },
        "pilot_updates": 2000,
        "max_updates": 10000,
        "automatic_G_U_FFT": False,
    }
    write(output / "plan.json", plan)
    write(
        output / "status.json",
        {
            "phase": "initialized_pending_resource_check",
            "at_beijing": now(),
            "P_training_started": False,
            "cases": {case: single.monitor(read(path)) for case, path in plans.items()},
        },
    )
    return plan


def verify_pair_identity(plan):
    children = {case: read(path) for case, path in plan["cases"].items()}
    if set(children) != {"A", "B"}:
        raise ValueError("Frozen pair case set changed.")
    validate_pair_protocols({case: read(child["protocol"]) for case, child in children.items()})
    pools = {case: child["training_gpus"] for case, child in children.items()}
    parsed = validate_gpu_pools(pools)
    if pools != plan["gpu_pools"]:
        raise ValueError("Pair GPU pools differ from their frozen child assignments.")
    resources = plan["resource_wait"]
    if (
        set(map(str, resources["gpus"])) != set(parsed["A"] + parsed["B"])
        or resources.get("enabled") is not True
        or resources.get("stable_seconds", 0) < 60
        or not 0 < resources.get("max_memory_MiB", 0) <= 2048
        or not 0 < resources.get("poll_seconds", 0) <= 30
    ):
        raise ValueError(
            "The resource guard must cover all8 assigned GPUs with the declared free-memory stability rule."
        )
    for child in children.values():
        single.verify_identity(child)
        if child["source_snapshot"] != plan["source_snapshot"] or child["source_hashes"] != plan["source_hashes"]:
            raise ValueError("A/B must execute the same frozen source snapshot.")
        if child.get("pair_managed") != str(Path(plan["output"]) / "plan.json"):
            raise ValueError("A/B child no longer binds its pair supervisor.")
    return children


def gpu_snapshot():
    """Read device memory and foreign compute PIDs without claiming devices."""

    def query(fields, flag):
        completed = subprocess.run(
            ["nvidia-smi", f"--query-{flag}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
        return [line.strip().split(",") for line in completed.stdout.splitlines() if line.strip()]

    cards = {
        parts[0].strip(): {"uuid": parts[1].strip(), "used_MiB": int(parts[2].strip())}
        for parts in query("index,uuid,memory.used", "gpu")
    }
    processes = []
    for parts in query("pid,gpu_uuid,used_gpu_memory", "compute-apps"):
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        processes.append(
            {
                "pid": int(parts[0]),
                "uuid": parts[1].strip(),
                "used_MiB": int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else None,
            }
        )
    return {"cards": cards, "processes": processes}


def resources_available(config, observed):
    requested = set(map(str, config["gpus"]))
    missing = requested - observed["cards"].keys()
    uuids = {observed["cards"][index]["uuid"] for index in requested if index in observed["cards"]}
    busy = [proc for proc in observed["processes"] if proc["uuid"] in uuids]
    high = [
        index
        for index in requested
        if index in observed["cards"] and observed["cards"][index]["used_MiB"] >= config["max_memory_MiB"]
    ]
    blocked = []
    for proc in config.get("blocking_processes", []):
        actual = process_identity(proc["pid"])
        if actual is not None and (proc.get("start_ticks") is None or proc["start_ticks"] == actual):
            # A launcher may exec its next stage without changing PID/start.
            # Only process exit/zombie or start-time mismatch releases it.
            command = command_identity(proc["pid"])
            blocked.append(
                {
                    **proc,
                    "observed_cmdline_sha256": command,
                    "command_changed": proc.get("cmdline_sha256") not in (None, command),
                }
            )
    return not (missing or busy or high or blocked), {
        "missing_GPUs": sorted(missing),
        "foreign_compute_processes": busy,
        "GPUs_over_memory_limit": sorted(high),
        "blocking_launchers_alive": blocked,
    }


def wait_for_gpus(plan):
    """Keep the durable supervisor lock while waiting; never preempt jobs."""
    config = plan["resource_wait"]
    if config.get("enabled") is not True:
        raise ValueError("P02 must wait for an explicit free-GPU resource check.")
    training_started = any(
        any((Path(read(path)["run_directory"]) / name).exists() for name in ("resume.pt", "progress.json"))
        for path in plan.get("cases", {}).values()
    )
    stable_since = None
    while True:
        try:
            observed = gpu_snapshot()
            free, reasons = resources_available(config, observed)
            error = None
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            observed, free, reasons, error = {}, False, {}, str(exc)
        current = time.monotonic()
        stable_since = current if free and stable_since is None else stable_since
        if not free:
            stable_since = None
        stable = current - stable_since if stable_since is not None else 0.0
        status = {
            "phase": "waiting_for_gpus",
            "at_beijing": now(),
            "P_training_started": training_started,
            "resuming_existing_training": training_started,
            "policy": "Wait for all8 free GPUs and known launcher completion; no sharing or preemption.",
            "observed": observed,
            "reasons": reasons,
            "probe_error": error,
            "stable_free_seconds": stable,
            "required_stable_seconds": config["stable_seconds"],
        }
        write(Path(plan["output"]) / "status.json", status)
        if free and stable >= config["stable_seconds"]:
            status["phase"] = "resources_ready"
            write(Path(plan["output"]) / "resource-verification.json", status)
            return status
        time.sleep(config["poll_seconds"])


def _rows_by_step(directory, prefix):
    rows = {}
    for text in (Path(directory) / "train.jsonl").read_text().splitlines():
        if not text.strip():
            continue
        row = json.loads(text)
        if row["step"] <= prefix:
            if row["step"] in rows and rows[row["step"]]["trace_hash"] != row["trace_hash"]:
                raise ValueError("Repeated recovered step has a different physical-query trace.")
            rows[row["step"]] = row
    if sorted(rows) != list(range(1, prefix + 1)):
        raise ValueError("A/B query trace has missing physical updates.")
    return rows


def audit_matched_pair(plan, children, *, preflight=False):
    locations = {
        case: Path(child["output"]) / "preflight/full" if preflight else Path(child["run_directory"])
        for case, child in children.items()
    }
    identities = {case: read(directory / "identity.json") for case, directory in locations.items()}
    if any(not identity for identity in identities.values()):
        raise ValueError("A/B initial identity is missing.")
    for case, identity in identities.items():
        if (
            identity.get("objective") != "common_cv_valid_state"
            or identity.get("model_config", {}).get("hold_beta", False) is not (case == "B")
            or identity.get("geometry_audit", {}).get("weight") != 0.008869197811349056
            or identity.get("geometry_audit", {}).get("weight_mode") != "P01_fixed_not_recalibrated"
        ):
            raise ValueError(f"{case} actual training objective/beta/fixed-weight identity differs from its protocol.")
    for key in ("random_initial_model_sha256", "bank_sha256", "seed", "batch_size", "microbatch_size"):
        if identities["A"][key] != identities["B"][key]:
            raise ValueError(f"A/B initial identity differs: {key}")
    steps = {case: single.checkpoint_step(directory) for case, directory in locations.items()}
    prefix = min(steps.values())
    if prefix < 1:
        raise ValueError("A/B matching audit requires actual completed updates.")
    logs = {case: _rows_by_step(directory, prefix) for case, directory in locations.items()}
    for step in range(1, prefix + 1):
        if logs["A"][step]["trace_hash"] != logs["B"][step]["trace_hash"]:
            raise ValueError(f"A/B physical-query trace differs at step{step}.")
        if not math.isclose(logs["A"][step]["lr"], logs["B"][step]["lr"], rel_tol=1e-12):
            raise ValueError(f"A/B learning-rate trace differs at step{step}.")
    for key in ("code", "geometry", "loss"):
        if not math.isclose(logs["A"][1][key], logs["B"][1][key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"A/B initial objective differs: {key}")
    report = {
        "status": "passed",
        "at_beijing": now(),
        "preflight": preflight,
        "matched_prefix_updates": prefix,
        "actual_checkpoint_steps": steps,
        "random_initial_model_sha256": identities["A"]["random_initial_model_sha256"],
        "bank_sha256": identities["A"]["bank_sha256"],
        "initial_objective_equal": True,
        "all_matched_query_and_LR_traces_equal": True,
        "weights_are_not_expected_equal_after_update": True,
    }
    write(Path(plan["output"]) / ("matched-preflight.json" if preflight else "matched-training.json"), report)
    return report


def _worker_spec(child, phase, *, stop_after=None, resume=False):
    directory = Path(child["run_directory"])
    return {
        "name": child["run_id"],
        "argv": single.train_command(child, directory, resume=resume, stop_after=stop_after),
        "directory": str(directory),
        "gpus": child["training_gpus"],
        "expected_output": str(directory / "resume.pt"),
        "phase": phase,
    }


def supervise_ab(plan):
    children = verify_pair_identity(plan)
    single.cleanup_owned_jobs(plan, reason="startup_recovery_before_resource_wait")
    for child in children.values():
        single.recover_checkpoint_exports(child["run_directory"])
    wait_for_gpus(plan)
    preflights = [
        {
            "name": case + "/preflight",
            "argv": [child["python"], "-m", "scripts.run_formal_prior", "preflight", "--plan", plan["cases"][case]],
            "directory": child["output"],
            "gpus": child["training_gpus"],
            "expected_output": str(Path(child["output"]) / "preflight-verification.json"),
        }
        for case, child in children.items()
    ]
    single.run_processes(plan, preflights, "AB-preflight")
    audit_matched_pair(plan, children, preflight=True)
    workers = []
    for child in children.values():
        step = single.checkpoint_step(child["run_directory"])
        if step < 2000:
            workers.append(_worker_spec(child, "pilot", resume=step > 0, stop_after=2000 - step))
    if workers:
        single.run_processes(plan, workers, "AB-training-to2000")
    for child in children.values():
        single.recover_checkpoint_exports(child["run_directory"])
    if any(single.checkpoint_step(c["run_directory"]) < 2000 for c in children.values()):
        raise RuntimeError("A/B pilot lacks a recoverable2k state.")
    audit_matched_pair(plan, children)
    decisions = {case: single.pilot_decision(child) for case, child in children.items()}
    write(Path(plan["output"]) / "pilot-decisions.json", decisions)
    workers = []
    for case, child in children.items():
        if not decisions[case]["passed"]:
            write(
                Path(child["output"]) / "status.json",
                {
                    "phase": "stopped_gate_failed",
                    "at_beijing": now(),
                    "step": single.checkpoint_step(child["run_directory"]),
                    "decision": decisions[case],
                    "P_training_started": True,
                    "auto_restart": False,
                },
            )
            continue
        step = single.checkpoint_step(child["run_directory"])
        if step < 10000:
            workers.append(_worker_spec(child, "continuation", resume=True, stop_after=10000 - step))
    if workers:
        single.run_processes(plan, workers, "AB-passed-continuations-to10000")
    for child in children.values():
        single.recover_checkpoint_exports(child["run_directory"])
    audit_matched_pair(plan, children)
    for case, child in children.items():
        if not decisions[case]["passed"]:
            continue
        if single.checkpoint_step(child["run_directory"]) != 10000:
            raise RuntimeError(f"{case} did not stop at the declared10k cap.")
        single.run_bank_jobs(child, "evaluation_jobs", "fixed-evaluation")
        final = single.final_decision(child)
        write(
            Path(child["output"]) / "status.json",
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
                "step": 10000,
                "P_training_started": True,
                "automatic_extension": False,
                "final_decision": final,
                "system_G_gain_established": False,
            },
        )
    write(
        Path(plan["output"]) / "status.json",
        {
            "phase": "AB_finished_awaiting_review",
            "at_beijing": now(),
            "cases": {case: single.monitor(child) for case, child in children.items()},
            "automatic_G_U_FFT": False,
        },
    )


def monitor_ab(plan):
    return {
        "status": read(Path(plan["output"]) / "status.json"),
        "matched_preflight": read(Path(plan["output"]) / "matched-preflight.json"),
        "matched_training": read(Path(plan["output"]) / "matched-training.json"),
        "cases": {case: single.monitor(read(path)) for case, path in plan["cases"].items()},
    }


def reexec_frozen(plan, action, plan_file):
    if action != "run":
        return False
    snapshot = Path(plan["source_snapshot"]).resolve()
    script = snapshot / "scripts/run_formal_prior_ab.py"
    if (
        Path(__file__).resolve() == script
        and Path.cwd().resolve() == snapshot
        and os.environ.get("PYTHONPATH") == str(snapshot)
        and Path(sys.executable).resolve() == Path(plan["python"]).resolve()
    ):
        return False
    verify_pair_identity(plan)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(snapshot)
    argv = [plan["python"], str(script), "run", "--plan", str(Path(plan_file).resolve())]
    os.chdir(snapshot)
    os.execve(plan["python"], argv, env)
    raise RuntimeError("Frozen A/B process replacement unexpectedly returned.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "run", "monitor"))
    p.add_argument("--plan", type=Path)
    p.add_argument("--protocol-a", type=Path)
    p.add_argument("--protocol-b", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--source", type=Path, default=Path.cwd())
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpus-a", default="0,1,2,3")
    p.add_argument("--gpus-b", default="4,5,6,7")
    p.add_argument("--port-a", type=int, default=31081)
    p.add_argument("--port-b", type=int, default=31091)
    p.add_argument("--wait-pid", type=int, action="append", default=[])
    args = p.parse_args()
    if args.action == "init":
        if args.protocol_a is None or args.protocol_b is None or args.output is None:
            p.error("init needs --protocol-a, --protocol-b and --output")
        result = initialize_ab(
            {"A": args.protocol_a, "B": args.protocol_b},
            args.output,
            args.source,
            python=args.python,
            gpu_pools={"A": args.gpus_a, "B": args.gpus_b},
            ports={"A": args.port_a, "B": args.port_b},
            blocking_pids=args.wait_pid,
        )
        print(
            json.dumps(
                {"plan": str(Path(result["output"]) / "plan.json"), "status": "initialized_waiting_for_gpus"},
                ensure_ascii=False,
            )
        )
        return
    if args.plan is None:
        p.error("--plan is required")
    plan = read(args.plan)
    if args.action == "monitor":
        print(json.dumps(monitor_ab(plan), ensure_ascii=False, indent=2))
        return
    reexec_frozen(plan, args.action, args.plan)
    output = Path(plan["output"])
    with (output / "controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("This A/B runtime is already supervised.") from exc
        write(output / "controller.json", {"pid": os.getpid(), "at_beijing": now()})
        with single.termination_signals():
            try:
                supervise_ab(plan)
            except BaseException as exc:
                write(
                    output / "status.json",
                    {
                        "phase": "interrupted" if isinstance(exc, single.SupervisorInterrupted) else "failed",
                        "at_beijing": now(),
                        "error": traceback.format_exc(),
                        "artifacts_preserved": True,
                    },
                )
                raise
            finally:
                single.cleanup_owned_jobs(plan, reason="pair_supervisor_exit")


if __name__ == "__main__":
    main()
