"""Supervise planned G training budgets and their fixed evaluations."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from scripts.run_history_ablation import now, read, write

METRICS = (
    "mpjpe_world_mm",
    "pa_mpjpe_mm",
    "velocity_error_mm_s",
    "acceleration_error_mm_s2",
    "foot_slide_uniegomotion_mm",
    "wrist_pe_mm",
    "head_joint_pe_mm",
    "mpjre_f",
    "head_rot_f",
    "root_mean_initial_translation_mm",
    "root_final_initial_translation_mm",
    "mpjpe_hand_mm",
    "head_translation_leye_mm",
    "foot_contact_mesh_mm",
)


def training_budgets(plan):
    """A single declared budget has no automatic extension stage."""
    standard = plan.get("standard_updates", 32000)
    cap = plan.get("native_checkpoint_total_cap", 40000)
    if (
        isinstance(standard, bool)
        or not isinstance(standard, int)
        or standard < 1
        or isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap < standard
    ):
        raise ValueError("Training budgets must start at the standard budget and increase within the native cap.")
    budgets = plan.get("training_budgets", [standard, cap] if cap > standard else [standard])
    if (
        not isinstance(budgets, (list, tuple))
        or not 1 <= len(budgets) <= 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in budgets)
        or budgets[0] != standard
        or any(a >= b for a, b in zip(budgets, budgets[1:]))
        or budgets[-1] > cap
    ):
        raise ValueError("Training budgets must start at the standard budget and increase within the native cap.")
    return tuple(budgets)


def command(plan, job, output, *, preflight=False, resume=False, stop_after=None):
    result = [
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
        "constant_velocity",
        "--from-scratch",
        "--device",
        "cuda",
        "--seed",
        "62",
        "--train-steps",
        "2" if preflight else str(plan.get("native_checkpoint_total_cap", 40000)),
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
        "--unroll-steps",
        "8",
        "--detach-between-steps",
        "--isolate-forward-rng",
        "--velocity-weight",
        str(plan["velocity_weight"]),
        "--slow-weight",
        "0",
        "--learning-rate",
        "3e-5",
        "--lr-schedule",
        "uniegomotion",
        "--lr-decay-step",
        "1" if preflight else str(plan.get("lr_decay_step", 12000)),
        "--lr-decay-gamma",
        str(plan.get("lr_decay_gamma", 0.3)),
        "--eval-every",
        "1" if preflight else "1000",
        "--dev-selection-file",
        plan["preflight_selection"] if preflight else plan["dev_selection"],
        "--output",
        str(output),
    ]
    if resume:
        result.append("--resume")
    if stop_after is not None:
        result += ["--stop-after", str(stop_after)]
    return result


def environment(plan, gpu):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONPATH=plan["source_snapshot"],
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        NCCL_DEBUG="WARN",
        SMPLX_MODEL_PATH=plan["smpl_directory"],
    )
    compile_threads = plan.get("compile_threads_per_rank", 1 if plan.get("runtime_revision") == "runtime3" else None)
    if compile_threads is not None:
        if isinstance(compile_threads, bool) or not isinstance(compile_threads, int) or compile_threads < 1:
            raise ValueError("Compile threads per rank must be a positive integer.")
        env["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    return env


def export_checkpoint(path, target, *, expected_step=None):
    import torch

    from egorecover.io import save_checkpoint

    saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if expected_step is not None and saved["step"] != expected_step:
        return False
    keys = ("format", "model_config", "identity", "protocol", "stats", "ema", "step", "best")
    save_checkpoint(target, {**{key: saved[key] for key in keys}, "purpose": "evaluation_or_initialization_only"})
    return True


def continuation_step(directory):
    """Progress can be newer than the recoverable optimizer/RNG state."""
    import torch

    checkpoint = Path(directory) / "resume.pt"
    if not checkpoint.exists():
        return 0
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    return saved["step"]


def checkpoint_milestones(plan):
    root = Path(plan["output"])
    for job in plan["jobs"]:
        directory = root / job["name"]
        selection = read(directory / "selection.json") or {}
        for step in plan.get("checkpoint_milestones", (12000, 24000)):
            target = directory / f"ema-step{step}.pt"
            if target.exists() or not any(row["step"] == step for row in selection.get("curve", [])):
                continue
            if (directory / "resume.pt").exists():
                export_checkpoint(directory / "resume.pt", target, expected_step=step)


def run_jobs(plan, specs, phase):
    root, processes = Path(plan["output"]), []
    try:
        for name, argv, directory, gpu in specs:
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / f"{phase}.log").open("x")
            process = subprocess.Popen(
                argv,
                cwd=plan["source_snapshot"],
                env=environment(plan, gpu),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((name, process, directory, log, gpu))
        write(root / f"{phase}-processes.json", [{"case": n, "pid": p.pid, "gpus": g} for n, p, _, _, g in processes])
        while True:
            states = [
                {
                    "case": n,
                    "pid": p.pid,
                    "gpus": g,
                    "exit_code": p.poll(),
                    "progress": read(d / "progress.json"),
                    "status": read(d / "status.json"),
                }
                for n, p, d, _, g in processes
            ]
            write(
                root / "status.json",
                {
                    "phase": phase,
                    "at_beijing": now(),
                    "jobs": states,
                    "standard_budget": plan.get("standard_updates", 32000),
                    "conditional_cap": plan.get("native_checkpoint_total_cap", 40000),
                    "training_role": plan.get("training_role", "foundation"),
                    "training_budgets": list(training_budgets(plan)),
                },
            )
            if any(row["exit_code"] not in (None, 0) for row in states):
                raise RuntimeError(f"A {phase} job failed; preserve its logs and checkpoint.")
            if phase.startswith("training"):
                checkpoint_milestones(plan)
                space = os.statvfs(root)
                if space.f_bavail * space.f_frsize < 10 * 1024**3:
                    raise RuntimeError("Project disk headroom dropped below10GiB; checkpoints preserved.")
            if all(row["exit_code"] == 0 for row in states):
                break
            time.sleep(10)
    finally:
        for _, process, _, log, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            log.close()


def assert_recovered_tensor(actual, expected, *, optimizer=False):
    """GPU model math is checked near float32 precision; RNG stays bit-exact."""
    import torch

    if actual.is_floating_point() and optimizer:
        scale = max(float(actual.abs().max()), float(expected.abs().max()))
        assert float((actual - expected).abs().max()) <= max(1e-5 * scale, torch.finfo(actual.dtype).tiny)
    elif actual.is_floating_point():
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-8)
    else:
        assert torch.equal(actual, expected)
    return float((actual - expected).abs().max())


def verify_preflight(plan):
    import torch

    from egorecover.io import state_sha256

    torch.set_num_threads(2)
    root, evidence, audits = Path(plan["output"]), {}, []
    for job in plan["jobs"]:
        base = root / "preflight" / job["name"]
        full = torch.load(base / "full/resume.pt", weights_only=True, map_location="cpu", mmap=True)
        resumed = torch.load(base / "resumed/resume.pt", weights_only=True, map_location="cpu", mmap=True)
        differences = {}
        exact = {}
        for group in ("model", "ema", "feedback_model"):
            differences[group] = max(
                assert_recovered_tensor(value, resumed[group][key]) for key, value in full[group].items()
            )
            exact[group] = all(torch.equal(value, resumed[group][key]) for key, value in full[group].items())
        assert exact["feedback_model"]
        for group in ("cpu_rng", "sample_rng"):
            assert torch.equal(full[group], resumed[group])
        assert full["scheduler"] == resumed["scheduler"]
        for left, right in zip(full["rank_rng_states"], resumed["rank_rng_states"]):
            for key in ("cpu_rng", "sample_rng", "cuda_rng"):
                assert torch.equal(left[key], right[key])
        for key, state in full["optimizer"]["state"].items():
            for field, value in state.items():
                other = resumed["optimizer"]["state"][key][field]
                if isinstance(value, torch.Tensor):
                    if field == "step":
                        assert torch.equal(value, other)
                    else:
                        assert_recovered_tensor(value, other, optimizer=True)
                else:
                    assert value == other
        assert full["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"]
        assert full["identity"]["prior"] == "constant_velocity" and full["identity"]["from_scratch"]
        assert full["identity"]["initialize"] is None and full["identity"]["e7"] is None
        rows = [json.loads(line) for line in (base / "full/train.jsonl").read_text().splitlines()]
        assert all(
            math.isfinite(row[key]) for row in rows for key in ("loss", "flow", "current", "velocity", "gradient_norm")
        )
        gamma = plan.get("lr_decay_gamma", 0.3)
        assert math.isclose(rows[-1]["learning_rate"], 3e-5 * gamma, rel_tol=1e-12)
        assert math.isclose(rows[-1]["next_learning_rate"], 3e-5 * gamma**2, rel_tol=1e-12)
        identity = full["identity"]
        evidence[job["name"]] = {
            "initial_model_sha256": identity["random_initial_model_sha256"],
            "ema_sha256": state_sha256(full["ema"]),
            "RNG_scheduler_optimizer_steps_exact": True,
            "weight_max_abs_differences": differences,
            "weight_bit_exact": exact,
            "GPU_weight_recovery": "float32 numerical precision, not bitwise identity",
            "two_step_validation_scores_mm": {
                "full": full["curve"][-1]["selection_body_mm"],
                "resumed": resumed["curve"][-1]["selection_body_mm"],
            },
            "updates": full["step"],
            "last_progress": rows[-1],
        }
        audit = read(base / "full/sampling-first-step.json")
        audits.append(
            [
                {
                    key: row[key]
                    for key in (
                        "global_offset",
                        "queries",
                        "target_local_frame",
                        "tau",
                        "current_epsilon_sha256",
                        "all_current_emission_noise_sha256",
                        "rollin_frames",
                    )
                }
                for row in audit
            ]
        )
    assert len({row["initial_model_sha256"] for row in evidence.values()}) == 1
    assert audits[0] == audits[1]
    write(
        root / "preflight-verification.json",
        {
            "status": "passed",
            "at_beijing": now(),
            "jobs": evidence,
            "matched_physical_queries_current_noise": True,
            "full_four_rank_numerical_resume_across_LR_drop": True,
            "CPU_exact_resume": "covered by full test suite",
            "GPU_precision_rule": "weights atol2e-8/rtol1e-7; optimizer tensor max-difference<=1e-5*tensor max-magnitude; RNG/steps/scheduler exact",
            "untrained_rollout_note": "Tiny parameter differences amplify in two-update untrained rollouts; those scores are logged, not treated as accuracy results or bitwise equivalence.",
        },
    )
    # Preflight tensors are reproducible and not scientific result checkpoints.
    for job in plan["jobs"]:
        for kind in ("full", "resumed"):
            for name in ("resume.pt", "best.pt"):
                (root / "preflight" / job["name"] / kind / name).unlink()


def platform_decision(curves, *, standard=32000, cap=40000, window=4000, threshold=0.5):
    gains = {}
    for name, curve in curves.items():

        def score(row):
            return row.get("selection_body_mm", row["scores"]["body_mm"]["episode_mean"])

        before = min(score(row) for row in curve if row["step"] <= standard - window)
        after = min(score(row) for row in curve if row["step"] <= standard)
        gains[name] = before - after
    return {
        "extend_both": any(value >= threshold for value in gains.values()),
        "best_MPJPE_gain_last4000_mm": gains,
        "threshold_mm": threshold,
        "rule": f"Any case still improves by>={threshold:g}mm: both continue from native state to{cap}; no further extension.",
    }


def evaluation_worker(plan, case, shard, *, long=False):
    import torch

    from egorecover.evaluate import load_evaluation_context
    from egorecover.evaluation import evaluate_episode
    from egorecover.fk import joint_decoder
    from egorecover.io import file_sha256, save_checkpoint
    from scripts.hand_metrics import EmissionRecorder, supplementary_metrics

    torch.set_num_threads(2)
    root = Path(plan["output"]) / ("evaluation-long" if long else "evaluation400")
    selection = read(Path(plan["long_selection"] if long else plan["evaluation_selection"]))
    job = next(item for item in plan["jobs"] if item["name"] == case)
    checkpoint = Path(plan["output"]) / case / "best.pt"
    args = SimpleNamespace(
        checkpoint=checkpoint,
        data_root=Path(plan["data_root"]),
        split="val",
        device="cuda",
        synthetic=False,
        data_format="official",
        prior_mode="constant_velocity",
        prior_checkpoint=None,
        limit=None,
    )
    context = load_evaluation_context(args)
    assert context.system.config.window == job["window"]
    assert context.system.config.shape_mode == "predicted"
    assert context.saved["identity"]["prior"] == "constant_velocity"
    out = root / case / f"shard{shard}"
    write(
        out / "identity.json",
        {
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_step": context.saved["step"],
            "selection_sha256": selection["sha256"],
            "source_code": context.saved["identity"]["code"],
            "prior": "constant_velocity",
            "shape": "predicted",
            "score": "first emissions after20frame startup",
        },
    )
    records = [row for row in selection["records"] if row["shard"] == shard]
    for index, record in enumerate(records, 1):
        if long:
            episode = context.data._episode(record["source"], record["start"], record["frames"])
        else:
            actual = context.data.records[record["record_index"]]
            assert actual["episode_id"] == record["episode_id"] and actual["num_frames"] == 80
            episode = context.data.load(record["record_index"])
        recorder = EmissionRecorder(context.system)
        row = evaluate_episode(
            recorder,
            context.model,
            episode,
            joint_decoder(episode, False, context.smpl, shape_mode="predicted"),
            seed=1062,
            steps=1,
            comparison=True,
        )
        predictions, rotations = row.pop("predictions")[0], row.pop("body_rotations")
        if not long:
            extra, _, _ = supplementary_metrics(context, record, episode, recorder, predictions)
            row["comparison"].update(extra)
        name = hashlib.sha256(record["episode_id"].encode()).hexdigest()
        target = out / "predictions" / f"{name}.pt"
        save_checkpoint(
            target,
            {
                "episode_id": record["episode_id"],
                "take": record["take"],
                "joints_floor_world": predictions,
                "body_rotations_world": rotations,
                "timestamps": episode.timestamps[20:].cpu(),
                "predicted_betas": torch.cat([state.auxiliary[:, -10:] for state in recorder.states]).cpu(),
                "score_frame_indices": torch.arange(20, len(episode)),
                "selection_sha256": selection["sha256"],
            },
        )
        row["prediction_file"] = str(target.relative_to(root))
        write(out / "rows" / f"{name}.json", row)
        write(
            out / "progress.json",
            {
                "completed_clips": index,
                "target_clips": len(records),
                "phase": "complete" if index == len(records) else "evaluating",
            },
        )
        del predictions, rotations, row, recorder


def collect(plan, *, long=False):
    from egorecover.evaluation import summarize
    from scripts.run_history_ablation import load_module

    root = Path(plan["output"]) / ("evaluation-long" if long else "evaluation400")
    selection = read(Path(plan["long_selection"] if long else plan["evaluation_selection"]))
    helper = load_module(Path(plan["paired_helper"]))
    groups, results, effects = {}, [], []
    for job in plan["jobs"]:
        rows = [
            read(
                root
                / job["name"]
                / f"shard{record['shard']}"
                / "rows"
                / (hashlib.sha256(record["episode_id"].encode()).hexdigest() + ".json")
            )
            for record in selection["records"]
        ]
        assert all(row is not None for row in rows)
        assert len({row["episode"] for row in rows}) == len(selection["records"])
        summary = summarize(rows)
        metrics = {key: value["episode_mean"] for key, value in summary["comparison"].items()}
        results.append(
            {
                "case": job["name"],
                "window": job["window"],
                "clips": len(rows),
                "takes": len({row["take"] for row in rows}),
                "scored_frames": sum(row["frames"] for row in rows),
                **{key: metrics[key] for key in METRICS if key in metrics},
            }
        )
        groups[job["name"]] = rows
    for metric in METRICS:
        values = [
            (a["take"], b["comparison"][metric] - a["comparison"][metric])
            for a, b in zip(*groups.values())
            if a["comparison"].get(metric) is not None and b["comparison"].get(metric) is not None
        ]
        if values:
            effects.append({"metric": metric, "contrast": "W3_minus_W1", **helper.paired_cluster(values)})
    write(
        root / "comparison.json",
        {
            "status": "complete",
            "at_beijing": now(),
            "results": results,
            "paired_effects": effects,
            "selection_sha256": selection["sha256"],
            "aggregation": "clip_equal primary; paired take-cluster bootstrap2000",
            "timing": "concurrent quality pass; latency is not an isolated hardware benchmark",
        },
    )


def supervise(plan):
    root = Path(plan["output"])
    budgets = training_budgets(plan)
    for rel, digest in plan["source_hashes"].items():
        assert hashlib.sha256((Path(plan["source_snapshot"]) / rel).read_bytes()).hexdigest() == digest
    if not (root / "preflight-verification.json").exists():
        for kind, resume, stop in (("full", False, None), ("resumed", False, 1), ("resumed", True, 1)):
            specs = []
            for job in plan["jobs"]:
                out = root / "preflight" / job["name"] / kind
                specs.append(
                    (
                        job["name"],
                        command(plan, job, out, preflight=True, resume=resume, stop_after=stop),
                        out,
                        job["gpus"],
                    )
                )
            run_jobs(plan, specs, "preflight-" + kind + ("-resume" if resume else "-start"))
        verify_preflight(plan)
    for index, budget in enumerate(budgets):
        if index:
            curves = {job["name"]: read(root / job["name"] / "selection.json")["curve"] for job in plan["jobs"]}
            decision = platform_decision(curves, standard=budgets[0], cap=budget)
            write(root / "platform-decision.json", decision)
            if not decision["extend_both"]:
                break
        specs = []
        for job in plan["jobs"]:
            out = root / job["name"]
            resume = (out / "resume.pt").exists()
            step = continuation_step(out) if resume else 0
            if step < budget:
                specs.append(
                    (job["name"], command(plan, job, out, resume=resume, stop_after=budget - step), out, job["gpus"])
                )
        if specs:
            revision = "-" + plan["runtime_revision"] if plan.get("runtime_revision") else ""
            run_jobs(plan, specs, f"training-to{budget}{revision}")
    for job in plan["jobs"]:
        export_checkpoint(root / job["name"] / "resume.pt", root / job["name"] / "final-ema.pt")
    for long in (False, True):
        phase = "evaluation-long" if long else "evaluation400"
        specs = []
        for index, job in enumerate(plan["jobs"]):
            for shard in range(4):
                out = root / phase / job["name"] / f"shard{shard}"
                argv = [
                    plan["python"],
                    "-m",
                    "scripts.run_formal_mainline",
                    "eval-worker",
                    "--plan",
                    str(root / "plan.json"),
                    "--case",
                    job["name"],
                    "--shard",
                    str(shard),
                ] + (["--long"] if long else [])
                specs.append((job["name"] + f"/{shard}", argv, out, str(index * 4 + shard)))
        run_jobs(plan, specs, phase)
        collect(plan, long=long)
    write(
        root / "status.json",
        {
            "phase": plan.get("training_role", "foundation") + "_complete_awaiting_review",
            "at_beijing": now(),
            "P_training_started": False,
            "training_role": plan.get("training_role", "foundation"),
            "standard_budget": plan.get("standard_updates", 32000),
            "conditional_cap": plan.get("native_checkpoint_total_cap", 40000),
            "training_budgets": list(budgets),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("supervise", "monitor", "eval-worker"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    sys.path.insert(0, plan["source_snapshot"])
    if args.action == "monitor":
        print(json.dumps(read(Path(plan["output"]) / "status.json"), ensure_ascii=False, indent=2))
    elif args.action == "eval-worker":
        evaluation_worker(plan, args.case, args.shard, long=args.long)
    else:
        with (Path(plan["output"]) / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                supervise(plan)
            except Exception as error:
                write(
                    Path(plan["output"]) / "status.json", {"phase": "failed", "at_beijing": now(), "error": str(error)}
                )
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
