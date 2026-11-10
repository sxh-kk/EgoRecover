"""Run the frozen-P history-training x rewrite pilot and fixed400 evaluation."""

import argparse
import csv
import hashlib
import importlib.util
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


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def load_module(path):
    spec = importlib.util.spec_from_file_location("fixed400_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_command(plan, job, output, *, preflight=False, resume=False):
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
        "2" if preflight else "2000",
        "--batch-size",
        "64",
        "--microbatch-size",
        "8",
        "--sample-reference-world-size",
        "2",
        "--sampling-rollin-frames",
        "20",
        "--rollin-frames",
        "20" if job["history"] == "mixed" else "4",
        "--history-mode",
        "mixed" if job["history"] == "mixed" else "fixed",
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
        "--output",
        str(output),
    ]
    command += ["--resume"] if resume else ["--initialize", plan["common_G_checkpoint"]]
    if not job["rewrite"]:
        command.append("--allow-initialize-rewrite-change")
    if preflight and not resume:
        command += ["--stop-after", "1"]
    return command


def environment(plan, gpus):
    result = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        result.pop(key, None)
    result.update(
        CUDA_VISIBLE_DEVICES=gpus,
        PYTHONPATH=plan["source_snapshot"],
        OMP_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        NCCL_DEBUG="WARN",
        SMPLX_MODEL_PATH=plan["smpl_directory"],
    )
    return result


def run_processes(plan, specs, *, phase):
    jobs = []
    root = Path(plan["output"])
    try:
        for name, command, directory, gpu in specs:
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / f"{phase}.log").open("x")
            process = subprocess.Popen(
                command,
                cwd=plan["source_snapshot"],
                env=environment(plan, gpu),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            jobs.append((name, process, directory, log, gpu))
        write(
            root / f"{phase}-processes.json", [{"case": name, "pid": p.pid, "gpus": gpu} for name, p, _, _, gpu in jobs]
        )
        while True:
            states = [
                {
                    "case": name,
                    "pid": p.pid,
                    "gpus": gpu,
                    "exit_code": p.poll(),
                    "progress": read(directory / "progress.json"),
                    "status": read(directory / "status.json"),
                }
                for name, p, directory, _, gpu in jobs
            ]
            write(
                root / "status.json",
                {"phase": phase, "at_beijing": now(), "jobs": states, "all_cases": [j["name"] for j in plan["jobs"]]},
            )
            if any(s["exit_code"] not in (None, 0) for s in states):
                raise RuntimeError(f"A {phase} worker failed; its log/checkpoints are preserved.")
            if all(s["exit_code"] == 0 for s in states):
                break
            time.sleep(10)
    finally:
        for _, process, _, log, _ in jobs:
            if process.poll() is None:
                process.terminate()
            log.close()


def verify_warm_start(plan):
    """Actual neutral-rollout provenance must preserve old ON/OFF outputs."""
    import torch

    from egorecover.evaluate import load_evaluation_context
    from egorecover.evaluation import evaluate_episode
    from egorecover.fk import joint_decoder
    from egorecover.system import WindowReconstructor

    torch.set_num_threads(2)
    args = SimpleNamespace(
        checkpoint=Path(plan["common_G_checkpoint"]),
        data_root=Path(plan["data_root"]),
        prior_checkpoint=Path(plan["prior_checkpoint"]),
        split="val",
        device="cuda",
        synthetic=False,
        data_format="official",
        prior_mode="learned",
        limit=None,
    )
    context = load_evaluation_context(args)
    original = context.system.config
    records = read(Path(plan["output"]) / "evaluation400/selection.json")["records"][:2]
    evidence = {}
    previous = Path(plan["rewrite_diagnostic"])
    for mode in ("on", "off"):
        system = WindowReconstructor(
            context.system.codec, context.system.predictor, replace(original, revise_recent=mode == "on")
        )
        differences = []
        for index, record in enumerate(records):
            episode = context.data.load(record["record_index"])
            row = evaluate_episode(
                system, context.model, episode, joint_decoder(episode, False, context.smpl), seed=1062, steps=1
            )
            code = hashlib.sha256(episode.identifier.encode()).hexdigest()
            payload = torch.load(
                previous / f"gaussian/{mode}/shard{index % 2}/traces/{code}.pt", map_location="cpu", weights_only=True
            )
            difference = float((row["predictions"][0] - payload["first_emission_joints"]).abs().max())
            if difference != 0:
                raise RuntimeError("Tracking actual reset provenance changed a standard rollout.")
            differences.append(difference)
        evidence[mode] = differences
    write(
        Path(plan["output"]) / "warmstart-rollout-verification.json",
        {
            "status": "passed",
            "max_abs_position_differences_m": evidence,
            "note": "Two selected clips; both original policies reproduce exactly before adaptation.",
        },
    )


def verify_training(plan, *, preflight):
    import torch

    from egorecover.checkpoint import load_predictor
    from egorecover.io import code_identity, file_sha256

    root = Path(plan["output"])
    prefix = root / "preflight" if preflight else root
    budget = 2 if preflight else 2000
    signatures, reports = [], {}
    for job in plan["jobs"]:
        directory = prefix / job["name"]
        initial, identity = read(directory / "initialization.json"), read(directory / "identity.json")
        assert initial["loaded_model_sha256"] == plan["common_G_model_sha256"]
        assert initial["checkpoint_sha256"] == plan["common_G_checkpoint_sha256"]
        assert initial["checkpoint_step"] == 12000 and initial["optimizer"] == "fresh"
        assert identity["prior"] == plan["prior_checkpoint_sha256"]
        assert identity["code"] == code_identity()
        assert identity["history_training"]["shared_sampling_rollin_frames"] == 20
        assert identity["protocol"] == read(job["config"])
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == list(range(1, budget + 1))
        signatures.append([r["sampling_sha256"] for r in rows])
        for row in rows:
            counts = row["rollin_frames_counts"]
            assert row["effective_batch_size"] == 64 and row["world_size"] == 4
            assert row["target_local_frame"] == 40 and sum(counts.values()) == 64
            if job["history"] == "mixed":
                assert counts["20"] == 48 and set(counts) <= {"0", "1", "4", "20"}
            else:
                assert counts == {"4": 64}
        saved = torch.load(directory / "resume.pt", map_location="cpu", weights_only=True, mmap=True)
        assert saved["step"] == budget and len(saved["rank_rng_states"]) == 4
        assert len({r["model_sha256"] for r in saved["rank_rng_states"]}) == 1
        assert all(int(x["step"]) == budget for x in saved["optimizer"]["state"].values())
        reports[job["name"]] = {
            "steps": budget,
            "same_loaded_G": True,
            "optimizer_steps": budget,
            "sample_stream_sha256": hashlib.sha256(json.dumps(signatures[-1]).encode()).hexdigest(),
        }
        if not preflight:
            keys = ("format", "model_config", "identity", "protocol", "stats", "ema", "step", "best")
            from egorecover.io import save_checkpoint

            save_checkpoint(
                directory / "final-ema.pt", {**{k: saved[k] for k in keys}, "purpose": "fixed2000_step_evaluation"}
            )
        del saved
    assert all(value == signatures[0] for value in signatures)
    assert file_sha256(plan["prior_checkpoint"]) == plan["prior_checkpoint_sha256"]
    prior = load_predictor(plan["prior_checkpoint"], "cpu", plan["manifest"])
    assert not prior.training and not any(p.requires_grad for p in prior.parameters())
    write(
        root / ("preflight-verification.json" if preflight else "training-verification.json"),
        {
            "status": "passed",
            "at_beijing": now(),
            "all_four_physical_query_tau_epsilon_streams_identical": True,
            "P_frozen_and_unchanged": True,
            "jobs": reports,
        },
    )


def evaluation_worker(plan, case, shard):
    """Reuse the verified hand/body evaluator; add only post-hoc mature scores."""
    import torch

    import egorecover.evaluation as evaluation
    from egorecover.metrics import comparison_metrics

    torch.set_num_threads(2)
    root = Path(plan["output"]) / "evaluation400"
    sys.path.insert(0, str(root))
    shared = load_module(root / "shared_worker.py")
    shared.ROOT = root
    original = evaluation.evaluate_episode

    def with_mature(*args, **kwargs):
        row = original(*args, **kwargs)
        episode = args[2]
        row["mature_comparison"] = comparison_metrics(
            row["predictions"][0, 20:],
            episode.truth.joints[40:, :, :3, 3],
            episode.timestamps[40:],
            predicted_rotations=row["body_rotations"][20:],
            truth_rotations=episode.truth.joints[40:, :, :3, :3],
        )
        return row

    evaluation.evaluate_episode = with_mature
    shared.worker(case, shard)


def collect_evaluation(plan):
    from egorecover.evaluation import summarize

    root = Path(plan["output"]) / "evaluation400"
    shared = load_module(root / "shared_worker.py")
    records = read(root / "selection.json")["records"]
    if len(records) != 400 or len({row["take"] for row in records}) != 212:
        raise ValueError("The four-cell evaluation requires the complete fixed400/212-take cohort.")
    groups, results, effects = {}, [], []
    for job in plan["jobs"]:
        rows = [
            read(root / job["name"] / f"shard{r['shard']}/rows" / (shared.name(r["episode_id"]) + ".json"))
            for r in records
        ]
        assert all(row is not None and row["frames"] == 60 for row in rows)
        assert [r["episode"] for r in rows] == [r["episode_id"] for r in records]
        assert [r["take"] for r in rows] == [r["take"] for r in records]
        groups[job["name"]] = rows
        result = {
            "case": job["name"],
            "history": job["history"],
            "rewrite": job["rewrite"],
            "checkpoint_step": 2000,
            "common_pretraining_steps": 12000,
            "clips": 400,
            "takes": 212,
        }
        for metric in shared.KEYS:
            result[metric] = sum(r["comparison"][metric] for r in rows) / len(rows)
        result["mature_metrics"] = {
            metric: sum(r["mature_comparison"][metric] for r in rows) / len(rows) for metric in shared.KEYS[:10]
        }
        results.append(result)
        write(
            root / job["name"] / "report.json",
            {
                "identity": read(root / job["name"] / "shard0/identity.json"),
                "results": {"1": {"rows": {"rollout": rows}, "summary": {"rollout": summarize(rows)}}},
            },
        )
    contrasts = [
        ("history_effect_ON", "C_mixed_on", "A_short_on"),
        ("history_effect_OFF", "D_mixed_off", "B_short_off"),
        ("rewrite_OFF_minus_ON_short", "B_short_off", "A_short_on"),
        ("rewrite_OFF_minus_ON_mixed", "D_mixed_off", "C_mixed_on"),
    ]
    for subset, key in (("all60", "comparison"), ("mature40", "mature_comparison")):
        metrics = shared.KEYS if key == "comparison" else shared.KEYS[:10]
        for metric in metrics:
            for label, b, a in contrasts:
                values = [(r["take"], s[key][metric] - r[key][metric]) for r, s in zip(groups[a], groups[b])]
                effects.append({"contrast": label, "subset": subset, "metric": metric, **shared.paired_cluster(values)})
            values = [
                (a["take"], d[key][metric] - b[key][metric] - c[key][metric] + a[key][metric])
                for a, b, c, d in zip(
                    groups["A_short_on"], groups["B_short_off"], groups["C_mixed_on"], groups["D_mixed_off"]
                )
            ]
            effects.append(
                {
                    "contrast": "history_rewrite_interaction",
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
        "selection_sha256": read(root / "selection.json")["sha256"],
        "protocol": plan["evaluation_protocol"],
        "uncertainty": "paired take-cluster2000; fixed single-seed checkpoints",
        "timing": "parallel quality pass; no isolated latency claim",
        "initialization": plan["initialization_note"],
    }
    write(root / "comparison.json", report)
    fields = ["case", "history", "rewrite", "checkpoint_step", "clips", "takes", *shared.KEYS]
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def monitor(plan):
    root = Path(plan["output"])
    status = read(root / "status.json") or {}
    print("phase:", status.get("phase"), "updated:", status.get("at_beijing"))
    for job in plan["jobs"]:
        directory = root / job["name"]
        p = read(directory / "progress.json") or {}
        s = read(directory / "status.json") or {}
        selection = read(directory / "selection.json") or {}
        curve = selection.get("curve", [])
        last = curve[-1] if curve else None
        print(
            job["name"],
            s.get("status", "queued"),
            "step",
            p.get("step", 0),
            "/2000",
            "loss",
            p.get("loss"),
            "rollins",
            p.get("rollin_frames_counts"),
            "dev MPJPE",
            last["scores"]["body_mm"]["mean"] if last else None,
        )


def supervise(plan):
    root = Path(plan["output"])
    if not (root / "warmstart-rollout-verification.json").exists():
        verify_warm_start(plan)
    if not (root / "preflight-verification.json").exists():
        for wave in (plan["jobs"][:2], plan["jobs"][2:]):
            for resume in (False, True):
                phase = "preflight-" + wave[0]["name"] + ("-resume" if resume else "-startup")
                specs = [
                    (
                        job["name"],
                        train_command(plan, job, root / "preflight" / job["name"], preflight=True, resume=resume),
                        root / "preflight" / job["name"],
                        job["gpus"],
                    )
                    for job in wave
                ]
                run_processes(plan, specs, phase=phase)
        verify_training(plan, preflight=True)
    for index, wave in enumerate((plan["jobs"][:2], plan["jobs"][2:]), start=1):
        specs = [
            (job["name"], train_command(plan, job, root / job["name"]), root / job["name"], job["gpus"]) for job in wave
        ]
        run_processes(plan, specs, phase=f"training-wave{index}")
    verify_training(plan, preflight=False)
    for wave_index, wave in enumerate((plan["jobs"][:2], plan["jobs"][2:]), start=1):
        specs = []
        for job in wave:
            for shard in range(4):
                gpu = str(len(specs))
                directory = root / "evaluation400" / job["name"] / f"shard{shard}"
                command = [
                    plan["python"],
                    str(Path(__file__).resolve()),
                    "eval-worker",
                    "--plan",
                    str(root / "plan.json"),
                    "--case",
                    job["name"],
                    "--shard",
                    str(shard),
                ]
                specs.append((job["name"] + f"/shard{shard}", command, directory, gpu))
        run_processes(plan, specs, phase=f"evaluation-wave{wave_index}")
    collect_evaluation(plan)
    write(
        root / "status.json",
        {"phase": "complete", "at_beijing": now(), "comparison": str(root / "evaluation400/comparison.json")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supervise", "monitor", "eval-worker", "collect"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    plan = read(args.plan)
    sys.path.insert(0, plan["source_snapshot"])
    if args.command == "monitor":
        monitor(plan)
    elif args.command == "eval-worker":
        evaluation_worker(plan, args.case, args.shard)
    elif args.command == "collect":
        collect_evaluation(plan)
    else:
        try:
            supervise(plan)
        except Exception as exc:
            write(Path(plan["output"]) / "status.json", {"phase": "failed", "at_beijing": now(), "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
