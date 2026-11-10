"""Supervise the fixed-C single-frame reference diagnostic and paired reports."""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    from egorecover.io import save_json

    save_json(path, value)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def command(plan, directory, shard, *, pilot=False):
    result = [
        plan["python"],
        "-m",
        "egorecover.reference_diagnostic",
        "--checkpoint",
        plan["checkpoint"],
        "--prior-checkpoint",
        plan["prior_checkpoint"],
        "--prior-mode",
        "learned",
        "--data-root",
        plan["data_root"],
        "--selection-file",
        plan["selection_file"],
        "--baseline-root",
        plan["baseline_root"],
        "--output",
        str(directory),
        "--device",
        "cuda",
        "--shard-index",
        str(shard),
        "--shard-count",
        "8",
        "--anchors",
        "20",
        "40",
        "60",
        "--horizons",
        "1",
        "2",
        "4",
        "10",
        "--future-seeds",
        "1062",
        "1063",
        "1064",
        "--prefix-seed",
        "1062",
        "--steps",
        "1",
        "--aggregation",
        "episode",
        "--verify-sham",
        "--bootstrap-samples",
        "0",
        "--threads",
        "2",
    ]
    if pilot:
        result += ["--episode-limit", "2"]
    return result


def verify_assets(plan):
    from egorecover.io import code_identity, file_sha256

    for key in ("checkpoint", "prior_checkpoint", "selection_file"):
        if file_sha256(plan[key]) != plan[key + "_sha256"]:
            raise ValueError(f"Diagnostic asset changed: {key}")
    if code_identity() != plan["code_identity"]:
        raise ValueError("Diagnostic source snapshot changed.")


def run_wave(plan, *, pilot):
    root = Path(plan["output"])
    processes = []
    phase = "pilot" if pilot else "diagnostic400"
    try:
        for shard in range(1 if pilot else 8):
            directory = root / ("pilot" if pilot else f"shard{shard}")
            directory.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                environment.pop(key, None)
            environment.update(
                CUDA_VISIBLE_DEVICES=str(shard),
                PYTHONPATH=plan["source_snapshot"],
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONUNBUFFERED="1",
                SMPLX_MODEL_PATH=plan["smpl_directory"],
            )
            log = (directory / "worker.log").open("a")
            process = subprocess.Popen(
                command(plan, directory, shard, pilot=pilot),
                cwd=plan["source_snapshot"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((shard, process, directory, log))
        while True:
            jobs = [
                {"gpu": shard, "pid": p.pid, "exit_code": p.poll(), "progress": read(directory / "progress.json")}
                for shard, p, directory, _ in processes
            ]
            write(root / "status.json", {"phase": phase, "at_beijing": now(), "jobs": jobs})
            if any(job["exit_code"] not in (None, 0) for job in jobs):
                raise RuntimeError(f"A {phase} worker failed; inspect its worker.log.")
            if all(job["exit_code"] == 0 for job in jobs):
                break
            time.sleep(10)
    finally:
        for _, process, _, log in processes:
            if process.poll() is None:
                process.terminate()
            log.close()


def collect_packets(plan, *, pilot=False):
    root = Path(plan["output"])
    selection = read(plan["selection_file"])
    if not pilot and (len(selection["records"]) != 400 or selection["takes"] != 212):
        raise ValueError("Reference diagnostic requires the complete fixed400/212-take cohort.")
    selected = selection["records"][::8][:2] if pilot else selection["records"]
    expected = {row["episode_id"]: row for row in selected}
    packets = []
    directories = [root / "pilot"] if pilot else [root / f"shard{s}" for s in range(8)]
    identity_keys = ("checkpoint_sha256", "prior_checkpoint_sha256", "selection_file_sha256")
    for directory in directories:
        identity = read(directory / "identity.json")
        if identity is None or any(identity[key] != plan[key] for key in identity_keys):
            raise ValueError("Missing or inconsistent reference worker identity.")
        if identity["selection"]["sha256"] != selection["sha256"] or identity["code"] != plan["code_identity"]:
            raise ValueError("Reference worker selection/source identity changed.")
        for path in sorted((directory / "episodes").glob("*.json")):
            packet = read(path)
            if packet["episode"] not in expected:
                raise ValueError("Unexpected episode in reference diagnostic.")
            record = expected[packet["episode"]]
            if packet["take"] != record["take"] or packet["frames"] != 80 or len(packet["rows"]) != 72:
                raise ValueError("Reference diagnostic does not match the complete planned clip.")
            combinations = {(row["intervention"], row["anchor"], row["horizon"], row["draw"]) for row in packet["rows"]}
            wanted = {
                (mode, anchor, horizon, draw)
                for mode in ("body_pose", "full_state")
                for anchor in (20, 40, 60)
                for horizon in (1, 2, 4, 10)
                for draw in (1062, 1063, 1064)
            }
            if combinations != wanted or any(
                row["episode"] != packet["episode"] or row["take"] != packet["take"] for row in packet["rows"]
            ):
                raise ValueError("Reference diagnostic has missing or mismatched paired conditions.")
            checks = packet["verification"]
            if checks["sham_checks"] != 3 or checks["baseline_comparisons"] != 71 or not packet["baseline_sha256"]:
                raise ValueError("Missing baseline reproduction or sham verification.")
            packets.append(packet)
    if len(packets) != len(expected) or {p["episode"] for p in packets} != set(expected):
        raise ValueError("Reference diagnostic is incomplete or contains duplicate clips.")
    return packets


def verification(packets):
    return {
        "status": "passed",
        "at_beijing": now(),
        "clips": len(packets),
        "takes": len({p["take"] for p in packets}),
        "sham_checks": sum(p["verification"]["sham_checks"] for p in packets),
        "baseline_comparisons": sum(p["verification"]["baseline_comparisons"] for p in packets),
        "baseline_max_abs_m": max(p["verification"]["baseline_max_abs_m"] for p in packets),
        "raw_paired_rows": sum(len(p["rows"]) for p in packets),
    }


def collect(plan):
    from egorecover.reference_statistics import summarize_reference_rows

    verify_assets(plan)
    packets = collect_packets(plan)
    rows = [row for packet in packets for row in packet["rows"]]
    mature = [row for row in rows if row["anchor"] in (40, 60)]
    options = {"bootstrap_samples": 2000, "bootstrap_seed": 1062}

    def summarize(selected, aggregation="episode"):
        return summarize_reference_rows(selected, aggregation=aggregation, **options)

    report = {
        "status": "complete",
        "at_beijing": now(),
        "plan": plan,
        "verification": verification(packets),
        "primary_mature": summarize(mature),
        "all_anchors": summarize(rows),
        "by_anchor": {str(t): summarize([r for r in rows if r["anchor"] == t]) for t in (20, 40, 60)},
        "take_equal_supplementary": {"mature": summarize(mature, "take"), "all": summarize(rows, "take")},
        "interpretation": "Primary: body_pose, anchors40/60, horizon1; positive delta favors reference. Single C checkpoint; diagnostic evidence does not establish a paired-loss training benefit.",
    }
    output = Path(plan["output"])
    write(output / "report.json", report)
    csv_rows = []
    groups = {"primary_mature40_60": report["primary_mature"], "all_anchors": report["all_anchors"]}
    groups.update({f"anchor{t}": result for t, result in report["by_anchor"].items()})
    for group, summary in groups.items():
        for mode, horizons in summary["main"].items():
            for horizon, table in horizons.items():
                csv_rows.append(
                    {
                        "group": group,
                        "intervention": mode,
                        "horizon": horizon,
                        **table["counts"],
                        **table["means"],
                        **{key + "_percent": value for key, value in table["percentages"].items()},
                        "strict_reference_worse_percent": 100 * table["sensitivity"]["0.0"]["reference_worse"],
                        "delta_ci95_low_mm": table["ci95"]["delta_mm"][0],
                        "delta_ci95_high_mm": table["ci95"]["delta_mm"][1],
                    }
                )
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write(output / "verification.json", report["verification"])
    write(output / "status.json", {"phase": "complete", "at_beijing": now(), "clips": len(packets)})
    primary = report["primary_mature"]["main"]["body_pose"]["1"]
    print(
        json.dumps(
            {"status": "complete", "primary": {key: primary[key] for key in ("means", "percentages", "ci95")}},
            ensure_ascii=False,
        ),
        flush=True,
    )


def supervise(plan):
    root = Path(plan["output"])
    try:
        verify_assets(plan)
        if not (root / "pilot-verification.json").exists():
            run_wave(plan, pilot=True)
            write(root / "pilot-verification.json", verification(collect_packets(plan, pilot=True)))
        run_wave(plan, pilot=False)
        write(root / "status.json", {"phase": "collecting", "at_beijing": now()})
        collect(plan)
    except Exception as error:
        write(root / "status.json", {"phase": "failed", "at_beijing": now(), "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("supervise", "collect", "monitor"))
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if args.action == "monitor":
        print(json.dumps(read(Path(plan["output"]) / "status.json"), ensure_ascii=False, indent=2))
    else:
        {"supervise": supervise, "collect": collect}[args.action](plan)


if __name__ == "__main__":
    main()
