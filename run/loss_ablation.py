"""Run two generator-loss settings with a common frozen P on four assigned GPUs."""

import argparse
import json
import statistics
from pathlib import Path

from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import (BOOTSTRAP, CHECKPOINT, ROOT, SIGNAL, SPLIT, Queue, Task,
                                 append_log, command, now)
from run.summarize_paired_development import interval, summarize


BASELINE = Path("exp/egorecover_stages_1_4/continuation_v1/stage2_budget/72take_2400")
BASELINE_EVAL = Path("exp/egorecover_stages_1_4/continuation_v1/stage2_eval/72take_2400/report.json")


def tasks_for(output, gpus, *, steps=2400, prior=BASELINE):
    tasks, cases = [], []
    for gpu, (geometry, fk, mode) in zip(gpus, ((0, 0, "gaussian"), (0, 0, "history"),
                                              (1, 1, "gaussian"), (1, 1, "history"))):
        label = f"g{geometry}f{fk}_{mode}"
        train = output / "train" / label
        evaluation = output / "eval" / label
        tasks.append(Task(
            f"train_{label}", command(
                "run.engineering_pilot", output=train, device="cuda", signal=SIGNAL,
                split_manifest=SPLIT, bootstrap_cache=BOOTSTRAP, checkpoint=CHECKPOINT, weight_source="ema",
                seed=62, sampling_seed=1062, prior_steps=0, fixed_prior_experiment=prior,
                source_modes=[mode], flow_steps=steps, batch_size=32, sigma=1,
                geometry_weight=geometry, fk_weight=fk, prior_selection="dense",
                flow_selection="closed_loop_fk", eval_every=200, record_loss_components=True,
            ), train, "train", gpu=gpu,
        ))
        cases.append({"label": label, "gpu": gpu, "geometry_weight": geometry, "fk_weight": fk,
                      "mode": mode, "train": str(train), "evaluation": str(evaluation)})
    for case in cases:
        label = case["label"]
        tasks.append(Task(
            f"eval_{label}", command(
                "run.evaluate_paired_development", signal=SIGNAL, split_manifest=SPLIT,
                bootstrap_cache=BOOTSTRAP, checkpoint=CHECKPOINT,
                experiment=f"{label}={case['train']}", source_modes=[case["mode"]],
                group="dev", seed=62, output=case["evaluation"],
            ), Path(case["evaluation"]), "eval", expected=36, deps=[f"train_{label}"], gpu=case["gpu"],
        ))
    return tasks, cases


def result_summary(cases, *, baseline_eval=BASELINE_EVAL):
    paths = [Path(case["evaluation"]) / "report.json" for case in cases]
    combined = summarize([baseline_eval, *paths])
    rows = json.loads(baseline_eval.read_text())["results"]
    for path in paths:
        rows.extend(json.loads(path.read_text())["results"])
    comparisons = {}
    pairs = [(case["label"], "72take_2400", case["mode"], f"{case['label']}_minus_g1f0") for case in cases]
    pairs.extend((f"g1f1_{mode}", f"g0f0_{mode}", mode, f"g1f1_minus_g0f0_{mode}")
                 for mode in ("gaussian", "history"))
    for label, reference_label, mode, comparison_label in pairs:
        for condition in ("clean", "all_variants"):
            def by_take(experiment):
                return {take: statistics.mean(row["smpl22_mpjpe_mm"] for row in rows
                        if row["experiment"] == experiment and row["mode"] == mode
                        and row["take"] == take and (condition == "all_variants" or row["variant"] == condition))
                        for take in combined["takes"]}
            reference, current = by_take(reference_label), by_take(label)
            differences = [current[take] - reference[take] for take in combined["takes"]]
            comparisons[f"{comparison_label}/{condition}"] = {
                "mean_difference_mm": statistics.mean(differences),
                "take_bootstrap_95ci_mm": interval(differences),
                "current_mean_mm": statistics.mean(current.values()),
                "baseline_mean_mm": statistics.mean(reference.values()),
            }
    return {"paired_reports": combined, "comparisons": comparisons,
            "primary_metric": "clean SMPL22 world MPJPE, take macro mean",
            "secondary_metric": "clean/freeze/drift macro mean",
            "scope": "development-set G-loss ablation with a common fixed P; one training seed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[4, 5, 6, 7])
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--minimum-free-gib", type=float, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or any(gpu not in range(8) for gpu in args.gpus) or args.steps < 1:
        parser.error("Provide four distinct GPU indices in 0..7 and positive steps.")
    tasks, cases = tasks_for(args.output, args.gpus, steps=args.steps)
    if args.dry_run:
        print(json.dumps([{"name": task.name, "gpu": task.gpu, "command": task.argv} for task in tasks], indent=2))
        return
    baseline = json.loads((BASELINE / "report.json").read_text())
    baseline_eval = json.loads(BASELINE_EVAL.read_text())
    if not baseline.get("completed") or not baseline_eval.get("completed"):
        raise ValueError("The common prior and comparison report must be complete.")
    queue = Queue(args.output, args.gpus, args.poll_seconds, allow_shared=True,
                  minimum_free_gib=args.minimum_free_gib)
    spec = {"cases": cases, "steps": args.steps, "loss_scope": "G only; shared frozen P",
            "prior_experiment": str(BASELINE), "prior_sha256": file_sha256(BASELINE / "prior.pt"),
            "baseline_evaluation": str(BASELINE_EVAL), "baseline_evaluation_sha256": file_sha256(BASELINE_EVAL),
            "checkpoint_sha256": file_sha256(Path(CHECKPOINT)),
            "bootstrap_sha256": file_sha256(Path(BOOTSTRAP)), "split_sha256": file_sha256(Path(SPLIT)),
            "primary": "clean world SMPL22 take-macro MPJPE", "secondary": "three-variant macro MPJPE",
            "code_sha256": {name: file_sha256(ROOT / name) for name in (
                "run/engineering_pilot.py", "run/loss_ablation.py", "egorecover/losses.py",
                "egorecover/fixed_prior.py", "model/history_uniegomotion.py", "egorecover/history_flow.py")},
            "train_seed": 62, "selection_seed": 1062, "evaluation_seed": 62,
            "replay": False, "holdout_used": False, "created_at": now()}
    spec_path = args.output / "experiment.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if any(old[key] != value for key, value in spec.items() if key != "created_at"):
            raise ValueError("Ablation protocol changed; use a new output directory.")
    else:
        atomic_json(spec_path, spec)
    try:
        queue.run(tasks)
        for case in cases:
            if file_sha256(Path(case["train"]) / "prior.pt") != spec["prior_sha256"]:
                raise ValueError("Ablation did not use the same fixed P.")
        result = result_summary(cases)
        atomic_json(args.output / "summary.json", result)
        lines = ["共同固定 P，Gaussian/History 各 2400 步；仅改变 G 的损失。",
                 "| 比较（当前减参照） | 当前 clean MPJPE (mm) | 差值 (mm) | take bootstrap 95% CI |",
                 "|---|---:|---:|---|"]
        for key, value in result["comparisons"].items():
            if key.endswith("/clean"):
                lines.append(f"| {key} | {value['current_mean_mm']:.3f} | {value['mean_difference_mm']:+.3f} | "
                             f"{value['take_bootstrap_95ci_mm']} |")
        (args.output / "results.md").write_text("\n".join(lines) + "\n")
        append_log("G 损失消融完成", [f"运行目录：`{args.output}`；完整结果 `summary.json`。", *lines])
        queue.state.update(status="complete", completed_at=now())
        queue.save()
    except Exception as error:
        queue.state.update(status="failed", error=str(error))
        queue.save()
        append_log("G 损失消融中断", [f"运行目录：`{args.output}`；原因：`{error}`；未完成结果不计入比较。"])
        raise


if __name__ == "__main__":
    main()
