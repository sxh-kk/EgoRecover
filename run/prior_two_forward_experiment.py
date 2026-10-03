"""Wait for existing EgoRecover experiments, then run the independent P comparisons."""

import argparse
import json
from pathlib import Path
import time

from egorecover.evaluation_resume import atomic_json
from run.adapt_prior_on_predictions import ABLATION, STAGES, dependency_status
from run.complete_stages import Queue, Task, append_log, command, now
from run.summarize_paired_development import interval


CASES = {"A_gt_dense": (False, 0), "B_two_forward_dense": (True, 0),
         "C_gt_fk": (False, 1), "D_two_forward_fk": (True, 1)}
OLD_PRIOR = STAGES / "stage2_budget/72take_2400/prior.pt"


def tasks_for(output, labels, seeds, *, baselines=False, gpus=(4, 5, 6, 7)):
    tasks = []
    for seed in seeds:
        for label in labels:
            two_forward, fk = CASES[label]
            path = output / "train" / f"{label}_s{seed}"
            baseline = baselines and label == "A_gt_dense" and seed == 62
            tasks.append(Task(f"{label}_s{seed}", command("run.train_prior_two_forward", data=output / "data",
                output=path, two_forward=two_forward, fk_weight=fk, seed=seed, baselines=baseline,
                old_prior=OLD_PRIOR if baseline else None), path, "prior", gpu=gpus[len(tasks) % len(gpus)]))
    return tasks


def collect_results(tasks):
    import torch
    reports = {task.name: json.loads(task.report.read_text()) for task in tasks}
    by_seed = {}
    matched_keys = ("data_sha256", "smplx_asset_sha256", "code_sha256", "steps", "batch_size",
                    "eval_every", "lr", "weight_decay", "selection", "reference_mode")
    reference_report = next(iter(reports.values()))
    for report in reports.values():
        if any(report[key] != reference_report[key] for key in matched_keys):
            raise ValueError("P comparisons use different data, code or matched training settings.")
    for name, report in reports.items():
        seed = report["training_seed"]
        state = torch.load(next(t.output for t in tasks if t.name == name) / "initial.pt",
                           map_location="cpu", weights_only=True)["state_dict"]
        if seed in by_seed and any(not torch.equal(value, by_seed[seed][key]) for key, value in state.items()):
            raise ValueError("Matched P cases did not start from identical weights.")
        by_seed[seed] = state
    scores = {name: report["results"]["prior"] for name, report in reports.items()}
    comparisons = {}
    for name, result in scores.items():
        seed = reports[name]["training_seed"]
        baseline = f"A_gt_dense_s{seed}"
        if name == baseline:
            continue
        current = result["single"]["fk22_mm"]["per_take"]
        reference = scores[baseline]["single"]["fk22_mm"]["per_take"]
        differences = [current[take] - reference[take] for take in sorted(current)]
        comparisons[f"{name}_minus_A"] = {"mean_mm": sum(differences) / len(differences),
                                            "take_bootstrap_95ci_mm": interval(differences)}
    return {"scores": scores, "comparisons": comparisons,
            "baselines": {key: value for key, value in reports["A_gt_dense_s62"]["results"].items() if key != "prior"},
            "holdout_used": False, "scope": "Development selection; seed62 screening, seeds63/64 confirmation."}


def conclusion_lines(result, winner):
    scores = result["scores"]
    a = scores["A_gt_dense_s62"]["single"]["fk22_mm"]["mean"]
    changes = {label: scores[f"{label}_s62"]["single"]["fk22_mm"]["mean"] - a
               for label in list(CASES)[1:]}
    deltas = result["candidate_minus_A_by_seed_mm"]
    trend = ("三个种子均降低下一帧误差" if all(v < 0 for v in deltas.values()) else
             "三个种子均未降低下一帧误差" if all(v >= 0 for v in deltas.values()) else
             "不同种子的改善方向不一致")
    rollout = {str(seed): scores[f"{winner}_s{seed}"]["rollout"]["1000"]["mean"] -
               scores[f"A_gt_dense_s{seed}"]["rollout"]["1000"]["mean"] for seed in (62, 63, 64)}
    cv = result["baselines"]["constant_velocity"]["single"]["fk22_mm"]["mean"]
    candidate = sum(scores[f"{winner}_s{seed}"]["single"]["fk22_mm"]["mean"] for seed in (62, 63, 64)) / 3
    return [f"seed62 下一帧 FK 相对 A 的变化：Two-Forward(B) {changes['B_two_forward_dense']:+.3f} mm；"
            f"FK监督(C) {changes['C_gt_fk']:+.3f} mm；组合(D) {changes['D_two_forward_fk']:+.3f} mm。负值表示改善。",
            f"候选 {winner} 的种子复核：{trend}；seed62/63/64 相对 A 分别为 "
            + "/".join(f"{deltas[str(seed)]:+.3f}" for seed in (62, 63, 64)) + " mm。",
            "1秒自反馈相对 A 的变化（seed62/63/64）："
            + "/".join(f"{rollout[str(seed)]:+.3f}" for seed in (62, 63, 64)) + " mm；正值表示自反馈退化。",
            f"候选三种子平均下一帧FK {candidate:.3f} mm；常速度基线 {cv:.3f} mm。"
            "以上是dev选模结果，尚未验证独立holdout或接入G后的收益。"]


def write_figures(output, summary, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from egorecover.annotations import BODY_PARENTS
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    cases = {label: summary["scores"][f"{label}_s62"] for label in CASES}
    cases.update(summary["baselines"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, case in cases.items():
        xs = sorted(int(x) for x in case["rollout"])
        ax.plot(xs, [case["rollout"][str(x)]["mean"] for x in xs], marker="o", label=label)
    ax.set(xlabel="Prediction horizon (ms)", ylabel="World SMPL22 FK MPJPE (mm)",
           title="P-only dev rollout: fixed GT starts, seed 62")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "rollout_errors.png", dpi=160)
    fig.savefig(figures / "rollout_errors.pdf")
    plt.close(fig)
    chosen = {name: torch.load(cases[name]["artifact"], weights_only=True, map_location="cpu")["rollout"]
              for name in ("A_gt_dense", winner, "constant_velocity")}
    reference = chosen["A_gt_dense"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), subplot_kw={"projection": "3d"})
    for take_id, (take, ax) in enumerate(zip(reference["takes"], axes.flat)):
        row = ((reference["indices"][:, 0] == take_id) & (reference["indices"][:, 1] == 80)).nonzero()[0, 0]
        truth = reference["gt_joints"][10][row]
        offset = truth[0].clone()
        for name, joints in [("GT", truth)] + [(name, arrays["joints"][10][row]) for name, arrays in chosen.items()]:
            joints = (joints - offset).numpy()
            for j, parent in enumerate(BODY_PARENTS[1:], 1):
                ax.plot(joints[[parent, j], 0], joints[[parent, j], 1], joints[[parent, j], 2],
                        color={"GT": "black", "A_gt_dense": "tab:red", winner: "tab:blue",
                               "constant_velocity": "tab:green"}[name], alpha=.65, label=name if j == 1 else None)
        ax.set_title(take, fontsize=8)
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
        radius = max(hi - lo for lo, hi in limits) / 2
        for setter, (lo, hi) in zip((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d), limits):
            setter((lo + hi) / 2 - radius, (lo + hi) / 2 + radius)
        ax.set_box_aspect((1, 1, 1))
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Every dev take; fixed start t=80; pose at +1000ms; GT-root centered for display only")
    fig.tight_layout()
    fig.savefig(figures / "pose_examples.png", dpi=160)
    fig.savefig(figures / "pose_examples.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--gpus", type=int, nargs=4, default=[4, 5, 6, 7])
    parser.add_argument("--skip-loss-ablation-wait", action="store_true",
                        help="Start P after the four stages without waiting for G loss evaluation.")
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or any(gpu not in range(8) for gpu in args.gpus):
        parser.error("Provide four distinct GPU indices in 0..7.")
    queue = Queue(args.output, args.gpus, 30, allow_shared=True, minimum_free_gib=32)
    plan = {"cases": {key: {"two_forward": value[0], "geometry_weight": 1, "fk_weight": value[1]}
                       for key, value in CASES.items()}, "steps": 2400, "batch_size": 32,
            "screening_seed": 62, "confirmation_seeds": [63, 64],
            "after_queues": [str(STAGES / "queue.json")] + (
                [] if args.skip_loss_ablation_wait else [str(ABLATION / "queue.json")]),
            "gpus": args.gpus, "allow_shared_with_lingbot": True, "minimum_free_gib": 32,
            "method": "docs/experiments/p-motionstreamer.md"}
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        previous = json.loads(plan_path.read_text())
        scheduling_only = all(previous.get(key) == value for key, value in plan.items()
                              if key not in ("after_queues", "gpus"))
        training_started = any(entry["kind"] != "prior_data" for entry in queue.state["tasks"].values())
        if training_started or not scheduling_only:
            raise ValueError("Deferred P experiment plan changed.")
        atomic_json(args.output / f"plan.before_schedule_change_{time.time_ns()}.json", previous)
        append_log("P 实验调度配置调整", [f"GPU {previous['gpus']} → {args.gpus}；等待队列 {plan['after_queues']}。",
                   "P 训练尚未开始，只改变资源/等待配置；沿用已完成的数据准备，训练与评估协议不变。"])
    atomic_json(plan_path, plan)
    if args.retry_failed:
        for entry in queue.state["tasks"].values():
            if entry["status"] == "failed":
                entry["status"] = "pending"
    try:
        previous_dependencies = None
        while True:
            ready, dependencies = dependency_status(plan["after_queues"])
            queue.state.update(status="waiting_for_existing_experiments", dependencies=dependencies)
            queue.save()
            if dependencies != previous_dependencies:
                queue.log(f"Existing EgoRecover dependencies: {dependencies}")
                previous_dependencies = dependencies
            if ready:
                break
            time.sleep(30)
        queue.state.update(status="running")
        queue.save()
        # A completed data task retains its original GPU as historical provenance.
        data_gpu = queue.state["tasks"].get("prepare_data", {}).get("assigned_gpu", args.gpus[0])
        data = Task("prepare_data", command("run.prepare_prior_sequences", output=args.output / "data"),
                    args.output / "data", "prior_data", 60, gpu=data_gpu)
        queue.run([data])
        screening = tasks_for(args.output, list(CASES), [62], baselines=True, gpus=args.gpus)
        queue.run(screening)
        summary = collect_results(screening)
        winner = min(list(CASES)[1:], key=lambda label: summary["scores"][f"{label}_s62"]["single"]["fk22_mm"]["mean"])
        atomic_json(args.output / "screening.json", {**summary, "candidate_for_confirmation": winner})
        confirmation = tasks_for(args.output, ["A_gt_dense", winner], [63, 64], gpus=args.gpus)
        queue.run(confirmation)
        result = collect_results(screening + confirmation)
        result.update(candidate_for_confirmation=winner, completed=True)
        result["candidate_minus_A_by_seed_mm"] = {str(seed): result["comparisons"][f"{winner}_s{seed}_minus_A"]["mean_mm"]
                                                 for seed in (62, 63, 64)}
        conclusions = conclusion_lines(result, winner)
        result["conclusions"] = conclusions
        write_figures(args.output, result, winner)
        atomic_json(args.output / "summary.json", result)
        rows = ["# 独立 P 改进实验结果", "", "固定 train/dev 划分；dev 选模，holdout 未使用。", "",
                "| 配置 | 下一帧 FK (mm) | 1 秒自反馈 FK (mm) |", "|---|---:|---:|"]
        for name, value in result["scores"].items():
            rows.append(f"| {name} | {value['single']['fk22_mm']['mean']:.3f} | {value['rollout']['1000']['mean']:.3f} |")
        rows += ["", "## 实验结论", "", *[f"- {line}" for line in conclusions],
                 "", "![误差曲线](figures/rollout_errors.png)", "", "![固定片段骨架](figures/pose_examples.png)"]
        (args.output / "RESULTS.md").write_text("\n".join(rows) + "\n")
        queue.state.update(status="complete", completed_at=now())
        queue.save()
        append_log("独立 P 改进实验结论", [*conclusions,
                   f"完整结果表与图：`{args.output / 'RESULTS.md'}`；配对区间和逐take指标：`{args.output / 'summary.json'}`。"])
    except Exception as error:
        queue.state.update(status="failed", error=repr(error))
        queue.save()
        append_log("独立 P 改进队列停止", [f"`{args.output}`；原因 `{error}`。"])
        raise


if __name__ == "__main__":
    main()
