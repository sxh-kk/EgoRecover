"""Adapt only P on frozen G histories, after both existing experiments finish."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import ROOT, SIGNAL, SPLIT, append_log, now, shared_gpu_indices
from run.summarize_paired_development import interval


STAGES = Path("exp/egorecover_stages_1_4/continuation_v1")
ABLATION = Path("exp/egorecover_loss_ablation/g00_g11_v1")


def dependency_status(paths):
    result = {}
    for path in paths:
        try:
            value = json.loads(Path(path).read_text())
            tasks = value.get("tasks", {})
            complete = value.get("status") == "complete" and bool(tasks) and all(
                task.get("status") == "complete" for task in tasks.values())
            result[str(path)] = {"status": value.get("status"), "ready": complete}
        except (OSError, ValueError):
            result[str(path)] = {"status": "missing_or_unreadable", "ready": False}
    return bool(result) and all(item["ready"] for item in result.values()), result


def save_torch(path, value):
    import torch
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args, status):
    import torch
    from dataset.smpl_utils import get_smpl
    from egorecover.codec import MotionCodec
    from egorecover.data import read_handoff, select_handoff_dataset
    from egorecover.losses import physical_objective
    from egorecover.prior import HistoryPrior
    from egorecover.prior_adaptation import FixedHistoryFrames, evaluate, predict, validate_teachers

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    split_path = Path(SPLIT)
    manifest = json.loads(split_path.read_text())
    handoff = read_handoff(SIGNAL)
    entry = select_handoff_dataset(handoff, profile="pilot", spec_sha256=manifest["dataset_spec_sha256"])
    records = {row["variant_id"]: row for row in
               (json.loads(line) for line in Path(entry["manifest"]).read_text().splitlines() if line.strip())}
    stats_path = Path(handoff["source_root"]) / "uniegomotion/v4_beta_ee_train_stats.pt"
    caches = {group: FixedHistoryFrames(args.histories / group / "frames.pt", group=group,
              split_path=split_path, stats_path=stats_path, records=records) for group in ("train", "dev")}
    teacher, dev = caches["train"], caches["dev"]
    validate_teachers(teacher, dev)
    source = Path(teacher.identity["generator_experiment"])
    source_report = json.loads((source / "report.json").read_text())
    if not source_report.get("completed"):
        raise ValueError("The frozen teacher experiment is incomplete.")
    frozen_files = {source / "prior.pt": teacher.identity["p_sha256"],
                    **{source / f"g_{mode}.pt": digest for mode, digest in teacher.identity["g_sha256"].items()}}
    if any(file_sha256(path) != digest for path, digest in frozen_files.items()):
        raise ValueError("Frozen teacher checkpoints differ from the history cache.")
    asset = Path(os.environ.get("SMPLX_MODEL_PATH", ROOT / "body_models/smplx")) / "SMPLX_NEUTRAL.npz"
    identity = {"scope": "P-only fixed-predicted-history adaptation; development selection",
                "teacher_identity": teacher.identity, "cache_sha256": {k: v.sha256 for k, v in caches.items()},
                "steps": args.steps, "selection_every": args.selection_every, "batch_size": args.batch_size,
                "seed": args.seed, "lr": 3e-4, "geometry_weight": 1., "fk_weight": 0.,
                "replay_probability": 1., "holdout_used": False,
                "selection": "next-frame world SMPL22 FK; equal take/mode/variant macro mean",
                "smplx_asset_sha256": file_sha256(asset),
                "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                    (Path(__file__).resolve(), ROOT / "egorecover/prior_adaptation.py",
                     ROOT / "egorecover/prior.py", ROOT / "egorecover/losses.py", ROOT / "egorecover/fk.py")}}
    if source_report.get("smplx_asset_sha256") != identity["smplx_asset_sha256"]:
        raise ValueError("SMPL-X asset differs from the audited frozen-teacher experiment.")
    identity_path = args.output / "experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cannot resume with changed data, teacher, code, or hyperparameters.")
    atomic_json(identity_path, identity)
    codec = MotionCodec(torch.load(stats_path, map_location="cpu", weights_only=False),
                        reference_mode=teacher.identity["reference_mode"]).to(device)
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    prior = HistoryPrior().to(device)
    initial = torch.load(source / "prior.pt", map_location="cpu", weights_only=True)
    prior.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    baseline_scores, baseline_errors = evaluate(prior, dev, codec, smpl, device, args.batch_size, baselines=True)
    best_score = baseline_scores["prior"]
    best_errors = baseline_errors["prior"]
    best_state = {key: value.detach().cpu().clone() for key, value in prior.state_dict().items()}
    best_step, start = 0, 0
    curve = [{"step": 0, **best_score}]
    resume_path = args.output / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch.")
        prior.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best_step, best_state = saved["step"], saved["best_step"], saved["best_state"]
        best_score, best_errors, curve = saved["best_score"], saved["best_errors"], saved["curve"]
    append_log("固定预测历史 P 适配开始/恢复", [f"目录 `{args.output}`；GPU{status['gpu']}；从额外 step {start} 开始。",
               f"两套前置队列已完成；冻结 teacher `{source}`；train/dev 帧数 {len(teacher)}/{len(dev)}。",
               "仅 train 缓存参与梯度更新；原 P/保持/常速度和新 P 共用固定 dev 历史，按 FK 选模，step0 可入选。"])
    print("Baselines", json.dumps(baseline_scores), flush=True)
    for step in range(start + 1, args.steps + 1):
        prior.train()
        batch = teacher.batch(torch.randint(len(teacher), (args.batch_size,), generator=generator), device)
        optimizer.zero_grad(set_to_none=True)
        loss = physical_objective(codec, predict(prior, batch), batch, geometry_weight=1., fk_weight=0.)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"Nonfinite training loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0:
            print(f"P adaptation {step}/{args.steps}: train={float(loss.detach()):.6f}", flush=True)
        if step % args.selection_every == 0 or step == args.steps:
            scores, errors = evaluate(prior, dev, codec, smpl, device, args.batch_size)
            score = scores["prior"]
            curve.append({"step": step, "train_loss": float(loss.detach()), **score})
            if score["macro_fk22_mm"] < best_score["macro_fk22_mm"]:
                best_step, best_score, best_errors = step, score, errors["prior"]
                best_state = {key: value.detach().cpu().clone() for key, value in prior.state_dict().items()}
            save_torch(resume_path, {"identity": identity, "step": step, "state_dict": prior.state_dict(),
                "optimizer": optimizer.state_dict(), "sampling_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "best_state": best_state, "best_step": best_step, "best_score": best_score,
                "best_errors": best_errors, "curve": curve})
            status.update(status="training", step=step, selected_step=best_step,
                          best_dev_fk22_mm=best_score["macro_fk22_mm"], updated_at=now())
            atomic_json(args.output / "status.json", status)
            atomic_json(args.output / "selection.json", curve)
            print(f"DEV step {step}: FK={score['macro_fk22_mm']:.3f} mm; best={best_step}", flush=True)
            append_log("P 适配 dev 选点", [f"`{args.output}`；step {step}/{args.steps}；"
                       f"固定 dev FK={score['macro_fk22_mm']:.3f} mm；"
                       f"当前最佳 {best_score['macro_fk22_mm']:.3f} mm @step {best_step}。"])
    if any(file_sha256(path) != digest for path, digest in frozen_files.items()) or any(
        file_sha256(args.histories / group / "frames.pt") != frames.sha256 for group, frames in caches.items()):
        raise ValueError("Frozen teacher or history data changed during P adaptation.")
    checkpoint = {**{key: value for key, value in initial.items() if key != "state_dict"},
                  "state_dict": best_state, "selected_step": best_step,
                  "scope": identity["scope"], "selection": "fixed_predicted_history_fk",
                  "adaptation_identity": identity}
    save_torch(args.output / "prior.pt", checkpoint)
    scores = {"original_prior": baseline_scores["prior"], "adapted_prior": best_score,
              "hold": baseline_scores["hold"], "constant_velocity": baseline_scores["constant_velocity"]}
    paired = {}
    for name in ("original_prior", "hold", "constant_velocity"):
        differences = [best_score["per_take_mm"][take] - scores[name]["per_take_mm"][take]
                       for take in sorted(best_score["per_take_mm"])]
        paired[f"adapted_prior_minus_{name}"] = {"mean_mm": sum(differences) / len(differences),
                                                 "take_bootstrap_95ci_mm": interval(differences)}
    save_torch(args.output / "dev_errors.pt", {"cache_sha256": dev.sha256, "metadata": dev.metadata,
               "variants": dev.variants, "errors_mm": {"original_prior": baseline_errors["prior"],
                "adapted_prior": best_errors, "hold": baseline_errors["hold"],
                "constant_velocity": baseline_errors["constant_velocity"]}})
    report = {**identity, "scores": scores, "paired": paired, "selected_step": best_step, "curve": curve,
              "prior_sha256": file_sha256(args.output / "prior.pt"), "completed": True,
              "interpretation": "Selected on these dev states; not a holdout or closed-loop improvement claim."}
    atomic_json(args.output / "report.json", report)
    lines = [f"{name}：{score['macro_fk22_mm']:.3f} mm" for name, score in scores.items()]
    append_log("固定预测历史 P 适配完成", [f"结果 `{args.output / 'report.json'}`；额外选中 step {best_step}。", *lines,
               "指标为固定 G 历史上的下一帧 world SMPL22 FK；配对 take 区间见报告。dev 同时用于选模，不能据此声称 holdout 或闭环性能提升。"])
    print(json.dumps({"scores": scores, "paired": paired, "selected_step": best_step}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--histories", type=Path, default=STAGES / "histories")
    parser.add_argument("--after-queue", type=Path, action="append")
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--selection-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()
    if not args.gpus or not set(args.gpus) <= {4, 5, 6, 7}:
        parser.error("This experiment is restricted to GPU4–7.")
    if min(args.steps, args.selection_every, args.batch_size) < 1:
        parser.error("Steps and batch sizes must be positive.")
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "report.json").exists():
            report = json.loads((args.output / "report.json").read_text())
            if report.get("completed") and file_sha256(args.output / "prior.pt") == report["prior_sha256"]:
                atomic_json(args.output / "status.json", {"status": "complete", "verified_at": now()})
                print("Experiment already complete.", flush=True)
                return
        dependencies = args.after_queue or [STAGES / "queue.json", ABLATION / "queue.json"]
        status = {"controller_pid": os.getpid(), "created_at": now(), "status": "waiting_for_existing_experiments"}
        plan = {"dependencies": [str(path) for path in dependencies], "histories": str(args.histories),
                "gpus": args.gpus, "steps": args.steps, "selection_every": args.selection_every,
                "batch_size": args.batch_size, "seed": args.seed,
                "allow_shared_with_lingbot": True, "minimum_free_gib": 32,
                "start_rule": "All dependency queues and all their tasks must complete successfully.",
                "training": "P only; 100% train predicted histories; geometry=1, fk=0",
                "selection": "Fixed dev next-frame world SMPL22 FK, equal take/mode/variant macro mean"}
        plan_path = args.output / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing deferred experiment plan differs from the requested arguments.")
        atomic_json(plan_path, plan)
        previous = None
        try:
            while True:
                ready, statuses = dependency_status(dependencies)
                status.update(dependencies=statuses, updated_at=now())
                status["status"] = "waiting_for_existing_experiments"
                if ready:
                    status["status"] = "waiting_for_history_caches"
                    if all((args.histories / group / "report.json").exists() for group in ("train", "dev")):
                        status["status"] = "waiting_for_gpu_memory"
                        env = os.environ.copy()
                        nvml = Path("/tmp/uniegomotion-nvml-host")
                        if (nvml / "libnvidia-ml.so.1").exists():
                            env["LD_LIBRARY_PATH"] = str(nvml) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                        available = shared_gpu_indices(args.gpus, env, 32)
                        if available:
                            gpu = min(available)
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                            status.update(gpu=gpu, status="preparing_training", started_at=now())
                            atomic_json(args.output / "status.json", status)
                            break
                atomic_json(args.output / "status.json", status)
                message = (status["status"], json.dumps(statuses, sort_keys=True))
                if message != previous:
                    print(now(), *message, flush=True)
                    previous = message
                time.sleep(30)
            train(args, status)
            status.update(status="complete", completed_at=now())
            atomic_json(args.output / "status.json", status)
        except Exception as error:
            status.update(status="failed", error=repr(error), updated_at=now())
            atomic_json(args.output / "status.json", status)
            append_log("固定预测历史 P 适配停止", [f"目录 `{args.output}`；原因 `{error}`。未完成结果不计入结论。"])
            raise


if __name__ == "__main__":
    main()
