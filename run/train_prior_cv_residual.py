"""Matched GT-history P training with hold or constant-velocity residual baselines."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from dataset.smpl_utils import get_smpl
from egorecover.codec import MotionCodec
from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from egorecover.losses import physical_objective
from egorecover.prior import HistoryPrior
from egorecover.prior_cv_residual import predict
from egorecover.prior_two_forward import replacement_probability
from egorecover.prior_cv_residual import PriorSequences, evaluate_rollout, evaluate_single, mixed_batch
from run.adapt_prior_on_predictions import save_torch
from run.complete_stages import ROOT, append_log, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-mode", choices=("hold", "constant_velocity"), required=True)
    parser.add_argument("--fk-weight", type=float, choices=(0., 1.), default=0.)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--old-prior", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.two_forward = False
    if min(args.steps, args.eval_every, args.batch_size) < 1:
        parser.error("Positive steps and batch sizes are required.")
    if args.output.exists() and not args.resume:
        parser.error("Existing output requires --resume.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    train, dev = PriorSequences(args.data, "train"), PriorSequences(args.data, "dev")
    device = torch.device(args.device)
    asset = Path(os.environ.get("SMPLX_MODEL_PATH", ROOT / "body_models/smplx")) / "SMPLX_NEUTRAL.npz"
    if file_sha256(asset) != train.identity["smplx_asset_sha256"]:
        raise ValueError("SMPL-X asset differs from the audited P data.")
    identity = {**train.identity, "data_sha256": file_sha256(args.data / "sequences.pt"),
                "base_mode": args.base_mode, "two_forward": args.two_forward, "geometry_weight": 1., "fk_weight": args.fk_weight,
                "steps": args.steps, "batch_size": args.batch_size, "training_seed": args.seed,
                "eval_every": args.eval_every, "lr": 3e-4, "weight_decay": .01,
                "replacement_maximum": .5 if args.two_forward else 0., "selection": "GT_history_next_frame_fk22",
                "baselines": args.baselines, "old_prior_sha256": file_sha256(args.old_prior) if args.old_prior else None,
                "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
                    Path(__file__).resolve(), ROOT / "egorecover/prior_cv_residual.py",
                    ROOT / "egorecover/prior_two_forward.py", ROOT / "eval/metrics.py", ROOT / "egorecover/prior.py",
                    ROOT / "egorecover/codec.py", ROOT / "egorecover/losses.py", ROOT / "egorecover/fk.py")}}
    path = args.output / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Resume protocol differs from the saved experiment.")
    atomic_json(path, identity)
    codec = MotionCodec(train.stats, reference_mode=train.identity["reference_mode"]).to(device)
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    torch.manual_seed(args.seed)
    model = HistoryPrior().to(device)
    model.base_mode = args.base_mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sampling = torch.Generator().manual_seed(args.seed)
    augmentation = torch.Generator().manual_seed(args.seed + 10000)
    indices = train.indices(40)
    curve, start, best_step, elapsed = [], 0, 0, 0.
    training_seconds = 0.
    resume = args.output / "resume.pt"
    if args.resume and resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampling.set_state(saved["sampling_rng"])
        augmentation.set_state(saved["augmentation_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        training_seconds = saved["training_seconds"]
        start, curve, best_step, best, best_state, elapsed = (saved[key] for key in (
            "step", "curve", "best_step", "best", "best_state", "elapsed_seconds"))
    else:
        scores, _ = evaluate_single(model, dev, codec, smpl, device, args.batch_size)
        best = scores["fk22_mm"]["mean"]
        curve.append({"step": 0, "dev": scores})
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_torch(args.output / "initial.pt", {"state_dict": best_state, "base_mode": args.base_mode})
    started = time.monotonic()
    append_log("P 常速度残差对照开始/恢复", [f"`{args.output}`；base={args.base_mode}，"
               f"geometry=1/fk={args.fk_weight:g}，seed={args.seed}，从 step{start} 开始。仅使用 train 身体状态训练。",
               f"运行配置：`{args.output / 'experiment.json'}`；初始权重：`{args.output / 'initial.pt'}`。"])
    for step in range(start + 1, args.steps + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_started = time.monotonic()
        chosen = indices[torch.randint(len(indices), (args.batch_size,), generator=sampling)]
        context, target, beta = train.examples(chosen, device, context_length=40)
        probability = replacement_probability(step, args.steps) if args.two_forward else 0.
        batch, actual = mixed_batch(model, codec, context, target, beta, probability=probability, generator=augmentation)
        optimizer.zero_grad(set_to_none=True)
        parts = physical_objective(codec, predict(model, batch), batch, geometry_weight=1.,
                                   fk_weight=args.fk_weight, smpl=smpl, return_components=True)
        if not torch.isfinite(parts["loss"]):
            raise ValueError(f"Nonfinite loss at step {step}.")
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.monotonic() - update_started
        if step % 20 == 0:
            print(f"P {step}/{args.steps} p={probability:.4f} replaced={actual:.4f}",
                  {key: float(value.detach()) for key, value in parts.items()}, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            scores, _ = evaluate_single(model, dev, codec, smpl, device, args.batch_size)
            score = scores["fk22_mm"]["mean"]
            curve.append({"step": step, "replacement_probability": probability, "actual_replaced_fraction": actual,
                          "losses": {key: float(value.detach()) for key, value in parts.items()}, "dev": scores})
            if score < best:
                best, best_step = score, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            seconds = elapsed + time.monotonic() - started
            save_torch(resume, {"identity": identity, "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "sampling_rng": sampling.get_state(),
                "augmentation_rng": augmentation.get_state(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "best": best, "best_state": best_state, "best_step": best_step, "curve": curve,
                "elapsed_seconds": seconds, "training_seconds": training_seconds})
            atomic_json(args.output / "progress.json", {"step": step, "selected_step": best_step,
                        "best_dev_fk22_mm": best, "elapsed_seconds": seconds, "updated_at": now()})
            atomic_json(args.output / "selection.json", curve)
            print(f"DEV {step}: FK={score:.3f} mm; best={best:.3f} @{best_step}", flush=True)
            append_log("独立 P 选点", [f"`{args.output}`；step{step}/{args.steps}；dev FK={score:.3f} mm；"
                       f"best={best:.3f} mm @step{best_step}；替换概率 {probability:.3f}。",
                       f"当前进度：`{args.output / 'progress.json'}`；完整选点曲线：`{args.output / 'selection.json'}`。"])
    final_single, _ = evaluate_single(model, dev, codec, smpl, device, args.batch_size)
    model.load_state_dict(best_state)
    save_torch(args.output / "prior.pt", {"state_dict": best_state, "kind": "history_prior", **identity,
                                         "selected_step": best_step})
    models = {"prior": model}
    if args.baselines:
        models.update(hold="hold", constant_velocity="constant_velocity")
        if args.old_prior:
            checkpoint = torch.load(args.old_prior, map_location="cpu", weights_only=True)
            for key in ("reference_mode", "stats_sha256", "split_manifest_sha256", "bootstrap_cache_sha256"):
                if checkpoint.get(key) != identity[key]:
                    raise ValueError(f"Old P identity mismatch: {key}.")
            old = HistoryPrior().to(device)
            old.base_mode = checkpoint.get("base_mode", "hold")
            old.load_state_dict(checkpoint["state_dict"])
            models["original_prior"] = old.freeze()
    results = {}
    for name, predictor in models.items():
        single, single_arrays = evaluate_single(predictor, dev, codec, smpl, device, args.batch_size)
        rollout, rollout_arrays = evaluate_rollout(predictor, dev, codec, smpl, device, args.batch_size)
        artifact = args.output / f"dev_{name}.pt"
        save_torch(artifact, {"identity": identity, "single": single_arrays, "rollout": rollout_arrays})
        results[name] = {"single": single, "rollout": rollout, "artifact": str(artifact),
                         "artifact_sha256": file_sha256(artifact)}
    first_pass_updates = sum(replacement_probability(step, args.steps) > 0 for step in range(1, args.steps + 1)) if args.two_forward else 0
    compute = {"supervised_samples": args.steps * args.batch_size,
               "supervised_forward_calls": args.steps,
               "candidate_samples": first_pass_updates * args.batch_size * 20,
               "candidate_forward_calls": first_pass_updates * ((args.batch_size * 20 + 63) // 64),
               "training_wall_seconds": training_seconds,
               "timing_scope": "Synchronized update wall time; includes sampling and transfers, excludes evaluation."}
    report = {**identity, "final_single": final_single, "compute": compute, "results": results, "selected_step": best_step, "curve": curve,
              "prior_sha256": file_sha256(args.output / "prior.pt"),
              "initial_sha256": file_sha256(args.output / "initial.pt"),
              "elapsed_seconds": elapsed + time.monotonic() - started, "completed": True}
    atomic_json(args.output / "report.json", report)
    append_log("P 常速度残差单组完成", [f"`{args.output}`；选中 step{best_step}；dev next-frame FK={best:.3f} mm。",
               f"1秒自反馈 FK={results['prior']['rollout']['1000']['mean']:.3f} mm；"
               f"完整指标/逐take结果：`{args.output / 'report.json'}`；选点曲线：`{args.output / 'selection.json'}`；"
               f"所选权重：`{args.output / 'prior.pt'}`。",
               "dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。"])


if __name__ == "__main__":
    main()
