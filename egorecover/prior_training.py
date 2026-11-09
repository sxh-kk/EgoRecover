"""Train a history-only P, with matched-CV evaluation and resumable DDP."""

import argparse
import copy
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bootstrap import HeadAlignedBootstrap
from .codec import MotionCodec
from .distributed import DistributedContext
from .fk import FixedShapeFK, load_smpl, smpl_identity
from .io import code_identity, file_sha256, save_checkpoint, save_json, state_sha256
from .losses import geometry_loss, update_ema
from .models import HistoryPrior
from .official import OfficialWindowData
from .prior_data import clean_batch, concatenate_examples, predictor_forward, select_examples
from .prior_two_forward import replacement_count, two_forward_example

MODEL_CONFIG = {"width": 256, "layers": 4, "heads": 8, "dropout": 0.1, "source_flags": True}


@dataclass
class PriorTrainOptions:
    output: Path
    data_root: Path
    train_bank: Path
    val_bank: Path
    device: str = "cuda"
    seed: int = 62
    train_steps: int = 12000
    clean_steps: int = 8000
    stop_after: int | None = None
    resume: bool = False
    batch_size: int = 64
    microbatch_size: int = 8
    learning_rate: float = 3e-5
    eval_every: int = 500
    initial_predicted_fraction: float = 0.0
    adapted_predicted_fraction: float = 0.5
    mature_predicted_fraction: float | None = None
    evaluation_min_frame: int = 21
    checkpoint_steps: tuple[int, ...] = ()
    two_forward_steps: int = 0
    two_forward_fraction: float = 0.8
    adapted_self_fraction: float = 0.0
    ema_decay: float = 0.999


def prior_losses(model, example, codec, fk):
    prediction = predictor_forward(model, example)
    error = (prediction - example["target"]).square()
    weights = error.new_ones(243)
    weights[198:207] = 8
    code = (error * weights).sum(-1).mean() / weights.sum()
    body = codec.decode_current(prediction[:, 0], example["reference"])
    geometry = geometry_loss(fk(body), example["truth_xyz"], 0.1)
    return {"code": code, "geometry": geometry, "hand_pca": error[..., 207:231].mean()}


def load_bank(path, split, manifest):
    bank = torch.load(path, map_location="cpu", weights_only=True)
    if bank.get("format") != "egorecover_prior_bank_v1" or bank["split"] != split:
        raise ValueError("P bank split or format mismatch.")
    if bank["manifest_sha256"] != manifest:
        raise ValueError("P bank dataset identity mismatch.")
    if any(len(value) != len(bank["rows"]) for value in bank["tensors"].values()):
        raise ValueError("P bank tensors and rows disagree.")
    if any(not bool(torch.isfinite(value).all()) for value in bank["tensors"].values()):
        raise ValueError("Nonfinite P bank.")
    return bank


def summarize_prior_rows(rows, minimum_frame=21):
    def summary(selected):
        if not selected:
            return None
        takes = sorted({row["take"] for row in selected})
        values = {}
        for take in takes:
            per_take = [row for row in selected if row["take"] == take]
            values[take] = {
                key: sum(row[key] for row in per_take) / len(per_take)
                for key in ("p_body_mm", "cv_body_mm", "delta_mm")
            }
        return {
            **{key: sum(value[key] for value in values.values()) / len(values) for key in next(iter(values.values()))},
            "samples": len(selected),
            "takes": len(takes),
            "per_take": values,
        }

    groups = {"clean": summary([row for row in rows if row["kind"] == "clean"])}
    predicted = [row for row in rows if row["kind"] == "predicted"]
    for window in (1, 3):
        selected = [row for row in predicted if row["window"] == window]
        groups[f"predicted_W{window}"] = summary(selected)
        groups[f"steady_W{window}"] = summary([row for row in selected if row["frame"] > 20])
        groups[f"mature_W{window}"] = summary([row for row in selected if row["frame"] >= 40])
        groups[f"selection_W{window}"] = summary([row for row in selected if row["frame"] >= minimum_frame])
    groups["predicted_all"] = summary(predicted)
    groups["predicted_steady"] = summary([row for row in predicted if row["frame"] > 20])
    groups["startup"] = summary([row for row in predicted if row["frame"] == 20])
    groups["selection_all"] = summary([row for row in predicted if row["frame"] >= minimum_frame])
    steady = groups["selection_all"]
    if steady is None or any(groups[f"selection_W{window}"] is None for window in (1, 3)):
        raise ValueError("P evaluation requires selected predicted histories for both W1 and W3.")
    deltas = np.asarray([value["delta_mm"] for value in steady["per_take"].values()])
    rng = np.random.default_rng(1062)
    boot = deltas[rng.integers(0, len(deltas), size=(2000, len(deltas)))].mean(1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    passed = all(groups[f"selection_W{window}"]["delta_mm"] < 0 for window in (1, 3)) and ci[1] < 0
    return {
        "groups": groups,
        "steady_delta_ci95_mm": ci,
        "gate_passed": bool(passed),
        "evaluation_min_frame": minimum_frame,
        "gate_rule": f"P beats CV on both W1/W3 matched histories at frame>={minimum_frame}; pooled take-bootstrap upper95CI below0",
        "uncertainty_scope": "fixed validation samples/checkpoint, not training-seed uncertainty",
    }


@torch.no_grad()
def evaluate_prior(model, bank, codec, fk, context, *, minimum_frame=21):
    model.eval()
    indices = list(range(context.rank, len(bank["rows"]), context.world_size))
    rows = []
    for start in range(0, len(indices), 32):
        selected = indices[start : start + 32]
        example = select_examples(bank["tensors"], selected, context.device)
        prediction = predictor_forward(model, example)
        state = codec.decode_current(prediction[:, 0], example["reference"])
        errors = (fk(state) - example["truth_xyz"]).norm(dim=-1).mean(-1).cpu() * 1000
        cv_state = codec.decode_current(example["base"][:, 0], example["reference"])
        cv_errors = (fk(cv_state) - example["truth_xyz"]).norm(dim=-1).mean(-1).cpu() * 1000
        for index, error, cv_error in zip(selected, errors.tolist(), cv_errors.tolist()):
            original = bank["rows"][index]
            rows.append(
                {
                    **original,
                    "producer_cv_body_mm": original["cv_body_mm"],
                    "cv_body_mm": cv_error,
                    "p_body_mm": error,
                    "delta_mm": error - cv_error,
                }
            )
    rows = context.gather_rows(rows)
    return {"summary": summarize_prior_rows(rows, minimum_frame), "rows": rows}


def predicted_sampling_groups(rows, minimum_frame=40):
    """Balance producer W independently of the startup/mature-history mixture."""
    groups = {
        (window, mature): torch.tensor(
            [
                index
                for index, row in enumerate(rows)
                if row["kind"] == "predicted" and row["window"] == window and (row["frame"] >= minimum_frame) == mature
            ],
            dtype=torch.long,
        )
        for window in (1, 3)
        for mature in (False, True)
    }
    if any(not len(indices) for indices in groups.values()):
        raise ValueError("Balanced P sampling requires early and mature histories from both W1 and W3.")
    return groups


def sample_predicted_indices(groups, generator, count, mature_fraction):
    windows = 1 + 2 * torch.randint(2, (count,), generator=generator)
    mature = torch.rand(count, generator=generator) < mature_fraction
    indices = torch.empty(count, dtype=torch.long)
    for (window, is_mature), candidates in groups.items():
        selected = (windows == window) & (mature == is_mature)
        indices[selected] = candidates[torch.randint(len(candidates), (int(selected.sum()),), generator=generator)]
    return indices


def sample_prior_batch(data, codec, bank, generator, count, fraction, device, *, groups=None, mature_fraction=None):
    predicted_count = int((torch.rand(count, generator=generator) < fraction).sum())
    examples = []
    if predicted_count < count:
        batch, frame = data.sample_batch(generator, count - predicted_count, 0)
        examples.append(clean_batch(codec, batch, frame))
    if predicted_count:
        indices = (
            torch.randint(len(bank["rows"]), (predicted_count,), generator=generator)
            if groups is None
            else sample_predicted_indices(groups, generator, predicted_count, mature_fraction)
        )
        examples.append(select_examples(bank["tensors"], indices, device))
    return concatenate_examples(examples), predicted_count


def prior_phase(args, step):
    if step <= args.clean_steps:
        name = "clean_pretraining" if args.initial_predicted_fraction == 0 else "mixed_history_pretraining"
        return name, args.initial_predicted_fraction, 0.0, 0
    if step <= args.clean_steps + args.two_forward_steps:
        replaced = replacement_count(step, args.clean_steps, args.two_forward_steps)
        return "two_forward_pretraining", 0.0, args.two_forward_fraction, replaced
    return "predicted_history_adaptation", args.adapted_predicted_fraction, args.adapted_self_fraction, 20


def sample_training_batch(
    model, data, codec, bank, generator, count, fraction, device, *, self_fraction, replaced, **kw
):
    if self_fraction == 0:
        example, predicted = sample_prior_batch(data, codec, bank, generator, count, fraction, device, **kw)
        return example, predicted, 0
    self_count = int((torch.rand(count, generator=generator) < self_fraction).sum())
    examples, predicted = [], 0
    if self_count < count:
        example, predicted = sample_prior_batch(
            data, codec, bank, generator, count - self_count, fraction / (1 - self_fraction), device, **kw
        )
        examples.append(example)
    if self_count:
        batch, frame = data.sample_batch(generator, self_count, replaced)
        examples.append(two_forward_example(model, codec, batch, frame, replaced))
    return concatenate_examples(examples), predicted, self_count


def calibrate_geometry(model, data, codec, fk, device):
    """Keep P's FK gradient below10% of the code-gradient norm initially."""
    model.eval()
    generator = torch.Generator().manual_seed(9062)
    batch, frame = data.sample_batch(generator, 8, 0)
    example = clean_batch(codec, batch, frame)
    losses = prior_losses(model, example, codec, fk)
    parameters = tuple(model.parameters())
    norms = {}
    for key in ("code", "geometry"):
        gradients = torch.autograd.grad(losses[key], parameters, retain_graph=True, allow_unused=True)
        norms[key] = math.sqrt(sum(float(value.square().sum()) for value in gradients if value is not None))
    weight = min(0.03, 0.1 * norms["code"] / max(norms["geometry"], 1e-12))
    model.zero_grad(set_to_none=True)
    return {"weight": weight, "gradient_norms": norms, "target_weighted_geometry_ratio": 0.1, "calibration_seed": 9062}


def train_prior(args):
    with DistributedContext.from_environment(args.device) as context:
        _train_prior(args, context)


def _train_prior(args, context):
    if (
        args.train_steps < 1
        or args.clean_steps < 0
        or args.eval_every < 1
        or args.batch_size < 1
        or args.microbatch_size < 1
        or (args.stop_after is not None and args.stop_after < 1)
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not 0 <= args.ema_decay < 1
        or args.evaluation_min_frame < 21
        or any(step < 1 or step > args.train_steps for step in args.checkpoint_steps)
        or args.two_forward_steps < 0
        or (args.two_forward_steps and args.clean_steps + args.two_forward_steps > args.train_steps)
    ):
        raise ValueError("Invalid P training budget or optimizer setting.")
    if args.batch_size % context.world_size or (args.batch_size // context.world_size) % args.microbatch_size:
        raise ValueError("P batch and microbatch must partition the DDP ranks.")
    if not 0 <= args.initial_predicted_fraction <= 1 or not 0 <= args.adapted_predicted_fraction <= 1:
        raise ValueError("Invalid predicted-history mixture.")
    if args.mature_predicted_fraction is not None and not 0 <= args.mature_predicted_fraction <= 1:
        raise ValueError("Invalid mature-history mixture.")
    if (
        not 0 <= args.two_forward_fraction <= 1
        or not 0 <= args.adapted_self_fraction <= 1
        or args.adapted_self_fraction + args.adapted_predicted_fraction > 1
    ):
        raise ValueError("Invalid self-predicted/G/GT history mixture.")
    torch.set_num_threads(2)
    device = context.device
    smpl = load_smpl(device)
    bootstrap = HeadAlignedBootstrap.from_smpl(smpl)
    data = OfficialWindowData(args.data_root, "train", device=device, bootstrap=bootstrap, load_images=False)
    codec = MotionCodec(data.stats).to(device)
    fk = FixedShapeFK(smpl, torch.zeros(10, device=device))
    train_bank = load_bank(args.train_bank, "train", data.digest)
    val_bank = load_bank(args.val_bank, "val", data.digest)
    sampling_groups = (
        predicted_sampling_groups(train_bank["rows"]) if args.mature_predicted_fraction is not None else None
    )
    if {row["take"] for row in train_bank["rows"]} & {row["take"] for row in val_bank["rows"]}:
        raise ValueError("P train and validation histories share takes.")
    torch.manual_seed(args.seed)
    model = HistoryPrior(**MODEL_CONFIG).to(device)
    initial_sha = state_sha256(model.state_dict())
    old_identity = json.loads((args.output / "identity.json").read_text()) if args.resume else None
    audit = (
        old_identity["geometry_audit"]
        if old_identity
        else (calibrate_geometry(model, data, codec, fk, device) if context.is_primary else None)
    )
    audit = context.all_gather_object(audit)[0]
    identity = {
        "model_config": MODEL_CONFIG,
        "seed": args.seed,
        "from_scratch": True,
        "random_initial_model_sha256": initial_sha,
        "manifest_sha256": data.digest,
        "smpl_asset_sha256": smpl_identity(),
        "train_bank_sha256": file_sha256(args.train_bank),
        "val_bank_sha256": file_sha256(args.val_bank),
        "train_steps": args.train_steps,
        "clean_steps": args.clean_steps,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "learning_rate": args.learning_rate,
        "eval_every": args.eval_every,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "ema_decay": args.ema_decay,
        "initial_predicted_fraction": args.initial_predicted_fraction,
        "adapted_predicted_fraction": args.adapted_predicted_fraction,
        "mature_predicted_fraction": args.mature_predicted_fraction,
        "evaluation_min_frame": args.evaluation_min_frame,
        "checkpoint_steps": list(args.checkpoint_steps),
        "two_forward_steps": args.two_forward_steps,
        "two_forward_fraction": args.two_forward_fraction,
        "adapted_self_fraction": args.adapted_self_fraction,
        "two_forward": {
            "enabled": bool(args.two_forward_steps or args.adapted_self_fraction),
            "replacement_counts": [1, 3, 5, 10, 20],
            "selection": "contiguous recent suffix; increasing counts during self pretraining;20 during G adaptation",
            "first_pass": "current raw P in eval/no_grad; independent GT-only preceding20-frame windows; no observations",
            "second_pass": "shared trainable P on physically mixed and re-encoded history; original current GT target",
            "sampler": "self examples need20+replacement_count past frames; take-balanced eligible official train sources",
        },
        "history_frames": 20,
        "current_observations_to_P": False,
        "geometry_audit": audit,
        "code": code_identity(),
        "world_size": context.world_size,
        "data_access": "motion-only; image features not loaded; training split only",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "identity.json").exists() and json.loads((args.output / "identity.json").read_text()) != identity:
        raise ValueError("P run identity changed; choose a new output directory.")
    if (args.output / "resume.pt").exists() and not args.resume:
        raise ValueError("P checkpoint exists; use explicit resume.")
    training_model = context.wrap(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed + 1 + context.rank)
    torch.manual_seed(args.seed + 100000 * context.rank)
    step, best, curve = 0, float("inf"), []
    if args.resume:
        saved = torch.load(args.output / "resume.pt", weights_only=True, map_location=device)
        if saved["identity"] != identity:
            raise ValueError("P resume identity changed.")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["rank_rng_states"][context.rank]
        generator.set_state(state["sample_rng"].cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        step, best, curve = saved["step"], saved["best"], saved["curve"]
        del saved
    if context.is_primary:
        save_json(args.output / "identity.json", identity)
        save_json(args.output / "status.json", {"phase": "training", "step": step})
    context.barrier()
    final = min(args.train_steps, step + args.stop_after) if args.stop_after else args.train_steps
    local_batch = args.batch_size // context.world_size
    for step in range(step + 1, final + 1):
        phase, fraction, self_fraction, replaced = prior_phase(args, step)
        training_model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = dict(
            code=0.0, geometry=0.0, hand_pca=0.0, loss=0.0, predicted_fraction=0.0, self_predicted_fraction=0.0
        )
        for offset in range(0, local_batch, args.microbatch_size):
            example, predicted_count, self_count = sample_training_batch(
                model,
                data,
                codec,
                train_bank,
                generator,
                args.microbatch_size,
                fraction,
                device,
                self_fraction=self_fraction,
                replaced=replaced,
                groups=sampling_groups,
                mature_fraction=args.mature_predicted_fraction,
            )
            synchronize = (
                training_model.no_sync()
                if context.world_size > 1 and offset + args.microbatch_size < local_batch
                else nullcontext()
            )
            with synchronize:
                losses = prior_losses(training_model, example, codec, fk)
                loss = losses["code"] + audit["weight"] * losses["geometry"]
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite P training objective.")
                (loss * args.microbatch_size / local_batch).backward()
            for key in losses:
                metrics[key] += float(losses[key].detach()) * args.microbatch_size / local_batch
            metrics["loss"] += float(loss.detach()) * args.microbatch_size / local_batch
            metrics["predicted_fraction"] += predicted_count / local_batch
            metrics["self_predicted_fraction"] += self_count / local_batch
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        metrics = context.sync_scalar_dict(metrics)
        progress = {
            "step": step,
            **metrics,
            "gradient_norm": float(norm),
            "geometry_weight": audit["weight"],
            "phase": phase,
            "self_replacement_frames": replaced if self_fraction else 0,
        }
        if context.is_primary:
            save_json(args.output / "progress.json", progress)
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
        if step % args.eval_every == 0 or step == final or step in args.checkpoint_steps:
            if context.is_primary:
                save_json(args.output / "status.json", {"phase": "validation", "step": step})
            report = evaluate_prior(ema, val_bank, codec, fk, context, minimum_frame=args.evaluation_min_frame)
            score = report["summary"]["groups"]["selection_all"]["p_body_mm"]
            improved = score < best
            best = min(best, score)
            curve.append({"step": step, "summary": report["summary"]})
            rng_state = {
                "sample_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "model_sha256": state_sha256(model.state_dict()),
            }
            rank_states = context.all_gather_object(rng_state)
            if len({row["model_sha256"] for row in rank_states}) != 1:
                raise RuntimeError("P DDP models disagree.")
            if context.is_primary:
                save_json(args.output / f"validation-step{step}.json", report)
                save_json(args.output / "validation.json", {"step": step, **report})
                save_json(
                    args.output / "selection.json",
                    {"best_selection_mm": best, "evaluation_min_frame": args.evaluation_min_frame, "curve": curve},
                )
                save_checkpoint(
                    args.output / "resume.pt",
                    {
                        "format": "egorecover_prior_resume_v1",
                        "identity": identity,
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rank_rng_states": rank_states,
                        "step": step,
                        "best": best,
                        "curve": curve,
                    },
                )
                export = {
                    "format": "egorecover_prior_v1",
                    "manifest_sha256": data.digest,
                    "model_config": MODEL_CONFIG,
                    "state_dict": ema.state_dict(),
                    "step": step,
                    "validation_summary": report["summary"],
                    "identity": identity,
                    "stats": data.stats,
                }
                save_checkpoint(args.output / "last.pt", export)
                if improved:
                    save_checkpoint(args.output / "best.pt", export)
                if step in args.checkpoint_steps:
                    milestone = args.output / "milestones" / f"step{step}"
                    milestone.mkdir(parents=True, exist_ok=False)
                    for name in ("resume.pt", "last.pt", "best.pt", f"validation-step{step}.json", "selection.json"):
                        (milestone / name).hardlink_to(args.output / name)
                save_json(args.output / "status.json", {"phase": "training", "step": step})
            context.barrier()
    if context.is_primary:
        save_json(
            args.output / "status.json",
            {"phase": "complete" if final == args.train_steps else "stopped_after_checkpoint", "step": final},
        )


def evaluate_checkpoint(args):
    with DistributedContext.from_environment(args.device) as context:
        torch.set_num_threads(2)
        saved = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
        codec = MotionCodec(saved["stats"]).to(context.device)
        smpl = load_smpl(context.device)
        fk = FixedShapeFK(smpl, torch.zeros(10, device=context.device))
        bank = load_bank(args.val_bank, "val", saved["manifest_sha256"])
        model = HistoryPrior(**saved["model_config"]).to(context.device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        report = evaluate_prior(model, bank, codec, fk, context, minimum_frame=args.evaluation_min_frame)
        if context.is_primary:
            args.output.mkdir(parents=True, exist_ok=True)
            save_json(
                args.output / "report.json",
                {
                    "checkpoint_step": saved["step"],
                    "checkpoint_sha256": file_sha256(args.evaluate_checkpoint),
                    "bank_sha256": file_sha256(args.val_bank),
                    **report,
                },
            )


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "data-root", "train-bank", "val-bank"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "seed",
        "train-steps",
        "clean-steps",
        "batch-size",
        "microbatch-size",
        "eval-every",
        "evaluation-min-frame",
        "two-forward-steps",
    ):
        parser.add_argument(
            "--" + name, type=int, default=PriorTrainOptions.__dataclass_fields__[name.replace("-", "_")].default
        )
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--mature-predicted-fraction", type=float)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=())
    parser.add_argument("--device", default="cuda")
    for name in (
        "learning-rate",
        "initial-predicted-fraction",
        "adapted-predicted-fraction",
        "ema-decay",
        "two-forward-fraction",
        "adapted-self-fraction",
    ):
        parser.add_argument(
            "--" + name, type=float, default=PriorTrainOptions.__dataclass_fields__[name.replace("-", "_")].default
        )
    args = parser.parse_args()
    if args.evaluate_checkpoint:
        evaluate_checkpoint(args)
    else:
        values = vars(args).copy()
        values.pop("evaluate_checkpoint")
        train_prior(PriorTrainOptions(**values))


if __name__ == "__main__":
    main()
