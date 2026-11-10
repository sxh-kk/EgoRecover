"""Choose an emitted-velocity weight using training-only global-batch gradients."""

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from egorecover.config import TrainOptions
from egorecover.evaluate import load_evaluation_context
from egorecover.fk import joint_decoder
from egorecover.history_training import HistoryTraining
from egorecover.io import save_json, state_sha256
from egorecover.losses import frozen_ema, two_step_objective
from egorecover.sampling import reference_microbatch_plan
from egorecover.system import WindowReconstructor
from scripts.diagnose_geometry_gradients import flat_grad


def gradient_summary(base, velocity):
    first, second = float(base.norm()), float(velocity.norm())
    if not first > 0 or not second > 0:
        raise RuntimeError("Calibration gradients must have positive norms.")
    cosine = float(torch.dot(base, velocity)) / (first * second)
    return {"base_norm": first, "velocity_norm": second, "cosine": cosine, "weight_10pct_norm": 0.1 * first / second}


def calibrate(plan):
    torch.set_num_threads(2)
    root = Path(plan["output"])
    args = SimpleNamespace(
        checkpoint=Path(plan["common_G_checkpoint"]),
        prior_checkpoint=Path(plan["prior_checkpoint"]),
        data_root=Path(plan["data_root"]),
        data_format="official",
        prior_mode="learned",
        split="train",
        synthetic=False,
        device="cuda",
        limit=None,
    )
    context = load_evaluation_context(args)
    config = replace(context.system.config, feedback="none", feedback_weight=0.0)
    system = WindowReconstructor(context.system.codec, context.system.predictor, config)
    model, teacher = context.model, frozen_ema(context.model)
    assert state_sha256(model.state_dict()) == plan["common_G_model_sha256"]
    parameters = list(model.parameters())
    options = TrainOptions(
        output=root,
        batch_size=64,
        microbatch_size=8,
        sample_reference_world_size=2,
        sampling_rollin_frames=20,
        rollin_frames=20,
        history_mode="mixed",
    )
    history = HistoryTraining.from_options(options, 2)
    generators = [torch.Generator().manual_seed(106062 + rank // 2) for rank in range(4)]
    rows, micro_rows = [], []
    for step in range(1, 5):
        base_sum = velocity_sum = None
        for rank in range(4):
            chunks = reference_microbatch_plan(
                context.data,
                generators[rank],
                rank=rank,
                world_size=4,
                reference_world_size=2,
                batch_size=64,
                microbatch_size=8,
                rollin_frames=20,
                endpoint_probability=config.endpoint_probability,
            )
            for chunk in chunks:
                batch = context.data.load_training_queries(chunk["queries"])
                frame, offset = chunk["frame"], chunk["global_offset"]
                cache = batch.cache(frame - history.frames(step, offset))
                model.eval()
                for t in range(cache.frame + 1, frame):
                    epsilon = batch.noise(4062 + step, t, 3, tag="rollin", offset=offset)
                    cache = system.online(model, cache, batch.observations(t), epsilon).cache.detach()
                model.train()
                seed = int.from_bytes(hashlib.sha256(f"calibration/{step}/{offset}".encode()).digest()[:8], "big") % (
                    2**62
                )
                result = two_step_objective(
                    system,
                    model,
                    teacher,
                    cache,
                    batch.observations(frame),
                    batch.target(frame, 3),
                    batch.noise(4062 + step, frame, 3, tag="train", offset=offset),
                    chunk["tau"].to("cuda"),
                    joint_decoder(batch, False, context.smpl),
                    next_observations=batch.observations(frame + 1),
                    next_truth_window=batch.target(frame + 1, 3),
                    next_epsilon=batch.noise(4062 + step, frame + 1, 3, tag="feedback", offset=offset),
                    forward_seed=seed,
                    velocity_weight=1.0,
                    velocity_scale_m_s=1.0,
                    dt=torch.stack([e.timestamps[frame + 1] - e.timestamps[frame] for e in batch.episodes]),
                )
                base = (result["first_loss"] + result["next_loss"]) / 2
                first = flat_grad(
                    torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True), parameters
                )
                second = flat_grad(torch.autograd.grad(result["velocity"], parameters, allow_unused=True), parameters)
                if not all(bool(torch.isfinite(value).all()) for value in (first, second)):
                    raise RuntimeError("Nonfinite calibration gradients.")
                if base_sum is None:
                    base_sum, velocity_sum = first.detach() / 8, second.detach() / 8
                else:
                    base_sum.add_(first.detach(), alpha=1 / 8)
                    velocity_sum.add_(second.detach(), alpha=1 / 8)
                micro_rows.append(
                    {
                        "batch": step,
                        "offset": offset,
                        "velocity": float(result["velocity"].detach()),
                        **gradient_summary(first, second),
                    }
                )
                print(json.dumps({"calibration_microbatches": len(micro_rows), "total": 32}), flush=True)
                del result, base, first, second, batch, cache
        rows.append({"batch": step, **gradient_summary(base_sum, velocity_sum)})
        del base_sum, velocity_sum
    weight = statistics.median(row["weight_10pct_norm"] for row in rows)
    assert state_sha256(model.state_dict()) == plan["common_G_model_sha256"]
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(not p.requires_grad and p.grad is None for p in system.predictor.parameters())
    result = {
        "status": "passed",
        "weight": weight,
        "velocity_scale_m_s": 1.0,
        "target_gradient_fraction": 0.1,
        "method": "median 0.1*||grad(mean(two base objectives))||/||grad(emitted FK velocity loss)||; four global64 batches",
        "data": "training split only; independent calibration sample/noise/dropout; 256 queries",
        "target": "two actual emitted world FK bodies; GT decoded with identical fixed bootstrap shape; measured dt",
        "G_unchanged": True,
        "P_and_teacher_frozen": True,
        "global_batches": rows,
        "microbatches": micro_rows,
        "realized_full_batch_ratios": [weight * row["velocity_norm"] / row["base_norm"] for row in rows],
    }
    save_json(root / "velocity-calibration.json", result)
    print(json.dumps({"calibration": "passed", "velocity_weight": weight, "global_batches": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    calibrate(json.loads(parser.parse_args().plan.read_text()))
