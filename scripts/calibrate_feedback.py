"""Calibrate a fixed paired-loss weight from training-only full-batch gradients."""

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from egorecover.config import TrainOptions, WindowConfig
from egorecover.evaluate import load_evaluation_context
from egorecover.fk import joint_decoder
from egorecover.history_training import HistoryTraining
from egorecover.io import save_json, state_sha256
from egorecover.losses import frozen_ema, training_objective
from egorecover.sampling import reference_microbatch_plan
from egorecover.system import WindowReconstructor
from scripts.diagnose_geometry_gradients import flat_grad, gradient_pair


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
    config = replace(context.system.config, feedback="paired", feedback_weight=1.0)
    system = WindowReconstructor(context.system.codec, context.system.predictor, config)
    model = context.model
    teacher = frozen_ema(model)
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
    # Independent training-only draws; calibration does not consume training RNG.
    generators = [torch.Generator().manual_seed(106062 + rank // 2) for rank in range(4)]
    rows, micro_rows = [], []
    for step in range(1, 5):
        base_sum = pair_sum = None
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
                rollin = history.frames(step, offset)
                cache = batch.cache(frame - rollin)
                model.eval()
                for t in range(frame - rollin, frame):
                    epsilon = batch.noise(4062 + step, t, 3, tag="rollin", offset=offset)
                    cache = system.online(model, cache, batch.observations(t), epsilon).cache.detach()
                model.train()
                seed = int.from_bytes(hashlib.sha256(f"calibration/{step}/{offset}".encode()).digest()[:8], "big") % (
                    2**62
                )
                result = training_objective(
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
                    next_epsilon=batch.noise(4062 + step, frame + 1, 3, tag="feedback", offset=offset)[None],
                    feedback_intervention="body_pose",
                    forward_seed=seed,
                )
                base = result["flow"] + config.current_weight * result["current"]
                first = flat_grad(
                    torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True), parameters
                )
                second = flat_grad(torch.autograd.grad(result["feedback"], parameters, allow_unused=True), parameters)
                if base_sum is None:
                    base_sum, pair_sum = first.detach() / 8, second.detach() / 8
                else:
                    base_sum.add_(first.detach(), alpha=1 / 8)
                    pair_sum.add_(second.detach(), alpha=1 / 8)
                micro_rows.append(
                    {
                        "batch": step,
                        "offset": offset,
                        "rollin": rollin,
                        "flow": float(result["flow"].detach()),
                        "current": float(result["current"].detach()),
                        "paired": float(result["feedback"].detach()),
                        **gradient_pair(first, second),
                    }
                )
                print(json.dumps({"calibration_microbatches": len(micro_rows), "total": 32}), flush=True)
                del result, base, first, second, batch, cache
        row = {"batch": step, **gradient_pair(base_sum, pair_sum)}
        if row["current_norm"] <= 0 or not all(torch.isfinite(v).all() for v in (base_sum, pair_sum)):
            raise RuntimeError("Paired gradient calibration is zero or nonfinite.")
        rows.append(row)
        del base_sum, pair_sum
    weight = statistics.median(row["lambda_10pct_norm"] for row in rows)
    assert state_sha256(model.state_dict()) == plan["common_G_model_sha256"]
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(not p.requires_grad and p.grad is None for p in system.predictor.parameters())
    result = {
        "status": "passed",
        "weight": weight,
        "target_gradient_fraction": 0.1,
        "method": "median of 0.1*||grad(Flow+0.03*current)||/||grad(paired)|| over four full global64 batches",
        "data": "training split only; independent calibration sample/noise/dropout streams; 256 queries",
        "intervention": "body_pose; same reference/auxiliary and revised earlier history",
        "P_and_teacher_frozen": True,
        "G_unchanged": True,
        "global_batches": rows,
        "microbatches": micro_rows,
        "realized_full_batch_ratios": [weight * r["current_norm"] / r["flow_norm"] for r in rows],
    }
    save_json(root / "feedback-calibration.json", result)
    for job in plan["jobs"]:
        protocol = WindowConfig(**plan["base_protocol"]).to_dict()
        protocol.update(
            feedback="paired" if job["unroll_steps"] == 1 else "none",
            feedback_weight=weight if job["unroll_steps"] == 1 else 0.0,
        )
        save_json(job["config"], protocol)
    print(json.dumps({"calibration": "passed", "paired_weight": weight, "global_batches": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    calibrate(json.loads(parser.parse_args().plan.read_text()))
