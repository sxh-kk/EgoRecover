"""Replay one saved global batch and separate Flow / geometry gradients."""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.bootstrap import HeadAlignedBootstrap
from egorecover.codec import MotionCodec
from egorecover.data import open_window_data
from egorecover.fk import joint_decoder, load_smpl, smpl_identity
from egorecover.io import code_identity, file_sha256, save_json, state_sha256
from egorecover.losses import training_objective
from egorecover.models import MotionModel
from egorecover.rotations import rotation_6d_to_matrix
from egorecover.sampling import reference_microbatch_plan


def flat_grad(gradients, parameters):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gradients, parameters)])


def magnitude(value):
    return float(value.norm())


def vector_summary(values):
    x = torch.as_tensor(values).detach().float().reshape(-1).cpu()
    return {
        "min": float(x.min()),
        "p01": float(torch.quantile(x, 0.01)),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def conditioning(codec, codes):
    raw = codec.denormalize(codes.detach())[:, -1]
    body = raw[:, :198].reshape(-1, 22, 9)[..., :6]
    reference = raw[:, 198:204]
    result = {}
    for name, value in (("body", body), ("reference", reference)):
        a, b = value[..., :3], value[..., 3:]
        first = a.norm(dim=-1)
        axis = a / first.clamp_min(1e-12)[..., None]
        residual = (b - (axis * b).sum(-1, keepdim=True) * axis).norm(dim=-1)
        result[f"{name}_first_axis_norm"] = vector_summary(first)
        result[f"{name}_orthogonal_residual"] = vector_summary(residual)
    rotation = rotation_6d_to_matrix(reference)
    result["reference_horizontal_axis"] = vector_summary(rotation[..., :2, 0].norm(dim=-1))
    return result


def code_gradient_groups(gradient):
    g = gradient[:, -1].detach()
    body = g[:, :198].reshape(-1, 22, 9)
    pieces = {
        "body_rotation6d": body[..., :6],
        "body_root_xyz": body[:, 0, 6:9],
        "body_nonroot_xyz": body[:, 1:, 6:9],
        "reference_rotation6d": g[:, 198:204],
        "reference_xyz": g[:, 204:207],
        "hand_pca": g[:, 207:231],
        "contact_beta": g[:, 231:],
    }
    return {name: magnitude(value) for name, value in pieces.items()}


def gradient_pair(first, second):
    nf, nc = magnitude(first), magnitude(second)
    dot = float(torch.dot(first, second))
    cosine = dot / max(1e-30, nf * nc)
    return {
        "flow_norm": nf,
        "current_norm": nc,
        "current_to_flow_ratio": nc / max(nf, 1e-30),
        "cosine": cosine,
        "dot": dot,
        "lambda_equal_norm": nf / max(nc, 1e-30),
        "lambda_10pct_norm": 0.1 * nf / max(nc, 1e-30),
    }


def probe_state(args, saved, data, smpl, state_name):
    config = replace(WindowConfig(**saved["protocol"]), current_weight=1.0, feedback="none")
    codec = MotionCodec(saved["stats"]).to(args.device)
    system = WindowReconstructor(codec, None, config)
    torch.manual_seed(saved["identity"]["seed"])
    model = MotionModel(config, ModelConfig(**saved["model_config"])).to(args.device)
    if state_name in ("raw", "ema"):
        model.load_state_dict(saved["model" if state_name == "raw" else "ema"])
    elif state_name == "baseline":
        baseline = torch.load(args.baseline, map_location="cpu", weights_only=True, mmap=True)
        assert all(torch.equal(baseline["stats"][k], saved["stats"][k]) for k in saved["stats"])
        model.load_state_dict(baseline["ema"])
    elif state_name == "initial":
        assert state_sha256(model.state_dict()) == saved["identity"]["random_initial_model_sha256"]
    else:
        raise ValueError(state_name)
    names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    accumulated_flow, accumulated_current = None, None
    batches = []
    next_step = saved["step"] + 1
    for rank in range(4):
        rng = saved["rank_rng_states"][rank]
        generator = torch.Generator().set_state(rng["sample_rng"])
        torch.cuda.set_rng_state(rng["cuda_rng"], args.device)
        plan = reference_microbatch_plan(
            data,
            generator,
            rank=rank,
            world_size=4,
            reference_world_size=2,
            batch_size=64,
            microbatch_size=8,
            rollin_frames=4,
            endpoint_probability=config.endpoint_probability,
        )
        for micro in plan:
            batch = data.load_training_queries(micro["queries"])
            frame, offset = micro["frame"], micro["global_offset"]
            model.eval()
            cache = batch.cache(frame - 4)
            for t in range(frame - 4, frame):
                noise = batch.noise(
                    saved["identity"]["seed"] + next_step, t, config.window, tag="rollin", offset=offset
                )
                cache = system.online(model, cache, batch.observations(t), noise).cache.detach()
            model.train()
            epsilon = batch.noise(
                saved["identity"]["seed"] + next_step, frame, config.window, tag="train", offset=offset
            )
            decoder = joint_decoder(batch, False, smpl)
            outputs, fk_outputs = [], []
            model_hook = model.register_forward_hook(
                lambda module, inputs, output, captured=outputs: captured.append(output)
            )
            fk_hook = decoder.register_forward_hook(
                lambda module, inputs, output, captured=fk_outputs: captured.append(output)
            )
            result = training_objective(
                system,
                model,
                model,
                cache,
                batch.observations(frame),
                batch.target(frame, config.window),
                epsilon,
                micro["tau"].to(args.device),
                decoder,
            )
            model_hook.remove()
            fk_hook.remove()
            assert len(outputs) == len(fk_outputs) == 2
            flow, current = result["flow"], result["current"]
            point_gradient = torch.autograd.grad(current, fk_outputs[0], retain_graph=True)[0]
            expected = 2 * (fk_outputs[0] - fk_outputs[1].detach()) / (8 * 22 * config.scale_m**2)
            torch.testing.assert_close(point_gradient, expected, rtol=2e-5, atol=1e-5)
            flow_codes = torch.autograd.grad(flow, outputs[0], retain_graph=True)[0]
            current_codes = torch.autograd.grad(current, outputs[1], retain_graph=True)[0]
            flow_gradients = torch.autograd.grad(flow, parameters, retain_graph=False, allow_unused=True)
            current_gradients = torch.autograd.grad(current, parameters, retain_graph=False, allow_unused=True)
            first, second = flat_grad(flow_gradients, parameters), flat_grad(current_gradients, parameters)
            assert bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
            if accumulated_flow is None:
                accumulated_flow, accumulated_current = first / 8, second / 8
            else:
                accumulated_flow.add_(first, alpha=1 / 8)
                accumulated_current.add_(second, alpha=1 / 8)
            dense = result["result"].current.joints[..., :3, 3]
            item = {
                "rank": rank,
                "global_offset": offset,
                "flow": float(flow.detach()),
                "current": float(current.detach()),
                "point_gradient_norm": magnitude(point_gradient),
                "code_gradient_norm": magnitude(current_codes),
                "code_to_point_amplification": magnitude(current_codes) / max(magnitude(point_gradient), 1e-30),
                "parameter_gradients": gradient_pair(first, second),
                "flow_code_groups": code_gradient_groups(flow_codes),
                "current_code_groups": code_gradient_groups(current_codes),
                "conditioning": conditioning(codec, outputs[1]),
                "dense_fk_rms_m": float((dense.detach() - fk_outputs[0].detach()).square().sum(-1).mean().sqrt()),
                "rms_joint_error_m": math.sqrt(float(current.detach())) * config.scale_m,
            }
            batches.append(item)
            print(
                json.dumps(
                    {
                        "state": state_name,
                        "rank": rank,
                        "offset": offset,
                        "flow": item["flow"],
                        "current": item["current"],
                        **item["parameter_gradients"],
                    }
                ),
                flush=True,
            )
            del (
                first,
                second,
                flow_gradients,
                current_gradients,
                result,
                outputs,
                fk_outputs,
                flow,
                current,
                point_gradient,
                expected,
                flow_codes,
                current_codes,
                decoder,
            )
    global_metrics = gradient_pair(accumulated_flow, accumulated_current)
    global_metrics["combined_norm"] = magnitude(accumulated_flow + accumulated_current)
    groups = defaultdict(lambda: {"flow_sq": 0.0, "current_sq": 0.0})
    position = 0
    top = []
    for name, parameter in zip(names, parameters):
        stop = position + parameter.numel()
        a, b = accumulated_flow[position:stop], accumulated_current[position:stop]
        nf, nc = magnitude(a), magnitude(b)
        prefix = name.split(".")[0]
        groups[prefix]["flow_sq"] += nf**2
        groups[prefix]["current_sq"] += nc**2
        top.append({"name": name, "flow_norm": nf, "current_norm": nc})
        position = stop
    weights = []
    for weight in (1.0, 0.1, 0.01, 0.001, 0.0001):
        norm = magnitude(accumulated_flow + weight * accumulated_current)
        clip = min(1.0, 1 / max(norm, 1e-30))
        weights.append(
            {
                "lambda": weight,
                "combined_norm": norm,
                "clip_coefficient": clip,
                "flow_component_norm_after_clip": clip * global_metrics["flow_norm"],
                "current_component_norm_after_clip": clip * weight * global_metrics["current_norm"],
            }
        )
    return {
        "state": state_name,
        "source_step": 0 if state_name == "initial" else 12000 if state_name == "baseline" else saved["step"],
        "flow": sum(r["flow"] for r in batches) / 8,
        "current": sum(r["current"] for r in batches) / 8,
        "global_gradients": global_metrics,
        "parameter_groups": {
            key: {"flow_norm": math.sqrt(v["flow_sq"]), "current_norm": math.sqrt(v["current_sq"])}
            for key, v in groups.items()
        },
        "top_current_parameters": sorted(top, key=lambda r: r["current_norm"], reverse=True)[:12],
        "weight_sensitivity": weights,
        "microbatches": batches,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "baseline", "data-root", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--states", nargs="+", default=["initial", "raw", "ema", "baseline"])
    args = p.parse_args()
    torch.set_num_threads(2)
    args.device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert saved["step"] == 1000 and saved["identity"]["distributed"]["world_size"] == 4
    smpl = load_smpl(args.device)
    assert smpl_identity() == saved["identity"]["smpl_asset"]
    data = open_window_data(
        args.data_root,
        "train",
        device=args.device,
        data_format="official",
        bootstrap=HeadAlignedBootstrap.from_smpl(smpl),
    )
    assert data.digest == saved["identity"]["manifest"]
    report = {
        "identity": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "baseline_sha256": file_sha256(args.baseline),
            "model_window": saved["protocol"]["window"],
            "replay_step": saved["step"] + 1,
            "global_batch": 64,
            "microbatch": 8,
            "scale_m": saved["protocol"]["scale_m"],
            "states": args.states,
            "code": code_identity(),
            "probe_sha256": file_sha256(__file__),
            "cache": "own four-frame roll-in for each state; same saved query/tau/noise/dropout plan",
            "mode": "training/dropout; gradients separated with autograd.grad; no optimizer steps",
        },
        "results": {},
    }
    for state in args.states:
        report["results"][state] = probe_state(args, saved, data, smpl, state)
        save_json(args.output, report)
        print(json.dumps({"state_complete": state, **report["results"][state]["global_gradients"]}), flush=True)


if __name__ == "__main__":
    main()
