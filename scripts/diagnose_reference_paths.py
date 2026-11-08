"""Matched t+1 CV/history condition interventions on frozen checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from egorecover.data import keyed_noise
from egorecover.evaluate import load_evaluation_context
from egorecover.fk import joint_decoder
from egorecover.flow import build_source, solve
from egorecover.io import code_identity, file_sha256, save_json
from egorecover.reference_diagnostic import _center_joints, _mpjpe, intervention_state
from egorecover.selection import select_dev_records
from egorecover.state import state_map
from egorecover.system import WindowResult


def changed_conditions(base, reference, width, cv, history):
    y = {key: value.clone() for key, value in base.items()}
    if cv:
        y["center"][:, -1] = reference["center"][:, -1]
    if history:
        key, slot = ("fixed_motion", -1) if width == 1 else ("center", -2)
        y[key][:, slot] = reference[key][:, slot]
    return y


@torch.no_grad()
def episode_rows(system, model, episode, joints, originals):
    rows, cache = [], episode.cache()
    anchors = [a for a in (20, 40, 60) if a + 1 < len(episode)]
    for frame in range(20, max(anchors) + 1):
        noise = keyed_noise(episode.identifier, 1062, frame, system.config.window, episode.beta.device)
        current = system.online(model, cache, episode.observations(frame), noise)
        cache = current.cache.detach()
        if frame not in anchors:
            continue
        gt = state_map(episode.target(frame), lambda x: x[:, -1])
        replacement = intervention_state(current.current, gt, joints, "body_pose")
        reference_cache = cache.replace_current(replacement).detach()
        obs = episode.observations(frame + 1)
        yp, refp = system.prepare(cache, obs)
        yr, refr = system.prepare(reference_cache, obs)
        assert torch.equal(refp, refr)
        y11 = changed_conditions(yp, yr, system.config.window, True, True)
        assert all(torch.equal(y11[k], yr[k]) for k in yp), "Uncontrolled condition difference"
        truth = state_map(episode.target(frame + 1), lambda x: x[:, -1])
        target, same = truth.joints[..., :3, 3], joints(truth)
        for draw in (1062, 1063, 1064):
            epsilon = keyed_noise(episode.identifier, draw, frame + 1, system.config.window, episode.beta.device)
            source = build_source(yp["center"], epsilon, mode="gaussian", sigma=system.config.sigma)
            outputs = {}
            for name, cv, history in (
                ("baseline", False, False),
                ("cv_only", True, False),
                ("history_only", False, True),
                ("both", True, True),
            ):
                y = changed_conditions(yp, yr, system.config.window, cv, history)
                codes = solve(model, source, y, steps=1)
                body = system.decode(codes, refp)
                active = state_map(body, lambda x: x[:, -1])
                points = joints(active)
                # A post-output cache write cannot alter the output just solved.
                update_true = cache.advance(body, revise_recent=True)
                update_false = cache.advance(body, revise_recent=False)
                for field in ("joints", "reference", "auxiliary"):
                    assert torch.equal(
                        getattr(update_true.body, field)[:, -1], getattr(update_false.body, field)[:, -1]
                    )
                    if system.config.window == 1:
                        assert torch.equal(getattr(update_true.body, field), getattr(update_false.body, field))
                result = WindowResult(codes, body, update_true, refp, y["center"])
                outputs[name] = {
                    "points": points,
                    "error": _mpjpe(points, target),
                    "same": _mpjpe(points, same),
                    "center": _mpjpe(_center_joints(system, result, joints), target),
                }
            original = originals[(frame, draw)]
            assert abs(outputs["baseline"]["error"] - original["predicted_mm"]) < 0.01
            assert abs(outputs["both"]["error"] - original["reference_mm"]) < 0.01
            baseline = outputs["baseline"]
            for path in ("cv_only", "history_only", "both"):
                result = outputs[path]
                rows.append(
                    {
                        "episode": episode.identifier,
                        "take": episode.take,
                        "anchor": frame,
                        "draw": draw,
                        "horizon": 1,
                        "intervention": "body_pose",
                        "path": path,
                        "predicted_mm": baseline["error"],
                        "reference_mm": result["error"],
                        "delta_mm": baseline["error"] - result["error"],
                        "response_mm": _mpjpe(baseline["points"], result["points"]),
                        "same_shape_predicted_mm": baseline["same"],
                        "same_shape_reference_mm": result["same"],
                        "same_shape_delta_mm": baseline["same"] - result["same"],
                        "center_predicted_mm": baseline["center"],
                        "center_reference_mm": result["center"],
                        "center_delta_mm": baseline["center"] - result["center"],
                    }
                )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=4)
    args = p.parse_args()
    args.synthetic, args.split, args.limit = False, "dev", 128
    args.data_format, args.prior_mode, args.prior_checkpoint = "official", "constant_velocity", None
    torch.set_num_threads(2)
    context = load_evaluation_context(args)
    assert context.saved["step"] == 12000
    assert context.system.config.source == "gaussian" and context.system.predictor is None
    indices, selection = select_dev_records(context.data.records, 128, 1062)
    indices = indices[args.shard_index :: args.shard_count]
    identity = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "step": 12000,
        "protocol": context.system.config.to_dict(),
        "selection": selection,
        "prefix_seed": 1062,
        "future_seeds": [1062, 1063, 1064],
        "anchors": [20, 40, 60],
        "horizon": 1,
        "source": "same Gaussian epsilon in all four cells",
        "reference": "body_pose; predicted reference/aux/cache.beta retained",
        "cv_path": "center[:,-1]",
        "history_path": "fixed_motion[:,-1] for W1; center[:,-2] otherwise",
        "code": code_identity(),
        "script_sha256": file_sha256(__file__),
    }
    old = args.output / "identity.json"
    if old.exists():
        assert json.loads(old.read_text()) == identity
    save_json(old, identity)
    width = context.system.config.window
    for number, index in enumerate(indices, 1):
        path = args.output / "episodes" / f"{index}.json"
        if not path.exists():
            original_paths = list((args.original_root / f"W{width}").glob(f"shard*/episodes/{index}.json"))
            assert len(original_paths) == 1
            packet = json.loads(original_paths[0].read_text())
            originals = {
                (r["anchor"], r["draw"]): r
                for r in packet["rows"]
                if r["intervention"] == "body_pose" and r["horizon"] == 1
            }
            episode = context.data.load(index)
            assert episode.identifier == packet["episode"]
            rows = episode_rows(
                context.system, context.model, episode, joint_decoder(episode, False, context.smpl), originals
            )
            save_json(path, {"episode": episode.identifier, "take": episode.take, "rows": rows})
        save_json(args.output / "progress.json", {"completed": number, "total": len(indices)})
        print(json.dumps({"completed": number, "total": len(indices)}), flush=True)


if __name__ == "__main__":
    main()
