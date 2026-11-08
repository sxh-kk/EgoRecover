"""Check whether a single ground-truth current-state intervention helps."""

import argparse
import json
from pathlib import Path

import torch

from .codec import BodyState
from .data import OFFICIAL_DATA_ROOT, keyed_noise
from .evaluate import load_evaluation_context
from .fk import joint_decoder
from .io import code_identity, file_sha256, save_json
from .selection import select_dev_records
from .state import state_map


def intervention_state(predicted, truth, joints, mode):
    """Separate pose-only correction from the original full-state oracle."""
    if mode == "full_state":
        return state_map(truth, lambda x: x.detach())
    if mode != "body_pose":
        raise ValueError("Unknown intervention mode.")
    # Never read GT hands/contact/beta or replace the predicted reference.
    pose = BodyState(truth.joints.detach(), predicted.reference.detach(), predicted.auxiliary.detach())
    transforms = pose.joints.clone()
    transforms[..., :3, 3] = joints(pose)
    return BodyState(transforms, pose.reference, pose.auxiliary)


def _mpjpe(first, second):
    return float((first - second).norm(dim=-1).mean() * 1000)


def _center_joints(system, result, joints):
    center = system.decode(result.center, result.reference)
    return joints(state_map(center, lambda x: x[:, -1]))


@torch.no_grad()
def diagnose_episode(
    system,
    model,
    episode,
    joints,
    *,
    anchors=(20, 40, 60),
    horizons=(1, 2, 4, 10),
    future_seeds=(1062, 1063, 1064),
    prefix_seed=1062,
    interventions=("body_pose", "full_state"),
    steps=1,
):
    """Keep one own-history prefix; fork paired continuations only at anchors."""
    if not anchors or any(isinstance(a, bool) or not isinstance(a, int) or a < 20 for a in anchors):
        raise ValueError("Anchors must be integers at or after frame 20.")
    if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integers.")
    if not future_seeds or len(set(future_seeds)) != len(future_seeds):
        raise ValueError("Provide distinct future noise seeds.")
    if not interventions or len(set(interventions)) != len(interventions):
        raise ValueError("Provide distinct intervention modes.")
    if set(interventions) - {"body_pose", "full_state"}:
        raise ValueError("Unknown intervention mode.")
    if not system.config.body_feedback:
        raise ValueError("Reference diagnostics require an active body-history path.")
    selected = sorted({a for a in anchors if a + min(horizons) < len(episode)})
    if not selected:
        return []
    model.eval()
    cache, rows = episode.cache(), []
    for frame in range(20, selected[-1] + 1):
        epsilon = keyed_noise(episode.identifier, prefix_seed, frame, system.config.window, episode.beta.device)
        current = system.online(model, cache, episode.observations(frame), epsilon, steps=steps)
        # Branches below never overwrite this prefix cache, including across seeds.
        cache = current.cache.detach()
        if frame not in selected:
            continue
        truth_current = state_map(episode.target(frame), lambda x: x[:, -1])
        replacements = {m: intervention_state(current.current, truth_current, joints, m) for m in interventions}
        available = sorted({h for h in horizons if frame + h < len(episode)})
        anchor_error = _mpjpe(joints(current.current), truth_current.joints[..., :3, 3])
        for draw in future_seeds:
            predicted_cache = cache
            reference_caches = {m: cache.replace_current(value).detach() for m, value in replacements.items()}
            for offset in range(1, max(available) + 1):
                t = frame + offset
                obs = episode.observations(t)
                noise = keyed_noise(episode.identifier, draw, t, system.config.window, episode.beta.device)
                predicted = system.online(model, predicted_cache, obs, noise, steps=steps)
                references = {
                    m: system.online(model, branch, obs, noise, steps=steps) for m, branch in reference_caches.items()
                }
                predicted_cache = predicted.cache.detach()
                reference_caches = {m: result.cache.detach() for m, result in references.items()}
                if offset not in available:
                    continue
                truth = state_map(episode.target(t), lambda x: x[:, -1])
                points, same_shape = truth.joints[..., :3, 3], joints(truth)
                pj = joints(predicted.current)
                ep, sp = _mpjpe(pj, points), _mpjpe(pj, same_shape)
                cp = _mpjpe(_center_joints(system, predicted, joints), points)
                for mode, reference in references.items():
                    rj = joints(reference.current)
                    er, sr = _mpjpe(rj, points), _mpjpe(rj, same_shape)
                    cr = _mpjpe(_center_joints(system, reference, joints), points)
                    rows.append(
                        {
                            "episode": episode.identifier,
                            "take": episode.take,
                            "anchor": frame,
                            "draw": draw,
                            "horizon": offset,
                            "intervention": mode,
                            "seconds": float(episode.timestamps[t] - episode.timestamps[frame]),
                            "predicted_mm": ep,
                            "reference_mm": er,
                            "delta_mm": ep - er,
                            "response_mm": _mpjpe(pj, rj),
                            "same_shape_predicted_mm": sp,
                            "same_shape_reference_mm": sr,
                            "same_shape_delta_mm": sp - sr,
                            "center_predicted_mm": cp,
                            "center_reference_mm": cr,
                            "center_delta_mm": cp - cr,
                            "anchor_current_mm": anchor_error,
                            "intervened_current_mm": _mpjpe(
                                joints(replacements[mode]), truth_current.joints[..., :3, 3]
                            ),
                        }
                    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=OFFICIAL_DATA_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=1062)
    parser.add_argument("--prefix-seed", type=int, default=1062)
    parser.add_argument("--future-seeds", type=int, nargs="+", default=[1062, 1063, 1064])
    parser.add_argument("--anchors", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument(
        "--interventions", nargs="+", choices=("body_pose", "full_state"), default=["body_pose", "full_state"]
    )
    parser.add_argument("--steps", type=int, choices=(1, 2, 4, 10), default=1)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    torch.set_num_threads(args.threads)
    args.synthetic, args.split = False, "dev"
    args.data_format, args.prior_mode, args.prior_checkpoint = "official", "constant_velocity", None
    context = load_evaluation_context(args)
    indices, selection = select_dev_records(context.data.records, args.limit, args.selection_seed)
    indices = indices[args.shard_index :: args.shard_count]
    task_path = args.data_root / "takes.json"
    tasks = {r["take_name"]: r for r in json.loads(task_path.read_text())}
    metadata = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": context.saved["step"],
        "protocol": context.system.config.to_dict(),
        "manifest": context.data.digest,
        "smpl_asset": context.saved["identity"]["smpl_asset"],
        "task_metadata_sha256": file_sha256(task_path),
        "selection": selection,
        "shard": {"index": args.shard_index, "count": args.shard_count, "episodes": len(indices)},
        "prefix_seed": args.prefix_seed,
        "future_seeds": args.future_seeds,
        "anchors": args.anchors,
        "horizons": args.horizons,
        "steps": args.steps,
        "interventions": args.interventions,
        "intervention_scope": {
            "body_pose": "GT rotations/root, fixed-startup-shape FK positions; predicted reference and auxiliary retained",
            "full_state": "original paired-training reference: all GT current joints/reference/auxiliary, including hands/contact/beta",
        },
        "history": "own continuous prefix; one intervention only; no future GT enters either continuation",
        "coupling": "same prefix cache per anchor; paired branches share future noise and observations",
        "delta_mm": "E_pred - E_ref; positive means reference improves accuracy",
        "tolerance_mm": args.tolerance_mm,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.selection_seed,
        "code": code_identity(),
        "scope": "explicit-checkpoint development diagnostic, not final benchmark or checkpoint selection",
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != metadata:
        raise ValueError("Diagnostic identity changed; choose a new output directory.")
    save_json(identity_path, metadata)
    from .reference_statistics import summarize_reference_rows

    all_rows, lengths = [], {}
    for number, index in enumerate(indices, 1):
        path = args.output / "episodes" / f"{index}.json"
        if path.exists():
            packet = json.loads(path.read_text())
        else:
            episode = context.data.load(index)
            rows = diagnose_episode(
                context.system,
                context.model,
                episode,
                joint_decoder(episode, False, context.smpl),
                anchors=tuple(args.anchors),
                horizons=tuple(args.horizons),
                future_seeds=tuple(args.future_seeds),
                prefix_seed=args.prefix_seed,
                interventions=tuple(args.interventions),
                steps=args.steps,
            )
            task = tasks[episode.take]
            for row in rows:
                row.update({k: task[k] for k in ("task_id", "task_name", "parent_task_id", "parent_task_name")})
            packet = {"episode": episode.identifier, "take": episode.take, "frames": len(episode), "rows": rows}
            save_json(path, packet)
        lengths[packet["episode"]] = packet["frames"]
        all_rows.extend(packet["rows"])
        save_json(args.output / "progress.json", {"completed": number, "total": len(indices), "rows": len(all_rows)})
        if number % 16 == 0 or number == len(indices):
            boot = args.bootstrap_samples if number == len(indices) else 0
            summary = summarize_reference_rows(
                all_rows, tolerance_mm=args.tolerance_mm, bootstrap_samples=boot, bootstrap_seed=args.selection_seed
            )
            complete = {e for e, length in lengths.items() if max(args.anchors) + max(args.horizons) < length}
            complete_summary = summarize_reference_rows(
                [r for r in all_rows if r["episode"] in complete],
                tolerance_mm=args.tolerance_mm,
                bootstrap_samples=boot,
                bootstrap_seed=args.selection_seed,
            )
            save_json(
                args.output / "report.json",
                {"identity": metadata, "completed": number, "summary": summary, "complete_case": complete_summary},
            )
            print(json.dumps({"completed": number, "total": len(indices), "rows": len(all_rows)}), flush=True)


if __name__ == "__main__":
    main()
