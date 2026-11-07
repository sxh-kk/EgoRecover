"""Matched-cache, independent rollout, and single-injection paired diagnostics."""

import math
import time

import torch

from .data import keyed_noise
from .losses import geometry_loss
from .metrics import comparison_metrics
from .state import state_map


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_episode(system, model, episode, joints, *, seed=1062, steps=None, matched_cache=False, comparison=False):
    model.eval()
    cache = episode.cache()
    predictions, errors, same_shape_errors, center_errors, timings = [], [], [], [], []
    rotations = []
    for frame in range(20, len(episode)):
        if matched_cache:
            cache = episode.cache(frame)
        obs = episode.observations(frame)
        noise = keyed_noise(episode.identifier, seed, frame, system.config.window, episode.beta.device)
        _synchronize(episode.beta.device)
        start = time.perf_counter()
        result = system.online(model, cache, obs, noise, steps=steps)
        pose = joints(result.current)
        _synchronize(episode.beta.device)
        timings.append(1000 * (time.perf_counter() - start))
        truth = state_map(episode.target(frame), lambda x: x[:, -1])
        errors.append((pose - truth.joints[..., :3, 3]).norm(dim=-1).mean().item() * 1000)
        same_shape_errors.append((pose - joints(truth)).norm(dim=-1).mean().item() * 1000)
        if system.config.body_feedback:
            center = system.decode(result.center, result.reference)
            center_errors.append(
                (joints(state_map(center, lambda x: x[:, -1])) - truth.joints[..., :3, 3]).norm(dim=-1).mean().item()
                * 1000
            )
        predictions.append(pose.cpu())
        if comparison:
            rotations.append(result.current.joints[0, :, :3, :3].cpu())
        cache = result.cache.detach()
    # Initial call is reported separately; remaining calls form steady latency.
    steady = timings[1:] or timings
    ordered = sorted(steady)
    interval_ms = 1000 * float(episode.timestamps[1] - episode.timestamps[0])
    latency_ms = sum(steady) / len(steady)
    row = {
        "episode": episode.identifier,
        "take": episode.take,
        "frames": len(errors),
        "body_mm": sum(errors) / len(errors),
        "frame_body_mm": errors,
        "same_shape_mm": sum(same_shape_errors) / len(same_shape_errors),
        "center_mm": sum(center_errors) / len(center_errors) if center_errors else None,
        "latency_ms": latency_ms,
        "first_frame_ms": timings[0],
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "frame_budget_exceed_ratio": sum(t > interval_ms for t in steady) / len(steady),
        "reconstruction_fps": 1000 / latency_ms,
        "observation_wait_ms": 0.0,
        "steady_frame_latencies_ms": steady,
        "frame_interval_ms": interval_ms,
        "steps": steps or system.config.steps,
        "matched_cache": matched_cache,
        "metric": "dense_diagnostic" if getattr(joints, "diagnostic_only", False) else "fixed_shape_smpl22",
        "synthetic": episode.synthetic,
        "floor_calibrated": episode.floor_calibrated,
        "predictions": torch.stack(predictions, 1),
    }
    if comparison:
        emitted_rotations = torch.stack(rotations)
        row["comparison"] = comparison_metrics(
            torch.cat(predictions),
            episode.truth.joints[20:, :, :3, 3],
            episode.timestamps[20:],
            predicted_rotations=emitted_rotations,
            truth_rotations=episode.truth.joints[20:, :, :3, :3],
        )
        if not episode.floor_calibrated:
            row["comparison"]["foot_slide_uniegomotion_mm"] = None
            row["comparison"]["gt_foot_slide_uniegomotion_mm"] = None
        row["body_rotations"] = emitted_rotations
    return row


@torch.no_grad()
def paired_horizons(
    system, model, episode, joints, *, frame=20, seed=1062, horizons=(1, 2, 4, 10), steps=None, current=None
):
    """Inject GT only once at t; both branches then advance their OWN caches."""
    if not horizons or min(horizons) < 1 or frame + max(horizons) >= len(episode):
        raise ValueError("Insufficient subsequent frames for requested horizons.")
    model.eval()
    if frame < 20:
        raise ValueError("The paired anchor must follow the twenty-frame startup.")
    if current is None:
        cache = episode.cache()
        for t in range(20, frame + 1):
            noise = keyed_noise(episode.identifier, seed, t, system.config.window, episode.beta.device)
            current = system.online(model, cache, episode.observations(t), noise, steps=steps)
            cache = current.cache.detach()
    if current.cache.frame != frame:
        raise ValueError("Paired diagnostic cache does not match the requested anchor.")
    predicted_cache = current.cache.detach()
    truth = state_map(episode.target(frame), lambda x: x[:, -1])
    reference_cache = predicted_cache.replace_current(truth).detach()
    result = {}
    for offset in range(1, max(horizons) + 1):
        t = frame + offset
        obs = episode.observations(t)
        noise = keyed_noise(episode.identifier, seed, t, system.config.window, episode.beta.device)
        predicted = system.online(model, predicted_cache, obs, noise, steps=steps)
        reference = system.online(model, reference_cache, obs, noise, steps=steps)
        # No future labels enter either branch's state, P, source or observations.
        if offset in horizons:
            pj, rj = joints(predicted.current), joints(reference.current)
            tj = state_map(episode.target(t), lambda x: x[:, -1]).joints[..., :3, 3]
            result[str(offset)] = {
                "seconds": float(episode.timestamps[t] - episode.timestamps[frame]),
                "predicted_mm": float((pj - tj).norm(dim=-1).mean() * 1000),
                "reference_mm": float((rj - tj).norm(dim=-1).mean() * 1000),
                "response_difference": float(geometry_loss(pj, rj, system.config.scale_m)),
            }
        predicted_cache, reference_cache = predicted.cache.detach(), reference.cache.detach()
    return result


@torch.no_grad()
def paired_anchor_diagnostics(
    system, model, episode, joints, *, anchors=(20, 60, 100, 140, 180), seed=1062, horizons=(1, 2, 4, 10), steps=None
):
    """Copy actual G rollout caches at multiple anchors; inject a label once."""
    if not anchors or any(isinstance(f, bool) or not isinstance(f, int) or f < 20 for f in anchors):
        raise ValueError("Diagnostic anchors must be integer frames >=20.")
    selected = sorted({f for f in anchors if f + max(horizons) < len(episode)})
    if not selected:
        raise ValueError("No diagnostic anchor has enough subsequent observations.")
    model.eval()
    cache = episode.cache()
    result = {}
    for frame in range(20, selected[-1] + 1):
        noise = keyed_noise(episode.identifier, seed, frame, system.config.window, episode.beta.device)
        current = system.online(model, cache, episode.observations(frame), noise, steps=steps)
        cache = current.cache.detach()
        if frame in selected:
            result[str(frame)] = paired_horizons(
                system, model, episode, joints, frame=frame, seed=seed, horizons=horizons, steps=steps, current=current
            )
    return result


def summarize(rows):
    """Take/episode weighted quality; pooled steady-frame timing statistics."""
    names = sorted({r["take"] for r in rows})
    if not names:
        raise ValueError("Cannot summarize an empty evaluation.")
    result = {}

    def aggregate(values):
        available = [(r["take"], value) for r, value in values if value is not None]
        per_take = {}
        for take in sorted({take for take, _ in available}):
            selected = [value for name, value in available if name == take]
            per_take[take] = sum(selected) / len(selected)
        if not per_take:
            return None
        return {
            "mean": sum(per_take.values()) / len(per_take),
            "episode_mean": sum(value for _, value in available) / len(available),
            "per_take": per_take,
            "available_episodes": len(available),
            "available_takes": len(per_take),
        }

    for metric in (
        "body_mm",
        "same_shape_mm",
        "center_mm",
        "first_frame_ms",
        "observation_wait_ms",
    ):
        value = aggregate([(r, r.get(metric)) for r in rows])
        if value is not None:
            result[metric] = value
    comparison_names = sorted({key for r in rows for key in r.get("comparison", {})})
    if comparison_names:
        result["comparison"] = {}
        for metric in comparison_names:
            value = aggregate([(r, r.get("comparison", {}).get(metric)) for r in rows])
            if value is not None:
                result["comparison"][metric] = value
    timed = [r for r in rows if "steady_frame_latencies_ms" in r]
    if timed:

        def timing(selected):
            samples = [(t, r["frame_interval_ms"]) for r in selected for t in r["steady_frame_latencies_ms"]]
            ordered = sorted(t for t, _ in samples)
            mean = sum(ordered) / len(ordered)
            return {
                "latency_ms": mean,
                "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
                "reconstruction_fps": 1000 / mean,
                "frame_budget_exceed_ratio": sum(t > interval for t, interval in samples) / len(samples),
            }

        per_take = {
            take: timing([r for r in timed if r["take"] == take]) for take in sorted({r["take"] for r in timed})
        }
        for key, value in timing(timed).items():
            result[key] = {
                "mean": value,
                "episode_mean": value,
                "per_take": {take: metrics[key] for take, metrics in per_take.items()},
                "aggregation": "pooled_steady_frames",
            }
    return result
