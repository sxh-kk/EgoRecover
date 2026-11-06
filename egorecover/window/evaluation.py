"""Matched-cache, independent rollout, and single-injection paired diagnostics."""

import time

import torch

from .data import keyed_noise
from .objective import geometry_loss
from .state import state_map


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_episode(system, model, episode, joints, *, seed=1062, steps=None, matched_cache=False):
    model.eval()
    cache = episode.cache()
    predictions, errors, same_shape_errors, center_errors, timings = [], [], [], [], []
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
        cache = result.cache.detach()
    ordered = sorted(timings)
    # Initial call is reported separately; remaining calls form steady latency.
    steady = timings[1:] or timings
    return {
        "episode": episode.identifier,
        "take": episode.take,
        "frames": len(errors),
        "body_mm": sum(errors) / len(errors),
        "frame_body_mm": errors,
        "same_shape_mm": sum(same_shape_errors) / len(same_shape_errors),
        "center_mm": sum(center_errors) / len(center_errors) if center_errors else None,
        "latency_ms": sum(steady) / len(steady),
        "first_frame_ms": timings[0],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "steps": steps or system.config.steps,
        "matched_cache": matched_cache,
        "metric": "dense_diagnostic" if getattr(joints, "diagnostic_only", False) else "fixed_shape_smpl22",
        "synthetic": episode.synthetic,
        "predictions": torch.stack(predictions, 1),
    }


@torch.no_grad()
def paired_horizons(system, model, episode, joints, *, frame=20, seed=1062, horizons=(1, 2, 4, 10), steps=None):
    """Inject GT only once at t; both branches then advance their OWN caches."""
    if not horizons or min(horizons) < 1 or frame + max(horizons) >= len(episode):
        raise ValueError("Insufficient subsequent frames for requested horizons.")
    model.eval()
    noise = keyed_noise(episode.identifier, seed, frame, system.config.window, episode.beta.device)
    current = system.online(model, episode.cache(frame), episode.observations(frame), noise, steps=steps)
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


def summarize(rows):
    """Equal take weighting after averaging each take's episodes."""
    names = sorted({r["take"] for r in rows})
    if not names:
        raise ValueError("Cannot summarize an empty evaluation.")
    result = {}
    for metric in ("body_mm", "same_shape_mm", "latency_ms", "center_mm"):
        per_take = {}
        for take in names:
            values = [r[metric] for r in rows if r["take"] == take and r[metric] is not None]
            if values:
                per_take[take] = sum(values) / len(values)
        if per_take:
            result[metric] = {"mean": sum(per_take.values()) / len(per_take), "per_take": per_take}
    return result
