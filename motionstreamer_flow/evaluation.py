import torch

from .data import online_inputs
from .streaming import rollout


def pa_align(prediction, truth):
    p = prediction.double()
    q = truth.double()
    pm, qm = p.mean(-2, keepdim=True), q.mean(-2, keepdim=True)
    pc, qc = p - pm, q - qm
    u, singular, vh = torch.linalg.svd(pc.transpose(-1, -2) @ qc)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.linalg.det(u @ vh)
    rotation = (u * correction[..., None, :]) @ vh
    scale = (singular * correction).sum(-1) / pc.square().sum((-2, -1)).clamp_min(1e-12)
    return ((pc @ rotation) * scale[..., None, None] + qm).float()


def metrics(prediction, truth, fps, mask=None):
    if len(prediction) < 1 or prediction.shape != truth.shape:
        raise ValueError("No evaluation frames or inconsistent joint shapes.")
    mask = torch.ones(len(truth), dtype=torch.bool) if mask is None else mask.cpu()
    if not mask.any():
        raise ValueError("No valid evaluation frames.")
    error = (prediction - truth).norm(dim=-1) * 1000
    pa = (pa_align(prediction[mask], truth[mask]) - truth[mask]).norm(dim=-1).mean() * 1000
    result = {
        "body_mm": float(error[mask].mean()),
        "root_mm": float(error[mask, 0].mean()),
        "head_mm": float(error[mask, 15].mean()),
        "pa_body_mm": float(pa),
        "frames": int(mask.sum()),
    }
    pairs = mask[1:] & mask[:-1]
    if pairs.any():
        velocity_error = ((prediction[1:] - prediction[:-1]) - (truth[1:] - truth[:-1])).norm(dim=-1)
        result["velocity_mm_s"] = float(velocity_error[pairs].mean() * fps * 1000)
        result["velocity_pairs"] = int(pairs.sum())
    return result


@torch.no_grad()
def evaluate(flow, tae, normalizer, episodes, split="dev", action="a11", seed=62, start=None, stop=None, limit=None):
    records = episodes.split(split)
    if limit is not None:
        records = records[:limit]
    results = []
    for record in records:
        packet = episodes.load(record)
        output = rollout(flow, tae, normalizer, online_inputs(packet), action=action, seed=seed)
        begin = max(flow.cfg.bootstrap, start if start is not None else flow.cfg.bootstrap)
        end = min(len(packet["pose"]), stop if stop is not None else len(packet["pose"]))
        # Predictions are closed-loop from startup, even when evaluation starts later.
        per_episode = metrics(
            output["joints"][begin - flow.cfg.bootstrap : end - flow.cfg.bootstrap],
            packet["joints"][begin:end],
            flow.cfg.fps,
            packet["valid"][begin:end],
        )
        results.append({"id": record["id"], "take": record["take"], **per_episode})
    averaged = {}
    for key in ("body_mm", "root_mm", "head_mm", "pa_body_mm", "velocity_mm_s"):
        weight = "velocity_pairs" if key == "velocity_mm_s" else "frames"
        supported = [r for r in results if key in r]
        if supported:
            averaged[key] = sum(r[key] * r[weight] for r in supported) / sum(r[weight] for r in supported)
    return {
        "split": split,
        "action": action,
        "seed": seed,
        "aggregation": "valid-frame weighted",
        "backend": episodes.manifest.get("backend"),
        "metrics": averaged,
        "episodes": results,
    }
