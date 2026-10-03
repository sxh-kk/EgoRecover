"""P-only Two-Forward training in physical coordinates; no generator G."""

import json
import math
from pathlib import Path

import torch

from egorecover.codec import BodyState, planar_reference
from egorecover.evaluation_protocol import file_sha256
from egorecover.losses import fk_position_errors
from egorecover.prior_adaptation import predict


FIELDS = ("joints", "reference", "auxiliary")


def state_map(state, fn):
    return BodyState(*(fn(getattr(state, key)) for key in FIELDS))


def replacement_probability(step, total_steps, maximum=0.5):
    if total_steps < 1 or not 0 <= step <= total_steps or not 0 <= maximum <= 1:
        raise ValueError("Invalid replacement schedule.")
    progress = min(1., max(0., (step / total_steps - .2) / .6))
    return maximum * (1 - math.cos(math.pi * progress)) / 2


def physical_batch(codec, history, target, beta):
    last = state_map(history, lambda x: x[:, -1])
    before = state_map(history, lambda x: x[:, -2])
    return {"history_motion": codec.encode_history(history, planar_reference(history.reference[:, 0])),
            "base_mu": codec.continuation_prior(last)[:, None],
            "velocity_mu": codec.constant_velocity_prior(before, last)[:, None],
            "previous_reference": last.reference, "target": codec.encode_current(target, last.reference)[:, None],
            "target_joints": target.joints[..., :3, 3], "beta_boot": beta,
            "beta_boot_is_model": torch.ones(len(beta), dtype=torch.bool, device=beta.device)}


@torch.no_grad()
def candidate_histories(prior, codec, context, chunk_size=64):
    """Predict frames 20..39, each using ONLY its preceding 20 GT frames."""
    if context.reference.shape[1] != 40 or chunk_size < 1:
        raise ValueError("Two-Forward requires exactly 40 past frames and a positive chunk size.")
    size = len(context.reference)
    indices = torch.arange(20, device=context.reference.device)[:, None] + torch.arange(
        20, device=context.reference.device)[None]
    windows = state_map(context, lambda x: x[:, indices].flatten(0, 1))
    predictions = []
    training = prior.training
    prior.eval()
    try:
        for start in range(0, size * 20, chunk_size):
            window = state_map(windows, lambda x: x[start:start + chunk_size])
            last = state_map(window, lambda x: x[:, -1])
            batch = {"history_motion": codec.encode_history(window, planar_reference(window.reference[:, 0])),
                     "base_mu": codec.continuation_prior(last)[:, None]}
            prediction = predict(prior, batch)
            predictions.append(codec.decode_current(prediction[:, 0], last.reference))
    finally:
        prior.train(training)
    return BodyState(*(torch.cat([getattr(state, key) for state in predictions]).reshape(
        size, 20, *getattr(predictions[0], key).shape[1:]).detach() for key in FIELDS))


def mixed_batch(prior, codec, context, target, beta, *, probability, generator, chunk_size=64):
    if not 0 <= probability <= 1 or context.reference.shape[1] != 40:
        raise ValueError("Invalid Two-Forward context or probability.")
    history = state_map(context, lambda x: x[:, 20:])
    if probability == 0:
        return physical_batch(codec, history, target, beta), 0.
    candidates = candidate_histories(prior, codec, context, chunk_size)
    mask = (torch.rand(len(beta), 20, generator=generator) < probability).to(beta.device)
    history = BodyState(*(torch.where(mask.reshape(*mask.shape, *([1] * (getattr(history, key).ndim - 2))),
                                      getattr(candidates, key), getattr(history, key)) for key in FIELDS))
    return physical_batch(codec, history, target, beta), float(mask.float().mean())


class PriorSequences:
    """Small, audited GT-body cache. Only a requested train/dev group is exposed."""
    def __init__(self, directory, group):
        if group not in ("train", "dev"):
            raise ValueError("P training/evaluation accepts only train or dev.")
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text())
        if not report.get("completed") or file_sha256(directory / "sequences.pt") != report["cache_sha256"]:
            raise ValueError("Incomplete or altered P sequence cache.")
        payload = torch.load(directory / "sequences.pt", map_location="cpu", weights_only=True, mmap=True)
        self.identity = payload["identity"]
        if self.identity != report["identity"] or self.identity.get("scope") != "audited_prior_train_dev_sequences":
            raise ValueError("P sequence cache identity mismatch.")
        splits = self.identity["splits"]
        names = [take for group_names in splits.values() for take in group_names]
        if len(names) != len(set(names)) or set(payload["take_names"]) != set(splits["train"] + splits["dev"]):
            raise ValueError("P sequence cache leaks across splits or lacks takes.")
        if len(payload["take_names"]) != len(set(payload["take_names"])):
            raise ValueError("Expected exactly one clean episode per take.")
        selected = [i for i, take in enumerate(payload["take_names"]) if take in splits[group]]
        self.takes = [payload["take_names"][i] for i in selected]
        self.group = group
        self.states = BodyState(*(payload[key][selected] for key in FIELDS))
        self.beta = payload["beta_boot"][selected]
        self.stats = payload["stats"]
        n = len(self.takes)
        shapes = {"joints": (n, 200, 22, 4, 4), "reference": (n, 200, 4, 4), "auxiliary": (n, 200, 36)}
        for key, shape in shapes.items():
            value = getattr(self.states, key)
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid P sequence tensor: {key}.")
        if self.beta.shape != (n, 10) or not torch.isfinite(self.beta).all():
            raise ValueError("Invalid model bootstrap shape.")

    def indices(self, minimum=20, maximum=200, stride=1):
        if not 20 <= minimum < maximum <= 200 or stride < 1:
            raise ValueError("Invalid P target frame range.")
        return torch.tensor([(take, t) for take in range(len(self.takes))
                             for t in range(minimum, maximum, stride)], dtype=torch.long)

    def examples(self, indices, device, context_length=20):
        indices = indices.cpu()
        seq, t = indices.T
        if context_length not in (20, 40) or bool((t < context_length).any()) or bool((t >= 200).any()):
            raise ValueError("Requested history would read outside the episode.")
        times = t[:, None] + torch.arange(-context_length, 0)[None]
        history = state_map(self.states, lambda x: x[seq[:, None], times].to(device))
        target = state_map(self.states, lambda x: x[seq, t].to(device))
        return history, target, self.beta[seq].to(device)


def summarize(values, indices, takes):
    if len(values) != len(indices) or not bool(torch.isfinite(values).all()):
        raise ValueError("Expected one finite metric per evaluated frame.")
    per_take = {name: float(values[indices[:, 0] == i].double().mean())
                for i, name in enumerate(takes) if bool((indices[:, 0] == i).any())}
    return {"mean": sum(per_take.values()) / len(per_take), "per_take": per_take}


@torch.no_grad()
def measure(codec, prediction, batch, target, smpl):
    error = fk_position_errors(codec, prediction, batch, smpl)
    state = codec.decode_current(prediction[:, 0], batch["previous_reference"])
    rotation = state.joints[..., :3, :3].transpose(-1, -2) @ target.joints[..., :3, :3]
    angle = torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    values = {"fk22_mm": error.norm(dim=-1).mean(-1) * 1000,
              "root_mm": error[:, 0].norm(dim=-1) * 1000,
              "root_relative_mm": (error - error[:, :1]).norm(dim=-1).mean(-1) * 1000,
              "dense22_mm": (state.joints[..., :3, 3] - batch["target_joints"]).norm(dim=-1).mean(-1) * 1000,
              "rotation_deg": torch.rad2deg(angle).mean(-1)}
    return {key: value.cpu() for key, value in values.items()}, (error + batch["target_joints"]).cpu(), state


def prediction_for(model, batch):
    if isinstance(model, str):
        return batch[{"hold": "base_mu", "constant_velocity": "velocity_mu"}[model]]
    return predict(model, batch)


@torch.no_grad()
def evaluate_single(model, data, codec, smpl, device, batch_size=32):
    if not isinstance(model, str):
        model.eval()
    indices = data.indices()
    arrays, poses, truth = {}, [], []
    for start in range(0, len(indices), batch_size):
        history, target, beta = data.examples(indices[start:start + batch_size], device)
        batch = physical_batch(codec, history, target, beta)
        metrics, joints, _ = measure(codec, prediction_for(model, batch), batch, target, smpl)
        for key, value in metrics.items():
            arrays.setdefault(key, []).append(value)
        poses.append(joints)
        truth.append(batch["target_joints"].cpu())
    arrays = {key: torch.cat(value) for key, value in arrays.items()}
    matched = indices[:, 1] >= 40
    summary = {key: summarize(value, indices, data.takes) for key, value in arrays.items()}
    summary["matched_fk22_mm"] = summarize(arrays["fk22_mm"][matched], indices[matched], data.takes)
    return summary, {"indices": indices, "takes": data.takes, "metrics": arrays,
                     "joints": torch.cat(poses), "gt_joints": torch.cat(truth)}


@torch.no_grad()
def evaluate_rollout(model, data, codec, smpl, device, batch_size=32):
    """108 fixed dev starts, ten predicted frames per start, no GT feedback."""
    if not isinstance(model, str):
        model.eval()
    indices = data.indices(20, 181, 20)
    metrics, poses, truth = {h: [] for h in (1, 2, 4, 8, 10)}, {}, {}
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        history, _, beta = data.examples(selected, device)
        for horizon in range(1, 11):
            future = selected.clone()
            future[:, 1] += horizon - 1
            # Target is offline scoring only; prediction_for reads history and base_mu only.
            _, target, _ = data.examples(future, device)
            batch = physical_batch(codec, history, target, beta)
            prediction = prediction_for(model, batch)
            state = codec.decode_current(prediction[:, 0], batch["previous_reference"])
            if horizon in metrics:
                measured, joints, _ = measure(codec, prediction, batch, target, smpl)
                metrics[horizon].append(measured["fk22_mm"])
                poses.setdefault(horizon, []).append(joints)
                truth.setdefault(horizon, []).append(batch["target_joints"].cpu())
            history = BodyState(*(torch.cat((getattr(history, key)[:, 1:], getattr(state, key)[:, None]), dim=1)
                                  for key in FIELDS))
    arrays = {h: torch.cat(value) for h, value in metrics.items()}
    return {str(h * 100): summarize(value, indices, data.takes) for h, value in arrays.items()}, {
        "indices": indices, "takes": data.takes, "errors_mm": arrays,
        "joints": {h: torch.cat(value) for h, value in poses.items()},
        "gt_joints": {h: torch.cat(value) for h, value in truth.items()}}
