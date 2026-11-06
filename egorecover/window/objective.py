"""Current supervision and stop-gradient paired full-response supervision."""

import copy

import torch

from .flow import build_source, flow_loss
from .state import state_map


def frozen_ema(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


@torch.no_grad()
def update_ema(ema, model, decay):
    source = model.state_dict()
    for key, value in ema.state_dict().items():
        if value.is_floating_point():
            value.lerp_(source[key], 1 - decay)
        else:
            value.copy_(source[key])
    ema.eval()


class DenseJoints:
    """Asset-free diagnostic only; production objectives use FixedShapeFK."""

    diagnostic_only = True

    def __call__(self, body):
        return body.joints[..., :3, 3]


def geometry_loss(prediction, target, scale):
    return (prediction - target).square().sum(-1).mean() / scale**2


def paired_feedback(
    system, ema, current, truth_current, next_observations, next_epsilon, joints, *, mode="paired", truth_next=None
):
    """Stop only shared revised past; retain predicted current's entire graph.

    Both branches reuse the raw next observation, standard epsilon and fixed FK.
    Each branch independently recomputes P, common reference, center and sensors.
    """
    if ema.training or any(p.requires_grad for p in ema.parameters()):
        raise ValueError("The subsequent EMA model must be frozen and in eval mode.")
    predicted_cache = current.cache.replace_current(current.current)
    reference_cache = current.cache.replace_current(state_map(truth_current, lambda x: x.detach()))
    if next_epsilon.ndim != 4 or next_epsilon.shape[0] < 1:
        raise ValueError("Feedback noise must be [draws,B,W,243].")
    losses = []
    for epsilon in next_epsilon:
        predicted = system.reconstruct(ema, predicted_cache, next_observations, epsilon)
        predicted_joints = joints(predicted.current)
        with torch.no_grad():
            if mode == "paired":
                reference = system.reconstruct(ema, reference_cache, next_observations, epsilon)
                target = joints(reference.current)
            elif mode == "direct" and truth_next is not None:
                target = joints(truth_next)
            else:
                raise ValueError("Feedback requires paired responses or direct next truth.")
        losses.append(geometry_loss(predicted_joints, target.detach(), system.config.scale_m))
    return torch.stack(losses).mean()


def training_objective(
    system,
    model,
    ema,
    cache,
    observations,
    truth_window,
    epsilon,
    tau,
    joints,
    *,
    next_observations=None,
    next_epsilon=None,
    truth_next=None
):
    cfg = system.config
    y, reference = system.prepare(cache.detach(), observations)
    clean = system.encode(truth_window, reference).detach()
    source = build_source(y["center"], epsilon, mode=cfg.source, sigma=cfg.sigma)
    base = flow_loss(model, clean, source, y, tau)
    zero = base.new_zeros(())
    current_loss, feedback_loss = zero, zero
    current = None
    if cfg.current_weight or (cfg.feedback != "none" and cfg.feedback_weight):
        current = system.reconstruct(model, cache.detach(), observations, epsilon)
        truth_current = state_map(truth_window, lambda x: x[:, -1])
        if cfg.current_weight:
            current_loss = geometry_loss(joints(current.current), joints(truth_current).detach(), cfg.scale_m)
        if cfg.feedback != "none" and cfg.feedback_weight:
            if cfg.feedback in ("paired", "direct"):
                if next_observations is None or next_epsilon is None:
                    raise ValueError("Subsequent supervision requires next observations and noise.")
                if next_epsilon.shape[0] != cfg.feedback_draws:
                    raise ValueError("Feedback draw count disagrees with configuration.")
                feedback_loss = paired_feedback(
                    system,
                    ema,
                    current,
                    truth_current,
                    next_observations,
                    next_epsilon,
                    joints,
                    mode=cfg.feedback,
                    truth_next=truth_next,
                )
            elif cfg.feedback == "feature":
                feedback_loss = (current.codes[:, -1] - clean[:, -1]).square().mean()
            elif cfg.feedback == "geometry":
                feedback_loss = geometry_loss(joints(current.current), joints(truth_current).detach(), cfg.scale_m)
    total = base + cfg.current_weight * current_loss + cfg.feedback_weight * feedback_loss
    return {"loss": total, "flow": base, "current": current_loss, "feedback": feedback_loss, "result": current}
