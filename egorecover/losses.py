"""Current supervision and stop-gradient paired full-response supervision."""

import copy
import hashlib
from contextlib import contextmanager

import torch

from .flow import build_source, flow_loss
from .intervention import intervention_state
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


def geometry_loss(prediction, target, scale):
    return (prediction - target).square().sum(-1).mean() / scale**2


@contextmanager
def forward_rng(tensor, seed, role):
    """Match dropout per objective call without advancing shared RNG streams."""
    if seed is None:
        yield
        return
    value = int.from_bytes(hashlib.sha256(f"{seed}/{role}".encode()).digest()[:8], "big") % (2**63 - 1)
    device = tensor.device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(value)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(value)
        yield


def paired_feedback(
    system,
    ema,
    current,
    truth_current,
    next_observations,
    next_epsilon,
    joints,
    *,
    mode="paired",
    truth_next=None,
    intervention="full_state",
):
    """Stop only shared revised past; retain predicted current's entire graph.

    Both branches reuse the raw next observation, standard epsilon and fixed FK.
    Each branch independently recomputes P, common reference, center and sensors.
    """
    if ema.training or any(p.requires_grad for p in ema.parameters()):
        raise ValueError("The subsequent EMA model must be frozen and in eval mode.")
    predicted_cache = current.cache.replace_current(current.current)
    reference_cache = current.cache.replace_current(
        intervention_state(current.current, truth_current, joints, intervention)
    )
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
    truth_next=None,
    detach_history=True,
    feedback_intervention="full_state",
    forward_seed=None,
):
    cfg = system.config
    input_cache = cache.detach() if detach_history else cache
    y, reference = system.prepare(input_cache, observations)
    clean = system.encode(truth_window, reference).detach()
    source = build_source(y["center"], epsilon, mode=cfg.source, sigma=cfg.sigma)
    with forward_rng(epsilon, forward_seed, "flow"):
        flow_components = flow_loss(model, clean, source, y, tau, return_components=True)
    base = flow_components["flow"]
    zero = base.new_zeros(())
    current_loss, feedback_loss = zero, zero
    current = None
    if cfg.current_weight or (cfg.feedback != "none" and cfg.feedback_weight):
        with forward_rng(epsilon, forward_seed, "endpoint"):
            current = system.reconstruct(model, input_cache, observations, epsilon)
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
                    intervention=feedback_intervention,
                )
            elif cfg.feedback == "feature":
                feedback_loss = (current.codes[:, -1] - clean[:, -1]).square().mean()
            elif cfg.feedback == "geometry":
                feedback_loss = geometry_loss(joints(current.current), joints(truth_current).detach(), cfg.scale_m)
    total = base + cfg.current_weight * current_loss + cfg.feedback_weight * feedback_loss
    return {
        "loss": total,
        "flow": base,
        "hand_pca": flow_components["hand_pca"],
        "current": current_loss,
        "feedback": feedback_loss,
        "result": current,
    }


def two_step_objective(
    system,
    model,
    teacher,
    cache,
    observations,
    truth_window,
    epsilon,
    tau,
    joints,
    *,
    next_observations,
    next_truth_window,
    next_epsilon,
    next_tau=None,
    forward_seed=None,
    detach_between=False,
):
    """Average two real online calls, keeping the cache path differentiable."""
    if system.config.feedback != "none" or not system.config.current_weight:
        raise ValueError("Two-step baseline requires current geometry and no feedback objective.")
    first = training_objective(
        system, model, teacher, cache, observations, truth_window, epsilon, tau, joints, forward_seed=forward_seed
    )
    second = training_objective(
        system,
        model,
        teacher,
        first["result"].cache,
        next_observations,
        next_truth_window,
        next_epsilon,
        tau if next_tau is None else next_tau,
        joints,
        detach_history=detach_between,
        forward_seed=None if forward_seed is None else forward_seed + 1,
    )
    return {
        **{key: (first[key] + second[key]) / 2 for key in ("loss", "flow", "hand_pca", "current", "feedback")},
        "result": first["result"],
        "next_result": second["result"],
        "first_loss": first["loss"],
        "next_loss": second["loss"],
    }
