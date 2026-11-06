"""x0 endpoint Flow, including differentiable one/two/four/ten-step Euler."""

import torch


def build_source(center, epsilon, *, mode, sigma):
    if center.shape != epsilon.shape or center.ndim != 3 or center.shape[-1] != 243:
        raise ValueError("Center and standard noise must be [B,W,243].")
    if center.device != epsilon.device or center.dtype != epsilon.dtype:
        raise ValueError("Center and noise must share device and dtype.")
    if not bool(torch.isfinite(epsilon).all()) or not bool(torch.isfinite(center).all()):
        raise ValueError("Nonfinite center or noise.")
    if mode not in ("gaussian", "prediction"):
        raise ValueError("Unknown source mode.")
    return sigma * epsilon + (center if mode == "prediction" else 0)


def solve(model, source, condition, *, steps):
    """At tau=1 start from S, at tau=0 reach clean body; one step is x0(S,1).

    No no_grad decorator: frozen EMA parameters still transmit input gradients.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    value = source
    for index in range(steps):
        tau = value.new_full((len(value),), 1 - index / steps)
        endpoint = model(value, tau, condition)
        if endpoint.shape != source.shape or not bool(torch.isfinite(endpoint).all()):
            raise ValueError("Invalid Flow endpoint.")
        value = endpoint if index == steps - 1 else value + (endpoint - value) / (steps - index)
    return value


def flow_loss(model, target, source, condition, tau):
    if target.shape != source.shape or not bool(torch.isfinite(target).all()):
        raise ValueError("Invalid clean target window.")
    if tau.shape != (len(source),) or not bool(((tau > 0) & (tau <= 1)).all()):
        raise ValueError("Training tau must be [B] in (0,1].")
    noisy = tau[:, None, None] * source + (1 - tau[:, None, None]) * target
    prediction = model(noisy, tau, condition)
    weights = target.new_ones(243)
    weights[198:207] = 8
    return ((prediction - target).square() * weights).sum(-1).mean() / weights.sum()
