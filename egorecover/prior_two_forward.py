"""Detached causal P forecasts replace a physical suffix before supervised P."""

import torch

from .prior_data import predictor_forward, prior_example, prior_inputs
from .state import state_cat, state_map

REPLACEMENT_COUNTS = (1, 3, 5, 10, 20)


def replacement_count(step, start, duration):
    if duration < 1 or not start < step <= start + duration:
        raise ValueError("Replacement curriculum step is outside its phase.")
    progress = (step - start - 1) / (duration - 1) if duration > 1 else 1.0
    return REPLACEMENT_COUNTS[min(int(progress * len(REPLACEMENT_COUNTS)), len(REPLACEMENT_COUNTS) - 1)]


@torch.no_grad()
def two_forward_example(model, codec, batch, frame, replace_frames):
    """Each first-pass forecast at i sees GT[i-20:i], never GT[i] or later.

    The second pass's history is GT[frame-20:frame-replace_frames] followed
    by detached, independently teacher-forced first-pass forecasts. Encoding,
    CV and supervised target coordinates are recomputed after physical mixing.
    The caller performs the differentiable second P pass with this example.
    """
    if not 1 <= replace_frames <= 20 or frame < 20 + replace_frames:
        raise ValueError("Two-forward needs twenty past states before every replaced frame.")
    histories = []
    for target_frame in range(frame - replace_frames, frame):
        history = state_map(batch.target(target_frame - 1, 20), lambda value: value.detach())
        histories.append(history)
    # Encode all shifted histories together instead of launching20 small GPU
    # geometry/validation pipelines when the entire suffix is replaced.
    first = prior_inputs(codec, state_cat(histories, dim=0))
    was_training = model.training
    model.eval()
    try:
        prediction = predictor_forward(model, first)
        forecast = codec.decode_current(prediction[:, 0], first["reference"])
    finally:
        model.train(was_training)
    forecast = state_map(
        forecast,
        lambda value: value.reshape(replace_frames, len(batch), *value.shape[1:]).transpose(0, 1).detach(),
    )
    history = state_map(batch.target(frame - 1, 20), lambda value: value[:, : 20 - replace_frames].detach())
    mixed = state_cat((history, forecast))
    truth = state_map(batch.target(frame), lambda value: value[:, -1].detach())
    return prior_example(codec, mixed, truth)
