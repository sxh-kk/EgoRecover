"""Causal examples for a20-frame history-only one-frame motion predictor."""

import torch

from .codec import planar_reference
from .state import state_map

TENSOR_FIELDS = ("history", "base", "target", "reference", "truth_xyz", "body_valid")


def reconstructed_history_count(generated_frames, window, revise_recent, history=20):
    count = max(0, generated_frames)
    if count and revise_recent:
        count += window - 1
    return min(history, count)


def prior_inputs(codec, history, *, generated_frames=None):
    """Encode predictor inputs using past physical states only."""
    if history.joints.ndim != 5 or history.joints.shape[1] != 20:
        raise ValueError("P requires exactly20 previous body states.")
    last = state_map(history, lambda value: value[:, -1])
    before = state_map(history, lambda value: value[:, -2])
    reference = last.reference
    encoded = codec.encode_history(history, planar_reference(history.reference[:, 0]))
    base = codec.constant_velocity_prior(before, last)[:, None]
    if generated_frames is None:
        body_valid = torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
    else:
        count = torch.as_tensor(generated_frames, device=encoded.device).clamp(0, 20).reshape(-1, 1)
        body_valid = torch.arange(20, device=encoded.device)[None] >= 20 - count
        body_valid = body_valid.expand(encoded.shape[:2])
    return {
        "history": encoded,
        "base": base,
        "reference": reference,
        "body_valid": body_valid,
    }


def prior_example(codec, history, truth, *, generated_frames=None):
    """Only target/truth_xyz depend on the current label; inputs use history."""
    inputs = prior_inputs(codec, history, generated_frames=generated_frames)
    return {
        **inputs,
        "target": codec.encode_current(truth, inputs["reference"])[:, None],
        "truth_xyz": truth.joints[..., :3, 3],
    }


def clean_batch(codec, batch, frame=20):
    history = state_map(batch.target(frame - 1, 20), lambda value: value.detach())
    truth = state_map(batch.target(frame), lambda value: value[:, -1])
    return prior_example(codec, history, truth)


def concatenate_examples(examples):
    if not examples:
        raise ValueError("No P examples to concatenate.")
    return {key: torch.cat([example[key] for example in examples], dim=0) for key in TENSOR_FIELDS}


def select_examples(tensors, indices, device):
    return {key: value[indices].to(device) for key, value in tensors.items()}


def predictor_forward(model, example):
    valid = torch.ones(example["history"].shape[:2], dtype=torch.bool, device=example["history"].device)
    return model(example["history"], valid, example["base"], history_body_valid=example["body_valid"])
