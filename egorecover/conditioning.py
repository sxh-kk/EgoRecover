"""Strict normalized-tensor boundary between the future codec and G."""

from collections.abc import Mapping

import torch

from egorecover.actions import Action, action_masks, binary_mask

REQUIRED_FIELDS = frozenset(
    {"history_motion", "history_valid", "prior_mu", "traj", "img_embs", "traj_mask", "img_mask", "valid_frames"}
)
ALLOWED_FIELDS = REQUIRED_FIELDS | {"loss_mask"}


def floating_tensor(value, shape, reference, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}.")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point.")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must match history_motion device and dtype.")
    return value


def finite_payload(value, visible, name):
    """Remove masked NaN/Inf before projections without editing caller tensors."""
    result = torch.where(visible[..., None], value, torch.zeros_like(value))
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} has NaN/Inf at an unmasked position.")
    return result


def prepare_conditioning(y, *, history_length=None):
    """Validate and sanitize G's conditions, returning a new dictionary.

    H, mu and the current target must already share coordinates/statistics.
    Sensor availability never clears valid_frames. Unknown keys are rejected
    so fault metadata, clean observations and old repaint/CFG fields cannot
    silently become model inputs. Masks are returned as bool tensors.
    """
    if not isinstance(y, Mapping):
        raise TypeError("y must be a mapping.")
    missing, extra = REQUIRED_FIELDS - y.keys(), y.keys() - ALLOWED_FIELDS
    if missing or extra:
        raise ValueError(f"Invalid G conditions: missing={sorted(missing)}, unexpected={sorted(extra)}.")
    history = y["history_motion"]
    if not isinstance(history, torch.Tensor) or history.ndim != 3 or history.shape[-1] != 243:
        raise ValueError("history_motion must have shape [B,L,243].")
    batch, length, _ = history.shape
    if batch < 1 or length < 1 or not history.is_floating_point():
        raise ValueError("history_motion requires a positive batch/history length and floating dtype.")
    if history_length is not None and length != history_length:
        raise ValueError(f"Expected history length {history_length}, got {length}.")
    result = {}
    for name, shape in (
        ("history_valid", (batch, length)),
        ("valid_frames", (batch, 1)),
        ("traj_mask", (batch, 1)),
        ("img_mask", (batch, 1)),
    ):
        result[name] = binary_mask(y[name], shape, history.device, name)
    if "loss_mask" in y:
        result["loss_mask"] = binary_mask(y["loss_mask"], (batch, 1), history.device, "loss_mask")
        if bool((result["loss_mask"] & ~result["valid_frames"]).any()):
            raise ValueError("loss_mask cannot supervise invalid current frames.")
    result["history_motion"] = finite_payload(history, result["history_valid"], "history_motion")
    for name, width, visible in (
        ("prior_mu", 243, result["valid_frames"]),
        ("traj", 18, result["valid_frames"] & ~result["traj_mask"]),
        ("img_embs", 1024, result["valid_frames"] & ~result["img_mask"]),
    ):
        value = floating_tensor(y[name], (batch, 1, width), history, name)
        result[name] = finite_payload(value, visible, name)
    return result


def build_conditioning(
    *,
    history_motion,
    history_valid,
    prior_mu,
    traj,
    img_embs,
    img_available,
    traj_available,
    action=Action.A11,
    valid_frames=None,
    loss_mask=None,
):
    """Build one physical time step from explicit availability and an action.

    traj/img_embs contain only the current observation. Missing payloads may
    contain NaN/Inf if availability or the chosen action masks them out.
    P and any history producer are frozen/detached by the caller.
    """
    img_mask, traj_mask = action_masks(action, img_available=img_available, traj_available=traj_available)
    if valid_frames is None:
        valid_frames = torch.ones_like(img_mask)
    y = dict(
        history_motion=history_motion,
        history_valid=history_valid,
        prior_mu=prior_mu,
        traj=traj,
        img_embs=img_embs,
        img_mask=img_mask,
        traj_mask=traj_mask,
        valid_frames=valid_frames,
    )
    if loss_mask is not None:
        y["loss_mask"] = loss_mask
    return prepare_conditioning(y)
