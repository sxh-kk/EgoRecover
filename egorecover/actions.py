"""Four actions, ordered a11/a10/a01/a00 (visual bit first)."""

from enum import IntEnum

import torch


class Action(IntEnum):
    A11 = 0
    A10 = 1
    A01 = 2
    A00 = 3


ACTION_NAMES = ("a11", "a10", "a01", "a00")


def binary_mask(value, shape, device, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be a tensor with shape {tuple(shape)}.")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}.")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only 0/1 or bool values.")
    return value.bool()


def _availability(img_available, traj_available):
    if not isinstance(img_available, torch.Tensor) or img_available.ndim != 2 or img_available.shape[1] != 1:
        raise ValueError("Availability must have shape [B,1].")
    shape, device = img_available.shape, img_available.device
    if shape[0] == 0:
        raise ValueError("Availability requires a nonempty batch.")
    return (
        binary_mask(img_available, shape, device, "img_available"),
        binary_mask(traj_available, shape, device, "traj_available"),
    )


def action_masks(action, *, img_available, traj_available):
    """Return (img_mask, traj_mask); True means replace by a missing token.

    action is an Action/int/name shared by the batch, or an integer [B] tensor.
    Actual unavailability always overrides a request to retain a modality.
    """
    img_available, traj_available = _availability(img_available, traj_available)
    batch, device = img_available.shape[0], img_available.device
    if isinstance(action, str):
        if action not in ACTION_NAMES:
            raise ValueError(f"Unknown action {action!r}; expected {ACTION_NAMES}.")
        action = ACTION_NAMES.index(action)
    ids = torch.as_tensor(action, device=device)
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        raise ValueError("Actions must be integer IDs 0..3.")
    if ids.ndim == 0:
        ids = ids.expand(batch)
    if ids.shape != (batch,) or not bool(((ids >= 0) & (ids < 4)).all()):
        raise ValueError("Actions must be scalar or [B] integer IDs 0..3.")
    ids = ids.long()[:, None]
    return (~img_available | (ids >= 2), ~traj_available | (ids.remainder(2) == 1))


def action_equivalence(*, img_available, traj_available):
    """Map each nominal action to its first equivalent action, shape [B,4].

    The earliest representative preserves a11 as the zero-gain baseline.
    Example: missing visual -> [0,1,0,1]; both missing -> [0,0,0,0].
    The same mapping should be used for future utility labels and selection.
    """
    img_available, traj_available = _availability(img_available, traj_available)
    ids = torch.arange(4, device=img_available.device)[None]
    img_drop = ~img_available | (ids >= 2)
    traj_drop = ~traj_available | (ids.remainder(2) == 1)
    effective = img_drop.long() * 2 + traj_drop.long()
    equal = effective[:, :, None] == effective[:, None, :]
    return equal.long().argmax(dim=-1)
