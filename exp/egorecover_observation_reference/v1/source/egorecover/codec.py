"""Dense v4_beta geometry without SMPL assets or ground-truth fallbacks.

The physical state keeps both predicted dense joints and its predicted internal
reference. FK projection, if used for evaluation, must not replace this state.
All world coordinates below must use one explicitly chosen floor/calibration.
"""

from dataclasses import dataclass

import torch
from torch import nn

from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


def rigid_transform(rotation, translation):
    top = torch.cat((rotation, translation[..., None]), dim=-1)
    bottom = top.new_zeros(*top.shape[:-2], 1, 4)
    bottom[..., 0, 3] = 1
    return torch.cat((top, bottom), dim=-2)


def rigid_inverse(transform):
    rotation = transform[..., :3, :3].transpose(-1, -2)
    return rigid_transform(rotation, -(rotation @ transform[..., :3, 3, None])[..., 0])


def transform_from_9d(value):
    if value.shape[-1] != 9 or not bool(torch.isfinite(value).all()):
        raise ValueError("A transform requires nine finite rotation6d+xyz channels.")
    rotation = rotation_6d_to_matrix(value[..., :6])
    # Zero/collinear rotation codes do not define a rotation. Do not quietly
    # turn them into an invalid SE(3) matrix and contaminate a physical cache.
    determinant = torch.linalg.det(rotation.float())
    if not bool((determinant > 0.99).all()):
        # A finite large code can overflow the squared float32 norm. This is
        # different from genuinely zero/collinear axes. Retry the SAME 6D
        # projection in float64; no replacement axis or clipping is invented.
        precise_rotation = rotation_6d_to_matrix(value[..., :6].double())
        determinant64 = torch.linalg.det(precise_rotation)
        if not bool((determinant64 > 0.99).all()):
            raise ValueError(
                "Degenerate 6D rotation cannot define a physical transform. "
                f"min_det32={float(determinant.min()):.6g}, min_det64={float(determinant64.min()):.6g}, "
                f"rotation_code_max_abs={float(value[..., :6].abs().max()):.6g}"
            )
        rotation = precise_rotation.to(value.dtype)
    return rigid_transform(rotation, value[..., 6:9])


def transform_to_9d(transform):
    return torch.cat((matrix_to_rotation_6d(transform[..., :3, :3]), transform[..., :3, 3]), dim=-1)


def planar_reference(head_transform):
    """World-from-heading reference: heading about z, horizontal translation.

    atan2 handles 180-degree headings. A vertical forward axis has no defined
    heading and raises instead of selecting one from hidden/future data.
    """
    forward = head_transform[..., :2, 0]
    if not bool((torch.linalg.vector_norm(forward, dim=-1) > 1e-6).all()):
        raise ValueError("A vertical forward axis has no planar heading.")
    angle = torch.atan2(forward[..., 1], forward[..., 0])
    cosine, sine = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    rotation = torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1)
    rotation = rotation.reshape(*angle.shape, 3, 3)
    translation = torch.cat((head_transform[..., :2, 3], zero[..., None]), dim=-1)
    return rigid_transform(rotation, translation)


@dataclass(frozen=True)
class BodyState:
    joints: torch.Tensor  # [...,22,4,4], dense joint transforms in world
    reference: torch.Tensor  # [...,4,4], world-from-internal-reference
    auxiliary: torch.Tensor  # [...,36], hands(24), contacts(2), beta(10)


class MotionCodec(nn.Module):
    """Normalize and re-encode dense states using original E7 training stats."""

    def __init__(self, stats, *, reference_mode="planar"):
        super().__init__()
        if reference_mode not in ("planar", "legacy_se3"):
            raise ValueError("reference_mode must be planar or legacy_se3.")
        self.reference_mode = reference_mode
        for kind, width in (("motion", 243), ("traj", 18)):
            for suffix in ("mean", "std"):
                name = f"{kind}_{suffix}"
                value = torch.as_tensor(stats[name], dtype=torch.float32).reshape(-1).clone()
                if value.shape != (width,) or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Invalid {name}; expected {width} finite values.")
                if suffix == "std":
                    if bool((value < 0).any()):
                        raise ValueError(f"Negative standard deviation in {name}.")
                    value = torch.where(value.abs() < 1e-8, torch.ones_like(value), value) + 1e-6
                self.register_buffer(name, value)

    def normalize(self, value, kind="motion"):
        return (value - getattr(self, kind + "_mean")) / getattr(self, kind + "_std")

    def denormalize(self, value, kind="motion"):
        return value * getattr(self, kind + "_std") + getattr(self, kind + "_mean")

    def canonical_reference(self, reference):
        """Enforce the v4 internal heading/xy frame; never flatten body/sensor pose.

        legacy_se3 is only for reproducing unconstrained historical experiments.
        Project the INITIAL frame too: planar deltas cannot repair a tilted base.
        """
        if reference.shape[-2:] != (4, 4) or not bool(torch.isfinite(reference).all()):
            raise ValueError("Expected a finite physical reference transform.")
        return planar_reference(reference) if self.reference_mode == "planar" else reference

    def project_motion_reference(self, motion):
        """Project only the reference channels of an x0/P prediction, never noise.

        Continuous Flow states are intentionally unconstrained. This operation
        is for decoding/inspection and leaves the other 234 channels untouched.
        """
        raw = self.denormalize(motion)
        reference = self.canonical_reference(transform_from_9d(raw[..., 198:207]))
        result = motion.clone()
        result[..., 198:207] = (transform_to_9d(reference) - self.motion_mean[198:207]) / self.motion_std[198:207]
        return result

    def encode_current(self, state, previous_reference):
        """The target delta bridges from the PREVIOUS PREDICTED reference."""
        reference = self.canonical_reference(state.reference)
        previous_reference = self.canonical_reference(previous_reference)
        local = rigid_inverse(reference)[..., None, :, :] @ state.joints
        delta = rigid_inverse(previous_reference) @ reference
        raw = torch.cat((transform_to_9d(local).flatten(-2), transform_to_9d(delta), state.auxiliary), dim=-1)
        if raw.shape[-1] != 243 or not bool(torch.isfinite(raw).all()):
            raise ValueError("Invalid dense body state.")
        return self.normalize(raw)

    def decode_current(self, motion, previous_reference):
        """Decode a submitted prediction without any current sensor or labels."""
        raw = self.denormalize(motion)
        local = transform_from_9d(raw[..., :198].unflatten(-1, (22, 9)))
        delta = self.canonical_reference(transform_from_9d(raw[..., 198:207]))
        reference = self.canonical_reference(self.canonical_reference(previous_reference) @ delta)
        return BodyState(reference[..., None, :, :] @ local, reference, raw[..., 207:])

    def encode_history(self, states, outer_reference):
        """states has [B,L,...]; first reference is absolute in the new window."""
        if states.reference.ndim != 4 or states.reference.shape[1] < 1:
            raise ValueError("History states require [B,L,4,4] references.")
        previous = torch.cat((outer_reference[:, None], states.reference[:, :-1]), dim=1)
        return self.encode_current(states, previous)

    def continuation_prior(self, last_state):
        """Repeat physical last state; the current reference delta is identity."""
        return self.encode_current(last_state, last_state.reference)

    def constant_velocity_prior(self, previous_state, last_state):
        """One-step world translation / SO(3) rotation extrapolation.

        Body joints and their internal reference are extrapolated physically.
        Hand PCA, contact and beta channels are held (no licensed hand basis is
        needed); the body baseline never linearly extrapolates rotation matrices.
        """

        def extrapolate(previous, last):
            delta_rotation = last[..., :3, :3] @ previous[..., :3, :3].transpose(-1, -2)
            rotation = delta_rotation @ last[..., :3, :3]
            translation = 2 * last[..., :3, 3] - previous[..., :3, 3]
            return rigid_transform(rotation, translation)

        predicted = BodyState(
            extrapolate(previous_state.joints, last_state.joints),
            extrapolate(previous_state.reference, last_state.reference),
            last_state.auxiliary,
        )
        return self.encode_current(predicted, last_state.reference)

    def encode_observation(self, absolute_traj, previous_reference, available):
        """18D current head condition in the committed body's reference chain.

        The last nine channels bridge previous_reference to the observed
        current heading. This agrees with E7 under clean, aligned history and
        stays meaningful under predicted-history drift. No past raw sensor is
        required by G. Unavailable payloads are sanitized BEFORE geometry.
        """
        if available.shape != absolute_traj.shape[:-1] or available.dtype != torch.bool:
            raise ValueError("available must be bool and match trajectory batch/time dimensions.")
        identity = absolute_traj.new_tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])
        safe = torch.where(available[..., None], absolute_traj, identity)
        head = transform_from_9d(safe)
        reference = planar_reference(head)
        local = rigid_inverse(reference) @ head
        delta = rigid_inverse(self.canonical_reference(previous_reference)) @ reference
        encoded = self.normalize(torch.cat((transform_to_9d(local), transform_to_9d(delta)), dim=-1), "traj")
        return torch.where(available[..., None], encoded, torch.zeros_like(encoded))
