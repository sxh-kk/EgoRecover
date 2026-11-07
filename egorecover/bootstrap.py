"""Pointwise head-aligned neutral-body startup without body labels or E7."""

import torch

from .codec import BodyState, planar_reference, rigid_transform, transform_from_9d


class HeadAlignedBootstrap:
    """Place a fixed neutral SMPL22 skeleton using each available head pose.

    This is an initialization prior, not a recovered full-body pose. The device
    origin is approximated by SMPL head joint 15; no device-to-head calibration,
    body labels, motion checkpoint, or future observation is consulted.
    """

    def __init__(self, rest_joints):
        rest = torch.as_tensor(rest_joints)
        if rest.shape != (22, 3) or not rest.is_floating_point() or not bool(torch.isfinite(rest).all()):
            raise ValueError("Provide 22 finite floating-point neutral SMPL joint positions in Y-up coordinates.")
        # Official clips are assembled on CPU. Cache this tiny template once,
        # avoiding a synchronous device-to-host copy for every startup frame.
        self.rest_joints = rest.detach().cpu().clone()

    @classmethod
    @torch.no_grad()
    def from_smpl(cls, smpl):
        """Read the licensed layer's neutral skeleton using only zero beta."""
        template = getattr(smpl, "v_template", None)
        if not isinstance(template, torch.Tensor) or not template.is_floating_point():
            template = next(
                (value for value in (*smpl.parameters(), *smpl.buffers()) if value.is_floating_point()),
                torch.empty((), dtype=torch.float32),
            )
        beta = template.new_zeros(1, 10)
        joints = smpl(betas=beta, return_verts=False).joints
        if joints.ndim != 3 or joints.shape[0] != 1 or joints.shape[1] < 22 or joints.shape[2] != 3:
            raise ValueError("SMPL must expose one neutral skeleton containing at least 22 joints.")
        return cls(joints[0, :22])

    @property
    def protocol(self):
        """A fresh JSON-compatible description suitable for run identity."""
        return {
            "version": "head_aligned_neutral_v1",
            "shape_policy": "neutral_beta_zero",
            "body_joints": 22,
            "head_anchor": 15,
            "device_origin_assumption": "device_origin_approximately_smpl_head_joint_15",
            "heading": "observed_local_X_projected_yaw_only",
            "rest_coordinates": "SMPL_Y_up_X_heading_axis",
            "axis_conversion": "Rx_positive_90_degrees_Y_up_to_Z_up_preserve_local_X",
            "output_coordinates": "floor_world_Z_up_X_heading_at_zero_yaw",
            "causality": "pointwise_current_head_only",
            "uses_gt_body": False,
            "uses_e7": False,
            "motion_checkpoint": None,
            "fine_hand_reconstruction": False,
        }

    @torch.no_grad()
    def __call__(self, head):
        """Return ``(BodyState[T,...], beta[10])`` for visible ``head[T,9]``.

        Head positions must already be in the chosen floor-world coordinates.
        Pitch and roll never tilt the whole neutral body. Undefined projected
        headings are rejected by ``planar_reference`` rather than filled from
        a future frame or a body label.
        """
        if (
            not isinstance(head, torch.Tensor)
            or head.ndim != 2
            or head.shape[1] != 9
            or len(head) < 1
            or not head.is_floating_point()
            or not bool(torch.isfinite(head).all())
        ):
            raise ValueError("Bootstrap requires finite floating-point head observations with shape [T>=1,9].")
        reference = planar_reference(transform_from_9d(head))
        # Upright Y-up -> Z-up, preserving the local +X axis used by the
        # observation/codec heading convention. No additional yaw is applied.
        basis = head.new_tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
        rotation = reference[:, :3, :3] @ basis
        rest = self.rest_joints.to(device=head.device, dtype=head.dtype)
        rotated = torch.einsum("tij,kj->tki", rotation, rest)
        translation = head[:, 6:9] - rotated[:, 15]
        positions = rotated + translation[:, None]
        joints = rigid_transform(rotation[:, None].expand(-1, 22, -1, -1), positions)
        return BodyState(joints, reference, head.new_zeros(len(head), 36)), head.new_zeros(10)
