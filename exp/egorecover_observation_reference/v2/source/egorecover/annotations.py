"""OFFLINE supervision -> dense physical states. Never imported by rollout."""

import torch

from egorecover.codec import BodyState, planar_reference, rigid_transform, transform_from_9d
from utils.rotation_conversions import rotation_6d_to_matrix

# Standard first 22 SMPL/SMPL-X body joints, in the repository's SMPL_JOINTS order.
BODY_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def body_states_from_supervision(supervision, clean_head, *, floor_height, contact_floor_height=None):
    """Use recorded dense positions and rotations; no SMPL mesh/FK required.

    clean_head and labels are OFFLINE target construction only. The caller
    supplies floor_height from its declared calibration/startup estimate; this
    function deliberately does not read supervision['floor_height']. An
    independent OFFLINE contact_floor_height may be supplied for contact
    labels; it never changes the body/reference coordinates or online input.
    Original hand PCA coefficients are preserved directly (no PCA roundtrip).
    """
    device, dtype = clean_head.device, clean_head.dtype

    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=dtype).clone()

    parameters = {name: tensor(value) for name, value in supervision["smpl_params"].items()}
    local_codes = torch.cat((parameters["global_orient"][:, None], parameters["body_pose"]), dim=1)
    local_rotations = rotation_6d_to_matrix(local_codes)
    rotations = []
    for joint, parent in enumerate(BODY_PARENTS):
        rotations.append(local_rotations[:, joint] if parent < 0 else rotations[parent] @ local_rotations[:, joint])
    rotation = torch.stack(rotations, dim=1)
    positions = tensor(supervision["kp3d"])[:, :22]
    positions[..., 2] -= floor_height
    head = clean_head.clone()
    head[..., 8] -= floor_height
    reference = planar_reference(transform_from_9d(head))
    # Contacts match the v4 representation's 0.02 m/frame and 0.10 m rules.
    feet = positions[:, [10, 11]]
    if len(feet) > 1:
        speed = (feet[1:] - feet[:-1]).norm(dim=-1)
        speed = torch.cat((speed[:1], speed), dim=0)
    else:
        speed = feet.new_zeros(1, 2)
    contact_floor = floor_height if contact_floor_height is None else contact_floor_height
    contact_height = feet[..., 2] + floor_height - contact_floor
    contacts = ((speed < 0.02) & (contact_height < 0.10)).to(dtype)
    beta = parameters["betas"].expand(len(head), 10)
    auxiliary = torch.cat((parameters["left_hand_pose"], parameters["right_hand_pose"], contacts, beta), dim=-1)
    if auxiliary.shape != (len(head), 36):
        raise ValueError("Expected original 12D PCA per hand, two contacts, and ten betas.")
    return BodyState(rigid_transform(rotation, positions), reference, auxiliary)
