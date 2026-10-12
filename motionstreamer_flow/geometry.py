"""Rotation convention matches PyTorch3D: the first two matrix ROWS form 6D."""

import torch
import torch.nn.functional as F

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(code):
    code = code.float()
    a, b = code[..., :3], code[..., 3:6]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    e1 = torch.where(a.norm(dim=-1, keepdim=True) > 1e-7, F.normalize(a, dim=-1), fallback)
    orth = b - (e1 * b).sum(-1, keepdim=True) * e1
    axis = F.one_hot(e1.abs().argmin(-1), 3).to(e1)
    alternative = axis - (axis * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(torch.where(orth.norm(dim=-1, keepdim=True) > 1e-7, orth, alternative), dim=-1)
    return torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), -2)


def axis_angle_to_matrix(v):
    v = v.float()
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)
    theta = v.norm(dim=-1, keepdim=True)[..., None]
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(skew)
    return eye + torch.sinc(theta / torch.pi) * skew + 0.5 * torch.sinc(theta / (2 * torch.pi)).square() * (skew @ skew)


def fk(pose, root, offsets):
    """SMPL22 skeleton FK, using offline shape-specific rest offsets; units are metres."""
    with torch.autocast(pose.device.type, enabled=False):
        return _fk(pose.float(), root.float(), offsets.float())


def _fk(pose, root, offsets):
    local = rotation_6d_to_matrix(pose[..., :126].reshape(*pose.shape[:-1], 21, 6))
    rotations = [rotation_6d_to_matrix(root[..., :6])]
    positions = [root[..., 6:9].float()]
    offsets = offsets.float()
    while offsets.ndim < positions[0].ndim + 1:
        offsets = offsets.unsqueeze(-3)
    for child, parent in enumerate(PARENTS[1:], start=1):
        rotations.append(rotations[parent] @ local[..., child - 1, :, :])
        positions.append(positions[parent] + (rotations[parent] @ offsets[..., child, :, None]).squeeze(-1))
    return torch.stack(positions, -2), torch.stack(rotations, -3)


def to_world(points, anchor):
    return points @ anchor[:3, :3].T + anchor[:3, 3]


def root_to_world(root, anchor):
    rotation = anchor[:3, :3] @ rotation_6d_to_matrix(root[..., :6])
    return torch.cat((matrix_to_rotation_6d(rotation), to_world(root[..., 6:9], anchor)), -1)


def head_features(transforms, anchor, available=None):
    """Fixed-anchor absolute pose + BACKWARD adjacent-frame relative motion."""
    available = (
        torch.ones(len(transforms), dtype=torch.bool, device=transforms.device) if available is None else available
    )
    safe = torch.where(available[:, None, None], transforms, torch.eye(4, device=transforms.device))
    relative = torch.linalg.inv(anchor) @ safe
    absolute = torch.cat((matrix_to_rotation_6d(relative[:, :3, :3]), relative[:, :3, 3]), -1)
    delta = torch.eye(4, device=transforms.device).expand(len(transforms), 4, 4).clone()
    delta[1:] = torch.linalg.inv(safe[:-1]) @ safe[1:]
    pair_valid = available.clone()
    pair_valid[1:] &= available[:-1]
    delta = torch.where(pair_valid[:, None, None], delta, torch.eye(4, device=transforms.device))
    result = torch.cat((absolute, matrix_to_rotation_6d(delta[:, :3, :3]), delta[:, :3, 3]), -1)
    return torch.where(available[:, None], result, torch.zeros_like(result))
