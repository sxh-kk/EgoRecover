"""Explicit asset-free integration fixtures, never formal accuracy evaluation."""

import torch

from .codec import BodyState, transform_to_9d
from .data import Episode
from .timing import validate_timing


class DenseJoints:
    diagnostic_only = True

    def __call__(self, body):
        return body.joints[..., :3, 3]


def synthetic_episode(*, device="cpu", length=36):
    """Explicit integration fixture, never a substitute for measured accuracy."""
    if length < 32:
        raise ValueError("Synthetic episode must support the ten-step diagnostic.")
    t = torch.arange(length, device=device).float()
    reference = torch.eye(4, device=device).repeat(length, 1, 1)
    reference[:, 0, 3] = t * 0.02
    joints = reference[:, None].repeat(1, 22, 1, 1)
    joints[:, :, 1, 3] += torch.linspace(-0.3, 0.3, 22, device=device)
    joints[:, :, 2, 3] += torch.linspace(0.2, 1.7, 22, device=device)
    auxiliary = torch.zeros(length, 36, device=device)
    truth = BodyState(joints, reference, auxiliary)
    teacher_joints = joints.clone()
    teacher_joints[..., 0, 3] += 0.01 * torch.sin(t)[:, None]
    teacher = BodyState(teacher_joints, reference.clone(), auxiliary.clone())
    head_transform = reference.clone()
    head_transform[:, 2, 3] = 1.7
    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
    }
    identity = torch.tensor([1, 0, 0, 0, 1, 0.0])
    for joint in range(23):
        stats["motion_mean"][joint * 9 : joint * 9 + 6] = identity
    episode = Episode(
        "synthetic",
        "synthetic",
        truth,
        teacher,
        torch.zeros(10, device=device),
        transform_to_9d(head_transform),
        torch.zeros(length, 1024, device=device),
        torch.ones(length, dtype=torch.bool, device=device),
        torch.ones(length, dtype=torch.bool, device=device),
        torch.arange(length).double() / 10,
        stats,
        torch.eye(4, device=device),
        synthetic=True,
    )
    validate_timing(episode.timestamps, 10, length)
    return episode
