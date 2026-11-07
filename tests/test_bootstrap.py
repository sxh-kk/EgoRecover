import json
import math
from types import SimpleNamespace

import pytest
import torch

from egorecover.bootstrap import HeadAlignedBootstrap
from egorecover.codec import rigid_transform, transform_from_9d, transform_to_9d


def neutral_joints(dtype=torch.float64):
    rest = torch.arange(66, dtype=dtype).reshape(22, 3) / 100
    rest[0] = torch.tensor([0.0, 0.9, 0.0], dtype=dtype)
    rest[15] = torch.tensor([0.0, 1.6, 0.0], dtype=dtype)
    rest[7] = torch.tensor([0.1, 0.0, 0.0], dtype=dtype)
    return rest


def head_observations(length=4, dtype=torch.float64):
    angles = torch.linspace(-0.8, 0.7, length, dtype=dtype)
    c, s, zero, one = angles.cos(), angles.sin(), torch.zeros_like(angles), torch.ones_like(angles)
    rotations = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), dim=-1).reshape(length, 3, 3)
    positions = torch.stack((angles, -angles, one * 1.7), dim=-1)
    return transform_to_9d(rigid_transform(rotations, positions))


def test_bootstrap_is_pointwise_causal_and_has_no_input_or_template_alias():
    rest = neutral_joints().requires_grad_(True)
    bootstrap = HeadAlignedBootstrap(rest)
    head = head_observations().requires_grad_(True)
    body, beta = bootstrap(head)
    prefix, _ = bootstrap(head[:2])
    changed = head.detach().clone()
    changed[2:, 6:9] += 100
    future_changed, _ = bootstrap(changed)
    for name in ("joints", "reference", "auxiliary"):
        # A shorter batched contraction may round at a different final bit.
        torch.testing.assert_close(getattr(body, name)[:2], getattr(prefix, name), rtol=0, atol=1e-14)
        torch.testing.assert_close(getattr(body, name)[:2], getattr(future_changed, name)[:2], rtol=0, atol=0)
        assert not getattr(body, name).requires_grad
    assert not bootstrap.rest_joints.requires_grad and not beta.requires_grad
    saved = body.joints.clone()
    with torch.no_grad():
        rest.add_(1000)
        head.add_(1000)
    torch.testing.assert_close(body.joints, saved, rtol=0, atol=0)
    assert not torch.equal(bootstrap.rest_joints, rest)


def test_neutral_skeleton_is_rigid_head_aligned_and_uses_fixed_zero_beta():
    rest, head = neutral_joints(), head_observations()
    body, beta = HeadAlignedBootstrap(rest)(head)
    torch.testing.assert_close(body.joints[:, 15, :3, 3], head[:, 6:9])
    distances = torch.cdist(body.joints[..., :3, 3], body.joints[..., :3, 3])
    torch.testing.assert_close(distances, torch.cdist(rest, rest).expand(len(head), -1, -1))
    rotations = body.joints[..., :3, :3]
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, torch.eye(3).double().expand_as(rotations))
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(len(head), 22).double())
    assert beta.shape == (10,) and torch.count_nonzero(beta) == 0
    assert body.auxiliary.shape == (len(head), 36) and torch.count_nonzero(body.auxiliary) == 0


def test_smpl_y_up_becomes_world_z_up_without_head_pitch_or_roll():
    dtype = torch.float64
    c, s = math.cos(0.4), math.sin(0.4)
    pitch = torch.tensor(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=dtype)
    roll = torch.tensor(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=dtype)
    position = torch.tensor([[2.0, 3.0, 1.8]], dtype=dtype)
    tilted = transform_to_9d(rigid_transform((pitch @ roll)[None], position))
    upright = transform_to_9d(rigid_transform(torch.eye(3, dtype=dtype)[None], position))
    bootstrap = HeadAlignedBootstrap(neutral_joints(dtype))
    tilted_body, _ = bootstrap(tilted)
    upright_body, _ = bootstrap(upright)
    torch.testing.assert_close(tilted_body.joints, upright_body.joints)
    basis = torch.tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=dtype)
    torch.testing.assert_close(upright_body.joints[0, 0, :3, :3], basis)
    assert upright_body.joints[0, 0, 2, 3] < upright_body.joints[0, 15, 2, 3]
    torch.testing.assert_close(upright_body.reference[0, :3, 3], torch.tensor([2.0, 3.0, 0.0], dtype=dtype))


def test_upright_axis_conversion_preserves_observed_heading_without_extra_yaw():
    head = head_observations()
    transform = transform_from_9d(head)
    c, s = math.cos(0.4), math.sin(0.4)
    pitch = head.new_tensor(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    transform[:, :3, :3] = transform[:, :3, :3] @ pitch
    head = transform_to_9d(transform)
    body, _ = HeadAlignedBootstrap(neutral_joints())(head)
    observed_heading = transform[:, :2, 0]
    observed_heading = observed_heading / observed_heading.norm(dim=-1, keepdim=True)
    body_heading = body.joints[..., :2, 0]
    torch.testing.assert_close(body_heading, observed_heading[:, None].expand(-1, 22, -1))
    torch.testing.assert_close(body.joints[..., 2, 1], torch.ones(len(head), 22, dtype=head.dtype))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_outputs_follow_head_dtype_and_device(dtype):
    head = head_observations(length=1, dtype=dtype)
    body, beta = HeadAlignedBootstrap(neutral_joints())(head)
    for value in (body.joints, body.reference, body.auxiliary, beta):
        assert value.dtype == head.dtype and value.device == head.device


def test_from_smpl_uses_only_zero_beta_and_detaches_neutral_template():
    class NeutralLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("v_template", neutral_joints())
            self.calls = []

        def forward(self, *, betas, return_verts):
            self.calls.append((betas.clone(), return_verts))
            return SimpleNamespace(joints=self.v_template[None])

    smpl = NeutralLayer()
    bootstrap = HeadAlignedBootstrap.from_smpl(smpl)
    assert len(smpl.calls) == 1
    assert torch.equal(smpl.calls[0][0], torch.zeros(1, 10, dtype=smpl.v_template.dtype))
    assert smpl.calls[0][1] is False
    assert smpl.training  # Reading the neutral template does not mutate the caller's mode.
    protocol = json.loads(json.dumps(bootstrap.protocol))
    assert protocol["head_anchor"] == 15 and protocol["shape_policy"] == "neutral_beta_zero"
    assert protocol["uses_gt_body"] is False and protocol["uses_e7"] is False
    assert protocol["causality"] == "pointwise_current_head_only"


@pytest.mark.parametrize("head", (torch.zeros(0, 9), torch.zeros(2, 8), torch.ones(2, 9, dtype=torch.long)))
def test_invalid_head_shapes_and_types_are_rejected(head):
    with pytest.raises(ValueError, match="head observations"):
        HeadAlignedBootstrap(neutral_joints())(head)


def test_nonfinite_or_invalid_neutral_and_head_are_rejected():
    with pytest.raises(ValueError, match="neutral SMPL"):
        HeadAlignedBootstrap(torch.zeros(23, 3))
    rest = neutral_joints()
    rest[0, 0] = float("nan")
    with pytest.raises(ValueError, match="neutral SMPL"):
        HeadAlignedBootstrap(rest)
    head = head_observations()
    head[0, 8] = float("inf")
    with pytest.raises(ValueError, match="head observations"):
        HeadAlignedBootstrap(neutral_joints())(head)
