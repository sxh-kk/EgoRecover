import math

import pytest
import torch

from egorecover.codec import (
    BodyState,
    MotionCodec,
    planar_reference,
    rigid_transform,
    transform_to_9d,
    transform_from_9d,
)


@pytest.fixture
def codec():
    return MotionCodec(
        {
            "motion_mean": torch.zeros(243),
            "motion_std": torch.ones(243),
            "traj_mean": torch.zeros(18),
            "traj_std": torch.ones(18),
        }
    )


def transforms(x, yaw=0.0):
    angle = torch.full_like(x, yaw)
    c, s, z, o = angle.cos(), angle.sin(), torch.zeros_like(x), torch.ones_like(x)
    rotation = torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(*x.shape, 3, 3)
    return rigid_transform(rotation, torch.stack((x, z, z), dim=-1))


def state_at(reference):
    shape = reference.shape[:-2]
    local = transforms(torch.linspace(-0.5, 0.5, 22).expand(*shape, 22))
    return BodyState(reference[..., None, :, :] @ local, reference, torch.randn(*shape, 36))


def test_dense_roundtrip_and_reanchoring_preserve_physical_states(codec):
    states = state_at(transforms(torch.tensor([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]), yaw=0.4))
    outer = transforms(torch.tensor([-5.0, 7.0]), yaw=-0.8)
    encoded = codec.encode_history(states, outer)
    previous = outer
    for frame in range(3):
        decoded = codec.decode_current(encoded[:, frame], previous)
        torch.testing.assert_close(decoded.joints, states.joints[:, frame], atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(decoded.auxiliary, states.auxiliary[:, frame])
        previous = decoded.reference


def test_target_delta_corrects_predicted_history_offset(codec):
    previous_prediction = transforms(torch.tensor([1.3]))
    target = state_at(transforms(torch.tensor([1.1])))
    encoded = codec.encode_current(target, previous_prediction)
    raw = codec.denormalize(encoded)
    assert raw[0, 204].item() == pytest.approx(-0.2, abs=1e-6)
    decoded = codec.decode_current(encoded, previous_prediction)
    torch.testing.assert_close(decoded.joints, target.joints)


def test_physical_continuation_has_identity_delta(codec):
    last = state_at(transforms(torch.tensor([3.7]), yaw=1.2))
    mu = codec.continuation_prior(last)
    torch.testing.assert_close(
        codec.denormalize(mu)[..., 198:207], torch.tensor([[1.0, 0, 0, 0, 1, 0, 0, 0, 0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(codec.decode_current(mu, last.reference).joints, last.joints)


def test_180_degree_heading_is_finite_and_correct():
    reference = transforms(torch.tensor([2.0]), yaw=math.pi)
    actual = planar_reference(reference)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=0)


def test_unavailable_sensor_cannot_affect_condition(codec):
    previous = transforms(torch.tensor([1.0, 2.0]))
    available = torch.tensor([True, False])
    raw = transform_to_9d(transforms(torch.tensor([1.2, 20.0])))
    before = codec.encode_observation(raw, previous, available)
    raw[1].fill_(float("nan"))
    after = codec.encode_observation(raw, previous, available)
    assert torch.equal(before, after)
    torch.testing.assert_close(codec.denormalize(after[:1], "traj")[:, 15], torch.tensor([0.2]))


def test_degenerate_rotation_is_rejected(codec):
    bad = codec.normalize(torch.zeros(1, 243))
    with pytest.raises(ValueError, match="Degenerate"):
        codec.decode_current(bad, transforms(torch.zeros(1)))


def test_large_finite_rotation_code_is_not_confused_with_degenerate_axes():
    code = torch.tensor([[1e20, 0, 0, 0, 1e20, 0, 1.0, 2.0, 3.0]])
    transform = transform_from_9d(code)
    torch.testing.assert_close(transform[0, :3, :3], torch.eye(3), atol=0, rtol=0)
    torch.testing.assert_close(transform[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]), atol=0, rtol=0)


def test_constant_velocity_extrapolates_so3_and_translation(codec):
    previous = state_at(transforms(torch.tensor([1.0]), yaw=0.2))
    last = state_at(transforms(torch.tensor([1.1]), yaw=0.4))
    encoded = codec.constant_velocity_prior(previous, last)
    decoded = codec.decode_current(encoded, last.reference)
    torch.testing.assert_close(decoded.reference, transforms(torch.tensor([1.2]), yaw=0.6), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded.joints[..., :3, 3], 2 * last.joints[..., :3, 3] - previous.joints[..., :3, 3])


def test_offline_contact_calibration_does_not_change_body_coordinates():
    from egorecover.annotations import body_states_from_supervision

    identity6 = torch.tensor([1.0, 0, 0, 0, 1, 0])
    labels = {
        "smpl_params": {
            "global_orient": identity6.repeat(3, 1),
            "body_pose": identity6.repeat(3, 21, 1),
            "left_hand_pose": torch.zeros(3, 12),
            "right_hand_pose": torch.zeros(3, 12),
            "betas": torch.zeros(1, 10),
        },
        "kp3d": torch.zeros(3, 76, 3),
    }
    labels["kp3d"][..., 2] = 0.05
    head = torch.cat((identity6.repeat(3, 1), torch.tensor([[0.0, 0, 1.6]]).repeat(3, 1)), dim=-1)
    first = body_states_from_supervision(labels, head, floor_height=-0.4)
    corrected = body_states_from_supervision(labels, head, floor_height=-0.4, contact_floor_height=0.0)
    assert torch.equal(first.joints, corrected.joints) and torch.equal(first.reference, corrected.reference)
    assert torch.equal(first.auxiliary[:, :24], corrected.auxiliary[:, :24])
    assert not first.auxiliary[:, 24:26].any() and corrected.auxiliary[:, 24:26].all()
