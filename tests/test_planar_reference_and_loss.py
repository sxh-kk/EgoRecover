import torch

from egorecover.codec import BodyState, MotionCodec, planar_reference, rigid_transform, transform_to_9d
from egorecover.history import HistoryBuffer
from egorecover.losses import physical_objective
from utils.rotation_conversions import axis_angle_to_matrix


def codec(mode="planar"):
    return MotionCodec(
        {
            "motion_mean": torch.zeros(243),
            "motion_std": torch.ones(243),
            "traj_mean": torch.zeros(18),
            "traj_std": torch.ones(18),
        },
        reference_mode=mode,
    )


def rotation(angle):
    return axis_angle_to_matrix(torch.tensor(angle, dtype=torch.float32))


def prediction(c):
    local = rigid_transform(rotation([0.4, 0.2, 0.1]).expand(22, 3, 3), torch.tensor([0.2, 0.3, 1.4]).expand(22, 3))
    delta = rigid_transform(rotation([0.15, 0.05, 0.02]), torch.tensor([0.04, 0.02, 0.12]))
    raw = torch.cat((transform_to_9d(local).flatten(), transform_to_9d(delta), torch.arange(36) / 10))
    return c.normalize(raw), local, delta


def test_bootstrap_and_long_commit_keep_reference_planar_but_body_three_dimensional():
    c = codec()
    pred, local, _ = prediction(c)
    initial = rigid_transform(rotation([0.3, 0.2, 0.1]), torch.tensor([1.0, 2.0, 3.0]))
    buffer = HistoryBuffer.from_bootstrap(c, pred.repeat(20, 1), initial)
    for frame in range(20, 220):
        state = buffer.commit(pred, frame)
        torch.testing.assert_close(state.reference[2], torch.tensor([0.0, 0.0, 1.0, 0.0]), atol=0, rtol=0)
        torch.testing.assert_close(state.reference[:2, 2], torch.zeros(2), atol=0, rtol=0)
        torch.testing.assert_close(state.joints, state.reference[None] @ local, atol=3e-6, rtol=1e-6)
        assert abs(float(state.joints[0, 2, 0])) > 0.05  # body pitch is preserved
    assert torch.equal(buffer.initial_reference, planar_reference(initial))
    assert torch.isfinite(buffer.encoded()).all()


def test_planar_reanchoring_keeps_world_body_and_legacy_preserves_old_se3():
    c = codec()
    pred, local, delta = prediction(c)
    prior = rigid_transform(rotation([0.2, 0.1, -0.4]), torch.tensor([2.0, 3.0, 4.0]))
    legacy = codec("legacy_se3")
    decoded_old = legacy.decode_current(pred, prior)
    torch.testing.assert_close(decoded_old.reference, prior @ delta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded_old.joints, (prior @ delta)[None] @ local, atol=2e-6, rtol=1e-6)
    encoded = c.encode_current(decoded_old, prior)
    roundtrip = c.decode_current(encoded, prior)
    torch.testing.assert_close(roundtrip.joints, decoded_old.joints, atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(roundtrip.reference, planar_reference(decoded_old.reference), atol=2e-6, rtol=1e-6)
    projected = c.project_motion_reference(pred)
    assert torch.equal(projected[:198], pred[:198]) and torch.equal(projected[207:], pred[207:])


def test_physical_loss_has_finite_gradients_and_penalizes_world_translation():
    c = codec()
    state = BodyState(torch.eye(4).expand(1, 22, 4, 4).clone(), torch.eye(4)[None], torch.zeros(1, 36))
    target = c.encode_current(state, state.reference)[:, None]
    batch = {"target": target, "previous_reference": state.reference, "target_joints": state.joints[..., :3, 3]}
    pred = target.clone()
    pred[..., 204] += 0.1
    pred.requires_grad_(True)
    rep_only = physical_objective(c, pred, batch, geometry_weight=0)
    total = physical_objective(c, pred, batch, geometry_weight=1)
    assert float((total - rep_only).detach()) > 0.99
    total.backward()
    assert torch.isfinite(pred.grad).all() and float(pred.grad[0, 0, 204]) > 0
    assert float(physical_objective(c, target, batch)) == 0
