import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.diagnostics import synthetic_episode
from egorecover.state import state_map
from scripts.diagnose_p01_gauge import losses, reencode_reference


def yaw(angle):
    angle = torch.as_tensor(angle)
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o)).reshape(3, 3)


def test_common_reference_preserves_physical_body_and_changes_code_loss():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    truth = state_map(episode.target(20), lambda x: x[:, -1])
    previous = episode.cache().body.reference[:, -1]
    reference = truth.reference.clone()
    reference[:, :3, :3] = yaw(0.7)
    reference[:, :2, 3] += 2
    original = codec.encode_current(truth, previous)
    changed = reencode_reference(codec, truth, reference, previous)
    decoded = codec.decode_current(changed, previous)
    torch.testing.assert_close(decoded.joints, truth.joints, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(decoded.auxiliary, truth.auxiliary, rtol=0, atol=0)
    assert not torch.allclose(original, changed)
    assert losses(changed, original)["code_loss"].item() > 0
    assert losses(changed, changed)["code_loss"].item() == 0


def test_sixd_scaling_can_reduce_code_loss_without_changing_physical_rotation():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    truth = state_map(episode.target(20), lambda x: x[:, -1])
    previous = episode.cache().body.reference[:, -1]
    target = codec.encode_current(truth, previous)
    joints = truth.joints.clone()
    joints[:, :, :3, :3] = yaw(torch.pi / 3)
    prediction = codec.encode_current(BodyState(joints, truth.reference, truth.auxiliary), previous)
    raw = codec.denormalize(prediction)
    raw[:, :198].reshape(-1, 22, 9)[..., :6] *= 0.5
    scaled = codec.normalize(raw)
    assert losses(scaled, target)["body6d_MSE"].item() < losses(prediction, target)["body6d_MSE"].item()
    first, second = [codec.decode_current(x, previous) for x in (prediction, scaled)]
    torch.testing.assert_close(first.joints, second.joints, atol=1e-6, rtol=1e-6)
