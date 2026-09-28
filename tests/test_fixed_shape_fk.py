from types import SimpleNamespace

import torch

from egorecover.codec import BodyState
from egorecover.codec import MotionCodec
from egorecover.fk import FixedShapeFK
from egorecover.losses import fk_position_errors


def test_fk_uses_bootstrap_shape_and_predicted_pelvis_only():
    class RecordingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("parents", torch.tensor([-1] + [0] * 21))
            self.register_buffer("left_hand_components", torch.eye(45)[:12])
            self.register_buffer("right_hand_components", torch.eye(45)[:12])
            self.last = None

        def forward(self, *, betas, transl=None, **kwargs):
            self.last = {"betas": betas.clone(), "transl": transl, **kwargs}
            root = betas[:, :3]
            if transl is not None:
                root = root + transl
            return SimpleNamespace(joints=root[:, None].expand(-1, 22, -1))

    smpl = RecordingLayer()
    bootstrap = torch.arange(10, dtype=torch.float32) * 0.1
    fk = FixedShapeFK(smpl, bootstrap)
    physical = torch.eye(4).repeat(2, 22, 1, 1)
    physical[:, 0, :3, 3] = torch.tensor([[3.0, 4.0, 5.0], [8.0, 9.0, 10.0]])
    physical[:, 0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    state = BodyState(physical, torch.eye(4).repeat(2, 1, 1), torch.zeros(2, 36))
    first = fk(state)
    torch.testing.assert_close(first[:, 0], physical[:, 0, :3, 3])
    assert torch.equal(smpl.last["betas"], bootstrap.expand(2, -1))
    state.auxiliary[:, -10:] = 12345  # Predicted per-frame beta never replaces beta_boot.
    assert torch.equal(first, fk(state))
    assert smpl.last["body_pose"].shape == (2, 21, 3, 3)
    torch.testing.assert_close(smpl.last["global_orient"], physical[:, 0, :3, :3])
    torch.testing.assert_close(
        smpl.last["body_pose"], physical[:, 0, :3, :3].transpose(-1, -2)[:, None].expand(-1, 21, -1, -1)
    )
    assert smpl.last["left_hand_pose"].shape == (2, 15, 3, 3)
    fk.train()
    assert not smpl.training


def test_batched_model_bootstrap_fk_has_pose_gradient_and_isolates_shape():
    class PoseLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("parents", torch.tensor([-1] + [0] * 21))
            self.register_buffer("left_hand_components", torch.eye(45)[:12])
            self.register_buffer("right_hand_components", torch.eye(45)[:12])
            self.last_betas = None

        def forward(self, *, betas, transl=None, global_orient=None, return_verts=False, **kwargs):
            self.last_betas = betas.detach().clone()
            root = betas[:, :3] + (torch.zeros_like(betas[:, :3]) if transl is None else transl)
            joints = root[:, None].expand(-1, 22, -1).clone()
            if global_orient is not None:
                limb = global_orient @ torch.tensor([0.5, 0.0, 0.0])
                joints[:, 1] = root + limb + betas[:, 3:4].expand(-1, 3) * 0.01
            return SimpleNamespace(joints=joints)

    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
    }
    codec = MotionCodec(stats)
    joints = torch.eye(4).expand(2, 22, 4, 4).clone()
    references = torch.eye(4).expand(2, 4, 4).clone()
    prediction = codec.encode_current(BodyState(joints, references, torch.zeros(2, 36)), references)
    prediction = prediction[:, None].detach().requires_grad_(True)
    beta = torch.zeros(2, 10)
    beta[1, 3] = 2.0
    target = torch.zeros(2, 22, 3)
    target[:, 1, 1] = 0.5
    batch = {
        "previous_reference": references,
        "target_joints": target,
        "beta_boot": beta,
        "beta_boot_is_model": torch.ones(2, dtype=torch.bool),
    }
    smpl = PoseLayer()
    error = fk_position_errors(codec, prediction, batch, smpl)
    assert torch.isfinite(error).all()
    torch.testing.assert_close(smpl.last_betas, beta)
    assert not torch.equal(error[0, 1], error[1, 1])
    error.square().mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 0, :6].abs().sum() > 0
    channel = int(prediction.grad[0, 0, :6].abs().argmax())
    plus, minus = prediction.detach().clone(), prediction.detach().clone()
    plus[0, 0, channel] += 1e-3
    minus[0, 0, channel] -= 1e-3
    numerical = (
        fk_position_errors(codec, plus, batch, smpl).square().mean()
        - fk_position_errors(codec, minus, batch, smpl).square().mean()
    ) / 2e-3
    torch.testing.assert_close(numerical, prediction.grad[0, 0, channel], rtol=0.03, atol=1e-4)
    batch["beta_boot_is_model"][0] = False
    try:
        fk_position_errors(codec, prediction, batch, smpl)
    except ValueError as exc:
        assert "model-generated bootstrap" in str(exc)
    else:
        raise AssertionError("GT-start bootstrap shape must not pass formal FK loss")
