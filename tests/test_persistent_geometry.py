from types import SimpleNamespace

import pytest
import torch

from egorecover.codec import BodyState
from egorecover.fk import FixedShapeFK
from egorecover.losses import emitted_velocity_loss, persistent_error_loss
from egorecover.official import OfficialWindowData


def test_constant_position_bias_is_visible_to_pooled_loss_even_when_velocity_is_exact():
    truth = torch.randn(2, 8, 22, 3)
    offset = torch.tensor([0.1, -0.2, 0.3])
    prediction = (truth + offset).requires_grad_()
    velocity = emitted_velocity_loss(
        prediction[:, :-1].reshape(-1, 22, 3),
        prediction[:, 1:].reshape(-1, 22, 3),
        truth[:, :-1].reshape(-1, 22, 3),
        truth[:, 1:].reshape(-1, 22, 3),
        torch.full((14,), 0.1),
    )
    assert float(velocity.detach()) < 1e-10
    loss = persistent_error_loss(prediction, truth)
    torch.testing.assert_close(loss, offset.square().sum() / 0.1**2)
    loss.backward()
    assert (prediction.grad.abs().sum((0, 2, 3)) > 0).all()


def test_exact_fast_motion_has_zero_persistent_error_and_labels_never_receive_gradients():
    truth = torch.randn(1, 8, 22, 3, requires_grad=True)
    prediction = truth.detach().clone().requires_grad_()
    loss = persistent_error_loss(prediction, truth)
    assert float(loss.detach()) == 0
    loss.backward()
    assert truth.grad is None
    with pytest.raises(ValueError, match="scale"):
        persistent_error_loss(prediction, truth, scales=(16,))


def test_predicted_shape_changes_limbs_preserves_pelvis_and_receives_geometry_gradients():
    class ShapeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("parents", torch.tensor([-1] + [0] * 21))
            self.register_buffer("left_hand_components", torch.eye(45)[:12])
            self.register_buffer("right_hand_components", torch.eye(45)[:12])

        def forward(self, *, betas, transl=None, **kwargs):
            root = betas[:, :3] + (0 if transl is None else transl)
            positions = root[:, None].expand(-1, 22, -1).clone()
            positions[:, 1, 0] = positions[:, 1, 0] + 0.5 + betas[:, 3] * 0.1
            return SimpleNamespace(joints=positions)

    auxiliary = torch.zeros(2, 36)
    auxiliary[1, -7] = 2
    auxiliary.requires_grad_()
    transforms = torch.eye(4).repeat(2, 22, 1, 1)
    transforms[:, 0, :3, 3] = torch.tensor([2.0, 3.0, 4.0])
    state = BodyState(transforms, torch.eye(4).repeat(2, 1, 1), auxiliary)
    decoder = FixedShapeFK(ShapeLayer(), torch.zeros(10), shape_mode="predicted")
    result = decoder(state)
    torch.testing.assert_close(result[:, 0], transforms[:, 0, :3, 3])
    assert float((result[1, 1, 0] - result[0, 1, 0]).detach()) == pytest.approx(0.2)
    result[:, 1].square().mean().backward()
    assert auxiliary.grad[:, -7].abs().sum() > 0
    assert torch.isfinite(auxiliary.grad).all()


def test_eight_frame_queries_fit_source_and_keep_the_complete_future_supervision():
    data = OfficialWindowData.__new__(OfficialWindowData)
    data.split = "train"
    data.sampling_inventory, data._sampling_groups = {}, {}
    data.source_metadata = {"exact": {"take": "only", "frames": 32}}
    data.inventory = {"source_takes": 1}
    queries, frame = data.sample_queries(torch.Generator().manual_seed(62), 4, 4, horizon=8)
    assert frame == 24
    assert queries == [("exact", 0, 32)] * 4
    assert frame + 7 == 31


def test_official_geometry_target_uses_stored_positions_instead_of_decoding_labels():
    from dataclasses import replace

    from egorecover import ModelConfig, WindowConfig, WindowReconstructor
    from egorecover.codec import MotionCodec
    from egorecover.diagnostics import synthetic_episode
    from egorecover.losses import frozen_ema, training_objective
    from egorecover.models import MotionModel

    torch.set_num_threads(2)
    episode = synthetic_episode()
    cfg = WindowConfig(current_weight=0.03, geometry_target="official")
    system = WindowReconstructor(MotionCodec(episode.stats), None, cfg)
    model = MotionModel(cfg, replace(ModelConfig.compact(), dropout=0))

    def deliberately_different_decoder(state):
        return state.joints[..., :3, 3] + 1

    result = training_objective(
        system,
        model,
        frozen_ema(model),
        episode.cache(),
        episode.observations(20),
        episode.target(20, 3),
        torch.randn(1, 3, 243),
        torch.tensor([0.4]),
        deliberately_different_decoder,
    )
    torch.testing.assert_close(result["truth_joints"], episode.truth.joints[20:21, :, :3, 3])


def test_all_eight_outputs_receive_persistent_error_gradients_with_detached_history(monkeypatch):
    from dataclasses import replace

    import egorecover.losses as losses
    from egorecover import ModelConfig, WindowConfig, WindowReconstructor
    from egorecover.codec import MotionCodec
    from egorecover.diagnostics import DenseJoints, synthetic_episode
    from egorecover.models import MotionModel

    torch.set_num_threads(2)
    episode = synthetic_episode()
    cfg = WindowConfig(current_weight=0.03, geometry_target="official")
    system = WindowReconstructor(MotionCodec(episode.stats), None, cfg)
    model = MotionModel(cfg, replace(ModelConfig.compact(), dropout=0))
    original, rows = losses.training_objective, []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        rows.append(result)
        return result

    monkeypatch.setattr(losses, "training_objective", record)
    result = losses.multi_step_objective(
        system,
        model,
        losses.frozen_ema(model),
        episode.cache(),
        [episode.observations(t) for t in range(20, 28)],
        [episode.target(t, 3) for t in range(20, 28)],
        torch.randn(8, 1, 3, 243),
        torch.tensor([0.4]),
        DenseJoints(),
        dt=torch.full((1, 7), 0.1),
        slow_weight=0.03,
    )
    gradients = torch.autograd.grad(result["slow"], [row["current_joints"] for row in rows], retain_graph=True)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)
    cross = torch.autograd.grad(rows[-1]["loss"], rows[0]["result"].codes, allow_unused=True, retain_graph=True)[0]
    assert cross is None
    result["loss"].backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
