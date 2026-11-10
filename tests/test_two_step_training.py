from dataclasses import replace

import pytest
import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.losses import (
    emitted_velocity_loss,
    forward_rng,
    frozen_ema,
    paired_feedback,
    training_objective,
    two_step_objective,
)
from egorecover.models import HistoryPrior, MotionModel
from egorecover.reference_diagnostic import intervention_state
from egorecover.state import state_map


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(62)
    episode = synthetic_episode()
    config = WindowConfig(current_weight=0.03)
    prior = HistoryPrior(width=24, layers=1, heads=4, dropout=0)
    system = WindowReconstructor(MotionCodec(episode.stats), prior, config)
    model = MotionModel(config, replace(ModelConfig.compact(), dropout=0.1)).train()
    return episode, system, model, frozen_ema(model)


def arguments(episode):
    return dict(
        next_observations=episode.observations(21),
        next_truth_window=episode.target(21, 3),
        next_epsilon=torch.randn(1, 3, 243),
        forward_seed=123,
    )


def test_future_loss_reaches_first_call_through_cache_and_detach_control_removes_only_that_path():
    episode, system, model, teacher = fixture()
    epsilon, tau = torch.randn(1, 3, 243), torch.tensor([0.4])
    kwargs = arguments(episode)
    results = []
    for detach in (False, True):
        result = two_step_objective(
            system,
            model,
            teacher,
            episode.cache(),
            episode.observations(20),
            episode.target(20, 3),
            epsilon,
            tau,
            DenseJoints(),
            detach_between=detach,
            **kwargs,
        )
        result["result"].cache.validate()
        assert result["next_result"].cache.frame == 21
        assert result["loss"].item() == pytest.approx((result["first_loss"] + result["next_loss"]).item() / 2)
        gradient = torch.autograd.grad(
            result["next_loss"], result["result"].codes, retain_graph=True, allow_unused=True
        )[0]
        if detach:
            assert gradient is None
        else:
            assert gradient is not None and gradient.abs().sum() > 0
        results.append(result)
    torch.testing.assert_close(results[0]["result"].codes, results[1]["result"].codes, atol=0, rtol=0)
    torch.testing.assert_close(results[0]["next_result"].codes, results[1]["next_result"].codes, atol=0, rtol=0)
    results[0]["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in system.predictor.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())


def test_isolated_rng_preserves_first_flow_and_endpoint_across_different_objective_routes():
    episode, system, model, teacher = fixture()
    epsilon, tau = torch.randn(1, 3, 243), torch.tensor([0.4])
    kwargs = arguments(episode)
    before = torch.get_rng_state().clone()
    single = training_objective(
        system,
        model,
        teacher,
        episode.cache(),
        episode.observations(20),
        episode.target(20, 3),
        epsilon,
        tau,
        DenseJoints(),
        forward_seed=123,
    )
    two = two_step_objective(
        system,
        model,
        teacher,
        episode.cache(),
        episode.observations(20),
        episode.target(20, 3),
        epsilon,
        tau,
        DenseJoints(),
        **kwargs,
    )
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(single["result"].codes, two["result"].codes, atol=0, rtol=0)
    torch.testing.assert_close(single["loss"], two["first_loss"], atol=0, rtol=0)
    with forward_rng(epsilon, 123, "endpoint"):
        first = torch.rand(10)
    with forward_rng(epsilon, 123, "flow"):
        second = torch.rand(10)
    assert not torch.equal(first, second)


def test_body_only_feedback_matches_diagnostic_intervention_and_keeps_common_past_detached():
    episode, system, model, teacher = fixture()
    # A trained teacher has a nonzero center projection. Its zero initialization
    # intentionally hides the active body-history center from a fresh G.
    with torch.no_grad():
        teacher.mu_proj.weight.normal_(std=0.01)
    current = system.reconstruct(model, episode.cache(), episode.observations(20), torch.randn(1, 3, 243))
    truth = state_map(episode.target(20), lambda x: x[:, -1])
    noise = torch.randn(1, 1, 3, 243)
    loss = paired_feedback(
        system, teacher, current, truth, episode.observations(21), noise, DenseJoints(), intervention="body_pose"
    )
    a = system.reconstruct(teacher, current.cache.replace_current(current.current), episode.observations(21), noise[0])
    intervention = intervention_state(current.current, truth, DenseJoints(), "body_pose")
    b = system.reconstruct(teacher, current.cache.replace_current(intervention), episode.observations(21), noise[0])
    expected = (DenseJoints()(a.current) - DenseJoints()(b.current).detach()).square().sum(-1).mean() / 0.1**2
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    gradient = torch.autograd.grad(loss, current.codes)[0]
    assert gradient[:, -1].abs().sum() > 0
    assert gradient[:, :-1].count_nonzero() == 0


def test_two_step_rejects_unintended_feedback_combination():
    episode, system, model, teacher = fixture()
    system.config = replace(system.config, feedback="paired")
    with pytest.raises(ValueError, match="no feedback"):
        two_step_objective(
            system,
            model,
            teacher,
            episode.cache(),
            episode.observations(20),
            episode.target(20, 3),
            torch.randn(1, 3, 243),
            torch.tensor([0.4]),
            DenseJoints(),
            **arguments(episode),
        )


def test_velocity_preserves_real_motion_and_ignores_shared_position_offset_and_truth_gradients():
    truth = torch.zeros(2, 22, 3, requires_grad=True)
    future_truth = torch.ones(2, 22, 3, requires_grad=True) * 0.1
    prediction = (truth.detach() + 2).requires_grad_()
    future_prediction = (future_truth.detach() + 2).requires_grad_()
    loss = emitted_velocity_loss(prediction, future_prediction, truth, future_truth, torch.tensor([0.1, 0.2]))
    assert loss.item() < 1e-11
    # A further 0.1m x displacement is 1m/s and 0.5m/s error in the two examples.
    shifted = future_prediction + torch.tensor([0.1, 0.0, 0.0])
    loss = emitted_velocity_loss(prediction, shifted, truth, future_truth, torch.tensor([0.1, 0.2]))
    assert loss.item() == pytest.approx((1.0 + 0.25) / 2, rel=1e-5)
    gradients = torch.autograd.grad(loss, (prediction, future_prediction, truth, future_truth), allow_unused=True)
    assert gradients[0].abs().sum() > 0 and gradients[1].abs().sum() > 0
    assert gradients[2:] == (None, None)


@pytest.mark.parametrize("dt", (0.0, -0.1, float("nan"), torch.ones(2)))
def test_velocity_rejects_invalid_timebase(dt):
    value = torch.zeros(1, 22, 3)
    with pytest.raises(ValueError, match="time intervals"):
        emitted_velocity_loss(value, value, value, value, dt)


def test_two_step_velocity_uses_actual_emissions_and_leaves_both_forward_predictions_unchanged():
    episode, system, model, teacher = fixture()
    epsilon, tau = torch.randn(1, 3, 243), torch.tensor([0.4])
    kwargs = arguments(episode)
    common = (system, model, teacher, episode.cache(), episode.observations(20), episode.target(20, 3), epsilon, tau)
    baseline = two_step_objective(*common, DenseJoints(), **kwargs)
    result = two_step_objective(*common, DenseJoints(), velocity_weight=0.2, dt=0.1, **kwargs)
    for key in ("result", "next_result"):
        torch.testing.assert_close(result[key].codes, baseline[key].codes, atol=0, rtol=0)
    expected = emitted_velocity_loss(
        DenseJoints()(result["result"].current),
        DenseJoints()(result["next_result"].current),
        DenseJoints()(state_map(episode.target(20), lambda x: x[:, -1])),
        DenseJoints()(state_map(episode.target(21), lambda x: x[:, -1])),
        0.1,
    )
    torch.testing.assert_close(result["velocity"], expected, atol=0, rtol=0)
    torch.testing.assert_close(result["loss"], baseline["loss"] + 0.2 * expected, atol=0, rtol=0)
    gradients = torch.autograd.grad(result["velocity"], (result["result"].codes, result["next_result"].codes))
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert all(not p.requires_grad and p.grad is None for p in system.predictor.parameters())
