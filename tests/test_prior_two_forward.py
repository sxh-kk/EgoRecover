import copy
from pathlib import Path

import pytest
import torch

from egorecover.codec import MotionCodec
from egorecover.data import TrainingBatch
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.models import HistoryPrior
from egorecover.prior_data import clean_batch
from egorecover.prior_training import PriorTrainOptions, prior_losses, prior_phase
from egorecover.prior_two_forward import replacement_count, two_forward_example


class RecordingPrior(HistoryPrior):
    def __init__(self):
        super().__init__(width=24, layers=1, heads=4, source_flags=True)
        self.calls = []
        with torch.no_grad():
            self.output.bias[204] = 0.05

    def forward(self, history, valid, base, **kwargs):
        self.calls.append(
            {"history": history.detach().clone(), "grad": torch.is_grad_enabled(), "training": self.training}
        )
        return super().forward(history, valid, base, **kwargs)


def fixture():
    episode = synthetic_episode(length=50)
    return episode, MotionCodec(episode.stats), RecordingPrior().train()


def test_self_history_is_detached_and_only_second_pass_backpropagates():
    episode, codec, model = fixture()
    example = two_forward_example(model, codec, TrainingBatch((episode,)), 40, 20)
    assert model.calls[0]["history"].shape == (20, 20, 243)
    assert model.calls[0]["grad"] is False and model.calls[0]["training"] is False
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not value.requires_grad for value in example.values())
    losses = prior_losses(model, example, codec, DenseJoints())
    (losses["code"] + 0.0158 * losses["geometry"]).backward()
    assert model.calls[1]["grad"] is True and model.calls[1]["training"] is True
    assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters() if parameter.grad is not None) > 0


def test_current_and_future_labels_or_observations_cannot_enter_two_forward_inputs():
    episode, codec, model = fixture()
    original = two_forward_example(model, codec, TrainingBatch((episode,)), 40, 20)
    changed = copy.deepcopy(episode)
    changed.truth.joints[40:, :, :3, 3] += 1.0
    changed.truth.reference[40:, :3, 3] += 2.0
    changed.truth.auxiliary[40:] += 0.1
    changed.images.fill_(float("nan"))
    changed.head.fill_(float("nan"))
    other = two_forward_example(model, codec, TrainingBatch((changed,)), 40, 20)
    for key in ("history", "base", "reference", "body_valid"):
        torch.testing.assert_close(original[key], other[key], atol=0, rtol=0)
    assert not torch.equal(original["target"], other["target"])
    assert not torch.equal(original["truth_xyz"], other["truth_xyz"])


def test_each_first_forecast_uses_only_its_own_preceding_twenty_frames():
    episode, codec, model = fixture()
    two_forward_example(model, codec, TrainingBatch((episode,)), 40, 20)
    original_inputs = model.calls[-1]["history"]
    changed = copy.deepcopy(episode)
    changed.truth.joints[25:, :, :3, 3] += 1.0
    two_forward_example(model, codec, TrainingBatch((changed,)), 40, 20)
    modified_inputs = model.calls[-1]["history"]
    # The first six forecasts predict indices20..25 and must not read index25.
    torch.testing.assert_close(original_inputs[:6], modified_inputs[:6], atol=0, rtol=0)
    assert not torch.equal(original_inputs[6:], modified_inputs[6:])


def test_physical_mixing_recomputes_reference_cv_and_target_coordinates():
    episode, codec, model = fixture()
    batch = TrainingBatch((episode,))
    clean = clean_batch(codec, batch, 40)
    mixed = two_forward_example(model, codec, batch, 40, 3)
    torch.testing.assert_close(clean["history"][:, :17], mixed["history"][:, :17], atol=0, rtol=0)
    assert not torch.equal(clean["reference"], mixed["reference"])
    assert not torch.equal(clean["base"], mixed["base"])
    assert not torch.equal(clean["target"], mixed["target"])
    decoded = codec.decode_current(mixed["target"][:, 0], mixed["reference"])
    torch.testing.assert_close(decoded.joints, episode.truth.joints[40:41], atol=1e-6, rtol=1e-6)
    assert bool(mixed["body_valid"].all())


def test_parallel_first_pass_keeps_sample_and_time_order():
    episode, codec, model = fixture()
    other = copy.deepcopy(episode)
    other.truth.joints[..., 0, 3] += 2.0
    other.truth.reference[..., 0, 3] += 2.0
    other.truth.auxiliary[..., :24] += 0.5
    together = two_forward_example(model, codec, TrainingBatch((episode, other)), 40, 5)
    separately = [two_forward_example(model, codec, TrainingBatch((value,)), 40, 5) for value in (episode, other)]
    for key in together:
        torch.testing.assert_close(
            together[key], torch.cat([example[key] for example in separately]), atol=2e-6, rtol=2e-6
        )


def test_two_forward_curriculum_and_adaptation_phases():
    empty = Path("unused")
    args = PriorTrainOptions(
        empty,
        empty,
        empty,
        empty,
        clean_steps=1000,
        two_forward_steps=1000,
        adapted_predicted_fraction=0.6,
        adapted_self_fraction=0.2,
    )
    assert prior_phase(args, 1000) == ("clean_pretraining", 0.0, 0.0, 0)
    assert [replacement_count(step, 1000, 1000) for step in (1001, 1201, 1401, 1601, 2000)] == [1, 3, 5, 10, 20]
    assert prior_phase(args, 2000) == ("two_forward_pretraining", 0.0, 0.8, 20)
    assert prior_phase(args, 2001) == ("predicted_history_adaptation", 0.6, 0.2, 20)


def test_invalid_two_forward_history_is_rejected_before_prediction():
    episode, codec, model = fixture()
    with pytest.raises(ValueError, match="twenty past states"):
        two_forward_example(model, codec, TrainingBatch((episode,)), 30, 20)
    assert not model.calls
