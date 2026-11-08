import math

import pytest
import torch

from egorecover import WindowConfig, WindowReconstructor
from egorecover.codec import BodyState, MotionCodec
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.reference_diagnostic import diagnose_episode, intervention_state
from egorecover.state import FIELDS, state_map


class CenterEndpoint(torch.nn.Module):
    def forward(self, value, timesteps, conditions):
        return conditions["center"]


class NoiseDependentEndpoint(torch.nn.Module):
    def forward(self, value, timesteps, conditions):
        # Small perturbations preserve valid rotation codes while making the
        # paired continuations depend on their explicitly supplied noise.
        return conditions["center"] + value * 0.01


class FixedTemplateJoints:
    """Asset-free fixed-shape FK stand-in using rotations and root only."""

    def __init__(self):
        rest = synthetic_episode().truth.joints[0, :, :3, 3]
        self.offsets = rest - rest[:1]
        self.inputs = []

    def __call__(self, body):
        self.inputs.append(body)
        rotations = body.joints[..., :3, :3]
        root = body.joints[..., 0, :3, 3]
        return root[..., None, :] + (rotations @ self.offsets[..., None])[..., 0]


def states_for_intervention():
    episode = synthetic_episode()
    predicted = state_map(episode.teacher, lambda value: value[20:21].clone())
    truth = state_map(episode.truth, lambda value: value[20:21].clone())
    predicted.reference[..., 0, 3] += 4
    predicted.auxiliary[:] = torch.arange(36)
    c, s = math.cos(0.4), math.sin(0.4)
    truth.joints[..., :3, :3] = torch.tensor(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    truth.joints[..., 0, :3, 3] = torch.tensor([2.0, -3.0, 1.0])
    truth.joints[..., 1:, :3, 3] = 77
    truth.reference[..., 1, 3] = -8
    truth.auxiliary[:] = -123
    return (
        state_map(predicted, lambda value: value.requires_grad_(True)),
        state_map(truth, lambda value: value.requires_grad_(True)),
    )


def test_body_pose_replaces_rotations_root_and_fixed_shape_positions_but_keeps_predicted_reference_aux():
    predicted, truth = states_for_intervention()
    decoder = FixedTemplateJoints()
    result = intervention_state(predicted, truth, decoder, "body_pose")
    torch.testing.assert_close(result.joints[..., :3, :3], truth.joints[..., :3, :3])
    torch.testing.assert_close(result.joints[..., 0, :3, 3], truth.joints[..., 0, :3, 3])
    expected = (
        truth.joints[..., 0, :3, 3, None].transpose(-1, -2)
        + (truth.joints[..., :3, :3] @ decoder.offsets[..., None])[..., 0]
    )
    torch.testing.assert_close(result.joints[..., :3, 3], expected)
    assert not torch.equal(result.joints[..., 1:, :3, 3], truth.joints[..., 1:, :3, 3])
    for name in ("reference", "auxiliary"):
        torch.testing.assert_close(getattr(result, name), getattr(predicted, name), atol=0, rtol=0)
        torch.testing.assert_close(getattr(decoder.inputs[0], name), getattr(predicted, name), atol=0, rtol=0)
    assert all(not getattr(result, name).requires_grad for name in FIELDS)

    poisoned = BodyState(truth.joints, torch.full_like(truth.reference, float("nan")), truth.auxiliary * float("nan"))
    clean_rotations_and_root = truth.joints.detach().clone()
    clean_rotations_and_root[..., 1:, :3, 3] = -999
    poisoned = BodyState(clean_rotations_and_root, poisoned.reference, poisoned.auxiliary)
    after = intervention_state(predicted, poisoned, decoder, "body_pose")
    for name in FIELDS:
        torch.testing.assert_close(getattr(after, name), getattr(result, name), atol=0, rtol=0)


def test_full_state_oracle_replaces_all_gt_fields_and_stops_their_gradients():
    predicted, truth = states_for_intervention()

    def forbidden_fk(body):
        raise AssertionError("The full-state intervention must preserve the original complete GT state")

    result = intervention_state(predicted, truth, forbidden_fk, "full_state")
    for name in FIELDS:
        torch.testing.assert_close(getattr(result, name), getattr(truth, name), atol=0, rtol=0)
        assert not getattr(result, name).requires_grad
        assert not torch.equal(getattr(result, name), getattr(predicted, name))


@pytest.mark.parametrize("mode", ("body_pose", "full_state"))
def test_intervention_changes_only_current_cache_slot_and_never_shape_anchor_or_common_history(mode):
    episode = synthetic_episode()
    cache = episode.cache(24)
    original = state_map(cache.body, lambda value: value.clone())
    current = state_map(cache.body, lambda value: value[:, -1])
    truth = state_map(episode.target(23), lambda value: value[:, -1].clone())
    truth.joints[..., 0, :3, 3] += 5
    truth.reference[..., 1, 3] += 7
    truth.auxiliary.fill_(99)
    replacement = intervention_state(current, truth, FixedTemplateJoints(), mode)
    changed = cache.replace_current(replacement)
    assert changed.frame == cache.frame
    torch.testing.assert_close(changed.beta, cache.beta, atol=0, rtol=0)
    torch.testing.assert_close(changed.anchor, cache.anchor, atol=0, rtol=0)
    for name in FIELDS:
        torch.testing.assert_close(getattr(cache.body, name), getattr(original, name), atol=0, rtol=0)
        torch.testing.assert_close(getattr(changed.body, name)[:, :-1], getattr(original, name)[:, :-1], atol=0, rtol=0)
        torch.testing.assert_close(getattr(changed.body, name)[:, -1], getattr(replacement, name), atol=0, rtol=0)
    assert not torch.equal(changed.body.joints[:, -1], original.joints[:, -1])


def test_future_noise_branches_cannot_pollute_later_own_history_prefix_anchors():
    episode = synthetic_episode()
    system = WindowReconstructor(MotionCodec(episode.stats), None, WindowConfig(window=3))
    before = state_map(episode.teacher, lambda value: value.clone())
    arguments = {"anchors": (20, 24, 28), "horizons": (1, 2), "prefix_seed": 2062}
    first = diagnose_episode(system, NoiseDependentEndpoint(), episode, DenseJoints(), future_seeds=(7,), **arguments)
    changed = diagnose_episode(
        system, NoiseDependentEndpoint(), episode, DenseJoints(), future_seeds=(97, 98), **arguments
    )

    def anchor_errors(rows):
        result = {}
        for row in rows:
            if row["anchor"] in result:
                assert result[row["anchor"]] == row["anchor_current_mm"]
            result[row["anchor"]] = row["anchor_current_mm"]
        return result

    assert set(anchor_errors(first)) == {20, 24, 28}
    assert anchor_errors(first) == anchor_errors(changed)
    assert first[0]["predicted_mm"] != changed[0]["predicted_mm"]
    for name in FIELDS:
        torch.testing.assert_close(getattr(episode.teacher, name), getattr(before, name), atol=0, rtol=0)


def test_positive_delta_means_reference_improvement_and_seconds_follow_episode_timebase():
    episode = synthetic_episode()
    episode.timestamps = episode.timestamps * 2
    system = WindowReconstructor(MotionCodec(episode.stats), None, WindowConfig(window=1))
    rows = diagnose_episode(
        system,
        CenterEndpoint(),
        episode,
        DenseJoints(),
        anchors=(20,),
        horizons=(1, 2),
        future_seeds=(42,),
    )
    assert len(rows) == 4
    for row in rows:
        assert row["delta_mm"] == pytest.approx(row["predicted_mm"] - row["reference_mm"])
        assert row["same_shape_delta_mm"] == pytest.approx(
            row["same_shape_predicted_mm"] - row["same_shape_reference_mm"]
        )
        assert row["center_delta_mm"] == pytest.approx(row["center_predicted_mm"] - row["center_reference_mm"])
        assert row["seconds"] == pytest.approx(row["horizon"] * 0.2)
        assert row["delta_mm"] > 0
        assert row["reference_mm"] < row["predicted_mm"]
        assert row["intervened_current_mm"] == pytest.approx(0, abs=1e-3)
        assert row["response_mm"] > 0 and row["anchor_current_mm"] > 0


def test_short_episode_without_a_valid_continuation_returns_empty_without_querying_model():
    episode = synthetic_episode()
    episode.truth = state_map(episode.truth, lambda value: value[:21])
    episode.teacher = state_map(episode.teacher, lambda value: value[:21])
    for name in ("head", "images", "head_valid", "image_valid", "timestamps"):
        setattr(episode, name, getattr(episode, name)[:21])
    system = WindowReconstructor(MotionCodec(episode.stats), None, WindowConfig())

    class ForbiddenEndpoint(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("No continuation is available for the requested anchor")

    assert diagnose_episode(system, ForbiddenEndpoint(), episode, DenseJoints(), anchors=(20,)) == []


def test_only_available_horizons_are_emitted_for_an_anchor_near_the_tail():
    episode = synthetic_episode()
    system = WindowReconstructor(MotionCodec(episode.stats), None, WindowConfig())
    rows = diagnose_episode(
        system,
        CenterEndpoint(),
        episode,
        DenseJoints(),
        anchors=(34,),
        horizons=(1, 2, 10),
        future_seeds=(42,),
    )
    assert len(rows) == 2 and {row["horizon"] for row in rows} == {1}
    assert all(row["seconds"] == pytest.approx(0.1) for row in rows)
