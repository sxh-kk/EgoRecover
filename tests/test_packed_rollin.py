"""Packed histories retain logical sampling, noise, cache and training units."""

from copy import copy, deepcopy

import pytest
import torch
from torch import nn

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.data import TrainingBatch
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.losses import multi_step_objective
from egorecover.models import MotionModel
from egorecover.rollin import RollinChunk, packed_rollin_caches
from egorecover.state import FIELDS


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class StableModel(nn.Module):
    """Trainable, dropout-bearing endpoint with valid rotations during roll-in."""

    def __init__(self):
        super().__init__()
        delta = torch.zeros(243)
        delta[6:9] = 0.001
        self.delta = nn.Parameter(delta)
        self.dropout = nn.Dropout(0.1)
        self.calls = []

    def forward(self, x, tau, conditions):
        self.calls.append((self.training, len(x), x.detach().clone()))
        return conditions["center"] + self.delta * self.dropout(torch.ones_like(x))


def fixture(lengths, *, offsets=(0, 6)):
    episode = synthetic_episode(length=56)
    episodes = []
    for index in range(4):
        item = copy(episode)
        item.identifier = f"independent_episode_{index}"
        item.bootstrap_only = True
        episodes.append(item)
    config = WindowConfig(window=3, current_weight=0.03, geometry_target="official")
    system = WindowReconstructor(MotionCodec(episode.stats), None, config)
    chunks = [
        RollinChunk(TrainingBatch(tuple(episodes[i * 2 : i * 2 + 2])), 40, offsets[i], lengths[i]) for i in range(2)
    ]
    return system, StableModel(), chunks


def legacy_rollin(system, model, chunk, *, seed):
    first = max(20, chunk.frame - chunk.rollin_frames)
    cache = chunk.batch.cache(first)
    model.eval()
    for frame in range(first, chunk.frame):
        epsilon = chunk.batch.noise(seed, frame, system.config.window, tag="rollin", offset=chunk.global_offset)
        cache = system.online(model, cache, chunk.batch.observations(frame), epsilon).cache.detach()
    return cache


def assert_cache_equal(actual, expected):
    for name in FIELDS:
        torch.testing.assert_close(getattr(actual.body, name), getattr(expected.body, name), rtol=0, atol=0)
        assert not getattr(actual.body, name).requires_grad
    for name in ("anchor", "beta"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert actual.frame == expected.frame
    assert actual.generated_frames == expected.generated_frames


@pytest.mark.parametrize("lengths", [(20, 20), (20, 4), (0, 1), (0, 0)])
def test_packed_rollins_keep_cache_alignment_noise_and_rng_and_reduce_eval_calls(lengths):
    system, model, chunks = fixture(lengths)
    before = torch.get_rng_state().clone()
    expected = [legacy_rollin(system, model, chunk, seed=123) for chunk in chunks]
    legacy_calls = list(model.calls)
    model.calls.clear()
    actual = packed_rollin_caches(system, model, chunks, seed=123)
    assert torch.equal(before, torch.get_rng_state())
    assert model.delta.grad is None
    assert len(model.calls) == sum(set(lengths))
    assert len(legacy_calls) == sum(lengths)
    for chunk, a, b in zip(chunks, actual, expected):
        assert_cache_equal(a, b)
        assert a.frame == chunk.frame - 1
        assert a.generated_frames == chunk.rollin_frames
    if lengths[0] == lengths[1] and lengths[0]:
        assert all(batch == 4 and not training for training, batch, _ in model.calls)
        for index, (_, _, noise) in enumerate(model.calls):
            frame = 40 - lengths[0] + index
            expected_noise = torch.cat(
                [chunk.batch.noise(123, frame, 3, tag="rollin", offset=chunk.global_offset) for chunk in chunks]
            )
            torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)


def objective(system, model, chunk, cache, *, forward_seed):
    batch, frame = chunk.batch, chunk.frame
    return multi_step_objective(
        system,
        model,
        None,
        cache,
        [batch.observations(frame + index) for index in range(8)],
        [batch.target(frame + index, 3) for index in range(8)],
        torch.stack(
            [batch.noise(123, frame + index, 3, tag="training", offset=chunk.global_offset) for index in range(8)]
        ),
        torch.tensor([0.3, 1.0]),
        DenseJoints(),
        dt=torch.full((2, 7), 0.1),
        velocity_weight=0.013409926806430328,
        forward_seed=forward_seed,
    )


@pytest.mark.parametrize("lengths", [(20, 20), (20, 1)])
@pytest.mark.parametrize("isolate_rng", [True, False])
def test_packed_history_preparation_keeps_micro2_training_order_losses_gradients_and_dropout(lengths, isolate_rng):
    system, model, chunks = fixture(lengths)
    packed_model = deepcopy(model)
    torch.manual_seed(1062)
    original_rows = []
    for chunk in chunks:
        cache = legacy_rollin(system, model, chunk, seed=123)
        model.train()
        row = objective(system, model, chunk, cache, forward_seed=chunk.global_offset if isolate_rng else None)
        (row["loss"] * 0.5).backward()
        original_rows.append({name: row[name].detach().clone() for name in ("loss", "flow", "current", "velocity")})
    original_rng = torch.get_rng_state()
    original_gradient = model.delta.grad.clone()
    torch.manual_seed(1062)
    caches = packed_rollin_caches(system, packed_model, chunks, seed=123)
    for chunk, cache, expected in zip(chunks, caches, original_rows):
        packed_model.train()
        row = objective(system, packed_model, chunk, cache, forward_seed=chunk.global_offset if isolate_rng else None)
        (row["loss"] * 0.5).backward()
        for name, value in expected.items():
            torch.testing.assert_close(row[name], value, rtol=0, atol=0)
    torch.testing.assert_close(packed_model.delta.grad, original_gradient, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), original_rng)
    training_batches = [batch for training, batch, _ in packed_model.calls if training]
    assert training_batches == [2] * 32  # Two micro2 chunks, each with eight Flow plus eight endpoint calls.


def test_equal_history_lengths_at_different_physical_frames_are_not_merged():
    system, model, chunks = fixture((20, 20))
    chunks[1] = RollinChunk(chunks[1].batch, 44, chunks[1].global_offset, 20)
    expected = [legacy_rollin(system, model, chunk, seed=123) for chunk in chunks]
    model.calls.clear()
    actual = packed_rollin_caches(system, model, chunks, seed=123)
    assert len(model.calls) == 40
    assert all(batch == 2 for _, batch, _ in model.calls)
    for a, b in zip(actual, expected):
        assert_cache_equal(a, b)


def test_neural_attention_rollin_batching_has_only_small_float32_differences():
    class ResidualModel(MotionModel):
        # Keep neutral rotations valid while exercising all transformer layers.
        def forward(self, x, tau, condition):
            return condition["center"] + 0.001 * super().forward(x, tau, condition)

    system, _, chunks = fixture((20, 20))
    torch.manual_seed(62)
    model = ResidualModel(system.config, ModelConfig.compact())
    before = torch.get_rng_state().clone()
    expected = [legacy_rollin(system, model, chunk, seed=123) for chunk in chunks]
    actual = packed_rollin_caches(system, model, chunks, seed=123)
    assert torch.equal(before, torch.get_rng_state())
    for a, b in zip(actual, expected):
        for name in FIELDS:
            torch.testing.assert_close(getattr(a.body, name), getattr(b.body, name), rtol=1e-5, atol=1e-5)
        assert a.frame == b.frame and a.generated_frames == b.generated_frames
