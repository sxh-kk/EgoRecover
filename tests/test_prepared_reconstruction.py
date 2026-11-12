from copy import deepcopy

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.flow import build_source
from egorecover.losses import training_objective
from egorecover.models import MotionModel


def test_shared_preparation_preserves_endpoint_loss_and_gradients():
    torch.manual_seed(62)
    episode = synthetic_episode()
    config = WindowConfig(current_weight=0.03)
    system = WindowReconstructor(MotionCodec(episode.stats), None, config)
    model = MotionModel(config, ModelConfig.compact()).train()
    reference = deepcopy(model)
    epsilon = torch.randn(1, 3, 243)
    tau = torch.tensor([0.4])
    original = system.reconstruct_prepared
    calls = []

    def recompute(model, cache, source, condition, anchor, **kwargs):
        # Reproduce the old endpoint path, which called prepare a second time.
        y, r = system.prepare(cache, episode.observations(20))
        s = build_source(y["center"], epsilon, mode=config.source, sigma=config.sigma)
        calls.append(True)
        return original(model, cache, s, y, r, **kwargs)

    args = (None, episode.cache(), episode.observations(20), episode.target(20, 3), epsilon, tau, DenseJoints())
    optimized = training_objective(system, model, *args, forward_seed=123)
    system.reconstruct_prepared = recompute
    legacy = training_objective(system, reference, *args, forward_seed=123)
    assert len(calls) == 1
    torch.testing.assert_close(optimized["result"].codes, legacy["result"].codes, rtol=0, atol=0)
    torch.testing.assert_close(optimized["loss"], legacy["loss"], rtol=0, atol=0)
    optimized["loss"].backward()
    legacy["loss"].backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
