"""Exercise real Dynamo capture on CPU; CUDAgraph fidelity is GPU-gated separately."""

from copy import deepcopy

import torch

from egorecover import ModelConfig, WindowConfig
from egorecover.models import MotionModel


def test_eval_compilation_reads_inplace_parameter_updates_and_preserves_eager_gradient_paths(monkeypatch):
    actual_compile = torch.compile
    invocations = []

    def cpu_compile(function, **options):
        assert options == {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}
        captured = actual_compile(function, backend="eager", fullgraph=True, dynamic=False)

        def run(*args):
            invocations.append(True)
            return captured(*args)

        return run

    monkeypatch.setattr(torch, "compile", cpu_compile)
    torch.manual_seed(62)
    protocol = WindowConfig(window=3)
    model = MotionModel(protocol, ModelConfig.compact()).eval()
    eager = deepcopy(model)
    b, f, h, w = 2, protocol.fixed, protocol.history, protocol.window
    x = torch.randn(b, w, 243)
    tau = torch.ones(b)
    conditions = {
        "fixed_motion": torch.randn(b, f, 243),
        "fixed_valid": torch.ones(b, f, dtype=torch.bool),
        "center": torch.randn(b, w, 243),
        "traj": torch.randn(b, h + 1, 18),
        "images": torch.randn(b, h + 1, 1024),
        "head_valid": torch.ones(b, h + 1, dtype=torch.bool),
        "image_valid": torch.ones(b, h + 1, dtype=torch.bool),
    }
    original_keys = tuple(model.state_dict())
    model.enable_eval_compile()
    assert tuple(model.state_dict()) == original_keys
    with torch.no_grad():
        first = model(x, tau, conditions)
        saved_first = first.clone()
        torch.testing.assert_close(first, eager(x, tau, conditions))
        model.tsfm[0].ff.net[4].bias.add_(0.3)
        eager.load_state_dict(model.state_dict())
        second = model(x, tau, conditions)
        torch.testing.assert_close(second, eager(x, tau, conditions))
    assert not torch.equal(first, second)
    assert torch.equal(first, saved_first)
    assert len(invocations) == 2
    # An eval-mode frozen teacher may still transmit gradients to its inputs.
    differentiable_input = x.clone().requires_grad_()
    output = model(differentiable_input, tau, conditions)
    gradient = torch.autograd.grad(output.square().sum(), differentiable_input)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    model.train()
    model(x, tau, conditions).sum().backward()
    assert len(invocations) == 2
