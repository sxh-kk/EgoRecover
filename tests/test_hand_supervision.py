"""Hands receive the existing Flow supervision without an additional loss."""

import torch
from torch import nn

from egorecover.flow import flow_loss


class Endpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1, 1, 243))
        self.calls = 0

    def forward(self, x, tau, condition):
        self.calls += 1
        return self.value.expand_as(x)


def test_both_hands_receive_gradients_from_existing_flow_loss():
    model = Endpoint()
    target = torch.zeros(2, 3, 243)
    target[..., 207:219] = 1
    target[..., 219:231] = -2
    terms = flow_loss(model, target, torch.zeros_like(target), {}, torch.ones(2), return_components=True)
    terms["flow"].backward()
    assert terms["hand_pca"].item() == 2.5
    assert torch.count_nonzero(model.value.grad[..., 207:219]) == 12
    assert torch.count_nonzero(model.value.grad[..., 219:231]) == 12
    assert torch.count_nonzero(model.value.grad[..., :207]) == 0
    assert torch.count_nonzero(model.value.grad[..., 231:]) == 0
    assert model.calls == 1


def test_hand_reporting_preserves_original_objective_and_gradients():
    torch.manual_seed(62)
    target, source = torch.randn(2, 3, 243), torch.randn(2, 3, 243)
    model = Endpoint()
    terms = flow_loss(model, target, source, {}, torch.ones(2), return_components=True)
    weights = target.new_ones(243)
    weights[198:207] = 8
    expected = ((model.value.expand_as(target) - target).square() * weights).sum(-1).mean() / weights.sum()
    actual_grad = torch.autograd.grad(terms["flow"], model.value)[0]
    expected_grad = torch.autograd.grad(expected, model.value)[0]
    torch.testing.assert_close(terms["flow"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert model.calls == 1
