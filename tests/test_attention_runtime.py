from copy import deepcopy

import pytest
import torch

from egorecover.models.layers import Attention


@pytest.mark.parametrize("training", [False, True])
def test_inference_attention_preserves_outputs_and_gradients_and_training_dropout_rng(training):
    torch.manual_seed(1062)
    layer = Attention(32, 4, 0.1).train(training)
    reference = deepcopy(layer)
    x = torch.randn(2, 7, 32, requires_grad=True)
    memory = torch.randn(2, 9, 32, requires_grad=True)
    mask = torch.rand(2, 9) > 0.2
    mask[:, 0] = True
    torch.manual_seed(62)
    expected = reference.out_dropout(
        reference.attn(reference.norm(x), memory, memory, key_padding_mask=~mask, need_weights=True)[0]
    )
    expected_rng = torch.get_rng_state()
    ga = torch.autograd.grad(expected.square().sum(), (x, memory, *reference.parameters()))
    torch.manual_seed(62)
    actual = layer(x, memory, mask[:, None, None])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    gb = torch.autograd.grad(actual.square().sum(), (x, memory, *layer.parameters()))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)
    if training:
        assert torch.equal(actual, expected)
        assert all(torch.equal(a, b) for a, b in zip(ga, gb))
