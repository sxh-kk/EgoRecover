import pytest
import torch

from egorecover.codec import rigid_inverse, rigid_transform, transform_from_9d
from egorecover.rotations import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


def assert_rigid(transform, tolerance=1e-6):
    rotation = transform[..., :3, :3]
    identity3 = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, identity3, atol=tolerance, rtol=0)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity3, atol=tolerance, rtol=0)
    torch.testing.assert_close(
        torch.linalg.det(rotation), rotation.new_ones(rotation.shape[:-2]), atol=tolerance, rtol=0
    )
    identity4 = torch.eye(4, dtype=rotation.dtype).expand_as(transform)
    torch.testing.assert_close(rigid_inverse(transform) @ transform, identity4, atol=tolerance, rtol=0)
    torch.testing.assert_close(transform @ rigid_inverse(transform), identity4, atol=tolerance, rtol=0)


def test_nearly_parallel_axes_retry_same_float64_projection_and_retain_gradients():
    value = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.000002, 0.2, 0.3, 0.4], requires_grad=True)
    original = rotation_6d_to_matrix(value[:6])
    assert torch.linalg.det(original) > 0.99
    assert (original.T @ original - torch.eye(3)).abs().max() > 0.05

    actual = transform_from_9d(value)
    assert_rigid(actual)
    expected_input = value.detach().clone().requires_grad_(True)
    expected = rigid_transform(
        rotation_6d_to_matrix(expected_input[:6].double()).to(expected_input.dtype), expected_input[6:9]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    coefficients = torch.arange(16, dtype=value.dtype).reshape(4, 4)
    gradient = torch.autograd.grad((actual * coefficients).sum(), value)[0]
    expected_gradient = torch.autograd.grad((expected * coefficients).sum(), expected_input)[0]
    assert bool(torch.isfinite(gradient).all())
    assert gradient[:6].abs().sum() > 0
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(4, 9), (2, 2, 9)])
def test_mixed_batch_preserves_healthy_values_and_has_finite_overflow_gradients(shape):
    # A mixed batch must not backpropagate through the discarded overflowing
    # float32 norm, even though good rows keep their original float32 values.
    value = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0.2, 0.3, 0.4],
            [1, 1, 1, 1, 1, 1.000002, 0.4, 0.5, 0.6],
            [1e30, 0, 0, 0, 1e30, 0, -0.4, -0.5, -0.6],
            [0.6, -0.8, 0, 0.8, 0.6, 0, -0.2, -0.3, -0.4],
        ],
        dtype=torch.float32,
    ).reshape(shape)
    value.requires_grad_(True)
    actual = transform_from_9d(value)
    assert_rigid(actual)
    good = value.reshape(-1, 9)[[0, 3]]
    original_good = rigid_transform(rotation_6d_to_matrix(good[:, :6]), good[:, 6:9])
    torch.testing.assert_close(actual.reshape(-1, 4, 4)[[0, 3]], original_good, atol=0, rtol=0)
    coefficients = torch.arange(16, dtype=value.dtype).reshape(4, 4)
    gradient = torch.autograd.grad((actual * coefficients).sum(), value)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient.reshape(-1, 9)[:, :6].abs().sum(-1) > 0).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_healthy_rotation_projection_is_bitwise_unchanged(dtype):
    generator = torch.Generator().manual_seed(1062)
    rotation = axis_angle_to_matrix(torch.randn(64, 3, generator=generator, dtype=dtype))
    value = torch.cat((matrix_to_rotation_6d(rotation), torch.randn(64, 3, generator=generator, dtype=dtype)), -1)
    expected = rigid_transform(rotation_6d_to_matrix(value[:, :6]), value[:, 6:9])
    actual = transform_from_9d(value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert_rigid(actual)


@pytest.mark.parametrize("axes", [[0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1]])
def test_zero_or_collinear_axes_remain_rejected(axes):
    with pytest.raises(ValueError, match="Degenerate 6D rotation"):
        transform_from_9d(torch.tensor([*axes, 0, 0, 0], dtype=torch.float32))


def test_float64_retry_is_also_required_to_be_orthogonal():
    value = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.00000000001, 0.2, 0.3, 0.4], dtype=torch.float64)
    original = rotation_6d_to_matrix(value[:6])
    assert torch.linalg.det(original) > 0.99
    assert (original.T @ original - torch.eye(3, dtype=value.dtype)).abs().max() > 1e-5
    with pytest.raises(ValueError, match="max_orth_error64"):
        transform_from_9d(value)
