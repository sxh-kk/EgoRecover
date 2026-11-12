"""Compare the optimized joint-only path against licensed full SMPL-X LBS."""

import os

import pytest
import torch

from egorecover.codec import BodyState, rigid_transform
from egorecover.fk import FixedShapeFK, load_smpl, smpl_path
from egorecover.rotations import rotation_6d_to_matrix


@pytest.fixture(scope="module")
def smpl():
    if not smpl_path().is_file():
        pytest.skip("Licensed SMPL-X asset is not installed.")
    return load_smpl(os.environ.get("SMPLX_TEST_DEVICE", "cpu"))


@pytest.mark.parametrize("shape_mode", ["bootstrap", "predicted"])
@pytest.mark.parametrize("prefix", [(1,), (2,), (2, 3)])
def test_body_only_preserves_positions_and_rotation_shape_root_gradients(smpl, shape_mode, prefix):
    device = smpl.v_template.device
    generator = torch.Generator(device=device).manual_seed(1062 + len(prefix))
    raw = torch.randn(*prefix, 22, 6, generator=generator, device=device).requires_grad_()
    root = (torch.randn(*prefix, 3, generator=generator, device=device) * 10).requires_grad_()
    auxiliary = (torch.randn(*prefix, 36, generator=generator, device=device) * 3).requires_grad_()
    bootstrap = torch.linspace(-2, 2, 10, device=device)
    full = FixedShapeFK(smpl, bootstrap, shape_mode=shape_mode, backend="full")
    fast = FixedShapeFK(smpl, bootstrap, shape_mode=shape_mode, backend="body")
    rotations = rotation_6d_to_matrix(raw)
    translation = torch.cat((root[..., None, :], root.new_zeros(*prefix, 21, 3)), -2)
    state = BodyState(
        rigid_transform(rotations, translation),
        torch.eye(4, device=device).expand(*prefix, 4, 4),
        auxiliary,
    )
    a, b = full(state), fast(state)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(b[..., 0, :], root, rtol=0, atol=0)
    probe = torch.randn(*prefix, 22, 3, generator=generator, device=device)
    parameters = (raw, auxiliary, root)
    ga = torch.autograd.grad((a * probe).sum(), parameters, retain_graph=True)
    gb = torch.autograd.grad((b * probe).sum(), parameters, allow_unused=True)
    for expected, actual in zip(ga, gb):
        actual = torch.zeros_like(expected) if actual is None else actual
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative < 1e-4
    assert torch.count_nonzero(ga[1][..., :26]) == 0  # No body geometry depends on hand/contact channels.
    if shape_mode == "predicted":
        assert gb[1][..., -10:].norm() > 0
    else:
        assert gb[1] is None  # Bootstrap beta is fixed; predicted beta must not leak into this mode.


def test_body_fk_per_sample_bootstrap_and_shared_geometry_cache(smpl):
    device = smpl.v_template.device
    beta = torch.stack((torch.zeros(10, device=device), torch.ones(10, device=device)))
    full = FixedShapeFK(smpl, beta, backend="full")
    fast = FixedShapeFK(smpl, beta, backend="body")
    geometry = smpl._body_joint_geometry
    assert FixedShapeFK(smpl, beta).smpl._body_joint_geometry is geometry
    state = BodyState(
        torch.eye(4, device=device).repeat(2, 22, 1, 1),
        torch.eye(4, device=device).repeat(2, 1, 1),
        torch.zeros(2, 36, device=device),
    )
    torch.testing.assert_close(full(state), fast(state), rtol=1e-6, atol=1e-6)
    assert not torch.equal(fast(state)[0], fast(state)[1])
    with pytest.raises(ValueError, match="flattened body batch"):
        fast(BodyState(state.joints[:1], state.reference[:1], state.auxiliary[:1]))


def test_body_fk_rejects_nonfinite_positions(smpl):
    device = smpl.v_template.device
    auxiliary = torch.zeros(1, 36, device=device)
    auxiliary[0, -1] = float("nan")
    state = BodyState(torch.eye(4, device=device).repeat(1, 22, 1, 1), torch.eye(4, device=device)[None], auxiliary)
    with pytest.raises(ValueError, match="nonfinite joints"):
        FixedShapeFK(smpl, torch.zeros(10, device=device), shape_mode="predicted", backend="body")(state)
