import itertools

import pytest
import torch

from egorecover.codec import BodyState, rigid_transform
from scripts.diagnose_p01_corrections import FACTORS, correction_energy, physical_mix, shapley


def state(phase):
    angle = torch.linspace(phase, phase + 0.4, 22)[None]
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    rotation = torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(1, 22, 3, 3)
    position = torch.arange(66).float().reshape(1, 22, 3) * 0.01 + phase
    auxiliary = torch.full((1, 36), phase)
    return BodyState(rigid_transform(rotation, position), torch.eye(4)[None], auxiliary)


def test_physical_intervention_endpoints_and_independent_factors():
    cv, prediction = state(0.1), state(0.5)
    for flag, original in ((0, cv), (1, prediction)):
        mixed = physical_mix(cv, prediction, (flag,) * 4)
        torch.testing.assert_close(mixed.joints[..., :3, :3], original.joints[..., :3, :3], atol=0, rtol=0)
        torch.testing.assert_close(mixed.joints[..., 0, :3, 3], original.joints[..., 0, :3, 3], atol=0, rtol=0)
        torch.testing.assert_close(mixed.auxiliary[..., -10:], original.auxiliary[..., -10:], atol=0, rtol=0)
    mixed = physical_mix(cv, prediction, (0, 1, 0, 1))
    relative = cv.joints[:, 0:1, :3, :3].transpose(-1, -2) @ cv.joints[..., :3, :3]
    expected = prediction.joints[:, 0:1, :3, :3] @ relative
    torch.testing.assert_close(mixed.joints[..., :3, :3], expected)
    torch.testing.assert_close(mixed.joints[..., 0, :3, 3], cv.joints[..., 0, :3, 3])
    torch.testing.assert_close(mixed.auxiliary[..., -10:], prediction.auxiliary[..., -10:])


def test_shapley_preserves_total_and_shares_interactions():
    values = {
        bits: sum((i + 1) * v for i, v in enumerate(bits)) + 3 * bits[0] * bits[1]
        for bits in itertools.product((0, 1), repeat=4)
    }
    result = shapley(values)
    assert result == pytest.approx(dict(zip(FACTORS, (2.5, 3.5, 3, 4))))
    assert sum(result.values()) == pytest.approx(values[(1,) * 4] - values[(0,) * 4])


def test_correction_energy_uses_signed_alignment_and_exact_MSE_identity():
    truth = torch.zeros(4, 22, 3)
    result = correction_energy(torch.ones_like(truth), torch.full_like(truth, 0.5), truth)
    assert result["CV_position_MSE_mm2"] == 3e6
    assert result["P_position_MSE_mm2"] == 0.75e6
    assert result["delta_MSE_mm2"] == pytest.approx(
        result["linear_alignment_term_mm2"] + result["squared_correction_term_mm2"]
    )
    assert result["position_linear_oracle_alpha"] == 2
    assert result["cosine_error_correction"] == -1
