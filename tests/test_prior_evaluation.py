import pytest
import torch

from egorecover.prior_evaluation import center_metrics, summarize_centers


def test_center_derivatives_have_physical_units_and_unknown_floor_is_excluded():
    times = torch.arange(8, dtype=torch.float64) / 10
    truth = torch.zeros(8, 22, 3, dtype=torch.float64)
    prediction = truth.clone()
    prediction[..., 0] = times[:, None] ** 2
    result = center_metrics(prediction, truth, times, False)
    assert result["acceleration_mm_s2"] == pytest.approx(2000)
    assert result["velocity_mm_s"] == pytest.approx(700)
    assert result["foot_slide_mm"] is None


def test_center_metrics_reject_irregular_sampling():
    with pytest.raises(ValueError, match="uniform"):
        center_metrics(torch.zeros(4, 22, 3), torch.zeros(4, 22, 3), torch.tensor([0.0, 0.1, 0.3, 0.4]), True)


def test_mature_temporal_gate_rejects_position_only_improvement():
    cv = dict(mpjpe_mm=110.0, velocity_mm_s=400.0, acceleration_mm_s2=6000.0, foot_slide_mm=20.0)
    rows = [
        {
            "take": take,
            "window": window,
            "phase": phase,
            "CV": cv,
            "P": {**cv, "mpjpe_mm": 100.0, "foot_slide_mm": slide},
        }
        for take in ("a", "b", "c")
        for window in (1, 3)
        for phase, slide in (("all", 10.0), ("mature", 22.0))
    ]
    summary = summarize_centers(rows)
    assert summary["position_gate_passed"]
    assert not summary["temporal_gate_passed"]
    assert not summary["gate_passed"]
    for row in rows:
        row["P"]["foot_slide_mm"] = 19.0
    assert summarize_centers(rows)["gate_passed"]
