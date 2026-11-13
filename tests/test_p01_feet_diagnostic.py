import pytest
import torch

from egorecover.metrics import foot_slide_uniegomotion
from scripts.diagnose_p01_feet import (
    FEET,
    contact_weights,
    cross_diagnostic,
    diagnose_clip,
    foot_statistics,
    paired_take_bootstrap,
    weighted_slide,
)


def trajectory(frames=10):
    value = torch.zeros(frames, 22, 3, dtype=torch.float64)
    value[:, 0, 2] = 1
    value[:, FEET, 0] = torch.arange(frames, dtype=torch.float64)[:, None] * 0.01
    value[:, FEET, 2] = torch.tensor([0.04, 0.02, -0.01, 0.07], dtype=value.dtype)
    return value


def test_contact_weighted_slide_matches_official_formula_including_penetration_and_airborne():
    value = trajectory()
    feet = value[:, FEET]
    actual = weighted_slide(contact_weights(feet), torch.diff(feet[..., :2], dim=0).norm(dim=-1), len(value))
    assert actual == pytest.approx(foot_slide_uniegomotion(value), abs=1e-12)
    assert contact_weights(feet)[0, 2] > 1
    assert contact_weights(feet)[0, 3] == 0


def test_lowering_feet_changes_height_weight_not_motion_and_GT_fixed_metric_stays_identical():
    cv = trajectory()
    p = cv.clone()
    p[:, FEET, 2] -= 0.02
    contact = torch.ones(len(cv), 2)
    result = cross_diagnostic(cv, p, cv, contact)
    assert result["total_slide_delta_mm"] > 0
    assert result["displacement_Shapley_mm"] == pytest.approx(0, abs=1e-12)
    assert result["height_weight_Shapley_mm"] == pytest.approx(result["total_slide_delta_mm"])
    assert result["fixed_GT_height_weight_delta_mm"] == 0
    assert result["fixed_GT_auxiliary_contact_delta_mm"] == 0


def test_speed_change_attributes_to_motion_and_two_factor_shapley_is_exact():
    cv = trajectory()
    p = cv.clone()
    p[:, FEET, 0] *= 1.4
    result = cross_diagnostic(cv, p, cv, None)
    assert result["height_weight_Shapley_mm"] == pytest.approx(0, abs=1e-12)
    assert result["displacement_Shapley_mm"] == pytest.approx(result["total_slide_delta_mm"])
    p[:, FEET, 2] -= 0.02
    result = cross_diagnostic(cv, p, cv, None)
    assert result["displacement_Shapley_mm"] + result["height_weight_Shapley_mm"] == pytest.approx(
        result["total_slide_delta_mm"], abs=1e-12
    )
    assert result["displacement_at_fixed_CV_weight_delta_mm"] + result[
        "weight_at_fixed_CV_displacement_delta_mm"
    ] + result["interaction_mm"] == pytest.approx(result["total_slide_delta_mm"], abs=1e-12)


def test_startup_mature_split_does_not_include_the_boundary_transition():
    cv = torch.zeros(60, 22, 3, dtype=torch.float64)
    p = cv.clone()
    p[20:, FEET, 0] = 1
    metadata = [
        {"frame": frame, "episode": "clip", "take": "take", "floor_calibrated": True} for frame in range(20, 80)
    ]
    result = {row["phase"]: row for row in diagnose_clip(metadata, cv, p, cv, torch.arange(60).double() / 10)}
    assert result["all"]["P"]["official_slide_mm"] > 0
    assert result["startup"]["P"]["official_slide_mm"] == 0
    assert result["mature"]["P"]["official_slide_mm"] == 0
    assert result["startup"]["frames"] == 20 and result["mature"]["frames"] == 40


def test_predicted_auxiliary_contact_is_not_used_by_official_slide():
    cv = trajectory()
    metadata = [
        {"frame": frame, "episode": "clip", "take": "take", "floor_calibrated": True} for frame in range(20, 30)
    ]
    zero = torch.zeros(10, 2)
    one = torch.ones(10, 2)
    row = diagnose_clip(metadata, cv, cv, cv, torch.arange(10).double() / 10, zero, one, one)[0]
    assert row["P"]["auxiliary_contact_mean"] == 1
    assert row["CV"]["auxiliary_contact_mean"] == 0
    assert row["P"]["official_slide_mm"] == row["CV"]["official_slide_mm"]


def test_paired_bootstrap_retains_clip_weighting_with_take_clusters():
    rows = [{"take": "A", "value": 1}, {"take": "A", "value": 3}, {"take": "B", "value": 8}]
    result = paired_take_bootstrap(rows, lambda row: row["value"])
    assert result["delta"] == 4
    assert result["ci95"] == [2, 8]


def test_GT_conditioned_heights_hold_contact_mask_fixed_and_mark_empty_groups_unavailable():
    gt = trajectory()
    gt[:, FEET, 2] = 0
    prediction = gt.clone()
    prediction[:, FEET, 2] = 0.2
    values = foot_statistics(prediction, gt, torch.arange(10).double() / 10, None, None)
    assert values["selected_fraction_GT_height_contact"] == 1
    assert values["foot_z_error_MAE_mm_on_GT_height_contact"] == pytest.approx(200)
    assert values["predicted_height_contact_fraction_on_GT_height_contact"] == 0
    assert values["foot_z_error_MAE_mm_on_GT_height_noncontact"] is None
