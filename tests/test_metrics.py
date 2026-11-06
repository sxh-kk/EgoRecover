import json
import math

import pytest
import torch

from egorecover.metrics import METRIC_DIRECTIONS, comparison_metrics, comparison_protocol


def trajectory(frames=8):
    generator = torch.Generator().manual_seed(2026)
    body = torch.randn(22, 3, generator=generator, dtype=torch.float64) * 0.2
    positions = body[None].repeat(frames, 1, 1)
    positions[..., 2] += 1.0
    positions[:, :, 0] += torch.arange(frames, dtype=torch.float64)[:, None] * 0.01
    return positions, torch.arange(frames, dtype=torch.float64) * 0.1


def test_world_error_retains_translation_while_aligned_metrics_remove_it():
    truth, times = trajectory()
    prediction = truth + torch.tensor([0.03, 0.04, 0.0])
    result = comparison_metrics(prediction, truth, times)
    assert result["mpjpe_world_mm"] == pytest.approx(50)
    assert result["head_joint_pe_mm"] == pytest.approx(50)
    assert result["wrist_pe_mm"] == pytest.approx(50)
    assert result["pa_mpjpe_mm"] < 1e-10
    assert result["seq_sim_mpjpe_mm"] < 1e-10
    assert result["root_mean_initial_translation_mm"] < 1e-10
    assert result["velocity_error_mm_s"] < 1e-10


def test_similarity_alignment_removes_rotation_scale_and_translation_but_not_reflection():
    truth, times = trajectory()
    rotation = truth.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    prediction = 2.5 * (truth @ rotation) + truth.new_tensor([4, -3, 2])
    aligned = comparison_metrics(prediction, truth, times)
    assert aligned["pa_mpjpe_mm"] < 1e-9
    assert aligned["seq_sim_mpjpe_mm"] < 1e-9
    reflected = truth * truth.new_tensor([-1, 1, 1])
    assert comparison_metrics(reflected, truth, times)["pa_mpjpe_mm"] > 10


def test_sequence_and_per_frame_similarity_alignment_remain_distinct():
    truth, times = trajectory()
    prediction = truth.clone()
    prediction[4:, :, 0] += 0.5
    result = comparison_metrics(prediction, truth, times)
    assert result["pa_mpjpe_mm"] < 1e-10
    assert result["seq_sim_mpjpe_mm"] > 10
    assert result["root_final_initial_translation_mm"] == pytest.approx(500)


def test_rotation_frobenius_error_uses_matrices_and_smpl_head_joint():
    truth, times = trajectory()
    rotations = torch.eye(3, dtype=torch.float64).repeat(len(truth), 22, 1, 1)
    predicted = rotations.clone()
    predicted[:, 15] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = comparison_metrics(truth, truth, times, predicted_rotations=predicted, truth_rotations=rotations)
    assert result["head_rot_f"] == pytest.approx(2)
    assert result["mpjre_f"] == pytest.approx(2 / 22)
    assert comparison_metrics(truth, truth, times)["mpjre_f"] is None
    with pytest.raises(ValueError, match="both"):
        comparison_metrics(truth, truth, times, predicted_rotations=predicted)


def test_physical_timebase_controls_derivative_errors():
    truth, times = trajectory()
    prediction = truth.clone()
    prediction[:, :, 0] += (0.5 * times.square())[:, None]
    result = comparison_metrics(prediction, truth, times)
    assert result["acceleration_error_mm_s2"] == pytest.approx(1000)
    slower = comparison_metrics(prediction, truth, 2 * times)
    assert slower["velocity_error_mm_s"] == pytest.approx(result["velocity_error_mm_s"] / 2)
    assert slower["acceleration_error_mm_s2"] == pytest.approx(250)
    assert result["diagnostic_jerk_peak_mm_s3"] < 1e-7


def test_jerk_peak_and_integral_have_separate_physical_units():
    truth, times = trajectory(frames=10)
    truth = truth[:1].repeat(len(times), 1, 1)
    prediction = truth.clone()
    prediction[:, :, 0] += (times**3 / 6)[:, None]
    result = comparison_metrics(prediction, truth, times)
    assert result["diagnostic_jerk_peak_mm_s3"] == pytest.approx(1000)
    assert result["diagnostic_jerk_auc_mm_s2"] == pytest.approx(600)
    assert result["gt_diagnostic_jerk_peak_mm_s3"] == 0
    assert result["diagnostic_jerk_peak_gap_mm_s3"] == pytest.approx(1000)
    assert result["diagnostic_jerk_auc_gap_mm_s2"] == pytest.approx(600)


def test_foot_slide_matches_published_weighting_and_sequence_denominator():
    truth, times = trajectory(frames=4)
    truth[:, [7, 10, 8, 11], 2] = truth.new_tensor([0.04, 0.02, 0.04, 0.02])
    truth[:, [7, 10, 8, 11], 0] = times[:, None] * 0.1
    result = comparison_metrics(truth, truth, times)
    expected = 3 * 0.01 * (2 - math.sqrt(2)) / 4 * 1000
    assert result["foot_slide_uniegomotion_mm"] == pytest.approx(expected)
    assert result["gt_foot_slide_uniegomotion_mm"] == pytest.approx(expected)
    airborne = truth.clone()
    airborne[:, [7, 10, 8, 11], 2] = 1000
    assert comparison_metrics(airborne, airborne, times)["foot_slide_uniegomotion_mm"] == 0


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_short_and_degenerate_trajectories_are_finite_with_explicit_missing_derivatives(frames):
    positions = torch.zeros(frames, 22, 3, dtype=torch.float64)
    times = torch.arange(frames, dtype=torch.float64) / 10
    result = comparison_metrics(positions, positions, times)
    assert all(value is None or math.isfinite(value) for value in result.values())
    assert result["pa_mpjpe_mm"] == 0
    assert result["seq_sim_mpjpe_mm"] == 0
    assert result["diagnostic_jerk_auc_mm_s2"] is None
    assert (result["velocity_error_mm_s"] is None) == (frames < 2)
    assert (result["acceleration_error_mm_s2"] is None) == (frames < 3)


def test_invalid_timebases_and_nonfinite_or_incompatible_positions_are_rejected():
    positions, times = trajectory()
    with pytest.raises(ValueError, match="strictly increasing"):
        comparison_metrics(positions, positions, times.flip(0))
    times[3] += 0.01
    with pytest.raises(ValueError, match="uniform"):
        comparison_metrics(positions, positions, times)
    times = torch.arange(len(positions), dtype=torch.float64) / 10
    with pytest.raises(ValueError, match="shapes"):
        comparison_metrics(positions[:, :21], positions[:, :21], times)
    positions[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        comparison_metrics(positions, positions, times)


def test_protocol_is_json_safe_and_labels_unverified_paper_definitions():
    protocol = comparison_protocol()
    json.dumps(protocol, allow_nan=False)
    assert protocol["head_joint"] == 15
    assert protocol["wrists"] == [20, 21]
    assert protocol["metrics"]["mpjpe_world_mm"]["units"] == "mm"
    assert set(protocol["metrics"]) == set(METRIC_DIRECTIONS)
    assert "egoforce_pj_auj" in protocol["paper_boundaries"]["pending"]
    positions, times = trajectory()
    assert set(comparison_metrics(positions, positions, times)) == set(METRIC_DIRECTIONS)
