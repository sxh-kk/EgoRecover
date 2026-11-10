import numpy as np
import pytest
import torch

from egorecover.metrics import foot_slide_uniegomotion
from scripts.analyze_rewrite_frame_traces import aligned_series, historical_accuracy, within_clip_association
from scripts.history_rewrite_metrics import association, diagnostic_metrics, paired_cluster, ranks


def trajectories(count=8):
    rng = np.random.default_rng(1062)
    initial = rng.normal(size=(22, 3)) * 0.1
    initial[:, 2] = 0.02
    positions = initial[None] + np.arange(count)[:, None, None] * np.array([0.01, 0.0, 0.0])
    previous = np.stack([positions[np.maximum(np.arange(count) - lag, 0)] for lag in (2, 1)], axis=1)
    windows = np.concatenate([previous, positions[:, None]], axis=1)
    return positions, windows, previous, np.arange(count) / 10


def test_no_corrections_has_no_seam_and_matches_native_foot_metric():
    positions, windows, previous, times = trajectories()
    summary, traces = diagnostic_metrics(positions, positions, windows, previous, times, rewrite=True)
    for key in [
        "candidate_cache_update_body_mm",
        "velocity_seam_gap_mm_s",
        "acceleration_seam_gap_mm_s2",
        "first_emission_velocity_error_mm_s",
        "within_window_acceleration_error_mm_s2",
    ]:
        assert summary[key] == pytest.approx(0)
    assert summary["first_emission_foot_slide_mm"] == pytest.approx(
        foot_slide_uniegomotion(torch.from_numpy(positions)), abs=1e-10
    )
    assert len(traces["velocity_error_mm_s"]) == len(times) - 1
    assert len(traces["acceleration_error_mm_s2"]) == len(times) - 2


def test_bootstrap_slots_are_excluded_and_disabled_writes_keep_proposals():
    positions, windows, previous, times = trajectories()
    windows[0, :2] += 10
    windows[1, 0] += 10
    windows[1:, 1, :, 0] += 0.02
    summary, _ = diagnostic_metrics(positions, positions, windows, previous, times, rewrite=False)
    assert summary["candidate_cache_update_lag1_body_mm"] == pytest.approx(20)
    assert summary["candidate_cache_update_lag2_body_mm"] == pytest.approx(0)
    assert summary["applied_cache_update_body_mm"] == 0
    assert summary["velocity_seam_gap_mm_s"] == pytest.approx(200)
    assert summary["acceleration_seam_gap_mm_s2"] == pytest.approx(4000)
    assert summary["valid_lag1_corrections"] == len(times) - 1
    assert summary["valid_lag2_corrections"] == len(times) - 2


def test_incremental_cache_rewrite_and_cumulative_first_emission_difference_are_distinct():
    positions, windows, previous, times = trajectories()
    previous[2:, 0, :, 0] += 0.01
    windows[2:, 0, :, 0] += 0.03
    summary, _ = diagnostic_metrics(positions, positions, windows, previous, times, rewrite=True)
    assert summary["candidate_cache_update_lag2_body_mm"] == pytest.approx(20)
    assert summary["cumulative_lag2_deviation_from_first_mm"] == pytest.approx(30)
    assert summary["acceleration_seam_gap_mm_s2"] == pytest.approx(3000)


def test_temporal_scale_and_motion_label_controls_are_posthoc_only():
    positions, windows, previous, times = trajectories()
    windows[1:, 1, :, 0] += 0.02
    normal, _ = diagnostic_metrics(positions, positions, windows, previous, times, rewrite=True)
    slower, _ = diagnostic_metrics(positions, positions, windows, previous, times * 2, rewrite=True)
    assert slower["velocity_seam_gap_mm_s"] == pytest.approx(normal["velocity_seam_gap_mm_s"] / 2)
    assert slower["acceleration_seam_gap_mm_s2"] == pytest.approx(normal["acceleration_seam_gap_mm_s2"] / 4)
    bad = windows.copy()
    bad[:, -1] += 1
    with pytest.raises(ValueError, match="terminal"):
        diagnostic_metrics(positions, positions, bad, previous, times, rewrite=True)


def test_take_cluster_pairing_and_rank_ties():
    assert ranks([3, 1, 1, 2]).tolist() == [3, 0.5, 0.5, 2]
    values = paired_cluster([-2, -2, -2, -2], ["a", "a", "b", "c"], draws=20)
    assert values["delta"] == -2
    assert values["ci95"] == [-2, -2]
    assert values["fraction_improved"] == 1
    result = association([1, 2, 3, 4], [2, 4, 6, 8], ["a", "a", "b", "c"], draws=20)
    assert result["spearman"] == pytest.approx(1)
    assert result["spearman_ci95"] == pytest.approx([1, 1])


def test_frame_associations_respect_write_then_next_output_order():
    positions, _, _, times = trajectories()
    payload = {
        "truth_joints": positions,
        "timestamps": times,
        "traces": {
            "lag1_correction_mm": np.arange(1, 8),
            "lag1_feet_xy_correction_mm": np.arange(1, 8),
            "velocity_error_mm_s": np.arange(1, 8) * 10,
            "acceleration_error_mm_s2": np.arange(2, 8) * 100,
            "foot_slide_transition_mm": np.arange(1, 8),
        },
    }
    for key in ["velocity_error_mm_s", "acceleration_error_mm_s2", "foot_slide_transition_mm"]:
        current, y, _ = aligned_series(payload, key, next_frame=False)
        previous, next_y, _ = aligned_series(payload, key, next_frame=True)
        assert current.tolist() == [2, 3, 4, 5, 6, 7]
        assert previous.tolist() == [1, 2, 3, 4, 5, 6]
        np.testing.assert_array_equal(y, next_y)


def test_within_clip_correlations_remove_between_clip_offsets():
    x = np.arange(1, 7)
    controls = np.zeros((6, 3))
    result = within_clip_association(
        [
            ("a", x, x * 2 + 100, controls),
            ("b", x + 100, x * 2, controls),
        ],
        draws=30,
    )
    assert result["within_clip_pearson"] == pytest.approx(1)
    assert result["within_clip_pearson_ci95"] == pytest.approx([1, 1])
    assert result["frames"] == 12


def test_historical_candidate_accuracy_compares_the_same_physical_frame():
    positions, windows, previous, _ = trajectories()
    emitted = positions.copy()
    emitted[:, :, 0] += 0.02
    previous[1:, 1, :, 0] += 0.02
    previous[2:, 0, :, 0] += 0.01
    payload = {
        "truth_joints": positions,
        "first_emission_joints": emitted,
        "candidate_window_joints": windows,
        "incoming_cached_past_joints": previous,
    }
    result = historical_accuracy(payload)
    assert result["lag1"]["candidate_minus_first_mm"] == pytest.approx(-20)
    assert result["lag2"]["candidate_minus_incoming_mm"] == pytest.approx(-10)
    assert result["lag1"]["fraction_better_than_incoming_over1mm"] == 1
