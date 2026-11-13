import copy
import json

import pytest
import torch

from egorecover.formal_prior_evaluation import (
    FOOT_DIAGNOSTIC_METRICS,
    FOOT_POINTS,
    FOOT_THRESHOLDS_M,
    METRICS,
    evaluate_formal_prior,
    formal_center_metrics,
    pilot_gate,
)


class CVPrior(torch.nn.Module):
    def forward(self, history, valid, base, *, history_body_valid):
        return base


class PositionCodec:
    """Asset-free positions let evaluator integration tests isolate scoring."""

    def decode_current(self, codes, reference):
        return codes[:, :66].reshape(-1, 22, 3)


def make_bank(specifications=(("contact", True, 0.0),), *, boundary_jump=False):
    rows, predictions, truths = [], [], []
    for episode, floor, height in specifications:
        for frame in range(20, 44):
            rows.append(
                {
                    "episode": episode,
                    "take": f"take-{episode}",
                    "frame": frame,
                    "timestamp_s": 100 + frame / 10,
                    "floor_calibrated": floor,
                    "kind": "predicted",
                    "window": 3,
                }
            )
            truth = torch.zeros(22, 3, dtype=torch.float64)
            truth[list(FOOT_POINTS.values()), 2] = height
            prediction = truth.clone()
            if boundary_jump and frame < 40:
                prediction[:, 0] = 1
            predictions.append(prediction)
            truths.append(truth)
    count = len(rows)
    base = torch.zeros(count, 1, 243, dtype=torch.float64)
    base[:, 0, :66] = torch.stack(predictions).flatten(1)
    tensors = {
        "history": torch.zeros(count, 20, 243),
        "base": base,
        "target": torch.zeros_like(base),
        "reference": torch.eye(4).expand(count, 4, 4).clone(),
        "truth_xyz": torch.stack(truths),
        "body_valid": torch.ones(count, 20, dtype=torch.bool),
    }
    return {"format": "egorecover_prior_bank_v2", "producer_window": 3, "rows": rows, "tensors": tensors}


def evaluate(bank, **kwargs):
    return evaluate_formal_prior(CVPrior(), bank, PositionCodec(), lambda state: state, bootstrap_samples=0, **kwargs)


def centers(prediction, truth, *, floor=True, enabled=True):
    times = torch.arange(len(truth), dtype=torch.float64) / 10
    return formal_center_metrics(prediction, truth, times, floor, include_foot_diagnostics=enabled)


def test_fixed_gt_height_weight_cannot_improve_by_raising_prediction_feet():
    truth = torch.zeros(3, 22, 3, dtype=torch.float64)
    truth[:, list(FOOT_POINTS.values()), 2] = torch.tensor(FOOT_THRESHOLDS_M).double() / 2
    low = truth.clone()
    low[..., 0] = torch.arange(3, dtype=torch.float64)[:, None] * 0.01
    low[..., 2] = 0
    high = low.clone()
    high[..., 2] = 1
    low_metrics, high_metrics = centers(low, truth), centers(high, truth)
    expected = 10 * (2 - 2**0.5) * 2 / 3
    assert low_metrics["foot_slide_gt_height_weight_mm"] == pytest.approx(expected, abs=1e-6)
    assert high_metrics["foot_slide_gt_height_weight_mm"] == low_metrics["foot_slide_gt_height_weight_mm"]
    assert high_metrics["foot_slide_mm"] == 0 < low_metrics["foot_slide_mm"]
    faster = low.clone()
    faster[..., 0] *= 2
    assert centers(faster, truth)["foot_slide_gt_height_weight_mm"] == pytest.approx(2 * expected, abs=1e-6)


def test_gt_height_conditions_use_all_four_thresholds_and_report_joint_proxies():
    truth = torch.zeros(2, 22, 3, dtype=torch.float64)
    truth[:, list(FOOT_POINTS.values()), 2] = torch.tensor([0.0, 0.04, 0.08, 0.03], dtype=torch.float64)
    prediction = truth.clone()
    prediction[:, list(FOOT_POINTS.values()), 2] = torch.tensor([-0.01, 0.03, 0.07, 0.08], dtype=torch.float64)
    metrics = centers(prediction, truth)
    assert metrics["foot_z_mae_mm"] == pytest.approx(20)
    assert metrics["foot_z_mae_gt_height_contact_mm"] == pytest.approx(30)
    assert metrics["foot_z_mae_gt_height_noncontact_mm"] == pytest.approx(10)
    assert metrics["foot_low_height_contact_fraction"] == 0.75
    assert metrics["foot_gt_noncontact_false_contact_fraction"] == 1
    assert metrics["foot_negative_z_fraction"] == 0.25
    assert metrics["gt_foot_negative_z_fraction"] == 0
    assert metrics["right_toe_z_mae_mm"] == pytest.approx(50)
    assert metrics["left_ankle_z_mae_mm"] == pytest.approx(10)


def test_unknown_floor_is_unavailable_and_one_frame_has_no_slide_stencil():
    truth = torch.zeros(2, 22, 3)
    prediction = truth + 0.01
    metrics = centers(prediction, truth, floor=False)
    assert all(metrics[name] is None for name in FOOT_DIAGNOSTIC_METRICS)
    assert metrics["foot_slide_mm"] is None
    assert metrics["mpjpe_mm"] > 0
    short = centers(prediction[:1], truth[:1])
    assert short["foot_slide_gt_height_weight_mm"] is None
    assert short["foot_z_mae_mm"] == pytest.approx(10)
    assert short["foot_z_mae_gt_height_noncontact_mm"] is None
    assert short["foot_gt_noncontact_false_contact_fraction"] is None


def test_evaluator_phase_stencils_never_cross_startup_to_mature():
    bank = make_bank(boundary_jump=True)
    report = evaluate(bank, include_foot_diagnostics=True, batch_size=7)
    for name in ("P", "CV"):
        assert report["summary"]["all"]["metrics"]["foot_slide_gt_height_weight_mm"][name] > 0
        assert report["summary"]["startup"]["metrics"]["foot_slide_gt_height_weight_mm"][name] == 0
        assert report["summary"]["mature"]["metrics"]["foot_slide_gt_height_weight_mm"][name] == 0
    assert report["summary"]["startup"]["foot_diagnostics"]["foot_samples"] == 80
    assert report["summary"]["mature"]["foot_diagnostics"]["foot_samples"] == 16
    json.dumps(report, allow_nan=False)


def test_conditional_empty_groups_and_floor_coverage_have_correct_sample_and_clip_counts():
    bank = make_bank((("contact", True, 0.0), ("noncontact", True, 0.2), ("unknown", False, 0.0)))
    report = evaluate(bank, include_foot_diagnostics=True)
    for phase, frame_count in (("all", 24), ("startup", 20), ("mature", 4)):
        summary = report["summary"][phase]
        metrics, counts = summary["metrics"], summary["foot_diagnostics"]
        assert summary["clips"] == 3
        assert counts == {
            "floor_calibrated_clips": 2,
            "foot_samples": 8 * frame_count,
            "gt_height_contact_samples": 4 * frame_count,
            "gt_height_noncontact_samples": 4 * frame_count,
            "gt_height_contact_valid_clips": 1,
            "gt_height_noncontact_valid_clips": 1,
        }
        assert metrics["foot_z_mae_mm"]["clips"] == 2
        assert metrics["foot_z_mae_gt_height_contact_mm"]["clips"] == 1
        assert metrics["foot_z_mae_gt_height_noncontact_mm"]["clips"] == 1
        assert metrics["foot_gt_noncontact_false_contact_fraction"]["clips"] == 1
        assert metrics["foot_z_mae_gt_height_noncontact_mm"]["takes"] == 1
    for row in report["rows"]:
        if row["episode"] == "unknown":
            assert row["foot_diagnostic_counts"] is None
        else:
            counts = row["foot_diagnostic_counts"]
            assert (
                counts["foot_samples"] == counts["gt_height_contact_samples"] + counts["gt_height_noncontact_samples"]
            )
            empty_kind = "noncontact" if row["episode"] == "contact" else "contact"
            assert row["P"][f"foot_z_mae_gt_height_{empty_kind}_mm"] is None
    unknown = evaluate(make_bank((("unknown", False, 0.0),)), include_foot_diagnostics=True)
    assert unknown["summary"]["all"]["metrics"]["foot_z_mae_mm"] is None
    assert unknown["summary"]["all"]["foot_diagnostics"]["foot_samples"] == 0


def test_default_report_and_formal_gates_remain_identical_and_fake_context_is_supported():
    bank = make_bank(boundary_jump=True)
    original = evaluate(bank)
    assert original == evaluate(bank, include_foot_diagnostics=False)
    enhanced = evaluate(bank, include_foot_diagnostics=True)
    stripped = copy.deepcopy(enhanced)
    del stripped["protocol"]["foot_diagnostics"]
    for row in stripped["rows"]:
        del row["foot_diagnostic_counts"]
        for name in ("P", "CV"):
            for metric in FOOT_DIAGNOSTIC_METRICS:
                del row[name][metric]
    for summary in stripped["summary"].values():
        del summary["foot_diagnostics"]
        for metric in FOOT_DIAGNOSTIC_METRICS:
            del summary["metrics"][metric]
    assert stripped == original
    assert METRICS == ("mpjpe_mm", "velocity_mm_s", "acceleration_mm_s2", "foot_slide_mm")
    assert pilot_gate({1500: original, 2000: original}, checks_passed=True) == pilot_gate(
        {1500: enhanced, 2000: enhanced}, checks_passed=True
    )

    class EmptyContext:
        rank, world_size = 3, 4

        def gather_rows(self, rows):
            assert rows == []
            return enhanced["rows"]

    assert evaluate(bank, context=EmptyContext(), include_foot_diagnostics=True) == enhanced
