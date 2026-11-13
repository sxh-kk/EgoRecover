import copy
import json

import pytest
import torch

from egorecover.codec import MotionCodec
from egorecover.diagnostics import DenseJoints
from egorecover.formal_prior_evaluation import (
    evaluate_formal_prior,
    final_gate,
    formal_center_metrics,
    pilot_gate,
    summarize_formal_prior_rows,
)
from egorecover.models import HistoryPrior


class CVPrior(torch.nn.Module):
    def forward(self, history, valid, base, *, history_body_valid):
        return base


class BetaPrior(torch.nn.Module):
    def forward(self, history, valid, base, *, history_body_valid):
        result = base.clone()
        result[..., 233] += 1
        return result


class ShapeJoints:
    shape_mode = "predicted"

    def __call__(self, state):
        result = state.joints[..., :3, 3].clone()
        result[..., 0] += state.auxiliary[..., -10, None] * torch.arange(22)[None] / 100
        return result


def make_bank(specifications=(("a", "take-a", 24, True),), *, boundary_error=False):
    rows = []
    for episode, take, length, floor in specifications:
        rows.extend(
            {
                "episode": episode,
                "take": take,
                "frame": frame,
                "timestamp_s": 100 + frame / 10,
                "floor_calibrated": floor,
                "kind": "predicted",
                "window": 3,
            }
            for frame in range(20, 20 + length)
        )
    count = len(rows)
    raw = torch.zeros(count, 243)
    transform = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 0.02])
    raw[:, :198] = transform.repeat(22)
    raw[:, :198].reshape(count, 22, 9)[..., 6] = torch.arange(22)[None] / 100
    raw[:, 198:207] = torch.tensor([1.0, 0, 0, 0, 1, 0, 0, 0, 0])
    truth = raw[:, :198].reshape(count, 22, 9)[..., 6:9].clone()
    if boundary_error:
        for index, row in enumerate(rows):
            if row["frame"] < 40:
                raw[index, :198].reshape(22, 9)[:, 6] += 1
    stats = {
        "motion_mean": torch.zeros(243),
        "motion_std": torch.ones(243),
        "traj_mean": torch.zeros(18),
        "traj_std": torch.ones(18),
    }
    codec = MotionCodec(stats)
    tensors = {
        "history": torch.zeros(count, 20, 243),
        "base": codec.normalize(raw)[:, None],
        "target": torch.zeros(count, 1, 243),
        "reference": torch.eye(4).expand(count, 4, 4).clone(),
        "truth_xyz": truth,
        "body_valid": torch.ones(count, 20, dtype=torch.bool),
    }
    return {"format": "egorecover_prior_bank_v2", "producer_window": 3, "rows": rows, "tensors": tensors}, codec


def evaluate(bank, codec, model=None, fk=None, **kw):
    return evaluate_formal_prior(
        model if model is not None else CVPrior(),
        bank,
        codec,
        fk if fk is not None else DenseJoints(),
        bootstrap_samples=32,
        **kw,
    )


def test_zero_residual_real_history_prior_is_exactly_cv_and_selection_is_mature_only():
    bank, codec = make_bank(boundary_error=True)
    model = HistoryPrior(width=16, layers=1, heads=4, dropout=0.5, source_flags=True)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = evaluate(bank, codec, model, batch_size=7, include_pa=True, include_parts=True)
    assert report["selection_key"] == report["best_key"] == "mature.mpjpe_mm.P"
    assert report["selection_score"] == pytest.approx(0, abs=1e-4)
    assert report["summary"]["startup"]["metrics"]["mpjpe_mm"]["P"] == pytest.approx(1000, abs=1e-3)
    assert report["summary"]["mature"]["metrics"]["velocity_mm_s"]["P"] == 0
    assert report["summary"]["mature"]["metrics"]["acceleration_mm_s2"]["P"] == 0
    assert report["summary"]["all"]["metrics"]["velocity_mm_s"]["P"] > 0
    assert report["summary"]["all"]["metrics"]["foot_slide_mm"]["P"] > 0
    for phase in report["summary"].values():
        for metric in phase["metrics"].values():
            if metric is not None:
                assert metric["P"] == metric["CV"] and metric["delta"] == 0
                assert metric["delta_ci95"] == [0, 0]
    assert not model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    json.dumps(report, allow_nan=False)


def test_physical_derivatives_and_short_stencils_and_unknown_floor():
    times = torch.arange(5, dtype=torch.float64) / 10
    truth = torch.zeros(5, 22, 3)
    predicted = truth.double()
    predicted[..., 0] = times[:, None].square()
    metrics = formal_center_metrics(predicted, truth, times, False)
    assert metrics["acceleration_mm_s2"] == pytest.approx(2000)
    assert metrics["velocity_mm_s"] == pytest.approx(400)
    assert metrics["foot_slide_mm"] is None
    short = formal_center_metrics(predicted[:1], truth[:1], times[:1], True)
    assert short["velocity_mm_s"] is short["acceleration_mm_s2"] is short["foot_slide_mm"] is None


def test_own_beta_primary_and_shared_cv_beta_auxiliary_are_distinct():
    bank, codec = make_bank()
    bank["tensors"]["truth_xyz"][..., 0] += torch.arange(22)[None] / 100
    report = evaluate(bank, codec, BetaPrior(), ShapeJoints(), include_beta=True, include_shared_beta=True)
    metrics = report["summary"]["mature"]["metrics"]
    assert metrics["mpjpe_mm"]["P"] == pytest.approx(0, abs=1e-3)
    assert metrics["mpjpe_mm"]["CV"] > 100
    assert metrics["shared_beta_mpjpe_mm"]["delta"] == 0
    assert metrics["beta_norm"]["P"] > metrics["beta_norm"]["CV"]


def clip_row(episode, take, delta, phase="mature", *, floor=True):
    cv = dict(mpjpe_mm=100.0, velocity_mm_s=100.0, acceleration_mm_s2=100.0, foot_slide_mm=100.0)
    p = {**cv, "mpjpe_mm": 100.0 + delta}
    if not floor:
        cv["foot_slide_mm"] = p["foot_slide_mm"] = None
    return {"episode": episode, "take": take, "phase": phase, "frames": 40, "floor_calibrated": floor, "P": p, "CV": cv}


def test_clip_equal_means_and_take_cluster_bootstrap_retain_cluster_multiplicity():
    rows = [clip_row("a1", "A", -4), clip_row("a2", "A", -4), clip_row("b", "B", 2)]
    summary = summarize_formal_prior_rows(rows, bootstrap_samples=2000)
    metric = summary["mature"]["metrics"]["mpjpe_mm"]
    assert metric["delta"] == -2  # take-equal would incorrectly yield -1.
    assert metric["P"] == 98 and metric["CV"] == 100
    assert metric["delta_ci95"] == [-4, 2]
    assert metric["clips"] == 3 and metric["takes"] == 2
    assert summary == summarize_formal_prior_rows(list(reversed(rows)), bootstrap_samples=2000)
    # Changing the value on every clip by the same amount changes only the
    # bootstrap effect, never its shared cluster draws.
    shifted = copy.deepcopy(rows)
    for row in shifted:
        row["P"]["mpjpe_mm"] += 5
    shifted_metric = summarize_formal_prior_rows(shifted)["mature"]["metrics"]["mpjpe_mm"]
    assert shifted_metric["bootstrap"] == metric["bootstrap"]
    assert shifted_metric["delta_ci95"] == [1, 7]


def test_unknown_floor_reports_paired_coverage_without_dropping_other_metrics():
    rows = [clip_row("a", "A", -1, floor=False), clip_row("b", "B", -2)]
    summary = summarize_formal_prior_rows(rows, bootstrap_samples=0)["mature"]
    assert summary["clips"] == 2 and summary["floor_calibrated_clips"] == 1
    assert summary["metrics"]["mpjpe_mm"]["clips"] == 2
    assert summary["metrics"]["foot_slide_mm"]["clips"] == 1
    assert summary["metrics"]["foot_slide_mm"]["takes"] == 1


def test_context_empty_rank_gathers_complete_report_without_any_forward():
    bank, codec = make_bank()
    expected = evaluate(bank, codec)

    class EmptyContext:
        rank, world_size = 3, 4

        def gather_rows(self, rows):
            assert rows == []
            return expected["rows"]

    class Forbidden(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("This worker owns no complete clips")

    actual = evaluate(bank, codec, Forbidden(), context=EmptyContext())
    assert actual == expected


@pytest.mark.parametrize(
    "change", ("duplicate", "gap", "irregular_time", "mixed_take", "mixed_window", "old_bank", "missing_time")
)
def test_bad_dense_metadata_is_rejected(change):
    bank, codec = make_bank()
    if change == "duplicate":
        bank["rows"][1]["frame"] = 20
    elif change == "gap":
        bank["rows"][1]["frame"] += 1
    elif change == "irregular_time":
        bank["rows"][1]["timestamp_s"] += 0.03
    elif change == "mixed_take":
        bank["rows"][1]["take"] = "another"
    elif change == "mixed_window":
        bank["rows"][1]["window"] = 1
    elif change == "missing_time":
        del bank["rows"][1]["timestamp_s"]
    else:
        bank["format"] = "egorecover_prior_bank_v1"
    with pytest.raises(ValueError):
        evaluate(bank, codec)


def test_formal_evaluation_rejects_fixed_zero_shape_fk():
    bank, codec = make_bank()
    fk = DenseJoints()
    fk.shape_mode = "bootstrap"
    with pytest.raises(ValueError, match="own beta"):
        evaluate(bank, codec, fk=fk)


def gate_report(delta=-2, startup_delta=0.5, temporal=105.0):
    rows = []
    for phase in ("all", "mature", "startup"):
        row = clip_row("a", "A", startup_delta if phase == "startup" else delta, phase)
        for name in ("velocity_mm_s", "acceleration_mm_s2", "foot_slide_mm"):
            row["P"][name] = temporal
        rows.append(row)
    return {"summary": summarize_formal_prior_rows(rows, bootstrap_samples=64)}


def test_pilot_gate_requires_both_checkpoints_external_checks_and_both_temporal_phases():
    curve = [{"step": 1500, **gate_report(-0.1)}, {"step": 2000, **gate_report(-0.5)}]
    assert pilot_gate(curve, checks_passed=True)["passed"]
    assert not pilot_gate(curve)["passed"]
    assert not pilot_gate(curve[:1], checks_passed=True)["passed"]
    report = curve[1]["summary"]
    report["all"]["metrics"]["velocity_mm_s"] = {"P": 105.01, "CV": 100.0, "delta": 5.01}
    result = pilot_gate(curve, checks_passed=True)
    assert not result["passed"]
    assert result["checks"]["D128_step2000_mature_velocity_mm_s_regression_le5pct"]
    assert "D128_step2000_all_velocity_mm_s_regression_le5pct" in result["failures"]


def test_final_gate_enforces_startup_and_long_all_temporal_constraints():
    d400, long = gate_report(), gate_report(-0.1)
    assert final_gate(d400, long)["passed"]
    bad = copy.deepcopy(d400)
    bad["summary"]["startup"]["metrics"]["mpjpe_mm"] = {"P": 102.0, "CV": 100.0, "delta": 2.0}
    assert not final_gate(bad, long)["passed"]
    bad_long = copy.deepcopy(long)
    bad_long["summary"]["all"]["metrics"]["acceleration_mm_s2"] = {"P": 106.0, "CV": 100.0, "delta": 6.0}
    assert not final_gate(d400, bad_long)["passed"]
    missing_ci = copy.deepcopy(d400)
    missing_ci["summary"]["mature"]["metrics"]["mpjpe_mm"]["delta_ci95"] = None
    assert not final_gate(missing_ci, long)["passed"]


def test_gates_fail_closed_for_missing_floor_and_zero_cv_is_handled_without_division():
    d400, long = gate_report(), gate_report()
    d400["summary"]["mature"]["metrics"]["foot_slide_mm"] = None
    assert not final_gate(d400, long)["passed"]
    good = gate_report()
    good["summary"]["all"]["metrics"]["foot_slide_mm"] = {"P": 0.0, "CV": 0.0, "delta": 0.0}
    assert final_gate(good, long)["passed"]
    good["summary"]["all"]["metrics"]["foot_slide_mm"] = {"P": 1e-9, "CV": 0.0, "delta": 1e-9}
    assert not final_gate(good, long)["passed"]
