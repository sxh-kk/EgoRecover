import csv
import json
from types import SimpleNamespace

import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.checkpoint import FORMAT
from egorecover.diagnostics import synthetic_episode
from egorecover.models import MotionModel
from egorecover.state import state_map


def evaluation_args(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(62)
    config, model_config = WindowConfig(), ModelConfig.compact()
    model = MotionModel(config, model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": FORMAT,
            "identity": {"synthetic": True, "manifest": "synthetic_diagnostic"},
            "protocol": config.to_dict(),
            "model_config": model_config.to_dict(),
            "stats": synthetic_episode().stats,
            "ema": model.state_dict(),
        },
        checkpoint,
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        output=tmp_path / "evaluation",
        device="cpu",
        synthetic=True,
        data_root=None,
        prior_checkpoint=None,
        split="dev",
        mode="rollout",
        steps=[1],
        anchors=[20],
        draw=1062,
        limit=None,
        save_predictions=False,
    )


def test_skip_paired_evaluates_only_one_continuous_rollout_and_keeps_metrics(tmp_path, monkeypatch):
    import egorecover.evaluate as evaluation

    args = evaluation_args(tmp_path)
    args.skip_paired = True
    monkeypatch.setattr(evaluation, "synthetic_episode", lambda *, device: synthetic_episode(device=device, length=200))

    def forbidden_diagnostics(*args, **kwargs):
        raise AssertionError("Skipping paired diagnostics must avoid the extra rollout queries")

    monkeypatch.setattr(evaluation, "paired_anchor_diagnostics", forbidden_diagnostics)
    original, calls = WindowReconstructor.online, []

    def count_queries(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(WindowReconstructor, "online", count_queries)
    evaluation.evaluate(args)
    report = json.loads((args.output / "report.json").read_text())
    assert len(calls) == 180
    assert report["identity"]["skip_paired"] is True
    assert report["results"]["1"]["paired_horizons"] == []
    assert set(report["results"]["1"]["summary"]) == {"rollout"}
    with (args.output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["mode"] == "rollout" and row["mpjpe_world_mm"] for row in rows)


def test_legacy_evaluation_options_keep_paired_diagnostics_enabled(tmp_path, monkeypatch):
    import egorecover.evaluate as evaluation

    args, calls = evaluation_args(tmp_path), []

    def diagnostics(*args, **kwargs):
        calls.append(kwargs["anchors"])
        return {"20": {"1": {"predicted_mm": 1.0}}}

    monkeypatch.setattr(evaluation, "paired_anchor_diagnostics", diagnostics)
    evaluation.evaluate(args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["identity"]["skip_paired"] is False
    assert calls == [[20]]
    assert report["results"]["1"]["paired_horizons"][0]["anchors"]["20"]


def test_twenty_two_frame_clip_keeps_reconstruction_and_marks_default_paired_diagnostic_unavailable(
    tmp_path, monkeypatch
):
    import egorecover.evaluate as evaluation

    args = evaluation_args(tmp_path)
    fixture = synthetic_episode()
    fixture.truth = state_map(fixture.truth, lambda value: value[:22])
    fixture.teacher = state_map(fixture.teacher, lambda value: value[:22])
    for name in ("head", "images", "head_valid", "image_valid", "timestamps"):
        setattr(fixture, name, getattr(fixture, name)[:22])
    monkeypatch.setattr(evaluation, "synthetic_episode", lambda *, device: fixture)

    def forbidden_diagnostics(*args, **kwargs):
        raise AssertionError("A short clip must report paired diagnostics as unavailable without launching them")

    monkeypatch.setattr(evaluation, "paired_anchor_diagnostics", forbidden_diagnostics)
    evaluation.evaluate(args)
    report = json.loads((args.output / "report.json").read_text())
    result = report["results"]["1"]
    row = result["rows"]["rollout"][0]
    assert report["identity"]["skip_paired"] is False
    assert row["frames"] == 2
    assert row["comparison"]["mpjpe_world_mm"] >= 0
    assert row["comparison"]["velocity_error_mm_s"] is not None
    assert row["comparison"]["acceleration_error_mm_s2"] is None
    assert result["paired_horizons"][0]["status"] == "not_available_short_clip"
    assert result["paired_horizons"][0]["anchors"] == {}
    with (args.output / "metrics.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    assert all(row["episodes"] == "1" and row["mpjpe_world_mm"] for row in csv_rows)


def test_unknown_floor_keeps_geometry_but_excludes_foot_slide_and_reports_csv_population(tmp_path, monkeypatch):
    import egorecover.evaluate as evaluation

    args = evaluation_args(tmp_path)
    args.skip_paired = True
    fixture = synthetic_episode()
    fixture.floor_calibrated = False
    monkeypatch.setattr(evaluation, "synthetic_episode", lambda *, device: fixture)
    evaluation.evaluate(args)
    report = json.loads((args.output / "report.json").read_text())
    result = report["results"]["1"]
    row = result["rows"]["rollout"][0]
    assert row["floor_calibrated"] is False
    for name in ("mpjpe_world_mm", "pa_mpjpe_mm", "mpjre_f", "wrist_pe_mm"):
        assert row["comparison"][name] is not None
        assert result["summary"]["rollout"]["comparison"][name]["available_episodes"] == 1
        assert result["summary"]["rollout"]["comparison"][name]["available_takes"] == 1
    for name in ("foot_slide_uniegomotion_mm", "gt_foot_slide_uniegomotion_mm"):
        assert row["comparison"][name] is None
        assert name not in result["summary"]["rollout"]["comparison"]
    with (args.output / "metrics.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    for entry in csv_rows:
        assert entry["episodes"] == "1" and entry["takes"] == "1"
        assert entry["floor_calibrated_episodes"] == "0"
        assert entry["mpjpe_world_mm"] and entry["pa_mpjpe_mm"]
        assert not entry.get("foot_slide_uniegomotion_mm")


def test_mixed_floor_csv_counts_all_geometry_and_only_calibrated_foot_metrics(tmp_path):
    from egorecover.codec import MotionCodec
    from egorecover.diagnostics import DenseJoints
    from egorecover.evaluate import save_metrics_csv
    from egorecover.evaluation import evaluate_episode, summarize

    torch.set_num_threads(2)
    torch.manual_seed(62)
    known, unknown = synthetic_episode(), synthetic_episode()
    known.take, unknown.take = "known_floor", "unknown_floor"
    unknown.floor_calibrated = False
    config = WindowConfig()
    system = WindowReconstructor(MotionCodec(known.stats), None, config)
    model = MotionModel(config, ModelConfig.compact()).eval()
    rows = [evaluate_episode(system, model, episode, DenseJoints(), comparison=True) for episode in (known, unknown)]
    summary = summarize(rows)
    assert summary["comparison"]["mpjpe_world_mm"]["available_episodes"] == 2
    assert summary["comparison"]["mpjpe_world_mm"]["available_takes"] == 2
    for name in ("foot_slide_uniegomotion_mm", "gt_foot_slide_uniegomotion_mm"):
        assert summary["comparison"][name]["available_episodes"] == 1
        assert summary["comparison"][name]["available_takes"] == 1
        assert set(summary["comparison"][name]["per_take"]) == {"known_floor"}
    report = {
        "identity": {"synthetic": True},
        "results": {"1": {"summary": {"rollout": summary}, "rows": {"rollout": rows}}},
    }
    path = tmp_path / "mixed.csv"
    save_metrics_csv(path, report)
    with path.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    for entry in csv_rows:
        assert entry["episodes"] == "2" and entry["takes"] == "2"
        assert entry["floor_calibrated_episodes"] == "1"
        assert float(entry["foot_slide_uniegomotion_mm"]) == rows[0]["comparison"]["foot_slide_uniegomotion_mm"]
