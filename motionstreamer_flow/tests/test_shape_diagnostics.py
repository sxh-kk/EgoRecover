"""Keep annotation accuracy distinct from fixed-bootstrap-skeleton accuracy."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from motionstreamer_flow import evaluation
from motionstreamer_flow.data import Episodes
from motionstreamer_flow.demo import make_dataset, tiny_config
from motionstreamer_flow.geometry import fk

METRIC_KEYS = {"body_mm", "root_mm", "head_mm", "pa_body_mm", "velocity_mm_s"}
DIAGNOSTIC_KEYS = ("fixed_bootstrap_shape_metrics", "target_geometry_gap_metrics")


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    cfg = replace(tiny_config(), selection_start=8, selection_stop=20)
    episodes = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    packet = episodes.load(episodes.split("test")[0])
    fixed, _ = fk(packet["pose"], packet["root"], packet["offsets"])
    case = SimpleNamespace(
        flow=SimpleNamespace(cfg=cfg),
        episodes=episodes,
        packet=packet,
        fixed=fixed,
        prediction=fixed.clone(),
        calls=[],
    )

    def fake_rollout(flow, tae, normalizer, inputs, **kwargs):
        assert "pose" not in inputs and "root" not in inputs and "joints" not in inputs
        case.calls.append(len(inputs["images"]))
        return {"joints": case.prediction[cfg.bootstrap :]}

    monkeypatch.setattr(evaluation, "rollout", fake_rollout)
    return case


def report(case):
    return evaluation.evaluate(case.flow, None, None, case.episodes, split="test")


def different_annotation_skeleton(case):
    original_offsets = case.packet["offsets"] * 1.15
    original, _ = fk(case.packet["pose"], case.packet["root"], original_offsets)
    case.packet["joints"] = original
    return original


def assert_signed_difference(result):
    # Differences describe the chosen target; counts are not error metrics.
    top = result["metric_difference_original_minus_fixed_shape"]
    assert set(top) == METRIC_KEYS
    for key in METRIC_KEYS:
        assert top[key] == pytest.approx(result["metrics"][key] - result["fixed_bootstrap_shape_metrics"][key])
    for episode in result["episodes"]:
        difference = episode["metric_difference_original_minus_fixed_shape"]
        assert set(difference) == METRIC_KEYS
        for key in METRIC_KEYS:
            assert difference[key] == pytest.approx(episode[key] - episode["fixed_bootstrap_shape_metrics"][key])


def test_matching_targets_preserve_primary_metrics_and_zero_target_gap(scenario):
    scenario.prediction += torch.tensor([0.01, 0.0, 0.0])
    result = report(scenario)
    assert result["metrics"]["body_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["metrics"]["root_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["metrics"]["head_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["fixed_bootstrap_shape_metrics"] == pytest.approx(result["metrics"])
    assert all(abs(value) < 0.001 for value in result["target_geometry_gap_metrics"].values())
    episode = result["episodes"][0]
    for key in METRIC_KEYS:
        assert episode["fixed_bootstrap_shape_metrics"][key] == pytest.approx(episode[key])
        assert abs(episode["target_geometry_gap_metrics"][key]) < 0.001
    assert_signed_difference(result)
    assert all(abs(value) < 0.001 for value in result["metric_difference_original_minus_fixed_shape"].values())


def test_fixed_skeleton_prediction_can_be_perfect_while_annotation_error_remains(scenario):
    different_annotation_skeleton(scenario)
    result = report(scenario)
    assert result["metrics"]["body_mm"] > 10
    assert result["metrics"]["head_mm"] > 10
    assert result["metrics"]["root_mm"] == pytest.approx(0, abs=0.001)
    assert all(abs(value) < 0.001 for value in result["fixed_bootstrap_shape_metrics"].values())
    assert result["target_geometry_gap_metrics"]["body_mm"] == pytest.approx(result["metrics"]["body_mm"])
    assert result["target_geometry_gap_metrics"]["root_mm"] == pytest.approx(0, abs=0.001)
    assert_signed_difference(result)
    assert result["metric_difference_original_minus_fixed_shape"]["body_mm"] > 0


def test_annotation_perfect_prediction_has_negative_signed_difference(scenario):
    scenario.prediction = different_annotation_skeleton(scenario).clone()
    result = report(scenario)
    assert all(abs(value) < 0.001 for value in result["metrics"].values())
    assert result["fixed_bootstrap_shape_metrics"]["body_mm"] > 10
    assert result["target_geometry_gap_metrics"]["root_mm"] == pytest.approx(0, abs=0.001)
    assert_signed_difference(result)
    assert result["metric_difference_original_minus_fixed_shape"]["body_mm"] < -10


def test_all_targets_share_scoring_slice_gap_mask_and_velocity_boundary(scenario):
    scenario.prediction += torch.tensor([0.01, 0.0, 0.0])
    scenario.packet["valid"][12] = False
    # Deliberately disagree before/after the slice and at the unavailable frame.
    # Neither target diagnostics nor speed may score or bridge these values.
    for frame in (7, 12, 20):
        scenario.packet["joints"][frame] += 500
        scenario.packet["root"][frame, 6:] -= 500
        scenario.prediction[frame] += 1000
    result = report(scenario)
    assert result["frames"] == 11 and result["velocity_pairs"] == 9
    assert result["protocol"]["fps"] == 10
    episode = result["episodes"][0]
    assert episode["scoring_window"] == {"start": 8, "stop": 20}
    assert episode["frames"] == 11 and episode["velocity_pairs"] == 9
    for key in DIAGNOSTIC_KEYS:
        assert episode[key]["frames"] == 11
        assert episode[key]["velocity_pairs"] == 9
        assert set(result[key]) == METRIC_KEYS
    assert result["metrics"]["body_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["fixed_bootstrap_shape_metrics"]["body_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["metrics"]["velocity_mm_s"] < 0.001
    assert result["fixed_bootstrap_shape_metrics"]["velocity_mm_s"] < 0.001
    assert all(abs(value) < 0.001 for value in result["target_geometry_gap_metrics"].values())
    assert_signed_difference(result)
    assert scenario.calls == [24]


def test_single_valid_frame_omits_velocity_for_every_target(scenario):
    different_annotation_skeleton(scenario)
    scenario.packet["valid"][8:20] = False
    scenario.packet["valid"][12] = True
    result = report(scenario)
    position_keys = METRIC_KEYS - {"velocity_mm_s"}
    assert result["frames"] == 1 and result["velocity_pairs"] == 0
    for name in ("metrics", *DIAGNOSTIC_KEYS, "metric_difference_original_minus_fixed_shape"):
        assert set(result[name]) == position_keys
    episode = result["episodes"][0]
    assert episode["frames"] == 1
    assert "velocity_mm_s" not in episode and "velocity_pairs" not in episode
    for name in DIAGNOSTIC_KEYS:
        assert episode[name]["frames"] == 1
        assert "velocity_mm_s" not in episode[name] and "velocity_pairs" not in episode[name]
    assert set(episode["metric_difference_original_minus_fixed_shape"]) == position_keys
    for key in position_keys:
        assert result["metric_difference_original_minus_fixed_shape"][key] == pytest.approx(
            result["metrics"][key] - result["fixed_bootstrap_shape_metrics"][key]
        )


def test_diagnostic_aggregation_uses_frame_and_pair_weights(tmp_path, monkeypatch):
    cfg = replace(tiny_config(), selection_start=8, selection_stop=20)
    episodes = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    records = episodes.split("train")
    predictions = []
    for index, record in enumerate(records):
        packet = episodes.load(record)
        fixed, _ = fk(packet["pose"], packet["root"], packet["offsets"])
        original_offsets = packet["offsets"] * (1.10 + index * 0.10)
        packet["joints"], _ = fk(packet["pose"], packet["root"], original_offsets)
        # Different drift, skeletons and validity counts distinguish weighting rules.
        prediction = fixed.clone()
        prediction[:, :, 0] += torch.arange(len(prediction))[:, None] * (0.001 + index * 0.002)
        if index:
            packet["valid"][12] = False
            packet["valid"][14] = False
        predictions.append(prediction[cfg.bootstrap :])
    pending = iter(predictions)
    monkeypatch.setattr(evaluation, "rollout", lambda *args, **kwargs: {"joints": next(pending)})
    result = evaluation.evaluate(SimpleNamespace(cfg=cfg), None, None, episodes, split="train")
    assert result["frames"] == 22 and result["velocity_pairs"] == 18
    rows = result["episodes"]
    assert rows[0]["frames"] == 12 and rows[1]["frames"] == 10
    assert rows[0]["velocity_pairs"] == 11 and rows[1]["velocity_pairs"] == 7
    for diagnostic in DIAGNOSTIC_KEYS:
        for key in METRIC_KEYS:
            weight = "velocity_pairs" if key == "velocity_mm_s" else "frames"
            expected = sum(row[diagnostic][key] * row[weight] for row in rows) / sum(row[weight] for row in rows)
            assert result[diagnostic][key] == pytest.approx(expected)
    assert_signed_difference(result)
