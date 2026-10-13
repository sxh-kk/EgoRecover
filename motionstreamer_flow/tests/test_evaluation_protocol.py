from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from motionstreamer_flow import evaluation
from motionstreamer_flow.data import Episodes, validate_episode
from motionstreamer_flow.demo import make_dataset, tiny_config
from motionstreamer_flow.train import run_training


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def timeline(tmp_path, monkeypatch):
    cfg = replace(tiny_config(), selection_start=40, selection_stop=200)
    episodes = Episodes(make_dataset(tmp_path / "data", cfg, frames=200), cfg)
    packet = episodes.load(episodes.split("test")[0])
    prediction = packet["joints"].clone() + torch.tensor([0.01, 0.0, 0.0])
    calls = []

    def fake_rollout(flow, tae, normalizer, inputs, **kwargs):
        calls.append(len(inputs["images"]))
        return {"joints": prediction[cfg.bootstrap :]}

    monkeypatch.setattr(evaluation, "rollout", fake_rollout)
    return SimpleNamespace(cfg=cfg), episodes, packet, prediction, calls


def test_default_selection_and_test_protocol_match_and_exclude_boundary_pair(timeline):
    flow, episodes, _, prediction, calls = timeline
    first = evaluation.evaluate(flow, None, None, episodes, split="test")
    assert first["frames"] == 160 and first["velocity_pairs"] == 159
    assert first["protocol"]["requested_window"] == {"start": 40, "stop": 200}
    assert first["protocol"]["fps"] == 10
    assert first["episodes"][0]["scoring_window"] == {"start": 40, "stop": 200}
    assert first["episodes"][0]["scoring_time_s"] == {"start": 4.0, "stop_exclusive": 20.0}
    assert first["metrics"]["body_mm"] == pytest.approx(10.0, abs=0.001)
    assert first["metrics"]["velocity_mm_s"] < 0.001
    prediction[39] += 1000
    after = evaluation.evaluate(flow, None, None, episodes, split="test", start=40, stop=200)
    assert after["metrics"] == first["metrics"]
    override = evaluation.evaluate(flow, None, None, episodes, split="test", start=60, stop=100)
    assert override["frames"] == 40 and override["velocity_pairs"] == 39
    assert override["episodes"][0]["scoring_window"] == {"start": 60, "stop": 100}
    assert calls == [200, 200, 200]


def test_invalid_frame_removes_two_velocity_pairs_without_bridging_gap(timeline):
    flow, episodes, packet, prediction, _ = timeline
    packet["valid"][50] = False
    prediction[50] += 1000
    result = evaluation.evaluate(flow, None, None, episodes, split="test")
    assert result["frames"] == 159 and result["velocity_pairs"] == 157
    assert result["metrics"]["body_mm"] == pytest.approx(10.0, abs=0.001)
    assert result["metrics"]["velocity_mm_s"] < 0.001


@pytest.mark.parametrize("start,stop", [(200, 200), (201, 300), (40, 39), (-1, 200), (40.5, 200), (True, 200)])
def test_invalid_or_empty_scoring_window_fails_before_rollout(timeline, start, stop):
    flow, episodes, _, _, calls = timeline
    with pytest.raises(ValueError):
        evaluation.evaluate(flow, None, None, episodes, split="test", start=start, stop=stop)
    assert not calls


def test_standalone_episode_timeline_rejects_relabelled_fps_and_frame_indices(tmp_path):
    cfg = tiny_config()
    episodes = Episodes(make_dataset(tmp_path / "data", cfg), cfg)
    packet = episodes.load(episodes.split("train")[0])
    with pytest.raises(ValueError, match="FPS"):
        validate_episode(packet, replace(cfg, fps=20))
    packet["source_frame_id_30fps"] = torch.arange(len(packet["pose"])) * 3 + 3
    with pytest.raises(ValueError, match="disagree"):
        validate_episode(packet, cfg)


def test_training_saves_selection_window_and_rejects_resume_protocol_change(tmp_path):
    cfg = replace(tiny_config(), total_steps=2, eval_every=1, selection_start=8, selection_stop=16)
    config = tmp_path / "config.json"
    cfg.save(config)
    manifest = make_dataset(tmp_path / "data", cfg)
    common = {"config": config, "manifest": manifest, "device_name": "cpu"}
    tae = run_training(stage="tae", output=tmp_path / "tae", **common)
    checkpoint = run_training(stage="flow", output=tmp_path / "flow", tae_path=tae, stop_after=1, **common)
    saved = torch.load(checkpoint, weights_only=True)
    report = saved["validation"]
    assert report["protocol"]["requested_window"] == {"start": 8, "stop": 16}
    assert report["frames"] == 8 and report["velocity_pairs"] == 7
    assert saved["config"]["selection_start"] == 8 and saved["config"]["selection_stop"] == 16
    replace(cfg, selection_start=9).save(config)
    with pytest.raises(ValueError, match="config"):
        run_training(stage="flow", output=tmp_path / "flow", tae_path=tae, resume=checkpoint, **common)


@pytest.mark.parametrize("fps", [0, -10, float("nan"), float("inf"), True])
def test_invalid_metric_and_configuration_fps_rejected(fps):
    with pytest.raises(ValueError, match="FPS"):
        replace(tiny_config(), fps=fps)
    with pytest.raises(ValueError, match="FPS"):
        evaluation.metrics(torch.zeros(2, 22, 3), torch.zeros(2, 22, 3), fps)
