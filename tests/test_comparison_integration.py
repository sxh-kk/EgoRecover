import pytest
import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.evaluation import evaluate_episode, summarize
from egorecover.models import MotionModel


def test_comparison_distinguishes_take_and_episode_weighting():
    rows = [
        {"take": take, "body_mm": value, "comparison": {"mpjpe_world_mm": value, "mpjre_f": None}}
        for take, value in (("A", 0.0), ("A", 2.0), ("B", 4.0))
    ]
    result = summarize(rows)
    for entry in (result["body_mm"], result["comparison"]["mpjpe_world_mm"]):
        assert entry["mean"] == 2.5
        assert entry["episode_mean"] == 2.0
        assert entry["per_take"] == {"A": 1.0, "B": 4.0}
    assert "mpjre_f" not in result["comparison"]


def test_new_metric_labels_do_not_change_online_outputs():
    torch.set_num_threads(2)
    episode = synthetic_episode()
    config = WindowConfig()
    system = WindowReconstructor(MotionCodec(episode.stats), None, config)
    model = MotionModel(config, ModelConfig.compact()).eval()
    original = evaluate_episode(system, model, episode, DenseJoints(), comparison=True)
    episode.truth.joints[20:, :, 0, 3] += 1.0
    changed = evaluate_episode(system, model, episode, DenseJoints(), comparison=True)
    torch.testing.assert_close(original["predictions"], changed["predictions"], atol=0, rtol=0)
    assert original["comparison"]["mpjpe_world_mm"] != changed["comparison"]["mpjpe_world_mm"]
    assert original["comparison"]["mpjpe_world_mm"] == pytest.approx(original["body_mm"], abs=0.002)
    assert original["observation_wait_ms"] == 0


def test_runtime_p95_and_fps_use_pooled_frames():
    rows = [
        {"take": "A", "steady_frame_latencies_ms": [10.0] * 100, "frame_interval_ms": 100.0},
        {"take": "B", "steady_frame_latencies_ms": [100.0] * 100, "frame_interval_ms": 100.0},
    ]
    result = summarize(rows)
    assert result["latency_ms"]["mean"] == 55.0
    assert result["p95_ms"]["mean"] == 100.0
    assert result["reconstruction_fps"]["mean"] == pytest.approx(1000 / 55)
    assert result["reconstruction_fps"]["mean"] != 55.0
    assert result["p95_ms"]["aggregation"] == "pooled_steady_frames"
