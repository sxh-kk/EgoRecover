import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from motionstreamer_flow.data import Episodes, save_json, sha256
from motionstreamer_flow.demo import make_dataset, tiny_config
from motionstreamer_flow.evaluation import metrics, pa_align
from motionstreamer_flow.train import load_bundle, resolve_device, run_training


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_metrics_units_and_procrustes():
    truth = torch.randn(4, 22, 3)
    translated = truth + torch.tensor([0.01, 0.0, 0.0])
    result = metrics(translated, truth, 10)
    assert abs(result["body_mm"] - 10) < 0.001
    assert abs(result["root_mm"] - 10) < 0.001
    assert result["pa_body_mm"] < 0.001 and result["velocity_mm_s"] < 0.001
    aligned = pa_align(truth * 2 + 5, truth)
    torch.testing.assert_close(aligned, truth, atol=1e-5, rtol=1e-5)


def test_split_leakage_and_changed_payload_are_rejected(tmp_path):
    cfg = tiny_config()
    manifest = make_dataset(tmp_path / "data", cfg)
    data = json.loads(manifest.read_text())
    data["records"][-1]["take"] = data["records"][0]["take"]
    save_json(manifest, data)
    with pytest.raises(ValueError, match="leakage"):
        Episodes(manifest, cfg)
    data["records"][-1]["take"] = "unique_test_take"
    save_json(manifest, data)
    episodes = Episodes(manifest, cfg)
    record = episodes.split("train")[0]
    path = episodes.resolve(record)
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        episodes.load(record)


def test_path_escape_is_rejected(tmp_path):
    cfg = tiny_config()
    manifest = make_dataset(tmp_path / "data", cfg)
    data = json.loads(manifest.read_text())
    external = tmp_path / "outside.pt"
    external.write_bytes(b"outside")
    data["records"][0]["path"] = "../outside.pt"
    save_json(manifest, data)
    with pytest.raises(ValueError, match="inside"):
        Episodes(manifest, cfg)


def assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_both_training_stages_resume_identically_and_preserve_sources(tmp_path, monkeypatch):
    cfg = tiny_config()
    cfg_path = tmp_path / "config.json"
    cfg.save(cfg_path)
    manifest = make_dataset(tmp_path / "data", cfg)
    original = {path: sha256(path) for path in (tmp_path / "data").rglob("*") if path.is_file()}
    common = {"config": cfg_path, "manifest": manifest, "device_name": "cpu"}
    tae_full = run_training(stage="tae", output=tmp_path / "tae_full", **common)
    tae_partial = run_training(stage="tae", output=tmp_path / "tae_partial", stop_after=1, **common)
    tae_resumed = run_training(stage="tae", output=tmp_path / "tae_partial", resume=tae_partial, **common)
    a, b = (torch.load(path, weights_only=True) for path in (tae_full, tae_resumed))
    assert_state_equal(a["tae"], b["tae"])
    assert a["best"] == b["best"]
    flow_full = run_training(stage="flow", output=tmp_path / "flow_full", tae_path=tae_full, **common)
    flow_partial = run_training(
        stage="flow", output=tmp_path / "flow_partial", tae_path=tae_full, stop_after=1, **common
    )
    flow_resumed = run_training(
        stage="flow", output=tmp_path / "flow_partial", tae_path=tae_full, resume=flow_partial, **common
    )
    a, b = (torch.load(path, weights_only=True) for path in (flow_full, flow_resumed))
    assert_state_equal(a["flow"], b["flow"])
    assert_state_equal(a["normalizer"], b["normalizer"])
    assert a["best"] == b["best"]
    assert torch.equal(a["cpu_rng"], b["cpu_rng"])
    assert torch.equal(a["data_rng"], b["data_rng"])
    assert original == {path: sha256(path) for path in original}
    assert len(a["runtime"]["sha256"]) == 9
    changed_runtime = {**a["runtime"], "sha256": {**a["runtime"]["sha256"], "model.py": "0" * 64}}
    with monkeypatch.context() as patch:
        patch.setattr("motionstreamer_flow.train.runtime_identity", lambda: changed_runtime)
        with pytest.raises(ValueError, match="runtime"):
            run_training(stage="flow", output=tmp_path / "flow_partial", tae_path=tae_full,
                         resume=flow_resumed, **common)
    cfg2 = replace(cfg, total_steps=5)
    cfg2.save(tmp_path / "changed.json")
    with pytest.raises(ValueError, match="config"):
        run_training(
            stage="flow",
            config=tmp_path / "changed.json",
            manifest=manifest,
            device_name="cpu",
            output=tmp_path / "flow_partial",
            tae_path=tae_full,
            resume=flow_resumed,
        )


def test_cuda_request_does_not_silently_fallback():
    if torch.cuda.is_available():
        assert resolve_device("cuda", "fp32").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda", "bf16")


def test_unindexed_cuda_is_resolved_before_selecting_gpu(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(device))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bool(selected) and selected[-1].index == 1)
    device = resolve_device("cuda", "bf16")
    assert device == torch.device("cuda:1") and selected == [device]


def test_directory_can_run_without_legacy_repository(tmp_path):
    package = Path(__file__).resolve().parents[1]
    independent = tmp_path / "independent"
    shutil.copytree(
        package, independent / "motionstreamer_flow", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "motionstreamer_flow.demo", "--device", "cpu", "--output", str(tmp_path / "smoke")],
        cwd=independent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "smoke" / "smoke_results.json").read_text())
    assert set(report["results"]) == {"a11", "a10", "a01", "a00"}
    assert all(value["backend"] == "synthetic_demo" for value in report["results"].values())
    checkpoint = tmp_path / "smoke" / "flow" / "best.pt"
    manifest = tmp_path / "smoke" / "data" / "manifest.json"
    prediction_path = tmp_path / "prediction.pt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "motionstreamer_flow.infer",
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--episode",
            "synthetic_3",
            "--output",
            str(prediction_path),
        ],
        cwd=independent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    prediction = torch.load(prediction_path, weights_only=True)
    assert prediction["prediction"]["joints"].shape == (20, 22, 3)
    assert prediction["first_frame"] == 4
