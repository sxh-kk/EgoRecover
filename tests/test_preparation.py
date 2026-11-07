import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from egorecover import ModelConfig, WindowConfig, WindowReconstructor
from egorecover.cli import main
from egorecover.codec import MotionCodec
from egorecover.config import TrainOptions
from egorecover.data import TrainingBatch, manifest_inventory
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.evaluation import paired_anchor_diagnostics, paired_horizons
from egorecover.losses import frozen_ema, training_objective
from egorecover.models import MotionModel


def test_prepare_produces_plan_and_pending_requirements_without_launching(tmp_path, monkeypatch):
    import egorecover.preparation as preparation

    monkeypatch.setattr(preparation, "smpl_path", lambda: tmp_path / "missing_smpl.npz")
    output = tmp_path / "plan"
    main(
        [
            "prepare",
            "--output",
            str(output),
            "--assets-root",
            str(tmp_path / "assets"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
        ]
    )
    report = json.loads((output / "preparation.json").read_text())
    assert report["status"] == "needs_assets_or_environment"
    assert report["experiments_started"] is False
    assert {item["kind"] for item in report["issues"]} >= {"missing_manifest", "missing_smplx"}
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["history_frames"] == 20 and protocol["reference_label_writes"] == 1
    jobs = json.loads((output / "jobs.json").read_text())
    assert all(job["started"] is False for job in jobs)
    assert all(job["argv"] is None for job in jobs if job["reuse"])
    assert all(
        job["argv"][:5] == ["conda", "run", "-n", "egorecover", "--no-capture-output"] for job in jobs if job["argv"]
    )
    initial_jobs = [job for job in jobs if job["argv"] and "--initialize" not in job["argv"]]
    assert initial_jobs and all("--from-scratch" in job["argv"] for job in initial_jobs)
    assert all("--e7-checkpoint" not in job["argv"] for job in jobs if job["argv"])
    assert protocol["g_initialization"] == "scratch"
    assert protocol["entire_system_from_scratch"] is False
    assert "pretrained/e7.ckpt" not in report["models"]
    assert not list(output.rglob("*.pt"))


@pytest.mark.parametrize("initialization", ["scratch", "e7"])
def test_prepare_requires_e7_only_when_selected_and_preserves_continuation_dependencies(
    tmp_path, monkeypatch, initialization
):
    import egorecover.preparation as preparation

    data = preparation.asset_registry()
    assets = tmp_path / "assets"
    prior_path = assets / "pretrained/prior.pt"
    prior_path.parent.mkdir(parents=True)
    payload = b"checked prior"
    prior_path.write_bytes(payload)
    prior_item = next(item for item in data["files"] if item["path"] == "pretrained/prior.pt")
    prior_item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(preparation, "asset_registry", lambda: data)
    smpl = tmp_path / "licensed-model.npz"
    smpl.write_bytes(b"asset identity; decoding audited separately")
    monkeypatch.setattr(preparation, "smpl_path", lambda: smpl)
    monkeypatch.setattr(preparation, "load_smpl", lambda _: object())
    data_root = tmp_path / "dataset"
    (data_root / "data").mkdir(parents=True)
    (data_root / "data/manifest.json").write_text("{}")
    audits = {split: {"available_episodes": 1, "total_episodes": 1, "audited": []} for split in ("train", "dev")}
    monkeypatch.setattr(
        preparation, "audit_available_episodes", lambda *_, **__: ({"sha256": "verified-manifest"}, audits)
    )
    args = SimpleNamespace(
        output=tmp_path / "plan",
        assets_root=assets,
        data_root=data_root,
        device="cpu",
        window=3,
        check_episodes=0,
        batch_size=64,
        microbatch_size=8,
        train_steps=2400,
        initialization=initialization,
    )
    report = preparation.prepare(args)
    assert report["models"]["pretrained/prior.pt"]["verified"] is True
    assert report["g_initialization"] == initialization
    assert "train-dev" in report["downloads"]
    if initialization == "scratch":
        assert report["status"] == "ready_for_training"
        assert report["issues"] == []
        assert set(report["models"]) == {"pretrained/prior.pt"}
    else:
        assert report["status"] == "needs_assets_or_environment"
        assert report["issues"] == [{"kind": "missing_or_changed_model", "path": str(assets / "pretrained/e7.ckpt")}]
    protocol = json.loads((args.output / "protocol.json").read_text())
    assert protocol["g_initialization"] == initialization
    assert protocol["predictor_initialization"] == "published_calibrated_prior"
    assert protocol["bootstrap_history_source"] == "published_teacher_predictions"
    assert protocol["bootstrap_shape_source"] == "published_beta_boot"
    assert protocol["entire_system_from_scratch"] is False
    jobs = json.loads((args.output / "jobs.json").read_text())
    continued = [job for job in jobs if job["argv"] and "--initialize" in job["argv"]]
    initial = [job for job in jobs if job["argv"] and "--initialize" not in job["argv"]]
    assert continued and initial
    assert all(job["depends_on"] and "--from-scratch" not in job["argv"] for job in continued)
    assert all("--e7-checkpoint" not in job["argv"] for job in continued)
    required_flag = "--from-scratch" if initialization == "scratch" else "--e7-checkpoint"
    excluded_flag = "--e7-checkpoint" if initialization == "scratch" else "--from-scratch"
    assert all(required_flag in job["argv"] and excluded_flag not in job["argv"] for job in initial)


def test_manifest_inventory_reports_whole_splits_without_episode_reads(tmp_path):
    (tmp_path / "data").mkdir()
    records = [
        {"episode_id": str(i), "base_take_name": f"take{i}", "group": group, "num_frames": 200}
        for i, group in enumerate(("train", "train", "dev", "test"))
    ]
    (tmp_path / "data/manifest.json").write_text(json.dumps({"records": records}))
    _, inventory = manifest_inventory(tmp_path)
    assert inventory["splits"]["train"] == {"episodes": 2, "takes": 2, "frames": 400}
    assert inventory["splits"]["dev"]["episodes"] == 1
    records[-1]["base_take_name"] = "take0"
    (tmp_path / "data/manifest.json").write_text(json.dumps({"records": records}))
    with pytest.raises(ValueError, match="leakage"):
        manifest_inventory(tmp_path)


def test_training_batch_matches_mean_of_individual_paired_objectives():
    torch.set_num_threads(2)
    episode = synthetic_episode()
    batch = TrainingBatch((episode, episode))
    batch.cache(20).validate()
    cfg = WindowConfig(feedback="paired")
    system = WindowReconstructor(MotionCodec(episode.stats), None, cfg)
    torch.manual_seed(62)
    model = MotionModel(cfg, ModelConfig.compact())
    teacher = frozen_ema(model)
    noise = batch.noise(62, 20, 3, tag="test")
    future = batch.noise(62, 21, 3, tag="future")[None]
    tau = torch.tensor([0.3, 0.7])
    joint = training_objective(
        system,
        model,
        teacher,
        batch.cache(20),
        batch.observations(20),
        batch.target(20, 3),
        noise,
        tau,
        DenseJoints(),
        next_observations=batch.observations(21),
        next_epsilon=future,
    )
    individual = []
    for i in range(2):
        value = training_objective(
            system,
            model,
            teacher,
            episode.cache(20),
            episode.observations(20),
            episode.target(20, 3),
            noise[i : i + 1],
            tau[i : i + 1],
            DenseJoints(),
            next_observations=episode.observations(21),
            next_epsilon=future[:, i : i + 1],
        )
        individual.append(value["loss"])
    torch.testing.assert_close(joint["loss"], torch.stack(individual).mean(), rtol=2e-5, atol=2e-4)
    joint["loss"].backward()
    assert model.output_process.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_later_paired_anchors_never_restart_from_cached_teacher_history(monkeypatch):
    episode = synthetic_episode()
    original = episode.cache
    calls = []

    def cache(frame=20):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(episode, "cache", cache)
    cfg = WindowConfig()
    system = WindowReconstructor(MotionCodec(episode.stats), None, cfg)
    model = MotionModel(cfg, ModelConfig.compact()).eval()
    result = paired_anchor_diagnostics(system, model, episode, DenseJoints(), anchors=(20, 24, 180))
    assert calls == [20]
    assert set(result) == {"20", "24"}
    assert all(set(group) == {"1", "2", "4", "10"} for group in result.values())
    calls.clear()
    paired_horizons(system, model, episode, DenseJoints(), frame=24)
    assert calls == [20]


def test_microbatch_resume_with_partial_last_batch_is_exact(tmp_path):
    from egorecover.training import train

    options = TrainOptions(
        output=tmp_path / "full",
        synthetic=True,
        compact=True,
        train_steps=2,
        batch_size=3,
        microbatch_size=2,
        rollin_frames=0,
        eval_every=1,
    )
    train(options)
    options = replace(options, output=tmp_path / "resumed", stop_after=1)
    train(options)
    options.resume, options.stop_after = True, None
    train(options)
    full = torch.load(tmp_path / "full/resume.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/resume.pt", weights_only=True)
    for group in ("model", "ema", "feedback_model"):
        for key in full[group]:
            torch.testing.assert_close(full[group][key], resumed[group][key], atol=0, rtol=0)
    progress = json.loads((tmp_path / "full/progress.json").read_text())
    assert progress["target_queries_seen"] == 6
