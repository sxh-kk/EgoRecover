"""Causality, physical coordinates and resume contracts for independent P training."""
import json
import sys

import pytest
import torch

from egorecover.codec import BodyState, MotionCodec
from egorecover.evaluation_protocol import file_sha256
from egorecover.losses import physical_objective
from egorecover.prior import HistoryPrior
from egorecover.prior_adaptation import predict
from egorecover.prior_two_forward import (FIELDS, PriorSequences, candidate_histories, evaluate_rollout,
    mixed_batch, physical_batch, replacement_probability, state_map)
from test_smplx_evaluation import SyntheticLayer


def stats():
    return {"motion_mean": torch.zeros(243), "motion_std": torch.ones(243),
            "traj_mean": torch.zeros(18), "traj_std": torch.ones(18)}


def trajectory(n=1, frames=200):
    reference = torch.eye(4).repeat(n, frames, 1, 1)
    reference[:, :, 0, 3] = torch.arange(frames) * .03
    joints = reference[:, :, None].repeat(1, 1, 22, 1, 1)
    joints[..., :3, 3] += SyntheticLayer().template[:22]
    return BodyState(joints, reference, torch.zeros(n, frames, 36))


def cache(folder):
    folder.mkdir()
    body = trajectory(2)
    identity = {"scope": "audited_prior_train_dev_sequences", "splits": {
        "train": ["train"], "dev": ["dev"], "holdout": ["unseen"]}, "reference_mode": "planar"}
    payload = {"identity": identity, "take_names": ["train", "dev"], "stats": stats(),
               "beta_boot": torch.zeros(2, 10), **{key: getattr(body, key) for key in FIELDS}}
    def save():
        torch.save(payload, folder / "sequences.pt")
        (folder / "report.json").write_text(json.dumps({"completed": True, "identity": identity,
            "cache_sha256": file_sha256(folder / "sequences.pt")}))
    save()
    return payload, save


def test_schedule_and_zero_probability_equivalence():
    assert [replacement_probability(s, 100) for s in (0, 20, 80, 100)] == [0., 0., .5, .5]
    assert replacement_probability(50, 100) == pytest.approx(.25)
    c, full = MotionCodec(stats()), trajectory(frames=41)
    context = state_map(full, lambda x: x[:, :40])
    target = state_map(full, lambda x: x[:, 40])
    class Forbidden:
        def __call__(self, *args):
            raise AssertionError("p=0 must not generate candidates")
    batch, actual = mixed_batch(Forbidden(), c, context, target, torch.zeros(1, 10), probability=0,
                               generator=torch.Generator())
    expected = physical_batch(c, state_map(context, lambda x: x[:, 20:]), target, torch.zeros(1, 10))
    for key in expected:
        torch.testing.assert_close(batch[key], expected[key])
    assert actual == 0


def test_candidates_are_causal_detached_and_restore_training_mode():
    c, context = MotionCodec(stats()), trajectory(frames=40)
    class Watch(HistoryPrior):
        def forward(self, *args):
            assert not torch.is_grad_enabled() and not self.training
            return super().forward(*args)
    model = Watch(width=16, layers=1, heads=4)
    model.train()
    first = candidate_histories(model, c, context, 7)
    assert model.training and not first.joints.requires_grad
    changed = state_map(context, lambda x: x.clone())
    changed.joints[:, 30:, :, 0, 3] += 8
    changed.reference[:, 30:, 0, 3] += 8
    second = candidate_histories(model, c, changed, 9)
    # Predictions for times20..30 cannot see frame30 or later.
    torch.testing.assert_close(first.joints[:, :11], second.joints[:, :11])
    assert not torch.equal(first.joints[:, 11:], second.joints[:, 11:])
    model.eval()
    candidate_histories(model, c, context)
    assert not model.training


def test_mixed_last_reference_reencodes_gt_and_second_forward_has_fk_gradients():
    c, full = MotionCodec(stats()), trajectory(frames=41)
    model = HistoryPrior(width=16, layers=1, heads=4)
    context, target = state_map(full, lambda x: x[:, :40]), state_map(full, lambda x: x[:, 40])
    batch, actual = mixed_batch(model, c, context, target, torch.zeros(1, 10), probability=1,
                               generator=torch.Generator())
    assert actual == 1
    # Zero residual P holds frame38 while true last history frame is39.
    torch.testing.assert_close(batch["previous_reference"], context.reference[:, 38])
    decoded = c.decode_current(batch["target"][:, 0], batch["previous_reference"])
    torch.testing.assert_close(decoded.joints, target.joints)
    assert all(parameter.grad is None for parameter in model.parameters())
    parts = physical_objective(c, predict(model, batch), batch, geometry_weight=1, fk_weight=1,
                               smpl=SyntheticLayer(), return_components=True)
    assert parts["weighted_fk"] > 0
    parts["loss"].backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_cache_rejects_holdout_leak_and_modified_contents(tmp_path):
    payload, save = cache(tmp_path / "data")
    data = PriorSequences(tmp_path / "data", "train")
    assert data.takes == ["train"] and len(data.indices(40)) == 160
    with pytest.raises(ValueError, match="outside"):
        data.examples(torch.tensor([[0, 20]]), "cpu", context_length=40)
    payload["identity"]["splits"]["holdout"] = ["train"]
    save()
    with pytest.raises(ValueError, match="leaks"):
        PriorSequences(tmp_path / "data", "train")
    with (tmp_path / "data/sequences.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="altered"):
        PriorSequences(tmp_path / "data", "train")


def test_rollout_uses_predictions_and_constant_velocity_baseline(tmp_path):
    cache(tmp_path / "data")
    data = PriorSequences(tmp_path / "data", "dev")
    c = MotionCodec(stats())
    hold, poses = evaluate_rollout("hold", data, c, SyntheticLayer(), "cpu")
    cv, _ = evaluate_rollout("constant_velocity", data, c, SyntheticLayer(), "cpu")
    assert len(poses["indices"]) == 9
    assert hold["100"]["mean"] == pytest.approx(30, abs=.001)
    assert hold["1000"]["mean"] == pytest.approx(300, abs=.001)
    assert cv["1000"]["mean"] < .05
    # Alter every scoring target after the first prefix; hold output remains the same.
    original = data.examples
    def poisoned(indices, device, context_length=20):
        history, target, beta = original(indices, device, context_length)
        target.joints[..., 0, 3] += 100
        return history, target, beta
    data.examples = poisoned
    _, changed = evaluate_rollout("hold", data, c, SyntheticLayer(), "cpu")
    torch.testing.assert_close(poses["joints"][10], changed["joints"][10], atol=1e-5, rtol=1e-5)


def test_training_and_resume_cpu_smoke(tmp_path, monkeypatch):
    from run import train_prior_two_forward as runner
    from run.complete_stages import Task, verify_task
    folder, output = tmp_path / "data", tmp_path / "run"
    payload, save = cache(folder)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (asset_dir / "SMPLX_NEUTRAL.npz").write_bytes(b"synthetic asset for test")
    payload["identity"]["smplx_asset_sha256"] = file_sha256(asset_dir / "SMPLX_NEUTRAL.npz")
    save()
    monkeypatch.setenv("SMPLX_MODEL_PATH", str(asset_dir))
    monkeypatch.setattr(runner, "get_smpl", SyntheticLayer)
    monkeypatch.setattr(runner, "HistoryPrior", lambda: HistoryPrior(width=16, layers=1, heads=4))
    monkeypatch.setattr(runner, "append_log", lambda *a: None)
    argv = ["train", "--data", str(folder), "--output", str(output), "--device", "cpu",
            "--steps", "2", "--eval-every", "1", "--batch-size", "2", "--two-forward", "--fk-weight", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    report = verify_task(Task("test", [], output, "prior"))
    assert report["completed"] and len(report["curve"]) == 3
    original = torch.load(output / "prior.pt", weights_only=True)["state_dict"]
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    runner.main()
    after = torch.load(output / "prior.pt", weights_only=True)["state_dict"]
    for key in original:
        torch.testing.assert_close(original[key], after[key], atol=0, rtol=0)


    # An actual interruption after step1 must reproduce uninterrupted step2,
    # including minibatch sampling, history masks and dropout RNG.
    interrupted = tmp_path / "interrupted"
    resumed_argv = [str(interrupted) if item == str(output) else item for item in argv]
    def stop_after_checkpoint(title, lines):
        if title == "独立 P 选点":
            raise RuntimeError("test interruption")
    monkeypatch.setattr(runner, "append_log", stop_after_checkpoint)
    monkeypatch.setattr(sys, "argv", resumed_argv)
    with pytest.raises(RuntimeError, match="test interruption"):
        runner.main()
    monkeypatch.setattr(runner, "append_log", lambda *a: None)
    monkeypatch.setattr(sys, "argv", resumed_argv + ["--resume"])
    runner.main()
    expected = torch.load(output / "resume.pt", weights_only=True)
    actual = torch.load(interrupted / "resume.pt", weights_only=True)
    for key in expected["model"]:
        torch.testing.assert_close(expected["model"][key], actual["model"][key], atol=0, rtol=0)
    assert expected["curve"] == actual["curve"]


@pytest.mark.parametrize("gpus", [(0, 1, 2, 3), (4, 5, 6, 7)])
def test_matched_matrix_and_gpu_assignment(tmp_path, gpus):
    from run.prior_two_forward_experiment import CASES, tasks_for
    tasks = tasks_for(tmp_path, CASES, [62], baselines=True, gpus=gpus)
    assert [task.gpu for task in tasks] == list(gpus)
    assert sum("--two-forward" in task.argv for task in tasks) == 2
    assert all(task.argv[task.argv.index("--seed") + 1] == "62" for task in tasks)
    assert sum("--baselines" in task.argv for task in tasks) == 1
