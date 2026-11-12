import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from egorecover.io import file_sha256, save_checkpoint, save_json
from egorecover.optimization import make_optimizer, make_scheduler, optimization_protocol
from scripts.revise_training_budget import (
    assert_retained_training_state,
    capture_boundary,
    complete_training_rows,
    install_continuations,
    migrate_budget_checkpoint,
    wait_and_continue,
)
from scripts.schedule_formal_runtime import verified_source


def fixture():
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
    options = SimpleNamespace(learning_rate=3e-5, lr_decay_step=8, lr_decay_gamma=0.3)
    protocol = optimization_protocol(options)
    optimizer = make_optimizer([parameter], protocol)
    scheduler = make_scheduler(optimizer, protocol)
    for step in range(1, 5):
        parameter.grad = torch.tensor([step / 10, -step / 20], dtype=torch.float64)
        optimizer.step()
        scheduler.step()
    code = {"training.py": "unchanged"}
    saved = {
        "format": "test_native",
        "identity": {"code": code, "train_steps": 12, "learning_rate": 3e-5, "optimization": protocol},
        "step": 4,
        "model": {"weight": parameter.detach().clone()},
        "ema": {"weight": parameter.detach().clone() * 0.9},
        "feedback_model": {"weight": torch.ones(2)},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rank_rng_states": [{"cpu_rng": torch.get_rng_state()}],
        "curve": [{"step": 4, "selection_body_mm": 95.0}],
        "protocol": {"window": 3, "unroll_steps": 8},
    }
    return saved, code


def test_revision_changes_only_authorized_budget_and_boundary_LR_fields(tmp_path):
    original, code = fixture()
    revised = migrate_budget_checkpoint(original, expected_code=code, total_updates=6, decay_step=4, gamma=0.1)
    torch.save(revised, tmp_path / "resume.pt")
    restored = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert_retained_training_state(original, restored)
    assert original["identity"]["train_steps"] == 12
    assert original["optimizer"]["param_groups"][0]["lr"] == 3e-5
    assert restored["identity"]["train_steps"] == 6
    assert restored["identity"]["learning_rate"] == 3e-5
    assert restored["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-6)
    assert restored["scheduler"]["last_epoch"] == 4
    assert restored["scheduler"]["_step_count"] == original["scheduler"]["_step_count"]
    assert restored["scheduler"]["base_lrs"] == [3e-5]
    for key in ("model", "ema", "feedback_model"):
        assert all(torch.equal(restored[key][name], value) for name, value in original[key].items())


def test_resumed_training_exactly_matches_new_schedule_without_resetting_Adam_moments():
    original, code = fixture()
    revised = migrate_budget_checkpoint(original, expected_code=code, total_updates=6, decay_step=4, gamma=0.1)
    protocol = optimization_protocol(SimpleNamespace(learning_rate=3e-5, lr_decay_step=4, lr_decay_gamma=0.1))
    expected = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
    expected_opt = make_optimizer([expected], protocol)
    expected_lr = make_scheduler(expected_opt, protocol)
    expected_rates = []
    for step in range(1, 7):
        expected_rates.append(expected_opt.param_groups[0]["lr"])
        expected.grad = torch.tensor([step / 10, -step / 20], dtype=torch.float64)
        expected_opt.step()
        expected_lr.step()
    actual = torch.nn.Parameter(revised["model"]["weight"].clone())
    actual_opt = make_optimizer([actual], protocol)
    actual_lr = make_scheduler(actual_opt, protocol)
    actual_opt.load_state_dict(revised["optimizer"])
    actual_lr.load_state_dict(revised["scheduler"])
    actual_rates = []
    for step in range(5, 7):
        actual_rates.append(actual_opt.param_groups[0]["lr"])
        actual.grad = torch.tensor([step / 10, -step / 20], dtype=torch.float64)
        actual_opt.step()
        actual_lr.step()
    assert actual_rates == expected_rates[4:] == [3e-6, 3e-6]
    assert torch.equal(actual, expected)
    assert actual_lr.state_dict() == expected_lr.state_dict()
    for key, value in expected_opt.state_dict()["state"][0].items():
        assert torch.equal(actual_opt.state_dict()["state"][0][key], value)


@pytest.mark.parametrize("corruption", ["moment", "seed", "scheduler_epoch", "weight_decay", "code"])
def test_retained_state_audit_rejects_unrelated_changes(corruption):
    original, code = fixture()
    revised = deepcopy(
        migrate_budget_checkpoint(original, expected_code=code, total_updates=6, decay_step=4, gamma=0.1)
    )
    if corruption == "moment":
        revised["optimizer"]["state"][0]["exp_avg"].add_(1)
    elif corruption == "seed":
        revised["rank_rng_states"][0]["cpu_rng"][0] += 1
    elif corruption == "scheduler_epoch":
        revised["scheduler"]["last_epoch"] = 0
    elif corruption == "weight_decay":
        revised["optimizer"]["param_groups"][0]["weight_decay"] = 0
    else:
        revised["identity"]["code"]["training.py"] = "other"
    with pytest.raises(AssertionError):
        assert_retained_training_state(original, revised)


def test_budget_revision_refuses_already_decayed_or_wrong_boundary_prefix():
    original, code = fixture()
    with pytest.raises(ValueError, match="first new decay boundary"):
        migrate_budget_checkpoint(original, expected_code=code, total_updates=6, decay_step=3, gamma=0.1)
    with pytest.raises(ValueError, match="same reviewed"):
        migrate_budget_checkpoint(
            original, expected_code={"training.py": "different"}, total_updates=6, decay_step=4, gamma=0.1
        )
    original["optimizer"]["param_groups"][0]["lr"] = 1e-5
    with pytest.raises(ValueError, match="already changed LR"):
        migrate_budget_checkpoint(original, expected_code=code, total_updates=6, decay_step=4, gamma=0.1)


def test_interrupted_unsaved_tail_is_ignored_but_corrupt_complete_logs_are_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"step": 9999}\n{"step": 10000}\n{"step": 1000')
    assert [row["step"] for row in complete_training_rows(path)] == [9999, 10000]
    path.write_text('{"step": 9999}\n{"step": broken}\n')
    with pytest.raises(ValueError):
        complete_training_rows(path)


def boundary_fixture(tmp_path):
    source = tmp_path / "source"
    package = source / "egorecover"
    package.mkdir(parents=True)
    (package / "training.py").write_text("same native package")
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old = {
        "output": str(old_root),
        "source_snapshot": str(source),
        "source_hashes": {"egorecover/training.py": file_sha256(package / "training.py")},
        "jobs": [{"name": f"oldW{w}", "window": w, "gpus": str(w)} for w in (1, 3)],
    }
    new = {
        **old,
        "output": str(new_root),
        "jobs": [{**job, "name": job["name"].replace("old", "new")} for job in old["jobs"]],
        "standard_updates": 6,
        "native_checkpoint_total_cap": 6,
        "training_budgets": [6],
        "lr_decay_step": 4,
        "lr_decay_gamma": 0.1,
        "learning_rates": {"1..4": 3e-5, "5..6": 3e-6},
        "workspace": str(tmp_path),
    }
    captured = new_root / "parent-boundary"
    originals = {}
    for job in old["jobs"]:
        directory = old_root / job["name"]
        directory.mkdir(parents=True)
        saved, _ = fixture()
        saved["identity"]["code"] = verified_source(old)
        saved["protocol"]["window"] = job["window"]
        originals[job["name"]] = deepcopy(saved)
        save_checkpoint(directory / "resume.pt", saved)
        save_checkpoint(directory / "best.pt", {"identity": saved["identity"], "ema": saved["ema"], "step": 3})
        (directory / "identity.json").write_text(json.dumps(saved["identity"]))
        (directory / "selection.json").write_text(json.dumps({"curve": saved["curve"]}))
        (directory / "train.jsonl").write_text(
            '{"step": 3}\n{"step": 4, "learning_rate": 3e-5, "next_learning_rate": 3e-5}\n'
        )
        assert capture_boundary(directory, captured / job["name"], 4)
        saved["step"] = 5
        save_checkpoint(directory / "resume.pt", saved)
        (directory / "progress.json").write_text('{"step": 5}')
        assert capture_boundary(directory, captured / job["name"], 4)
        assert torch.load(captured / job["name"] / "resume.pt", weights_only=True)["step"] == 4
    return old, new, captured, originals


def test_two_window_file_handoff_keeps_parent_artifacts_and_pins_overwritten_checkpoint(tmp_path):
    old, new, captured, originals = boundary_fixture(tmp_path)
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    records = install_continuations(old, new, 4, captured_root=captured)
    assert len(records) == 2
    for old_job, new_job in zip(old["jobs"], new["jobs"]):
        output = new_root / new_job["name"]
        restored = torch.load(output / "resume.pt", weights_only=True)
        assert_retained_training_state(originals[old_job["name"]], restored)
        assert restored["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-6)
        assert torch.load(old_root / old_job["name"] / "resume.pt", weights_only=True)["step"] == 5
        assert torch.load(output / "best.pt", weights_only=True)["identity"]["train_steps"] == 12
        assert json.loads((output / "progress.json").read_text())["next_learning_rate"] == 3e-5
    assert (new_root / "preflight-verification.json").exists()


def test_orchestrator_uses_captured_boundary_and_starts_only_the_compressed_budget(tmp_path, monkeypatch):
    old, new, _, _ = boundary_fixture(tmp_path)
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    save_json(old_root / "plan.json", old)
    save_json(new_root / "plan.json", new)
    save_json(old_root / "status.json", {"phase": "training", "jobs": [{"exit_code": None}]})
    actions = []

    def stop(plan, state):
        assert plan == old and state["phase"] == "training"
        actions.append("stop")

    def start(plan, path, suffix):
        assert plan == new and suffix == "budget12k"
        assert plan["training_budgets"] == [6]
        actions.append("start")
        for job in plan["jobs"]:
            directory = new_root / job["name"]
            saved = torch.load(directory / "resume.pt", weights_only=True)
            assert saved["step"] == 4
            save_json(
                directory / "progress.json", {"step": 5, "learning_rate": saved["optimizer"]["param_groups"][0]["lr"]}
            )
        return SimpleNamespace(pid=123, poll=lambda: None)

    monkeypatch.setattr("scripts.revise_training_budget.stop_jobs", stop)
    monkeypatch.setattr("scripts.revise_training_budget.start_supervisor", start)
    args = SimpleNamespace(parent_plan=old_root / "plan.json", revised_plan=new_root / "plan.json", checkpoint_step=4)
    wait_and_continue(args)
    assert actions == ["stop", "start"]
    state = json.loads((new_root / "budget-revision-status.json").read_text())
    assert state["phase"] == "budget12k_training_running" and state["auto_extension"] is False
    assert state["learning_rate"] == 3e-6
    assert json.loads((old_root / "status.json").read_text())["phase"] == "continued_under_budget_revision"
    assert (tmp_path / "docs/reports/data/formal_budget12k_revision_20261112.json").exists()
