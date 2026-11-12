import json
from pathlib import Path

import pytest
import torch

from scripts import run_formal_mainline as runner
from scripts.run_formal_mainline import assert_recovered_tensor, command, continuation_step, platform_decision
from scripts.run_history_ablation import load_module


def test_gpu_precision_check_accepts_float32_roundoff_but_rejects_missing_update():
    reference = torch.tensor([0.2, -0.01], dtype=torch.float32)
    roundoff = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert assert_recovered_tensor(reference, roundoff) > 0
    with pytest.raises(AssertionError):
        assert_recovered_tensor(reference, reference + 1e-4)


def test_optimizer_precision_check_rejects_lost_momentum():
    reference = torch.tensor([1e-4, -2e-5])
    with pytest.raises(AssertionError):
        assert_recovered_tensor(reference, torch.zeros_like(reference), optimizer=True)


def test_continuation_budget_uses_saved_optimizer_step_instead_of_unsaved_progress(tmp_path):
    assert continuation_step(tmp_path) == 0
    (tmp_path / "progress.json").write_text('{"step": 1087}')
    torch.save({"step": 1000}, tmp_path / "resume.pt")
    assert continuation_step(tmp_path) == 1000


def test_formal_foundation_cannot_load_old_P_or_G_and_keeps_the_declared_budget():
    plan = {
        "python": "python",
        "data_root": "data",
        "velocity_weight": 0.01341,
        "preflight_selection": "two.json",
        "dev_selection": "128.json",
    }
    job = {"port": 30911, "config": "W1.json"}
    argv = command(plan, job, Path("out"), stop_after=32000)
    assert "--from-scratch" in argv
    assert not {"--initialize", "--prior-checkpoint", "--e7-checkpoint"} & set(argv)
    assert argv[argv.index("--prior-mode") + 1] == "constant_velocity"
    assert argv[argv.index("--stop-after") + 1] == "32000"
    assert argv[argv.index("--train-steps") + 1] == "40000"
    assert argv[argv.index("--lr-decay-step") + 1] == "12000"
    assert argv[argv.index("--dev-selection-file") + 1] == "128.json"
    assert int(argv[argv.index("--batch-size") + 1]) * int(argv[argv.index("--unroll-steps") + 1]) == 128
    probe = command(plan, job, Path("probe"), preflight=True, resume=True, stop_after=1)
    assert "--resume" in probe and probe[probe.index("--dev-selection-file") + 1] == "two.json"


def curve(before, after):
    return [
        {"step": 28000, "selection_body_mm": before, "scores": {"body_mm": {"episode_mean": before}}},
        {"step": 32000, "selection_body_mm": after, "scores": {"body_mm": {"episode_mean": after}}},
    ]


def test_only_a_common_extension_is_allowed_if_either_window_has_not_plateaued():
    decision = platform_decision({"W1": curve(90, 89.3), "W3": curve(90, 89.9)})
    assert decision["extend_both"]
    assert not platform_decision({"W1": curve(90, 89.8), "W3": curve(90, 90.2)})["extend_both"]


def screening_plan(tmp_path):
    return {
        "python": "python",
        "output": str(tmp_path),
        "source_snapshot": str(tmp_path / "source"),
        "source_hashes": {},
        "smpl_directory": str(tmp_path / "smpl"),
        "data_root": "data",
        "velocity_weight": 0.01341,
        "preflight_selection": "two.json",
        "dev_selection": "128.json",
        "training_budgets": [12000],
        "standard_updates": 12000,
        "native_checkpoint_total_cap": 12000,
        "lr_decay_step": 10000,
        "lr_decay_gamma": 0.1,
        "checkpoint_milestones": [10000],
        "training_role": "budget_screening",
        "runtime_revision": "budget-screening-v2",
        "compile_threads_per_rank": 1,
        "jobs": [
            {"name": "W1", "port": 30911, "config": "W1.json", "gpus": "0,1,2,3"},
            {"name": "W3", "port": 30915, "config": "W3.json", "gpus": "4,5,6,7"},
        ],
    }


def test_screening_plan_uses_twelve_thousand_updates_and_late_single_decay(tmp_path):
    plan = screening_plan(tmp_path)
    argv = command(plan, plan["jobs"][0], tmp_path / "W1", resume=True, stop_after=2000)
    values = {flag: argv[argv.index(flag) + 1] for flag in ("--train-steps", "--lr-decay-step", "--lr-decay-gamma")}
    assert values == {"--train-steps": "12000", "--lr-decay-step": "10000", "--lr-decay-gamma": "0.1"}
    assert argv[argv.index("--stop-after") + 1] == "2000"
    assert "--resume" in argv
    probe = command(plan, plan["jobs"][0], tmp_path / "probe", preflight=True)
    assert probe[probe.index("--train-steps") + 1] == "2"
    assert probe[probe.index("--lr-decay-step") + 1] == "1"
    assert probe[probe.index("--lr-decay-gamma") + 1] == "0.1"


def test_compile_threads_follow_plan_for_new_revisions_and_preserve_runtime3_default(tmp_path, monkeypatch):
    plan = screening_plan(tmp_path)
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "99")
    assert runner.environment(plan, "0,1,2,3")["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    del plan["compile_threads_per_rank"]
    plan["runtime_revision"] = "runtime3"
    assert runner.environment(plan, "0,1,2,3")["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    plan["runtime_revision"] = "runtime2"
    assert runner.environment(plan, "0,1,2,3")["TORCHINDUCTOR_COMPILE_THREADS"] == "99"


@pytest.mark.parametrize("milestones", [None, [10000]])
def test_milestone_exports_use_the_plan_and_keep_the_original_default(tmp_path, monkeypatch, milestones):
    plan = {"output": str(tmp_path), "jobs": [{"name": "W1"}]}
    if milestones is not None:
        plan["checkpoint_milestones"] = milestones
    directory = tmp_path / "W1"
    directory.mkdir()
    (directory / "resume.pt").touch()
    (directory / "selection.json").write_text(json.dumps({"curve": [{"step": s} for s in (10000, 12000, 24000)]}))
    exported = []
    monkeypatch.setattr(
        runner, "export_checkpoint", lambda path, target, expected_step: exported.append((target.name, expected_step))
    )
    runner.checkpoint_milestones(plan)
    expected = (12000, 24000) if milestones is None else milestones
    assert exported == [(f"ema-step{step}.pt", step) for step in expected]


def test_status_uses_screening_budget_and_role_instead_of_legacy_cap(tmp_path):
    plan = screening_plan(tmp_path)
    runner.run_jobs(plan, [], "preflight-status-check")
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["standard_budget"] == status["conditional_cap"] == 12000
    assert status["training_budgets"] == [12000]
    assert status["training_role"] == "budget_screening"


def test_screening_finishes_once_then_runs_both_fixed_evaluations_without_extending_or_starting_P(
    tmp_path, monkeypatch
):
    plan = screening_plan(tmp_path)
    (tmp_path / "preflight-verification.json").write_text('{"status":"passed"}')
    for job in plan["jobs"]:
        directory = tmp_path / job["name"]
        directory.mkdir()
        (directory / "resume.pt").touch()
    monkeypatch.setattr(runner, "continuation_step", lambda path: 10000)

    def reject_extension(*args, **kwargs):
        raise AssertionError("A single screening budget must not enter the extension decision.")

    monkeypatch.setattr(runner, "platform_decision", reject_extension)
    phases, exports, collections = [], [], []
    monkeypatch.setattr(runner, "run_jobs", lambda plan, specs, phase: phases.append((phase, specs)))
    monkeypatch.setattr(runner, "export_checkpoint", lambda path, target: exports.append((path, target)))
    monkeypatch.setattr(runner, "collect", lambda plan, long: collections.append(long))
    runner.supervise(plan)
    assert [phase for phase, _ in phases] == [
        "training-to12000-budget-screening-v2",
        "evaluation400",
        "evaluation-long",
    ]
    assert len(phases[0][1]) == 2
    for _, argv, _, _ in phases[0][1]:
        assert argv[argv.index("--stop-after") + 1] == "2000"
        assert argv[argv.index("--train-steps") + 1] == "12000"
        assert "--resume" in argv
    assert all(len(specs) == 8 for _, specs in phases[1:])
    assert collections == [False, True]
    assert len(exports) == 2 and all(target.name == "final-ema.pt" for _, target in exports)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "budget_screening_complete_awaiting_review"
    assert status["training_budgets"] == [12000]
    assert status["standard_budget"] == status["conditional_cap"] == 12000
    assert status["P_training_started"] is False


@pytest.mark.parametrize("budgets", [[], [0], [12000, 10000], [12000, 40000], [12000, 13000, 14000]])
def test_invalid_budget_plans_cannot_silently_exceed_the_declared_cap(tmp_path, budgets):
    plan = {**screening_plan(tmp_path), "training_budgets": budgets}
    with pytest.raises(ValueError, match="native cap"):
        runner.training_budgets(plan)


def test_default_budget_sequence_remains_legacy_standard_then_conditional_extension():
    assert runner.training_budgets({}) == (32000, 40000)


def test_formal_collector_helper_is_loadable_and_keeps_clip_weighting_and_take_clusters():
    helper = load_module(Path(runner.__file__).with_name("formal_paired_helper.py"))
    values = [("A", 1.0), ("A", 3.0), ("B", 8.0)]
    expected = {"delta": 4.0, "ci95": [2.0, 8.0], "valid_clips": 3, "takes": 2}
    assert helper.paired_cluster(values) == expected
    assert helper.paired_cluster(values) == expected
