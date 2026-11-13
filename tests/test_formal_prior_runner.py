import json
import sys
from pathlib import Path

import pytest
import torch

from scripts import run_formal_prior as runner


def plan_at(tmp_path):
    return {
        "output": str(tmp_path),
        "protocol": str(tmp_path / "protocol.json"),
        "protocol_sha256": "protocol-digest",
        "source_snapshot": str(tmp_path / "source"),
        "source_hashes": {"egorecover/formal.py": "frozen-code"},
        "python": "python",
        "data_root": "official-train-val",
        "smpl_directory": "smplx",
        "training_gpus": "0,1,2,3",
        "port": 31071,
        "run_id": "P01_W3G10k_s62",
        "run_directory": str(tmp_path / "P01_W3G10k_s62"),
        "train_bank": str(tmp_path / "train.pt"),
        "val_bank": str(tmp_path / "val.pt"),
        "preflight_train_bank": str(tmp_path / "small-train.pt"),
        "preflight_val_bank": str(tmp_path / "small-val.pt"),
        "pilot_updates": 2000,
        "max_updates": 10000,
        "minimum_free_GiB": 0,
        "bank_jobs": [],
        "preflight_bank_jobs": [],
        "evaluation_jobs": [],
    }


def test_pilot_and_continuation_share_native_cap_and_preserve_optimizer_resume(tmp_path):
    plan = plan_at(tmp_path)
    initial = runner.train_command(plan, tmp_path / "run", stop_after=2000)
    resumed = runner.train_command(plan, tmp_path / "run", resume=True, stop_after=8000)
    for argv in (initial, resumed):
        assert argv[argv.index("--train-steps") + 1] == "10000"
        assert argv[argv.index("--batch-size") + 1] == "64"
        assert argv[argv.index("--microbatch-size") + 1] == "8"
        assert argv[argv.index("--nproc_per_node=4")] == "--nproc_per_node=4"
        assert argv[argv.index("--lr-decay-step") + 1] == "8000"
        assert argv[argv.index("--formal-protocol") + 1] == plan["protocol"]
        assert not {"--initialize", "--prior-checkpoint"} & set(argv)
    assert "--resume" not in initial and "--resume" in resumed
    assert initial[initial.index("--stop-after") + 1] == "2000"
    assert resumed[resumed.index("--stop-after") + 1] == "8000"


def test_preflight_has_its_own_small_banks_and_lr_boundary(tmp_path):
    plan = plan_at(tmp_path)
    argv = runner.train_command(plan, tmp_path / "probe", preflight=True, resume=True, stop_after=1)
    assert "--formal-preflight" in argv and "--resume" in argv
    assert argv[argv.index("--train-steps") + 1] == "2"
    assert argv[argv.index("--lr-decay-step") + 1] == "1"
    assert argv[argv.index("--train-bank") + 1] == plan["preflight_train_bank"]
    assert argv[argv.index("--val-bank") + 1] == plan["preflight_val_bank"]
    assert argv[argv.index("--eval-every") + 1] == "1"
    plan["training_gpus"] = "0,1"
    with pytest.raises(ValueError, match="four GPUs"):
        runner.train_command(plan, tmp_path / "probe")


def test_recoverable_step_does_not_follow_unsaved_progress(tmp_path):
    (tmp_path / "progress.json").write_text('{"step": 2507}')
    assert runner.checkpoint_step(tmp_path) == 0
    torch.save({"step": 2500}, tmp_path / "resume.pt")
    assert runner.checkpoint_step(tmp_path) == 2500


def health(step=2000):
    return {"step": step, "passed": True, "checks": {name: True for name in runner.HEALTH_CHECKS}}


def test_gate_health_cannot_be_carried_by_an_empty_or_stale_success_marker(tmp_path):
    plan = plan_at(tmp_path)
    path = Path(plan["run_directory"]) / "health_checks.json"
    path.parent.mkdir()
    assert not runner.health_passed(plan, 2000)
    for invalid in (
        {"step": 2000, "passed": True, "checks": {}},
        health(1500),
        {**health(), "checks": {"finite": True}},
    ):
        path.write_text(json.dumps(invalid))
        assert not runner.health_passed(plan, 2000)
    path.write_text(json.dumps(health()))
    assert runner.health_passed(plan, 2000)
    invalid = health()
    invalid["checks"]["exact_phase_quota"] = False
    path.write_text(json.dumps(invalid))
    assert not runner.health_passed(plan, 2000)


@pytest.mark.parametrize("passed", [False, True])
def test_supervisor_stops_failed_pilot_or_resumes_same_run_to_exact_cap(tmp_path, monkeypatch, passed):
    plan = plan_at(tmp_path)
    for key in ("train_bank", "val_bank"):
        Path(plan[key]).touch()
    state = {"step": 0}
    calls = []
    monkeypatch.setattr(runner, "verify_identity", lambda plan: None)
    monkeypatch.setattr(runner, "preflight", lambda plan: {"status": "passed"})
    monkeypatch.setattr(runner, "checkpoint_step", lambda directory: state["step"])
    monkeypatch.setattr(runner, "pilot_decision", lambda plan: {"passed": passed})

    def execute(plan, specs, phase):
        calls.append((phase, specs))
        state["step"] += int(specs[0]["argv"][specs[0]["argv"].index("--stop-after") + 1])

    monkeypatch.setattr(runner, "run_processes", execute)
    runner.supervise(plan)
    assert calls[0][0] == "training-to2000"
    status = json.loads((tmp_path / "status.json").read_text())
    if passed:
        assert len(calls) == 2
        assert calls[1][0] == "training-to10000"
        assert "--resume" in calls[1][1][0]["argv"]
        assert state["step"] == 10000
        assert status["phase"] == "training_complete_pending_fixed_evaluation"
        assert status["system_G_gain_established"] is False
    else:
        assert len(calls) == 1 and state["step"] == 2000
        assert status["phase"] == "stopped_gate_failed"
        assert status["auto_restart"] is False


def make_preflight(plan):
    root = Path(plan["output"])
    for key in ("preflight_train_bank", "preflight_val_bank"):
        Path(plan[key]).write_bytes(key.encode())
    saved = {
        "step": 2,
        "identity": {"protocol": "same", "preflight": True},
        "model": {"weight": torch.tensor([0.1, 0.2])},
        "ema": {"weight": torch.tensor([0.1, 0.2])},
        "optimizer": {
            "state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.tensor([0.1, 0.2])}},
            "param_groups": [{"lr": 3e-7}],
        },
        "scheduler": {"last_epoch": 2, "_last_lr": [3e-7]},
        "rank_rng_states": [
            {
                "cpu_rng": torch.tensor([rank, 2], dtype=torch.uint8),
                "cuda_rng": torch.tensor([rank, 3], dtype=torch.uint8),
                "sample_rng": torch.tensor([rank, 4], dtype=torch.uint8),
                "visited_indices": [rank, 4],
                "model_sha256": "allowed-numerical-digest",
            }
            for rank in range(4)
        ],
    }
    for name in ("full", "resumed"):
        directory = root / "preflight" / name
        directory.mkdir(parents=True)
        torch.save(saved, directory / "resume.pt")
        (directory / "health_checks.json").write_text(json.dumps(health(2)))
        rows = [
            {
                "step": step,
                "lr": 3e-5 if step == 1 else 3e-6,
                "loss": 1.0,
                "gradient_norm": 2.0,
                "trace_hash": f"sample-{step}",
            }
            for step in (1, 2)
        ]
        (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return saved


def test_preflight_evidence_binds_real_states_sampling_lr_and_bank_identities(tmp_path):
    plan = plan_at(tmp_path)
    make_preflight(plan)
    report = runner.verify_preflight(plan)
    assert report["status"] == "passed" and report["ranks"] == 4
    assert report["protocol_sha256"] == plan["protocol_sha256"]
    assert report["preflight_bank_hashes"]["preflight_train_bank"] == runner.sha256(plan["preflight_train_bank"])
    Path(plan["preflight_train_bank"]).write_bytes(b"changed-bank")
    with pytest.raises(ValueError, match="bank changed"):
        runner.preflight(plan)


def test_preflight_rejects_lost_rng_even_when_final_weights_are_identical(tmp_path):
    plan = plan_at(tmp_path)
    saved = make_preflight(plan)
    saved["rank_rng_states"][3]["sample_rng"] = torch.tensor([99], dtype=torch.uint8)
    torch.save(saved, tmp_path / "preflight/resumed/resume.pt")
    with pytest.raises(AssertionError, match="RNG"):
        runner.verify_preflight(plan)


def test_cpu_rank_preflight_cannot_be_marked_as_a_real_four_gpu_preflight(tmp_path):
    plan = plan_at(tmp_path)
    saved = make_preflight(plan)
    for row in saved["rank_rng_states"]:
        row["cuda_rng"] = None
    for kind in ("full", "resumed"):
        torch.save(saved, tmp_path / "preflight" / kind / "resume.pt")
    with pytest.raises(AssertionError, match="actual CUDA RNG"):
        runner.verify_preflight(plan)


def test_preflight_rejects_different_actual_sampling_trace(tmp_path):
    plan = plan_at(tmp_path)
    make_preflight(plan)
    path = tmp_path / "preflight/resumed/train.jsonl"
    path.write_text(path.read_text().replace("sample-2", "other-sample"))
    with pytest.raises(AssertionError):
        runner.verify_preflight(plan)


def test_bank_generation_stages_wait_for_outputs_and_skip_completed_artifacts(tmp_path, monkeypatch):
    plan = plan_at(tmp_path)
    existing = tmp_path / "already.pt"
    existing.touch()
    first = {"name": "one", "expected_output": str(existing)}
    second = {"name": "two", "expected_output": str(tmp_path / "missing.pt")}
    plan["bank_jobs"] = [[first, second], {"name": "merge"}]
    calls = []
    monkeypatch.setattr(runner, "run_processes", lambda plan, specs, phase: calls.append((phase, specs)))
    runner.run_bank_jobs(plan, "bank_jobs", "banks")
    assert calls == [("banks-0", [second]), ("banks-1", [{"name": "merge"}])]


def test_default_pipeline_builds_distinct_mini_and_full_banks_and_scores_only_best_checkpoint(tmp_path):
    plan = runner.configure_bank_jobs(plan_at(tmp_path))
    mini_workers = plan["preflight_bank_jobs"][0]
    formal_workers = plan["bank_jobs"][0]
    assert len(mini_workers) == len(formal_workers) == 4
    assert {job["gpus"] for job in mini_workers} == {"0", "1", "2", "3"}
    for job in mini_workers:
        assert job["argv"][job["argv"].index("--limit-contexts") + 1] == "16"
        assert "preflight" not in job["name"] or "--resume" in job["argv"]
    assert all("--limit-contexts" not in job["argv"] for job in formal_workers)
    evaluation = [job for stage in plan["evaluation_jobs"] for job in stage if job["name"].endswith("/evaluate")]
    assert len(evaluation) == 2
    for job in evaluation:
        assert job["argv"][job["argv"].index("--evaluate-checkpoint") + 1] == str(
            Path(plan["run_directory"]) / "best.pt"
        )
        assert "--resume" not in job["argv"]


def test_final_acceptance_rejects_reports_from_a_different_best_checkpoint(tmp_path):
    plan = plan_at(tmp_path)
    directory = Path(plan["run_directory"])
    directory.mkdir()
    (directory / "best.pt").write_bytes(b"selected-by-D128")
    for cohort in ("D400", "long50"):
        path = tmp_path / "evaluation" / cohort / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"checkpoint_sha256": "some-other-checkpoint"}))
    with pytest.raises(ValueError, match="D128-selected best"):
        runner.final_decision(plan)


@pytest.mark.parametrize("action", ["run", "preflight", "verify-preflight"])
def test_checkout_execution_replaces_controller_before_importing_mutable_gate_code(tmp_path, monkeypatch, action):
    plan = plan_at(tmp_path)
    snapshot = Path(plan["source_snapshot"])
    snapshot.mkdir()
    plan["python"] = sys.executable
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "mutable-checkout"))
    monkeypatch.setattr(runner, "verify_identity", lambda plan: None)
    captured = {}

    class Reexecuted(Exception):
        pass

    def execute(python, argv, env):
        captured.update(python=python, argv=argv, env=env, cwd=Path.cwd())
        raise Reexecuted

    monkeypatch.setattr(runner.os, "execve", execute)
    with pytest.raises(Reexecuted):
        runner.reexec_frozen_controller(plan, action, Path("plan.json"))
    assert captured["argv"] == [
        sys.executable,
        str(snapshot / "scripts/run_formal_prior.py"),
        action,
        "--plan",
        str(tmp_path / "plan.json"),
    ]
    assert captured["cwd"] == snapshot
    assert captured["env"]["PYTHONPATH"] == str(snapshot)


def test_already_frozen_controller_and_read_only_monitor_do_not_reexecute(tmp_path, monkeypatch):
    plan = plan_at(tmp_path)
    plan["python"] = sys.executable
    snapshot = Path(plan["source_snapshot"])
    snapshot.mkdir()
    monkeypatch.chdir(snapshot)
    monkeypatch.setenv("PYTHONPATH", str(snapshot))
    monkeypatch.setattr(runner, "__file__", str(snapshot / "scripts/run_formal_prior.py"))
    monkeypatch.setattr(runner.os, "execve", lambda *args: pytest.fail("unexpected recursive reexecution"))
    assert runner.reexec_frozen_controller(plan, "run", tmp_path / "plan.json") is False
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "mutable-runner.py"))
    assert runner.reexec_frozen_controller(plan, "monitor", tmp_path / "plan.json") is False


def test_reused_bank_requires_matching_completion_sha(tmp_path):
    plan = plan_at(tmp_path)
    bank = tmp_path / "bank.pt"
    bank.write_bytes(b"some-cache")
    plan["bank_jobs"] = [[{"name": "merge", "expected_output": str(bank)}]]
    (tmp_path / "complete.json").write_text(json.dumps({"bank_sha256": "different-cache"}))
    with pytest.raises(ValueError, match="completion identity"):
        runner.run_bank_jobs(plan, "bank_jobs", "banks")
    (tmp_path / "complete.json").write_text(json.dumps({"bank_sha256": runner.sha256(bank)}))
    runner.run_bank_jobs(plan, "bank_jobs", "banks")


def test_final_acceptance_rejects_wrong_cohort_even_from_the_same_P_checkpoint(tmp_path):
    plan = plan_at(tmp_path)
    directory = Path(plan["run_directory"])
    directory.mkdir()
    (directory / "best.pt").write_bytes(b"selected-by-D128")
    Path(plan["protocol"]).write_text(json.dumps({"evaluation": {"D400": {"selection_id": "full400", "clips": 400}}}))
    for cohort in ("D400", "long50"):
        path = tmp_path / "evaluation" / cohort / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"checkpoint_sha256": runner.sha256(directory / "best.pt"), "bank_cohort": "D128"}))
    with pytest.raises(ValueError, match="bank/cohort/selection"):
        runner.final_decision(plan)
