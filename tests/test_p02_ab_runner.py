import copy
import json
import sys
from pathlib import Path

import pytest
import torch

from scripts import run_formal_prior as single
from scripts import run_formal_prior_ab as ab


def protocol_pair(tmp_path):
    base = json.loads(Path("docs/reports/data/formal_p_training_plan_20261113.json").read_text())
    checkpoint = tmp_path / "G.pt"
    checkpoint.write_bytes(b"frozen-G")
    base["producer"].update(checkpoint=str(checkpoint), sha256=single.sha256(checkpoint))
    base["experiment_family"] = "P02_common_cv_ab"
    base["loss"].update(objective="common_cv_valid_state", fixed_geometry_weight=0.008869197811349056)
    base["existing_banks"] = {}
    for name in ("train", "D128", "preflight_train", "preflight_D128"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        base["existing_banks"][name] = {"path": str(path), "sha256": single.sha256(path)}
    protocols = {}
    files = {}
    for case, run_id in (("A", "A_P02_commonCV_s62"), ("B", "B_P02_commonCV_holdBeta_s62")):
        p = copy.deepcopy(base)
        p.update(case=case, run_id=run_id)
        p["model"]["hold_beta"] = case == "B"
        files[case] = tmp_path / f"{case}.json"
        files[case].write_text(json.dumps(p))
        protocols[case] = p
    return protocols, files


def source_at(tmp_path):
    root = tmp_path / "repo"
    for name in ("egorecover", "scripts"):
        p = root / name
        p.mkdir(parents=True)
        (p / "__init__.py").write_text("")
    return root


def initialized_pair(tmp_path):
    protocols, files = protocol_pair(tmp_path)
    plan = ab.initialize_ab(files, tmp_path / "runtime", source_at(tmp_path), python=sys.executable)
    return protocols, plan


def test_pair_freezes_shared_code_and_reuses_all_banks_without_generation(tmp_path):
    protocols, plan = initialized_pair(tmp_path)
    children = ab.verify_pair_identity(plan)
    assert plan["resource_wait"]["stable_seconds"] == 60
    for case, p in children.items():
        assert p["source_snapshot"] == plan["source_snapshot"]
        assert p["pair_managed"] == str(Path(plan["output"]) / "plan.json")
        assert p["preflight_bank_jobs"] == p["bank_jobs"] == []
        assert p["train_bank"] == protocols[case]["existing_banks"]["train"]["path"]
        assert p["val_bank"] == protocols[case]["existing_banks"]["D128"]["path"]
        with pytest.raises(ValueError, match="matched A/B"):
            single.supervise(p)
    assert not (Path(plan["output"]) / "A/source").exists()
    assert not (Path(plan["output"]) / "B/source").exists()
    assert not (Path(plan["output"]) / "A/banks/train/bank.pt").exists()
    source = Path(plan["source_snapshot"]) / "egorecover/__init__.py"
    source.write_text("# unexpected mutation")
    with pytest.raises(ValueError, match="implementation changed"):
        ab.verify_pair_identity(plan)


def test_reused_bank_binding_detects_changed_content(tmp_path):
    protocols, plan = initialized_pair(tmp_path)
    Path(protocols["A"]["existing_banks"]["train"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing-bank identity"):
        ab.verify_pair_identity(plan)


def test_runtime_bank_alias_cannot_override_a_frozen_content_binding(tmp_path):
    _, plan = initialized_pair(tmp_path)
    path = Path(plan["cases"]["A"])
    child = json.loads(path.read_text())
    child["train_bank"] = child["val_bank"]
    path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match="bank path differs"):
        ab.verify_pair_identity(plan)


def test_resource_guard_must_cover_both_four_gpu_groups(tmp_path):
    _, plan = initialized_pair(tmp_path)
    plan["resource_wait"]["gpus"] = ["0", "1", "2", "3"]
    with pytest.raises(ValueError, match="all8 assigned GPUs"):
        ab.verify_pair_identity(plan)


@pytest.mark.parametrize("kind", ("hold", "weight", "sampling", "architecture"))
def test_matched_protocols_reject_unapproved_factors(tmp_path, kind):
    p, _ = protocol_pair(tmp_path)
    if kind == "hold":
        p["B"]["model"]["hold_beta"] = False
    elif kind == "weight":
        p["B"]["loss"]["fixed_geometry_weight"] = 0.03
    elif kind == "sampling":
        p["B"]["sampling"]["global_mature"] = 32
    else:
        p["B"]["model"]["width"] = 128
    with pytest.raises(ValueError):
        ab.validate_pair_protocols(p)


def test_gpu_pool_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        ab.validate_gpu_pools({"A": "0,1,2,3", "B": "3,4,5,6"})


def resource_config():
    return {
        "enabled": True,
        "gpus": ["0", "1"],
        "max_memory_MiB": 2048,
        "stable_seconds": 30,
        "poll_seconds": 15,
        "blocking_processes": [],
    }


def free_observation():
    return {"cards": {"0": {"uuid": "uuid0", "used_MiB": 1}, "1": {"uuid": "uuid1", "used_MiB": 1}}, "processes": []}


def test_gpu_guard_requires_no_processes_low_memory_and_original_launcher_exit(monkeypatch):
    config = resource_config()
    obs = free_observation()
    assert ab.resources_available(config, obs)[0]
    obs["processes"] = [{"pid": 90, "uuid": "uuid1", "used_MiB": 12}]
    assert not ab.resources_available(config, obs)[0]
    obs = free_observation()
    obs["cards"]["0"]["used_MiB"] = 2048
    assert not ab.resources_available(config, obs)[0]
    obs = free_observation()
    config["blocking_processes"] = [{"pid": 50, "start_ticks": 100, "cmdline_sha256": "cmd"}]
    monkeypatch.setattr(ab, "process_identity", lambda pid: 100)
    monkeypatch.setattr(ab, "command_identity", lambda pid: "cmd")
    assert not ab.resources_available(config, obs)[0]
    monkeypatch.setattr(ab, "command_identity", lambda pid: "new-owner")
    assert not ab.resources_available(config, obs)[0]
    assert ab.resources_available(config, obs)[1]["blocking_launchers_alive"][0]["command_changed"]
    monkeypatch.setattr(ab, "process_identity", lambda pid: 101)
    assert ab.resources_available(config, obs)[0]
    monkeypatch.setattr(ab, "command_identity", lambda pid: "cmd")
    monkeypatch.setattr(ab, "process_identity", lambda pid: None)
    assert ab.resources_available(config, obs)[0]


def test_gpu_guard_resets_stability_after_an_occupied_interval(tmp_path, monkeypatch):
    busy = free_observation()
    busy["processes"] = [{"pid": 30, "uuid": "uuid0"}]
    snapshots = iter(
        [busy, free_observation(), free_observation(), busy, free_observation(), free_observation(), free_observation()]
    )
    times = iter([0, 15, 30, 45, 60, 75, 90])
    sleeps = []
    monkeypatch.setattr(ab, "gpu_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(ab.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(ab.time, "sleep", lambda seconds: sleeps.append(seconds))
    plan = {"output": str(tmp_path), "resource_wait": resource_config()}
    report = ab.wait_for_gpus(plan)
    assert report["phase"] == "resources_ready" and report["stable_free_seconds"] == 30
    assert len(sleeps) == 6
    assert json.loads((tmp_path / "resource-verification.json").read_text())["stable_free_seconds"] == 30


def setup_pair_evidence(plan, *, steps=2):
    children = {case: single.read(path) for case, path in plan["cases"].items()}
    for case, p in children.items():
        directory = Path(p["output"]) / "preflight/full"
        directory.mkdir(parents=True)
        identity = {
            "random_initial_model_sha256": "matched-init",
            "bank_sha256": {"train": "train", "val": "val"},
            "seed": 62,
            "batch_size": 64,
            "microbatch_size": 8,
            "objective": "common_cv_valid_state",
            "model_config": {"hold_beta": case == "B"},
            "geometry_audit": {"weight": 0.008869197811349056, "weight_mode": "P01_fixed_not_recalibrated"},
        }
        (directory / "identity.json").write_text(json.dumps(identity))
        torch.save({"step": steps}, directory / "resume.pt")
        logs = [
            {
                "step": step,
                "trace_hash": f"query{step}",
                "lr": 3e-5 if step == 1 else 3e-6,
                "code": 0.7 if step == 1 else 0.6,
                "geometry": 1.2,
                "loss": 0.71 if step == 1 else 0.61,
            }
            for step in range(1, steps + 1)
        ]
        (directory / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in logs))
    return children


def test_pair_audit_checks_initial_weights_objective_and_query_trace(tmp_path):
    _, plan = initialized_pair(tmp_path)
    children = setup_pair_evidence(plan)
    report = ab.audit_matched_pair(plan, children, preflight=True)
    assert report["matched_prefix_updates"] == 2 and report["initial_objective_equal"]
    path = Path(children["B"]["output"]) / "preflight/full/train.jsonl"
    path.write_text(path.read_text().replace("query2", "wrong-query"))
    with pytest.raises(ValueError, match="query trace differs"):
        ab.audit_matched_pair(plan, children, preflight=True)


def test_pair_audit_rejects_different_initial_weights(tmp_path):
    _, plan = initialized_pair(tmp_path)
    children = setup_pair_evidence(plan)
    path = Path(children["B"]["output"]) / "preflight/full/identity.json"
    value = json.loads(path.read_text())
    value["random_initial_model_sha256"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="initial identity"):
        ab.audit_matched_pair(plan, children, preflight=True)


def test_pair_audit_rejects_recalibrated_weight_or_old_objective(tmp_path):
    _, plan = initialized_pair(tmp_path)
    children = setup_pair_evidence(plan)
    path = Path(children["B"]["output"]) / "preflight/full/identity.json"
    initial = json.loads(path.read_text())
    for key in ("weight", "objective"):
        changed = copy.deepcopy(initial)
        if key == "weight":
            changed["geometry_audit"]["weight"] = 0.03
        else:
            changed["objective"] = "raw_code"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="actual training objective"):
            ab.audit_matched_pair(plan, children, preflight=True)


@pytest.mark.parametrize("passing", ([], ["A"], ["A", "B"]))
def test_ab_continues_only_passing_cases_after_both_pilots(tmp_path, monkeypatch, passing):
    plan = {"output": str(tmp_path), "cases": {"A": "A.json", "B": "B.json"}}
    children = {
        case: {
            "output": str(tmp_path / case),
            "run_id": case,
            "run_directory": str(tmp_path / case / "run"),
            "python": "python",
            "training_gpus": "0,1,2,3" if case == "A" else "4,5,6,7",
        }
        for case in ("A", "B")
    }
    steps = {"A": 0, "B": 0}
    calls = []
    events = []
    monkeypatch.setattr(ab, "verify_pair_identity", lambda plan: children)
    monkeypatch.setattr(ab, "wait_for_gpus", lambda plan: events.append("wait"))
    monkeypatch.setattr(ab, "audit_matched_pair", lambda *a, **kw: events.append("audit"))
    monkeypatch.setattr(single, "checkpoint_step", lambda directory: steps["A" if "/A/" in str(directory) else "B"])
    monkeypatch.setattr(
        single,
        "train_command",
        lambda p, directory, **kw: [p["run_id"], str(kw.get("stop_after")), str(kw.get("resume"))],
    )

    def execute(plan, specs, phase):
        calls.append((phase, [s["name"] for s in specs]))
        events.append(phase)
        if phase == "AB-training-to2000":
            steps.update(A=2000, B=2000)
        if phase == "AB-passed-continuations-to10000":
            for case in passing:
                steps[case] = 10000

    monkeypatch.setattr(single, "run_processes", execute)
    monkeypatch.setattr(single, "pilot_decision", lambda p: {"passed": p["run_id"] in passing})
    monkeypatch.setattr(single, "run_bank_jobs", lambda *a: None)
    monkeypatch.setattr(single, "final_decision", lambda p: None)
    monkeypatch.setattr(single, "monitor", lambda p: {"step": steps[p["run_id"]]})
    ab.supervise_ab(plan)
    assert events[0] == "wait"
    assert calls[0][0] == "AB-preflight" and calls[1] == ("AB-training-to2000", ["A", "B"])
    if passing:
        assert calls[2] == ("AB-passed-continuations-to10000", passing)
    else:
        assert len(calls) == 2
    for case in ("A", "B"):
        status = json.loads((tmp_path / case / "status.json").read_text())
        assert status["phase"] == (
            "training_complete_pending_fixed_evaluation" if case in passing else "stopped_gate_failed"
        )
