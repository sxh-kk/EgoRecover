import subprocess

import pytest

from run.complete_stages import idle_gpu_indices, replay_selection, stage2_tasks, training
from run import complete_stages


def test_busy_or_unverifiable_gpus_are_never_selected(monkeypatch):
    def query(argv, **kwargs):
        if any("query-gpu=" in value for value in argv):
            return "0, GPU-a, 100, 0\n1, GPU-b, 100, 0\n2, GPU-c, 2000, 0\n3, GPU-d, 100, 50\n4, GPU-e, 100, 0\n"
        return "GPU-a, 123\n"
    monkeypatch.setattr(subprocess, "check_output", query)
    assert idle_gpu_indices([0, 1, 2, 3], {}) == {1}
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", failure)
    with pytest.raises(subprocess.CalledProcessError):
        idle_gpu_indices([0, 1, 2, 3], {})


def test_shared_policy_checks_scope_memory_and_compute_mode(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs:
                        "0, 98000, 1000, Default\n4, 98000, 49000, Default\n"
                        "5, 98000, 80000, Default\n6, 98000, 49000, Exclusive_Process\n"
                        "7, 98000, 49000, Default\n")
    assert complete_stages.shared_gpu_indices([4, 5, 6, 7], {}, 32) == {4, 7}


def test_policy_change_preserves_pending_tasks_but_rejects_running(tmp_path):
    output = tmp_path / "queue"
    first = complete_stages.Queue(output, [0, 1, 2, 3], 10)
    first.register(complete_stages.Task("training", [], tmp_path / "train", "train"))
    first.save()
    first.lock.close()
    shared = complete_stages.Queue(output, [4, 5, 6, 7], 10, allow_shared=True)
    assert shared.state["tasks"]["training"]["status"] == "pending"
    assert len(shared.state["policy_changes"]) == 1
    shared.state["tasks"]["training"]["status"] = "running"
    shared.save()
    shared.lock.close()
    with pytest.raises(ValueError, match="while experiment jobs are running"):
        complete_stages.Queue(output, [0], 10)


def test_six_budget_cells_and_training_dependencies(tmp_path):
    tasks, experiments = stage2_tasks(tmp_path)
    assert len(experiments) == 6
    assert len([task for task in tasks if task.kind == "eval"]) == 6
    for task in tasks:
        assert "holdout" not in task.argv
        if task.kind == "eval":
            assert task.expected == 108
            assert task.deps == ([task.name.replace("eval_", "train_")] if task.name.endswith("2400") else [])
    assert "--resume-incomplete" in tasks[1].argv
    assert "--resume-incomplete" not in tasks[0].argv
    for probability in (0., .25, .5):
        argv = training(tmp_path / str(probability), initial="selected", cache="train_cache", probability=probability)
        assert argv[argv.index("--prior-steps") + 1] == "400"
        assert argv[argv.index("--flow-steps") + 1] == "400"
        assert argv[argv.index("--history-cache") + 1] == "train_cache"


def test_replay_requires_paired_improvement_and_ties_prefer_less_replay():
    summary = {"takes": ["a", "b"], "scores": {}}
    for label in ("before_replay", "replay_0", "replay_0p25", "replay_0p5"):
        for mode in ("gaussian", "history"):
            summary["scores"][f"{label}/{mode}"] = {"per_take_macro_smpl22_mm": {"a": 100., "b": 200.}}
    result = replay_selection(summary)
    assert result["best_probability_label"] == "replay_0"
    assert result["replay_improvement_supported"] is False
    for mode in ("gaussian", "history"):
        summary["scores"][f"replay_0p25/{mode}"]["per_take_macro_smpl22_mm"] = {"a": 90., "b": 190.}
    result = replay_selection(summary)
    assert result["best_probability_label"] == "replay_0p25"
    assert result["replay_improvement_supported"] is True


def test_queue_waits_for_idle_confirmation_and_obeys_dependencies(tmp_path, monkeypatch):
    queue = complete_stages.Queue(tmp_path / "queue", [0, 1], 10)
    tasks = [complete_stages.Task("first", [], tmp_path / "first", "train"),
             complete_stages.Task("second", [], tmp_path / "second", "train", deps=["first"])]
    snapshots = iter([set(), {0}, {0}, {0}])
    sleeps, launched = [], []
    monkeypatch.setattr(complete_stages, "idle_gpu_indices", lambda *args: next(snapshots))
    monkeypatch.setattr(complete_stages.time, "sleep", lambda seconds: sleeps.append(seconds))
    def launch(task, gpu):
        assert len(sleeps) >= 2  # Busy poll, then first idle poll cannot launch.
        assert all(queue.state["tasks"][dep]["status"] == "complete" for dep in task.deps)
        launched.append((task.name, gpu))
        queue.state["tasks"][task.name]["status"] = "complete"
    monkeypatch.setattr(queue, "launch", launch)
    queue.run(tasks)
    assert launched == [("first", 0), ("second", 0)]
    queue.lock.close()
