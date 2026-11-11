import subprocess
from types import SimpleNamespace

import pytest

from scripts.run_history_ablation import write
from scripts.run_shape_persistent import queue_evidence, training_command


@pytest.mark.parametrize(
    "finished,occupied,expected", [(False, False, False), (True, True, False), (True, False, True)]
)
def test_queue_waits_for_the_whole_simple_queue_and_all_eight_cards(
    tmp_path, monkeypatch, finished, occupied, expected
):
    status = tmp_path / "simple.json"
    write(
        status,
        {
            "status": "all_three_models_completed" if finished else "task_completed",
            "completed_tasks": 3 if finished else 1,
            "tasks": {},
        },
    )

    def execute(command, **kwargs):
        if "--query-compute-apps=pid" in command:
            return SimpleNamespace(stdout="1234\n" if occupied else "")
        return SimpleNamespace(stdout="\n".join(f"{gpu}, {60000 if occupied else 0}, 0" for gpu in range(8)))

    monkeypatch.setattr(subprocess, "run", execute)
    assert queue_evidence({"simple_queue_status": str(status)})["ready"] is expected


def test_both_commands_keep_supervised_frame_budget_and_differ_only_by_slow_weight():
    plan = {
        "python": "python",
        "data_root": "data",
        "prior_checkpoint": "P.pt",
        "common_G_checkpoint": "G.pt",
        "training_steps": 2000,
        "velocity_weight": 0.013409926806430328,
    }
    first = training_command(plan, {"port": 30811, "config": "shape.json", "slow_weight": 0}, "first")
    second = training_command(plan, {"port": 30811, "config": "shape.json", "slow_weight": 0.03}, "first")
    differences = [(a, b) for a, b in zip(first, second) if a != b]
    assert differences == [("0", "0.03")]
    assert int(first[first.index("--batch-size") + 1]) * int(first[first.index("--unroll-steps") + 1]) == 128
    assert first[first.index("--nproc_per_node=4")] == "--nproc_per_node=4"
