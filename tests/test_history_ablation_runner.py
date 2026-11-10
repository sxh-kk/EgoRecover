import json

import pytest

from scripts.run_history_ablation import collect_evaluation, train_command


def test_four_gpu_continuation_and_resume_keep_controls_explicit(tmp_path):
    plan = {"python": "python", "data_root": "data", "prior_checkpoint": "P.pt", "common_G_checkpoint": "G.pt"}
    job = {"port": 30211, "config": "off.json", "history": "mixed", "rewrite": False}
    fresh = train_command(plan, job, tmp_path, preflight=True)
    resumed = train_command(plan, job, tmp_path, preflight=True, resume=True)
    assert fresh[fresh.index("--initialize") + 1] == "G.pt"
    assert "--initialize" not in resumed and "--resume" in resumed
    assert "--allow-initialize-rewrite-change" in fresh and "--allow-initialize-rewrite-change" in resumed
    assert fresh[fresh.index("--sampling-rollin-frames") + 1] == "20"
    assert fresh[fresh.index("--history-mode") + 1] == "mixed"
    assert "--from-scratch" not in fresh and "--e7-checkpoint" not in fresh


def test_collector_rejects_partial_cohort_before_reporting_400(tmp_path):
    root = tmp_path / "evaluation400"
    root.mkdir()
    (root / "shared_worker.py").write_text("KEYS = ()\n")
    (root / "selection.json").write_text(json.dumps({"records": [{"take": "a"}]}))
    with pytest.raises(ValueError, match="complete fixed400"):
        collect_evaluation({"output": str(tmp_path)})
