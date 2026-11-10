import pytest
import torch

from scripts.calibrate_velocity import gradient_summary
from scripts.run_ac_training import ac_command


def test_ac_keep_identical_training_stream_and_change_only_the_intended_objective(tmp_path):
    plan = {"python": "python", "data_root": "data", "prior_checkpoint": "P.pt", "common_G_checkpoint": "C.pt"}
    common = {"port": 30321, "config": "config.json", "history": "mixed", "rewrite": True, "unroll_steps": 2}
    a = ac_command(plan, {**common, "detach_between": True, "velocity_weight": 0.0}, tmp_path)
    c = ac_command(plan, {**common, "detach_between": False, "velocity_weight": 0.02}, tmp_path)
    assert "--detach-between-steps" in a and "--detach-between-steps" not in c
    assert a[a.index("--velocity-weight") + 1] == "0.0"
    assert c[c.index("--velocity-weight") + 1] == "0.02"
    for flag in (
        "--initialize",
        "--seed",
        "--batch-size",
        "--microbatch-size",
        "--unroll-steps",
        "--rollin-frames",
        "--sample-reference-world-size",
    ):
        assert a[a.index(flag) + 1] == c[c.index(flag) + 1]
    resumed = ac_command(
        plan, {**common, "detach_between": False, "velocity_weight": 0.02}, tmp_path, preflight=True, resume=True
    )
    assert "--resume" in resumed and "--initialize" not in resumed
    assert resumed[resumed.index("--velocity-weight") + 1] == "0.02"


def test_velocity_gradient_calibration_uses_direction_and_global_norm():
    result = gradient_summary(torch.tensor([3.0, 4.0]), torch.tensor([-4.0, 3.0]))
    assert result["base_norm"] == result["velocity_norm"] == 5.0
    assert result["weight_10pct_norm"] == pytest.approx(0.1)
    assert result["cosine"] == 0.0
    with pytest.raises(RuntimeError):
        gradient_summary(torch.zeros(2), torch.ones(2))
