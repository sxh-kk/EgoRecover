from pathlib import Path

import pytest
import torch

from scripts.run_formal_mainline import assert_recovered_tensor, command, continuation_step, platform_decision


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
