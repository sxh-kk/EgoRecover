import json
from dataclasses import replace

import pytest
import torch

from egorecover.cli import build_parser
from egorecover.config import ModelConfig, TrainOptions
from egorecover.training import train


def test_official_defaults_and_compact_diagnostic_dropout():
    args = build_parser().parse_args(["train", "--output", "unused"])
    assert args.batch_size == 64 and args.microbatch_size == 8
    assert args.learning_rate == 3e-5 and args.weight_decay == 0.01
    assert args.grad_clip_norm == 1.0 and args.lr_schedule == "uniegomotion"
    assert args.lr_decay_step is None and args.lr_decay_gamma == 0.1
    assert ModelConfig().dropout == 0.1 and ModelConfig.compact().dropout == 0.0


def test_native_resume_preserves_optimizer_across_learning_rate_drop(tmp_path):
    options = TrainOptions(
        output=tmp_path / "uninterrupted",
        synthetic=True,
        compact=True,
        from_scratch=True,
        train_steps=4,
        batch_size=2,
        microbatch_size=1,
        rollin_frames=0,
        eval_every=2,
        lr_decay_step=2,
    )
    train(options)
    interrupted = replace(options, output=tmp_path / "resumed", stop_after=2)
    train(interrupted)
    half = torch.load(interrupted.output / "resume.pt", weights_only=True)
    assert half["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-6)
    assert half["scheduler"]["last_epoch"] == 2
    train(replace(interrupted, stop_after=None, resume=True))
    first = torch.load(options.output / "resume.pt", weights_only=True)
    resumed = torch.load(interrupted.output / "resume.pt", weights_only=True)
    for group in ("model", "ema", "feedback_model"):
        for key, value in first[group].items():
            assert torch.equal(value, resumed[group][key]), (group, key)
    assert first["scheduler"] == resumed["scheduler"]
    assert torch.equal(first["sample_rng"], resumed["sample_rng"])
    assert torch.equal(first["cpu_rng"], resumed["cpu_rng"])
    final = json.loads((interrupted.output / "progress.json").read_text())
    assert final["learning_rate"] == pytest.approx(3e-6)
    assert final["next_learning_rate"] == pytest.approx(3e-7)
