import json
from dataclasses import replace

import pytest

from egorecover.cli import build_parser
from egorecover.config import TrainOptions
from egorecover.training import train


@pytest.mark.parametrize("kind", ["e7_checkpoint", "initialize"])
def test_random_g_rejects_loading_another_initialization(tmp_path, kind):
    options = TrainOptions(output=tmp_path / "out", from_scratch=True, **{kind: tmp_path / "weights.pt"})
    with pytest.raises(ValueError, match="cannot be combined"):
        train(options)
    assert not options.output.exists()


def test_real_training_requires_an_explicit_initialization_choice(tmp_path):
    options = TrainOptions(output=tmp_path / "out", data_root=tmp_path, prior_checkpoint=tmp_path / "prior.pt")
    with pytest.raises(ValueError, match="Choose --from-scratch"):
        train(options)


def test_random_g_identity_persists_across_resume(tmp_path):
    options = TrainOptions(
        output=tmp_path / "run",
        synthetic=True,
        compact=True,
        from_scratch=True,
        train_steps=2,
        batch_size=1,
        rollin_frames=0,
        eval_every=1,
        stop_after=1,
    )
    train(options)
    identity = json.loads((options.output / "identity.json").read_text())
    assert identity["from_scratch"] is True
    assert identity["e7"] is None and identity["initialize"] is None
    with pytest.raises(ValueError, match="identity changed"):
        train(replace(options, from_scratch=False, resume=True, stop_after=None))
    train(replace(options, resume=True, stop_after=None))
    assert json.loads((options.output / "progress.json").read_text())["step"] == 2


def test_training_startup_flags_are_available_on_the_public_cli():
    parser = build_parser()
    args = parser.parse_args(["train", "--from-scratch", "--output", "exp/new"])
    assert args.from_scratch and args.e7_checkpoint is None
    assert parser.parse_args(["prepare"]).initialization == "scratch"
    assert parser.parse_args(["evaluate", "--checkpoint", "a.pt", "--output", "b", "--skip-paired"]).skip_paired


def test_initialized_continuation_can_resume_without_reinitializing(tmp_path):
    base = TrainOptions(
        output=tmp_path / "base",
        synthetic=True,
        compact=True,
        from_scratch=True,
        train_steps=1,
        batch_size=1,
        rollin_frames=0,
        eval_every=1,
    )
    train(base)
    options = replace(
        base,
        output=tmp_path / "continued",
        from_scratch=False,
        initialize=base.output / "best.pt",
        train_steps=2,
        stop_after=1,
    )
    train(options)
    origin = json.loads((options.output / "identity.json").read_text())["initialize"]
    assert origin
    train(replace(options, initialize=None, resume=True, stop_after=None))
    assert json.loads((options.output / "progress.json").read_text())["step"] == 2
    assert json.loads((options.output / "identity.json").read_text())["initialize"] == origin
