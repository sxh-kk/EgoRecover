"""The adaptation opt-in changes P identity, never G resume semantics."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from egorecover import ModelConfig, WindowConfig
from egorecover.checkpoint import FORMAT, load_predictor
from egorecover.cli import build_parser
from egorecover.config import TrainOptions
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.io import file_sha256, state_sha256
from egorecover.models import HistoryPrior
from egorecover.training import compatible_initialization, prior_change_provenance, train


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def payloads(tmp_path):
    cfg = WindowConfig(shape_mode="predicted", geometry_target="official")
    stats = synthetic_episode().stats
    original = {
        "manifest": "manifest",
        "prior": "constant_velocity",
        "prior_mode": "constant_velocity",
        "synthetic": False,
        "compact": False,
        "smpl_asset": "licensed-smpl-sha",
        "model_config": ModelConfig.compact().to_dict(),
        "data_format": "official",
        "data_protocol": {"scope": "official"},
    }
    saved = {"format": FORMAT, "protocol": cfg.to_dict(), "identity": original, "stats": stats}
    model_config = {"width": 24, "layers": 1, "heads": 4, "dropout": 0.1, "source_flags": True, "hold_beta": True}
    prior = HistoryPrior(**model_config)
    p = {
        "format": "egorecover_prior_v1",
        "manifest_sha256": "manifest",
        "stats": copy.deepcopy(stats),
        "model_config": model_config,
        "state_dict": prior.state_dict(),
        "identity": {
            "shape_mode": "predicted",
            "geometry_target": "official",
            "smpl_asset_sha256": "licensed-smpl-sha",
            "producer": {
                "checkpoint_sha256": "base-G-sha",
                "training_manifest": "manifest",
                "window": cfg.window,
                "shape_mode": cfg.shape_mode,
                "source": cfg.source,
                "NFE": cfg.steps,
                "revise_recent": cfg.revise_recent,
                "prior": "constant_velocity",
            },
        },
    }
    path = tmp_path / "prior.pt"
    torch.save(p, path)
    identity = {**original, "prior": file_sha256(path), "prior_mode": "learned", "initialize": "base-G-sha"}
    return cfg, saved, identity, p, path


def test_only_explicit_cv_to_learned_prior_identity_change_is_allowed(tmp_path):
    cfg, saved, identity, _, _ = payloads(tmp_path)
    with pytest.raises(ValueError, match="identity changed: prior"):
        compatible_initialization(saved, cfg, identity)
    compatible_initialization(saved, cfg, identity, allow_prior_change=True)
    for key in ("manifest", "smpl_asset", "model_config", "data_format", "data_protocol"):
        changed = {**identity, key: "different"}
        with pytest.raises(ValueError, match=f"identity changed: {key}"):
            compatible_initialization(saved, cfg, changed, allow_prior_change=True)
    with pytest.raises(ValueError, match="rewrite"):
        compatible_initialization(saved, cfg, identity, allow_prior_change=True, allow_rewrite_change=True)
    saved["identity"]["prior"] = "another-learned-P"
    with pytest.raises(ValueError, match="only permits"):
        compatible_initialization(saved, cfg, identity, allow_prior_change=True)


def test_prior_swap_checks_and_records_the_exact_producer_and_normalization(tmp_path):
    cfg, saved, identity, _, path = payloads(tmp_path)
    loaded = load_predictor(path, "cpu", identity["manifest"])
    assert not loaded.training and loaded.hold_beta
    assert all(not p.requires_grad for p in loaded.parameters())
    provenance = prior_change_provenance(saved, cfg, identity, saved["stats"], path)
    assert provenance["from_prior"] == "constant_velocity"
    assert provenance["to_prior"] == file_sha256(path)
    assert provenance["source_G_checkpoint_sha256"] == identity["initialize"]
    assert provenance["manifest_stats_shape_smpl_and_producer_checked"] is True
    assert provenance["optimizer"] == "fresh"


@pytest.mark.parametrize(
    "fault", ["manifest", "stats", "shape", "geometry", "smpl", "producer", "format", "missing_stats"]
)
def test_prior_swap_rejects_a_mismatched_native_p_even_with_explicit_opt_in(tmp_path, fault):
    cfg, saved, identity, prior, path = payloads(tmp_path)
    if fault == "manifest":
        prior["manifest_sha256"] = "other-manifest"
    elif fault == "stats":
        prior["stats"]["motion_mean"] = prior["stats"]["motion_mean"] + 1
    elif fault == "shape":
        prior["identity"]["shape_mode"] = "bootstrap"
    elif fault == "geometry":
        prior["identity"]["geometry_target"] = "fixed_fk"
    elif fault == "smpl":
        prior["identity"]["smpl_asset_sha256"] = "other-smpl"
    elif fault == "producer":
        prior["identity"]["producer"]["checkpoint_sha256"] = "other-G"
    elif fault == "format":
        prior["format"] = "unknown"
    elif fault == "missing_stats":
        prior.pop("stats")
    torch.save(prior, path)
    identity["prior"] = file_sha256(path)
    with pytest.raises(ValueError):
        prior_change_provenance(saved, cfg, identity, saved["stats"], path)


def test_prior_swap_never_weakens_the_existing_manifest_or_parameter_shape_loader(tmp_path):
    _, _, _, prior, path = payloads(tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        load_predictor(path, "cpu", "other-manifest")
    prior["state_dict"]["output.weight"] = torch.zeros(1)
    torch.save(prior, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_predictor(path, "cpu", "manifest")


@pytest.mark.parametrize("resume", [False, True])
def test_prior_swap_override_requires_new_initialization_and_is_forbidden_on_resume(tmp_path, resume):
    options = TrainOptions(output=tmp_path / "absent", allow_initialize_prior_change=True, resume=resume)
    with pytest.raises(ValueError, match="requires --initialize"):
        train(options)
    assert not options.output.exists()


def test_prior_swap_flag_is_opt_in_on_the_public_cli():
    parser = build_parser()
    assert not parser.parse_args(["train", "--output", "out"]).allow_initialize_prior_change
    args = parser.parse_args(["train", "--output", "out", "--initialize", "G.pt", "--allow-initialize-prior-change"])
    assert args.allow_initialize_prior_change and args.initialize.name == "G.pt"


def test_new_prior_adaptation_inherits_only_g_ema_then_resumes_the_same_prior_and_optimizer(tmp_path, monkeypatch):
    """Exercise the production initialization branch with tiny CPU-only data."""
    import egorecover.training as training

    episode = synthetic_episode()
    episode.synthetic = False

    class Data:
        stats = episode.stats
        digest = "manifest"
        records = [{"episode_id": episode.identifier, "base_take_name": episode.take}]

        def load(self, _):
            return episode

        def sample(self, _):
            return episode

    monkeypatch.setattr(training, "open_window_data", lambda *a, **k: Data())
    monkeypatch.setattr(training, "load_smpl", lambda _: object())
    monkeypatch.setattr(training, "smpl_identity", lambda: "licensed-smpl-sha")
    monkeypatch.setattr(training, "joint_decoder", lambda *a, **k: DenseJoints())
    monkeypatch.setattr(training, "ModelConfig", lambda: ModelConfig.compact())
    cfg = WindowConfig(current_weight=0.03, shape_mode="predicted", geometry_target="official")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg.to_dict()))
    base = TrainOptions(
        output=tmp_path / "base",
        config=config,
        data_root=tmp_path,
        data_format="paired",
        from_scratch=True,
        train_steps=1,
        batch_size=1,
        microbatch_size=1,
        rollin_frames=0,
        eval_every=1,
        lr_schedule="constant",
    )
    train(base)
    original = torch.load(base.output / "best.pt", map_location="cpu", weights_only=True)
    _, _, _, prior, prior_path = payloads(tmp_path)
    prior["identity"]["producer"]["checkpoint_sha256"] = file_sha256(base.output / "best.pt")
    torch.save(prior, prior_path)
    target = replace(
        base,
        output=tmp_path / "adaptation",
        from_scratch=False,
        initialize=base.output / "best.pt",
        prior_checkpoint=prior_path,
        prior_mode="learned",
        allow_initialize_prior_change=True,
        train_steps=2,
        stop_after=1,
    )
    train(target)
    identity = json.loads((target.output / "identity.json").read_text())
    provenance = identity["initialization_prior_change"]
    receipt = json.loads((target.output / "initialization.json").read_text())
    assert receipt["prior_change"] == provenance
    assert receipt["loaded_model_sha256"] == state_sha256(original["ema"])
    saved = torch.load(target.output / "resume.pt", map_location="cpu", weights_only=True)
    assert state_sha256(saved["feedback_model"]) == state_sha256(original["ema"])
    assert saved["step"] == 1 and all(int(s["step"]) == 1 for s in saved["optimizer"]["state"].values())
    continued = replace(target, initialize=None, allow_initialize_prior_change=False, resume=True, stop_after=None)
    train(continued)
    saved = torch.load(target.output / "resume.pt", map_location="cpu", weights_only=True)
    assert saved["identity"]["initialization_prior_change"] == provenance
    assert saved["step"] == 2 and all(int(s["step"]) == 2 for s in saved["optimizer"]["state"].values())
    assert file_sha256(prior_path) == provenance["to_prior"]
    prior["state_dict"]["output.bias"] = prior["state_dict"]["output.bias"] + 0.01
    torch.save(prior, prior_path)
    with pytest.raises(ValueError, match="identity changed"):
        train(continued)
