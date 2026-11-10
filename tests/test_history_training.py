import json
from dataclasses import replace

import pytest
import torch

from egorecover import WindowConfig, WindowReconstructor
from egorecover.codec import MotionCodec
from egorecover.config import TrainOptions
from egorecover.data import TrainingBatch, keyed_noise
from egorecover.diagnostics import synthetic_episode
from egorecover.history_training import HistoryTraining, sampling_signature
from egorecover.models import HistoryPrior
from egorecover.sampling import reference_microbatch_plan
from egorecover.state import state_map
from egorecover.training import compatible_initialization, train


def options(tmp_path, **kwargs):
    return TrainOptions(output=tmp_path, sampling_rollin_frames=20, sample_reference_world_size=2, **kwargs)


def test_mixed_history_has_exact_fraction_and_does_not_consume_rng(tmp_path):
    cfg = HistoryTraining.from_options(options(tmp_path, history_mode="mixed", rollin_frames=20), 2)
    before = torch.get_rng_state().clone()
    seen = set()
    for step in range(1, 101):
        draws = [cfg.frames(step, offset) for offset in range(0, 64, 8)]
        assert sum(x == 20 for x in draws) == 6
        assert all(x in (0, 1, 4, 20) for x in draws)
        assert draws == [cfg.frames(step, offset) for offset in range(0, 64, 8)]
        seen.update(draws)
    assert seen == {0, 1, 4, 20}
    assert torch.equal(before, torch.get_rng_state())


class SharedQueries:
    def sample_queries(self, generator, count, rollin_frames):
        assert rollin_frames == 20
        queries = [("source", int(torch.randint(0, 100, (), generator=generator)), 42) for _ in range(count)]
        return queries, 40


def test_four_cells_share_physical_targets_times_noise_and_rng(tmp_path):
    signatures = []
    for mode, revise in (("fixed", True), ("fixed", False), ("mixed", True), ("mixed", False)):
        del revise  # Rewrite is a downstream cache policy, never a sampling input.
        cfg = HistoryTraining.from_options(
            options(tmp_path, history_mode=mode, rollin_frames=4 if mode == "fixed" else 20), 2
        )
        all_rows = []
        states = []
        for rank in range(4):
            generator = torch.Generator().manual_seed(63 + rank // 2)
            chunks = reference_microbatch_plan(
                SharedQueries(),
                generator,
                rank=rank,
                world_size=4,
                reference_world_size=2,
                batch_size=64,
                microbatch_size=8,
                rollin_frames=cfg.sampling_frames,
                endpoint_probability=0.1,
            )
            for chunk in chunks:
                noise = keyed_noise(str(chunk["queries"]), 62, chunk["frame"], 3, "cpu")
                all_rows.append(
                    {
                        "global_offset": chunk["global_offset"],
                        "queries": chunk["queries"],
                        "target_local_frame": chunk["frame"],
                        "tau": chunk["tau"].tolist(),
                        "noise": noise.tolist(),
                        "rollin_frames": cfg.frames(1, chunk["global_offset"]),
                    }
                )
            states.append(generator.get_state().tolist())
        signatures.append((sampling_signature(all_rows), states))
        assert all(row["target_local_frame"] == 40 for row in all_rows)
    assert all(value == signatures[0] for value in signatures)


@pytest.mark.parametrize(
    "changes",
    [
        {"sampling_rollin_frames": 3},
        {"history_mode": "bad"},
        {"history_mode": "mixed", "rollin_frames": 4},
        {"history_mode": "mixed", "rollin_frames": 20, "history_startup_fraction": 0.3},
        {"history_startup_fraction": float("nan")},
    ],
)
def test_invalid_history_controls_are_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        HistoryTraining.from_options(replace(options(tmp_path), **changes), 2)


def test_initialization_override_is_explicit_and_limited_to_rewrite():
    identity = {
        key: None
        for key in (
            "manifest",
            "prior",
            "synthetic",
            "compact",
            "smpl_asset",
            "model_config",
            "data_format",
            "data_protocol",
        )
    }
    cfg = WindowConfig(window=3)
    saved = {"format": "egorecover_window_v2", "protocol": cfg.to_dict(), "identity": identity}
    with pytest.raises(ValueError, match="revise_recent"):
        compatible_initialization(saved, replace(cfg, revise_recent=False), identity)
    compatible_initialization(saved, replace(cfg, revise_recent=False), identity, allow_rewrite_change=True)
    with pytest.raises(ValueError, match="sigma"):
        compatible_initialization(
            saved, replace(cfg, revise_recent=False, sigma=0.5), identity, allow_rewrite_change=True
        )


def test_rewrite_override_continuation_keeps_provenance_on_native_resume(tmp_path):
    base = TrainOptions(
        output=tmp_path / "base",
        synthetic=True,
        compact=True,
        from_scratch=True,
        train_steps=1,
        batch_size=1,
        microbatch_size=1,
        rollin_frames=0,
        eval_every=1,
    )
    train(base)
    cfg = tmp_path / "off.json"
    cfg.write_text(json.dumps(WindowConfig(revise_recent=False).to_dict()))
    changed = replace(
        base,
        output=tmp_path / "off",
        config=cfg,
        from_scratch=False,
        initialize=base.output / "best.pt",
        allow_initialize_rewrite_change=True,
        train_steps=2,
        stop_after=1,
    )
    train(changed)
    initial = json.loads((changed.output / "initialization.json").read_text())
    assert initial["weight_source"] == "ema" and initial["optimizer"] == "fresh"
    train(replace(changed, initialize=None, resume=True, stop_after=None))
    assert json.loads((changed.output / "progress.json").read_text())["step"] == 2
    assert json.loads((changed.output / "identity.json").read_text())["allow_initialize_rewrite_change"]


@pytest.mark.parametrize("rewrite,expected", [(True, 6), (False, 4)])
def test_neutral_reset_at_later_frame_tracks_actual_rollins_for_prior_flags(rewrite, expected):
    episode = synthetic_episode(length=72)
    episode.bootstrap_only = True
    prior = HistoryPrior(width=24, layers=1, heads=4, source_flags=True).freeze()
    system = WindowReconstructor(MotionCodec(episode.stats), prior, WindowConfig(window=3, revise_recent=rewrite))
    counts = []
    handle = prior.register_forward_pre_hook(
        lambda module, args, kwargs: counts.append(int(kwargs["history_body_valid"].sum())), with_kwargs=True
    )

    class Center(torch.nn.Module):
        def forward(self, x, tau, y):
            return y["center"]

    model = Center().eval()
    cache = episode.cache(36)
    assert cache.generated_frames == 0 and TrainingBatch((episode, episode)).cache(36).generated_frames == 0
    system.prepare(cache, episode.observations(36))
    assert counts[-1] == 0
    for frame in range(36, 40):
        cache = system.online(model, cache, episode.observations(frame), torch.zeros(1, 3, 243)).cache.detach()
    system.prepare(cache, episode.observations(40))
    assert cache.generated_frames == 4 and counts[-1] == expected
    assert cache.replace_current(state_map(cache.body, lambda value: value[:, -1])).generated_frames == 4
    handle.remove()
