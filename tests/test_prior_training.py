import copy

import torch

from egorecover.codec import MotionCodec
from egorecover.data import TrainingBatch
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.distributed import DistributedContext
from egorecover.models import HistoryPrior
from egorecover.prior_data import (
    clean_batch,
    concatenate_examples,
    predictor_forward,
    prior_example,
    reconstructed_history_count,
)
from egorecover.prior_training import evaluate_prior, summarize_prior_rows
from egorecover.state import state_map


def test_current_label_cannot_enter_predictor_inputs():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    example = clean_batch(codec, TrainingBatch((episode,)))
    changed = copy.deepcopy(episode)
    changed.truth.joints[20, :, :3, 3] += 1
    other = clean_batch(codec, TrainingBatch((changed,)))
    for key in ("history", "base", "reference", "body_valid"):
        torch.testing.assert_close(example[key], other[key], atol=0, rtol=0)
    assert not torch.equal(example["target"], other["target"])
    assert not torch.equal(example["truth_xyz"], other["truth_xyz"])


def test_random_prior_initially_equals_cv_with_full_history_and_source_flags():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    example = clean_batch(codec, TrainingBatch((episode,)))
    model = HistoryPrior(width=24, layers=1, heads=4, source_flags=True).eval()
    torch.testing.assert_close(predictor_forward(model, example), example["base"], atol=0, rtol=0)
    example["body_valid"].zero_()
    torch.testing.assert_close(predictor_forward(model, example), example["base"], atol=0, rtol=0)


def test_generated_history_flags_distinguish_startup_without_masking_all_inputs():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    truth = state_map(episode.target(20), lambda value: value[:, -1])
    for generated in (0, 1, 4, 20, 32):
        example = prior_example(codec, episode.cache().body, truth, generated_frames=generated)
        assert int(example["body_valid"].sum()) == min(generated, 20)
        assert example["history"].shape == (1, 20, 243)


def test_history_flags_include_revised_startup_states():
    assert reconstructed_history_count(0, 3, True) == 0
    assert reconstructed_history_count(1, 3, True) == 3
    assert reconstructed_history_count(4, 3, True) == 6
    assert reconstructed_history_count(4, 3, False) == 4
    assert reconstructed_history_count(19, 3, True) == 20


def test_frozen_source_aware_prior_keeps_input_gradients():
    model = HistoryPrior(width=24, layers=1, heads=4, source_flags=True)
    with torch.no_grad():
        model.output.weight.normal_(std=0.02)
    model.freeze()
    history = torch.randn(1, 20, 243, requires_grad=True)
    result = model(
        history,
        torch.ones(1, 20, dtype=torch.bool),
        torch.zeros(1, 1, 243),
        history_body_valid=torch.ones(1, 20, dtype=torch.bool),
    )
    result.square().sum().backward()
    assert history.grad.abs().sum() > 0
    assert history.grad[:, :18].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gate_cannot_pass_only_by_improving_startup_or_clean_history():
    rows = []
    for take in ("a", "b", "c"):
        rows.append(
            {
                "take": take,
                "frame": 40,
                "kind": "clean",
                "window": 0,
                "p_body_mm": 5.0,
                "cv_body_mm": 20.0,
                "delta_mm": -15.0,
            }
        )
        for window in (1, 3):
            for frame, delta in [(20, -100.0), (40, 2.0), (60, 2.0)]:
                rows.append(
                    {
                        "take": take,
                        "frame": frame,
                        "kind": "predicted",
                        "window": window,
                        "p_body_mm": 110.0 + delta,
                        "cv_body_mm": 110.0,
                        "delta_mm": delta,
                    }
                )
    assert not summarize_prior_rows(rows)["gate_passed"]
    for row in rows:
        if row["kind"] == "predicted" and row["frame"] > 20:
            row.update(p_body_mm=108.0, delta_mm=-2.0)
    assert summarize_prior_rows(rows)["gate_passed"]


def test_initial_predictor_and_cv_are_evaluated_with_identical_batch_arithmetic():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    example = clean_batch(codec, TrainingBatch((episode,)))
    rows = [
        {"take": take, "frame": 40, "kind": kind, "window": window, "cv_body_mm": 999.0}
        for take in ["a", "b"]
        for kind, window in [("clean", 0), ("predicted", 1), ("predicted", 3)]
    ]
    bank = {"rows": rows, "tensors": concatenate_examples([example] * len(rows))}
    model = HistoryPrior(width=24, layers=1, heads=4, source_flags=True).eval()
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    report = evaluate_prior(model, bank, codec, DenseJoints(), context)
    assert all(row["delta_mm"] == 0 for row in report["rows"])
    assert not report["summary"]["gate_passed"]
