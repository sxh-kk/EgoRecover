import pytest
import torch

from egorecover.codec import BodyState, rigid_transform
from egorecover.diagnostics import DenseJoints
from scripts.diagnose_p01_gradients import (
    GROUPS,
    aggregate_metrics,
    code_vectors,
    gradient_relationships,
    paired_loss_vectors,
    phase_gradient_audit,
    sample_train_indices,
    split_phases,
)


class OffsetPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(243))

    def forward(self, history, valid, base, *, history_body_valid):
        return base + self.offset[None, None]


class ToyCodec:
    def decode_current(self, prediction, reference):
        rotation = torch.eye(3).expand(len(prediction), 22, 3, 3)
        positions = prediction[:, None, :3].expand(-1, 22, -1)
        return BodyState(rigid_transform(rotation, positions), reference, prediction[:, -36:])


def bank():
    rows = [
        {
            "episode": f"e{i}",
            "take": f"take{i % 2}",
            "source": f"s{i}",
            "kind": "predicted",
            "window": 3,
            "frame": 40 if i < 6 else 20,
        }
        for i in range(8)
    ]
    target = torch.zeros(8, 1, 243)
    target[:6, :, 0], target[6:, :, 0] = -1, 10
    xyz = torch.zeros(8, 22, 3)
    xyz[:6, :, 0], xyz[6:, :, 0] = -1, 10
    return {
        "rows": rows,
        "tensors": {
            "history": torch.zeros(8, 20, 243),
            "body_valid": torch.ones(8, 20, dtype=torch.bool),
            "base": torch.zeros(8, 1, 243),
            "target": target,
            "truth_xyz": xyz,
            "reference": torch.eye(4).expand(8, 4, 4),
        },
    }


def test_code_groups_partition243_and_contributions_reproduce_weighted_loss():
    assert sorted(index for values in GROUPS.values() for index in values) == list(range(243))
    prediction = torch.arange(243).float()[None, None]
    result = code_vectors(prediction, torch.zeros_like(prediction))
    assert torch.allclose(sum(result[f"code_contribution/{name}"] for name in GROUPS), result["code_loss"])
    assert result["code_mse/reference9"] > 0


def test_phase_weighted_sum_matches_full_gradient_and_detects_startup_domination():
    data, model, codec, fk = bank(), OffsetPrior(), ToyCodec(), DenseJoints()
    selected = split_phases(data, list(range(8)))
    before = model.offset.detach().clone()
    audits = {
        phase: phase_gradient_audit(model, data, indices, codec, fk, 0.03, microbatch=2)
        for phase, indices in selected.items()
    }
    example = data["tensors"]
    prediction = model(example["history"], None, example["base"], history_body_valid=None)
    vectors = paired_loss_vectors(prediction, example, codec, fk, 0.03)
    actual = torch.autograd.grad(vectors["P"]["total_loss"].mean(), model.parameters())[0].double()
    expected = sum(
        weight * (audits[phase]["gradients"]["code"] + 0.03 * audits[phase]["gradients"]["geometry"])
        for phase, weight in (("mature", 0.75), ("startup", 0.25))
    )
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-6)
    analysis = gradient_relationships(audits, 0.03)
    report = analysis["quota_weighted_relationships"]["weighted_total"]
    assert report["phase_cosine"] == pytest.approx(-1)
    assert report["weighted_startup_over_mature_norm"] == pytest.approx(10 / 3)
    assert report["sum_opposes_mature"] and not report["sum_opposes_startup"]
    assert report["cos_sum_mature"] == pytest.approx(-1)
    assert analysis["mixed_objective_direction"]["mature_geometry"]["mixed_descent_increases_this_loss_locally"]
    assert not analysis["mixed_objective_direction"]["startup_geometry"]["mixed_descent_increases_this_loss_locally"]
    torch.testing.assert_close(model.offset, before, atol=0, rtol=0)
    assert model.offset.grad is None


def test_fixed_sampler_uses_exact75_25_with_reproducible_hierarchical_draws():
    data = bank()
    first = sample_train_indices(data, samples=1024)
    assert first == sample_train_indices(data, samples=1024)
    phases = split_phases(data, first)
    assert len(phases["mature"]) == 768 and len(phases["startup"]) == 256


def test_row_equal_and_clip_equal_are_explicitly_different_estimands():
    rows = [
        {
            "index": index,
            "episode": episode,
            "take": episode,
            "source": episode,
            "frame": 40,
            "P": {"code_loss": value},
            "CV": {"code_loss": 0},
        }
        for index, (episode, value) in enumerate((("long", 1), ("long", 1), ("short", 10)))
    ]
    assert aggregate_metrics(rows)["mature"]["P"]["code_loss"] == 4
    assert aggregate_metrics(rows, clip_equal=True)["mature"]["P"]["code_loss"] == 5.5
