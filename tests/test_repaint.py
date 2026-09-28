import copy

import pytest
import torch
from torch import nn

from mydiffusion.flow_matching import FlowMatching


class CoupledModel(nn.Module):
    """Each frame depends on history, so final-only pasting fails these checks."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, t, y):
        assert "repaint_mask" not in y and "repaint_value" not in y
        self.inputs.append((x.clone(), t.clone()))
        return x.mean(dim=1, keepdim=True).expand_as(x) * 0.25


@pytest.fixture
def case():
    g = torch.Generator().manual_seed(17)
    noise = torch.randn(2, 5, 243, generator=g)
    value = torch.randn(2, 5, 243, generator=g)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :2, :198] = True
    y = {"valid_frames": torch.ones(2, 5), "repaint_mask": mask, "repaint_value": value}
    return noise, y


def test_disabled_is_bitwise_baseline_and_does_not_mutate_inputs(case):
    noise, y = case
    saved = copy.deepcopy(y)
    original_noise = noise.clone()
    flow = FlowMatching()
    plain = {"valid_frames": y["valid_frames"]}
    baseline = flow.sample_loop(CoupledModel(), noise.shape, {"y": plain}, noise=noise)
    disabled = flow.sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise)
    assert torch.equal(baseline, disabled)
    flow.sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise, repaint_enabled=True)
    assert torch.equal(noise, original_noise)
    assert y.keys() == saved.keys()
    for key in y:
        assert torch.equal(y[key], saved[key])


def test_enabled_follows_path_on_every_model_call_and_output(case):
    noise, y = case
    mask, known = y["repaint_mask"], y["repaint_value"]
    model = CoupledModel()
    outputs = list(
        FlowMatching(repaint_enabled=True).sample_loop_progressive(model, noise.shape, {"y": y}, noise=noise)
    )
    assert len(outputs) == len(model.inputs) == 10
    for (x, t), output in zip(model.inputs, outputs):
        expected = (1 - t[:, None, None]) * known + t[:, None, None] * noise
        torch.testing.assert_close(x[mask], expected[mask], atol=0, rtol=0)
        next_t = output["next_t"][:, None, None]
        expected_next = (1 - next_t) * known + next_t * noise
        torch.testing.assert_close(output["sample"][mask], expected_next[mask], atol=0, rtol=0)
        assert torch.equal(output["pred_xstart"][mask], known[mask])
    assert torch.equal(outputs[-1]["sample"][mask], known[mask])
    baseline = FlowMatching().sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise)
    assert not torch.equal(outputs[-1]["sample"][~mask], baseline[~mask])


def test_per_call_override_and_no_history(case):
    noise, y = case
    flow = FlowMatching(repaint_enabled=True)
    release = flow.sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise, repaint_enabled=False)
    plain = {"valid_frames": y["valid_frames"]}
    cold = flow.sample_loop(CoupledModel(), noise.shape, {"y": plain}, noise=noise)
    assert torch.equal(release, cold)
    y["repaint_mask"] = torch.zeros_like(noise, dtype=torch.bool)
    empty = flow.sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise)
    assert torch.equal(empty, cold)


def test_full_mask_and_returned_clean_estimates(case):
    noise, y = case
    y["repaint_mask"] = torch.ones_like(noise)
    result, estimates = FlowMatching().sample_loop(
        CoupledModel(), noise.shape, {"y": y}, noise=noise, repaint_enabled=True, return_all_pred_xstart=True
    )
    assert torch.equal(result, y["repaint_value"])
    assert len(estimates) == 10
    assert all(torch.equal(p, y["repaint_value"]) for p in estimates)


@pytest.mark.parametrize("fault", ["missing_mask", "missing_value", "shape", "soft_mask", "nan_value"])
def test_invalid_enabled_constraints_fail(case, fault):
    noise, y = case
    if fault == "missing_mask":
        del y["repaint_mask"]
    elif fault == "missing_value":
        del y["repaint_value"]
    elif fault == "shape":
        y["repaint_value"] = y["repaint_value"][:, :1]
    elif fault == "soft_mask":
        y["repaint_mask"] = y["repaint_mask"].float() * 0.5
    else:
        y["repaint_value"][y["repaint_mask"]] = float("nan")
    with pytest.raises(ValueError):
        FlowMatching(repaint_enabled=True).sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise)


def test_switch_off_ignores_even_stale_constraints(case):
    noise, y = case
    y["repaint_mask"] = torch.tensor(float("nan"))
    del y["repaint_value"]
    result = FlowMatching().sample_loop(CoupledModel(), noise.shape, {"y": y}, noise=noise)
    assert torch.isfinite(result).all()


def test_training_constraints_are_not_silently_used(case):
    noise, y = case
    with pytest.raises(ValueError, match="inference-only"):
        FlowMatching(repaint_enabled=True).training_losses(CoupledModel(), noise, {"y": y})
    result = FlowMatching(repaint_enabled=False).training_losses(
        CoupledModel(), noise, {"y": y}, t=torch.tensor([0.5, 0.7])
    )
    assert torch.isfinite(result["loss"]).all()


def test_weighted_loss_and_padding():
    class Zero(nn.Module):
        def forward(self, x, t, y):
            return torch.zeros_like(x)

    target = torch.zeros(2, 3, 243)
    target[0, 0, :198] = 1
    target[0, 0, 198:207] = 2
    target[:, 1:] = 10000  # Padding must not affect the loss.
    valid = torch.tensor([[1, 0, 0], [0, 0, 0]])
    result = FlowMatching().training_losses(
        Zero(), target, {"y": {"valid_frames": valid}}, noise=torch.zeros_like(target), t=torch.ones(2) * 0.5
    )
    expected = (198 + 9 * 4 * 8) / (234 + 9 * 8)
    assert result["loss"][0].item() == pytest.approx(expected)
    assert result["loss"][1].item() == 0


def test_oracle_euler_recovers_clean_motion(case):
    noise, y = case
    clean = y["repaint_value"]

    class Oracle(nn.Module):
        def forward(self, x, t, y):
            return clean

    result = FlowMatching().sample_loop(Oracle(), noise.shape, {"y": {}}, noise=noise)
    torch.testing.assert_close(result, clean, atol=1e-6, rtol=1e-6)
