import copy

import pytest
import torch
from torch import nn

from config.defaults import get_cfg_defaults
from egorecover.actions import Action, action_equivalence, action_masks
from egorecover.checkpoint import load_e7_weights
from egorecover.conditioning import build_conditioning, prepare_conditioning
from egorecover.history_flow import HistoryFlow
from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import UniEgoMotion
from mydiffusion.flow_matching import FlowMatching


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(62)
    return HistoryUniEgoMotion(get_cfg_defaults(), dropout=0.0)


@pytest.fixture
def model(backbone):
    backbone.zero_grad(set_to_none=True)
    backbone.eval()
    return backbone


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(18)
    return dict(
        history_motion=torch.randn(2, 20, 243, generator=generator),
        history_valid=torch.ones(2, 20, dtype=torch.bool),
        prior_mu=torch.randn(2, 1, 243, generator=generator),
        traj=torch.randn(2, 1, 18, generator=generator),
        img_embs=torch.randn(2, 1, 1024, generator=generator),
        img_available=torch.ones(2, 1, dtype=torch.bool),
        traj_available=torch.ones(2, 1, dtype=torch.bool),
    )


def test_action_bits_and_missing_equivalence():
    img_available = torch.tensor([[True], [False], [True], [False]])
    traj_available = torch.tensor([[True], [True], [False], [False]])
    expected = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1], [0, 0, 2, 2], [0, 0, 0, 0]])
    assert torch.equal(action_equivalence(img_available=img_available, traj_available=traj_available), expected)
    available = torch.ones(4, 1, dtype=torch.bool)
    img_mask, traj_mask = action_masks(torch.arange(4), img_available=available, traj_available=available)
    assert img_mask[:, 0].tolist() == [False, False, True, True]
    assert traj_mask[:, 0].tolist() == [False, True, False, True]


def test_real_backbone_training_and_new_branch_gradients(model, inputs):
    model.train()  # Dropout=0 fixture also checks no hidden random condition masking.
    y = build_conditioning(**inputs)
    flow = HistoryFlow(sigma=0.3)
    epsilon = torch.randn_like(y["prior_mu"])
    target = torch.randn_like(epsilon)
    result = flow.training_losses(
        model, target, y, epsilon=epsilon, t=torch.tensor([0.4, 0.8]), return_diagnostics=True
    )
    assert result["model_output"].shape == (2, 1, 243)
    assert torch.isfinite(result["loss"]).all()
    result["loss"].mean().backward()
    parameters = dict(model.named_parameters())
    for name in ("input_process.weight", "tsfm.11.ff.net.4.weight", *model.NEW_PARAMETER_NAMES):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    assert model.embed_text_cond.weight.grad is None  # Legacy text branch stays frozen.


@pytest.mark.parametrize(
    "action,masked_fields", [(Action.A10, ["traj"]), (Action.A01, ["img_embs"]), (Action.A00, ["traj", "img_embs"])]
)
def test_masked_payload_cannot_change_output_or_poison_gradients(model, inputs, action, masked_fields):
    y = build_conditioning(**inputs, action=action)
    x, t = torch.randn(2, 1, 243), torch.tensor([0.2, 0.6])
    with torch.no_grad():
        reference = model(x, t, y)
        changed = copy.deepcopy(y)
        for name in masked_fields:
            changed[name].fill_(float("nan"))
        actual = model(x, t, changed)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    actual = model(x, t, changed)
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_kept_observations_and_history_are_used(model, inputs):
    y = build_conditioning(**inputs)
    x, t = torch.randn(2, 1, 243), torch.tensor([0.3, 0.5])
    with torch.no_grad():
        reference = model(x, t, y)
        for field in ("history_motion", "traj", "img_embs"):
            changed = dict(y)
            changed[field] = -y[field]
            assert not torch.allclose(model(x, t, changed), reference), field


def test_padding_does_not_leak_and_no_sensors_still_predicts(model, inputs):
    inputs["history_valid"][:, :5] = False
    inputs["img_available"].zero_()
    inputs["traj_available"].zero_()
    y = build_conditioning(**inputs)
    assert y["valid_frames"].all()
    x, t = torch.randn(2, 1, 243), torch.tensor([0.7, 0.9])
    with torch.no_grad():
        reference = model(x, t, y)
        y["history_motion"][:, :5] = float("inf")
        result = model(x, t, y)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_single_frame_euler_keeps_history_mu_masks_and_noise_fixed(model, inputs):
    y = build_conditioning(**inputs, action=torch.tensor([0, 3]))
    epsilon = torch.randn(2, 1, 243)
    saved, noise_before = copy.deepcopy(y), epsilon.clone()
    full_inputs, current_shapes = [], []

    def capture_motion(module, args):
        full_inputs.append(args[0].detach().clone())

    def capture_current(module, args, kwargs):
        current_shapes.append(args[0].shape)
        for name in ("prior_mu", "img_mask", "traj_mask"):
            assert torch.equal(kwargs["y"][name], saved[name])

    motion_hook = model.input_process.register_forward_pre_hook(capture_motion)
    current_hook = model.register_forward_pre_hook(capture_current, with_kwargs=True)
    try:
        output, estimates = HistoryFlow(sigma=0.3).sample(model, y, epsilon=epsilon, return_all_pred_xstart=True)
    finally:
        motion_hook.remove()
        current_hook.remove()
    assert output.shape == (2, 1, 243) and torch.isfinite(output).all()
    assert len(full_inputs) == len(current_shapes) == len(estimates) == 10
    assert all(shape == (2, 1, 243) for shape in current_shapes)
    assert all(value.shape == (2, 21, 243) for value in full_inputs)
    assert all(torch.equal(value[:, :20], saved["history_motion"]) for value in full_inputs)
    assert torch.equal(epsilon, noise_before)
    assert all(torch.equal(y[name], saved[name]) for name in y)


def test_sources_share_mu_and_generator_is_reproducible(inputs):
    mu = inputs["prior_mu"]
    epsilon = torch.randn_like(mu)
    gaussian, history = HistoryFlow(source_mode="gaussian", sigma=0.3), HistoryFlow(sigma=0.3)
    torch.testing.assert_close(
        history.build_source(mu, epsilon=epsilon) - gaussian.build_source(mu, epsilon=epsilon), mu
    )
    first = history.build_source(mu, generator=torch.Generator().manual_seed(42))
    second = history.build_source(mu, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="not both"):
        history.build_source(mu, epsilon=epsilon, generator=torch.Generator())


def test_current_loss_weighting_and_unlabelled_nan_targets(inputs):
    class Zero(nn.Module):
        def forward(self, x, t, y):
            return torch.zeros_like(x)

    y = build_conditioning(**inputs, loss_mask=torch.tensor([[1], [0]]))
    target = torch.ones(2, 1, 243)
    target[0, :, 198:207] = 2
    target[1] = float("nan")
    result = HistoryFlow().training_losses(Zero(), target, y, epsilon=torch.zeros_like(target), t=0.5)
    assert result["loss"].tolist() == pytest.approx([(234 + 9 * 4 * 8) / (234 + 9 * 8), 0])


@pytest.mark.parametrize("fault", ["long_loss", "long_current", "extra_gt", "kept_nan", "invalid_loss", "soft_mask"])
def test_invalid_conditioning_fails_early(inputs, fault):
    y = build_conditioning(**inputs)
    if fault == "long_loss":
        y["loss_mask"] = torch.ones(2, 21)
    elif fault == "long_current":
        y["traj"] = torch.zeros(2, 21, 18)
    elif fault == "extra_gt":
        y["gt_motion"] = torch.zeros(2, 1, 243)
    elif fault == "kept_nan":
        y["img_embs"][0, 0, 0] = float("nan")
    elif fault == "invalid_loss":
        y["valid_frames"].zero_()
        y["loss_mask"] = torch.ones(2, 1)
    else:
        y["img_mask"] = torch.full((2, 1), 0.5)
    with pytest.raises(ValueError):
        prepare_conditioning(y)


def test_unsupported_cfg_repaint_and_training_mode_sampling(model, inputs):
    y = build_conditioning(**inputs)
    with pytest.raises(ValueError, match="CFG"):
        model(torch.zeros(2, 1, 243), torch.ones(2), y, cond_scale=1.0)
    with pytest.raises(ValueError, match="repaint"):
        HistoryFlow(flow=FlowMatching(repaint_enabled=True))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        HistoryFlow().sample(model, y)


def test_checkpoint_migration_raw_lightning_and_ema(model):
    original = UniEgoMotion(get_cfg_defaults(), dropout=0.0)
    state = original.state_dict()
    cpu_rng = torch.random.get_rng_state().clone()
    report = load_e7_weights(model, state)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert set(report.new_parameters) == model.NEW_PARAMETER_NAMES
    assert sum(p.numel() for name, p in model.named_parameters() if name in report.new_parameters) == 779520
    destination = model.state_dict()
    assert all(torch.equal(value, destination[name]) for name, value in state.items())
    assert all(torch.count_nonzero(destination[name]) == 0 for name in report.new_parameters)

    checkpoint = {"state_dict": {"model." + name: value for name, value in state.items()}}
    with pytest.raises(ValueError, match="EMA initialization"):
        load_e7_weights(model, checkpoint, weight_source="ema")
    checkpoint["ema_state_format"] = "original_model_with_ema_optimizer_v1"
    ema = [p.detach().clone() for p in original.parameters() if p.requires_grad]
    ema[0] = ema[0] + 0.125
    checkpoint["optimizer_states"] = [{"ema": ema}]
    report = load_e7_weights(model, checkpoint, weight_source="ema")
    assert report.weight_source == "ema"
    assert torch.equal(model.input_process.weight, ema[0])
    assert torch.equal(model.embed_text_cond.weight, original.embed_text_cond.weight)
    checkpoint["optimizer_states"][0]["ema"] = ema[:-1]
    with pytest.raises(ValueError, match="EMA has"):
        load_e7_weights(model, checkpoint, weight_source="ema")
    broken = dict(state)
    del broken["input_process.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_e7_weights(model, broken)
