import copy

import pytest
import torch
from torch import nn

from egorecover.checkpoint import load_predictor
from egorecover.codec import BodyState, MotionCodec
from egorecover.data import TrainingBatch
from egorecover.diagnostics import DenseJoints, synthetic_episode
from egorecover.io import state_sha256
from egorecover.models import HistoryPrior
from egorecover.prior_data import clean_batch, predictor_forward
from egorecover.prior_objectives import common_cv_valid_state_codes, prior_objective_losses
from egorecover.prior_training import prior_losses


def fixture():
    episode = synthetic_episode()
    codec = MotionCodec(episode.stats)
    example = clean_batch(codec, TrainingBatch((episode,)))
    return codec, example


def yaw(angle):
    angle = torch.as_tensor(angle)
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one)).reshape(3, 3)


def predicted_codes(codec, example, *, reference_angle=0.4):
    state = codec.decode_current(example["base"][:, 0], example["reference"])
    joints = state.joints.clone()
    joints[..., :3, :3] = yaw(0.3)
    joints[..., :3, 3] += 0.02
    reference = state.reference.clone()
    reference[..., :3, :3] = yaw(reference_angle)
    return codec.encode_current(BodyState(joints, reference, state.auxiliary), example["reference"])[:, None]


class FixedPrediction(nn.Module):
    def __init__(self, codes):
        super().__init__()
        self.codes = nn.Parameter(codes.clone())

    def forward(self, *args, **kwargs):
        return self.codes


def small_prior(*, hold_beta=False):
    return HistoryPrior(width=24, layers=1, heads=4, dropout=0.0, source_flags=True, hold_beta=hold_beta).eval()


def test_raw_objective_matches_existing_p01_loss_and_gradients():
    codec, example = fixture()
    model = small_prior()
    with torch.no_grad():
        model.output.weight.normal_(std=0.001)
    first = prior_losses(model, example, codec, DenseJoints())
    other = prior_objective_losses(model, example, codec, DenseJoints(), objective="raw_code")
    assert first.keys() == other.keys()
    parameters = tuple(model.parameters())
    for key in first:
        torch.testing.assert_close(first[key], other[key], atol=0, rtol=0)
    a = torch.autograd.grad(first["code"] + 0.01 * first["geometry"], parameters, allow_unused=True)
    b = torch.autograd.grad(other["code"] + 0.01 * other["geometry"], parameters, allow_unused=True)
    for expected, actual in zip(a, b):
        assert (expected is None) == (actual is None)
        if expected is not None:
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)


def test_common_cv_reference_codes_preserve_world_states_and_zero_reference_residual():
    codec, example = fixture()
    prediction = predicted_codes(codec, example)
    codes, target, physical = common_cv_valid_state_codes(prediction, example, codec)
    decoded_prediction = codec.decode_current(codes[:, 0], example["reference"])
    decoded_target = codec.decode_current(target[:, 0], example["reference"])
    original_target = codec.decode_current(example["target"][:, 0], example["reference"])
    assert codes.shape == target.shape == (1, 1, 243)
    torch.testing.assert_close(decoded_prediction.joints, physical.joints, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded_prediction.auxiliary, physical.auxiliary, atol=0, rtol=0)
    torch.testing.assert_close(decoded_target.joints, original_target.joints, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded_target.auxiliary, original_target.auxiliary, atol=0, rtol=0)
    torch.testing.assert_close(codes[..., 198:207], target[..., 198:207], atol=0, rtol=0)


def test_common_loss_ignores_reference_gauge_of_the_same_predicted_body():
    codec, example = fixture()
    first = predicted_codes(codec, example, reference_angle=0.4)
    other = predicted_codes(codec, example, reference_angle=-0.6)
    a = prior_objective_losses(FixedPrediction(first), example, codec, DenseJoints(), objective="common_cv_valid_state")
    b = prior_objective_losses(FixedPrediction(other), example, codec, DenseJoints(), objective="common_cv_valid_state")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-6)


def test_valid_state_objective_removes_sixd_scale_gain_and_keeps_prediction_gradient():
    codec, example = fixture()
    prediction = predicted_codes(codec, example)
    raw = codec.denormalize(prediction[:, 0])
    raw[:, :198].reshape(-1, 22, 9)[..., :6] *= 0.5
    scaled = codec.normalize(raw)[:, None]
    model_a, model_b = FixedPrediction(prediction), FixedPrediction(scaled)
    a = prior_objective_losses(model_a, example, codec, DenseJoints(), objective="common_cv_valid_state")
    b = prior_objective_losses(model_b, example, codec, DenseJoints(), objective="common_cv_valid_state")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-6)
    a["code"].backward()
    gradient = model_a.codes.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert gradient[..., :198].reshape(-1, 22, 9)[..., :6].abs().sum() > 0


def test_ground_truth_changes_only_loss_targets_and_never_predictor_inputs():
    codec, example = fixture()
    changed = copy.deepcopy(example)
    truth = codec.decode_current(changed["target"][:, 0], changed["reference"])
    joints = truth.joints.clone()
    joints[..., :3, 3] += 0.2
    changed["target"] = codec.encode_current(BodyState(joints, truth.reference, truth.auxiliary), changed["reference"])[
        :, None
    ]
    changed["truth_xyz"] += 0.2
    model = small_prior()
    captured = []

    def capture(module, args, kwargs):
        captured.append(
            ([arg.detach().clone() for arg in args], {key: value.detach().clone() for key, value in kwargs.items()})
        )

    hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    first = prior_objective_losses(model, example, codec, DenseJoints(), objective="common_cv_valid_state")
    other = prior_objective_losses(model, changed, codec, DenseJoints(), objective="common_cv_valid_state")
    hook.remove()
    assert not torch.equal(first["code"], other["code"])
    assert not torch.equal(first["geometry"], other["geometry"])
    for a, b in zip(captured[0][0], captured[1][0]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in captured[0][1]:
        torch.testing.assert_close(captured[0][1][key], captured[1][1][key], atol=0, rtol=0)


def test_beta_policy_preserves_matched_initialization_and_initial_losses():
    codec, example = fixture()
    torch.manual_seed(62)
    first = small_prior(hold_beta=False)
    torch.manual_seed(62)
    other = small_prior(hold_beta=True)
    assert state_sha256(first.state_dict()) == state_sha256(other.state_dict())
    assert first.state_dict().keys() == other.state_dict().keys()
    torch.testing.assert_close(predictor_forward(first, example), example["base"], atol=0, rtol=0)
    torch.testing.assert_close(predictor_forward(other, example), example["base"], atol=0, rtol=0)
    a = prior_objective_losses(first, example, codec, DenseJoints(), objective="common_cv_valid_state")
    b = prior_objective_losses(other, example, codec, DenseJoints(), objective="common_cv_valid_state")
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_hold_beta_masks_residual_in_train_eval_and_after_freeze_without_detaching_base():
    codec, example = fixture()
    model = small_prior(hold_beta=True)
    with torch.no_grad():
        model.output.weight.normal_(std=0.002)
        model.output.bias[-10:].fill_(7)
        model.output.bias[207:231].fill_(0.1)
    base = example["base"].clone().requires_grad_(True)
    input_example = {**example, "base": base}
    for mode in (True, False):
        model.train(mode)
        prediction = predictor_forward(model, input_example)
        torch.testing.assert_close(prediction[..., -10:], base[..., -10:], atol=0, rtol=0)
        assert not torch.equal(prediction[..., 207:231], base[..., 207:231])
    prediction = predictor_forward(model, input_example)
    prediction[..., -10:].sum().backward()
    assert torch.count_nonzero(model.output.weight.grad[-10:]) == 0
    assert torch.count_nonzero(model.output.bias.grad[-10:]) == 0
    torch.testing.assert_close(base.grad[..., -10:], torch.ones_like(base.grad[..., -10:]), atol=0, rtol=0)
    model.freeze()
    torch.testing.assert_close(predictor_forward(model, input_example)[..., -10:], base[..., -10:], atol=0, rtol=0)


def test_inherited_beta_remains_part_of_shape_aware_fk():
    codec, example = fixture()
    model = small_prior(hold_beta=True)
    prediction = predictor_forward(model, example)
    body = codec.decode_current(prediction[:, 0], example["reference"])
    altered = {**example, "base": example["base"].clone()}
    altered["base"][..., -10] += 2
    other = codec.decode_current(predictor_forward(model, altered)[:, 0], example["reference"])

    def shape_dependent_fk(state):
        return state.joints[..., :3, 3] + state.auxiliary[..., -10:-9, None] * 0.01

    assert not torch.equal(shape_dependent_fk(body), shape_dependent_fk(other))
    torch.testing.assert_close(body.joints, other.joints, atol=0, rtol=0)


@pytest.mark.parametrize("hold_beta", [False, True])
def test_checkpoint_loader_preserves_beta_policy(tmp_path, hold_beta):
    _, example = fixture()
    config = {"width": 24, "layers": 1, "heads": 4, "dropout": 0.0, "source_flags": True, "hold_beta": hold_beta}
    model = HistoryPrior(**config).eval()
    with torch.no_grad():
        model.output.bias[-10:].fill_(2)
    checkpoint = tmp_path / "prior.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict(), "manifest_sha256": "fixture"}, checkpoint)
    loaded = load_predictor(checkpoint, "cpu", manifest="fixture")
    assert loaded.hold_beta is hold_beta
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    prediction = predictor_forward(loaded, example)
    torch.testing.assert_close(prediction, predictor_forward(model, example), atol=0, rtol=0)
    if hold_beta:
        torch.testing.assert_close(prediction[..., -10:], example["base"][..., -10:], atol=0, rtol=0)
    else:
        torch.testing.assert_close(prediction[..., -10:], example["base"][..., -10:] + 2, atol=0, rtol=0)


def test_unknown_objective_and_non_boolean_shape_policy_are_rejected():
    codec, example = fixture()
    with pytest.raises(ValueError, match="P objective must"):
        prior_objective_losses(small_prior(), example, codec, DenseJoints(), objective="unreviewed")
    with pytest.raises(ValueError, match="hold_beta must"):
        small_prior(hold_beta=1)
