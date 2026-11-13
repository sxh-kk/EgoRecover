"""Physical-state supervision for the history-only P, without new inputs."""

from .codec import BodyState
from .losses import geometry_loss
from .prior_data import predictor_forward

OBJECTIVES = ("raw_code", "common_cv_valid_state")


def common_cv_valid_state_codes(prediction, example, codec):
    """Loss-only P/GT encodings under the same causal current CV reference.

    Decoding projects the predicted 6D rotations into valid physical states.
    Reencoding changes their representation, not their world body transforms.
    Current GT contributes only to the target code; it never reaches P's
    forward inputs. No prediction or input gradients are detached here.
    """
    previous = example["reference"]
    predicted_state = codec.decode_current(prediction[:, 0], previous)
    cv_state = codec.decode_current(example["base"][:, 0], previous)
    truth_state = codec.decode_current(example["target"][:, 0], previous)
    common_reference = cv_state.reference

    def encode(state):
        physical = BodyState(state.joints, common_reference, state.auxiliary)
        return codec.encode_current(physical, previous)[:, None]

    return encode(predicted_state), encode(truth_state), predicted_state


def prior_objective_losses(model, example, codec, fk, *, objective="raw_code"):
    """Keep evaluation/physical output unchanged while selecting P's objective."""
    if objective not in OBJECTIVES:
        raise ValueError(f"P objective must be one of {OBJECTIVES}.")
    prediction = predictor_forward(model, example)
    if objective == "common_cv_valid_state":
        loss_prediction, target, body = common_cv_valid_state_codes(prediction, example, codec)
    else:
        loss_prediction, target = prediction, example["target"]
        body = codec.decode_current(prediction[:, 0], example["reference"])
    error = (loss_prediction - target).square()
    weights = error.new_ones(243)
    weights[198:207] = 8
    code = (error * weights).sum(-1).mean() / weights.sum()
    geometry = geometry_loss(fk(body), example["truth_xyz"], 0.1)
    return {"code": code, "geometry": geometry, "hand_pca": error[..., 207:231].mean()}
