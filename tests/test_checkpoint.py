import copy

import pytest
import torch

from egorecover import ModelConfig, WindowConfig
from egorecover.checkpoint import LEGACY_TEXT, load_e7_weights
from egorecover.models import MotionModel


def fixture():
    model = MotionModel(WindowConfig(), ModelConfig.compact())
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name not in model.NEW_PARAMETER_NAMES
    }
    state.update({name: torch.zeros(1) for name in LEGACY_TEXT})
    parameters = [
        parameter.detach().clone() + 0.25
        for name, parameter in model.named_parameters()
        if name not in model.NEW_PARAMETER_NAMES
    ]
    checkpoint = {
        "state_dict": {"model." + key: value for key, value in state.items()},
        "ema_state_format": "original_model_with_ema_optimizer_v1",
        "optimizer_states": [{"ema": parameters}],
    }
    return model, checkpoint


def test_ordered_e7_ema_is_applied_without_text_branches_or_rng_changes():
    model, checkpoint = fixture()
    rng = torch.get_rng_state().clone()
    report = load_e7_weights(model, checkpoint, weight_source="ema")
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert set(report.ignored_parameters) == LEGACY_TEXT
    assert not any("text" in name for name in model.state_dict())
    params = [p for name, p in model.named_parameters() if name not in model.NEW_PARAMETER_NAMES]
    for actual, expected in zip(params, checkpoint["optimizer_states"][0]["ema"]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(torch.count_nonzero(model.state_dict()[name]) == 0 for name in model.NEW_PARAMETER_NAMES)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "ema_count", "ema_shape", "format"])
def test_bad_e7_checkpoint_fails_before_changing_model(fault):
    model, checkpoint = fixture()
    before = copy.deepcopy(model.state_dict())
    if fault == "missing":
        del checkpoint["state_dict"]["model.input_process.weight"]
    elif fault == "extra":
        checkpoint["state_dict"]["unrelated"] = torch.zeros(1)
    elif fault == "shape":
        checkpoint["state_dict"]["model.input_process.weight"] = torch.zeros(1)
    elif fault == "ema_count":
        checkpoint["optimizer_states"][0]["ema"].pop()
    elif fault == "ema_shape":
        checkpoint["optimizer_states"][0]["ema"][0] = torch.zeros(1)
    elif fault == "format":
        checkpoint["ema_state_format"] = "unknown"
    with pytest.raises(ValueError):
        load_e7_weights(model, checkpoint, weight_source="ema")
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
