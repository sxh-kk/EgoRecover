"""Load published E7/P weights; native v2 checkpoints use strict loading."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike

import torch

from .models import HistoryPrior, MotionModel

FORMAT = "egorecover_window_v2"
LEGACY_TEXT = frozenset({"embed_text_cond.weight", "embed_text_cond.bias", "mask_tokens.text"})


def resolve_prior_mode(mode, path):
    if mode not in ("auto", "constant_velocity", "learned"):
        raise ValueError("Prior mode must be auto, constant_velocity or learned.")
    resolved = ("learned" if path is not None else "constant_velocity") if mode == "auto" else mode
    if resolved == "learned" and path is None:
        raise ValueError("Learned P requires --prior-checkpoint trained for this dataset identity.")
    if resolved == "constant_velocity" and path is not None:
        raise ValueError("Constant-velocity P does not load --prior-checkpoint.")
    return resolved


@dataclass(frozen=True)
class MigrationReport:
    weight_source: str
    loaded_tensors: int
    new_parameters: tuple[str, ...]
    ignored_parameters: tuple[str, ...]


def load_predictor(path, device, manifest=None):
    if path is None:
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if manifest is not None and saved.get("manifest_sha256") != manifest:
        raise ValueError("Use the calibrated P checkpoint for this data manifest.")
    prior = HistoryPrior(**saved.get("model_config", {})).to(device)
    prior.load_state_dict(saved["state_dict"], strict=True)
    return prior.freeze()


@torch.no_grad()
def load_e7_weights(model, checkpoint, *, weight_source="ema"):
    if not isinstance(model, MotionModel):
        raise TypeError("Expected the window MotionModel.")
    if weight_source not in ("model", "ema"):
        raise ValueError("weight_source must be model or ema.")
    if isinstance(checkpoint, (str, PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a path or mapping.")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint requires a nonempty state_dict.")
    if all(isinstance(key, str) and key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    destination = model.state_dict()
    required = set(destination) - model.NEW_PARAMETER_NAMES
    ignored = set(state) & LEGACY_TEXT
    missing, extra = required - state.keys(), state.keys() - required - LEGACY_TEXT
    if missing or extra:
        raise ValueError(f"Invalid E7 keys: missing={sorted(missing)}, extra={sorted(extra)}.")
    for name in required:
        if not isinstance(state[name], torch.Tensor) or state[name].shape != destination[name].shape:
            raise ValueError(f"E7 tensor shape mismatch at {name}.")
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if name not in model.NEW_PARAMETER_NAMES
    ]
    ema = None
    if weight_source == "ema":
        if checkpoint.get("ema_state_format") != "original_model_with_ema_optimizer_v1":
            raise ValueError("E7 EMA requires original_model_with_ema_optimizer_v1 format.")
        optimizers = checkpoint.get("optimizer_states", [])
        if not optimizers or "ema" not in optimizers[0]:
            raise ValueError("Missing optimizer_states[0]['ema'].")
        ema = optimizers[0]["ema"]
        if len(ema) != len(parameters):
            raise ValueError("E7 EMA parameter count mismatch.")
        for (name, parameter), value in zip(parameters, ema):
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"E7 EMA tensor shape mismatch at {name}.")
    incompatible = model.load_state_dict({key: state[key] for key in required}, strict=False)
    if set(incompatible.missing_keys) != model.NEW_PARAMETER_NAMES or incompatible.unexpected_keys:
        raise RuntimeError("Unexpected E7 loading result.")
    if ema is not None:
        for (_, parameter), value in zip(parameters, ema):
            parameter.copy_(value.to(parameter))
    model.reset_history_parameters()
    return MigrationReport(
        weight_source, len(required), tuple(sorted(model.NEW_PARAMETER_NAMES)), tuple(sorted(ignored))
    )
