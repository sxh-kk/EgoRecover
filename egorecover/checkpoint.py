"""Explicit E7 -> history-G weight initialization, never optimizer resume."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike

import torch

from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import UniEgoMotion


@dataclass(frozen=True)
class MigrationReport:
    weight_source: str
    loaded_tensors: int
    new_parameters: tuple[str, ...]


def _backbone_state(checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint requires a nonempty state_dict or raw backbone state.")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Backbone state must map parameter/buffer names to tensors.")
    # UEM_Module stores its only learned network under model.; raw
    # UniEgoMotion state_dicts already have the desired parameter names.
    if all(key.startswith("model.") for key in state):
        return {key.removeprefix("model."): value for key, value in state.items()}
    return dict(state)


@torch.no_grad()
def load_e7_weights(model, checkpoint, *, weight_source="model"):
    """Initialize a HistoryUniEgoMotion from a standard dense E7 checkpoint.

    checkpoint is a path or an already loaded mapping. Paths use weights_only
    loading on CPU. Choose 'model' or 'ema' explicitly; an unrecognized/missing
    EMA format raises instead of silently using raw weights. The original E7
    architecture is strictly loaded before ordered EMA tensors are applied.

    New history branches are reset to zero. Old optimizer/scheduler/EMA state
    is not restored. Use ordinary strict loading to resume a NEW G checkpoint.
    """
    if not isinstance(model, HistoryUniEgoMotion):
        raise TypeError("Expected a HistoryUniEgoMotion target.")
    if weight_source not in ("model", "ema"):
        raise ValueError("weight_source must be 'model' or 'ema'.")
    if isinstance(checkpoint, (str, PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a path or mapping.")
    state = _backbone_state(checkpoint)

    # Constructing the temporary model must not change the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        original = UniEgoMotion(model.cfg, dropout=model.dropout)
    original.load_state_dict(state, strict=True)
    if weight_source == "ema":
        if checkpoint.get("ema_state_format") != "original_model_with_ema_optimizer_v1":
            raise ValueError("EMA initialization requires original_model_with_ema_optimizer_v1 format.")
        optimizer_states = checkpoint.get("optimizer_states", [])
        if not optimizer_states or "ema" not in optimizer_states[0]:
            raise ValueError("EMA checkpoint is missing optimizer_states[0]['ema'].")
        ema = optimizer_states[0]["ema"]
        parameters = [(name, value) for name, value in original.named_parameters() if value.requires_grad]
        if len(ema) != len(parameters):
            raise ValueError(f"EMA has {len(ema)} tensors; original E7 expects {len(parameters)}.")
        for (name, parameter), value in zip(parameters, ema):
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"EMA tensor shape mismatch at {name}.")
        # Reuse the repository's format-specific helper only on the OLD model.
        from module.ema import apply_ema_weights_from_checkpoint

        apply_ema_weights_from_checkpoint(original, checkpoint)

    state = original.state_dict()
    destination = model.state_dict()
    missing, extra = destination.keys() - state.keys(), state.keys() - destination.keys()
    if missing != HistoryUniEgoMotion.NEW_PARAMETER_NAMES or extra:
        raise ValueError(f"Unexpected migration keys: missing={sorted(missing)}, extra={sorted(extra)}.")
    for name, value in state.items():
        if value.shape != destination[name].shape:
            raise ValueError(f"Backbone tensor shape mismatch at {name}.")
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise RuntimeError("Unexpected state_dict loading result.")
    model.reset_history_parameters()
    return MigrationReport(weight_source, len(state), tuple(sorted(missing)))
