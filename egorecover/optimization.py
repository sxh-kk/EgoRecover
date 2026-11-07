"""Explicit optimizer settings and the pinned UniEgoMotion LR reference."""

import math
from numbers import Integral, Real

import torch

from .config import TrainOptions

UEM_COMMIT = "580c92c6d70a91672c4106bab30ca82cdd80f379"
UEM_ROOT = f"https://github.com/chaitanya100100/UniEgoMotion/blob/{UEM_COMMIT}"


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    invalid_range = value <= 0 if positive else value < 0
    if invalid_range:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}.")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def optimization_protocol(args, reference_clips=None):
    """Describe optimization, without compressing an epoch schedule into a pilot.

    ``reference_clips`` is supplied only for the official training-window
    inventory. Sampling online target queries has different supervision volume
    from the upstream sequence task, so this is an explicit update reference,
    not a claim that the two objectives have equivalent epochs.
    """

    def option(name):
        return getattr(args, name, getattr(TrainOptions, name))

    learning_rate = _finite_number(option("learning_rate"), "learning_rate", positive=True)
    weight_decay = _finite_number(option("weight_decay"), "weight_decay")
    clip_norm = _finite_number(option("grad_clip_norm"), "grad_clip_norm", positive=True)
    gamma = _finite_number(option("lr_decay_gamma"), "lr_decay_gamma", positive=True)
    if gamma >= 1:
        raise ValueError("lr_decay_gamma must be strictly between zero and one.")
    batch_size = _positive_integer(option("batch_size"), "batch_size")
    mode = option("lr_schedule")
    if mode not in ("constant", "uniegomotion"):
        raise ValueError("lr_schedule must be constant or uniegomotion.")
    explicit_step = option("lr_decay_step")
    if explicit_step is not None:
        explicit_step = _positive_integer(explicit_step, "lr_decay_step")
    if mode == "constant" and explicit_step is not None:
        raise ValueError("A constant LR schedule cannot specify lr_decay_step.")
    steps_per_epoch = None
    if reference_clips is not None:
        reference_clips = _positive_integer(reference_clips, "reference_clips")
        steps_per_epoch = (reference_clips + batch_size - 1) // batch_size
    decay_step = None
    if mode == "uniegomotion":
        decay_step = (
            explicit_step
            if explicit_step is not None
            else (steps_per_epoch * 300 if steps_per_epoch is not None else None)
        )
    return {
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "grad_clip_norm": clip_norm,
        "scheduler": {"mode": mode, "decay_step": decay_step, "gamma": gamma},
        "reference": {
            "repository": "chaitanya100100/UniEgoMotion",
            "commit": UEM_COMMIT,
            "source_links": {
                "config": f"{UEM_ROOT}/config/uem.yaml",
                "optimizer_scheduler": f"{UEM_ROOT}/module/uem_module.py",
                "gradient_clip_ema_seed": f"{UEM_ROOT}/run/train_uem.py",
            },
            "train_epochs": 350,
            "decay_epoch": 300,
            "source_clips": reference_clips,
            "reference_steps_per_epoch": steps_per_epoch,
            "full_reference_updates": steps_per_epoch * 350 if steps_per_epoch is not None else None,
            "interpretation": "official window-count update reference; not equivalent supervised-body volume",
        },
    }


def make_optimizer(parameters, protocol):
    """Create AdamW with all comparison-relevant parameters explicit."""
    return torch.optim.AdamW(
        parameters,
        lr=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"],
        betas=tuple(protocol["betas"]),
        eps=protocol["eps"],
    )


def make_scheduler(optimizer, protocol):
    """Step once per optimizer update; a missing milestone means constant LR."""
    settings = protocol["scheduler"]
    if settings["decay_step"] is None:
        return None
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings["decay_step"], gamma=settings["gamma"])
