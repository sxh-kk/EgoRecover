"""Shared controls for all four experiments; no machine-specific paths."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WindowConfig:
    history: int = 20
    window: int = 3
    source: str = "gaussian"
    sigma: float = 1.0
    steps: int = 1
    revise_recent: bool = True
    body_feedback: bool = True
    endpoint_probability: float = 0.1
    current_weight: float = 1.0
    feedback_weight: float = 1.0
    feedback: str = "none"
    feedback_draws: int = 1
    scale_m: float = 0.1
    ema_decay: float = 0.999

    def __post_init__(self):
        for key in ("history", "window", "steps", "feedback_draws"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.history != 20 or self.window not in (1, 3, 5, 10):
            raise ValueError("The plan fixes history=20 and window in 1,3,5,10.")
        if self.steps not in (1, 2, 4, 10):
            raise ValueError("Solver steps must be 1,2,4,10.")
        if self.source not in ("gaussian", "prediction"):
            raise ValueError("source must be gaussian or prediction.")
        if self.feedback not in ("none", "paired", "direct", "feature", "geometry"):
            raise ValueError("Unknown feedback objective.")
        for key in ("sigma", "scale_m", "current_weight", "feedback_weight", "endpoint_probability", "ema_decay"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}.")
        if self.sigma == 0 or self.scale_m == 0:
            raise ValueError("sigma and scale_m must be positive.")
        if self.endpoint_probability > 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid endpoint probability or EMA decay.")
        if not isinstance(self.revise_recent, bool) or not isinstance(self.body_feedback, bool):
            raise ValueError("Ablation switches must be boolean.")
        if not self.body_feedback and self.source != "gaussian":
            raise ValueError("The no-body-feedback control uses a Gaussian source.")

    @property
    def fixed(self):
        return self.history - self.window + 1

    def to_dict(self):
        return asdict(self)
