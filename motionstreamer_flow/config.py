import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    pose_dim: int = 152
    latent_dim: int = 16
    image_dim: int = 1024
    head_dim: int = 18
    history: int = 20
    bootstrap: int = 20
    target_frames: int = 32
    tae_width: int = 512
    tae_dilations: tuple = (1, 2, 4)
    width: int = 768
    layers: int = 12
    heads: int = 12
    fusion_layers: int = 2
    flow_width: int = 1792
    flow_layers: int = 9
    dropout: float = 0.0
    checkpoint_layers: bool = True
    flow_steps: int = 10
    tau_min: float = 0.001
    kl_weight: float = 0.0001
    geometry_weight: float = 1.0
    root_weight: float = 8.0
    two_forward: bool = True
    modality_drop: float = 0.1
    batch_size: int = 2
    accumulation: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    total_steps: int = 24000
    eval_every: int = 500
    seed: int = 62
    fps: int = 10
    precision: str = "bf16"

    def __post_init__(self):
        if self.pose_dim != 152 or self.head_dim != 18:
            raise ValueError("This representation requires pose=152 and head=18.")
        positive = (
            self.history,
            self.bootstrap,
            self.target_frames,
            self.latent_dim,
            self.layers,
            self.flow_steps,
            self.batch_size,
            self.accumulation,
            self.total_steps,
            self.eval_every,
        )
        if min(positive) < 1 or self.heads < 1 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Invalid lengths, or attention head width must be even.")
        if self.precision not in ("bf16", "fp16", "fp32") or not 0 < self.tau_min < 1:
            raise ValueError("Invalid precision or flow time lower bound.")
        if not 0 <= self.modality_drop < 1 or self.dropout < 0 or self.dropout >= 1:
            raise ValueError("Drop probabilities must be in [0,1).")

    @property
    def state_dim(self):
        return self.latent_dim + 9

    @property
    def receptive_field(self):
        return 1 + 4 + 2 * sum(self.tae_dilations)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        if "tae_dilations" in values:
            values["tae_dilations"] = tuple(values["tae_dilations"])
        return cls(**values)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
