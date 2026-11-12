"""The only G architecture: joint rolling-window endpoint prediction."""

import torch
from torch import nn

from egorecover.config import ModelConfig

from .layers import DecoderBlock, PositionalEncoding, TimestepEmbedder


class MotionModel(nn.Module):
    NEW_PARAMETER_NAMES = frozenset(
        {
            "state_embedding.weight",
            "frame_time_proj.weight",
            "frame_time_proj.bias",
            "mu_proj.weight",
            "mu_proj.bias",
        }
    )

    def __init__(self, protocol, model_config=None):
        super().__init__()
        self.protocol = protocol
        self.model_config = model_config or ModelConfig()
        cfg = self.model_config
        self.input_feats, self.latent_dim = 243, cfg.width
        self.input_process = nn.Linear(243, cfg.width)
        self.pos_enc = PositionalEncoding(cfg.width, cfg.dropout)
        self.embed_timestep = TimestepEmbedder(cfg.width, self.pos_enc)
        self.embed_traj_cond = nn.Linear(18, cfg.width)
        self.embed_clip_cond = nn.Linear(1024, cfg.width)
        self.mask_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(cfg.width) * 0.05) for name in ("traj", "clip")}
        )
        self.tsfm = nn.ModuleList([DecoderBlock(cfg.width, cfg.heads, cfg.dropout, 2) for _ in range(cfg.layers)])
        self.output_process = nn.Linear(cfg.width, 243)
        self.state_embedding = nn.Embedding(2, cfg.width)
        self.frame_time_proj = nn.Linear(cfg.width, cfg.width)
        self.mu_proj = nn.Linear(243, cfg.width)
        self._compiled_eval_core = None
        self.reset_history_parameters()

    def enable_eval_compile(self):
        """Compile only the no-grad decoder; training retains its eager graph."""
        self._compiled_eval_core = torch.compile(
            self._decode_layers, mode="reduce-overhead", fullgraph=True, dynamic=False
        )

    def _decode_layers(self, motion, memory, body_valid, obs_valid):
        for block in self.tsfm:
            motion = block(motion, memory, mask=body_valid[:, None, None], context_mask=obs_valid[:, None, None])
        return motion

    def reset_history_parameters(self):
        for layer in (self.state_embedding, self.frame_time_proj, self.mu_proj):
            for parameter in layer.parameters():
                nn.init.zeros_(parameter)

    def forward(self, x, timesteps, y):
        required = {"fixed_motion", "fixed_valid", "center", "traj", "images", "head_valid", "image_valid"}
        if set(y) != required:
            raise ValueError("Only documented online window conditions are accepted.")
        b, w, d = x.shape
        h, f = self.protocol.history, self.protocol.fixed
        if (w, d) != (self.protocol.window, 243):
            raise ValueError("Active window shape disagrees with protocol.")
        shapes = {
            "fixed_motion": (b, f, 243),
            "fixed_valid": (b, f),
            "center": (b, w, 243),
            "traj": (b, h + 1, 18),
            "images": (b, h + 1, 1024),
            "head_valid": (b, h + 1),
            "image_valid": (b, h + 1),
        }
        for key, shape in shapes.items():
            if y[key].shape != shape or y[key].device != x.device:
                raise ValueError(f"Invalid window condition: {key}.")
        for key in ("fixed_valid", "head_valid", "image_valid"):
            if y[key].dtype != torch.bool:
                raise ValueError(f"{key} must be boolean.")
        if timesteps.shape != (b,):
            raise ValueError("Flow times must be [B] in (0,1].")
        fixed = torch.where(y["fixed_valid"][..., None], y["fixed_motion"], 0)
        traj = torch.where(y["head_valid"][..., None], y["traj"], 0)
        images = torch.where(y["image_valid"][..., None], y["images"], 0)
        valid_time = ((timesteps > 0) & (timesteps <= 1)).all()
        finite = [torch.isfinite(value).all() for value in (x, fixed, traj, images, y["center"])]
        if not bool(torch.stack([valid_time, *finite]).all()):
            if not bool(valid_time):
                raise ValueError("Flow times must be [B] in (0,1].")
            raise ValueError("Nonfinite available model input.")
        motion = self.pos_enc(self.input_process(torch.cat((fixed, x), 1)))
        ids = torch.cat((torch.zeros(f, device=x.device), torch.ones(w, device=x.device))).long()
        motion = motion + self.state_embedding(ids)[None]
        times = torch.cat((timesteps.new_zeros(b, f), timesteps[:, None].expand(-1, w)), 1)
        motion = motion + self.frame_time_proj(self.embed_timestep(times.flatten()).reshape(b, h + 1, -1))
        # BOTH source modes receive exactly the same explicit center condition.
        motion = motion + torch.cat((motion.new_zeros(b, f, self.latent_dim), self.mu_proj(y["center"])), 1)
        head_tokens = self.embed_traj_cond(traj)
        motion = motion + torch.where(y["head_valid"][..., None], head_tokens, self.mask_tokens["traj"])
        image_tokens = self.embed_clip_cond(images)
        image_tokens = torch.where(y["image_valid"][..., None], image_tokens, self.mask_tokens["clip"])
        # Head memory remains visible even when body-history keys are masked.
        memory = self.pos_enc(
            image_tokens + torch.where(y["head_valid"][..., None], head_tokens, self.mask_tokens["traj"])
        )
        time_token = self.embed_timestep(timesteps)[:, None]
        motion, memory = torch.cat((time_token, motion), 1), torch.cat((time_token, memory), 1)
        body_valid = torch.cat(
            (
                torch.ones(b, 1, dtype=torch.bool, device=x.device),
                y["fixed_valid"],
                torch.ones(b, w, dtype=torch.bool, device=x.device),
            ),
            1,
        )
        obs_valid = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=x.device), y["head_valid"] | y["image_valid"]), 1
        )
        core = self._decode_layers
        if not self.training and not torch.is_grad_enabled() and self._compiled_eval_core is not None:
            core = self._compiled_eval_core
        motion = core(motion, memory, body_valid, obs_valid)
        # Eager projection creates independent output storage before the next
        # no-grad CUDAgraph replay; emitted codes must never be overwritten.
        return self.output_process(motion[:, -w:])
