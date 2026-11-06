"""Joint active-window endpoint prediction with the dense E7 parameter layout."""

import torch
from torch import nn

from model.core import DecoderBlock
from model.history_uniegomotion import HistoryUniEgoMotion
from model.uniegomotion import PositionalEncoding, TimestepEmbedder


class WindowUniEgoMotion(HistoryUniEgoMotion):
    def __init__(self, cfg, *, protocol, dropout=0.1, compact=False):
        self.protocol = protocol
        self.compact = compact
        if not compact:
            super().__init__(cfg, dropout=dropout, history_length=protocol.history)
        else:
            # CPU integration checks use the same forward path and attention
            # blocks. Compact checkpoints cannot be confused with full E7.
            nn.Module.__init__(self)
            self.cfg, self.dropout = cfg, dropout
            self.input_feats, self.latent_dim = 243, 48
            self.history_length = protocol.history
            self.input_process = nn.Linear(243, 48)
            self.output_process = nn.Linear(48, 243)
            self.pos_enc = PositionalEncoding(48, dropout)
            self.embed_timestep = TimestepEmbedder(48, self.pos_enc)
            self.embed_traj_cond = nn.Linear(18, 48)
            self.embed_clip_cond = nn.Linear(1024, 48)
            self.mask_tokens = nn.ParameterDict({k: nn.Parameter(torch.zeros(48)) for k in ("traj", "clip")})
            self.tsfm = nn.ModuleList([DecoderBlock(48, 4, dropout, 2) for _ in range(2)])
            self.state_embedding = nn.Embedding(2, 48)
            self.frame_time_proj = nn.Linear(48, 48)
            self.mu_proj = nn.Linear(243, 48)
            self.reset_history_parameters()

    def forward(self, x, timesteps, y, cond_scale=None):
        if cond_scale is not None:
            raise ValueError("CFG is not part of the window protocol.")
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
        if timesteps.shape != (b,) or not bool(((timesteps > 0) & (timesteps <= 1)).all()):
            raise ValueError("Flow times must be [B] in (0,1].")
        fixed = torch.where(y["fixed_valid"][..., None], y["fixed_motion"], 0)
        traj = torch.where(y["head_valid"][..., None], y["traj"], 0)
        images = torch.where(y["image_valid"][..., None], y["images"], 0)
        for value in (x, fixed, traj, images, y["center"]):
            if not bool(torch.isfinite(value).all()):
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
        for block in self.tsfm:
            motion = block(motion, memory, mask=body_valid[:, None, None], context_mask=obs_valid[:, None, None])
        return self.output_process(motion[:, -w:])
