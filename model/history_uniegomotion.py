"""E7 backbone with fixed predicted history and one active Flow frame."""

import torch
from torch import nn

from egorecover.conditioning import finite_payload, floating_tensor, prepare_conditioning
from model.uniegomotion import UniEgoMotion


class HistoryUniEgoMotion(UniEgoMotion):
    """Preserve E7 parameter names; add three zero-initialized conditions.

    forward accepts/returns [B,1,243]. History is concatenated inside the
    network, outside the Flow state. Full attention is causal with respect
    to physical time when callers supply only past bodies/current sensors.
    """

    NEW_PARAMETER_NAMES = frozenset(
        {"state_embedding.weight", "frame_time_proj.weight", "frame_time_proj.bias", "mu_proj.weight", "mu_proj.bias"}
    )

    def __init__(self, cfg, dropout=0.1, *, history_length=20):
        if not isinstance(history_length, int) or isinstance(history_length, bool) or history_length < 1:
            raise ValueError("history_length must be a positive integer.")
        super().__init__(cfg, dropout=dropout)
        if history_length + 1 > self.pos_enc.pe.shape[1]:
            raise ValueError("History plus current frame exceeds positional encoding capacity.")
        self.history_length = history_length
        self.state_embedding = nn.Embedding(2, self.latent_dim)
        self.frame_time_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.mu_proj = nn.Linear(self.input_feats, self.latent_dim)
        self.reset_history_parameters()

    def reset_history_parameters(self):
        for layer in (self.state_embedding, self.frame_time_proj, self.mu_proj):
            for parameter in layer.parameters():
                nn.init.zeros_(parameter)

    def forward(self, x, timesteps, y, cond_scale=None):
        if cond_scale is not None:
            raise ValueError("CFG is not supported by the current-frame history interface.")
        y = prepare_conditioning(y, history_length=self.history_length)
        history = y["history_motion"]
        batch, length, _ = history.shape
        x = floating_tensor(x, (batch, 1, self.input_feats), history, "x")
        x = finite_payload(x, y["valid_frames"], "x")
        if not isinstance(timesteps, torch.Tensor) or timesteps.shape != (batch,):
            raise ValueError("timesteps must be a floating [B] tensor.")
        if not timesteps.is_floating_point() or timesteps.device != x.device:
            raise ValueError("timesteps must be floating point and on the motion device.")
        if not bool(((timesteps > 0) & (timesteps <= 1)).all()):
            raise ValueError("Current Flow times must be finite and in (0,1].")

        motion = self.pos_enc(self.input_process(torch.cat((history, x), dim=1)))
        state_ids = torch.cat((torch.zeros(length, device=x.device), torch.ones(1, device=x.device))).long()
        motion = motion + self.state_embedding(state_ids)[None]
        frame_times = torch.cat((timesteps.new_zeros(batch, length), timesteps[:, None]), dim=1)
        frame_time = self.embed_timestep(frame_times.reshape(-1)).reshape(batch, length + 1, self.latent_dim)
        motion = motion + self.frame_time_proj(frame_time)
        mu_condition = torch.cat((motion.new_zeros(batch, length, self.latent_dim), self.mu_proj(y["prior_mu"])), dim=1)
        motion = motion + mu_condition

        # Only the current position ever receives a raw observation. Use
        # torch.where rather than the legacy multiply-mask to isolate payloads.
        enc_traj = self.embed_traj_cond(y["traj"])
        enc_traj = torch.where(y["traj_mask"][..., None], self.mask_tokens["traj"], enc_traj)
        past_traj = self.mask_tokens["traj"].view(1, 1, -1).expand(batch, length, -1)
        motion = motion + torch.cat((past_traj, enc_traj), dim=1)

        enc_img = self.embed_clip_cond(y["img_embs"])
        enc_img = torch.where(y["img_mask"][..., None], self.mask_tokens["clip"], enc_img)
        past_img = self.mask_tokens["clip"].view(1, 1, -1).expand(batch, length, -1)
        enc_img = self.pos_enc(torch.cat((past_img, enc_img), dim=1))
        enc_time = self.embed_timestep(timesteps)[:, None]
        motion = torch.cat((enc_time, motion), dim=1)
        context = torch.cat((enc_time, enc_img), dim=1)
        valid = torch.cat(
            (torch.ones(batch, 1, device=x.device, dtype=torch.bool), y["history_valid"], y["valid_frames"]), dim=1
        )
        attention_mask = valid[:, None, None, :]
        for decoder in self.tsfm:
            motion = decoder(motion, context, mask=attention_mask, context_mask=attention_mask)
        return self.output_process(motion[:, -1:])
