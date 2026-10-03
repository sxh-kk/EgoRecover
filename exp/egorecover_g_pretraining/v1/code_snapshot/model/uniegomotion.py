"""Dense E7 motion network; parameter names match upstream E7 checkpoints."""

import math
import numpy as np
import torch
import torch.nn as nn
from model.core import DecoderBlock
from loguru import logger


def mask_it(mask, cond, replace_token):
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).float()
    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


class UniEgoMotion(nn.Module):

    def __init__(self, cfg, dropout=0.1):
        super().__init__()
        self.cfg = cfg
        latent_dim = 768
        ff_size = 768 * 2
        num_layers = 12
        num_heads = 12
        self.img_feat_type = "dinov2"
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.cond_betas = False
        self.encoder_tsfm = None
        self.finetune_type = None
        self.repre_type = "v4_beta"
        self.input_feats = 243
        traj_dim = 18
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.img_feat_dim = 1024
        self.cond_mask_prob = {"traj": 0.5, "clip": 0.1, "subseq_frames": 0.5}
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_traj_cond = nn.Linear(traj_dim, self.latent_dim)
        self.embed_clip_cond = nn.Linear(self.img_feat_dim, self.latent_dim)
        self.embed_text_cond = nn.Linear(768, self.latent_dim)
        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "clip": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "text": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
            }
        )
        self.zero_mask_token = False
        self.tsfm = nn.ModuleList(
            [DecoderBlock(self.latent_dim, self.num_heads, self.dropout, 2) for _ in range(self.num_layers)]
        )
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)
        self.embed_text_cond.requires_grad_(False)
        self.mask_tokens["text"].requires_grad_(False)

    def mask_cond(self, cond, cond_type, cond_mask=None):
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond
        if not self.training:
            return cond
        inp_shape = cond.shape
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob[cond_type]
        cond = mask_it(mask, cond, self.mask_tokens[cond_type])
        assert cond_type in ["traj", "clip"]
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob["subseq_frames"]
        if cond_type == "traj":
            cond = mask_it(mask, cond, self.mask_tokens[cond_type])
        else:
            subsequent_cond = mask_it(mask, cond[:, 1:], self.mask_tokens[cond_type])
            cond = torch.cat((cond[:, :1], subsequent_cond), axis=1)
        assert cond.shape == inp_shape
        return cond

    def pos_enc_and_process_img_feat(self, x, enc_imgs):
        enc_imgs = self.pos_enc(enc_imgs)
        return enc_imgs

    def forward(self, x, timesteps, y, cond_scale=None):
        """
        x_t: B x T x F
        timesteps: B
        y: dict
        """
        if cond_scale is not None:
            x_cond = self.forward(x, timesteps, y)
            x_uncond = self.forward(x, timesteps, {"valid_frames": y["valid_frames"]})
            x_scaled = x_uncond + (x_cond - x_uncond) * cond_scale
            return x_scaled
        B, T, F = x.shape
        x = self.input_process(x)
        x = self.pos_enc(x)
        traj_mask = y["traj_mask"] if "traj_mask" in y else None
        img_mask = y["img_mask"] if "img_mask" in y else None
        enc_time = self.embed_timestep(timesteps)
        if "traj" in y:
            enc_traj = self.embed_traj_cond(y["traj"])
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]
        if "betas" in y:
            raise ValueError("E7 predicts shape and does not accept betas as a condition.")
        all_cond = [enc_time[:, None]]
        all_cond_mask = [torch.ones(B, 1, dtype=torch.long, device=x.device)]
        if "img_embs" in y:
            enc_imgs = self.embed_clip_cond(y["img_embs"])
            enc_imgs = self.mask_cond(enc_imgs, "clip", img_mask)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)
            enc_img_mask = y["valid_img_embs"]
            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)
        else:
            enc_imgs = self.mask_tokens["clip"].view(1, 1, -1).repeat(B, T, 1)
            enc_imgs = self.pos_enc_and_process_img_feat(x, enc_imgs)
            enc_img_mask = y["valid_frames"]
            all_cond.append(enc_imgs)
            all_cond_mask.append(enc_img_mask)
        x = torch.cat((enc_time[:, None], x), axis=1)
        mask = y["valid_frames"]
        mask = torch.cat((torch.ones((B, 1), device=mask.device, dtype=mask.dtype), mask), dim=1)
        mask = mask[:, None, None, :]
        context_mask = torch.cat(all_cond_mask, dim=1)
        context_mask = context_mask[:, None, None, :]
        context = torch.cat(all_cond, dim=1)
        for dec in self.tsfm:
            x = dec(x=x, context=context, mask=mask, context_mask=context_mask)
        x = self.output_process(x[:, 1 : 1 + T])
        x = x.view(B, T, F)
        if "repaint_mask" in y:
            raise ValueError("Pass repaint constraints to FlowMatching.sample_loop, not the network directly.")
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe[None])

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim, pos_enc, min_period=0.004, max_period=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc
        if not 0 < min_period < max_period:
            raise ValueError(f"Expected 0 < min_period < max_period, got {min_period} and {max_period}.")
        half_dim = self.latent_dim // 2
        periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), half_dim))
        angular_frequencies = 2.0 * math.pi / periods
        self.register_buffer("continuous_angular_frequencies", angular_frequencies, persistent=False)
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, timesteps):
        if timesteps.ndim != 1:
            raise ValueError(f"Expected one timestep per batch item, got shape {tuple(timesteps.shape)}.")
        if timesteps.dtype.is_floating_point:
            continuous_time = timesteps.to(dtype=torch.float32)
            frequencies = self.continuous_angular_frequencies.to(dtype=torch.float32)
            angles = continuous_time[:, None] * frequencies[None, :]
            t = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
            if t.shape[-1] < self.latent_dim:
                t = torch.cat((t, torch.zeros_like(t[:, :1])), dim=-1)
            t = t.to(dtype=self.time_embed[0].weight.dtype)
        else:
            t = self.pos_enc.pe[0, timesteps.long()]
        return self.time_embed(t)
