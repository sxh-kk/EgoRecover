"""Dense E7 attention and continuous time embeddings, without training-framework dependencies."""

import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_x, mask=None):
        # mask B x 1 x 1 x N denotes which elements of kv_x are valid

        x = self.norm(x)
        if mask is not None:
            mask = mask.squeeze([1, 2]).bool().logical_not()
        out = self.attn(x, kv_x, kv_x, key_padding_mask=mask)[0]
        out = self.out_dropout(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x, context, mask=None, context_mask=None):
        x = self.attn1(x, kv_x=x, mask=mask) + x
        x = self.attn2(x, kv_x=context, mask=context_mask) + x
        x = self.ff(x) + x
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
