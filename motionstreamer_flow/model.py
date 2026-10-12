import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def windows(values, length, include_current=False):
    """[B,T,...] -> [B,T,length,...], with zero left padding."""
    if not include_current:
        values = torch.cat((torch.zeros_like(values[:, :1]), values[:, :-1]), 1)
    pad = values.new_zeros(values.shape[0], length - 1, *values.shape[2:])
    padded = torch.cat((pad, values), 1)
    return padded.unfold(1, length, 1).movedim(-1, 2)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normed.to(x.dtype) * self.weight


class CausalConv(nn.Module):
    def __init__(self, input_width, output_width, dilation=1, kernel=3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(input_width, output_width, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2)


class CausalStack(nn.Module):
    def __init__(self, input_width, output_width, width, dilations):
        super().__init__()
        self.input = CausalConv(input_width, width)
        self.shape = nn.Linear(10, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in dilations])
        self.convs = nn.ModuleList([CausalConv(width, width, d) for d in dilations])
        self.output = CausalConv(width, output_width)

    def forward(self, x, beta, valid=None):
        gate = torch.ones_like(x[..., :1]) if valid is None else valid[..., None].to(x.dtype)
        x = torch.where(gate.bool(), x, torch.zeros_like(x))
        x = (self.input(x) + self.shape(beta)[:, None]) * gate
        for norm, conv in zip(self.norms, self.convs):
            x = (x + conv(F.silu(norm(x)) * gate)) * gate
        return self.output(F.silu(x)) * gate


class CausalTAE(nn.Module):
    """Stride-one variational temporal autoencoder, with framewise normalization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = CausalStack(cfg.pose_dim, 2 * cfg.latent_dim, cfg.tae_width, cfg.tae_dilations)
        self.decoder = CausalStack(cfg.latent_dim, cfg.pose_dim, cfg.tae_width, cfg.tae_dilations)

    def encode(self, pose, beta, valid=None):
        mean, logvar = self.encoder(pose, beta, valid).chunk(2, -1)
        return mean, logvar.clamp(-12, 8)

    def decode(self, latent, beta, valid=None):
        return self.decoder(latent, beta, valid)

    def forward(self, pose, beta, valid):
        mean, logvar = self.encode(pose, beta, valid)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return self.decode(latent, beta, valid), mean, logvar

    def decode_queries(self, past_states, current_latent, beta, valid):
        context = windows(past_states[..., : self.cfg.latent_dim], self.cfg.receptive_field - 1)
        mask = windows(valid, self.cfg.receptive_field - 1)
        context = torch.cat((context, current_latent.unsqueeze(2)), 2)
        mask = torch.cat((mask, valid.unsqueeze(2)), 2)
        b, t, r, d = context.shape
        decoded = self.decode(
            context.reshape(b * t, r, d), beta[:, None].expand(-1, t, -1).reshape(b * t, 10), mask.reshape(b * t, r)
        )
        return decoded[:, -1].reshape(b, t, self.cfg.pose_dim)


def rope(x):
    # x: [batch,heads,time,head_width], positions are relative to the fixed history window.
    half = x.shape[-1] // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(half, device=x.device).float() / half)
    angles = torch.arange(x.shape[-2], device=x.device).float()[:, None] * frequency
    a, b = x.chunk(2, -1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)


class Attention(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        self.heads = cfg.heads
        self.cross = cross
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.k = nn.Linear(cfg.width, cfg.width, bias=False)
        self.v = nn.Linear(cfg.width, cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.dropout = cfg.dropout

    def forward(self, x, memory, available):
        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, -1).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(memory)), split(self.v(memory))
        if not self.cross:
            q, k = rope(q), rope(k)
        q = F.normalize(q.float(), dim=-1).to(q.dtype) * self.log_scale.clamp(max=math.log(100)).exp().to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
        allowed = available[:, None, None, :].expand(-1, 1, x.shape[1], -1)
        if not self.cross:
            causal = torch.ones(x.shape[1], memory.shape[1], dtype=torch.bool, device=x.device).tril()
            allowed = allowed & causal
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, scale=1.0, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(result.transpose(1, 2).reshape_as(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, cross=False):
        super().__init__()
        self.norm = RMSNorm(cfg.width)
        self.memory_norm = RMSNorm(cfg.width) if cross else None
        self.attention = Attention(cfg, cross)
        self.ff_norm = RMSNorm(cfg.width)
        self.ff = nn.Linear(cfg.width, cfg.width * 8 // 3 * 2, bias=False)
        self.ff_out = nn.Linear(cfg.width * 8 // 3, cfg.width, bias=False)

    def forward(self, x, memory, valid):
        normed = self.norm(x)
        x = x + self.attention(normed, self.memory_norm(memory) if self.memory_norm else normed, valid)
        gate, value = self.ff(self.ff_norm(x)).chunk(2, -1)
        return x + self.ff_out(F.silu(gate) * value)


class Conditioner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.state = nn.Linear(cfg.state_dim, cfg.width)
        self.shape = nn.Linear(10, cfg.width)
        self.bos = nn.Parameter(torch.randn(1, 1, cfg.width) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.width) * 0.02)
        self.history_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
        self.image = nn.Sequential(nn.Linear(cfg.image_dim, cfg.width), RMSNorm(cfg.width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(18, cfg.width), RMSNorm(cfg.width), nn.SiLU())
        self.position = nn.Parameter(torch.randn(1, cfg.history + 1, cfg.width) * 0.02)
        self.types = nn.Parameter(torch.randn(2, 1, cfg.width) * 0.02)
        self.null = nn.Parameter(torch.randn(1, 1, cfg.width) * 0.02)
        self.fusion = nn.ModuleList([TransformerBlock(cfg, cross=True) for _ in range(cfg.fusion_layers)])
        self.output_norm = RMSNorm(cfg.width)

    def forward(self, states, images, head, image_valid, head_valid, body_valid, beta, actions=None, last_only=False):
        b, original_t, _ = states.shape
        t = 1 if last_only else original_t

        def select(value):
            return value[:, -1:] if last_only else value

        # Mask raw payload before projection: invalid NaNs must never reach linear layers.
        images = torch.where(image_valid[..., None], images, torch.zeros_like(images))
        head = torch.where(head_valid[..., None], head, torch.zeros_like(head))
        past = select(windows(states, self.cfg.history)).reshape(b * t, self.cfg.history, -1)
        mask = select(windows(body_valid, self.cfg.history)).reshape(b * t, self.cfg.history)
        past = torch.where(mask[..., None], past, torch.zeros_like(past))
        shape = self.shape(beta)[:, None].expand(-1, t, -1).reshape(b * t, 1, -1)
        tokens = (
            torch.cat((self.bos.expand(b * t, -1, -1), self.state(past), self.query.expand(b * t, -1, -1)), 1) + shape
        )
        allowed = torch.cat(
            (
                torch.ones(b * t, 1, dtype=torch.bool, device=states.device),
                mask,
                torch.ones(b * t, 1, dtype=torch.bool, device=states.device),
            ),
            1,
        )
        for block in self.history_blocks:
            if self.cfg.checkpoint_layers and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, tokens, allowed, use_reentrant=False)
            else:
                tokens = block(tokens, tokens, allowed)
        query = tokens[:, -1:]
        observations, availability = [], []
        for i, (encoder, payload, valid) in enumerate(
            ((self.image, images, image_valid), (self.head, head, head_valid))
        ):
            memory = select(windows(payload, self.cfg.history + 1, True)).reshape(b * t, self.cfg.history + 1, -1)
            available = select(windows(valid, self.cfg.history + 1, True)).reshape(b * t, self.cfg.history + 1)
            if actions is not None:
                # A query's action gates its ENTIRE observation window, not the source frame's action.
                available = available & select(actions)[..., i].reshape(b * t, 1).bool()
            memory = torch.where(available[..., None], memory, torch.zeros_like(memory))
            observations.append(encoder(memory) + self.position + self.types[i])
            availability.append(available)
        memory = torch.cat((*observations, self.null.expand(b * t, -1, -1)), 1)
        available = torch.cat((*availability, torch.ones(b * t, 1, dtype=torch.bool, device=states.device)), 1)
        for block in self.fusion:
            query = block(query, memory, available)
        return self.output_norm(query[:, 0]).reshape(b, t, -1)


class FlowBlock(nn.Module):
    def __init__(self, width, condition_width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(condition_width, width * 3)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, c):
        shift, scale, gate = self.modulation(F.silu(c)).chunk(3, -1)
        return x + gate * self.ff(self.norm(x) * (1 + scale) + shift)


class MotionFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conditioner = Conditioner(cfg)
        self.time = nn.Sequential(nn.Linear(64, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.input = nn.Linear(cfg.state_dim, cfg.flow_width)
        self.blocks = nn.ModuleList([FlowBlock(cfg.flow_width, cfg.width) for _ in range(cfg.flow_layers)])
        self.output = nn.Sequential(nn.LayerNorm(cfg.flow_width), nn.Linear(cfg.flow_width, cfg.state_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def endpoint(self, noisy, tau, condition):
        frequency = torch.exp(-math.log(10000) * torch.arange(32, device=noisy.device).float() / 32)
        angle = tau.float()[..., None] * 1000 * frequency
        c = condition + self.time(torch.cat((angle.sin(), angle.cos()), -1).to(condition.dtype))
        x = self.input(noisy)
        for block in self.blocks:
            x = block(x, c)
        return self.output(x)

    def forward(self, noisy, tau, states, images, head, image_valid, head_valid, body_valid, beta, actions=None):
        return self.endpoint(
            noisy, tau, self.conditioner(states, images, head, image_valid, head_valid, body_valid, beta, actions)
        )

    @torch.no_grad()
    def sample(self, condition, noise, steps=None):
        return euler_sample(self.endpoint, condition, noise, steps or self.cfg.flow_steps)


def euler_sample(endpoint, condition, noise, steps):
    if steps < 1:
        raise ValueError("Euler steps must be positive.")
    x = noise.float().clone()
    for i in range(steps):
        tau = 1 - i / steps
        time = x.new_full(x.shape[:-1], tau)
        x0 = endpoint(x, time, condition).float()
        x = x + (-1 / steps) * (x - x0) / tau
    return x
