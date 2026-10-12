import math
from contextlib import contextmanager

import torch

from .geometry import fk


@contextmanager
def temporarily_eval(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def masked_mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return torch.where(expanded, value, torch.zeros_like(value)).sum() / expanded.sum().clamp_min(1)


def draw_noisy(target, cfg):
    concentration = target.new_tensor(1.5)
    tau = torch.distributions.Beta(concentration, target.new_tensor(1.0)).sample(target.shape[:-1]).float()
    tau = tau.clamp_min(cfg.tau_min)
    noise = torch.randn_like(target)
    return (1 - tau[..., None]) * target + tau[..., None] * noise, tau


def tae_loss(tae, normalizer, batch, cfg):
    target = normalizer.normalize("pose", batch["pose"])
    recon, mean, logvar = tae(target, batch["beta"], batch["valid"])
    reconstruction = masked_mean((recon.float() - target).square(), batch["valid"])
    kl = masked_mean(0.5 * (mean.float().square() + logvar.float().exp() - 1 - logvar.float()), batch["valid"])
    pose = normalizer.restore("pose", recon)
    joints, _ = fk(pose, batch["root"], batch["offsets"])
    # Fixed-shape reconstruction target: do not force bootstrap beta to match GT beta.
    target_joints, _ = fk(batch["pose"], batch["root"], batch["offsets"])
    geometry = masked_mean((joints - target_joints).square(), batch["valid"])
    total = reconstruction + cfg.kl_weight * kl + cfg.geometry_weight * geometry
    return total, {"reconstruction": reconstruction.detach(), "kl": kl.detach(), "geometry": geometry.detach()}


def replacement_mask(valid, prefix, ratio):
    """Shared time positions across batch; invalid frames and startup remain immutable."""
    t = valid.shape[1]
    candidates = torch.arange(prefix, t, device=valid.device)
    count = min(len(candidates), int(len(candidates) * ratio + 1e-8))
    mask = torch.zeros_like(valid)
    if count:
        chosen = candidates[torch.randperm(len(candidates), device=valid.device)[:count]]
        mask[:, chosen] = True
    return mask & valid


def two_forward_loss(flow, tae, norm, batch, cfg, step, total_steps=None):
    progress = min(max(step / (total_steps or cfg.total_steps), 0), 1)
    ratio = 0.5 * (1 - math.cos(math.pi * progress)) if cfg.two_forward else 0.0
    # Frozen target/startup encoding matches train-fitted latent statistics and online initialization.
    with torch.no_grad(), torch.autocast(batch["pose"].device.type, enabled=False):
        mean, _ = tae.encode(norm.normalize("pose", batch["pose"]), batch["beta"], batch["valid"])
        target = torch.cat((norm.normalize("latent", mean), norm.normalize("root", batch["root"])), -1)
        startup, _ = tae.encode(norm.normalize("pose", batch["startup_pose"]), batch["beta"])
        history = target.clone()
        history[:, : cfg.bootstrap] = torch.cat(
            (norm.normalize("latent", startup), norm.normalize("root", batch["startup_root"])), -1
        )
    head = norm.normalize("head", batch["head"])
    actions = torch.rand(len(target), 1, 2, device=target.device) >= cfg.modality_drop
    actions = actions.expand(-1, target.shape[1], -1)
    args = (batch["images"], head, batch["image_valid"], batch["head_valid"], batch["valid"], batch["beta"], actions)
    selected = replacement_mask(batch["valid"], cfg.bootstrap, ratio)
    if selected.any():
        # The first pass has noisy *current GT* as its flow input. Its causal history has no current GT.
        with torch.no_grad(), temporarily_eval(flow):
            noisy, tau = draw_noisy(target, cfg)
            first = flow(noisy, tau, history, *args).float()
            first = norm.project_state(first, cfg.latent_dim)
            history = torch.where(selected[..., None], first, history).detach()
    noisy, tau = draw_noisy(target, cfg)  # Independent time and noise in the second pass.
    prediction = flow(noisy, tau, history, *args).float()
    supervise = batch["valid"].clone()
    supervise[:, : cfg.bootstrap] = False
    pose_loss = masked_mean((prediction[..., : cfg.latent_dim] - target[..., : cfg.latent_dim]).square(), supervise)
    root_loss = masked_mean((prediction[..., cfg.latent_dim :] - target[..., cfg.latent_dim :]).square(), supervise)
    # Crucial: decoder receives the same mixed past that conditioned the second pass.
    physical_past = history.clone()
    physical_past[..., : cfg.latent_dim] = norm.restore("latent", history[..., : cfg.latent_dim])
    latent = norm.restore("latent", prediction[..., : cfg.latent_dim])
    pose = norm.restore("pose", tae.decode_queries(physical_past, latent, batch["beta"], batch["valid"]))
    root = norm.restore("root", prediction[..., cfg.latent_dim :])
    joints, _ = fk(pose, root, batch["offsets"])
    target_joints, _ = fk(batch["pose"], batch["root"], batch["offsets"])
    geometry = masked_mean((joints - target_joints).square(), supervise)
    total = pose_loss + cfg.root_weight * root_loss + cfg.geometry_weight * geometry
    return total, {
        "latent": pose_loss.detach(),
        "root": root_loss.detach(),
        "geometry": geometry.detach(),
        "gamma": ratio,
        "replaced_fraction": float(selected.sum() / supervise.sum().clamp_min(1)),
    }
