"""Complete differentiable history -> P -> source -> G -> world-state map."""

from dataclasses import dataclass

import torch

from egorecover.codec import BodyState, planar_reference
from .flow import build_source, solve
from .state import WindowCache, state_cat, state_map


@dataclass(frozen=True)
class WindowResult:
    codes: torch.Tensor
    body: BodyState
    cache: WindowCache
    reference: torch.Tensor
    center: torch.Tensor

    @property
    def current(self):
        return state_map(self.body, lambda x: x[:, -1])


class WindowReconstructor:
    def __init__(self, codec, predictor, config):
        self.codec, self.predictor, self.config = codec, predictor, config
        # requires_grad_(False) freezes parameters without disabling input grads.
        if predictor is not None:
            predictor.requires_grad_(False).eval()

    def prepare(self, cache, observations):
        cfg = self.config
        cache.validate(cfg.history)
        observations.validate(len(cache.beta), cache.frame + 1, cfg.history)
        if observations.head.device != cache.beta.device or observations.head.dtype != cache.beta.dtype:
            raise ValueError("Cache and observations must share device and dtype.")
        history = cache.body
        last = state_map(history, lambda x: x[:, -1])
        reference = self.codec.canonical_reference(last.reference if cfg.body_feedback else cache.anchor)
        if cfg.body_feedback:
            before = state_map(history, lambda x: x[:, -2])
            mode = getattr(self.predictor, "base_mode", "constant_velocity")
            if mode not in ("hold", "constant_velocity"):
                raise ValueError("Unknown predictor baseline.")
            base = (
                self.codec.continuation_prior(last)
                if mode == "hold"
                else self.codec.constant_velocity_prior(before, last)
            )[:, None]
            if self.predictor is not None:
                self.predictor.eval()
                prior_history = self.codec.encode_history(history, planar_reference(history.reference[:, 0]))
                valid = torch.ones(prior_history.shape[:2], dtype=torch.bool, device=base.device)
                base = self.predictor(prior_history, valid, base)
            predicted = self.codec.decode_current(base[:, 0], last.reference)
            recent = state_map(history, lambda x: x[:, cfg.fixed :])
            center_body = state_cat((recent, state_map(predicted, lambda x: x[:, None])))
            # Every code has its reference delta relative to ONE shared frame.
            center = self.encode(center_body, reference)
            fixed = self.encode(state_map(history, lambda x: x[:, : cfg.fixed]), reference)
        else:
            center = cache.beta.new_zeros(len(cache.beta), cfg.window, 243)
            fixed = cache.beta.new_zeros(len(cache.beta), cfg.fixed, 243)
        fixed_valid = torch.full(fixed.shape[:2], cfg.body_feedback, dtype=torch.bool, device=fixed.device)
        traj = self.codec.encode_observation(observations.head, reference[:, None], observations.head_valid)
        images = torch.where(observations.image_valid[..., None], observations.images, 0)
        y = {
            "fixed_motion": fixed,
            "fixed_valid": fixed_valid,
            "center": center,
            "traj": traj,
            "images": images,
            "head_valid": observations.head_valid,
            "image_valid": observations.image_valid,
        }
        return y, reference

    def encode(self, body, reference):
        return self.codec.encode_current(body, reference[:, None])

    def decode(self, codes, reference):
        return self.codec.decode_current(codes, reference[:, None])

    def reconstruct(self, model, cache, observations, epsilon, *, steps=None):
        y, reference = self.prepare(cache, observations)
        source = build_source(y["center"], epsilon, mode=self.config.source, sigma=self.config.sigma)
        codes = solve(model, source, y, steps=self.config.steps if steps is None else steps)
        body = self.decode(codes, reference)
        updated = cache.advance(body, revise_recent=self.config.revise_recent)
        return WindowResult(codes, body, updated, reference, y["center"])

    @torch.no_grad()
    def online(self, model, cache, observations, epsilon, *, steps=None):
        if model.training:
            raise ValueError("Online reconstruction requires model.eval().")
        return self.reconstruct(model, cache, observations, epsilon, steps=steps)
