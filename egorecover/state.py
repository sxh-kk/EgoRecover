"""Immutable world-space cache and an explicitly bounded online sensor packet."""

from dataclasses import dataclass

import torch

from egorecover.codec import BodyState

FIELDS = ("joints", "reference", "auxiliary")


def state_map(state, function):
    return BodyState(*(function(getattr(state, key)) for key in FIELDS))


def state_cat(states, dim=1):
    return BodyState(*(torch.cat([getattr(state, key) for state in states], dim) for key in FIELDS))


@dataclass(frozen=True)
class ObservationWindow:
    head: torch.Tensor  # [B,21,9], absolute, floor-calibrated world pose
    images: torch.Tensor  # [B,21,1024]
    head_valid: torch.Tensor  # [B,21]
    image_valid: torch.Tensor  # [B,21]
    frame: int  # terminal physical frame; no future payload is accepted

    def validate(self, batch, frame, history=20):
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame != frame:
            raise ValueError("Observation window must end at the requested physical frame.")
        for name, width, valid_name in (("head", 9, "head_valid"), ("images", 1024, "image_valid")):
            value, valid = getattr(self, name), getattr(self, valid_name)
            if value.shape != (batch, history + 1, width) or not value.is_floating_point():
                raise ValueError(f"Invalid observation shape/dtype: {name}.")
            if valid.shape != (batch, history + 1) or valid.dtype != torch.bool or valid.device != value.device:
                raise ValueError(f"Invalid availability: {valid_name}.")
            if not bool(torch.isfinite(value[valid]).all()):
                raise ValueError(f"Nonfinite available observation: {name}.")
        if self.head.device != self.images.device or self.head.dtype != self.images.dtype:
            raise ValueError("Observation payloads must share device and dtype.")


@dataclass(frozen=True)
class WindowCache:
    body: BodyState  # [B,20,...], world coordinates
    beta: torch.Tensor  # [B,10], fixed model-generated startup shape
    anchor: torch.Tensor  # [B,4,4], fixed startup reference for the no-feedback control
    frame: int  # last committed physical frame

    def validate(self, history=20):
        b = len(self.beta)
        for name, shape in (
            ("joints", (b, history, 22, 4, 4)),
            ("reference", (b, history, 4, 4)),
            ("auxiliary", (b, history, 36)),
        ):
            value = getattr(self.body, name)
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid world cache: {name}.")
            if value.device != self.beta.device or value.dtype != self.beta.dtype:
                raise ValueError("Cache fields must share device and dtype.")
        if self.beta.shape != (b, 10) or b < 1 or not bool(torch.isfinite(self.beta).all()):
            raise ValueError("Expected fixed finite startup beta [B,10].")
        if self.anchor.shape != (b, 4, 4) or not bool(torch.isfinite(self.anchor).all()):
            raise ValueError("Expected a fixed finite startup anchor.")
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < history - 1:
            raise ValueError("Invalid cache frame index.")

    def detach(self):
        return WindowCache(
            state_map(self.body, lambda x: x.detach()), self.beta.detach(), self.anchor.detach(), self.frame
        )

    def advance(self, window, *, revise_recent=True):
        """Rewrite W-1 internal states, append current, then evict the oldest.

        Public outputs already emitted at previous times are never changed.
        No in-place write is used, so feedback can retain just the current graph.
        """
        width = window.reference.shape[1]
        if not 1 <= width <= self.body.reference.shape[1] + 1:
            raise ValueError("Invalid active window length.")
        if revise_recent:
            fixed = state_map(self.body, lambda x: x[:, : self.body.reference.shape[1] - width + 1])
            combined = state_cat((fixed, window))
        else:
            combined = state_cat((self.body, state_map(window, lambda x: x[:, -1:])))
        return WindowCache(state_map(combined, lambda x: x[:, 1:]), self.beta, self.anchor, self.frame + 1)

    def replace_current(self, current):
        body = state_cat((state_map(self.body, lambda x: x[:, :-1].detach()), state_map(current, lambda x: x[:, None])))
        return WindowCache(body, self.beta.detach(), self.anchor.detach(), self.frame)
