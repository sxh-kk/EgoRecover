"""History-only P with a codec-defined physical continuation baseline."""

import torch
from torch import nn

from egorecover.actions import binary_mask
from egorecover.conditioning import finite_payload
from model.uniegomotion import PositionalEncoding


class HistoryPrior(nn.Module):
    def __init__(self, width=256, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(243, width)
        self.positions = PositionalEncoding(width, dropout)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 243)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self._frozen = False

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode=True):
        return super().train(False if self._frozen else mode)

    def forward(self, history_motion, history_valid, base_mu):
        if history_motion.ndim != 3 or history_motion.shape[-1] != 243:
            raise ValueError("P expects [B,L,243] body history.")
        batch, length, _ = history_motion.shape
        valid = binary_mask(history_valid, (batch, length), history_motion.device, "history_valid")
        if not bool(valid.any(dim=1).all()) or base_mu.shape != (batch, 1, 243):
            raise ValueError("P requires nonempty histories and a codec-defined [B,1,243] base_mu.")
        if not bool(torch.isfinite(base_mu).all()):
            raise ValueError("base_mu must be finite.")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._frozen):
            history = finite_payload(history_motion, valid, "history_motion")
            encoded = self.encoder(self.positions(self.input(history)), src_key_padding_mask=~valid)
            indices = (
                torch.arange(length, device=history.device).expand(batch, -1).masked_fill(~valid, -1).max(1).values
            )
            correction = self.output(encoded[torch.arange(batch, device=history.device), indices])[:, None]
            return base_mu + correction
