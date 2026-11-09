"""History-only P with a codec-defined physical continuation baseline."""

import torch
from torch import nn

from .layers import PositionalEncoding


class HistoryPrior(nn.Module):
    def __init__(self, width=256, layers=4, heads=8, dropout=0.1, source_flags=False):
        super().__init__()
        self.input = nn.Linear(243, width)
        self.positions = PositionalEncoding(width, dropout)
        self.source_flags = source_flags
        if source_flags:
            self.source_embedding = nn.Embedding(2, width)
            nn.init.zeros_(self.source_embedding.weight)
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

    def forward(self, history_motion, history_valid, base_mu, *, history_body_valid=None):
        if history_motion.ndim != 3 or history_motion.shape[-1] != 243:
            raise ValueError("P expects [B,L,243] body history.")
        batch, length, _ = history_motion.shape
        if (
            history_valid.shape != (batch, length)
            or history_valid.dtype != torch.bool
            or history_valid.device != history_motion.device
        ):
            raise ValueError("P requires a boolean history mask on the input device.")
        valid = history_valid
        if not bool(valid.any(dim=1).all()) or base_mu.shape != (batch, 1, 243):
            raise ValueError("P requires nonempty histories and a codec-defined [B,1,243] base_mu.")
        if not bool(torch.isfinite(base_mu).all()):
            raise ValueError("base_mu must be finite.")
        # Frozen parameters must still transmit gradients to predicted history:
        # the paired-window objective differentiates through this complete map.
        history = torch.where(valid[..., None], history_motion, 0)
        if not bool(torch.isfinite(history).all()):
            raise ValueError("Nonfinite available P history.")
        embedded = self.input(history)
        if self.source_flags:
            if history_body_valid is None:
                history_body_valid = valid
            if (
                history_body_valid.shape != valid.shape
                or history_body_valid.dtype != torch.bool
                or history_body_valid.device != valid.device
            ):
                raise ValueError("Body-history source flags must be boolean [B,L] on the input device.")
            embedded = embedded + self.source_embedding(history_body_valid.long())
        encoded = self.encoder(self.positions(embedded), src_key_padding_mask=~valid)
        indices = torch.arange(length, device=history.device).expand(batch, -1).masked_fill(~valid, -1).max(1).values
        correction = self.output(encoded[torch.arange(batch, device=history.device), indices])[:, None]
        return base_mu + correction
