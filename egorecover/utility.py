"""Pre-sampling utility Q and offline-only shared-noise label generation."""

import torch
from torch import nn

from egorecover.actions import action_equivalence, action_masks, binary_mask
from egorecover.conditioning import finite_payload, prepare_conditioning
from model.uniegomotion import PositionalEncoding


class UtilityPredictor(nn.Module):
    def __init__(self, width=256, layers=2, heads=8, dropout=0.1, compatibility_dim=0):
        super().__init__()
        self.compatibility_dim = compatibility_dim
        self.body_input = nn.Linear(243, width)
        self.observation_input = nn.Linear(1024 + 18 + 2, width)
        self.mu_input = nn.Linear(243, width)
        self.positions = PositionalEncoding(width, dropout)
        body_layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, norm_first=True)
        observation_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, batch_first=True, norm_first=True
        )
        self.body_encoder = nn.TransformerEncoder(body_layer, layers, enable_nested_tensor=False)
        self.observation_encoder = nn.TransformerEncoder(observation_layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(width * 3 + compatibility_dim, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _last(encoded, valid):
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Q requires nonempty body and observation histories.")
        indices = (
            torch.arange(valid.shape[1], device=valid.device).expand_as(valid).masked_fill(~valid, -1).max(1).values
        )
        return encoded[torch.arange(len(valid), device=valid.device), indices]

    def forward(
        self,
        *,
        history_motion,
        history_valid,
        prior_mu,
        img_embs,
        traj,
        img_available,
        traj_available,
        observation_valid,
        compatibility=None
    ):
        """No G candidate/GT/fault arguments; observations end at current time."""
        batch, length, width = history_motion.shape
        if width != 243 or img_embs.ndim != 3 or img_embs.shape[0] != batch or img_embs.shape[-1] != 1024:
            raise ValueError("Invalid body/image shapes for Q.")
        count, device = img_embs.shape[1], history_motion.device
        if traj.shape != (batch, count, 18) or prior_mu.shape != (batch, 1, 243):
            raise ValueError("Q expects 18D observed trajectory and one 243D mu.")
        hv = binary_mask(history_valid, (batch, length), device, "history_valid")
        ov = binary_mask(observation_valid, (batch, count), device, "observation_valid")
        iv = binary_mask(img_available, (batch, count), device, "img_available") & ov
        tv = binary_mask(traj_available, (batch, count), device, "traj_available") & ov
        if not bool(hv.any(1).all()) or not bool(ov.any(1).all()):
            raise ValueError("Q requires nonempty histories.")
        body = self.positions(self.body_input(finite_payload(history_motion, hv, "history_motion")))
        observations = torch.cat(
            (
                finite_payload(img_embs, iv, "img_embs"),
                finite_payload(traj, tv, "traj"),
                iv[..., None].to(traj.dtype),
                tv[..., None].to(traj.dtype),
            ),
            dim=-1,
        )
        observations = self.positions(self.observation_input(observations))
        body = self._last(self.body_encoder(body, src_key_padding_mask=~hv), hv)
        observations = self._last(self.observation_encoder(observations, src_key_padding_mask=~ov), ov)
        features = [body, observations, self.mu_input(prior_mu[:, 0])]
        if self.compatibility_dim:
            if compatibility is None or compatibility.shape != (batch, self.compatibility_dim):
                raise ValueError("Expected the configured legal compatibility features.")
            features.append(compatibility)
        elif compatibility is not None:
            raise ValueError("This Q was configured without compatibility features.")
        joined = torch.cat(features, dim=-1)
        if not bool(torch.isfinite(joined).all()):
            raise ValueError("Nonfinite utility features.")
        return self.output(joined)


def choose_action(gains, *, img_available, traj_available, min_gain=0.0):
    if gains.ndim != 2 or gains.shape[1] != 3 or not bool(torch.isfinite(gains).all()) or min_gain < 0:
        raise ValueError("Expected finite [B,3] gains and nonnegative min_gain.")
    canonical = action_equivalence(img_available=img_available, traj_available=traj_available)
    values = torch.cat((torch.zeros_like(gains[:, :1]), gains), dim=1)
    values = values.gather(1, canonical)
    values = torch.where(values > min_gain, values, torch.zeros_like(values))
    # argmax picks the first representative (a11 on ties), including total dropout.
    return canonical.gather(1, values.argmax(1, keepdim=True))[:, 0]


@torch.no_grad()
def generate_utility_labels(model, flow, y, epsilon, error_fn, *, checkpoint_id):
    """OFFLINE: error_fn(candidate) -> [B] using fixed external supervision.

    y must describe a11 with ONLY actual-unavailability masks; do not pass
    an already action-masked payload. Candidates share H/mu/epsilon and cannot
    access/commit a HistoryBuffer.
    Caller supplies an explicit metric and stores that metric's identity with
    the returned metadata; production labels require world SMPL22 FK error.
    """
    if not checkpoint_id:
        raise ValueError("Utility labels require a G checkpoint identity.")
    y = prepare_conditioning(y)
    if not bool(y["valid_frames"].all()):
        raise ValueError("Utility labels require valid current physical frames.")
    img_available, traj_available = ~y["img_mask"], ~y["traj_mask"]
    errors = []
    for action in range(4):
        candidate_y = dict(y)
        candidate_y["img_mask"], candidate_y["traj_mask"] = action_masks(
            action, img_available=img_available, traj_available=traj_available
        )
        prediction = flow.sample(model, candidate_y, epsilon=epsilon)
        error = error_fn(prediction)
        if error.shape != (len(epsilon),) or not bool(torch.isfinite(error).all()):
            raise ValueError("error_fn must return one finite scalar per batch item.")
        errors.append(error)
    errors = torch.stack(errors, dim=1)
    canonical = action_equivalence(img_available=img_available, traj_available=traj_available)
    errors = errors.gather(1, canonical)
    return {
        "errors": errors,
        "gains": errors[:, :1] - errors[:, 1:],
        "canonical_actions": canonical,
        "identity": {
            "checkpoint_id": checkpoint_id,
            "source_mode": flow.source_mode,
            "sigma": flow.sigma,
            "num_steps": flow.flow.num_steps,
        },
    }
