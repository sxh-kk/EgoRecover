"""History-only velocity features and a matched residual-prior input branch."""
import math
import torch
from torch import nn

from egorecover.codec import MotionCodec, transform_from_9d
from egorecover.prior import HistoryPrior
from utils.rotation_conversions import matrix_to_axis_angle

# SMPL22 parent order; no supervision or SMPL assets are needed at inference.
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
VELOCITY_SCHEMA = 'root_linear3_root_spatial_angular3_local_angular63_valid1_dt0.1_v1'


@torch.no_grad()
def history_velocities(codec, history_motion, history_valid, dt=0.1):
    """70 features/frame from past states, expressed in a common window frame.

    root translation: m/s divided by 1 m/s; angular velocities: rad/s divided
    by pi rad/s. Local angular velocities use previous local joint axes.
    The first frame has zero features and a false difference-valid bit.
    Current GT, sensors and shape labels are never read here.
    """
    if history_motion.ndim != 3 or history_motion.shape[-1] != 243 or dt <= 0:
        raise ValueError('Expected [B,L,243] history and positive dt.')
    if history_valid.shape != history_motion.shape[:2] or not bool(history_valid.bool().all()):
        raise ValueError('Velocity v1 requires a complete, valid history window.')
    if not bool(torch.isfinite(history_motion).all()):
        raise ValueError('Nonfinite history.')
    with torch.autocast(device_type=history_motion.device.type, enabled=False):
        raw = codec.denormalize(history_motion.float())
        local = transform_from_9d(raw[..., :198].unflatten(-1, (22, 9)))
        delta = codec.canonical_reference(transform_from_9d(raw[..., 198:207]))
        # encode_history stores the first delta relative to the outer window
        # frame, so cumulative transforms recover all states in that same frame.
        reference = torch.eye(4, device=raw.device, dtype=raw.dtype).expand(len(raw), 4, 4)
        references = []
        for t in range(raw.shape[1]):
            reference = reference @ delta[:, t]
            references.append(reference)
        references = torch.stack(references, 1)
        body = references[:, :, None] @ local
        rotations = body[..., :3, :3]
        positions = body[:, :, 0, :3, 3]
        root_linear = (positions[:, 1:] - positions[:, :-1]) / dt
        root_delta = rotations[:, 1:, 0] @ rotations[:, :-1, 0].transpose(-1, -2)
        root_angular = matrix_to_axis_angle(root_delta) / (dt * math.pi)
        joint_local = rotations[:, :, PARENTS[1:]].transpose(-1, -2) @ rotations[:, :, 1:]
        local_delta = joint_local[:, :-1].transpose(-1, -2) @ joint_local[:, 1:]
        local_angular = matrix_to_axis_angle(local_delta).flatten(-2) / (dt * math.pi)
        features = raw.new_zeros((*raw.shape[:2], 70))
        features[:, 1:] = torch.cat((root_linear, root_angular, local_angular,
                                    raw.new_ones((len(raw), raw.shape[1] - 1, 1))), -1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite physical velocity features.')
        return features


class VelocityHistoryPrior(HistoryPrior):
    """Original state tokens plus a zero-initialized linear velocity embedding."""
    def __init__(self, stats, *, reference_mode='planar', width=256, layers=4, heads=8, dropout=.1):
        super().__init__(width, layers, heads, dropout)
        self.codec = MotionCodec(stats, reference_mode=reference_mode)
        # Preserve shared initialization and subsequent dropout RNG vs control.
        with torch.random.fork_rng(devices=[]):
            self.velocity_input = nn.Linear(70, width, bias=False)
            nn.init.zeros_(self.velocity_input.weight)
        self.base_mode = 'constant_velocity'

    def forward(self, history_motion, history_valid, base_mu):
        if history_motion.ndim != 3 or history_motion.shape[-1] != 243:
            raise ValueError('Expected [B,L,243] body history.')
        if base_mu.shape != (len(history_motion), 1, 243) or not bool(torch.isfinite(base_mu).all()):
            raise ValueError('Invalid constant-velocity baseline.')
        velocity = history_velocities(self.codec, history_motion, history_valid)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._frozen):
            tokens = self.input(history_motion) + self.velocity_input(velocity)
            encoded = self.encoder(self.positions(tokens))
            return base_mu + self.output(encoded[:, -1])[:, None]


def create_prior(stats, velocity_input=False, reference_mode='planar', **network):
    model = (VelocityHistoryPrior(stats, reference_mode=reference_mode, **network) if velocity_input
             else HistoryPrior(**network))
    model.base_mode = 'constant_velocity'
    return model


def load_prior(path, stats, device='cpu', reference_mode='planar'):
    """Explicit loader: preserve the checkpoint's feature schema and CV baseline."""
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if saved.get('base_mode') != 'constant_velocity' or saved.get('reference_mode') != reference_mode:
        raise ValueError('Checkpoint baseline/reference mode differs.')
    enabled = saved.get('velocity_input', False)
    if enabled and saved.get('velocity_schema') != VELOCITY_SCHEMA:
        raise ValueError('Checkpoint velocity schema differs.')
    model = create_prior(stats, enabled, reference_mode)
    # Avoid silently replacing caller statistics with differently normalized history.
    if enabled:
        for key, value in model.codec.state_dict().items():
            if not torch.equal(value.cpu(), saved['state_dict']['codec.' + key].cpu()):
                raise ValueError('Checkpoint normalization differs from input codec.')
    model.load_state_dict(saved['state_dict'])
    return model.to(device).freeze()
