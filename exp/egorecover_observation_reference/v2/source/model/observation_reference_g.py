"""E7-compatible parameters, independent causal observation-memory length."""
import torch

from model.history_uniegomotion import HistoryUniEgoMotion
from egorecover.observation_reference import prepare
from egorecover.conditioning import floating_tensor


class ObservationReferenceG(HistoryUniEgoMotion):
    def __init__(self, cfg, *, observation_length=21, anchored=False, dropout=.1):
        super().__init__(cfg, dropout=dropout, past_observations=False)
        self.observation_length = observation_length
        self.anchored = anchored
        # Preserve E7's head-to-body branch. Extra head memory starts silent.
        self.memory_traj_adapter = torch.nn.Linear(18, self.latent_dim)
        torch.nn.init.zeros_(self.memory_traj_adapter.weight)
        torch.nn.init.zeros_(self.memory_traj_adapter.bias)

    def initialize_e7(self, checkpoint):
        from egorecover.checkpoint import load_e7_weights
        with torch.random.fork_rng(devices=[]):
            base = HistoryUniEgoMotion(self.cfg, dropout=self.dropout)
        report = load_e7_weights(base, checkpoint, weight_source='ema')
        incompatible = self.load_state_dict(base.state_dict(), strict=False)
        assert set(incompatible.missing_keys) == {'memory_traj_adapter.weight', 'memory_traj_adapter.bias'}
        assert not incompatible.unexpected_keys
        torch.nn.init.zeros_(self.memory_traj_adapter.weight)
        torch.nn.init.zeros_(self.memory_traj_adapter.bias)
        return report

    def forward(self, x, timesteps, y, cond_scale=None):
        if cond_scale is not None:
            raise ValueError('CFG not supported.')
        y = prepare(y); history = y['history_motion']; b, h, _ = history.shape
        m = y['memory_img'].shape[1]
        if m != self.observation_length:
            raise ValueError('Observation length differs from model configuration.')
        x = floating_tensor(x, (b, 1, 243), history, 'x')
        if timesteps.shape != (b,) or not bool(((timesteps > 0) & (timesteps <= 1)).all()):
            raise ValueError('Invalid Flow times.')
        motion = self.pos_enc(self.input_process(torch.cat((history, x), 1)))
        ids = torch.cat((torch.zeros(h, device=x.device), torch.ones(1, device=x.device))).long()
        motion = motion + self.state_embedding(ids)[None]
        ft = torch.cat((timesteps.new_zeros(b, h), timesteps[:, None]), 1)
        motion = motion + self.frame_time_proj(self.embed_timestep(ft.reshape(-1)).reshape(b, h+1, -1))
        motion = motion + torch.cat((motion.new_zeros(b, h, self.latent_dim), self.mu_proj(y['prior_mu'])), 1)
        motion = motion + self.embed_traj_cond(y['memory_traj'][:, -h-1:])
        memory = self.embed_clip_cond(y['memory_img']) + self.memory_traj_adapter(y['memory_traj'])
        # Final 21 positions remain 0..20, matching E7 and previous G. Extend
        # the same sinusoid to negative positions for additional older tokens.
        positions = torch.arange(h+1-m, h+1, device=x.device)
        pe = self.pos_enc.pe[:, positions.abs()].clone().to(memory)
        pe[:, positions < 0, 0::2] *= -1
        memory = self.pos_enc.dropout(memory + pe)
        et = self.embed_timestep(timesteps)[:, None]
        motion = torch.cat((et, motion), 1); context = torch.cat((et, memory), 1)
        body_valid = torch.cat((torch.ones(b, 1, device=x.device, dtype=torch.bool),
                                y['history_valid'], y['valid_frames']), 1)
        memory_valid = torch.cat((torch.ones(b, 1, device=x.device, dtype=torch.bool), y['memory_valid']), 1)
        for layer in self.tsfm:
            motion = layer(motion, context, mask=body_valid[:, None, None, :],
                           context_mask=memory_valid[:, None, None, :])
        result = self.output_process(motion[:, -1:])
        if self.anchored:
            # Observed reference removes predicted reference delta. Its target
            # is normalized identity, also used by the re-encoded P condition.
            result = torch.cat((result[..., :198], y['prior_mu'][..., 198:207], result[..., 207:]), -1)
        return result
