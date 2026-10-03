"""Independent observation memory and observed-reference G ablation.

P remains in its existing physical-history protocol. An observed reference is
a coordinate frame, never an assertion that the SMPL head equals the camera.
"""
import torch

from egorecover.codec import (BodyState, planar_reference, rigid_inverse,
                             transform_from_9d, transform_to_9d)
from egorecover.conditioning import prepare_conditioning, finite_payload, binary_mask, floating_tensor
from egorecover.history_flow import HistoryFlow
from egorecover.losses import weighted_representation_mse
from egorecover.prior_cv_residual import predict
from egorecover.prior_real_history import input_batch
from egorecover.g_pretraining import PairedSequences

GROUPS = {'R0': (21, False), 'R1': (80, False), 'R2': (21, True), 'R3': (80, True)}
MEMORY = {'memory_traj', 'memory_img', 'memory_valid'}


def prepare(y):
    if not MEMORY <= y.keys():
        raise ValueError('Complete observation memory required.')
    clean = prepare_conditioning({k: v for k, v in y.items() if k not in MEMORY}, history_length=20)
    h = clean['history_motion']; b = len(h)
    if y['memory_img'].ndim != 3:
        raise ValueError('Memory must be [B,M,D].')
    m = y['memory_img'].shape[1]
    if m not in (21, 80):
        raise ValueError('Expected 21 or 80 observation slots.')
    valid = binary_mask(y['memory_valid'], (b, m), h.device, 'memory_valid')
    if not bool(valid[:, -21:].all()):
        raise ValueError('At least the current and previous 20 observations are required.')
    clean['memory_valid'] = valid
    for k, width in [('memory_img', 1024), ('memory_traj', 18)]:
        value = floating_tensor(y[k], (b, m, width), h, k)
        clean[k] = finite_payload(value, valid, k)
    return clean


def encode_in_reference(codec, state, reference):
    local = rigid_inverse(reference)[..., None, :, :] @ state.joints
    identity = torch.eye(4, device=reference.device, dtype=reference.dtype).expand_as(reference)
    return codec.normalize(torch.cat((transform_to_9d(local).flatten(-2),
        transform_to_9d(identity), state.auxiliary), -1))


def decode(codec, code, reference, anchored):
    if not anchored:
        return codec.decode_current(code, reference)
    raw = codec.denormalize(code)
    local = transform_from_9d(raw[..., :198].unflatten(-1, (22, 9)))
    return BodyState(reference[..., None, :, :] @ local, reference, raw[..., 207:])


def gather_observations(head, images, times, length):
    """Batch timelines; only indices <= each target are gathered, left padding."""
    if length not in (21, 80) or times.shape != (len(head),):
        raise ValueError('Invalid observation context.')
    if not bool(((times >= 20) & (times < head.shape[1])).all()):
        raise ValueError('Invalid target times.')
    indices = times[:, None] + torch.arange(1-length, 1, device=times.device)
    valid = indices >= 0
    rows = torch.arange(len(head), device=times.device)[:, None]
    h = head[rows, indices.clamp_min(0)]
    im = images[rows, indices.clamp_min(0)]
    return h, im, valid


def conditions(codec, history, head, images, valid, prior, anchored):
    """No targets/labels accepted; returns y, decode reference and physical P."""
    inp = input_batch(codec, history)
    with torch.no_grad():
        mu = predict(prior, inp)
        pstate = codec.decode_current(mu[:, 0], history.reference[:, -1])
    reference = (planar_reference(transform_from_9d(head[:, -1])) if anchored
                 else history.reference[:, -1])
    if anchored:
        mu = encode_in_reference(codec, pstate, reference)[:, None]
    # Same reference for all sensor tokens, independent of body-history length.
    traj = codec.encode_observation(head, reference[:, None].expand(-1, head.shape[1], -1, -1), valid)
    b = len(head); visible = torch.ones((b, 1), dtype=torch.bool, device=head.device)
    y = {'history_motion': inp['history_motion'],
         'history_valid': visible.expand(-1, 20), 'prior_mu': mu,
         'traj': traj[:, -1:], 'img_embs': images[:, -1:],
         'traj_mask': ~visible, 'img_mask': ~visible, 'valid_frames': visible,
         'memory_traj': traj, 'memory_img': images, 'memory_valid': valid}
    return prepare(y), reference, pstate


def target_batch(codec, target, reference, anchored):
    encoded = encode_in_reference(codec, target, reference) if anchored else codec.encode_current(target, reference)
    return {'target': encoded[:, None], 'reference': reference,
            'target_joints': target.joints[..., :3, 3]}


def objective(codec, prediction, batch, anchored):
    state = decode(codec, prediction[:, 0].float(), batch['reference'].float(), anchored)
    geometry = (state.joints[..., :3, 3] - batch['target_joints']).square().sum(-1).mean() / .01
    return weighted_representation_mse(prediction, batch['target']) + geometry


class ReferenceSequences(PairedSequences):
    def __init__(self, output, split, *, observation_length, anchored, **kwargs):
        super().__init__(output, split, **kwargs)
        self.observation_length, self.anchored = observation_length, anchored

    def batch(self, codec, prior, indices, choose, device):
        history, target, _, _, _ = self.examples(indices, choose, device)
        e, t = indices.T
        head, images, valid = gather_observations(self.head[e], self.images[e], t, self.observation_length)
        y, reference, _ = conditions(codec, history, head.to(device), images.to(device), valid.to(device), prior, self.anchored)
        return target_batch(codec, target, reference, self.anchored), y


class ObservationFlow(HistoryFlow):
    def __init__(self):
        super().__init__(source_mode='gaussian', sigma=1.)

    def training_losses(self, model, target_current, y, *, epsilon, t):
        y = prepare(y)
        return self.flow.training_losses(model, target_current, model_kwargs={'y': y},
            noise=self.build_source(y['prior_mu'], epsilon=epsilon), t=t, return_diagnostics=True)

    @torch.no_grad()
    def sample(self, model, y, *, epsilon, num_steps=10):
        if model.training:
            raise ValueError('Sampling requires eval mode.')
        y = prepare(y)
        source = self.build_source(y['prior_mu'], epsilon=epsilon)
        return self.flow.sample_loop(model, source.shape, model_kwargs={'y': y}, noise=source,
            num_steps=num_steps, device=source.device, repaint_enabled=False)
