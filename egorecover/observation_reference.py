"""Independent observation memory and observed-reference G ablation.

P remains in its existing physical-history protocol. An observed reference is
a coordinate frame, never an assertion that the SMPL head equals the camera.
"""
import torch

from egorecover.actions import action_masks
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
MEMORY_MASKS = {'memory_traj_mask', 'memory_img_mask'}


def _action_masks(img_mask, traj_mask, action, scope):
    if scope not in ('current', 'window'):
        raise ValueError('Action scope must be current or window.')
    available = torch.ones_like(img_mask[:, -1:])
    image_drop, trajectory_drop = action_masks(
        action, img_available=available, traj_available=available)
    image, trajectory = img_mask.clone(), traj_mask.clone()
    if scope == 'window':
        image = image | image_drop
        trajectory = trajectory | trajectory_drop
    else:
        image[:, -1:] |= image_drop
        trajectory[:, -1:] |= trajectory_drop
    return image, trajectory


def prepare(y):
    if not MEMORY <= y.keys():
        raise ValueError('Complete observation memory required.')
    core = {k: v for k, v in y.items() if k not in MEMORY | MEMORY_MASKS}
    h = core['history_motion']; b = len(h)
    if h.ndim != 3 or h.shape[1:] != (20, 243):
        raise ValueError('Expected [B,20,243] body history.')
    if y['memory_img'].ndim != 3:
        raise ValueError('Memory must be [B,M,D].')
    m = y['memory_img'].shape[1]
    if m not in (21, 80):
        raise ValueError('Expected 21 or 80 observation slots.')
    valid = binary_mask(y['memory_valid'], (b, m), h.device, 'memory_valid')
    if not bool(valid[:, -21:].all()):
        raise ValueError('The current and previous 20 physical time slots are required.')
    present = MEMORY_MASKS & y.keys()
    if present and present != MEMORY_MASKS:
        raise ValueError('Both observation-memory masks must be supplied together.')
    masks = {}
    for kind in ('img', 'traj'):
        key = f'memory_{kind}_mask'
        mask = binary_mask(y[key], (b, m), h.device, key) if present else torch.zeros_like(valid)
        mask = mask | ~valid
        mask[:, -1:] |= binary_mask(core[f'{kind}_mask'], (b, 1), h.device, f'{kind}_mask')
        core[f'{kind}_mask'] = mask[:, -1:]
        masks[key] = mask
    clean = prepare_conditioning(core, history_length=20)
    clean['memory_valid'] = valid
    clean.update(masks)
    for k, width in [('memory_img', 1024), ('memory_traj', 18)]:
        value = floating_tensor(y[k], (b, m, width), h, k)
        clean[k] = finite_payload(value, valid & ~masks[k + '_mask'], k)
    return clean


def apply_action(y, action, *, scope='current'):
    """Action scope is explicit; a00/window also hides all past raw sensors."""
    if not MEMORY <= y.keys():
        raise ValueError('Complete observation memory required.')
    present = MEMORY_MASKS & y.keys()
    if present and present != MEMORY_MASKS:
        raise ValueError('Both observation-memory masks must be supplied together.')
    candidate = dict(y)
    h = y['history_motion']; b, m = len(h), y['memory_img'].shape[1]
    masks = []
    for kind in ('img', 'traj'):
        key = f'memory_{kind}_mask'
        mask = binary_mask(y[key], (b, m), h.device, key) if key in y else torch.zeros_like(y['memory_valid'])
        mask = mask | ~y['memory_valid'].bool()
        mask[:, -1:] |= binary_mask(y[f'{kind}_mask'], (b, 1), h.device, f'{kind}_mask')
        masks.append(mask)
    image, trajectory = _action_masks(*masks, action, scope)
    candidate.update(memory_img_mask=image, memory_traj_mask=trajectory,
                     img_mask=image[:, -1:], traj_mask=trajectory[:, -1:])
    return prepare(candidate)


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


def conditions(codec, history, head, images, valid, prior, anchored, *,
               img_available=None, traj_available=None, action='a11', action_scope='current'):
    """No targets/labels accepted; returns y, decode reference and physical P."""
    b, m = head.shape[:2]
    valid = binary_mask(valid, (b, m), head.device, 'memory_valid')
    available = []
    for name, value in [('img_available', img_available), ('traj_available', traj_available)]:
        available.append(valid if value is None else binary_mask(value, (b, m), head.device, name) & valid)
    image_mask, trajectory_mask = _action_masks(~available[0], ~available[1], action, action_scope)
    if anchored and bool(trajectory_mask[:, -1].any()):
        raise ValueError('Observed-reference G requires the current trajectory to be retained and available.')
    identity = head.new_tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])
    safe_head = torch.where(trajectory_mask[..., None], identity, head)
    inp = input_batch(codec, history)
    with torch.no_grad():
        mu = predict(prior, inp)
        pstate = codec.decode_current(mu[:, 0], history.reference[:, -1])
    reference = (planar_reference(transform_from_9d(safe_head[:, -1])) if anchored
                 else history.reference[:, -1])
    if anchored:
        mu = encode_in_reference(codec, pstate, reference)[:, None]
    # Keep the pretrained delta scale: older sensors use adjacent observed
    # references. The final 21 slots exactly retain the previous G interface.
    refs = planar_reference(transform_from_9d(safe_head))
    previous = torch.cat((refs[:, :1], refs[:, :-1]), 1)
    # An older sensor slot without its preceding observation has no measured
    # adjacent delta. Use identity there; the final 21 slots still use H.
    preceding_available = torch.cat((valid[:, :1] & ~trajectory_mask[:, :1],
                                    valid[:, :-1] & ~trajectory_mask[:, :-1]), 1)
    previous = torch.where(preceding_available[..., None, None], previous, refs)
    previous[:, -21:] = torch.cat((planar_reference(history.reference[:, :1]), history.reference), 1)
    traj = codec.encode_observation(safe_head, previous, valid & ~trajectory_mask)
    b = len(head); visible = torch.ones((b, 1), dtype=torch.bool, device=head.device)
    y = {'history_motion': inp['history_motion'],
         'history_valid': visible.expand(-1, 20), 'prior_mu': mu,
         'traj': traj[:, -1:], 'img_embs': images[:, -1:],
         'traj_mask': trajectory_mask[:, -1:], 'img_mask': image_mask[:, -1:], 'valid_frames': visible,
         'memory_traj': traj, 'memory_img': images, 'memory_valid': valid,
         'memory_traj_mask': trajectory_mask, 'memory_img_mask': image_mask}
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
    conditioning_protocol = 'observation-memory-masks-v2'

    def __init__(self, *, action_scope='current', flow=None):
        if action_scope not in ('current', 'window'):
            raise ValueError('Action scope must be current or window.')
        super().__init__(source_mode='gaussian', sigma=1., flow=flow)
        self.action_scope = action_scope

    def prepare_conditioning(self, y):
        return prepare(y)

    def action_availability(self, y):
        y = prepare(y)
        if self.action_scope == 'current':
            return ~y['img_mask'], ~y['traj_mask']
        return ((y['memory_valid'] & ~y['memory_img_mask']).any(1, keepdim=True),
                (y['memory_valid'] & ~y['memory_traj_mask']).any(1, keepdim=True))

    def apply_action(self, y, action):
        return apply_action(y, action, scope=self.action_scope)

    def training_losses(self, model, target_current, y, *, epsilon, t):
        y = prepare(y)
        return self.flow.training_losses(model, target_current, model_kwargs={'y': y},
            noise=self.build_source(y['prior_mu'], epsilon=epsilon), t=t, return_diagnostics=True)

    @torch.no_grad()
    def sample(self, model, y, *, epsilon, num_steps=None):
        if model.training:
            raise ValueError('Sampling requires eval mode.')
        y = prepare(y)
        source = self.build_source(y['prior_mu'], epsilon=epsilon)
        return self.flow.sample_loop(model, source.shape, model_kwargs={'y': y}, noise=source,
            num_steps=num_steps, device=source.device, repaint_enabled=False)
