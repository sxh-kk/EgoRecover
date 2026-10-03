"""Controlled E7 inference factors; no learned parameter or training changes."""
import hashlib

import torch

from dataset.canonicalization import rotation_to_make_this_forward_batch, rot_trans_to_matrix
from dataset.representation_utils import repre_to_full_sequence_v4_beta
from dataset.smpl_utils import evaluate_smpl
from egorecover.codec import transform_from_9d, transform_to_9d


def matrix():
    """Final four experiments: window x floor, all other choices upstream."""
    settings = [('E0', 'offline', 'source'), ('E1', 'causal', 'source'),
                ('E2', 'offline', 'startup'), ('E3', 'causal', 'startup')]
    return [dict(id=name, inference_id=name, window=window, floor=floor,
                 padding='pad80', noise='independent', encoder='native',
                 beta='window', decoder='native') for name,window,floor in settings]


def bounds(frame, mode, frames=200):
    if not 0 <= frame < frames or frames < 80:
        raise ValueError('Invalid episode/frame.')
    if mode == 'causal':
        return max(0, frame - 79), frame + 1
    if mode == 'offline':
        first = min((frame // 80) * 80, frames - 80)
        return first, first + 80
    raise ValueError(mode)


def draw_noise(take, draw, first, last, mode, length):
    # Independent mode matches the previous history experiment's seed formula.
    seed_text = f'{take}/{draw}/{last-1}'
    if mode != 'independent':
        raise ValueError(mode)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'big') % (2**63-1)
    generator = torch.Generator().manual_seed(seed)
    # Generate full width before truncating: padding has identical valid noise.
    return torch.randn(80, 243, generator=generator)[:length]


def floor_for(case, first, last, mode):
    if mode == 'startup':
        return float(case['startup']['floor_estimate_m'])
    if mode != 'source':
        raise ValueError(mode)
    value = float(case['supervision']['floor_height'])
    if value == 0:
        value = float(case['supervision']['kp3d'][first:last, [10, 11], 2].mean()) - .02
    return value


def native_head_encoding(head, floor):
    """Observation-only extraction of upstream v4's trajectory operations (CPU).

    Upstream quaternion helpers allocate CPU tensors. Keep preprocessing on CPU
    rather than silently rewriting upstream numerics for GPU operation.
    """
    head = transform_from_9d(head).clone()
    rotation = rotation_to_make_this_forward_batch(head[:1, :3, :3])
    translation = (rotation @ (head[:1, :3, 3] * head.new_tensor([-1, -1, 0]))[..., None])[..., 0]
    canonical = rot_trans_to_matrix(rotation, translation)[0]
    h = canonical[None] @ head
    h[:, 2, 3] -= floor
    r = rotation_to_make_this_forward_batch(h[:, :3, :3])
    t = (r @ (h[:, :3, 3] * h.new_tensor([-1, -1, 0]))[..., None])[..., 0]
    inverse_refs = rot_trans_to_matrix(r, t)
    refs = inverse_refs.inverse()
    delta = torch.cat((refs[:1], refs[:-1].inverse() @ refs[1:]))
    return torch.cat((transform_to_9d(inverse_refs @ h), transform_to_9d(delta)), -1), canonical.inverse()


def conditions(case, first, last, row, codec):
    obs = case['observations']
    if not bool(obs['traj_available'][first:last].all()):
        raise ValueError('Missing trajectory requires a separate anchoring policy.')
    floor = floor_for(case, first, last, row['floor'])
    head = obs['aria_traj_obs'][first:last].clone()
    if row['encoder'] != 'native' or row['padding'] != 'pad80':
        raise ValueError('The four-cell protocol uses native encoding and padding.')
    raw, anchor = native_head_encoding(head, floor)
    traj = codec.normalize(raw, 'traj')
    count = last - first
    length = 80
    def pad(x):
        return torch.cat((x, x.new_zeros(length-count, *x.shape[1:])))
    valid = torch.arange(length) < count
    y = {'traj': pad(traj), 'img_embs': pad(obs['img_feats'][first:last]),
         'valid_frames': valid.long(),
         'valid_img_embs': pad(obs['img_available'][first:last]).long(),
         'traj_mask': torch.zeros(length, dtype=torch.long),
         'img_mask': (~pad(obs['img_available'][first:last]).bool() & valid).long()}
    return y, anchor, floor


@torch.no_grad()
def decode(codes, anchor, floor, row, codec, smpl):
    """Valid codes only. Return FK22 and dense22 in the original source world."""
    raw = codec.denormalize(codes)
    if row['decoder'] != 'native' or row['beta'] != 'window':
        raise ValueError('The four-cell protocol preserves native decoder and window beta.')
    _, params, dense = repre_to_full_sequence_v4_beta(raw, None, smpl, None, None)
    joints = evaluate_smpl(smpl, params, return_joints_only=True)[:, :22]
    joints = (anchor[:3, :3] @ joints[..., None])[..., 0] + anchor[:3, 3]
    dense = (anchor[:3, :3] @ dense[..., None])[..., 0] + anchor[:3, 3]
    joints = joints.clone(); dense = dense.clone()
    joints[..., 2] += floor; dense[..., 2] += floor
    if not torch.isfinite(joints).all() or not torch.isfinite(dense).all():
        raise ValueError('Nonfinite reconstructed joints.')
    return joints, dense


def metric_arrays(pred, dense, gt, times):
    from eval.metrics import reconstruction_error
    pred, dense, gt = pred.cpu(), dense.cpu(), gt.cpu()
    errors = (pred-gt).norm(dim=-1)
    local = ((pred-pred[:, :1]) - (gt-gt[:, :1])).norm(dim=-1)
    result = {'body_mpjpe_mm': errors.mean(-1)*1000,
              'pa_mpjpe_mm': torch.as_tensor(reconstruction_error(pred.numpy(), gt.numpy(), reduction=None))*1000,
              'root_mm': errors[:, 0]*1000, 'head_joint15_mm': errors[:, 15]*1000,
              'root_relative_mm': local.mean(-1)*1000,
              'dense_mpjpe_mm': (dense-gt).norm(dim=-1).mean(-1)*1000}
    consecutive = times[1:] - times[:-1] == 1
    result['velocity_error_mmps'] = ((pred[1:]-pred[:-1])-(gt[1:]-gt[:-1])).norm(dim=-1).mean(-1)[consecutive]*10000
    return result
