"""Explicit physical baselines and shared metrics for the CV residual experiment."""
import torch
from egorecover.codec import BodyState
from egorecover.fk import FixedShapeFK
from egorecover.prior_two_forward import (FIELDS, PriorSequences, physical_batch, state_map,
                                         summarize, measure as original_measure)
from eval.metrics import reconstruction_error


def predict(prior, batch):
    mode = getattr(prior, 'base_mode', 'hold')
    if mode not in ('hold', 'constant_velocity'):
        raise ValueError('Invalid physical baseline mode.')
    history = batch['history_motion']
    base = batch['base_mu' if mode == 'hold' else 'velocity_mu']
    return prior(history, torch.ones(history.shape[:2], dtype=torch.bool, device=history.device), base)


def mixed_batch(prior, codec, context, target, beta, *, probability, generator):
    if probability != 0:
        raise ValueError('CV residual experiment requires GT histories.')
    return physical_batch(codec, state_map(context, lambda x: x[:, -20:]), target, beta), 0.


def prediction_for(model, batch):
    if isinstance(model, str):
        return batch[{'hold': 'base_mu', 'constant_velocity': 'velocity_mu'}[model]]
    return predict(model, batch)


def rotation_angle(predicted, target):
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1)
    sine = skew.norm(dim=-1) / 2
    return torch.rad2deg(torch.atan2(sine, cosine))


@torch.no_grad()
def measure(codec, prediction, batch, target, smpl):
    values, joints_cpu, state = original_measure(codec, prediction, batch, target, smpl)
    values['body_mpjpe_mm'] = values['fk22_mm']
    gt_cpu = batch['target_joints'].cpu()
    pa = reconstruction_error(joints_cpu.numpy(), gt_cpu.numpy(), reduction=None)
    values['pa_mpjpe_mm'] = torch.as_tensor(pa * 1000)
    fk = FixedShapeFK(smpl, batch['beta_boot'])
    target_same = fk.project(target)[0]
    values['same_shape_mpjpe_mm'] = ((joints_cpu.to(target_same.device) - target_same)
                                    .norm(dim=-1).mean(-1) * 1000).cpu()
    pred_r, gt_r = state.joints[..., :3, :3], target.joints[..., :3, :3]
    parents = fk.parents[1:]
    local_pred = pred_r[:, parents].transpose(-1, -2) @ pred_r[:, 1:]
    local_gt = gt_r[:, parents].transpose(-1, -2) @ gt_r[:, 1:]
    values['local_rotation_deg'] = rotation_angle(local_pred, local_gt).mean(-1).cpu()
    values['root_rotation_deg'] = rotation_angle(pred_r[:, 0], gt_r[:, 0]).cpu()
    return values, joints_cpu, state


@torch.no_grad()
def evaluate_single(model, data, codec, smpl, device, batch_size=32):
    if not isinstance(model, str):
        model.eval()
    indices = data.indices()
    arrays, poses, truth = {}, [], []
    for start in range(0, len(indices), batch_size):
        history, target, beta = data.examples(indices[start:start + batch_size], device)
        batch = physical_batch(codec, history, target, beta)
        metrics, joints, _ = measure(codec, prediction_for(model, batch), batch, target, smpl)
        for key, value in metrics.items():
            arrays.setdefault(key, []).append(value)
        poses.append(joints)
        truth.append(batch['target_joints'].cpu())
    arrays = {key: torch.cat(value) for key, value in arrays.items()}
    scores = {key: summarize(value, indices, data.takes) for key, value in arrays.items()}
    matched = indices[:, 1] >= 40
    scores['matched_fk22_mm'] = summarize(arrays['fk22_mm'][matched], indices[matched], data.takes)
    return scores, {'indices': indices, 'takes': data.takes, 'metrics': arrays,
                    'joints': torch.cat(poses), 'gt_joints': torch.cat(truth)}


@torch.no_grad()
def evaluate_rollout(model, data, codec, smpl, device, batch_size=32):
    if not isinstance(model, str):
        model.eval()
    indices = data.indices(20, 181, 20)
    metrics, poses, truth = {h: {} for h in (1, 2, 4, 8, 10)}, {}, {}
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        history, _, beta = data.examples(selected, device)
        for horizon in range(1, 11):
            future = selected.clone()
            future[:, 1] += horizon - 1
            _, target, _ = data.examples(future, device)
            batch = physical_batch(codec, history, target, beta)
            prediction = prediction_for(model, batch)
            state = codec.decode_current(prediction[:, 0], batch['previous_reference'])
            if horizon in metrics:
                measured, joints, _ = measure(codec, prediction, batch, target, smpl)
                for key, value in measured.items():
                    metrics[horizon].setdefault(key, []).append(value)
                poses.setdefault(horizon, []).append(joints)
                truth.setdefault(horizon, []).append(batch['target_joints'].cpu())
            history = BodyState(*(torch.cat((getattr(history, key)[:, 1:], getattr(state, key)[:, None]), 1)
                                  for key in FIELDS))
    arrays = {h: {k: torch.cat(v) for k, v in group.items()} for h, group in metrics.items()}
    summaries = {str(h * 100): {**summarize(group['fk22_mm'], indices, data.takes),
                   'metrics': {k: summarize(v, indices, data.takes) for k, v in group.items()}}
                 for h, group in arrays.items()}
    return summaries, {'indices': indices, 'takes': data.takes, 'metrics': arrays,
                      'errors_mm': {h: v['fk22_mm'] for h, v in arrays.items()},
                      'joints': {h: torch.cat(v) for h, v in poses.items()},
                      'gt_joints': {h: torch.cat(v) for h, v in truth.items()}}
