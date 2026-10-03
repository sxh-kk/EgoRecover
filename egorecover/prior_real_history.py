"""Original-E7 causal windows and immutable physical states for P evaluation."""
import hashlib
import torch
from egorecover.codec import BodyState, planar_reference, transform_from_9d
from egorecover.prior_two_forward import FIELDS


def noise_seed(take, draw, frame):
    return int.from_bytes(hashlib.sha256(f'{take}/{draw}/{frame}'.encode()).digest()[:8], 'big') % (2**63-1)


def window_conditions(codec, observations, floors, frame):
    """Only observations <= frame enter the original E7 model."""
    if frame < 20 or frame >= observations['aria_traj_obs'].shape[1]:
        raise ValueError('Window frame outside committed rollout range.')
    first=max(0,frame-79);last=frame+1
    head=observations['aria_traj_obs'][:,first:last].clone()
    head[...,8]-=floors[:,None]
    available=observations['traj_available'][:,first:last]
    if not bool(available.all()):
        raise ValueError('This protocol requires available head trajectories; missing-head cases need a separate anchor policy.')
    refs=planar_reference(transform_from_9d(head))
    previous=torch.cat((refs[:,:1],refs[:,:-1]),1)
    encoded=codec.encode_observation(head,previous,available)
    valid=torch.ones(head.shape[:2],dtype=torch.bool,device=head.device)
    y={'traj':encoded,'img_embs':observations['img_feats'][:,first:last],
       'valid_frames':valid,'valid_img_embs':valid,'traj_mask':~available,
       'img_mask':~observations['img_available'][:,first:last]}
    return y,refs[:,0],(first,last)


def decode_window(codec, codes, anchor):
    """Decode each delta in order; callers commit only the terminal state."""
    previous=anchor
    for t in range(codes.shape[1]):
        state=codec.decode_current(codes[:,t],previous)
        previous=state.reference
    return state


def stack_states(states, dim=1):
    return BodyState(*(torch.stack([getattr(x,k) for x in states],dim) for k in FIELDS))


def input_batch(codec, history):
    """Inference-only P inputs; contains no target or current observation."""
    last=BodyState(*(getattr(history,k)[:,-1] for k in FIELDS))
    before=BodyState(*(getattr(history,k)[:,-2] for k in FIELDS))
    return {'history_motion':codec.encode_history(history,planar_reference(history.reference[:,0])),
            'base_mu':codec.continuation_prior(last)[:,None],
            'velocity_mu':codec.constant_velocity_prior(before,last)[:,None],
            'previous_reference':last.reference}


def paired_take_summary(predicted, original, takes, draws, frame_mask):
    """predicted [seed,case,time], original [seed,take,time]; equal take weight."""
    if predicted.ndim!=3 or original.ndim!=3 or len(takes)!=predicted.shape[1]:
        raise ValueError('Invalid paired score shapes.')
    names=sorted(set(takes));differences={};scores={};references={}
    if original.shape[1]!=len(names):raise ValueError('Original take axis must match sorted names.')
    if not torch.isfinite(predicted).all() or not torch.isfinite(original).all():raise ValueError('Nonfinite score.')
    for i,take in enumerate(names):
        selected=[j for j,t in enumerate(takes) if t==take]
        if sorted(draws[j] for j in selected)!=sorted(set(draws)):
            raise ValueError('Missing or duplicated draw for a take.')
        a=predicted[:,selected][:,:,frame_mask].double().mean((1,2))
        b=original[:,i,frame_mask].double().mean(1)
        differences[take]=(a-b).tolist();scores[take]=a.tolist();references[take]=b.tolist()
    return scores,references,differences
