"""Paired histories and causal observations for the four-way G experiment."""
import hashlib
import json
from pathlib import Path

import torch

from egorecover.codec import BodyState, MotionCodec, planar_reference, transform_from_9d
from egorecover.conditioning import prepare_conditioning
from egorecover.prior_cv_residual import predict
from egorecover.prior_real_history import input_batch
from egorecover.prior_two_forward import FIELDS, physical_batch, state_map


GROUPS = {'T0': ('gaussian', False), 'T1': ('history', False),
          'T2': ('gaussian', True), 'T3': ('history', True)}


def mixture(step):
    return .5 * min(1., max(0., (step - 2000) / 4000))


def keyed_seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'big') % (2**63-1)


def native_states(codec, codes, anchor):
    """Full E7 SE(3) accumulation, then express unchanged bodies in planar refs.

    anchor is source-world-from-canonical; world z remains floor-relative.
    Unlike decode_current(planar), no delta is projected before accumulation.
    """
    raw = codec.denormalize(codes.float())
    delta = transform_from_9d(raw[..., 198:207])
    local = transform_from_9d(raw[..., :198].unflatten(-1, (22, 9)))
    ref = anchor
    refs = []
    for t in range(len(raw)):
        ref = ref @ delta[t]
        refs.append(ref)
    full = torch.stack(refs)
    return BodyState(full[:, None] @ local, planar_reference(full), raw[..., 207:])


def sensor_conditions(codec, history, head, images, prior):
    """head/images are exactly 20 past + current; no labels are accepted."""
    b, h = history.reference.shape[:2]
    if h != 20 or head.shape != (b, 21, 9) or images.shape != (b, 21, 1024):
        raise ValueError('Expected exactly 20 past + one current observation.')
    inputs = input_batch(codec, history)
    with torch.no_grad():
        mu = predict(prior, inputs)
    # Each observation delta connects the previous submitted body reference.
    # At the start of the visible buffer, use the same outer reference as H.
    previous = torch.cat((planar_reference(history.reference[:, :1]), history.reference), 1)
    traj = codec.encode_observation(head, previous, torch.ones((b, 21), dtype=torch.bool, device=head.device))
    y = {'history_motion': inputs['history_motion'], 'history_valid': torch.ones((b,h),dtype=torch.bool,device=head.device),
         'prior_mu': mu, 'traj': traj[:, -1:], 'img_embs': images[:, -1:],
         'traj_mask': torch.zeros((b,1),dtype=torch.bool,device=head.device),
         'img_mask': torch.zeros((b,1),dtype=torch.bool,device=head.device),
         'valid_frames': torch.ones((b,1),dtype=torch.bool,device=head.device),
         'past_traj':traj[:,:-1], 'past_img_embs':images[:,:-1],
         'past_traj_mask':torch.zeros((b,h),dtype=torch.bool,device=head.device),
         'past_img_mask':torch.zeros((b,h),dtype=torch.bool,device=head.device)}
    return prepare_conditioning(y)


class PairedSequences:
    def __init__(self, output, group, *, verify=True, limit=None, take_names=None):
        from egorecover.evaluation_protocol import file_sha256
        self.output = Path(output)
        manifest = json.loads((self.output/'data/manifest.json').read_text())
        if group not in ('train','dev','test'):
            raise ValueError(group)
        records = [r for r in manifest['records'] if r['group']==group]
        if take_names is not None:
            records = [r for r in records if r['base_take_name'] in take_names]
        if limit:
            records = records[:limit]
        if not records:
            raise ValueError('Empty data group.')
        self.records, self.group = records, group
        self.names = [r['base_take_name'] for r in records]
        self.ids = [r['episode_id'] for r in records]
        self.manifest_sha = file_sha256(self.output/'data/manifest.json')
        packets = []
        for r in records:
            path = self.output/'data/episodes'/f"{r['episode_id']}.pt"
            mark = json.loads(path.with_suffix('.json').read_text())
            if mark['manifest_sha256'] != self.manifest_sha or not mark['completed']:
                raise ValueError('Data identity mismatch.')
            if verify and file_sha256(path) != mark['sha256']:
                raise ValueError('Altered episode.')
            p = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
            if p['record'] != r:
                raise ValueError('Episode target identity mismatch.')
            packets.append(p)
        self.stats = packets[0]['stats']
        self.truth = BodyState(*(torch.stack([p['truth'][k] for p in packets]) for k in FIELDS))
        self.teacher = BodyState(*(torch.stack([p['teacher'][k] for p in packets]) for k in FIELDS))
        self.head = torch.stack([p['head'] for p in packets])
        self.images = torch.stack([p['images'] for p in packets])
        self.beta = torch.stack([p['beta_boot'] for p in packets])
        self.teacher_window_beta = (torch.stack([p['teacher_window_beta'] for p in packets])
                                   if all('teacher_window_beta' in p for p in packets) else None)
        self.take_episodes = [torch.tensor([i for i,n in enumerate(self.names) if n==take])
                              for take in sorted(set(self.names))]

    def sample(self, count, generator):
        if self.group != 'train':
            raise ValueError('Random sampling only on train.')
        takes = torch.randint(len(self.take_episodes),(count,),generator=generator)
        u = torch.rand(count,generator=generator)
        rows = torch.tensor([self.take_episodes[int(t)][int(float(v)*len(self.take_episodes[int(t)]))]
                             for t,v in zip(takes,u)])
        times = torch.randint(20,200,(count,),generator=generator)
        return torch.stack((rows,times),1)

    def examples(self, indices, choose, device):
        if indices.ndim != 2 or indices.shape[1]!=2 or choose.shape != (len(indices),):
            raise ValueError('Invalid paired example indices.')
        e,t = indices.T
        if not bool(((t>=20)&(t<200)).all()):
            raise ValueError('Target outside causal range.')
        ts = t[:,None] + torch.arange(-20,0)
        selected=[]
        for k in FIELDS:
            gt = getattr(self.truth,k)[e[:,None],ts]
            pred = getattr(self.teacher,k)[e[:,None],ts]
            value = torch.where(choose.reshape(-1,*([1]*(gt.ndim-1))),pred,gt)
            selected.append(value.to(device))
        history = BodyState(*selected)
        target = state_map(self.truth,lambda x:x[e,t].to(device))
        obs_t = t[:,None] + torch.arange(-20,1)
        return history,target,self.beta[e].to(device),self.head[e[:,None],obs_t].to(device),self.images[e[:,None],obs_t].to(device)

    def batch(self, codec, prior, indices, choose, device):
        history,target,beta,head,images = self.examples(indices,choose,device)
        batch = physical_batch(codec,history,target,beta)
        y = sensor_conditions(codec,history,head,images,prior)
        return batch,y
