"""Expanded train episodes with fixed original dev tensors and take-balanced sampling."""
import json
from pathlib import Path
import torch
from egorecover.codec import BodyState
from egorecover.evaluation_protocol import file_sha256
from egorecover.prior_two_forward import PriorSequences, FIELDS

SCOPE = 'audited_prior_expanded_train_fixed_dev_v1'


class ExpandedPriorSequences(PriorSequences):
    def __init__(self, directory, group):
        if group not in ('train', 'dev'):
            raise ValueError('Only train/dev may be loaded.')
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text())
        if not report.get('completed') or file_sha256(directory/'sequences.pt') != report['cache_sha256']:
            raise ValueError('Incomplete or altered expanded cache.')
        payload = torch.load(directory/'sequences.pt', map_location='cpu', weights_only=True, mmap=True)
        self.identity = payload['identity']
        if self.identity != report['identity'] or self.identity['scope'] != SCOPE:
            raise ValueError('Expanded cache identity mismatch.')
        splits = self.identity['splits']
        all_takes = sum(splits.values(), [])
        if len(all_takes) != len(set(all_takes)):
            raise ValueError('Take split leakage.')
        names, ids = payload['take_names'], payload['episode_ids']
        if len(ids) != len(set(ids)) or len(names) != len(ids) or set(names) != set(splits['train']+splits['dev']):
            raise ValueError('Unexpected episode coverage.')
        if any(names.count(t) != 1 for t in splits['dev']):
            raise ValueError('Dev must retain exactly one original episode per take.')
        selected = [i for i,t in enumerate(names) if t in splits[group]]
        self.base_takes = [names[i] for i in selected]
        self.takes = self.base_takes if group == 'dev' else [ids[i] for i in selected]
        self.group = group
        self.states = BodyState(*(payload[k][selected] for k in FIELDS))
        self.beta = payload['beta_boot'][selected]
        self.stats = payload['stats']
        n = len(selected)
        for k, shape in {'joints':(n,200,22,4,4),'reference':(n,200,4,4),'auxiliary':(n,200,36)}.items():
            v=getattr(self.states,k)
            if v.shape != shape or not bool(torch.isfinite(v).all()):
                raise ValueError('Invalid expanded body tensors.')
        if self.beta.shape != (n,10) or not bool(torch.isfinite(self.beta).all()):
            raise ValueError('Invalid model bootstrap shapes.')
        self.take_episodes = [torch.tensor([i for i,t in enumerate(self.base_takes) if t==name])
                              for name in sorted(set(self.base_takes))]
        if group == 'dev':
            original = PriorSequences(self.identity['original_data'], 'dev')
            if self.takes != original.takes:
                raise ValueError('Fixed dev order changed.')
            for k in FIELDS:
                if not torch.equal(getattr(self.states,k),getattr(original.states,k)):
                    raise ValueError('Fixed dev tensors changed.')
            if not torch.equal(self.beta,original.beta):
                raise ValueError('Fixed dev shapes changed.')

    def sample(self, count, generator):
        if self.group != 'train':
            raise ValueError('Sampling is restricted to train.')
        take_ids = torch.randint(len(self.take_episodes),(count,),generator=generator)
        uniform = torch.rand(count,generator=generator)
        sequences = torch.tensor([self.take_episodes[int(t)][int(float(u)*len(self.take_episodes[int(t)]))]
                                  for t,u in zip(take_ids,uniform)])
        times = torch.randint(40,200,(count,),generator=generator)
        return torch.stack((sequences,times),1)
