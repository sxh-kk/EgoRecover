"""Prepare take-disjoint paired GT/native-E7 histories for G training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import time

import torch

from data_pipeline.ee4d_mismatch.source import EE4DSource, annotation_digest
from egorecover.annotations import body_states_from_supervision
from egorecover.bootstrap_shapes import load_e7_initializer
from egorecover.codec import BodyState, MotionCodec
from egorecover.e7_ablation import conditions, matrix
from egorecover.evaluation_protocol import file_sha256
from egorecover.g_pretraining import native_states, keyed_seed
from egorecover.prior_two_forward import FIELDS, state_map
from mydiffusion.flow_matching import FlowMatching
from run.e7_ablation import save, write_json
from run.prepare_prior_expanded import select_clips


def plan(cfg, output):
    output = Path(output)
    if (output/'data/manifest.json').exists():
        old=json.loads((output/'plan.json').read_text())
        if old != cfg: raise ValueError('Existing plan differs.')
        return
    excluded_train = json.loads(Path('config/egorecover_train_72take_split_v1.json').read_text())['splits']
    exclude = set(excluded_train['dev']+excluded_train['holdout'])
    excluded_val=set()
    # All historical project val manifests, including the two original smoke takes.
    for path in Path('config').glob('*split*.json'):
        doc=json.loads(path.read_text())
        if doc.get('base_split')=='val':
            excluded_val.update(sum(doc.get('splits',{}).values(),[]))
    for path in Path('data').glob('ee4d_mismatch*/*jsonl'):
        if 'train_' in str(path.parent): continue
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r.get('base_split','val')=='val': excluded_val.add(r['base_take_name'])
    groups={}; records=[]; reports={}; discarded=[]; source_ids={}
    for split in ('train','val'):
        print('Inventory',split,flush=True)
        source=EE4DSource(cfg['source_root'],split)
        candidates,inventory=source.inventory(200)
        official=json.loads(source.split_path.read_text())['take_uid_to_split']
        eligible={}
        for take,names in sorted(candidates.items()):
            if official.get(source.metadata[take]['take_uid'])!=split:
                raise ValueError('Official split mismatch.')
            if take in (exclude if split=='train' else excluded_val): continue
            accepted=[n for n in names if torch.isfinite(torch.as_tensor(source.motion[n]['floor_height'])).all()
                      and float(source.motion[n]['floor_height'])!=0]
            if accepted: eligible[take]=accepted
            else: discarded.append({'take':take,'reason':'missing_or_zero_floor_not_verified'})
        if split=='train':
            groups['train']=sorted(eligible)
        else:
            strata={}
            for take in eligible:
                meta=source.metadata[take]; task=meta.get('parent_task_name') or meta.get('task_name') or 'unknown'
                strata.setdefault(task,[]).append(take)
            groups['dev'],groups['test']=[],[]
            rng=random.Random(cfg['split_seed'])
            for task,names in sorted(strata.items()):
                rng.shuffle(names)
                for i,take in enumerate(names):
                    # Balance singleton strata too without querying any outcomes.
                    group=('dev' if len(groups['dev'])<=len(groups['test']) else 'test') if len(names)==1 else ('dev' if i%2==0 else 'test')
                    groups[group].append(take)
            for g in ('dev','test'): groups[g].sort()
        for group in (('train',) if split=='train' else ('dev','test')):
            for take in groups[group]:
                clips=select_clips(source,take,eligible[take],cap=cfg['clips_per_train_take'] if group=='train' else 1)
                for r in clips:
                    r.update(group=group,base_split=split,
                        task=source.metadata[take].get('parent_task_name') or source.metadata[take].get('task_name'),
                        floor_height=float(source.motion[r['base_seq_name']]['floor_height']))
                    records.append(r)
        reports[split]=inventory
        # Stat identity here; immutable raw packet/annotation hashes are recorded per episode.
        source_ids[split]={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                           for p in (source.motion_path,source.feature_path)}
        del source
    all_takes=sum(groups.values(),[])
    if len(all_takes)!=len(set(all_takes)) or not all(groups.values()): raise ValueError('Invalid take split.')
    previous=json.loads(Path('exp/egorecover_prior_data_budget/v1/data/manifest.json').read_text())['train_takes']
    missing=sorted(set(previous)-set(groups['train']))
    stats_path=Path(cfg['source_root'])/'uniegomotion/v4_beta_ee_train_stats.pt'
    manifest={'protocol':cfg['protocol'],'groups':groups,'records':records,'inventory':reports,
        'excluded_old_train_diagnostic':sorted(exclude),'excluded_project_val':sorted(excluded_val),
        'discarded':discarded,'old192_not_eligible':missing,'source_assets':source_ids,
        'stats_sha256':file_sha256(stats_path),'checkpoint_sha256':file_sha256(cfg['e7_checkpoint']),
        'old_prior_sha256':file_sha256(cfg['prior_checkpoint']),
        'source_split_sha256':file_sha256(Path(cfg['source_root'])/'annotations/splits.json'),
        'initialization_coverage': 'Repository E7 loader uses official train for fitting and val for validation. Checkpoint contains no take-level training manifest; official val is held out from gradient training under this documented recipe, but was used by upstream validation. This is a newly held-out project comparison, not a previously untouched benchmark.'}
    write_json(output/'plan.json',cfg);write_json(output/'data/manifest.json',manifest)
    write_json(output/'data/split.json',{'groups':groups,'initialization_coverage':manifest['initialization_coverage']})
    print('Frozen', {k:len(v) for k,v in groups.items()},'clips',len(records),'old192 exclusions',missing,flush=True)


@torch.no_grad()
def prepare(cfg, output, shard, shards, device, limit=None):
    output=Path(output); manifest=json.loads((output/'data/manifest.json').read_text())
    manifest_sha=file_sha256(output/'data/manifest.json')
    if file_sha256(cfg['e7_checkpoint'])!=manifest['checkpoint_sha256']: raise ValueError('E7 changed.')
    records=manifest['records'][shard::shards]
    if limit:
        # Train and dev examples for an actual end-to-end smoke, never test.
        records=([r for r in manifest['records'] if r['group']=='train'][:limit]
                 + [r for r in manifest['records'] if r['group']=='dev'][:1])
    stats=torch.load(Path(cfg['source_root'])/'uniegomotion/v4_beta_ee_train_stats.pt',weights_only=False,map_location='cpu')
    stats={k:torch.as_tensor(stats[k]).float() for k in ('motion_mean','motion_std','traj_mean','traj_std')}
    codec=MotionCodec(stats)
    model=load_e7_initializer(cfg['e7_checkpoint'],device=device,weight_source='ema')
    flow=FlowMatching(); row=matrix()[1]
    source=None; source_split=None
    for index,r in enumerate(records):
        path=output/'data/episodes'/f"{r['episode_id']}.pt"; marker=path.with_suffix('.json')
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['manifest_sha256']!=manifest_sha or file_sha256(path)!=m['sha256']: raise ValueError('Altered prepared cache.')
            if m.get('producer_sha256')==file_sha256(__file__):
                print('REUSE',r['episode_id'],flush=True);continue
            print('REBUILD new preparation code',r['episode_id'],flush=True)
        if source_split!=r['base_split']:
            if source is not None: del source
            print('Loading source',r['base_split'],flush=True)
            source=EE4DSource(cfg['source_root'],r['base_split']); source_split=r['base_split']
            print('Loaded source',source_split,flush=True)
            for p in (source.motion_path,source.feature_path):
                expected=manifest['source_assets'][source_split][str(p)]
                if {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}!=expected:
                    raise ValueError('Source asset changed after manifest freeze.')
        start=time.monotonic(); clean=source.clean(r); supervision=source.supervision(r)
        observations={k:torch.as_tensor(clean[k]).clone() for k in ('aria_traj_obs','img_available','traj_available')}
        observations['img_feats']=source.visual_payload(r['base_take_name'],clean['img_source_idx'],clean['img_available'])
        if not bool(observations['img_available'].all() & observations['traj_available'].all()):
            raise ValueError('Clean paired protocol requires available observations throughout.')
        case={'observations':observations,'supervision':{'floor_height':torch.tensor(r['floor_height'])}}
        windows=[(0,20)]+[(max(0,t-79),t+1) for t in range(20,200)]
        encoded=[]
        for first,last in windows:
            y,anchor,_=conditions(case,first,last,row,codec)
            epsilon=torch.randn(80,243,generator=torch.Generator().manual_seed(keyed_seed(r['episode_id'],1062,last-1)))
            encoded.append((y,anchor,epsilon))
        states=[]; native_betas=[]; beta=None; max_roundtrip=0.
        for offset in range(0,len(windows),16):
            jobs=encoded[offset:offset+16]
            y={k:torch.stack([j[0][k] for j in jobs]).to(device) for k in jobs[0][0]}
            noise=torch.stack([j[2] for j in jobs]).to(device)
            codes=flow.sample_loop(model,noise.shape,{'y':y},noise=noise).float().cpu()
            for j,(_,anchor,_) in enumerate(jobs):
                first,last=windows[offset+j]
                state=native_states(codec,codes[j,:last-first],anchor)
                if offset+j==0:
                    states.extend(state_map(state,lambda x:x[t]) for t in range(20))
                    beta=codec.denormalize(codes[j,:20])[:,-10:].mean(0)
                    native_betas.extend([beta]*20)
                    # Independent native decoder parity on the startup window.
                    from dataset.representation_utils import repre_to_full_sequence_v4_beta
                    from dataset.smpl_utils import get_smpl
                    if not hasattr(prepare,'smpl'): prepare.smpl=get_smpl().eval()
                    _,_,dense=repre_to_full_sequence_v4_beta(codec.denormalize(codes[j,:20]),None,prepare.smpl,None,None)
                    expected=(anchor[:3,:3]@dense[...,None])[...,0]+anchor[:3,3]
                    max_roundtrip=float((expected-state.joints[...,:3,3]).abs().max())
                    if max_roundtrip>2e-5: raise ValueError('Native decoder parity failed.')
                else:
                    states.append(state_map(state,lambda x:x[-1]))
                    native_betas.append(codec.denormalize(codes[j,:last-first])[:,-10:].mean(0))
        teacher=BodyState(*(torch.stack([getattr(s,k) for s in states]) for k in FIELDS))
        truth=body_states_from_supervision(supervision,observations['aria_traj_obs'],floor_height=r['floor_height'])
        from egorecover.smpl_evaluation import prepare_ground_truth
        asset_audit=prepare_ground_truth(prepare.smpl,supervision,[0,19,199])['asset_audit']
        # Verify planar re-encoding preserves the teacher's dense world positions.
        code=codec.encode_current(teacher,teacher.reference)
        restored=codec.decode_current(code,teacher.reference)
        err=float((restored.joints-teacher.joints).abs().max())
        if err>5e-4: raise ValueError('Planar state roundtrip failed.')
        head=observations['aria_traj_obs'].clone();head[:,8]-=r['floor_height']
        packet={'record':r,'stats':stats,'truth':{k:getattr(truth,k) for k in FIELDS},
                'teacher':{k:getattr(teacher,k) for k in FIELDS},'beta_boot':beta,
                'head':head,'images':observations['img_feats'],'teacher_window_beta':torch.stack(native_betas)}
        if any(not torch.isfinite(v).all() for v in (head,packet['images'],beta,*packet['truth'].values(),*packet['teacher'].values())):
            raise ValueError('Nonfinite episode.')
        save(path,packet)
        write_json(marker,{'completed':True,'manifest_sha256':manifest_sha,'sha256':file_sha256(path),'producer_sha256':file_sha256(__file__),
            'annotation_sha256':annotation_digest(supervision),'native_startup_max_abs_m':max_roundtrip,
            'planar_roundtrip_max_abs':err,'smpl_asset_audit':asset_audit,'seconds':time.monotonic()-start})
        write_json(output/f'data/shard{shard}.progress.json',{'completed':index+1,'total':len(records),
            'episode':r['episode_id'],'take':r['base_take_name'],'pid':os.getpid()})
        print(f'{index+1}/{len(records)} {r["base_take_name"]} {r["episode_id"]} {time.monotonic()-start:.1f}s',flush=True)
    write_json(output/f'data/shard{shard}.done.json',{'completed':True,'manifest_sha256':manifest_sha,'records':len(records),'limited':limit is not None})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','prepare']);p.add_argument('--config',default='config/egorecover_g_pretraining_v1.json')
    p.add_argument('--output',type=Path,default=Path('exp/egorecover_g_pretraining/v1'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4)
    cfg=json.loads(Path(args.config).read_text())
    if args.stage=='plan': plan(cfg,args.output)
    else:
        lock_path=args.output/f'data/shard{args.shard}.lock'
        lock=lock_path.open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(cfg,args.output,args.shard,args.shards,args.device,args.limit)


if __name__=='__main__': main()
