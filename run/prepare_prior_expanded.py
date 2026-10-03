"""Build additional audited train clips while preserving the original dev tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from dataset.smpl_utils import get_smpl
from data_pipeline.ee4d_mismatch.source import annotation_digest
from egorecover.annotations import body_states_from_supervision
from egorecover.bootstrap_shapes import load_e7_initializer
from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from egorecover.prior_expanded import SCOPE, ExpandedPriorSequences
from egorecover.prior_two_forward import FIELDS
from egorecover.rollout import initialize_history
from egorecover.smpl_evaluation import prepare_ground_truth
from mydiffusion.flow_matching import FlowMatching
from run.adapt_prior_on_predictions import save_torch
from run.complete_stages import ROOT, SIGNAL, SPLIT, BOOTSTRAP, CHECKPOINT, append_log, now

ORIGINAL = ROOT/'exp/egorecover_prior_two_forward/v1/data'


def interval(record):
    start = int(record['base_seq_name'].rsplit('___',2)[1]) + 3*record['episode_start_motion_idx']
    return start, start + 3*record['num_frames']


def select_clips(source, take, names, original=None, cap=4):
    selected = [dict(original)] if original else []
    options = []
    for name in sorted(names):
        count = int(source.motion[name]['num_frames'])
        starts = set(range(0, count-199, 200)) | {count-200}
        if original:
            end = interval(original)[1]
            start30 = int(name.rsplit('___',2)[1])
            first = max(0, (end-start30+2)//3)
            starts.update(range(first, count-199, 200))
        for start in sorted(starts):
            ident = f'{name}:{start}:200'
            options.append(dict(base_take_name=take, base_seq_name=name, episode_start_motion_idx=start,
                num_frames=200, bootstrap_frames=20, episode_id='expanded_'+hashlib.sha256(ident.encode()).hexdigest()[:16]))
    # Stable random coverage of the whole take, without intersecting absolute time intervals.
    random.Random('expanded192-v1:'+take).shuffle(options)
    for record in options:
        if len(selected) >= cap: break
        a,b = interval(record)
        if all(b <= interval(r)[0] or a >= interval(r)[1] for r in selected):
            selected.append(record)
    if not selected: raise ValueError(f'No clips for {take}')
    return selected


def select_takes(candidates, metadata, official, splits, target):
    original = list(splits['train'])
    excluded = set(splits['dev']+splits['holdout'])
    if target < len(original): raise ValueError('Cannot drop original train takes.')
    allowed = [t for t in sorted(candidates) if t not in excluded and
               official.get(metadata[t]['take_uid']) == 'train']
    if not set(original).issubset(allowed): raise ValueError('Original train take is not eligible official train.')
    groups = {}
    for t in allowed:
        if t in original: continue
        task = metadata[t].get('parent_task_name') or metadata[t].get('task_name') or 'unknown'
        groups.setdefault(task, []).append(t)
    rng = random.Random(62)
    keys = sorted(groups); rng.shuffle(keys)
    for names in groups.values(): rng.shuffle(names)
    chosen = original[:]
    while len(chosen) < target:
        before = len(chosen)
        for key in keys:
            if groups[key] and len(chosen)<target: chosen.append(groups[key].pop())
        if len(chosen)==before: raise ValueError('Insufficient official train takes.')
    return sorted(chosen)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--train-takes', type=int, default=192)
    p.add_argument('--clips-per-take', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True); args.shards.mkdir(parents=True, exist_ok=True)
    print('Loading verified official train source', flush=True)
    old_report = json.loads((ORIGINAL/'report.json').read_text())
    if file_sha256(ORIGINAL/'sequences.pt') != old_report['cache_sha256']: raise ValueError('Original cache changed.')
    old = torch.load(ORIGINAL/'sequences.pt', weights_only=True, map_location='cpu')
    manifest = json.loads(Path(SPLIT).read_text())
    dataset,_ = open_dataset(signal=Path(SIGNAL), spec_sha256=manifest['dataset_spec_sha256'])
    source = dataset.source; candidates,inventory = source.inventory(200)
    official = json.loads(source.split_path.read_text())['take_uid_to_split']
    splits = old['identity']['splits']
    chosen = select_takes(candidates, source.metadata, official, splits, args.train_takes)
    originals = {r['base_take_name']:r for r in dataset.records if r['variant_name']=='clean'}
    episodes = {t:select_clips(source,t,candidates[t],originals[t] if t in splits['train'] else None,args.clips_per_take) for t in chosen}
    plan = {'train_takes':chosen,'dev_takes':splits['dev'],'holdout_takes':splits['holdout'],
            'episodes':episodes,'inventory':inventory,'original_cache_sha256':old_report['cache_sha256'],
            'official_split_sha256':file_sha256(source.split_path), 'metadata_sha256':file_sha256(source.metadata_path),
            'source_spec_sha256':manifest['dataset_spec_sha256'], 'code_sha256':file_sha256(__file__),
            'tasks':{t:source.metadata[t].get('parent_task_name') or source.metadata[t].get('task_name') for t in chosen}}
    plan_path = args.shards/'manifest.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan: raise ValueError('Frozen data manifest changed.')
    atomic_json(plan_path,plan); atomic_json(args.output/'manifest.json',plan)
    plan_hash=file_sha256(plan_path)
    print(f'Frozen manifest: {len(chosen)} train takes, {sum(map(len,episodes.values()))} train clips, 12 fixed dev',flush=True)
    codec=MotionCodec(old['stats']).to(args.device)
    smpl=get_smpl().to(args.device).eval().requires_grad_(False)
    asset=ROOT/'body_models/smplx/SMPLX_NEUTRAL.npz'
    if file_sha256(asset)!=old['identity']['smplx_asset_sha256']: raise ValueError('SMPL asset changed.')
    if file_sha256(CHECKPOINT)!=old['identity']['bootstrap_identity']['e7_checkpoint_sha256']: raise ValueError('E7 checkpoint changed.')
    if file_sha256(BOOTSTRAP)!=old['identity']['bootstrap_cache_sha256']: raise ValueError('Old bootstrap changed.')
    boot = torch.load(BOOTSTRAP,weights_only=True,map_location='cpu')['episodes']
    initializer=None; shards=[]; shard_hashes={}
    for index,take in enumerate(chosen):
        path=args.shards/(take+'.pt'); marker=path.with_suffix('.json')
        if marker.exists():
            meta=json.loads(marker.read_text())
            if meta['manifest_sha256']!=plan_hash or file_sha256(path)!=meta['sha256']: raise ValueError('Shard identity changed.')
            shard=torch.load(path,weights_only=True,map_location='cpu')
        else:
            records=episodes[take]
            if take in splits['train']:
                startup=boot[originals[take]['episode_id']]
            else:
                if initializer is None: initializer=load_e7_initializer(CHECKPOINT,device=args.device,weight_source='ema')
                clean=source.clean(records[0])
                prefix={k:torch.from_numpy(clean[k][:20].copy()).to(args.device) for k in ('aria_traj_obs','img_available','traj_available')}
                feats=source.visual_payload(take,clean['img_source_idx'][:20],clean['img_available'][:20])
                prefix['img_feats']=torch.as_tensor(feats,device=args.device)
                buffer,floor,_=initialize_history(initializer,FlowMatching(),codec,prefix,seed=1062+index)
                startup={'take':take,'beta_boot':buffer.beta_boot.cpu(),'floor_estimate_m':float(floor),
                         'sampling_seed':1062+index,'startup_episode_id':records[0]['episode_id'],
                         'normalized_motion':buffer.bootstrap_motion.cpu()}
            values={k:[] for k in FIELDS}; audits=[]; annotation_hashes=[]
            for j,record in enumerate(records):
                if take in splits['train'] and j==0:
                    i=old['take_names'].index(take)
                    for k in FIELDS: values[k].append(old[k][i])
                    audits.append(old_report['audits'][take]); annotation_hashes.append('original_cache:'+old_report['cache_sha256'])
                else:
                    clean=source.clean(record); supervision=source.supervision(record)
                    audit=prepare_ground_truth(smpl,supervision,list(range(200)))['asset_audit']
                    state=body_states_from_supervision(supervision,torch.from_numpy(clean['aria_traj_obs'].copy()),
                        floor_height=startup['floor_estimate_m'],contact_floor_height=startup['floor_estimate_m'])
                    for k in FIELDS: values[k].append(getattr(state,k).cpu())
                    audits.append(audit); annotation_hashes.append(annotation_digest(supervision))
            shard={'take':take,'records':records,'startup':startup,'audits':audits,'annotation_sha256':annotation_hashes,
                   **{k:torch.stack(v) for k,v in values.items()}}
            save_torch(path,shard); atomic_json(marker,{'manifest_sha256':plan_hash,'sha256':file_sha256(path)})
        shards.append(shard); shard_hashes[take]=json.loads(marker.read_text())['sha256']
        atomic_json(args.output/'progress.json',{'completed_takes':index+1,'total_takes':len(chosen),'updated_at':now()})
        print(f'Audited {index+1}/{len(chosen)} {take}: {len(shard["records"])} clips',flush=True)
    names=[]; ids=[]; betas=[]; values={k:[] for k in FIELDS}
    for shard in shards:
        for j,r in enumerate(shard['records']):
            names.append(shard['take']); ids.append(r['episode_id']); betas.append(shard['startup']['beta_boot'])
            for k in FIELDS: values[k].append(shard[k][j])
    # Original cached order is the exact evaluation order.
    for i,take in enumerate(old['take_names']):
        if take not in splits['dev']: continue
        names.append(take); ids.append(originals[take]['episode_id']);betas.append(old['beta_boot'][i])
        for k in FIELDS: values[k].append(old[k][i])
    identity={**old['identity'],'scope':SCOPE,'splits':{**splits,'train':chosen},'original_data':str(ORIGINAL),
              'original_cache_sha256':old_report['cache_sha256'],'expanded_manifest_sha256':plan_hash,
              'expanded_shard_sha256':shard_hashes,'preparation_code_sha256':file_sha256(__file__),
              'bootstrap_policy':'original startup retained; new take frozen E7 EMA 20-frame startup, shared across its clips'}
    save_torch(args.output/'sequences.pt',{'identity':identity,'stats':old['stats'],'take_names':names,
        'episode_ids':ids,'beta_boot':torch.stack(betas),**{k:torch.stack(v) for k,v in values.items()}})
    report={'completed':True,'identity':identity,'sequences':len(names),'train_takes':len(chosen),'train_clips':len(names)-12,
            'unique_train_minutes':(len(names)-12)/3,'cache_sha256':file_sha256(args.output/'sequences.pt'),
            'audit_shards':str(args.shards),'holdout_used':False}
    atomic_json(args.output/'report.json',report)
    ExpandedPriorSequences(args.output,'train');ExpandedPriorSequences(args.output,'dev')
    append_log('P D组扩量数据审计完成',[f'{len(chosen)}个train take、{len(names)-12}个不重叠20秒片段；原12个dev逐张量一致；holdout未使用。',
        f'数据报告：`{args.output / "report.json"}`；冻结片段清单：`{args.output / "manifest.json"}`。'])

if __name__=='__main__': main()
