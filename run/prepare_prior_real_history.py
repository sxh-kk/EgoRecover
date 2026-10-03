"""Materialize audited online observations once; original GT is evaluation-only."""
import argparse,json
from pathlib import Path
import torch
from egorecover.data import open_dataset
from egorecover.bootstrap_shapes import ModelBootstrapShapes
from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from egorecover.prior_two_forward import FIELDS
from run.adapt_prior_on_predictions import save_torch
from run.complete_stages import SIGNAL,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    plan=json.loads(a.plan.read_text());split=json.loads(Path(plan['split_manifest']).read_text())
    print('Loading official source once to materialize 12 dev + 2 train smoke takes',flush=True)
    dataset,_=open_dataset(signal=Path(SIGNAL),spec_sha256=split['dataset_spec_sha256'])
    original_dir=Path('exp/egorecover_prior_two_forward/v1/data');report=json.loads((original_dir/'report.json').read_text())
    if file_sha256(original_dir/'sequences.pt')!=report['cache_sha256']:raise ValueError('Original data changed.')
    original=torch.load(original_dir/'sequences.pt',weights_only=True,map_location='cpu')
    if file_sha256(plan['startup']['cache'])!=plan['startup']['cache_sha256']:raise ValueError('Startup changed.')
    stats=dataset.source.root/'uniegomotion/v4_beta_ee_train_stats.pt'
    boots=ModelBootstrapShapes(plan['startup']['cache'],allowed_takes=sum(split['splits'].values(),[]),
        stats_sha256=file_sha256(stats),split_manifest_sha256=file_sha256(plan['split_manifest']),dataset_spec_sha256=split['dataset_spec_sha256'])
    if boots.identity['e7_checkpoint_sha256']!=plan['generator']['checkpoint_declared_sha256']:raise ValueError('E7 identity differs.')
    if set(plan['takes'])!=set(split['splits']['dev']):raise ValueError('Dev split changed.')
    groups={'dev':plan['takes'],'smoke':sorted(split['splits']['train'])[:2]};packs={}
    for group,names in groups.items():
        packs[group]=[]
        for take in names:
            idx=original['take_names'].index(take)
            records={r['variant_name']:r for r in dataset.records if r['base_take_name']==take}
            clean=records['clean'];startup=boots.startup_for_record(clean)
            if not torch.equal(startup['beta_boot'],original['beta_boot'][idx]):raise ValueError('Startup beta changed.')
            truth={k:original[k][idx].clone() for k in FIELDS}
            variants=plan['variants'] if group=='dev' else ['clean']
            for variant in variants:
                r=records[variant]
                if r['num_frames']!=200 or r['bootstrap_frames']!=20:raise ValueError('Episode changed.')
                dataset.arrays(r['variant_id'],verify_hash=True)
                obs=dataset.observations(r['variant_id'],as_of=199,history_frames=200)
                packs[group].append({'take':take,'variant':variant,'record':r,'observations':obs,
                                     'startup':startup,'truth':truth})
                print('Prepared',group,take,variant,flush=True)
    identity={'plan_sha256':file_sha256(a.plan),'original_cache_sha256':report['cache_sha256'],
              'original_identity':original['identity'],'preparation_code_sha256':file_sha256(__file__)}
    save_torch(a.output/'packets.pt',{'identity':identity,'stats':original['stats'],'packs':packs})
    atomic_json(a.output/'report.json',{'completed':True,'identity':identity,'cache_sha256':file_sha256(a.output/'packets.pt'),
                'dev_takes':12,'smoke_train_takes':2,'updated_at':now()})
if __name__=='__main__':main()
