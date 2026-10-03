import copy

import pytest
import torch

from config.defaults import get_cfg_defaults
from egorecover.codec import BodyState, MotionCodec, rigid_transform
from egorecover.conditioning import prepare_conditioning
from egorecover.g_pretraining import mixture, native_states, sensor_conditions
from egorecover.history_flow import HistoryFlow
from egorecover.prior import HistoryPrior
from egorecover.prior_two_forward import physical_batch
from model.history_uniegomotion import HistoryUniEgoMotion


@pytest.fixture
def codec():
    return MotionCodec({k:torch.zeros(n) if k.endswith('mean') else torch.ones(n)
                        for k,n in [('motion_mean',243),('motion_std',243),('traj_mean',18),('traj_std',18)]})


def states(batch=2,length=20):
    r=torch.eye(3).expand(batch,length,22,3,3).clone()
    p=torch.zeros(batch,length,22,3);p[...,2]=1.
    ref=torch.eye(4).expand(batch,length,4,4).clone()
    ref[...,0,3]=torch.arange(length)*.01
    p[...,0]+=ref[...,0,3,None]
    return BodyState(rigid_transform(r,p),ref,torch.zeros(batch,length,36))


def conditions(codec):
    history=states();head=torch.zeros(2,21,9);head[...,0]=1;head[...,4]=1;head[...,8]=1.5
    images=torch.randn(2,21,1024)
    prior=HistoryPrior(width=32,layers=1,heads=4,dropout=0).freeze();prior.base_mode='constant_velocity'
    return sensor_conditions(codec,history,head,images,prior)


def test_mixture_warmup_and_ramp():
    assert [mixture(x) for x in (1,2000,4000,6000,24000)]==[0.,0.,.25,.5,.5]


def test_target_reencoded_against_actual_history(codec):
    gt=states();pred=states()
    pred.reference[:,-1,0,3]+=2
    target=BodyState(*(getattr(gt,k)[:,-1].clone() for k in ('joints','reference','auxiliary')))
    a=physical_batch(codec,gt,target,torch.zeros(2,10))
    b=physical_batch(codec,pred,target,torch.zeros(2,10))
    assert torch.equal(a['target_joints'],b['target_joints'])
    assert not torch.equal(a['target'],b['target'])
    decoded=codec.decode_current(b['target'][:,0],b['previous_reference'])
    torch.testing.assert_close(decoded.joints,target.joints)


def test_past_masks_isolate_nan_and_require_complete_fields(codec):
    y=conditions(codec)
    y['past_img_mask'][:]=True;y['past_img_embs'][:]=float('nan')
    clean=prepare_conditioning(y)
    assert torch.equal(clean['past_img_embs'],torch.zeros_like(clean['past_img_embs']))
    del y['past_traj']
    with pytest.raises(ValueError,match='together'):prepare_conditioning(y)


def test_sensor_boundary_rejects_future_length(codec):
    p=HistoryPrior(width=32,layers=1,heads=4).freeze()
    with pytest.raises(ValueError,match='exactly'):
        sensor_conditions(codec,states(),torch.zeros(2,22,9),torch.zeros(2,22,1024),p)


def test_new_g_uses_past_images_and_backpropagates(codec):
    torch.set_num_threads(2);torch.manual_seed(62)
    model=HistoryUniEgoMotion(get_cfg_defaults(),dropout=0,past_observations=True)
    # Two genuine decoder blocks exercise the new path without duplicating a full backbone test.
    model.tsfm=torch.nn.ModuleList(list(model.tsfm)[:2]);model.eval()
    y=conditions(codec);x=torch.randn(2,1,243);t=torch.tensor([.3,.7])
    first=model(x,t,y)
    changed={**y,'past_img_embs':y['past_img_embs']+1}
    second=model(x,t,changed)
    assert not torch.allclose(first,second)
    first.square().mean().backward()
    assert model.embed_clip_cond.weight.grad.abs().sum()>0
    model.past_observations=False
    with pytest.raises(ValueError,match='opt-in'):model(x,t,y)


def test_source_pair_has_identical_noise_and_conditions(codec):
    y=conditions(codec);noise=torch.randn_like(y['prior_mu'])
    gaussian=HistoryFlow(source_mode='gaussian').build_source(y['prior_mu'],epsilon=noise)
    history=HistoryFlow(source_mode='history').build_source(y['prior_mu'],epsilon=noise)
    torch.testing.assert_close(history-gaussian,y['prior_mu'])


def test_native_decode_keeps_vertical_reference_delta(codec):
    state=states(batch=1,length=2)
    raw=codec.denormalize(codec.encode_current(state,state.reference))[0]
    raw[:,206]=.2  # z component of the full native SE(3) delta
    result=native_states(codec,codec.normalize(raw),torch.eye(4))
    torch.testing.assert_close(result.joints[:,0,2,3],torch.tensor([1.2,1.4]))
    assert torch.equal(result.reference[:,2,3],torch.zeros(2))


def test_score_summary_excludes_frame_arrays_and_weights_takes():
    from run.train_g_pretraining import summarize_rows
    rows=[{'take':t,'body_mm':v,'frame_body_mm':torch.tensor([v,v])}
          for t,v in [('a',10.),('a',30.),('b',40.)]]
    result=summarize_rows(rows)
    assert set(result)=={'body_mm'}
    assert result['body_mm']=={'mean':30.,'per_take':{'a':20.,'b':40.}}


def test_completed_factorial_report_uses_paired_effects(tmp_path,monkeypatch):
    import json
    from run import g_pretraining_experiment as queue
    from run.e7_ablation import save,write_json
    calls=[];monkeypatch.setattr(queue,'log',lambda title,lines:calls.append(lines))
    cfg={'groups':['T0','T1','T2','T3'],'seeds':[62,63,64],'draws':[1062,1063,1064],'nfe':[10,1,3,5]}
    levels={'T0':100.,'T1':95.,'T2':98.,'T3':91.}
    for g,value in levels.items():
        for seed in cfg['seeds']:
            directory=tmp_path/g/f'seed{seed}'/'test'
            payload={'summary':{'scores':{'body_mm':{'mean':value,'per_take':{'a':value-1,'b':value+1}},
                'p_to_g_improvement_mm':{'mean':2.}}}}
            for draw in cfg['draws']:
                save(directory/f'last_nfe10_draw{draw}.pt',payload)
                for nfe in cfg['nfe']:save(directory/f'best_nfe{nfe}_draw{draw}.pt',payload)
            save(directory/'auxiliary.pt',{'rows':[{'take':t,'history':kind,'horizon':h,'body_mm':value}
                for t in ('a','b') for kind,h in [('GT',1),('teacher',1),('teacher',4),('teacher',8)]]})
    write_json(tmp_path/'evaluation/e7_baseline.json',{'scores':{'common_bootstrap':{'mean':110.},'native_window':{'mean':108.}}})
    # Reporting should support both absolute production paths and test artifacts.
    monkeypatch.setattr(queue,'ROOT',tmp_path.parent)
    queue.summarize(tmp_path,cfg)
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['paired_differences']['T1-T0']['mean']==-5
    assert report['paired_differences']['T3-T2']['mean']==-7
    assert report['interaction_mm']==-2
    assert calls and '支持降低误差' in (tmp_path/'RESULTS.md').read_text()
