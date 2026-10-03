import torch
import pytest
from egorecover.codec import MotionCodec,BodyState,transform_to_9d
from egorecover.prior import HistoryPrior
from egorecover.prior_two_forward import physical_batch,FIELDS
from egorecover.prior_cv_residual import predict
from egorecover.prior_real_history import window_conditions,decode_window,input_batch,noise_seed
from run.prior_real_history_experiment import score_cell
from test_prior_two_forward import stats,trajectory


def observations(n=2):
    state=trajectory(n=n)
    return {'aria_traj_obs':transform_to_9d(state.reference),'img_feats':torch.randn(n,200,1024),
            'traj_available':torch.ones(n,200,dtype=torch.bool),'img_available':torch.ones(n,200,dtype=torch.bool)}


def test_window_excludes_future_and_has_80_frame_limit():
    codec=MotionCodec(stats());obs=observations();floor=torch.zeros(2)
    for frame in (20,79,80,150):
        y,anchor,bounds=window_conditions(codec,obs,floor,frame)
        changed={k:v.clone() for k,v in obs.items()}
        changed['aria_traj_obs'][:,frame+1:,6:]+=1000
        changed['img_feats'][:,frame+1:]=float('nan')
        z,b,_=window_conditions(codec,changed,floor,frame)
        for k in y:torch.testing.assert_close(y[k],z[k],atol=0,rtol=0)
        torch.testing.assert_close(anchor,b,atol=0,rtol=0)
        assert bounds==(max(0,frame-79),frame+1)
        assert y['traj'].shape[1]==min(80,frame+1)


def test_decode_window_world_anchor_and_sequential_deltas():
    codec=MotionCodec(stats());state=trajectory(n=2,frames=80)
    anchor=state.reference[:,0].clone();anchor[:,0,3]-=7
    codes=codec.encode_history(state,anchor)
    result=decode_window(codec,codes,anchor)
    for k in FIELDS:torch.testing.assert_close(getattr(result,k),getattr(state,k)[:,-1],atol=1e-5,rtol=1e-5)


def test_inference_batch_matches_original_without_targets():
    codec=MotionCodec(stats());state=trajectory(n=2,frames=21)
    history=BodyState(*(getattr(state,k)[:,:20] for k in FIELDS));target=BodyState(*(getattr(state,k)[:,20] for k in FIELDS))
    batch=input_batch(codec,history);old=physical_batch(codec,history,target,torch.zeros(2,10))
    assert set(batch)=={'history_motion','base_mu','velocity_mu','previous_reference'}
    for k in batch:torch.testing.assert_close(batch[k],old[k],atol=0,rtol=0)
    p=HistoryPrior(width=16,layers=1,heads=4).eval();p.base_mode='constant_velocity'
    torch.testing.assert_close(predict(p,batch),predict(p,old),atol=0,rtol=0)


def test_noise_independent_of_variant_sharding_and_future():
    assert noise_seed('a',1062,35)==noise_seed('a',1062,35)
    assert len({noise_seed(t,d,f) for t in ['a','b'] for d in [1062,1063] for f in [35,36]})==8


def test_take_pairing_does_not_count_draws_as_takes():
    cases=[['a','clean',1],['a','clean',2],['b','clean',1],['b','clean',2]]
    gt=torch.ones(3,4,5)*10;pred=gt.clone();pred[:,:2]+=2;pred[:,2:]+=6
    s=score_cell(pred,gt,cases,torch.ones(4,5,dtype=torch.bool))
    assert s['difference']==4 and s['difference_by_seed']==[4,4,4]
    assert set(s['per_take_difference_by_seed'])=={'a','b'}
    assert s['take_bootstrap_95ci']==[2.,6.]


def test_worker_offline_metrics_and_partial_history_resume(tmp_path,monkeypatch):
    import json,sys
    from run import evaluate_prior_real_history as worker
    from run.adapt_prior_on_predictions import save_torch
    from egorecover.evaluation_protocol import file_sha256
    from test_smplx_evaluation import SyntheticLayer
    codec=MotionCodec(stats());body=trajectory(n=1)
    startup_codes=codec.encode_history(BodyState(*(getattr(body,k)[:,:20] for k in FIELDS)),body.reference[:,0])[0]
    startup={'normalized_motion':startup_codes,'initial_reference':body.reference[0,0],
             'beta_boot':torch.zeros(10),'floor_estimate_m':0.}
    obs={k:v[0] for k,v in observations(n=1).items()}
    record={'base_take_name':'train','variant_name':'clean'}
    pack={'take':'train','variant':'clean','record':record,'observations':obs,'startup':startup,
          'truth':{k:getattr(body,k)[0] for k in FIELDS}}
    asset=tmp_path/'asset';asset.mkdir();(asset/'SMPLX_NEUTRAL.npz').write_bytes(b'asset')
    monkeypatch.setenv('SMPLX_MODEL_PATH',str(asset));monkeypatch.setattr(worker,'get_smpl',SyntheticLayer)
    torch.manual_seed(62);prior=HistoryPrior(width=16,layers=1,heads=4)
    ckpt=tmp_path/'p.pt';torch.save({'state_dict':prior.state_dict(),'base_mode':'constant_velocity','selected_step':1,'smplx_asset_sha256':file_sha256(asset/'SMPLX_NEUTRAL.npz')},ckpt)
    plan={'generator':{'draw_seeds':[1062]},'predictors':[{'name':'new_P_s62','checkpoint':str(ckpt),'checkpoint_sha256':file_sha256(ckpt),'selected_step':1,'training_seed':62}]}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));data=tmp_path/'data';data.mkdir()
    payload={'identity':{'plan_sha256':file_sha256(path),'original_identity':{'smplx_asset_sha256':file_sha256(asset/'SMPLX_NEUTRAL.npz')}},'stats':stats(),'packs':{'smoke':[pack]}}
    save_torch(data/'packets.pt',payload);(data/'report.json').write_text(json.dumps({'completed':True,'cache_sha256':file_sha256(data/'packets.pt')}))
    class E7(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('code',startup_codes[0])
        def forward(self,x,t,y):return self.code.expand_as(x)
    monkeypatch.setattr(worker,'original_e7',lambda *a:E7().eval())
    monkeypatch.setattr(worker,'HistoryPrior',lambda:HistoryPrior(width=16,layers=1,heads=4))
    out=tmp_path/'output';work=tmp_path/'work'
    argv=['worker','--plan',str(path),'--data',str(data),'--output',str(out),'--work',str(work),'--device','cpu','--smoke']
    monkeypatch.setattr(sys,'argv',argv);worker.main()
    expected=torch.load(out/'frames.pt',weights_only=True)
    assert expected['metrics']['62']['fk22_mm'].shape==(1,65)
    assert expected['history_quality']['last_velocity_error_mmps'].shape==(1,65)
    assert expected['history_quality']['last_angular_velocity_error_degps'].shape==(1,65)
    saved=torch.load(work/'history_resume.pt',weights_only=True)
    work2=tmp_path/'resumed_work';work2.mkdir()
    saved['next_frame']=40
    for k in FIELDS:saved[k]=saved[k][:,:40]
    save_torch(work2/'history_resume.pt',saved)
    out2=tmp_path/'resumed_output'
    monkeypatch.setattr(sys,'argv',[str(out2) if x==str(out) else str(work2) if x==str(work) else x for x in argv]);worker.main()
    actual=torch.load(out2/'frames.pt',weights_only=True)
    for k in actual['metrics']['62']:torch.testing.assert_close(actual['metrics']['62'][k],expected['metrics']['62'][k],atol=0,rtol=0)
