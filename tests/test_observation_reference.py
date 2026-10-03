import pytest
import torch

from config.defaults import get_cfg_defaults
from egorecover.codec import MotionCodec, BodyState, rigid_transform
from egorecover.prior import HistoryPrior
from egorecover.observation_reference import (gather_observations, conditions,
    encode_in_reference, decode, target_batch, objective)
from model.observation_reference_g import ObservationReferenceG


@pytest.fixture
def inputs():
    torch.set_num_threads(2)
    codec=MotionCodec({k:torch.zeros(n) if k.endswith('mean') else torch.ones(n)
        for k,n in [('motion_mean',243),('motion_std',243),('traj_mean',18),('traj_std',18)]})
    ref=torch.eye(4).expand(1,20,4,4).clone();ref[...,0,3]=.5
    joints=ref[:,:,None].expand(1,20,22,4,4).clone();joints[...,2,3]=1
    h=BodyState(joints,ref,torch.zeros(1,20,36))
    head=torch.zeros(1,200,9);head[...,0]=head[...,4]=1;head[...,8]=1.6
    head[...,6]=torch.arange(200)*.01
    prior=HistoryPrior(width=32,layers=1,heads=4,dropout=0).freeze();prior.base_mode='constant_velocity'
    return codec,h,head,torch.randn(1,200,1024),prior


def test_future_and_padding_are_never_observed(inputs):
    _,_,head,im,_=inputs
    a=gather_observations(head,im,torch.tensor([20]),80)
    head[:,21:]=float('nan');im[:,21:]=float('nan')
    b=gather_observations(head,im,torch.tensor([20]),80)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    assert a[2].sum()==21 and not a[2][0,:59].any()


def test_observed_reference_preserves_world_body_and_device_offset(inputs):
    codec,h,head,im,p=inputs
    oh,oi,v=gather_observations(head,im,torch.tensor([40]),21)
    y,ref,pstate=conditions(codec,h,oh,oi,v,p,True)
    restored=decode(codec,y['prior_mu'][:,0],ref,True)
    torch.testing.assert_close(restored.joints,pstate.joints,atol=1e-6,rtol=1e-6)
    assert not torch.allclose(restored.joints[:,15,:3,3],oh[:,-1,6:9])
    # Ignoring the old predicted reference delta is intentional and explicit.
    changed=y['prior_mu'][:,0].clone();changed[:,198:207]=float('nan')
    torch.testing.assert_close(decode(codec,changed,ref,True).joints,pstate.joints)
    target=BodyState(h.joints[:,-1],h.reference[:,-1],h.auxiliary[:,-1])
    batch=target_batch(codec,target,ref,True)
    assert objective(codec,batch['target'],batch,True).item()<1e-10


def test_shared_interface_padding_equivalence_and_real_past_use(inputs):
    codec,h,head,im,p=inputs
    short=ObservationReferenceG(get_cfg_defaults(),observation_length=21,dropout=0).eval()
    short.tsfm=torch.nn.ModuleList(list(short.tsfm)[:2])
    long=ObservationReferenceG(get_cfg_defaults(),observation_length=80,dropout=0).eval()
    long.tsfm=torch.nn.ModuleList(list(long.tsfm)[:2]);long.load_state_dict(short.state_dict())
    x=torch.randn(1,1,243);t=torch.tensor([.5])
    ys=[]
    for n in (21,80):
        obs=gather_observations(head,im,torch.tensor([20]),n)
        ys.append(conditions(codec,h,*obs,p,False)[0])
    with torch.no_grad():
        a=short(x,t,ys[0]);b=long(x,t,ys[1])
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    obs=gather_observations(head,im,torch.tensor([90]),80)
    y=conditions(codec,h,*obs,p,False)[0]
    changed={**y,'memory_img':y['memory_img'].clone()};changed['memory_img'][:,0]+=2
    assert not torch.allclose(long(x,t,y),long(x,t,changed))
    long(x,t,y).square().mean().backward()
    assert long.embed_traj_cond.weight.grad.abs().sum()>0


def test_anchored_output_has_no_learned_reference_delta(inputs):
    codec,h,head,im,p=inputs
    obs=gather_observations(head,im,torch.tensor([40]),21)
    y,ref,_=conditions(codec,h,*obs,p,True)
    model=ObservationReferenceG(get_cfg_defaults(),anchored=True,dropout=0).eval()
    model.tsfm=torch.nn.ModuleList(list(model.tsfm)[:1])
    code=model(torch.randn(1,1,243),torch.tensor([.5]),y)
    torch.testing.assert_close(code[...,198:207],y['prior_mu'][...,198:207])
    assert torch.isfinite(decode(codec,code[:,0],ref,True).joints).all()


def test_short_initial_interface_matches_previous_g(inputs):
    from model.history_uniegomotion import HistoryUniEgoMotion
    from egorecover.g_pretraining import sensor_conditions
    codec,h,head,im,p=inputs
    old=HistoryUniEgoMotion(get_cfg_defaults(),dropout=0,past_observations=True).eval()
    new=ObservationReferenceG(get_cfg_defaults(),dropout=0).eval()
    old.tsfm=torch.nn.ModuleList(list(old.tsfm)[:2])
    new.tsfm=torch.nn.ModuleList(list(new.tsfm)[:2])
    new.load_state_dict(old.state_dict(),strict=False)
    obs=gather_observations(head,im,torch.tensor([90]),21)
    yo=sensor_conditions(codec,h,obs[0],obs[1],p)
    yn=conditions(codec,h,*obs,p,False)[0]
    torch.testing.assert_close(yn['memory_traj'],torch.cat((yo['past_traj'],yo['traj']),1))
    x=torch.randn(1,1,243);t=torch.tensor([.5])
    torch.testing.assert_close(new(x,t,yn),old(x,t,yo),atol=2e-6,rtol=2e-6)
    # Additional trajectory information can begin learning from the first step.
    new(x,t,yn).square().mean().backward()
    assert new.memory_traj_adapter.weight.grad.abs().sum()>0


def test_older_observation_deltas_are_incremental(inputs):
    codec,h,head,im,p=inputs
    obs=gather_observations(head,im,torch.tensor([90]),80)
    y=conditions(codec,h,*obs,p,False)[0]
    raw=codec.denormalize(y['memory_traj'],'traj')
    torch.testing.assert_close(raw[:,1:-21,15],torch.full_like(raw[:,1:-21,15],.01),atol=1e-6,rtol=1e-6)
