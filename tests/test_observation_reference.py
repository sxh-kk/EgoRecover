import pytest
import torch

from config.defaults import get_cfg_defaults
from egorecover.codec import MotionCodec, BodyState, rigid_transform
from egorecover.prior import HistoryPrior
from egorecover.observation_reference import (gather_observations, conditions,
    encode_in_reference, decode, target_batch, objective, apply_action, prepare, ObservationFlow)
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


@pytest.mark.parametrize('action,kinds', [('a10', ('traj',)), ('a01', ('img',)), ('a00', ('img', 'traj'))])
def test_current_action_reaches_all_model_paths_and_isolates_masked_nan(inputs, action, kinds):
    codec, h, head, im, p = inputs
    y = conditions(codec, h, *gather_observations(head, im, torch.tensor([40]), 21), p, False)[0]
    original = {k: v.clone() for k, v in y.items()}
    candidate = apply_action(y, action)
    model = ObservationReferenceG(get_cfg_defaults(), dropout=0).eval()
    model.tsfm = torch.nn.ModuleList(list(model.tsfm)[:1])
    # Exercise the additional trajectory path after it has learned nonzero weights.
    with torch.no_grad():
        model.memory_traj_adapter.weight.normal_(std=.01)
    x, t = torch.randn(1, 1, 243), torch.tensor([.5])
    with torch.no_grad():
        expected = model(x, t, candidate)
        assert not torch.allclose(expected, model(x, t, y))
        # Existing callers may set only the current masks. They must reach memory too.
        direct = {**y, 'img_mask': candidate['img_mask'], 'traj_mask': candidate['traj_mask']}
        torch.testing.assert_close(model(x, t, direct), expected, atol=0, rtol=0)
    poisoned = dict(candidate)
    for kind in kinds:
        key = f'memory_{kind}'
        value = candidate[key].clone()
        value[:, -1] = float('nan')
        poisoned[key] = value.requires_grad_()
        current = 'img_embs' if kind == 'img' else 'traj'
        poisoned[current] = torch.full_like(candidate[current], float('nan'))
        assert not candidate[f'memory_{kind}_mask'][:, :-1].any()
    result = model(x, t, poisoned)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    result.square().mean().backward()
    for kind in kinds:
        grad = poisoned[f'memory_{kind}'].grad
        assert torch.isfinite(grad).all()
        assert not grad[:, -1].any()
    assert all(torch.equal(y[k], original[k]) for k in y)
    assert candidate['valid_frames'].all() and candidate['memory_valid'].all()


def test_window_action_hides_raw_sensors_before_geometry(inputs):
    codec, h, head, im, p = inputs
    oh, oi, valid = gather_observations(head, im, torch.tensor([90]), 80)
    expected = conditions(codec, h, oh, oi, valid, p, False, action='a00', action_scope='window')[0]
    result, ref, _ = conditions(codec, h, torch.full_like(oh, float('nan')),
        torch.full_like(oi, float('nan')), valid, p, False, action='a00', action_scope='window')
    assert all(torch.equal(result[k], expected[k]) for k in result)
    torch.testing.assert_close(ref, h.reference[:, -1])
    assert result['memory_img_mask'].all() and result['memory_traj_mask'].all()
    assert result['valid_frames'].all() and result['memory_valid'].all()


def test_real_missingness_preserves_physical_time_and_partial_masks_fail(inputs):
    codec, h, head, im, p = inputs
    oh, oi, valid = gather_observations(head, im, torch.tensor([40]), 21)
    available = torch.ones_like(valid); available[:, -1] = False
    oh[:, -1] = float('nan'); oi[:, -1] = float('nan')
    y = conditions(codec, h, oh, oi, valid, p, False,
                   img_available=available, traj_available=available)[0]
    assert y['img_mask'].all() and y['traj_mask'].all()
    assert y['valid_frames'].all() and y['memory_valid'].all()
    assert not y['memory_img'][:, -1].any() and not y['memory_traj'][:, -1].any()
    partial = {k: v for k, v in y.items() if k != 'memory_img_mask'}
    for check in (prepare, lambda value: apply_action(value, 'a11')):
        with pytest.raises(ValueError, match='supplied together'):
            check(partial)


def test_utility_labels_use_scope_availability_shared_noise_and_actual_nfe(inputs):
    from egorecover.utility import generate_utility_labels
    from mydiffusion.flow_matching import FlowMatching

    codec, h, head, im, p = inputs
    oh, oi, valid = gather_observations(head, im, torch.tensor([40]), 21)
    available = torch.ones_like(valid); available[:, -1] = False
    y = conditions(codec, h, oh, oi, valid, p, False,
                   img_available=available, traj_available=available)[0]
    original = {k: v.clone() for k, v in y.items()}

    class MemoryActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sources = []
            self.calls = 0

        def forward(self, x, t, y):
            self.calls += 1
            if bool((t == 1).all()):
                self.sources.append(x.clone())
            image = (~y['memory_img_mask'] & y['memory_valid']).any(1)
            trajectory = (~y['memory_traj_mask'] & y['memory_valid']).any(1)
            return (2 * image.float() + 3 * trajectory.float())[:, None, None].expand_as(x)

    epsilon = torch.randn(1, 1, 243)
    for scope in ('current', 'window'):
        model = MemoryActionModel().eval()
        flow = ObservationFlow(action_scope=scope); flow.flow = FlowMatching(num_steps=3)
        labels = generate_utility_labels(model, flow, y, epsilon,
            lambda value: (value - 2).abs().mean((1, 2)), checkpoint_id='synthetic-only', num_steps=2)
        assert model.calls == 8 and len(model.sources) == 4
        assert all(torch.equal(value, epsilon) for value in model.sources)
        assert labels['identity']['num_steps'] == 2
        assert labels['identity']['action_scope'] == scope
        assert labels['identity']['conditioning_protocol'] == 'observation-memory-masks-v2'
        canonical = [0, 0, 0, 0] if scope == 'current' else [0, 1, 2, 3]
        assert labels['canonical_actions'].tolist() == [canonical]
        errors = [[3., 3., 3., 3.]] if scope == 'current' else [[3., 0., 1., 2.]]
        torch.testing.assert_close(labels['errors'], torch.tensor(errors), atol=1e-6, rtol=0)
    assert all(torch.equal(y[k], original[k]) for k in y)


def test_observed_reference_cannot_claim_to_mask_its_current_anchor(inputs):
    from egorecover.utility import generate_utility_labels

    codec, h, head, im, p = inputs
    obs = gather_observations(head, im, torch.tensor([40]), 21)
    with pytest.raises(ValueError, match='current trajectory'):
        conditions(codec, h, *obs, p, True, action='a10')
    y = conditions(codec, h, *obs, p, True)[0]
    model = ObservationReferenceG(get_cfg_defaults(), anchored=True, dropout=0).eval()
    model.tsfm = torch.nn.ModuleList(list(model.tsfm)[:1])
    with pytest.raises(ValueError, match='current trajectory'):
        model(torch.randn(1, 1, 243), torch.tensor([.5]), apply_action(y, 'a10'))
    with pytest.raises(ValueError, match='prediction-defined reference'):
        generate_utility_labels(model, ObservationFlow(), y, torch.randn(1, 1, 243),
            lambda value: value.mean((1, 2)), checkpoint_id='synthetic-only')
