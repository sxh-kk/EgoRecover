"""Frozen native E7 causal histories, then new-P-only next-frame evaluation."""
import argparse,json,time,os
from pathlib import Path
import torch
from config.defaults import get_cfg_defaults,validate_e7
from model.uniegomotion import UniEgoMotion
from module.ema import apply_ema_weights_from_checkpoint
from mydiffusion.flow_matching import FlowMatching
from dataset.smpl_utils import get_smpl
from egorecover.codec import BodyState,MotionCodec
from egorecover.evaluation_protocol import file_sha256
from egorecover.evaluation_resume import atomic_json
from egorecover.history import HistoryBuffer
from egorecover.prior import HistoryPrior
from egorecover.prior_cv_residual import predict,measure,rotation_angle
from egorecover.prior_two_forward import FIELDS
from egorecover.prior_real_history import noise_seed,window_conditions,decode_window,stack_states,input_batch
from run.adapt_prior_on_predictions import save_torch
from run.complete_stages import now


def original_e7(plan,device):
    spec=plan['generator'];path=Path(spec['checkpoint'])
    if file_sha256(path)!=spec['checkpoint_declared_sha256']:raise ValueError('E7 checkpoint changed.')
    cfg=get_cfg_defaults();cfg.merge_from_file(spec['configuration']);validate_e7(cfg)
    model=UniEgoMotion(cfg).to(device).eval()
    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
    state=checkpoint.get('state_dict',checkpoint)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True)
    if not apply_ema_weights_from_checkpoint(model,checkpoint):raise ValueError('E7 EMA missing.')
    return model.requires_grad_(False)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);started=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=True);a.work.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());dr=json.loads((a.data/'report.json').read_text())
    if not dr['completed'] or dr['cache_sha256']!=file_sha256(a.data/'packets.pt'):raise ValueError('Data cache changed.')
    payload=torch.load(a.data/'packets.pt',weights_only=True,map_location='cpu')
    if payload['identity']['plan_sha256']!=file_sha256(a.plan):raise ValueError('Plan/cache mismatch.')
    device=torch.device(a.device);codec=MotionCodec(payload['stats']).to(device)
    originals=payload['packs']['smoke' if a.smoke else 'dev']
    takes=sorted({r['take'] for r in originals});takes=takes if a.smoke else takes[a.shard::a.shards]
    if not takes:raise ValueError('Empty shard.')
    draws=plan['generator']['draw_seeds'][:1] if a.smoke else plan['generator']['draw_seeds']
    cases=[{**r,'draw':draw} for r in originals if r['take'] in takes for draw in draws]
    keys=[(r['take'],r['variant'],r['draw']) for r in cases]
    end=85 if a.smoke else 200;n=len(cases)
    identity={'plan_sha256':file_sha256(a.plan),'data_sha256':dr['cache_sha256'],'cases':keys,'end_frame':end,
       'code_sha256':{str(q):file_sha256(q) for q in [Path(__file__),Path('egorecover/prior_real_history.py'),Path('egorecover/codec.py'),Path('egorecover/prior_cv_residual.py'),Path('mydiffusion/flow_matching.py')]}}
    # JSON round trip removes tuple/list differences in reports and resume checks.
    identity=json.loads(json.dumps(identity));atomic_json(a.output/'identity.json',identity)
    obs={k:torch.stack([r['observations'][k] for r in cases]).to(device) for k in cases[0]['observations']}
    floors=torch.tensor([r['startup']['floor_estimate_m'] for r in cases],device=device)
    betas=torch.stack([r['startup']['beta_boot'] for r in cases]).to(device)
    cp=a.work/'history_resume.pt';hp=a.work/'histories.pt';hm=a.work/'histories.json'
    if hm.exists():
        meta=json.loads(hm.read_text())
        if meta['identity']!=identity or file_sha256(hp)!=meta['sha256']:raise ValueError('Saved history identity changed.')
        h=torch.load(hp,weights_only=True,map_location='cpu');states=BodyState(*(h[k].to(device) for k in FIELDS))
        print('Reusing completed immutable E7 histories',flush=True)
    else:
        if cp.exists():
            resume=torch.load(cp,weights_only=True,map_location='cpu')
            if resume['identity']!=identity:raise ValueError('Resume history identity changed.')
            committed=[BodyState(*(resume[k][:,t].to(device) for k in FIELDS)) for t in range(resume['next_frame'])]
        else:
            codes=torch.stack([r['startup']['normalized_motion'] for r in cases]).to(device)
            previous=torch.stack([r['startup']['initial_reference'] for r in cases]).to(device)
            committed=[]
            for t in range(20):
                state=codec.decode_current(codes[:,t],previous);committed.append(state);previous=state.reference
            if not torch.allclose(torch.stack([x.auxiliary[...,-10:] for x in committed]).mean(0),betas,atol=1e-5,rtol=0):
                raise ValueError('Startup decoded beta differs.')
        print(f'Loading original E7 EMA: {n} histories, frames {len(committed)}..{end-1}',flush=True)
        model=original_e7(plan,device);flow=FlowMatching()
        for s in range(len(committed),end):
            y,anchor,bounds=window_conditions(codec,obs,floors,s)
            noise=torch.stack([torch.randn(s-bounds[0]+1,243,generator=torch.Generator().manual_seed(noise_seed(r['take'],r['draw'],s))) for r in cases]).to(device)
            codes=flow.sample_loop(model,noise.shape,{'y':y},noise=noise,repaint_enabled=False)
            state=decode_window(codec,codes,anchor)
            if not all(torch.isfinite(getattr(state,k)).all() for k in FIELDS):raise ValueError(f'Nonfinite E7 frame {s}.')
            committed.append(state)
            if (s+1)%20==0 or s==end-1:
                states=stack_states(committed)
                save_torch(cp,{'identity':identity,'next_frame':s+1,**{k:getattr(states,k).cpu() for k in FIELDS}})
                atomic_json(a.output/'progress.json',{'phase':'E7_history','committed_frames':s+1,'total_frames':end,'cases':n,
                   'elapsed_seconds_this_attempt':time.monotonic()-started,'updated_at':now()})
                print(f'E7 {s+1}/{end}, cases={n}, window={bounds}, elapsed={time.monotonic()-started:.1f}s',flush=True)
        del model;torch.cuda.empty_cache()
        states=stack_states(committed)
        save_torch(hp,{'identity':identity,'cases':identity['cases'],
            'observation_ranges':[[max(0,s-79),s+1] for s in range(20,end)],
            'fixed_beta_boot':betas.cpu(),'fixed_floor':floors.cpu(),**{k:getattr(states,k).cpu() for k in FIELDS}})
        atomic_json(hm,{'identity':identity,'sha256':file_sha256(hp)})
    # Store all P predictions before attaching any target fields or loading SMPL.
    indices=torch.tensor([(c,t) for c in range(n) for t in range(20,end)])
    predictions={};models=[]
    for spec in plan['predictors']:
        if file_sha256(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('P checkpoint changed.')
        saved=torch.load(spec['checkpoint'],weights_only=True,map_location='cpu')
        if saved['base_mode']!='constant_velocity' or saved['selected_step']!=spec['selected_step']:raise ValueError('P configuration differs.')
        for key in ('stats_sha256','reference_mode','smplx_asset_sha256','bootstrap_cache_sha256'):
            expected=payload['identity']['original_identity'].get(key)
            if expected is not None and saved.get(key)!=expected:raise ValueError(f'P/data convention differs: {key}')
        model=HistoryPrior().to(device).eval().requires_grad_(False);model.base_mode='constant_velocity';model.load_state_dict(saved['state_dict'])
        output=[]
        for start in range(0,len(indices),32):
            c,t=indices[start:start+32].to(device).T;history_times=t[:,None]+torch.arange(-20,0,device=device)[None]
            history=BodyState(*(getattr(states,k)[c[:,None],history_times] for k in FIELDS))
            output.append(predict(model,input_batch(codec,history)).cpu())
        predictions[str(spec['training_seed'])]=torch.cat(output);del model
        print('P predictions complete',spec['name'],len(indices),flush=True)
    atomic_json(a.output/'progress.json',{'phase':'offline_metrics','cases':n,'updated_at':now()})
    # Offline labels only enter below this line.
    gt=BodyState(*(torch.stack([r['truth'][k] for r in cases]).to(device) for k in FIELDS))
    smpl=get_smpl().to(device).eval().requires_grad_(False)
    asset=Path(os.environ.get('SMPLX_MODEL_PATH','body_models/smplx'))/'SMPLX_NEUTRAL.npz'
    if file_sha256(asset)!=payload['identity']['original_identity']['smplx_asset_sha256']:raise ValueError('SMPL asset changed.')
    metrics={}
    for seed,codes in predictions.items():
        values={}
        for start in range(0,len(indices),32):
            c,t=indices[start:start+32].to(device).T
            target=BodyState(*(getattr(gt,k)[c,t] for k in FIELDS))
            batch={'previous_reference':states.reference[c,t-1],'target_joints':target.joints[...,:3,3],
                   'beta_boot':betas[c],'beta_boot_is_model':torch.ones(len(c),device=device,dtype=torch.bool)}
            measured,_,_=measure(codec,codes[start:start+32].to(device),batch,target,smpl)
            for k,v in measured.items():values.setdefault(k,[]).append(v)
        metrics[seed]={k:torch.cat(v).reshape(n,end-20) for k,v in values.items()}
    # History quality at each physical frame, with same beta and metric conventions.
    quality={};qindices=torch.tensor([(c,t) for c in range(n) for t in range(end)])
    for start in range(0,len(qindices),32):
        c,t=qindices[start:start+32].to(device).T
        state=BodyState(*(getattr(states,k)[c,t] for k in FIELDS));target=BodyState(*(getattr(gt,k)[c,t] for k in FIELDS))
        batch={'previous_reference':state.reference,'target_joints':target.joints[...,:3,3],
               'beta_boot':betas[c],'beta_boot_is_model':torch.ones(len(c),device=device,dtype=torch.bool)}
        v,_,_=measure(codec,codec.encode_current(state,state.reference)[:,None],batch,target,smpl)
        for k,value in v.items():quality.setdefault(k,[]).append(value)
    quality={k:torch.cat(v).reshape(n,end) for k,v in quality.items()}
    c,t=indices.to(device).T;hist_times=t[:,None]+torch.arange(-20,0,device=device)[None]
    hist_quality={k:v.to(device)[c[:,None],hist_times].mean(1).cpu().reshape(n,end-20) for k,v in quality.items()}
    pred_vel=(states.joints[:,1:,:,:3,3]-states.joints[:,:-1,:,:3,3])*10
    gt_vel=(gt.joints[:,1:end,:,:3,3]-gt.joints[:,:end-1,:,:3,3])*10
    hist_quality['last_velocity_error_mmps']=(pred_vel-gt_vel).norm(dim=-1).mean(-1)[:,18:end-2].cpu()*1000
    pred_delta=states.joints[:,1:,:,:3,:3]@states.joints[:,:-1,:,:3,:3].transpose(-1,-2)
    gt_delta=gt.joints[:,1:end,:,:3,:3]@gt.joints[:,:end-1,:,:3,:3].transpose(-1,-2)
    hist_quality['last_angular_velocity_error_degps']=rotation_angle(pred_delta,gt_delta).mean(-1)[:,18:end-2].cpu()*10
    result={'identity':identity,'cases':identity['cases'],'times':torch.arange(20,end),'metrics':metrics,
            'history_quality':hist_quality,'history_last_frame_quality':{k:v[:,19:end-1] for k,v in quality.items()},
            'records':[r['record'] for r in cases], 'predictions':predictions,'indices':indices}
    if not all(torch.isfinite(v).all() for group in metrics.values() for v in group.values()):raise ValueError('Invalid metrics.')
    save_torch(a.output/'frames.pt',result)
    atomic_json(a.output/'report.json',{'completed':True,'identity':identity,'cases':n,'target_frames':len(indices),
                'artifact_sha256':file_sha256(a.output/'frames.pt'),'histories':str(hp),'history_sha256':file_sha256(hp),
                'elapsed_seconds':time.monotonic()-started,'smoke':a.smoke,'holdout_used':False})
    print('COMPLETE',a.output,flush=True)
if __name__=='__main__':main()
