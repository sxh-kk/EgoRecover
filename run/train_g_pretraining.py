"""P protocol calibration and four controlled history-G training runs."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch

from config.defaults import get_cfg_defaults
from dataset.smpl_utils import get_smpl
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import BodyState, MotionCodec
from egorecover.evaluation_protocol import file_sha256
from egorecover.fk import FixedShapeFK
from egorecover.g_pretraining import GROUPS, PairedSequences, keyed_seed, mixture, sensor_conditions
from egorecover.history_flow import HistoryFlow
from egorecover.losses import physical_objective
from egorecover.prior import HistoryPrior
from egorecover.prior_cv_residual import predict, rotation_angle
from egorecover.prior_two_forward import FIELDS, physical_batch, state_map
from model.history_uniegomotion import HistoryUniEgoMotion
from run.e7_ablation import save, write_json


def load_prior(path,device,freeze=True,manifest_sha=None):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if manifest_sha is not None and ckpt.get('manifest_sha256')!=manifest_sha:
        raise ValueError('P checkpoint belongs to a different data protocol.')
    p=HistoryPrior().to(device);p.base_mode='constant_velocity'
    p.load_state_dict(ckpt['state_dict'],strict=True)
    return p.freeze() if freeze else p


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def macro(values, names):
    per={name:float(values[[i for i,n in enumerate(names) if n==name]].mean()) for name in sorted(set(names))}
    return {'mean':sum(per.values())/len(per),'per_take':per}


def summarize_rows(rows):
    names=[r['take'] for r in rows]
    return {k:macro(torch.tensor([r[k] for r in rows]),names)
            for k,value in rows[0].items()
            if isinstance(value,(float,int)) and k.endswith(('_mm','_mmps','_deg'))}


@torch.no_grad()
def prior_score(prior,data,codec,smpl,device):
    prior.eval(); errors=[]
    # Fixed, evenly spaced targets on the predeclared monitor takes.
    for e in range(len(data.ids)):
        idx=torch.tensor([[e,t] for t in range(20,200,10)])
        h,target,beta,_,_=data.examples(idx,torch.zeros(len(idx),dtype=torch.bool),device)
        batch=physical_batch(codec,h,target,beta)
        state=codec.decode_current(predict(prior,batch)[:,0],batch['previous_reference'])
        joints=FixedShapeFK(smpl,beta)(state)
        errors.append((joints-batch['target_joints']).norm(dim=-1).mean().cpu()*1000)
    return macro(torch.stack(errors),data.names)


def monitor_data(output,cfg,limit=None):
    # Manifest records are grouped by take. Select a deterministic spread of tasks
    # before looking at any scores, then persist the exact monitor names.
    manifest=json.loads((output/'data/manifest.json').read_text())
    records=[r for r in manifest['records'] if r['group']=='dev']
    strata={}
    for r in records:strata.setdefault(r.get('task') or 'unknown',[]).append(r['base_take_name'])
    chosen=[]
    while len(chosen)<min(cfg['dev_monitor_takes'],len(records)):
        for names in strata.values():
            if names and len(chosen)<cfg['dev_monitor_takes']:chosen.append(names.pop(0))
    monitor={'takes':chosen,'selection':'task round robin, fixed before scores'}
    monitor_path=output/'data/dev_monitor.json'
    if monitor_path.exists():
        if json.loads(monitor_path.read_text())!=monitor:raise ValueError('Monitor set changed.')
    else:write_json(monitor_path,monitor)
    return PairedSequences(output,'dev',take_names=chosen[:limit] if limit else chosen)


@torch.no_grad()
def evaluate(model,prior,data,codec,smpl,device,*,mode,draw=1062,nfe=10,max_frames=200):
    model.eval(); prior.eval(); flow=HistoryFlow(source_mode=mode,sigma=1.)
    rows=[]; start=time.monotonic()
    for offset in range(0,len(data.ids),8):
        end=min(len(data.ids),offset+8); ids=list(range(offset,end)); b=len(ids)
        history=state_map(data.teacher,lambda x:x[ids,:20].to(device))
        beta=data.beta[ids].to(device);fk=FixedShapeFK(smpl,beta)
        head=data.head[ids].to(device); images=data.images[ids].to(device)
        gt=data.truth.joints[ids,...,:3,3].to(device)
        poses=[];p_errors=[];rotations=[]
        for t in range(20,max_frames):
            y=sensor_conditions(codec,history,head[:,t-20:t+1],images[:,t-20:t+1],prior)
            noise=torch.stack([torch.randn(1,243,generator=torch.Generator().manual_seed(keyed_seed(data.ids[i],draw,t))) for i in ids]).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                code=flow.sample(model,y,epsilon=noise,num_steps=nfe)
            code=code.float();previous=history.reference[:,-1]
            state=codec.decode_current(code[:,0],previous)
            pose=fk(state);poses.append(pose.cpu())
            p_state=codec.decode_current(y['prior_mu'][:,0],previous)
            p_errors.append((fk(p_state)-gt[:,t]).norm(dim=-1).mean(-1).cpu()*1000)
            target_rot=data.truth.joints[ids,t,:,:3,:3].to(device)
            rotations.append(rotation_angle(state.joints[...,:3,:3],target_rot).mean(-1).cpu())
            history=BodyState(*(torch.cat((getattr(history,k)[:,1:],getattr(state,k)[:,None]),1) for k in FIELDS))
        poses=torch.stack(poses,1);truth=gt[:,20:max_frames].cpu()
        errors=(poses-truth).norm(dim=-1)*1000
        rootrelative=((poses-poses[:,:,:1])-(truth-truth[:,:,:1])).norm(dim=-1).mean(-1)*1000
        from eval.metrics import reconstruction_error
        pa=torch.as_tensor(reconstruction_error(poses.flatten(0,1).numpy(),truth.flatten(0,1).numpy(),reduction=None)).reshape(b,-1)*1000
        velocity=((poses[:,1:]-poses[:,:-1])-(truth[:,1:]-truth[:,:-1])).norm(dim=-1).mean(-1)*10000
        p_errors=torch.stack(p_errors,1);rotations=torch.stack(rotations,1)
        for j,i in enumerate(ids):
            late=slice(20,None) if max_frames>40 else slice(None)
            rows.append({'take':data.names[i],'episode':data.ids[i],
                'body_mm':float(errors[j,late].mean()),'all_body_mm':float(errors[j].mean()),
                'startup_body_mm':float(errors[j,:20].mean()),'pa_mm':float(pa[j,late].mean()),
                'root_mm':float(errors[j,late,0].mean()),'root_relative_mm':float(rootrelative[j,late].mean()),
                'head_mm':float(errors[j,late,15].mean()),'global_joint_rotation_deg':float(rotations[j,late].mean()),
                'velocity_mmps':float(velocity[j].mean()) if velocity.shape[1] else 0.,
                'prior_body_mm':float(p_errors[j,late].mean()),
                'p_to_g_improvement_mm':float((p_errors[j,late]-errors[j,late].mean(-1)).mean()),
                'joints':poses[j],'gt':truth[j],'frame_body_mm':errors[j].mean(-1),'frame_prior_mm':p_errors[j]})
    scores=summarize_rows(rows)
    return {'scores':scores,'nfe':nfe,'draw':draw,'seconds':time.monotonic()-start,'frames':sum(len(r['joints']) for r in rows)},rows


def calibrate(args,cfg):
    out=args.output;device=torch.device(args.device);directory=out/'prior'
    train=PairedSequences(out,'train',limit=2 if args.smoke else None)
    dev=monitor_data(out,cfg,limit=1 if args.smoke else None)
    codec=MotionCodec(train.stats).to(device);smpl=get_smpl().to(device).eval().requires_grad_(False)
    torch.manual_seed(62);prior=load_prior(cfg['prior_checkpoint'],device,freeze=False)
    initial=prior_score(prior,dev,codec,smpl,device)
    best=initial['mean'];best_step=0;best_state=cpu_state(prior)
    save(directory/'prior.pt',{'state_dict':best_state,'base_mode':'constant_velocity','selected_step':0,
                             'manifest_sha256':train.manifest_sha})
    optimizer=torch.optim.AdamW(prior.parameters(),lr=cfg['prior_calibration_lr'],weight_decay=.01)
    rng=torch.Generator().manual_seed(62);curve=[{'step':0,**initial}]
    steps=2 if args.smoke else cfg['prior_calibration_steps']
    for step in range(1,steps+1):
        prior.train();idx=train.sample(32,rng)
        h,target,beta,_,_=train.examples(idx,torch.zeros(32,dtype=torch.bool),device)
        batch=physical_batch(codec,h,target,beta)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            pred=predict(prior,batch)
        loss=physical_objective(codec,pred.float(),batch,geometry_weight=1.,fk_weight=0.)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%20==0 or step==steps:
            write_json(directory/'progress.json',{'step':step,'total':steps,'loss':float(loss.detach()),'pid':os.getpid()})
            print('P calibration',step,steps,float(loss.detach()),flush=True)
        if step%400==0 or step==steps:
            score=prior_score(prior,dev,codec,smpl,device);curve.append({'step':step,**score})
            if score['mean']<best:
                best=score['mean'];best_step=step;best_state=cpu_state(prior)
                save(directory/'prior.pt',{'state_dict':best_state,'base_mode':'constant_velocity','selected_step':step,
                                         'manifest_sha256':train.manifest_sha})
            write_json(directory/'selection.json',{'curve':curve,'selected_step':best_step})
    write_json(directory/'report.json',{'completed':True,'smoke':args.smoke,'initial_body_mm':initial['mean'],
               'selected_body_mm':best,'selected_step':best_step,'manifest_sha256':train.manifest_sha,
               'prior_sha256':file_sha256(directory/'prior.pt'),'meaning':'GT-history protocol calibration; not actual-history accuracy'})


def train(args,cfg):
    out=args.output;directory=out/args.group/f'seed{args.seed}';device=torch.device(args.device)
    if args.smoke:directory=out/'verification'/f'{args.group}_smoke'
    directory.mkdir(parents=True,exist_ok=True)
    data=PairedSequences(out,'train',limit=2 if args.smoke else None)
    dev=monitor_data(out,cfg,limit=1 if args.smoke else None)
    prior=load_prior(out/'prior/prior.pt',device,manifest_sha=data.manifest_sha)
    codec=MotionCodec(data.stats).to(device);smpl=get_smpl().to(device).eval().requires_grad_(False)
    torch.manual_seed(args.seed)
    model=HistoryUniEgoMotion(get_cfg_defaults(),past_observations=True).to(device)
    load_e7_weights(model,cfg['e7_checkpoint'],weight_source='ema')
    flow=HistoryFlow(source_mode=GROUPS[args.group][0],sigma=cfg['sigma'])
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    rng=torch.Generator().manual_seed(args.seed+10);mix_rng=torch.Generator().manual_seed(args.seed+20)
    noise_rng=torch.Generator().manual_seed(args.seed+30)
    ident={'group':args.group,'seed':args.seed,'manifest_sha256':data.manifest_sha,
        'prior_sha256':file_sha256(out/'prior/prior.pt'),'config':cfg,
        'code':{p:file_sha256(p) for p in ('run/train_g_pretraining.py','egorecover/g_pretraining.py',
                 'model/history_uniegomotion.py','egorecover/conditioning.py','egorecover/history_flow.py','egorecover/losses.py')}}
    identity_path=directory/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=ident:raise ValueError('Run identity changed.')
    write_json(identity_path,ident)
    steps=2 if args.smoke else cfg['steps'];start_step=0;curve=[];best=float('inf');best_step=0
    resume=directory/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=True)
        if saved['identity']!=ident:raise ValueError('Resume identity changed.')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);ema=saved['ema']
        start_step=saved['step'];curve=saved['curve'];best=saved['best'];best_step=saved['best_step']
        rng.set_state(saved['sample_rng'].cpu());mix_rng.set_state(saved['mix_rng'].cpu());noise_rng.set_state(saved['noise_rng'].cpu())
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        del saved
    begin=time.monotonic();model.train()
    for step in range(start_step+1,steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        p=mixture(step) if GROUPS[args.group][1] else 0.
        if args.smoke and GROUPS[args.group][1]:p=.5
        indices=data.sample(cfg['batch_size'],rng)
        choose=torch.rand(cfg['batch_size'],generator=mix_rng)<p
        epsilon=torch.randn(cfg['batch_size'],1,243,generator=noise_rng)
        # Separate deterministic time stream so model dropout does not change it.
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
            torch.manual_seed(keyed_seed(args.seed,step,'flow_time'))
            times=flow.flow.sample_timesteps(cfg['batch_size'],'cpu')
        factor=min(1.,step/cfg['warmup_steps']) if step<=cfg['warmup_steps'] else .1+.9*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/(cfg['steps']-cfg['warmup_steps'])))/2
        for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*factor
        losses=[]
        for offset in range(0,cfg['batch_size'],cfg['microbatch']):
            sl=slice(offset,min(offset+cfg['microbatch'],cfg['batch_size']))
            batch,y=data.batch(codec,prior,indices[sl],choose[sl],device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                result=flow.training_losses(model,batch['target'],y,epsilon=epsilon[sl].to(device),t=times[sl].to(device),return_diagnostics=True)
            loss=physical_objective(codec,result['model_output'].float(),batch,geometry_weight=1.,fk_weight=0.)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss.')
            (loss*len(indices[sl])/cfg['batch_size']).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for name,value in model.state_dict().items():
                if value.is_floating_point():ema[name].lerp_(value,1-cfg['ema_decay'])
                else:ema[name].copy_(value)
        if step%20==0 or step in (1,steps):
            progress={'step':step,'total':steps,'loss':sum(losses)/len(losses),'mixture_probability':p,
                'actual_mixture':float(choose.float().mean()),'gradient_norm':float(norm),'pid':os.getpid(),
                'elapsed_this_session_seconds':time.monotonic()-begin,'steps_this_session':step-start_step,
                'phase':'training','gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
            write_json(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step%cfg['eval_every']==0 or step==steps:
            inference=copy.deepcopy(model).eval();inference.load_state_dict(ema)
            score,rows=evaluate(inference,prior,dev,codec,smpl,device,mode=GROUPS[args.group][0],max_frames=24 if args.smoke else 200)
            del inference
            value=score['scores']['body_mm']['mean'];curve.append({'step':step,**score})
            checkpoint={'state_dict':{k:v.cpu() for k,v in ema.items()},'identity':ident,'step':step,'past_observations':True}
            if value<best:
                best=value;best_step=step;save(directory/'best.pt',checkpoint)
            save(directory/'last.pt',checkpoint)
            save(directory/'dev'/f'{step}.pt',{'summary':score,'rows':rows})
            write_json(directory/'selection.json',{'best_step':best_step,'best_body_mm':best,'curve':curve})
            save(resume,{'identity':ident,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'ema':ema,'sample_rng':rng.get_state(),'mix_rng':mix_rng.get_state(),'noise_rng':noise_rng.get_state(),
                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                'curve':curve,'best':best,'best_step':best_step})
    write_json(directory/'report.json',{'completed':True,'smoke':args.smoke,'identity':ident,'best_step':best_step,
        'best_dev_body_mm':best,'steps':steps,'last_sha256':file_sha256(directory/'last.pt'),'best_sha256':file_sha256(directory/'best.pt')})


def final_eval(args,cfg):
    out=args.output;device=torch.device(args.device);directory=out/args.group/f'seed{args.seed}'
    data=PairedSequences(out,'test');codec=MotionCodec(data.stats).to(device)
    smpl=get_smpl().to(device).eval().requires_grad_(False);prior=load_prior(out/'prior/prior.pt',device,manifest_sha=data.manifest_sha)
    model=HistoryUniEgoMotion(get_cfg_defaults(),past_observations=True).to(device)
    for selected in ('last','best'):
        checkpoint=torch.load(directory/f'{selected}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        for nfe in (cfg['nfe'] if selected=='best' else [10]):
            for draw in cfg['draws']:
                path=directory/'test'/f'{selected}_nfe{nfe}_draw{draw}.pt'
                if path.exists():continue
                summary,rows=evaluate(model,prior,data,codec,smpl,device,mode=GROUPS[args.group][0],draw=draw,nfe=nfe)
                save(path,{'identity':checkpoint['identity'],'checkpoint_sha256':file_sha256(directory/f'{selected}.pt'),
                           'summary':summary,'rows':rows})
                print('TEST',args.group,args.seed,selected,nfe,draw,summary['scores']['body_mm']['mean'],flush=True)
    auxiliary=directory/'test/auxiliary.pt'
    if not auxiliary.exists():
        # model now holds best EMA; fixed teacher starts isolate correction ability.
        measured=fixed_history_diagnostics(model,prior,data,codec,smpl,device,mode=GROUPS[args.group][0])
        save(auxiliary,measured)
    write_json(directory/'test/report.json',{'completed':True})


@torch.no_grad()
def fixed_history_diagnostics(model,prior,data,codec,smpl,device,*,mode):
    model.eval();flow=HistoryFlow(source_mode=mode);rows=[]
    for offset in range(0,len(data.ids),8):
        ids=list(range(offset,min(offset+8,len(data.ids))))
        beta=data.beta[ids].to(device);fk=FixedShapeFK(smpl,beta)
        for start in (40,80,120,160):
            for kind in ('teacher','GT'):
                base=data.teacher if kind=='teacher' else data.truth
                h=state_map(base,lambda x:x[ids,start-20:start].to(device))
                for horizon in range(1,9 if kind=='teacher' else 2):
                    t=start+horizon-1
                    head=data.head[ids,t-20:t+1].to(device);images=data.images[ids,t-20:t+1].to(device)
                    y=sensor_conditions(codec,h,head,images,prior)
                    eps=torch.stack([torch.randn(1,243,generator=torch.Generator().manual_seed(keyed_seed(data.ids[i],1062,t))) for i in ids]).to(device)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        code=flow.sample(model,y,epsilon=eps,num_steps=10)
                    state=codec.decode_current(code[:,0].float(),h.reference[:,-1])
                    if horizon in (1,4,8):
                        gt=data.truth.joints[ids,t,:,:3,3].to(device)
                        errors=(fk(state)-gt).norm(dim=-1).mean(-1)*1000
                        p_state=codec.decode_current(y['prior_mu'][:,0],h.reference[:,-1])
                        p_errors=(fk(p_state)-gt).norm(dim=-1).mean(-1)*1000
                        rows.extend({'take':data.names[i],'start':start,'history':kind,'horizon':horizon,
                                     'body_mm':float(errors[j]),'prior_mm':float(p_errors[j])} for j,i in enumerate(ids))
                    h=BodyState(*(torch.cat((getattr(h,k)[:,1:],getattr(state,k)[:,None]),1) for k in FIELDS))
    return {'rows':rows,'scope':'same teacher histories at horizon1; separate short closed loops at horizon4/8; GT diagnostic only'}


@torch.no_grad()
def baseline(args,cfg):
    device=torch.device(args.device);data=PairedSequences(args.output,'test')
    if data.teacher_window_beta is None:raise ValueError('Native window shape cache missing.')
    smpl=get_smpl().to(device).eval().requires_grad_(False);rows=[]
    for i,take in enumerate(data.names):
        state=state_map(data.teacher,lambda x:x[i,20:].to(device));gt=data.truth.joints[i,20:,:,:3,3].to(device)
        values={}
        for policy,beta in [('common_bootstrap',data.beta[i].to(device)),('native_window',data.teacher_window_beta[i,20:].to(device))]:
            pred=FixedShapeFK(smpl,beta)(state)
            errors=(pred-gt).norm(dim=-1)*1000
            values[policy]={'body_mm':float(errors[20:].mean()),'root_mm':float(errors[20:,0].mean()),
                'all_body_mm':float(errors.mean()),'velocity_mmps':float(((pred[1:]-pred[:-1])-(gt[1:]-gt[:-1])).norm(dim=-1).mean()*10000)}
        rows.append({'take':take,'episode':data.ids[i],'metrics':values})
    scores={p:macro(torch.tensor([r['metrics'][p]['body_mm'] for r in rows]),data.names) for p in ('common_bootstrap','native_window')}
    write_json(args.output/'evaluation/e7_baseline.json',{'completed':True,'draw':1062,'scores':scores,'rows':rows,
        'note':'Single shared teacher draw; compare G draw1062 for matched baseline reporting. Ground-truth floor; startup beta or native per-window beta as labeled.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prior','train','evaluate','baseline'])
    p.add_argument('--output',type=Path,default=Path('exp/egorecover_g_pretraining/v1'))
    p.add_argument('--group',choices=list(GROUPS),default='T0');p.add_argument('--seed',type=int,default=62)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    lock_dir=args.output/'locks';lock_dir.mkdir(parents=True,exist_ok=True)
    lock=(lock_dir/f'{args.stage}_{args.group}_{args.seed}_{args.smoke}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((args.output/'plan.json').read_text())
    {'prior':calibrate,'train':train,'evaluate':final_eval,'baseline':baseline}[args.stage](args,cfg)


if __name__=='__main__':main()
