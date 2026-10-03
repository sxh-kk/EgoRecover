"""Observation context x device-reference controlled G training."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
import torch
from config.defaults import get_cfg_defaults
from dataset.smpl_utils import get_smpl
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import BodyState, MotionCodec
from egorecover.evaluation_protocol import file_sha256
from egorecover.fk import FixedShapeFK
from egorecover.g_pretraining import keyed_seed, mixture, PairedSequences
from egorecover.prior_cv_residual import rotation_angle
from egorecover.prior_two_forward import FIELDS, state_map
from egorecover.observation_reference import (GROUPS, ReferenceSequences, ObservationFlow,
    conditions, gather_observations, decode, objective)
from model.observation_reference_g import ObservationReferenceG
from run.e7_ablation import save, write_json
from run.train_g_pretraining import load_prior, macro, summarize_rows, monitor_data

@torch.no_grad()
def evaluate(model,prior,data,codec,smpl,device,*,draw=1062,nfe=10,max_frames=200):
    model.eval(); prior.eval(); flow=ObservationFlow()
    rows=[]; start=time.monotonic()
    for offset in range(0,len(data.ids),8):
        end=min(len(data.ids),offset+8); ids=list(range(offset,end)); b=len(ids)
        history=state_map(data.teacher,lambda x:x[ids,:20].to(device))
        beta=data.beta[ids].to(device);fk=FixedShapeFK(smpl,beta)
        head=data.head[ids].to(device); images=data.images[ids].to(device)
        gt=data.truth.joints[ids,...,:3,3].to(device)
        poses=[];p_errors=[];rotations=[]
        for t in range(20,max_frames):
            oh, oi, valid = gather_observations(head, images, torch.full((b,),t,device=device,dtype=torch.long), model.observation_length)
            y, reference, p_state = conditions(codec,history,oh,oi,valid,prior,model.anchored)
            noise=torch.stack([torch.randn(1,243,generator=torch.Generator().manual_seed(keyed_seed(data.ids[i],draw,t))) for i in ids]).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                code=flow.sample(model,y,epsilon=noise,num_steps=nfe)
            code=code.float();previous=history.reference[:,-1]
            state=decode(codec,code[:,0],reference,model.anchored)
            pose=fk(state);poses.append(pose.cpu())
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


def train(args,cfg):
    out=args.output;directory=out/args.group/f'seed{args.seed}';device=torch.device(args.device)
    if args.smoke:directory=out/'verification'/f'{args.group}_smoke'
    directory.mkdir(parents=True,exist_ok=True)
    length,anchored=GROUPS[args.group]
    data=ReferenceSequences(out,'train',observation_length=length,anchored=anchored,limit=2 if args.smoke else None)
    dev=monitor_data(out,cfg,limit=1 if args.smoke else None)
    prior=load_prior(out/'prior/prior.pt',device,manifest_sha=data.manifest_sha)
    codec=MotionCodec(data.stats).to(device);smpl=get_smpl().to(device).eval().requires_grad_(False)
    torch.manual_seed(args.seed)
    model=ObservationReferenceG(get_cfg_defaults(),observation_length=length,anchored=anchored).to(device)
    model.initialize_e7(cfg['e7_checkpoint'])
    flow=ObservationFlow()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    rng=torch.Generator().manual_seed(args.seed+10);mix_rng=torch.Generator().manual_seed(args.seed+20)
    noise_rng=torch.Generator().manual_seed(args.seed+30)
    ident={'group':args.group,'seed':args.seed,'manifest_sha256':data.manifest_sha,
        'prior_sha256':file_sha256(out/'prior/prior.pt'),'config':cfg,
        'code':{p:file_sha256(p) for p in ('run/train_observation_reference.py','egorecover/observation_reference.py','model/observation_reference_g.py')}}
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
        p=mixture(step)
        if args.smoke:p=.5
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
                result=flow.training_losses(model,batch['target'],y,epsilon=epsilon[sl].to(device),t=times[sl].to(device))
            loss=objective(codec,result['model_output'].float(),batch,anchored)
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
            score,rows=evaluate(inference,prior,dev,codec,smpl,device,max_frames=96 if args.smoke else 200)
            del inference
            value=score['scores']['body_mm']['mean'];curve.append({'step':step,**score})
            checkpoint={'state_dict':{k:v.cpu() for k,v in ema.items()},'identity':ident,'step':step,'observation_length':length,'anchored':anchored}
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


def final_eval(args, cfg):
    out = args.output; device = torch.device(args.device)
    directory = out / args.group / f'seed{args.seed}'
    length, anchored = GROUPS[args.group]
    frozen=json.loads((out/'evaluation/selected_checkpoints.json').read_text())
    if file_sha256(directory/'best.pt')!=frozen[f'{args.group}/seed{args.seed}']:
        raise ValueError('Selected checkpoint changed after evaluation freeze.')
    data = PairedSequences(out, 'test')
    codec = MotionCodec(data.stats).to(device)
    smpl = get_smpl().to(device).eval().requires_grad_(False)
    prior = load_prior(out/'prior/prior.pt',device,manifest_sha=data.manifest_sha)
    model = ObservationReferenceG(get_cfg_defaults(), observation_length=length, anchored=anchored).to(device)
    checkpoint = torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    summary, rows = evaluate(model, prior, data, codec, smpl, device)
    save(directory/'test/best_nfe10_draw1062.pt', {'summary': summary, 'rows': rows,
        'identity': checkpoint['identity'], 'checkpoint_sha256':file_sha256(directory/'best.pt')})
    write_json(directory/'test/report.json', {'completed':True,'body_mm':summary['scores']['body_mm']['mean']})
    print('TEST',args.group,args.seed,summary['scores']['body_mm']['mean'],flush=True)


@torch.no_grad()
def diagnostic(args, cfg):
    """Frozen native E7: 21 versus cached 80 observations on fixed dev only."""
    from egorecover.bootstrap_shapes import load_e7_initializer
    from egorecover.e7_ablation import native_head_encoding
    from egorecover.g_pretraining import native_states
    from mydiffusion.flow_matching import FlowMatching
    data = monitor_data(args.output,cfg)
    device = torch.device(args.device); codec = MotionCodec(data.stats)
    model = load_e7_initializer(cfg['e7_checkpoint'],device=device,weight_source='ema')
    smpl = get_smpl().to(device).eval().requires_grad_(False); flow = FlowMatching()
    rows=[]
    for i,take in enumerate(data.names):
        predictions=[]
        for start in range(40,200,16):
            targets=list(range(start,min(start+16,200))); batch=[]; anchors=[]; noise=[]
            for t in targets:
                raw,anchor=native_head_encoding(data.head[i,t-20:t+1],0.)
                trajectory=codec.normalize(raw,'traj')
                pad=lambda x:torch.cat((x,x.new_zeros(80-len(x),*x.shape[1:])))
                valid=torch.arange(80)<21
                batch.append({'traj':pad(trajectory),'img_embs':pad(data.images[i,t-20:t+1]),
                    'valid_frames':valid.long(),'valid_img_embs':valid.long(),
                    'traj_mask':torch.zeros(80,dtype=torch.long),'img_mask':torch.zeros(80,dtype=torch.long)})
                anchors.append(anchor)
                noise.append(torch.randn(80,243,generator=torch.Generator().manual_seed(keyed_seed(data.ids[i],1062,t))))
            y={k:torch.stack([v[k] for v in batch]).to(device) for k in batch[0]}
            eps=torch.stack(noise).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                codes=flow.sample_loop(model,eps.shape,{'y':y},noise=eps).float().cpu()
            for j,t in enumerate(targets):
                state=native_states(codec,codes[j,:21],anchors[j])
                predictions.append(BodyState(*(getattr(state,k)[-1] for k in FIELDS)))
        pred=BodyState(*(torch.stack([getattr(s,k) for s in predictions]).to(device) for k in FIELDS))
        teacher=state_map(data.teacher,lambda x:x[i,40:].to(device))
        gt=data.truth.joints[i,40:,:,:3,3].to(device)
        fk=FixedShapeFK(smpl,data.beta[i].to(device))
        values={str(n):float((fk(s)-gt).norm(dim=-1).mean()*1000) for n,s in [(21,pred),(80,teacher)]}
        rows.append({'take':take,**values})
        print('E7 CONTEXT',take,values,flush=True)
        write_json(args.output/'diagnostic/progress.json',{'completed_takes':len(rows),'total':len(data.names),'rows':rows})
    write_json(args.output/'diagnostic/report.json',{'completed':True,'scope':'16 fixed dev takes, t40..199, common startup beta, draw1062',
        'mean_mm':{str(n):sum(r[str(n)] for r in rows)/len(rows) for n in (21,80)},'rows':rows,
        'note':'80-frame results reuse verified native E7 cache; padding/canonicalization follow native protocol. Diagnostic, not G accuracy.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','evaluate','diagnostic'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group',choices=list(GROUPS),default='R0')
    p.add_argument('--seed',type=int,default=62)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.output=args.output.resolve();torch.set_num_threads(4)
    cfg=json.loads((args.output/'plan.json').read_text())
    if args.stage=='train':train(args,cfg)
    elif args.stage=='evaluate':final_eval(args,cfg)
    else:diagnostic(args,cfg)


if __name__=='__main__':main()
