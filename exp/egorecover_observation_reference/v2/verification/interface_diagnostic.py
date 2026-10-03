import importlib.util,json,torch
from pathlib import Path
from config.defaults import get_cfg_defaults
from egorecover.codec import MotionCodec
from egorecover.g_pretraining import sensor_conditions
from egorecover.observation_reference import ReferenceSequences,conditions,gather_observations,objective,target_batch,ObservationFlow
from model.observation_reference_g import ObservationReferenceG
from model.history_uniegomotion import HistoryUniEgoMotion
from run.train_g_pretraining import load_prior
from egorecover.checkpoint import load_e7_weights

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
oldm=module('v1model','exp/egorecover_observation_reference/v1/source/model/observation_reference_g.py')
oldc=module('v1conditions','exp/egorecover_observation_reference/v1/source/egorecover/observation_reference.py')
out=Path('exp/egorecover_observation_reference/v1');device='cuda';torch.set_num_threads(4)
data=ReferenceSequences(out,'train',observation_length=80,anchored=False,limit=2)
codec=MotionCodec(data.stats).to(device);prior=load_prior(out/'prior/prior.pt',device,manifest_sha=data.manifest_sha)
idx=torch.tensor([[0,90],[1,120],[0,150],[1,199]])
h,target,*_=data.examples(idx,torch.zeros(4,dtype=torch.bool),device)
results={}
for name,cls,fn in [('v1',oldm.ObservationReferenceG,oldc.conditions),('v2',ObservationReferenceG,conditions)]:
 model=cls(get_cfg_defaults(),observation_length=21,dropout=0).to(device).eval()
 if name=='v1':load_e7_weights(model,'exp/e7/last.ckpt',weight_source='ema')
 else:model.initialize_e7('exp/e7/last.ckpt')
 obs=gather_observations(data.head[idx[:,0]],data.images[idx[:,0]],idx[:,1],21)
 y,ref,_=fn(codec,h,*(x.to(device) for x in obs),prior,False)
 batch=target_batch(codec,target,ref,False)
 noise=torch.randn(4,1,243,generator=torch.Generator().manual_seed(62)).to(device)
 r=ObservationFlow().training_losses(model,batch['target'],y,epsilon=noise,t=torch.full((4,),.5,device=device))
 loss=objective(codec,r['model_output'],batch,False);loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 results[name]={'loss':float(loss),'gradient_norm':float(norm),'trajectory_absmax':float(y['memory_traj'].abs().max())}
 if name=='v2':
  legacy=HistoryUniEgoMotion(get_cfg_defaults(),dropout=0,past_observations=True).to(device).eval()
  legacy.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('memory_traj_adapter.')})
  yo=sensor_conditions(codec,h,obs[0].to(device),obs[1].to(device),prior)
  with torch.no_grad():
   x=noise;t=torch.full((4,),.5,device=device)
   results['v2']['legacy_output_maxdiff']=float((model(x,t,y)-legacy(x,t,yo)).abs().max())
 del model
print(json.dumps(results,indent=2))
Path('exp/egorecover_observation_reference/v2/verification/interface_diagnostic.json').write_text(json.dumps(results,indent=2))
