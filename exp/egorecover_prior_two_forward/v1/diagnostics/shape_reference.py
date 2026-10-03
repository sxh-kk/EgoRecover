import json,time
from pathlib import Path
import torch
from dataset.smpl_utils import get_smpl
from egorecover.fk import FixedShapeFK
from egorecover.prior_two_forward import PriorSequences,summarize
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import now,append_log

torch.set_num_threads(4)
root=Path('exp/egorecover_prior_two_forward/v1')
data=PriorSequences(root/'data','dev');indices=data.indices();smpl=get_smpl().eval().requires_grad_(False)
fixed_poses=[];same_beta=[];recorded=[]
with torch.no_grad():
 for start in range(0,len(indices),32):
  _,target,beta=data.examples(indices[start:start+32],'cpu')
  fixed_poses.append(FixedShapeFK(smpl,beta)(target))
  same_beta.append(FixedShapeFK(smpl,target.auxiliary[:,-10:])(target))
  recorded.append(target.joints[...,:3,3])
fixed=torch.cat(fixed_poses);gtbeta=torch.cat(same_beta);recorded=torch.cat(recorded)
def metric(a,b):return summarize((a-b).norm(dim=-1).mean(-1)*1000,indices,data.takes)
result={'at':now(),'scope':'Offline dev diagnostic only; GT pose/root with model startup shape is a reference, not a strict achievable lower bound. GT shape is diagnostic only, not used by deployed P.', 'frames':len(indices),'fixed_shape_gt_pose_vs_recorded_gt_mm':metric(fixed,recorded),'gt_shape_gt_pose_vs_recorded_gt_mm':metric(gtbeta,recorded),'models':{}}
for path in sorted((root/'train').glob('*_s62/dev_*.pt')):
 if path.name=='dev_prior.pt':name=path.parent.name
 elif path.parent.name=='A_gt_dense_s62':name=path.stem.removeprefix('dev_')
 else:continue
 saved=torch.load(path,weights_only=True,map_location='cpu')['single']
 assert saved['takes']==data.takes and torch.equal(saved['indices'],indices)
 pred=saved['joints'];error=pred-fixed
 result['models'][name]={'formal_fk_mm':metric(pred,recorded),'same_shape_pose_prediction_mm':metric(pred,fixed),'same_shape_root_relative_mm':summarize((error-error[:,:1]).norm(dim=-1).mean(-1)*1000,indices,data.takes)}
folder=root/'diagnostics';folder.mkdir(exist_ok=True);atomic_json(folder/'shape_reference.json',result)
print(json.dumps({k:({n:{x:round(y['mean'],3) for x,y in v.items()} for n,v in value.items()} if k=='models' else value['mean'] if isinstance(value,dict) and 'mean' in value else value) for k,value in result.items()},indent=2),flush=True)
