"""Read-only inventory of workspace files; writes reports, never uploads.

Symlinks are recorded without traversing them. Checksums cover small release
candidates only. Checkpoint payloads are not loaded by this scanner.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from datetime import datetime
from zoneinfo import ZoneInfo

SOURCE_DIRS = {'egorecover','model','mydiffusion','module','dataset','data_pipeline','run','eval','utils','tests','tools','config','docs'}
BINARY = {'.pt','.pth','.ckpt','.pkl','.npz','.npy','.safetensors','.bin'}
TEXT = {'.py','.md','.txt','.json','.jsonl','.yaml','.yml','.toml','.sh','.cfg','.ini','.csv','.tsv'}
KEY_PATTERNS = {
 'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
 'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
 'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 'credential_url': re.compile(r'https?://[^\s/<>:]+:[^\s/<>@]+@'),
}
PATH_RE=re.compile(r'/(?:gaozt-test1|home|root|mnt|data)(?:/[^\s\"\'<>),;]+)+')


def category(rel):
 p=Path(rel);parts=p.parts;top=parts[0];name=p.name;ext=p.suffix.lower()
 if top=='.git':return 'local_git'
 if any(x in {'__pycache__','.pytest_cache','.venv','.vscode','.codex','.agents'} for x in parts) or ext in {'.pyc','.pyo'}:return 'local_ephemeral'
 if name=='.env' or name.startswith('.env.') and name!='.env.example' or ext in {'.pem','.key'}:return 'local_credentials'
 if top=='body_models':return 'restricted_smplx'
 if top=='data':return 'external_dataset_reference' if 'ee4d_motion_uniegomotion' in parts else 'derived_dataset_review'
 if name=='cv.md' or top=='图片和附件':return 'personal_review'
 if top=='exp':
  if top and len(parts)>1 and parts[1]=='release_audit':return 'local_audit'
  if 'source' in parts:return 'hf_source_snapshots'
  if ext in {'.pid','.lock'} or 'locks' in parts:return 'local_ephemeral'
  if 'data' in parts or 'episodes' in parts or any('cache' in x for x in parts) or 'histories' in parts:return 'hf_cache_review'
  if name=='resume.pt' or name.startswith('resume') and ext in BINARY or name in {'optimizer.pt','trainer_state.pt'}:return 'hf_training_resume'
  if len(parts)>1 and parts[1]=='e7' and ext=='.ckpt':return 'baseline_checkpoint_review'
  if ext in {'.pt','.pth','.ckpt','.safetensors'}:
   if name in {'prior.pt','best.pt','last.pt','model.pt','g.pt','p.pt','checkpoint.pt','g_gaussian.pt','g_history.pt','initial.pt'} or name.startswith(('checkpoint','ckpt')):return 'hf_model_checkpoints'
   return 'hf_tensor_artifacts_review'
  if ext in {'.pkl','.npz','.npy'}:return 'hf_tensor_artifacts_review'
  if ext=='.log' or 'events.out.tfevents.' in name:return 'hf_logs_private'
  if ext in TEXT:return 'github_experiment_metadata'
  return 'hf_other_artifacts_review'
 if top=='verification':return 'github_verification' if ext in TEXT else 'local_verification_binary'
 if top in SOURCE_DIRS or top in {'README.md','SOURCE.md','LOG.md','.gitignore','environment.yml','requirements.txt','requirements-dev.txt','requirements-vis.txt','requirements-release.txt','requirements-reproduction-lock.txt','pyproject.toml'}:return 'github_source'
 return 'manual_review'


def aggregate(rows,key):
 d={}
 for r in rows:
  k=r[key];x=d.setdefault(k,{'files':0,'logical_bytes':0,'allocated_bytes':0,'symlinks':0})
  x['files']+=r['kind']=='file';x['symlinks']+=r['kind']=='symlink'
  if r['kind']=='file':x['logical_bytes']+=r['size_bytes'];x['allocated_bytes']+=r['allocated_bytes']
 return d


def write_release_plan_manifests(out, rows, root):
 """Suggest destinations only; these reports are not an approved upload list."""
 def is_formal_g(path):
  return bool(re.fullmatch(r'exp/(egorecover_g_pretraining/v1/T[0-3]|egorecover_observation_reference/v2/R[0-3])/seed6[234]/best\.pt', path))
 core = {f'exp/egorecover_observation_reference/v2/R0/seed{s}/best.pt' for s in (62, 63, 64)}
 core.add('exp/egorecover_g_pretraining/v1/prior/prior.pt')
 def route(r):
  c, p = r['category'], r['path']; name = Path(p).name
  if r['kind']=='symlink':return 'restore_alias', 'restore_only', 'shared target stored once'
  if c in {'github_source','github_verification','github_experiment_metadata','hf_source_snapshots'}:
   return 'github:sxh-kk/EgoRecover', 'review_text_and_provenance', 'preserve relative path; portable copies must not replace immutable identities'
  if c=='hf_cache_review' and r['extension'] in TEXT and name in {'manifest.json','split.json','dev_monitor.json','report.json'} and '/episodes/' not in p:
   return 'github:sxh-kk/EgoRecover', 'review_metadata', 'split/cache provenance metadata; review for embedded observations'
  if c in {'hf_other_artifacts_review','local_verification_binary'} and r['extension'] in {'.png','.pdf','.xml'}:
   return 'github:sxh-kk/EgoRecover', 'review_figure_or_report', 'pose visualizations require derived-data review'
  if c=='hf_model_checkpoints':
   diagnostic = name in {'initial.pt','last.pt'} or any(x in p for x in ('/verification/','/smoke/','interrupted')) or '/egorecover_observation_reference/v1/' in p
   if diagnostic:return 'hf-dataset:<HF_OWNER>/EgoRecover-experiment-archive', 'optional_archive_review', 'initial/last/smoke/failed/interrupted; never advertise as a final selected model'
   return 'hf-model:<HF_OWNER>/EgoRecover', 'review_payload_and_rights', 'core inference' if p in core else 'formal G ablation' if is_formal_g(p) else 'historical P/G checkpoint; verify completion and identity'
  if c=='hf_training_resume':return 'hf-dataset:<HF_OWNER>/EgoRecover-experiment-archive', 'optional_archive_review', 'optimizer/EMA/RNG required for continuation, not inference'
  if c in {'hf_cache_review','hf_tensor_artifacts_review','derived_dataset_review'}:
   return 'hf-dataset:<HF_OWNER>/EgoRecover-experiment-archive', 'hold_for_data_rights_review', 'may contain GT, observations or source-derived features; private visibility does not confer redistribution rights'
  if c in {'hf_logs_private','hf_other_artifacts_review'}:
   return 'hf-dataset:<HF_OWNER>/EgoRecover-experiment-archive', 'optional_private_archive_review', 'review local paths, identifiers and provenance'
  if c=='baseline_checkpoint_review':return 'external_reference_or_hf_model_after_review', 'hold_for_checkpoint_provenance', 'E7 EMA stored in optimizer_states; dedicated export required'
  if c=='external_dataset_reference':return 'official_download_reference', 'do_not_duplicate_by_default', 'record upstream URL and restore path'
  if c=='restricted_smplx':return 'licensed_user_download', 'exclude_from_github_and_hf', 'SMPL-X redistribution requires prior permission'
  if c=='personal_review':return 'local_or_optional_document', 'manual_personal_review', 'CV/attachments are not required to reproduce code'
  if c in {'local_git','local_ephemeral','local_credentials','local_audit'}:return 'local_only', 'exclude', 'exclude Git objects, credentials and transient files'
  return 'local_review', 'manual_review', 'unclassified or synthetic verification binary'
 planned=[]
 for r in rows:
  target,status,note=route(r)
  planned.append({'path':r['path'],'kind':r['kind'],'category':r['category'],'size_bytes':r['size_bytes'],'proposed_destination':target,'review_status':status,'priority':'core' if r['path'] in core else 'formal_g_best' if is_formal_g(r['path']) else 'standard','planned_destination_path':r['path'],'sha256_if_computed':r['sha256'],'note':note})
 with (out/'release-manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(planned[0]));w.writeheader();w.writerows(planned)
 github=[r for r in planned if r['kind']=='file' and r['proposed_destination']=='github:sxh-kk/EgoRecover']
 with (out/'github-candidates.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(planned[0]));w.writeheader();w.writerows(github)
 weights=[r for r in planned if r['category'] in {'hf_model_checkpoints','baseline_checkpoint_review'} and r['kind']=='file']
 (out/'model-release-index.json').write_text(json.dumps({'status':'planning_only_not_upload_authorization','core_inference_paths':sorted(core),'core_inference_bytes':sum(r['size_bytes'] for r in rows if r['path'] in core),'formal_g_best_count':sum(is_formal_g(r['path']) for r in rows),'formal_g_best_bytes':sum(r['size_bytes'] for r in rows if is_formal_g(r['path'])),'github_candidate_files':len(github),'github_candidate_bytes':sum(r['size_bytes'] for r in github),'checkpoint_files':weights},ensure_ascii=False,indent=2)+'\n')
 aliases=[]
 for r in rows:
  if r['kind']=='symlink':
   target=Path(r['resolved_target'])
   aliases.append({'path':r['path'],'canonical_target':target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target),'relative_link_target':os.path.relpath(target,root/Path(r['path']).parent),'exists':r['target_exists']})
 (out/'symlink-restore.json').write_text(json.dumps({'status':'proposed_restore_map_no_links_changed','aliases':aliases},ensure_ascii=False,indent=2)+'\n')


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 started=datetime.now(ZoneInfo('Asia/Singapore')).isoformat();rows=[];errors=[];issues=[];links=[];seen={};sha_count=0
 def visit(folder):
  nonlocal sha_count
  try:entries=sorted(os.scandir(folder),key=lambda e:e.name)
  except OSError as e:errors.append({'path':str(folder.relative_to(root)),'error':str(e)});return
  for ent in entries:
   p=Path(ent.path)
   if p==out or out in p.parents:continue
   rel=p.relative_to(root).as_posix()
   try:s=ent.stat(follow_symlinks=False)
   except OSError as e:errors.append({'path':rel,'error':str(e)});continue
   if stat.S_ISDIR(s.st_mode):visit(p);continue
   kind='symlink' if stat.S_ISLNK(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'special'
   rec={'path':rel,'kind':kind,'category':category(rel),'top_directory':Path(rel).parts[0] if len(Path(rel).parts)>1 else '(root)','extension':p.suffix.lower() or '(none)','size_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'symlink_target':'','resolved_target':'','target_exists':'','sha256':''}
   if kind=='symlink':
    rec['symlink_target']=os.readlink(p)
    try:target=p.resolve();rec['resolved_target']=str(target);rec['target_exists']=target.exists();inside=target.is_relative_to(root)
    except (OSError,RuntimeError) as e:inside=False;rec['target_exists']=False;errors.append({'path':rel,'error':str(e)})
    links.append({'path':rel,'target':rec['symlink_target'],'resolved_target':rec['resolved_target'],'exists':rec['target_exists'],'internal':inside})
   elif kind=='file':
    seen.setdefault((s.st_dev,s.st_ino),{'size':s.st_size,'allocated':s.st_blocks*512,'paths':[]})['paths'].append(rel)
    # Inspect modest text files. Never print matched credential contents.
    if s.st_size<=16*1024**2 and (p.suffix.lower() in TEXT or p.name in {'.gitignore','config'}) and not rel.startswith('.git/objects/'):
     try:
      data=p.read_bytes();text=data.decode('utf-8',errors='replace');labels=[k for k,v in KEY_PATTERNS.items() if v.search(text)]
      if labels:issues.append({'path':rel,'type':'possible_secret','patterns':labels})
      n=len(PATH_RE.findall(text))
      if n:issues.append({'path':rel,'type':'absolute_paths','occurrences':n})
      if rec['category'].startswith('github_'):
       rec['sha256']=hashlib.sha256(data).hexdigest();sha_count+=1
     except OSError as e:errors.append({'path':rel,'error':str(e)})
   rows.append(rec)
 visit(root)
 fields=list(rows[0]) if rows else []
 def write_csv(name,selected):
  with (out/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
 write_csv('files.csv',rows)
 write_csv('github-source.csv',[r for r in rows if r['kind']=='file' and r['category'] in {'github_source','github_verification'}])
 write_csv('experiment-metadata.csv',[r for r in rows if r['kind']=='file' and r['category']=='github_experiment_metadata'])
 write_csv('model-checkpoints.csv',[r for r in rows if r['kind']=='file' and r['category'] in {'hf_model_checkpoints','baseline_checkpoint_review'}])
 write_csv('training-resume.csv',[r for r in rows if r['kind']=='file' and r['category']=='hf_training_resume'])
 write_csv('symlinks.csv',[r for r in rows if r['kind']=='symlink'])
 summary={'started_at':started,'finished_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'root':str(root),'scope':'All regular files/symlinks under root including .git; no symlink traversal; excludes this audit output. No uploads, archive construction, file deletion or checkpoint loading.','file_count':sum(r['kind']=='file' for r in rows),'symlink_count':len(links),'path_logical_bytes':sum(r['size_bytes'] for r in rows if r['kind']=='file'),'unique_inode_logical_bytes':sum(v['size'] for v in seen.values()),'unique_inode_allocated_bytes':sum(v['allocated'] for v in seen.values()),'by_category':aggregate(rows,'category'),'by_top_directory':aggregate(rows,'top_directory'),'by_extension':aggregate(rows,'extension'),'top_30_files':sorted([{'path':r['path'],'size_bytes':r['size_bytes'],'category':r['category']} for r in rows if r['kind']=='file'],key=lambda r:r['size_bytes'],reverse=True)[:30],'hardlink_groups':[v for v in seen.values() if len(v['paths'])>1],'links':links,'possible_secret_findings':sum(x['type']=='possible_secret' for x in issues),'absolute_path_files':sum(x['type']=='absolute_paths' for x in issues),'small_candidate_sha256_files':sha_count,'errors':errors,'classification_note':'Destination suggestions only, based on paths/extensions. Binary artifacts can embed GT/observations/licensed assets; public release requires payload/provenance review. Categories are mutually exclusive; content deduplication was not run.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 (out/'content-review.json').write_text(json.dumps({'findings':issues,'note':'Credential values are never recorded. Pattern scan of UTF-8 text <=16MiB, not a security guarantee. Binary payloads and Git object histories were not inspected.'},ensure_ascii=False,indent=2)+'\n')
 experiment_rows=collections.defaultdict(list)
 for r in rows:
  parts=Path(r['path']).parts
  if len(parts)>2 and parts[0]=='exp':experiment_rows[parts[1]].append(r)
 exps={k:{'file_count':sum(x['kind']=='file' for x in v),'logical_bytes':sum(x['size_bytes'] for x in v if x['kind']=='file'),'by_category':aggregate(v,'category')} for k,v in sorted(experiment_rows.items())}
 (out/'experiments.json').write_text(json.dumps(exps,ensure_ascii=False,indent=2)+'\n')
 write_release_plan_manifests(out,rows,root)
 print(json.dumps({k:summary[k] for k in ['finished_at','file_count','symlink_count','path_logical_bytes','unique_inode_logical_bytes','unique_inode_allocated_bytes','possible_secret_findings','absolute_path_files','errors']},ensure_ascii=False))
 for k,v in sorted(summary['by_category'].items()):print(k,v['files'],round(v['logical_bytes']/1024**3,6),'GiB',v['symlinks'],'links')

if __name__=='__main__':main()
