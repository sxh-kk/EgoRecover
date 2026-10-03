"""Bundle selected small binary data artifacts, preserving exact file bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staging',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();stage=args.staging.resolve()
    if stage.is_relative_to(ROOT):raise ValueError('Use a staging directory outside the source workspace.')
    manifest=json.loads((ROOT/'docs/release/artifacts.json').read_text())
    g=json.loads((ROOT/'exp/egorecover_g_pretraining/v1/data/manifest.json').read_text());groups={r['episode_id']:r['group'] for r in g['records']}
    bins={}
    for f in manifest['files']:
        if f['repo_type']!='dataset' or 'archive' in f or Path(f['path']).suffix not in {'.pt','.pkl','.npz','.npy'} or f['bytes']<64*1024:continue
        path=f['path']
        if '/egorecover_g_pretraining/v1/data/episodes/' in path:kind='g-'+groups[Path(path).stem]
        elif f['role'] in {'final_prediction_or_metric_evidence'}:kind='final-evaluation'
        elif '/histories/' in path or '/replay_cache/' in path:kind='legacy-history'
        else:kind='prior-and-startup'
        bins.setdefault(kind,[]).append(f)
    jobs=[]
    for kind,files in bins.items():
        part=[];size=0;index=0
        for f in sorted(files,key=lambda x:x['path']):
            if part and size+f['bytes']>1024**3:
                jobs.append((kind,index,part));index+=1;part=[];size=0
            part.append(f);size+=f['bytes']
        if part:jobs.append((kind,index,part))
    print('Packing',sum(len(x[2]) for x in jobs),'files into',len(jobs),'bundles',flush=True)
    def pack(job):
        kind,index,files=job;relative=f'bundles/{kind}/part-{index:03d}.zip';path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for f in files:
                h=hashlib.sha256()
                with (ROOT/f['path']).open('rb') as src,z.open(f['path'],'w',force_zip64=True) as out:
                    for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);out.write(b)
                if h.hexdigest()!=f['sha256']:raise ValueError('Source changed before bundle: '+f['path'])
        record={'repo_id':'sxhkk/EgoRecover-data','repo_type':'dataset','path':relative,'bytes':path.stat().st_size,'sha256':digest(path),'member_count':len(files),'role':'lossless_prepared_data_bundle'}
        print(relative,len(files),round(record['bytes']/1024**2,2),'MiB',flush=True)
        return record,files
    with ThreadPoolExecutor(max_workers=args.workers) as pool:packed=list(pool.map(pack,jobs))
    for record,files in packed:
        manifest.setdefault('archives',[]).append(record)
        for f in files:
            f['archive']={'path':record['path'],'member':f['path']}
            path=stage/f['path']
            if path.exists():path.unlink()
    (ROOT/'docs/release/artifacts.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (stage/'release-index.json').write_text(json.dumps({'schema_version':manifest['schema_version'],'github_repo':manifest['github_repo'],'repository':manifest['repositories']['dataset'],'files':[f for f in manifest['files'] if f['repo_type']=='dataset'],'aliases':manifest['aliases'],'archives':manifest['archives'],'note':'Exact final revisions pinned in GitHub.'},ensure_ascii=False,indent=2)+'\n')
    print('Bundle source hashes verified; originals preserved. No uploads.',flush=True)


if __name__=='__main__':main()
