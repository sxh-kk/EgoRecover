"""Split large release payloads into content-addressed, exact byte chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SIZE=64*1024*1024


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--staging',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();stage=args.staging.resolve()
    if stage.is_relative_to(ROOT):raise ValueError('Use an independent staging directory.')
    manifest=json.loads((ROOT/'docs/release/artifacts.json').read_text());jobs=[]
    for f in manifest['files']:
        if f['repo_type']=='model' and f['bytes']>SIZE and 'parts' not in f:jobs.append((f,ROOT/f['path'],stage/'model'))
    for a in manifest.get('archives',[]):
        if a.get('role')=='lossless_prepared_data_bundle' and a['bytes']>SIZE and 'parts' not in a:jobs.append((a,stage/'dataset'/a['path'],stage/'dataset'))
    def split(job):
        record,source,folder=job;parts=[];whole=hashlib.sha256()
        with source.open('rb') as f:
            while True:
                block=f.read(SIZE)
                if not block:break
                whole.update(block);sha=hashlib.sha256(block).hexdigest();relative=f'parts/{sha}.bin';dest=folder/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                # Publish only complete parts; interrupted temporary writes are
                # never mistaken for reusable content-addressed cache entries.
                with tempfile.NamedTemporaryFile(dir=dest.parent, prefix='.part-', delete=False) as out:
                    temporary=Path(out.name);out.write(block)
                try:
                    os.link(temporary,dest)
                except FileExistsError:
                    if dest.stat().st_size!=len(block) or hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:
                        raise ValueError('Existing transport part is incomplete or changed: '+relative)
                finally:
                    temporary.unlink(missing_ok=True)
                parts.append({'path':relative,'bytes':len(block),'sha256':sha})
        if whole.hexdigest()!=record['sha256'] or sum(x['bytes'] for x in parts)!=record['bytes']:raise ValueError('Payload changed while splitting: '+record['path'])
        print(record['path'],len(parts),'parts',flush=True)
        return record,parts,folder
    with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(split,jobs))
    for record,parts,folder in results:
        record['parts']=parts
        direct=folder/record['path']
        if direct.exists():direct.unlink()
    (ROOT/'docs/release/artifacts.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for kind in ['model','dataset']:
        files=[f for f in manifest['files'] if f['repo_type']==kind]
        d={'schema_version':manifest['schema_version'],'github_repo':manifest['github_repo'],'repository':manifest['repositories'][kind],'files':files,'aliases':manifest['aliases'],'archives':manifest.get('archives',[]) if kind=='dataset' else [],'required_local_assets':manifest.get('required_local_assets',[]),'note':'Large payloads use 64MiB parts. Exact final revisions pinned in GitHub.'}
        (stage/kind/'release-index.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        payloads=list((stage/kind/'parts').glob('*.bin'))
        print(kind,'unique large-payload parts',len(payloads),'GiB',sum(x.stat().st_size for x in payloads)/1024**3,flush=True)
    print('Exact source hashes verified; nothing uploaded.',flush=True)


if __name__=='__main__':main()
