"""Run the four-cell E7 audit: prepare completion, parity, smoke, GPUs 4-7, summary."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'exp/e7_minimal_ablation/v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation-pid',type=int)
    a=p.parse_args();os.chdir(ROOT);OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=OUTPUT/'jobs';jobs.mkdir(exist_ok=True)
    state={'controller_pid':os.getpid(),'status':'starting','gpus':[4,5,6,7],'workers':[]}

    def update(status):
        state['status']=status
        state['updated_at']=datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
        tmp=OUTPUT/'queue.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUTPUT/'queue.json')
        print(state['updated_at'],status,flush=True)

    def run_job(name, args, gpu=None):
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (jobs/f'{name}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','run.e7_ablation',*args],
                                  stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        return proc

    try:
        if a.preparation_pid:
            update('waiting_for_data_preparation')
            while Path(f'/proc/{a.preparation_pid}/stat').exists():
                try:
                    if Path(f'/proc/{a.preparation_pid}/stat').read_text().split()[2]=='Z':break
                except FileNotFoundError:break
                time.sleep(2)
        for name,args,gpu in [('verify',['verify'],None),('smoke',['evaluate','--smoke'],4)]:
            update(name);proc=run_job(name,args,gpu);state['active_pid']=proc.pid;update(name)
            code=proc.wait()
            if code:raise RuntimeError(f'{name} failed with exit {code}; see jobs/{name}.log')
        workers=[]
        for shard,gpu in enumerate((4,5,6,7)):
            proc=run_job(f'shard{shard}',['evaluate','--shard',str(shard),'--shards','4'],gpu)
            workers.append(proc);state['workers'].append({'shard':shard,'gpu':gpu,'pid':proc.pid,'exit_code':None})
        state.pop('active_pid',None);update('formal_evaluation')
        while any(proc.poll() is None for proc in workers):
            for spec,proc in zip(state['workers'],workers):spec['exit_code']=proc.poll()
            update('formal_evaluation');time.sleep(20)
        for spec,proc in zip(state['workers'],workers):spec['exit_code']=proc.returncode
        if any(proc.returncode for proc in workers):raise RuntimeError('Formal shard failed; see jobs/shard*.log')
        update('summarize');proc=run_job('summarize',['summarize'])
        if proc.wait():raise RuntimeError('Summary failed; see jobs/summarize.log')
        update('completed')
    except Exception as exc:
        state['error']=str(exc);update('failed')
        title=datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d--%H：%M')
        with (ROOT/'LOG.md').open('a') as f:
            f.write(f'\n## {title}：E7四组对照执行中断\n\n- {exc}\n- 状态：[queue.json](exp/e7_minimal_ablation/v1/queue.json)；[运行目录](exp/e7_minimal_ablation/v1/jobs)。未生成完整四组结论。\n')
        raise


if __name__=='__main__':main()
