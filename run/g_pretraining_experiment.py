"""Persistent data -> P calibration -> 12 G runs -> held-out evaluation queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from run.e7_ablation import write_json


ROOT=Path(__file__).resolve().parents[1]


def log(title,lines):
    stamp=datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d--%H：%M')
    with (ROOT/'LOG.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(f'\n## {stamp}：{title}\n\n'+''.join('- '+s+'\n' for s in lines))


def complete(path,**required):
    if not path.exists():return False
    value=json.loads(path.read_text())
    return value.get('completed') is True and all(value.get(k)==v for k,v in required.items())


def stage(output,name,jobs):
    running=[];env=os.environ.copy()
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SMPLX_MODEL_PATH=str(ROOT/'body_models/smplx'))
    env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08'
    for label,gpu,module,arguments,report,required in jobs:
        if complete(report,**required):
            print('REUSE',label,flush=True);continue
        logfile=output/'jobs'/f'{label}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
        while True:
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                                 env=env,text=True,capture_output=True,check=True)
            free={int(line.split(',')[0]):int(line.split(',')[1]) for line in query.stdout.strip().splitlines()}
            if free[gpu]>=24576:break
            write_json(output/'queue.json',{'phase':'waiting_for_memory','next_stage':name,'gpu':gpu,
                                          'free_mib':free[gpu],'controller_pid':os.getpid()})
            time.sleep(30)
        e={**env,'CUDA_VISIBLE_DEVICES':str(gpu)}
        cmd=[sys.executable,'-u','-m',module,*arguments]
        with logfile.open('a') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,env=e,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        running.append((label,process,report,required))
        print('START',label,'gpu',gpu,'pid',process.pid,flush=True)
    while running:
        write_json(output/'queue.json',{'phase':name,'controller_pid':os.getpid(),
            'updated_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
            'jobs':[{'name':label,'pid':p.pid,'running':p.poll() is None} for label,p,_,_ in running]})
        for job in list(running):
            label,p,report,required=job
            rc=p.poll()
            if rc is not None:
                if rc!=0 or not complete(report,**required):
                    raise RuntimeError(f'{label} failed (exit={rc}); see jobs/{label}.log. Other launched jobs may still be running.')
                running.remove(job);print('DONE',label,flush=True)
        if running:time.sleep(10)


def summarize(output,cfg):
    import torch
    summaries={};diffs={};best_models={};short_horizons={}
    for group in cfg['groups']:
        seeds=[]
        for seed in cfg['seeds']:
            draws=[]
            for draw in cfg['draws']:
                p=torch.load(output/group/f'seed{seed}/test/last_nfe10_draw{draw}.pt',weights_only=True)
                draws.append(p['summary']['scores']['body_mm']['per_take'])
            takes=sorted(draws[0]);per={t:sum(d[t] for d in draws)/len(draws) for t in takes}
            seeds.append({'seed':seed,'per_take':per,'mean':sum(per.values())/len(per)})
        values=torch.tensor([s['mean'] for s in seeds])
        summaries[group]={'mean':float(values.mean()),'std':float(values.std()),'seeds':seeds}
    pairs=[('T1','T0'),('T2','T0'),('T3','T2'),('T3','T1')]
    generator=torch.Generator().manual_seed(20260929)
    for a,b in pairs:
        per=[];seed_deltas=[]
        for x,y in zip(summaries[a]['seeds'],summaries[b]['seeds']):
            if x['per_take'].keys()!=y['per_take'].keys():raise ValueError('Unpaired evaluation.')
            per.append([x['per_take'][t]-y['per_take'][t] for t in sorted(x['per_take'])])
            seed_deltas.append(x['mean']-y['mean'])
        values=torch.tensor(per).mean(0)
        samples=values[torch.randint(len(values),(10000,len(values)),generator=generator)].mean(1)
        diffs[a+'-'+b]={'mean':float(values.mean()),'seed_deltas':seed_deltas,'take_bootstrap95':torch.quantile(samples,torch.tensor([.025,.975])).tolist()}
    interaction=diffs['T3-T2']['mean']-diffs['T1-T0']['mean']
    for group in cfg['groups']:
        best_models[group]={}
        for nfe in cfg['nfe']:
            values=[];corrections=[]
            for seed in cfg['seeds']:
                for draw in cfg['draws']:
                    p=torch.load(output/group/f'seed{seed}/test/best_nfe{nfe}_draw{draw}.pt',weights_only=True)
                    values.append(p['summary']['scores']['body_mm']['mean'])
                    corrections.append(p['summary']['scores']['p_to_g_improvement_mm']['mean'])
            best_models[group][str(nfe)]={'body_mm':sum(values)/len(values),'p_to_g_improvement_mm':sum(corrections)/len(corrections)}
        cells={}
        for seed in cfg['seeds']:
            p=torch.load(output/group/f'seed{seed}/test/auxiliary.pt',weights_only=True)
            for row in p['rows']:
                key=f"{row['history']}_h{row['horizon']}"
                cells.setdefault(key,{}).setdefault(row['take'],[]).append(row['body_mm'])
        short_horizons[group]={k:sum(sum(v)/len(v) for v in takes.values())/len(takes) for k,takes in cells.items()}
    baseline=json.loads((output/'evaluation/e7_baseline.json').read_text())
    summary={'completed':True,'same_budget_last_nfe10':summaries,'paired_differences':diffs,'interaction_mm':interaction,
             'selected_models_by_nfe':best_models,'fixed_history_short_rollouts':short_horizons,
             'e7_reference_draw1062':baseline['scores'],
             'limitations':'New project holdout from official val; upstream E7 used official val for validation. Known-floor, fixed bootstrap beta, planar online reference.'}
    write_json(output/'summary.json',summary)
    lines=['# G四组训练结果','', '| 组别 | Body MPJPE，三seed均值±标准差（mm） |','|---|---:|']
    for group,v in summaries.items():lines.append(f'| {group} | {v["mean"]:.3f} ± {v["std"]:.3f} |')
    lines+=['','同预算末点、NFE10、三draw平均；主要时段t40…199。','']
    conclusions=[]
    for label,v in diffs.items():
        lo,hi=v['take_bootstrap95'];seeds=v['seed_deltas']
        verdict=('支持降低误差' if hi<0 and all(d<0 for d in seeds) else
                 '一致增加误差' if lo>0 and all(d>0 for d in seeds) else '方向或区间不足以支持稳定收益')
        conclusions.append(f'{label} {v["mean"]:+.3f}mm，{verdict}')
        lines.append(f'- {label}: {v["mean"]:+.3f}mm；{verdict}；take配对95%区间{v["take_bootstrap95"]}；三seed差值{seeds}。')
    lines += [f'- 交互：{interaction:+.3f}mm；负值表示混合历史下History起点更有利。',
              '- 区间含0或seed方向不一致时，证据不足；此表不单独识别共同架构改动的贡献。',
              '- dev最优点、低NFE、P→G修正及逐take数据保存在各组test目录；本表仅用于同预算主对照。',
              '',summary['limitations'],'','## dev选中模型：少步推理与纠错','',
              '| 组别 | NFE10 | NFE5 | NFE3 | NFE1 | NFE10的P→G改善（mm） |',
              '|---|---:|---:|---:|---:|---:|']
    for g,values in best_models.items():
        lines.append(f'| {g} | '+ ' | '.join(f'{values[str(n)]["body_mm"]:.3f}' for n in (10,5,3,1))+f' | {values["10"]["p_to_g_improvement_mm"]:+.3f} |')
    lines+=['','上表为三seed×三draw平均；正的P→G改善表示平均单帧误差降低。低NFE是同一选中权重的推理对照。',
            '', '## 固定历史与短闭环','', '| 组别 | GT历史下一帧 | 教师历史下一帧 | 0.4秒闭环 | 0.8秒闭环 |',
            '|---|---:|---:|---:|---:|']
    for g,values in short_horizons.items():
        lines.append(f'| {g} | '+' | '.join(f'{values[k]:.3f}' for k in ('GT_h1','teacher_h1','teacher_h4','teacher_h8'))+' |')
    lines+=['','单位mm；短闭环从相同教师历史开始，之后各组提交自身输出。',
            '',f'共享E7参照（draw1062）：公共启动β {baseline["scores"]["common_bootstrap"]["mean"]:.3f}mm；原生窗口β {baseline["scores"]["native_window"]["mean"]:.3f}mm。它只有一个draw，不直接作上述三draw均值的配对显著性结论。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    relative=output.relative_to(ROOT) if output.is_absolute() else output
    log('G四组重新训练与留出集评价完成',[
        '；'.join(f'{g} Body {v["mean"]:.3f}±{v["std"]:.3f}mm' for g,v in summaries.items()),
        '；'.join(conclusions)+f'；交互{interaction:+.3f}mm。',
        '；'.join(f'{g}选中模型NFE10的P→G平均改善{v["10"]["p_to_g_improvement_mm"]:+.3f}mm' for g,v in best_models.items())+'；短闭环和低NFE结果见完整表。',
        f'完整结果：[RESULTS.md]({relative}/RESULTS.md)；[配对统计]({relative}/summary.json)。评价范围与限制见结果文件。'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'exp/egorecover_g_pretraining/v1')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((out/'plan.json').read_text());output_args=['--output',str(out)]
    check=json.loads((out/'verification/smoke.json').read_text())
    if not check.get('passed'):raise ValueError('Pass real-data smoke before launching this queue.')
    from egorecover.evaluation_protocol import file_sha256
    for path,digest in check['code_sha256'].items():
        if file_sha256(ROOT/path)!=digest:raise ValueError(f'Code changed after smoke: {path}')
    try:
        stage(out,'preparing_paired_histories',[
            (f'prepare_gpu{gpu}',gpu,'run.g_pretraining_data',['prepare',*output_args,'--shard',str(i),'--shards','4'],out/f'data/shard{i}.done.json',{'limited':False})
            for i,gpu in enumerate(cfg['gpus'])])
        write_json(out/'data/report.json',{'completed':True,'shards':4})
        stage(out,'calibrating_shared_P',[
            ('prior_calibration',5,'run.train_g_pretraining',['prior',*output_args],out/'prior/report.json',{'smoke':False})])
        rel=out.relative_to(ROOT)
        log('G四组正式梯度训练阶段启动',[
            '配对数据准备、坐标核对、真实片段试跑及共享P协议校准已通过。T0/T1/T2/T3按GPU4/5/6/7调度，三个seed逐轮运行，每组24,000步。',
            f'进度：[queue.json]({rel}/queue.json)；共享P校准：[report.json]({rel}/prior/report.json)。此时尚无四组精度结论。'])
        for seed in cfg['seeds']:
            stage(out,f'training_seed{seed}',[
                (f'{group}_seed{seed}',gpu,'run.train_g_pretraining',['train',*output_args,'--group',group,'--seed',str(seed)],
                 out/group/f'seed{seed}/report.json',{'smoke':False}) for group,gpu in zip(cfg['groups'],cfg['gpus'])])
        # All checkpoint decisions are now frozen before opening test metrics.
        stage(out,'shared_e7_reference',[
            ('e7_reference',5,'run.train_g_pretraining',['baseline',*output_args],out/'evaluation/e7_baseline.json',{})])
        for seed in cfg['seeds']:
            stage(out,f'heldout_evaluation_seed{seed}',[
                (f'eval_{group}_seed{seed}',gpu,'run.train_g_pretraining',['evaluate',*output_args,'--group',group,'--seed',str(seed)],
                 out/group/f'seed{seed}/test/report.json',{}) for group,gpu in zip(cfg['groups'],cfg['gpus'])])
        summarize(out,cfg);write_json(out/'queue.json',{'phase':'completed','completed':True,'controller_pid':os.getpid()})
    except Exception as error:
        write_json(out/'queue.json',{'phase':'failed','controller_pid':os.getpid(),'error':str(error)})
        log('G四组队列中断',[str(error),f'状态：[queue.json]({out.relative_to(ROOT)}/queue.json)。已完成产物保留，未生成完整结论。'])
        raise


if __name__=='__main__':main()
