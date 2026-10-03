"""D only: 192 train takes, up to four clips, three seeds and 9600 updates."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import ROOT, Queue, Task, command, append_log, now
from run.prior_cv_residual_experiment import paired_interval

SEEDS=(62,63,64)
BASE=ROOT/'exp/egorecover_prior_cv_residual/v1/train'


def collect_results(tasks):
    expanded={s:json.loads(t.report.read_text()) for s,t in zip(SEEDS,tasks)}
    baseline={s:json.loads((BASE/f'V0_cv_dense_s{s}'/'report.json').read_text()) for s in SEEDS}
    for s in SEEDS:
        a,b=expanded[s],baseline[s]
        for key in ('stats_sha256','smplx_asset_sha256','reference_mode','base_mode','geometry_weight','fk_weight','batch_size','lr','weight_decay','eval_every'):
            if a[key]!=b[key]: raise ValueError(f'Baseline protocol differs: {key}')
        if a['original_cache_sha256']!=b['data_sha256']: raise ValueError('Original dev cache differs.')
        x=torch.load(tasks[SEEDS.index(s)].output/'initial.pt',weights_only=True,map_location='cpu')['state_dict']
        y=torch.load(BASE/f'V0_cv_dense_s{s}'/'initial.pt',weights_only=True,map_location='cpu')['state_dict']
        if set(x)!=set(y) or any(not torch.equal(x[k],y[k]) for k in x): raise ValueError('Seed initialization differs.')
    metrics=list(expanded[62]['results']['prior']['single'])
    aggregates={}; differences={}
    for label,reports in [('baseline',baseline),('D',expanded)]:
        aggregates[label]={k:{'mean':statistics.mean(r['results']['prior']['single'][k]['mean'] for r in reports.values()),
            'std':statistics.stdev(r['results']['prior']['single'][k]['mean'] for r in reports.values())} for k in metrics}
    for k in metrics:
        per_seed={}; takes={}
        for s in SEEDS:
            a=expanded[s]['results']['prior']['single'][k]['per_take'];b=baseline[s]['results']['prior']['single'][k]['per_take']
            if set(a)!=set(b): raise ValueError('Dev take sets differ.')
            per_seed[str(s)]=statistics.mean(a[t]-b[t] for t in a)
            for t in a: takes.setdefault(t,[]).append(a[t]-b[t])
        ds=[statistics.mean(takes[t]) for t in sorted(takes)]
        differences[k]={'mean':statistics.mean(ds),'by_seed':per_seed,'take_bootstrap_95ci':paired_interval(ds)}
    v=aggregates['D'];d=differences['fk22_mm'];ci=d['take_bootstrap_95ci']
    reliable=all(x<0 for x in d['by_seed'].values()) and ci[1]<0
    conclusions=[f"D组Body MPJPE {v['fk22_mm']['mean']:.3f}±{v['fk22_mm']['std']:.3f} mm，PA {v['pa_mpjpe_mm']['mean']:.3f} mm，统一体型 {v['same_shape_mpjpe_mm']['mean']:.3f} mm。",
        f"相对原48take/2400步：Body变化{d['mean']:+.3f} mm，逐take配对95%区间[{ci[0]:+.3f},{ci[1]:+.3f}]；三个seed差值"+'/'.join(f'{x:+.3f}' for x in d['by_seed'].values())+' mm。',
        f"三个seed均改善且配对区间低于0：{'达到' if reliable else '未达到'}。",
        '本轮同时增加take、片段与训练步数，只能评价组合收益，无法拆分各因素贡献；dev参与多轮选模，holdout和接入G收益尚未验证。']
    return {'completed':True,'aggregates':aggregates,'differences':differences,'conclusions':conclusions,
        'selected_steps':{str(s):r['selected_step'] for s,r in expanded.items()},
        'fixed_budget_dev':{str(s):{str(p['step']):p['dev'] for p in r['curve'] if p['step'] in (2400,4800,9600)} for s,r in expanded.items()},
        'final_single':{str(s):r['final_single'] for s,r in expanded.items()},'holdout_used':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args();o=args.output
    queue=Queue(o,[4,5,6,7],15,allow_shared=True,minimum_free_gib=12)
    prep=Task('prepare_data',command('run.prepare_prior_expanded',output=o/'data',shards=o/'data_shards'),o/'data','expanded_data')
    trains=[Task(f'D_s{s}',command('run.train_prior_expanded',data=o/'data',output=o/'train'/f'D_s{s}',seed=s),
                 o/'train'/f'D_s{s}','prior',deps=['prepare_data']) for s in SEEDS]
    tasks=[prep,*trains]
    plan={'group':'D','train_takes':192,'max_nonoverlapping_clips_per_take':4,'steps':9600,'seeds':list(SEEDS),
          'gpus':[4,5,6,7],'geometry_weight':1,'fk_weight':0,'velocity_input':False,'base_mode':'constant_velocity',
          'dev':'exact original 12 takes and cached tensors','holdout_used':False,'method':'docs/experiments/p-data-budget.md'}
    if (o/'plan.json').exists() and json.loads((o/'plan.json').read_text())!=plan: raise ValueError('Plan changed.')
    atomic_json(o/'plan.json',plan)
    for t in tasks: queue.register(t)
    if args.retry_failed:
        for e in queue.state['tasks'].values():
            if e['status']=='failed': e['status']='pending'
    queue.save()
    try:
        if not queue.state.get('start_logged'):
            append_log('P D组扩量队列启动',[ '按用户决定只做D组，目标192个train take、每take最多4个独特20秒片段；seeds62/63/64，各9600步，GPU4–7。',
                '常速度残差、geometry=1、所有FK损失=0、不加显式速度；先审计数据再自动训练，当前尚无新效果结论。',
                f'计划：`{o / "plan.json"}`；队列：`{o / "queue.json"}`；完成后结果：[{o / "RESULTS.md"}]({o / "RESULTS.md"})。'])
            queue.state['start_logged']=True;queue.save()
        queue.run(tasks)
        result=collect_results(trains);atomic_json(o/'summary.json',result)
        rows=['# P D组扩量结果','','| 配置 | Body MPJPE | PA-MPJPE | 统一体型MPJPE |','|---|---:|---:|---:|']
        for label,m in result['aggregates'].items():
            rows.append(f"| {label} | {m['fk22_mm']['mean']:.3f} ± {m['fk22_mm']['std']:.3f} | {m['pa_mpjpe_mm']['mean']:.3f} | {m['same_shape_mpjpe_mm']['mean']:.3f} |")
        rows+=['','单位mm。','','## 结论','',*['- '+x for x in result['conclusions']],'','[完整统计与2400/4800/9600固定预算点](summary.json)']
        (o/'RESULTS.md').write_text('\n'.join(rows)+'\n')
        append_log('P D组扩量实验结论',[*result['conclusions'],f'结果：`{o / "RESULTS.md"}`；统计：`{o / "summary.json"}`。'])
        queue.state.update(status='complete',completed_at=now());queue.save()
    except Exception as error:
        queue.state.update(status='failed',error=repr(error));queue.save()
        append_log('P D组扩量队列异常',[f'{error!r}；状态：`{o / "queue.json"}`。'])
        raise

if __name__=='__main__': main()
