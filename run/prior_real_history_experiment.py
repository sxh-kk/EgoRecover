"""Queue the new-P-only actual-history evaluation and record paired conclusions."""
import argparse,json,statistics
from pathlib import Path
import torch
import numpy as np
from egorecover.evaluation_protocol import file_sha256,event_window
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import Queue,Task,command,append_log,now
from run.adapt_prior_on_predictions import save_torch
from run.prior_cv_residual_experiment import paired_interval


def score_cell(predicted,gt,cases,mask):
    # All seeds and generator draws share one set of independent take units.
    per_take={};per_seed=[];absolute={};original={};p95={}
    for take in sorted({r[0] for r in cases}):
        selected=[i for i,c in enumerate(cases) if c[0]==take]
        pa=[];pb=[];tails=[]
        for i in selected:
            m=mask[i]
            if not bool(m.any()):continue
            pa.append(predicted[:,i,m].double().mean(1));pb.append(gt[:,i,m].double().mean(1))
            tails.append(torch.quantile(predicted[:,i,m].double(),.95,dim=1))
        if not pa:continue
        a=torch.stack(pa).mean(0);b=torch.stack(pb).mean(0)
        absolute[take]=a.tolist();original[take]=b.tolist();per_take[take]=(a-b).tolist();p95[take]=torch.stack(tails).mean().item()
    if not per_take:return {'available':False}
    differences=[statistics.mean(v) for v in per_take.values()]
    A=statistics.mean(statistics.mean(v) for v in absolute.values());B=statistics.mean(statistics.mean(v) for v in original.values())
    return {'available':True,'E7_history':A,'GT_history':B,'difference':A-B,'relative_percent':100*(A-B)/B if B else None,
        'take_bootstrap_95ci':paired_interval(differences),'per_take_difference_by_seed':per_take,
        'difference_by_seed':[statistics.mean(v[s] for v in per_take.values()) for s in range(predicted.shape[0])],
        'E7_history_by_seed':[statistics.mean(v[s] for v in absolute.values()) for s in range(predicted.shape[0])],
        'per_take_E7_history':absolute,'per_take_GT_history':original,'per_take_frame_p95':p95}


def collect(output,plan,tasks):
    results=[]
    for t in tasks:
        r=json.loads(t.report.read_text())
        if not r['completed'] or r['artifact_sha256']!=file_sha256(t.output/'frames.pt'):raise ValueError('Evaluation artifact changed.')
        if file_sha256(r['histories'])!=r['history_sha256']:raise ValueError('History artifact changed.')
        results.append(torch.load(t.output/'frames.pt',weights_only=True,map_location='cpu'))
    cases=sum((r['cases'] for r in results),[]);records=sum((r['records'] for r in results),[])
    expected={(t,v,d) for t in plan['takes'] for v in plan['variants'] for d in plan['generator']['draw_seeds']}
    if len(cases)!=len(expected) or set(map(tuple,cases))!=expected:raise ValueError('Missing or duplicated history case.')
    times=torch.arange(20,200)
    if any(not torch.equal(r['times'],times) for r in results):raise ValueError('Target frames changed.')
    seeds=[str(x['training_seed']) for x in plan['predictors']]
    keys=list(results[0]['metrics'][seeds[0]])
    predicted={k:torch.stack([torch.cat([r['metrics'][s][k] for r in results]) for s in seeds]) for k in keys}
    references={}
    for spec in plan['existing_GT_results']:
        if file_sha256(spec['artifact'])!=spec['artifact_sha256']:raise ValueError('Original GT artifact changed.')
        x=torch.load(spec['artifact'],weights_only=True,map_location='cpu')['single']
        per_take={}
        for take in plan['takes']:
            idx=x['takes'].index(take);mask=x['indices'][:,0]==idx
            if not torch.equal(x['indices'][mask,1],times):raise ValueError('GT target frame order differs.')
            per_take[take]={k:x['metrics'][k][mask] for k in keys}
        references[str(spec['training_seed'])]=per_take
    gt={k:torch.stack([torch.stack([references[s][c[0]][k] for c in cases]) for s in seeds]) for k in keys}
    quality={k:torch.cat([r['history_quality'][k] for r in results]) for k in results[0]['history_quality']}
    aggregates={};phases={};strata={}
    for variant in plan['variants']:
        choose=torch.tensor([i for i,c in enumerate(cases) if c[1]==variant]);c=[cases[i] for i in choose.tolist()]
        a={k:v[:,choose] for k,v in predicted.items()};b={k:v[:,choose] for k,v in gt.items()}
        aggregates[variant]={}
        for label,begin in [('all',20),('post_startup',40),('full_window',100)]:
            mask=(times>=begin)[None].expand(len(c),-1)
            aggregates[variant][label]={k:score_cell(a[k],b[k],c,mask) for k in keys}
        phases[variant]={}
        # Clean phase boundaries are paired separately to each corrupted variant.
        for event_variant in (plan['variants'][1:] if variant=='clean' else [variant]):
            masks={k:[] for k in ('pre_fault','fault','recovery')}
            for idx in choose.tolist():
                record=next(r for r in records if r['base_take_name']==cases[idx][0] and r['variant_name']==event_variant)
                window=event_window(record)
                if window is None:raise ValueError('Missing corruption event window.')
                start,end=window
                masks['pre_fault'].append(times<start);masks['fault'].append((times>=start)&(times<end));masks['recovery'].append(times>=end)
            phases[variant][event_variant]={phase:{k:score_cell(a[k],b[k],c,torch.stack(ms)) for k in keys} for phase,ms in masks.items()}
        q=quality['fk22_mm'][choose];threshold=torch.quantile(q.flatten().double(),torch.tensor([1/3,2/3],dtype=torch.double))
        strata[variant]={'history_body_thresholds_mm':threshold.tolist(),'groups':{}}
        for label,mask in [('low',q<=threshold[0]),('middle',(q>threshold[0])&(q<=threshold[1])),('high',q>threshold[1])]:
            strata[variant]['groups'][label]={'frames':int(mask.sum()),'takes':len({c[i][0] for i in range(len(c)) if mask[i].any()}),
                'Body':score_cell(a['fk22_mm'],b['fk22_mm'],c,mask)}
    summary={'completed':True,'scope':plan['scope'],'holdout_used':False,'cases':len(cases),'target_pairs':len(cases)*180,
             'aggregates':aggregates,'phases':phases,'history_quality_strata':strata,'training_seeds':seeds,
             'interpretation':plan['interpretation']}
    conclusions=[]
    for variant in plan['variants']:
        m=aggregates[variant]['all'];b=m['fk22_mm'];ci=b['take_bootstrap_95ci']
        conclusions.append(f"{variant}：新P Body {b['E7_history']:.3f}mm，原GT历史 {b['GT_history']:.3f}mm，变化{b['difference']:+.3f}mm（{b['relative_percent']:+.1f}%），按take配对95%区间[{ci[0]:+.3f},{ci[1]:+.3f}]；PA {m['pa_mpjpe_mm']['E7_history']:.3f}mm，统一体型 {m['same_shape_mpjpe_mm']['E7_history']:.3f}mm。")
    conclusions+=['本轮只量化同一个新P更换历史输入后的变化，没有重新训练或评估旧P/CV/Hold；不能据此判断预测历史上替代方法的优劣。',
                  '历史来源为冻结原E7的因果窗口推理；不是训练后G闭环，且dev已参与先前选模，holdout未使用。']
    summary['conclusions']=conclusions
    (output/'evaluation').mkdir(exist_ok=True)
    save_torch(output/'evaluation/frames.pt',{'cases':cases,'times':times,'metrics':predicted,'GT_metrics':gt,'history_quality':quality})
    atomic_json(output/'summary.json',summary)
    rows=['# 新P实际历史适应性结果','','同一新P、相同dev与目标帧；三个P seed与三个E7 draw取均值。单位mm。','',
          '| 输入 | Body MPJPE | PA-MPJPE | 统一体型MPJPE | Body相对GT变化 |','|---|---:|---:|---:|---:|']
    ref=aggregates['clean']['all'];rows.append(f"| 原GT历史 | {ref['fk22_mm']['GT_history']:.3f} | {ref['pa_mpjpe_mm']['GT_history']:.3f} | {ref['same_shape_mpjpe_mm']['GT_history']:.3f} | — |")
    for variant in plan['variants']:
        m=aggregates[variant]['all'];rows.append(f"| E7/{variant} | {m['fk22_mm']['E7_history']:.3f} | {m['pa_mpjpe_mm']['E7_history']:.3f} | {m['same_shape_mpjpe_mm']['E7_history']:.3f} | {m['fk22_mm']['difference']:+.3f} |")
    rows+=['','## 结论','',*['- '+x for x in conclusions],'','[完整统计、时间子集及故障阶段](summary.json)','', '![历史质量与P误差](figures/history_quality.png)']
    (output/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,variant in zip(axes,plan['variants']):
        selected=torch.tensor([i for i,c in enumerate(cases) if c[1]==variant])
        x=quality['fk22_mm'][selected].flatten().numpy();y=predicted['fk22_mm'][:,selected].mean(0).flatten().numpy()
        ax.scatter(x,y,s=3,alpha=.15);ax.set(xlabel='E7 history Body MPJPE (mm)',ylabel='New P next-frame MPJPE (mm)',title=variant)
    fig.tight_layout();(output/'figures').mkdir(exist_ok=True);fig.savefig(output/'figures/history_quality.png',dpi=160);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('config/egorecover_prior_real_history_v1.json'))
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args();plan=json.loads(a.config.read_text());o=Path(plan['output'])
    queue=Queue(o,plan['gpus'],15,allow_shared=True,minimum_free_gib=16)
    path=o/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen execution plan differs.')
    atomic_json(path,plan)
    prep=Task('prepare_data',command('run.prepare_prior_real_history',plan=path,output=o/'data'),o/'data','real_history_data')
    smoke=Task('smoke',command('run.evaluate_prior_real_history',plan=path,data=o/'data',output=o/'smoke',work=o/'work/smoke',smoke=True),o/'smoke','real_history',deps=['prepare_data'])
    workers=[Task(f'eval_shard{i}',command('run.evaluate_prior_real_history',plan=path,data=o/'data',output=o/'shards'/f'shard{i}',work=o/'work'/f'shard{i}',shard=i,shards=4),o/'shards'/f'shard{i}','real_history',deps=['smoke']) for i in range(4)]
    tasks=[prep,smoke,*workers]
    for t in tasks:queue.register(t)
    if a.retry_failed:
        for r in queue.state['tasks'].values():
            if r['status']=='failed':r['status']='pending'
    queue.save()
    try:
        if not queue.state.get('start_logged'):
            append_log('新P实际历史评估启动（冻结E7，复用GT对照）',[
                '按用户指令启动：GPU4–7，先准备观测和2个train take工程检查，再并行生成12dev×3变体×3采样的因果E7历史，仅评估新P三个已选seed。旧P/CV/Hold不重跑，不训练P/G。',
                f'队列：`{o / "queue.json"}`；控制日志：`{o / "controller.log"}`；计划：`{path}`。结束后自动写入比较结论与结果链接。'])
            queue.state['start_logged']=True;queue.save()
        queue.run(tasks)
        result=collect(o,plan,workers)
        append_log('新P实际历史适应性结论（E7历史对照已有GT指标）',[*result['conclusions'],f'结果：`{o / "RESULTS.md"}`；完整统计：`{o / "summary.json"}`。'])
        queue.state.update(status='complete',completed_at=now());queue.save()
    except Exception as error:
        queue.state.update(status='failed',error=repr(error));queue.save()
        append_log('新P实际历史评估异常',[f'{error!r}；队列：`{o / "queue.json"}`。未完成结果不作正式结论。']);raise
if __name__=='__main__':main()
