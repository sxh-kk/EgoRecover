"""Verified four-GPU observation length x observed reference experiment queue."""
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
from run.e7_ablation import write_json
from run.g_pretraining_experiment import ROOT, complete, log, stage
from egorecover.evaluation_protocol import file_sha256

PROTOCOL='g-observation-reference-v1'
CODE=('run/train_observation_reference.py','egorecover/observation_reference.py',
      'model/observation_reference_g.py','run/observation_reference_experiment.py')


def verify_geometry(out, cfg):
    import torch
    from egorecover.codec import MotionCodec
    from egorecover.observation_reference import ReferenceSequences, GROUPS, decode
    from run.train_g_pretraining import load_prior
    errors={}
    for group,(length,anchored) in GROUPS.items():
        data=ReferenceSequences(out,'train',observation_length=length,anchored=anchored,limit=2)
        codec=MotionCodec(data.stats)
        prior=load_prior(out/'prior/prior.pt','cpu',manifest_sha=data.manifest_sha)
        indices=torch.tensor([[0,20],[1,40],[0,120],[1,199]])
        batch,y=data.batch(codec,prior,indices,torch.tensor([False,True,True,False]),'cpu')
        state=decode(codec,batch['target'][:,0],batch['reference'],anchored)
        error=float((state.joints[...,:3,3]-batch['target_joints']).abs().max())
        if error>5e-4:raise ValueError(f'Real geometry roundtrip failed: {group}: {error}')
        errors[group]=error
    return errors

def summarize(out, cfg):
    import torch
    groups = {}
    for group in cfg['groups']:
        seeds = []
        for seed in cfg['seeds']:
            p = torch.load(out / group / f'seed{seed}/test/best_nfe10_draw1062.pt',
                           map_location='cpu', weights_only=True)
            seeds.append({'seed': seed, 'scores': p['summary']['scores']})
        means = torch.tensor([s['scores']['body_mm']['mean'] for s in seeds], dtype=torch.float64)
        metrics = {k: sum(s['scores'][k]['mean'] for s in seeds) / len(seeds)
                   for k in seeds[0]['scores']}
        if not all(torch.isfinite(torch.tensor(list(metrics.values())))):
            raise ValueError(f'Nonfinite metrics in {group}; inspect failures before summarizing.')
        groups[group] = {'body_mean_mm': float(means.mean()), 'body_std_mm': float(means.std()),
                         'metrics': metrics, 'seeds': seeds}
    diffs = {}
    rng = torch.Generator().manual_seed(20260929)
    for a, b in [('R1', 'R0'), ('R2', 'R0'), ('R3', 'R2'), ('R3', 'R1'), ('R3', 'R0')]:
        paired = []
        for x, y in zip(groups[a]['seeds'], groups[b]['seeds']):
            px, py = x['scores']['body_mm']['per_take'], y['scores']['body_mm']['per_take']
            if x['seed'] != y['seed'] or set(px) != set(py):
                raise ValueError('Unpaired seeds or takes.')
            paired.append([px[t] - py[t] for t in sorted(px)])
        values = torch.tensor(paired, dtype=torch.float64)
        by_take = values.mean(0)
        samples = by_take[torch.randint(len(by_take), (10000, len(by_take)), generator=rng)].mean(1)
        diffs[a + '-' + b] = {'mean_mm': float(values.mean()),
            'seed_deltas_mm': values.mean(1).tolist(),
            'take_bootstrap95_mm': torch.quantile(samples, torch.tensor([.025, .975], dtype=torch.float64)).tolist()}
    baseline = json.loads((out / 'evaluation/e7_baseline.json').read_text())
    ref = baseline['scores']['common_bootstrap']['mean']
    summary = {'completed': True, 'protocol': PROTOCOL, 'checkpoint': 'dev-selected best',
        'nfe': 10, 'draw': 1062, 'groups': groups, 'paired_differences': diffs,
        'e7_reference': baseline['scores'],
        'g_minus_e7_mm': {g: v['body_mean_mm'] - ref for g, v in groups.items()},
        'omitted': ['multiple draws', 'low NFE', 'fixed history diagnostics'],
        'limitations': 'One sampling draw; dev-selected checkpoints; known floor and fixed bootstrap shape. '
                        'Repeated project benchmark already used in prior analysis; upstream E7 used official val for validation.'}
    write_json(out / 'summary.json', summary)
    lines = ['# G观测长度×设备参考四组结果', '',
        '222个本轮留出take，开发集选中的best模型，三个训练seed，单draw1062，Euler10；主时段t40…199。', '',
        '| 组别 | Body均值±种子标准差 | PA | Root | Head | P→G改善 | 相对E7差值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g, v in groups.items():
        m = v['metrics']
        lines.append(f'| {g} | {v["body_mean_mm"]:.3f} ± {v["body_std_mm"]:.3f} | '
            f'{m["pa_mm"]:.3f} | {m["root_mm"]:.3f} | {m["head_mm"]:.3f} | '
            f'{m["p_to_g_improvement_mm"]:+.3f} | {v["body_mean_mm"]-ref:+.3f} |')
    lines += ['', '单位mm。P→G为各组自身历史上的单步改善，不是独立P-only闭环对照。', '', '## 配对结论', '']
    verdicts = []
    for name, d in diffs.items():
        lo, hi = d['take_bootstrap95_mm']
        verdict = ('支持降低误差' if hi < 0 and all(x < 0 for x in d['seed_deltas_mm']) else
                   '一致增加误差' if lo > 0 and all(x > 0 for x in d['seed_deltas_mm']) else '稳定收益证据不足')
        verdicts.append(f'{name} {d["mean_mm"]:+.3f}mm，{verdict}')
        lines.append(f'- {verdicts[-1]}；take配对95%区间[{lo:.3f}, {hi:.3f}]；种子差值{d["seed_deltas_mm"]}。')
    lines += ['', f'原E7共同启动体型参照：{ref:.3f}mm。相对E7差值为描述性比较；推理上下文与计算成本不同。',
        '区间按take重采样、先平均三个训练种子；不覆盖多次采样不确定性，未作多重比较校正。', '',
        '仅最小评估完成；未完成原计划的多draw、低NFE和固定历史诊断。R0/R1为预测参考，R2/R3为观测参考；R0/R2读21帧，R1/R3最多80帧。',
        '本轮使用已知地面和预测启动体型；222take已用于前轮方法讨论，属于重复使用的基准。E7直接读取最多80帧；21帧组观测量不同。', '',
        '[完整统计](summary.json)；[最小评价协议](plan.json)。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    rel = out.relative_to(ROOT)
    log('G观测长度×设备参考四组评估完成', [
        '；'.join(f'{g} Body {v["body_mean_mm"]:.3f}±{v["body_std_mm"]:.3f}mm' for g, v in groups.items()),
        '；'.join(verdicts),
        f'原E7共同体型参照{ref:.3f}mm；best权重、三训练seed、Euler10、单draw1062。',
        f'仅完成缩减后的评价范围；[完整结论]({rel}/RESULTS.md)；[配对统计]({rel}/summary.json)。'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((out/'plan.json').read_text());par=Path(cfg['parent_run'])
    check=json.loads((par/'verification/smoke.json').read_text())
    if not check['passed']:raise ValueError('Parent preparation verification missing.')
    for path,digest in check['code_sha256'].items():
        if file_sha256(ROOT/path)!=digest:raise ValueError(f'Parent dependency changed: {path}')
    hashes={p:file_sha256(ROOT/p) for p in CODE}
    identity={'code_sha256':hashes,'parent_manifest_sha256':file_sha256(out/'data/manifest.json'),
              'prior_sha256':file_sha256(out/'prior/prior.pt'),'config':cfg}
    identity_path=out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Experiment identity changed.')
    write_json(identity_path,identity)
    for path in (*CODE,*check['code_sha256']):
        dst=out/'source'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dst)
    base=['--output',str(out)]
    try:
        stage(out,'e7_context_diagnostic',[
            ('e7_context_diagnostic',4,'run.train_observation_reference',['diagnostic',*base],out/'diagnostic/report.json',{})])
        geom=verify_geometry(out,cfg)
        stage(out,'real_data_smoke',[
            (f'smoke_{g}',gpu,'run.train_observation_reference',['train',*base,'--group',g,'--seed','62','--smoke'],
             out/'verification'/f'{g}_smoke/report.json',{'smoke':True})
            for g,gpu in zip(cfg['groups'],cfg['gpus'])])
        write_json(out/'verification/report.json',{'passed':True,'geometry_max_abs_m':geom,
                   'code_sha256':hashes,'smoke_optimizer_steps':2,'smoke_frames':96})
        diag=json.loads((out/'diagnostic/report.json').read_text())
        rel=out.relative_to(ROOT)
        log('观测长度×设备参考四组正式训练启动',[
            f'冻结E7开发集观测诊断：21帧{diag["mean_mm"]["21"]:.3f}mm，80帧{diag["mean_mm"]["80"]:.3f}mm；不是新G的成绩。',
            'R0/R1为预测参考，R2/R3为当前设备planar参考；R0/R2读取21帧，R1/R3最多80帧；全部Gaussian源、混合历史、相同冻结P和E7初始化。',
            '四组真实数据反传和96帧闭环试跑通过，坐标往返核验通过。GPU4–7每卡一组，24,000步×三seed；之后仅best/Euler10/draw1062评价。',
            f'[状态]({rel}/queue.json)；[控制日志]({rel}/controller.log)；[配置]({rel}/plan.json)；[诊断]({rel}/diagnostic/report.json)。'])
        for seed in cfg['seeds']:
            stage(out,f'training_seed{seed}',[
                (f'{g}_seed{seed}',gpu,'run.train_observation_reference',['train',*base,'--group',g,'--seed',str(seed)],
                 out/g/f'seed{seed}/report.json',{'smoke':False,'steps':cfg['steps']})
                for g,gpu in zip(cfg['groups'],cfg['gpus'])])
        selected={f'{g}/seed{s}':file_sha256(out/g/f'seed{s}/best.pt') for g in cfg['groups'] for s in cfg['seeds']}
        selected_path=out/'evaluation/selected_checkpoints.json'
        if selected_path.exists() and json.loads(selected_path.read_text())!=selected:
            raise ValueError('Selected checkpoints changed after evaluation.')
        write_json(selected_path,selected)
        for seed in cfg['seeds']:
            stage(out,f'evaluation_seed{seed}',[
                (f'eval_{g}_seed{seed}',gpu,'run.train_observation_reference',['evaluate',*base,'--group',g,'--seed',str(seed)],
                 out/g/f'seed{seed}/test/report.json',{}) for g,gpu in zip(cfg['groups'],cfg['gpus'])])
        summarize(out,cfg)
        write_json(out/'queue.json',{'phase':'completed','completed':True,'controller_pid':os.getpid()})
    except Exception as exc:
        write_json(out/'queue.json',{'phase':'failed','error':str(exc),'controller_pid':os.getpid()})
        log('观测长度×设备参考实验队列中断',[str(exc),f'[状态]({out.relative_to(ROOT)}/queue.json)。已完成产物保留。'])
        raise


if __name__=='__main__':main()
