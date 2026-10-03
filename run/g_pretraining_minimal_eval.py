"""Evaluate only dev-selected G checkpoints: three seeds, Euler10, one draw."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from run.e7_ablation import save, write_json
from run.g_pretraining_experiment import ROOT, complete, log, stage


PROTOCOL = 'best-three-seeds-nfe10-draw1062-v1'


def worker(out, group, seed):
    import torch
    from run.train_g_pretraining import (
        PairedSequences, MotionCodec, get_smpl, load_prior, HistoryUniEgoMotion,
        get_cfg_defaults, evaluate, GROUPS, file_sha256,
    )
    directory = out / group / f'seed{seed}'
    checkpoint_path = directory / 'best.pt'
    digest = file_sha256(checkpoint_path)
    frozen = json.loads((out / 'evaluation/minimal_plan.json').read_text())
    if digest != frozen['best_checkpoint_sha256'][f'{group}/seed{seed}']:
        raise ValueError('Selected checkpoint changed after protocol freeze.')
    path = directory / 'test/best_nfe10_draw1062.pt'
    if path.exists():
        result = torch.load(path, map_location='cpu', weights_only=True)
        if result['checkpoint_sha256'] != digest:
            raise ValueError('Existing evaluation belongs to another checkpoint.')
    else:
        device = torch.device('cuda')
        data = PairedSequences(out, 'test')
        codec = MotionCodec(data.stats).to(device)
        smpl = get_smpl().to(device).eval().requires_grad_(False)
        prior = load_prior(out / 'prior/prior.pt', device, manifest_sha=data.manifest_sha)
        model = HistoryUniEgoMotion(get_cfg_defaults(), past_observations=True).to(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        summary, rows = evaluate(model, prior, data, codec, smpl, device,
                                 mode=GROUPS[group][0], draw=1062, nfe=10)
        result = {'identity': checkpoint['identity'], 'checkpoint_sha256': digest,
                  'summary': summary, 'rows': rows}
        save(path, result)
    print('MINIMAL TEST', group, seed, result['summary']['scores']['body_mm']['mean'], flush=True)
    write_json(directory / 'test/minimal_report.json', {
        'completed': True, 'protocol': PROTOCOL, 'checkpoint_sha256': digest,
        'body_mm': result['summary']['scores']['body_mm']['mean'],
    })


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
    for a, b in [('T1', 'T0'), ('T2', 'T0'), ('T3', 'T2'), ('T3', 'T1'), ('T3', 'T0')]:
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
        'omitted': ['remaining last checkpoints', 'multiple draws', 'low NFE', 'fixed history diagnostics'],
        'limitations': 'One sampling draw; dev-selected checkpoints; known floor and fixed bootstrap shape. '
                        'Project holdout is from official val used by upstream E7 for validation.'}
    write_json(out / 'minimal_summary.json', summary)
    lines = ['# G四组最小评估结果', '',
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
        '仅最小评估完成；未完成原计划的多draw、低NFE和固定历史诊断。已完成末点结果保留，不混入本表。',
        '本轮使用已知地面和预测启动体型；留出集不是确认上游完全未见的测试集。', '',
        '[完整统计](minimal_summary.json)；[最小评价协议](evaluation/minimal_plan.json)。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    rel = out.relative_to(ROOT)
    log('G四组最小留出评估完成', [
        '；'.join(f'{g} Body {v["body_mean_mm"]:.3f}±{v["body_std_mm"]:.3f}mm' for g, v in groups.items()),
        '；'.join(verdicts),
        f'原E7共同体型参照{ref:.3f}mm；best权重、三训练seed、Euler10、单draw1062。',
        f'仅完成缩减后的评价范围；[完整结论]({rel}/RESULTS.md)；[配对统计]({rel}/minimal_summary.json)。'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--group', choices=['T0', 'T1', 'T2', 'T3'])
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out, args.group, args.seed)
        return
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((out / 'plan.json').read_text())
    from egorecover.evaluation_protocol import file_sha256
    check = json.loads((out / 'verification/smoke.json').read_text())
    if not check['passed']:
        raise ValueError('Original verification did not pass.')
    for path, digest in check['code_sha256'].items():
        if file_sha256(ROOT / path) != digest:
            raise ValueError(f'Original inference/training code changed: {path}')
    hashes = {}
    for g in cfg['groups']:
        for seed in cfg['seeds']:
            d = out / g / f'seed{seed}'
            if not complete(d / 'report.json', smoke=False, steps=cfg['steps']):
                raise ValueError(f'Training incomplete: {d}')
            hashes[f'{g}/seed{seed}'] = file_sha256(d / 'best.pt')
    if not complete(out / 'evaluation/e7_baseline.json'):
        raise ValueError('Shared E7 reference missing.')
    plan = {'protocol': PROTOCOL, 'reason': 'User requested minimal evaluation',
            'groups': cfg['groups'], 'seeds': cfg['seeds'], 'checkpoint': 'best',
            'draws': [1062], 'nfe': [10], 'takes': 222, 'best_checkpoint_sha256': hashes,
            'runner_sha256': file_sha256(Path(__file__))}
    plan_path = out / 'evaluation/minimal_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Minimal evaluation plan changed; audit before resuming.')
    write_json(plan_path, plan)
    try:
        for seed in cfg['seeds']:
            stage(out, f'minimal_evaluation_seed{seed}', [
                (f'minimal_eval_{g}_seed{seed}', gpu, 'run.g_pretraining_minimal_eval',
                 ['--worker', '--output', str(out), '--group', g, '--seed', str(seed)],
                 out / g / f'seed{seed}/test/minimal_report.json', {'protocol': PROTOCOL})
                for g, gpu in zip(cfg['groups'], cfg['gpus'])])
        summarize(out, cfg)
        write_json(out / 'queue.json', {'phase': 'minimal_evaluation_completed', 'completed': True,
            'protocol': PROTOCOL, 'full_original_evaluation_completed': False, 'controller_pid': os.getpid()})
    except Exception as error:
        write_json(out / 'queue.json', {'phase': 'minimal_evaluation_failed', 'error': str(error),
                                      'controller_pid': os.getpid()})
        log('G最小评估中断', [str(error), '已完成结果保留，可按同一协议恢复。'])
        raise


if __name__ == '__main__':
    main()
