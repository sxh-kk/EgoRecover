"""Run the preregistered four-case, three-seed CV residual experiment."""
import argparse
import json
from pathlib import Path
import statistics

from egorecover.evaluation_resume import atomic_json
from run.complete_stages import Queue, Task, append_log, command, now

CASES = {'H0_hold_dense': ('hold', 0), 'V0_cv_dense': ('constant_velocity', 0),
         'H1_hold_fk': ('hold', 1), 'V1_cv_fk': ('constant_velocity', 1)}
SEEDS = (62, 63, 64)
DATA = Path('exp/egorecover_prior_two_forward/v1/data')


def tasks_for(output, data=DATA, gpus=(4, 5, 6, 7)):
    return [Task(f'{case}_s{seed}', command('run.train_prior_cv_residual', data=data,
                output=output / 'train' / f'{case}_s{seed}', base_mode=mode, fk_weight=fk,
                seed=seed, baselines=(case == 'H0_hold_dense' and seed == 62)),
                output / 'train' / f'{case}_s{seed}', 'prior', gpu=gpus[i])
            for seed in SEEDS for i, (case, (mode, fk)) in enumerate(CASES.items())]


def paired_interval(differences):
    import numpy as np
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(62)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def collect_results(tasks):
    import torch
    reports = {task.name: json.loads(task.report.read_text()) for task in tasks}
    reference = next(iter(reports.values()))
    matched = ('data_sha256', 'code_sha256', 'steps', 'batch_size', 'eval_every', 'lr',
               'weight_decay', 'reference_mode', 'smplx_asset_sha256', 'selection')
    initial = {}
    for task in tasks:
        report = reports[task.name]
        if any(report[k] != reference[k] for k in matched):
            raise ValueError('Unmatched data/code/training settings.')
        checkpoint = torch.load(task.output / 'prior.pt', weights_only=True, map_location='cpu')
        if checkpoint['base_mode'] != report['base_mode']:
            raise ValueError('Checkpoint physical baseline mismatch.')
        seed = report['training_seed']
        weights = torch.load(task.output / 'initial.pt', weights_only=True, map_location='cpu')['state_dict']
        if seed in initial and any(not torch.equal(v, initial[seed][k]) for k, v in weights.items()):
            raise ValueError('Paired initial network parameters differ.')
        initial[seed] = weights
    scores = {name: r['results']['prior'] for name, r in reports.items()}
    baselines = {k: v for k, v in reports['H0_hold_dense_s62']['results'].items() if k != 'prior'}
    metrics = list(next(iter(scores.values()))['single'])
    aggregates = {}
    for case in CASES:
        group = [scores[f'{case}_s{seed}'] for seed in SEEDS]
        aggregates[case] = {'single': {k: {'mean': statistics.mean(v['single'][k]['mean'] for v in group),
                                           'std': statistics.stdev(v['single'][k]['mean'] for v in group)}
                                       for k in metrics},
                            'rollout': {h: {'mean': statistics.mean(v['rollout'][h]['mean'] for v in group),
                                            'std': statistics.stdev(v['rollout'][h]['mean'] for v in group)}
                                        for h in group[0]['rollout']}}
    comparisons = {}
    for left, right in [('V0_cv_dense', 'H0_hold_dense'), ('V1_cv_fk', 'H1_hold_fk'),
                        ('V0_cv_dense', 'constant_velocity'), ('V1_cv_fk', 'constant_velocity'),
                        ('H1_hold_fk', 'H0_hold_dense'), ('V1_cv_fk', 'V0_cv_dense')]:
        comparisons[f'{left}_minus_{right}'] = {}
        for metric in metrics:
            by_seed, by_take = {}, {}
            for seed in SEEDS:
                a = scores[f'{left}_s{seed}']['single'][metric]['per_take']
                b = (baselines[right] if right == 'constant_velocity' else
                     scores[f'{right}_s{seed}'])['single'][metric]['per_take']
                if set(a) != set(b):
                    raise ValueError('Unmatched take sets.')
                by_seed[str(seed)] = statistics.mean(a[t] - b[t] for t in a)
                for t in a:
                    by_take.setdefault(t, []).append(a[t] - b[t])
            differences = [statistics.mean(by_take[t]) for t in sorted(by_take)]
            comparisons[f'{left}_minus_{right}'][metric] = {
                'mean': statistics.mean(differences), 'by_seed': by_seed,
                'seed_averaged_take_differences': dict(zip(sorted(by_take), differences)),
                'take_bootstrap_95ci': paired_interval(differences)}
    interaction = {}
    for metric in metrics:
        d1 = comparisons['V1_cv_fk_minus_H1_hold_fk'][metric]['seed_averaged_take_differences']
        d0 = comparisons['V0_cv_dense_minus_H0_hold_dense'][metric]['seed_averaged_take_differences']
        ds = [d1[t] - d0[t] for t in sorted(d1)]
        interaction[metric] = {'mean': statistics.mean(ds), 'take_bootstrap_95ci': paired_interval(ds)}
    winner = min(('V0_cv_dense', 'V1_cv_fk'), key=lambda k: aggregates[k]['single']['fk22_mm']['mean'])
    lines = []
    for left, right in [('V0_cv_dense', 'H0_hold_dense'), ('V1_cv_fk', 'H1_hold_fk')]:
        c = comparisons[f'{left}_minus_{right}']['fk22_mm']
        lines.append(f"{left} 相对 {right}：Body MPJPE变化 {c['mean']:+.3f} mm，"
                     f"take配对95%区间 [{c['take_bootstrap_95ci'][0]:+.3f}, {c['take_bootstrap_95ci'][1]:+.3f}]。")
    c = comparisons[f'{winner}_minus_constant_velocity']['fk22_mm']
    values = aggregates[winner]['single']
    learned = all(v < 0 for v in c['by_seed'].values()) and c['take_bootstrap_95ci'][1] < 0
    target = values['fk22_mm']['mean'] <= 35 and values['same_shape_mpjpe_mm']['mean'] <= 15
    lines.append(f"候选 {winner}：Body MPJPE {values['fk22_mm']['mean']:.3f} mm，"
                 f"PA-MPJPE {values['pa_mpjpe_mm']['mean']:.3f} mm，统一体型 {values['same_shape_mpjpe_mm']['mean']:.3f} mm；"
                 f"超过纯CV的跨种子/区间要求：{'达到' if learned else '未达到'}；35/15 mm工程目标：{'达到' if target else '未达到'}。")
    lines.append('候选相对纯CV的seed62/63/64差值：' + '/'.join(f'{v:+.3f}' for v in c['by_seed'].values()) + ' mm。')
    zeros = [name for name, r in reports.items() if r['selected_step'] == 0]
    if zeros:
        lines.append('以下任务选中step0，残差学习未超过其物理基准：' + '、'.join(zeros) + '。')
    hold_case = 'H0_hold_dense' if winner == 'V0_cv_dense' else 'H1_hold_fk'
    for h in ('200', '400', '1000'):
        delta = aggregates[winner]['rollout'][h]['mean'] - aggregates[hold_case]['rollout'][h]['mean']
        best_physical = min(v['rollout'][h]['mean'] for v in baselines.values())
        lines.append(f"候选{h}ms自反馈MPJPE {aggregates[winner]['rollout'][h]['mean']:.3f} mm；"
                     f"相对配对保持残差组 {delta:+.3f} mm；该时长最强物理基线 {best_physical:.3f} mm。")
    lines.append('以上为dev选模结果；holdout尚未评估，尚未验证接入G的收益。')
    return {'scores': scores, 'baselines': baselines, 'aggregates': aggregates,
            'comparisons': comparisons, 'interaction': interaction, 'candidate': winner,
            'selected_steps': {k: v['selected_step'] for k, v in reports.items()},
            'final_single': {k: v['final_single'] for k, v in reports.items()},
            'conclusions': lines, 'holdout_used': False, 'completed': True}


def write_results(output, result):
    rows = ['# P 常速度残差实验结果', '', '三个种子全部完成；dev选模，holdout尚未使用。位置单位mm。', '',
            '| 配置 | Body MPJPE均值±标准差 | PA-MPJPE | 统一体型MPJPE | 1秒自反馈MPJPE |',
            '|---|---:|---:|---:|---:|']
    for name, value in result['aggregates'].items():
        m = value['single']
        rows.append(f"| {name} | {m['fk22_mm']['mean']:.3f} ± {m['fk22_mm']['std']:.3f} | "
                    f"{m['pa_mpjpe_mm']['mean']:.3f} | {m['same_shape_mpjpe_mm']['mean']:.3f} | {value['rollout']['1000']['mean']:.3f} |")
    for name, value in result['baselines'].items():
        m = value['single']
        rows.append(f"| {name} | {m['fk22_mm']['mean']:.3f} | {m['pa_mpjpe_mm']['mean']:.3f} | "
                    f"{m['same_shape_mpjpe_mm']['mean']:.3f} | {value['rollout']['1000']['mean']:.3f} |")
    rows += ['', '## 结论', '', *['- ' + x for x in result['conclusions']], '',
             '[详细结果与配对区间](summary.json)', '', '![自反馈误差](figures/rollout_errors.png)']
    (output / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, value in {**result['aggregates'], **result['baselines']}.items():
        xs = sorted(int(h) for h in value['rollout'])
        ax.plot(xs, [value['rollout'][str(h)]['mean'] for h in xs], marker='o', label=name)
    ax.set(xlabel='Prediction horizon (ms)', ylabel='Body MPJPE (mm)', title='P-only dev rollout; learned models averaged over 3 seeds')
    ax.legend(fontsize=8)
    fig.tight_layout()
    (output / 'figures').mkdir(exist_ok=True)
    fig.savefig(output / 'figures/rollout_errors.png', dpi=160)
    fig.savefig(output / 'figures/rollout_errors.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--gpus', type=int, nargs=4, default=[4, 5, 6, 7])
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or any(g not in range(8) for g in args.gpus):
        parser.error('Four distinct GPUs in 0..7 required.')
    queue = Queue(args.output, args.gpus, 15, allow_shared=True, minimum_free_gib=32)
    plan = {'cases': CASES, 'seeds': list(SEEDS), 'gpus': args.gpus, 'data': str(args.data),
            'steps': 2400, 'minimum_free_gib': 32, 'method': 'docs/experiments/p-cv-residual.md'}
    plan = json.loads(json.dumps(plan))
    path = args.output / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Experiment plan changed.')
    atomic_json(path, plan)
    tasks = tasks_for(args.output, args.data, args.gpus)
    for task in tasks:
        queue.register(task)
    if args.retry_failed:
        for entry in queue.state['tasks'].values():
            if entry['status'] == 'failed':
                entry['status'] = 'pending'
    queue.save()
    try:
        if not queue.state.get('start_logged'):
            append_log('P 常速度残差实验启动', [f'GPU{args.gpus}；与现有任务共享，空闲显存门槛32 GiB；GT历史四组×三个种子，共12次训练。',
                       f'计划：`{path}`；状态：`{args.output / "queue.json"}`；设计：[常速度残差](docs/experiments/p-cv-residual.md)。'])
            queue.state['start_logged'] = True
            queue.save()
        for seed in SEEDS:
            queue.run([task for task in tasks if task.name.endswith(f'_s{seed}')])
        result = collect_results(tasks)
        atomic_json(args.output / 'summary.json', result)
        write_results(args.output, result)
        queue.state.update(status='complete', completed_at=now(), holdout_status='not_evaluated')
        queue.save()
        append_log('P 常速度残差实验结论', [*result['conclusions'],
                   f'完整结果：`{args.output / "RESULTS.md"}`；统计：`{args.output / "summary.json"}`。'])
    except Exception as error:
        queue.state.update(status='failed', error=repr(error))
        queue.save()
        append_log('P 常速度残差队列异常', [f'原因：{error!r}；状态：`{args.output / "queue.json"}`。'])
        raise


if __name__ == '__main__':
    main()
