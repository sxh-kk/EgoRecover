"""Four matched pose-supervision weights; unchanged original-GT evaluation."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from egorecover.evaluation_resume import atomic_json
from run.complete_stages import Queue, Task, append_log, command, now
from run.prior_cv_residual_experiment import DATA, paired_interval

CASES = {'control': 0., 'pose01': .1, 'pose03': .3, 'pose10': 1.}
SEEDS = (62, 63, 64)


def tasks_for(output, data=DATA, gpus=(4, 5, 6, 7)):
    return [Task(f'{case}_s{seed}', command('run.train_prior_same_shape', data=data,
                output=output / 'train' / f'{case}_s{seed}', same_shape_weight=weight,
                seed=seed, baselines=(case == 'control' and seed == 62)),
                output / 'train' / f'{case}_s{seed}', 'prior', gpu=gpus[i])
            for seed in SEEDS for i, (case, weight) in enumerate(CASES.items())]


def collect_results(tasks):
    reports = {t.name: json.loads(t.report.read_text()) for t in tasks}
    reference = next(iter(reports.values()))
    matched = ('data_sha256', 'code_sha256', 'steps', 'batch_size', 'eval_every', 'lr', 'weight_decay',
               'base_mode', 'fk_weight', 'geometry_weight', 'supervision_schema', 'velocity_input',
               'reference_mode', 'smplx_asset_sha256', 'selection')
    initial = {}
    for task in tasks:
        r = reports[task.name]
        if any(r[k] != reference[k] for k in matched):
            raise ValueError('Unmatched training protocols.')
        checkpoint = torch.load(task.output / 'prior.pt', weights_only=True, map_location='cpu')
        if any(checkpoint[k] != r[k] for k in ('same_shape_weight', 'base_mode', 'supervision_schema')):
            raise ValueError('Checkpoint supervision identity differs.')
        state = torch.load(task.output / 'initial.pt', weights_only=True, map_location='cpu')['state_dict']
        seed = r['training_seed']
        if seed in initial and (set(state) != set(initial[seed]) or any(not torch.equal(v, initial[seed][k]) for k, v in state.items())):
            raise ValueError('Paired initial parameters differ.')
        initial[seed] = state
    scores = {k: v['results']['prior'] for k, v in reports.items()}
    baselines = {k: v for k, v in reports['control_s62']['results'].items() if k != 'prior'}
    metrics = list(scores['control_s62']['single'])
    aggregates = {}
    for case in CASES:
        values = [scores[f'{case}_s{s}'] for s in SEEDS]
        aggregates[case] = {'single': {k: {'mean': statistics.mean(v['single'][k]['mean'] for v in values),
                                           'std': statistics.stdev(v['single'][k]['mean'] for v in values)} for k in metrics},
                            'rollout': {h: {'mean': statistics.mean(v['rollout'][h]['mean'] for v in values),
                                            'std': statistics.stdev(v['rollout'][h]['mean'] for v in values)}
                                        for h in values[0]['rollout']}}
    comparisons = {}
    for case in list(CASES)[1:]:
        comparisons[case] = {}
        for ref in ('control', 'constant_velocity'):
            comparisons[case][ref] = {}
            for metric in metrics:
                per_seed, per_take = {}, {}
                for seed in SEEDS:
                    a = scores[f'{case}_s{seed}']['single'][metric]['per_take']
                    b = (scores[f'control_s{seed}'] if ref == 'control' else baselines[ref])['single'][metric]['per_take']
                    if set(a) != set(b):
                        raise ValueError('Take sets differ.')
                    per_seed[str(seed)] = statistics.mean(a[t] - b[t] for t in a)
                    for t in a:
                        per_take.setdefault(t, []).append(a[t] - b[t])
                ds = [statistics.mean(per_take[t]) for t in sorted(per_take)]
                comparisons[case][ref][metric] = {'mean': statistics.mean(ds), 'by_seed': per_seed,
                    'take_bootstrap_95ci': paired_interval(ds), 'seed_averaged_take_differences': dict(zip(sorted(per_take), ds))}
    candidate = min(list(CASES)[1:], key=lambda k: aggregates[k]['single']['fk22_mm']['mean'])
    conclusions = []
    for case in list(CASES)[1:]:
        d = comparisons[case]['control']; main = d['fk22_mm']; ci = main['take_bootstrap_95ci']
        conclusions.append(f"同体型权重{CASES[case]:g}：Body MPJPE变化{main['mean']:+.3f} mm，95%区间[{ci[0]:+.3f}, {ci[1]:+.3f}]；"
                           f"统一体型变化{d['same_shape_mpjpe_mm']['mean']:+.3f} mm；局部旋转变化{d['local_rotation_deg']['mean']:+.3f}°。")
        conclusions.append(f"权重{CASES[case]:g}的seed62/63/64正式MPJPE差值：" + '/'.join(f'{v:+.3f}' for v in main['by_seed'].values()) + ' mm。')
    v = aggregates[candidate]['single']; d = comparisons[candidate]['control']['fk22_mm']
    reliable = all(x < 0 for x in d['by_seed'].values()) and d['take_bootstrap_95ci'][1] < 0
    conclusions += [f"按三种子正式MPJPE选出的候选为{candidate}：Body {v['fk22_mm']['mean']:.3f} mm、PA {v['pa_mpjpe_mm']['mean']:.3f} mm、"
                    f"统一体型 {v['same_shape_mpjpe_mm']['mean']:.3f} mm。",
                    f"候选相对配对对照的各seed/区间改善要求：{'达到' if reliable else '未达到'}；"
                    f"35/15 mm工程目标：{'达到' if v['fk22_mm']['mean'] <= 35 and v['same_shape_mpjpe_mm']['mean'] <= 15 else '未达到'}。",
                    '主指标始终对原始GT计算；纯P长程仅作诊断。dev用于checkpoint及三个权重的选择，区间未作多重比较校正，不能代替独立验证。',
                    'holdout和接入G收益尚未验证；显式速度分支未启用。']
    return {'completed': True, 'scores': scores, 'aggregates': aggregates, 'baselines': baselines,
            'comparisons': comparisons, 'candidate': candidate, 'conclusions': conclusions, 'holdout_used': False,
            'selected_steps': {k: r['selected_step'] for k, r in reports.items()},
            'final_single': {k: r['final_single'] for k, r in reports.items()},
            'compute': {k: r['compute'] for k, r in reports.items()}}


def write_results(output, result):
    rows = ['# P 统一体型FK监督实验结果', '', '常速度残差，无显式速度；原FK损失=0，geometry=1。三个种子dev选模结果。', '',
            '| 新损失权重 | Body MPJPE均值±标准差 | PA-MPJPE | 统一体型MPJPE | 局部旋转° |', '|---|---:|---:|---:|---:|']
    for case, value in result['aggregates'].items():
        m = value['single']
        rows.append(f"| {CASES[case]:g} | {m['fk22_mm']['mean']:.3f} ± {m['fk22_mm']['std']:.3f} | "
                    f"{m['pa_mpjpe_mm']['mean']:.3f} | {m['same_shape_mpjpe_mm']['mean']:.3f} | {m['local_rotation_deg']['mean']:.3f} |")
    for case, value in result['baselines'].items():
        m = value['single']
        rows.append(f"| {case} | {m['fk22_mm']['mean']:.3f} | {m['pa_mpjpe_mm']['mean']:.3f} | "
                    f"{m['same_shape_mpjpe_mm']['mean']:.3f} | {m['local_rotation_deg']['mean']:.3f} |")
    rows += ['', '位置单位mm。', '', '## 结论', '', *['- ' + line for line in result['conclusions']], '',
             '[完整统计与辅助自反馈](summary.json)', '', '![单步指标](figures/single_step.png)']
    (output / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ['fk22_mm', 'pa_mpjpe_mm', 'same_shape_mpjpe_mm']):
        for seed in SEEDS:
            ax.plot(range(4), [result['scores'][f'{case}_s{seed}']['single'][key]['mean'] for case in CASES], marker='o', label=f'seed {seed}')
        ax.axhline(result['baselines']['constant_velocity']['single'][key]['mean'], color='gray', linestyle='--', label='CV')
        ax.set(xticks=range(4), xticklabels=['0', '0.1', '0.3', '1.0'], xlabel='Same-shape loss weight', ylabel='mm', title=key)
    axes[0].legend(fontsize=8);fig.tight_layout();(output/'figures').mkdir(exist_ok=True)
    fig.savefig(output/'figures/single_step.png', dpi=160);fig.savefig(output/'figures/single_step.pdf');plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--gpus', type=int, nargs=4, default=[4, 5, 6, 7])
    p.add_argument('--minimum-free-gib', type=float, default=12)
    p.add_argument('--retry-failed', action='store_true')
    args = p.parse_args()
    if len(set(args.gpus)) != 4 or any(g not in range(8) for g in args.gpus) or args.minimum_free_gib < 8:
        p.error('Four distinct GPUs and at least 8 GiB free-memory threshold required.')
    queue = Queue(args.output, args.gpus, 15, allow_shared=True, minimum_free_gib=args.minimum_free_gib)
    plan = {'cases': CASES, 'seeds': list(SEEDS), 'gpus': args.gpus, 'data': str(args.data), 'steps': 2400,
            'base_mode': 'constant_velocity', 'original_fk_weight': 0, 'velocity_input': False,
            'minimum_free_gib': args.minimum_free_gib, 'method': 'docs/experiments/p-same-shape-fk.md',
            'selection': 'original_GT_world_Body_MPJPE', 'rollout_role': 'diagnostic_only'}
    path = args.output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Experiment plan changed.')
    atomic_json(path, plan);tasks = tasks_for(args.output, args.data, args.gpus)
    for task in tasks:
        queue.register(task)
    if args.retry_failed:
        for entry in queue.state['tasks'].values():
            if entry['status'] == 'failed': entry['status'] = 'pending'
    queue.save()
    try:
        if not queue.state.get('start_logged'):
            append_log('P 统一体型监督实验启动', [f'GPU{args.gpus}；四个权重×三个种子，共12次训练；显式速度关闭。',
                f'空闲显存门槛{args.minimum_free_gib:g}GiB；计划：`{path}`；状态：`{args.output / "queue.json"}`。'])
            queue.state['start_logged'] = True;queue.save()
        queue.run(tasks)
        result = collect_results(tasks);atomic_json(args.output/'summary.json', result);write_results(args.output, result)
        queue.state.update(status='complete', completed_at=now());queue.save()
        append_log('P 统一体型监督实验结论', [*result['conclusions'],
            f'结果：`{args.output / "RESULTS.md"}`；完整统计：`{args.output / "summary.json"}`。'])
    except Exception as error:
        queue.state.update(status='failed', error=repr(error));queue.save()
        append_log('P 统一体型监督队列异常', [f'{error!r}；状态：`{args.output / "queue.json"}`。'])
        raise


if __name__ == '__main__':
    main()
