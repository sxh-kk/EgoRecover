"""Matched CV/FK=0 experiments with and without explicit historical velocities."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from egorecover.evaluation_resume import atomic_json
from egorecover.prior_velocity import VELOCITY_SCHEMA
from run.complete_stages import Queue, Task, append_log, command, now
from run.prior_cv_residual_experiment import DATA, paired_interval

CASES = {'control': False, 'velocity': True}
SEEDS = (62, 63, 64)


def tasks_for(output, data=DATA, gpus=(4, 5, 6, 7)):
    tasks = []
    for seed in SEEDS:
        for case, enabled in CASES.items():
            path = output / 'train' / f'{case}_s{seed}'
            tasks.append(Task(f'{case}_s{seed}', command('run.train_prior_velocity', data=data,
                output=path, seed=seed, velocity_input=enabled, baselines=(case == 'control' and seed == 62)),
                path, 'prior', gpu=gpus[len(tasks) % len(gpus)]))
    return tasks


def collect_results(tasks):
    reports = {t.name: json.loads(t.report.read_text()) for t in tasks}
    first = next(iter(reports.values()))
    keys = ('data_sha256', 'code_sha256', 'steps', 'batch_size', 'eval_every', 'lr', 'weight_decay',
            'base_mode', 'fk_weight', 'geometry_weight', 'velocity_schema', 'reference_mode', 'selection')
    initial = {}
    for task in tasks:
        r = reports[task.name]
        if any(r[k] != first[k] for k in keys):
            raise ValueError('Training protocols differ.')
        saved = torch.load(task.output / 'prior.pt', map_location='cpu', weights_only=True)
        if any(saved[k] != r[k] for k in ('velocity_input', 'base_mode', 'velocity_schema')):
            raise ValueError('Checkpoint feature contract differs.')
        state = torch.load(task.output / 'initial.pt', map_location='cpu', weights_only=True)['state_dict']
        common = {k: v for k, v in state.items() if not k.startswith(('codec.', 'velocity_input.'))}
        seed = r['training_seed']
        if seed in initial and (set(common) != set(initial[seed]) or
                               any(not torch.equal(v, initial[seed][k]) for k, v in common.items())):
            raise ValueError('Shared initial parameters differ within a seed.')
        initial[seed] = common
    scores = {k: r['results']['prior'] for k, r in reports.items()}
    baselines = {k: v for k, v in reports['control_s62']['results'].items() if k != 'prior'}
    metrics = list(scores['control_s62']['single'])
    aggregates = {}
    for case in CASES:
        group = [scores[f'{case}_s{s}'] for s in SEEDS]
        aggregates[case] = {'single': {k: {'mean': statistics.mean(v['single'][k]['mean'] for v in group),
                                           'std': statistics.stdev(v['single'][k]['mean'] for v in group)} for k in metrics},
                            'rollout': {h: {'mean': statistics.mean(v['rollout'][h]['mean'] for v in group),
                                            'std': statistics.stdev(v['rollout'][h]['mean'] for v in group)}
                                        for h in group[0]['rollout']}}
    comparisons = {}
    for reference in ('control', 'constant_velocity'):
        comparisons[reference] = {}
        for metric in metrics:
            by_seed, take_values = {}, {}
            for seed in SEEDS:
                a = scores[f'velocity_s{seed}']['single'][metric]['per_take']
                b = (scores[f'control_s{seed}'] if reference == 'control' else baselines[reference])['single'][metric]['per_take']
                if set(a) != set(b):
                    raise ValueError('Take mismatch.')
                by_seed[str(seed)] = statistics.mean(a[t] - b[t] for t in a)
                for t in a:
                    take_values.setdefault(t, []).append(a[t] - b[t])
            ds = [statistics.mean(take_values[t]) for t in sorted(take_values)]
            comparisons[reference][metric] = {'mean': statistics.mean(ds), 'by_seed': by_seed,
                'take_bootstrap_95ci': paired_interval(ds),
                'seed_averaged_take_differences': dict(zip(sorted(take_values), ds))}
    main = comparisons['control']['fk22_mm']
    strong = all(x < 0 for x in main['by_seed'].values()) and main['take_bootstrap_95ci'][1] < 0
    v = aggregates['velocity']['single']
    c = aggregates['control']['single']
    lines = [f"显式速度相对配对对照：Body MPJPE {c['fk22_mm']['mean']:.3f}→{v['fk22_mm']['mean']:.3f} mm，"
             f"差值 {main['mean']:+.3f} mm，take配对95%区间 [{main['take_bootstrap_95ci'][0]:+.3f}, {main['take_bootstrap_95ci'][1]:+.3f}]。",
             'seed62/63/64差值：' + '/'.join(f'{x:+.3f}' for x in main['by_seed'].values()) + ' mm。',
             f"PA-MPJPE {c['pa_mpjpe_mm']['mean']:.3f}→{v['pa_mpjpe_mm']['mean']:.3f} mm；"
             f"统一体型 {c['same_shape_mpjpe_mm']['mean']:.3f}→{v['same_shape_mpjpe_mm']['mean']:.3f} mm。",
             f"单步稳定改善要求（各seed均改善且配对区间低于0）：{'达到' if strong else '未达到'}。"
             '纯P长程只作辅助诊断，不作为本轮单独淘汰条件。',
             f"相对纯CV Body MPJPE差值 {comparisons['constant_velocity']['fk22_mm']['mean']:+.3f} mm；"
             f"35/15 mm工程目标：{'达到' if v['fk22_mm']['mean'] <= 35 and v['same_shape_mpjpe_mm']['mean'] <= 15 else '未达到'}。",
             'dev参与选模；holdout、G预测历史上的单步适配及接入G收益尚未验证。']
    zeros = [k for k, r in reports.items() if r['selected_step'] == 0]
    if zeros:
        lines.append('选中step0，未学到超过CV的修正：' + ', '.join(zeros))
    return {'completed': True, 'scores': scores, 'baselines': baselines, 'aggregates': aggregates,
            'comparisons': comparisons, 'conclusions': lines, 'holdout_used': False,
            'selected_steps': {k: r['selected_step'] for k, r in reports.items()},
            'final_single': {k: r['final_single'] for k, r in reports.items()},
            'compute': {k: r['compute'] for k, r in reports.items()}}


def write_results(output, result):
    rows = ['# 显式速度输入实验结果', '', '常速度＋残差，geometry=1/FK=0；GT历史，三种子dev结果。位置单位mm。', '',
            '| 方法 | Body MPJPE均值±标准差 | PA-MPJPE | 统一体型MPJPE | 1秒自反馈（辅助） |', '|---|---:|---:|---:|---:|']
    for case, value in result['aggregates'].items():
        m = value['single']
        rows.append(f"| {case} | {m['fk22_mm']['mean']:.3f} ± {m['fk22_mm']['std']:.3f} | "
                    f"{m['pa_mpjpe_mm']['mean']:.3f} | {m['same_shape_mpjpe_mm']['mean']:.3f} | {value['rollout']['1000']['mean']:.3f} |")
    for case, value in result['baselines'].items():
        m = value['single']
        rows.append(f"| {case} | {m['fk22_mm']['mean']:.3f} | {m['pa_mpjpe_mm']['mean']:.3f} | "
                    f"{m['same_shape_mpjpe_mm']['mean']:.3f} | {value['rollout']['1000']['mean']:.3f} |")
    rows += ['', '## 结论', '', *['- ' + line for line in result['conclusions']], '',
             '[完整指标、配对区间和计算量](summary.json)', '', '![配对单步指标](figures/single_step.png)']
    (output / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, key in zip(axes, ['fk22_mm', 'pa_mpjpe_mm', 'same_shape_mpjpe_mm']):
        for seed in SEEDS:
            ax.plot([0, 1], [result['scores'][f'{case}_s{seed}']['single'][key]['mean'] for case in CASES],
                    marker='o', label=f'seed {seed}')
        ax.axhline(result['baselines']['constant_velocity']['single'][key]['mean'], color='gray', linestyle='--', label='CV')
        ax.set(xticks=[0, 1], xticklabels=['Control', '+ velocity'], ylabel='mm', title=key)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (output / 'figures').mkdir(exist_ok=True)
    fig.savefig(output / 'figures/single_step.png', dpi=160)
    fig.savefig(output / 'figures/single_step.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs=4, type=int, default=[4, 5, 6, 7])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or any(g not in range(8) for g in args.gpus):
        parser.error('Four distinct GPUs in 0..7 required.')
    queue = Queue(args.output, args.gpus, 15, allow_shared=True, minimum_free_gib=32)
    plan = {'cases': CASES, 'seeds': list(SEEDS), 'gpus': args.gpus, 'data': str(args.data),
            'base_mode': 'constant_velocity', 'fk_weight': 0, 'steps': 2400, 'velocity_schema': VELOCITY_SCHEMA,
            'method': 'docs/experiments/p-explicit-velocity.md', 'rollout_role': 'diagnostic_only'}
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
            append_log('P 显式速度输入实验启动', [f'GPU{args.gpus}；常速度/FK=0，两组×seeds62/63/64，共6次训练。',
                       f'计划：`{path}`；队列：`{args.output / "queue.json"}`；[方案](docs/experiments/p-explicit-velocity.md)。'])
            queue.state['start_logged'] = True
            queue.save()
        queue.run(tasks)
        result = collect_results(tasks)
        atomic_json(args.output / 'summary.json', result)
        write_results(args.output, result)
        queue.state.update(status='complete', completed_at=now())
        queue.save()
        append_log('P 显式速度输入实验结论', [*result['conclusions'],
                   f'结果：`{args.output / "RESULTS.md"}`；统计：`{args.output / "summary.json"}`。'])
    except Exception as error:
        queue.state.update(status='failed', error=repr(error))
        queue.save()
        append_log('P 显式速度输入队列异常', [f'{error!r}；状态：`{args.output / "queue.json"}`。'])
        raise


if __name__ == '__main__':
    main()
