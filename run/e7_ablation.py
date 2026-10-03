"""Minimal E7 inference audit. Upstream model/sampler/decoder remain unmodified.

python -m run.e7_ablation plan
python -m run.e7_ablation prepare
python -m run.e7_ablation verify
python -m run.e7_ablation evaluate --smoke --device cpu
python -m run.e7_ablation evaluate --shard 0 --shards 4
python -m run.e7_ablation summarize
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def save(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp); tmp.replace(path)


def install_upstream(plan):
    upstream = Path(plan['upstream']).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=upstream, text=True)
    if commit != plan['upstream_commit'] or dirty:
        raise ValueError('Expected a clean, pinned upstream checkout.')
    # Called before importing any model, dataset, utils or config package.
    sys.path.insert(0, str(upstream))
    os.environ.setdefault('SMPLX_MODEL_PATH', str(ROOT/'body_models/smplx'))


def identity(plan, plan_path):
    return {'plan_sha256': sha(plan_path), 'upstream_commit': plan['upstream_commit'],
            'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                (ROOT/'run/e7_ablation.py', ROOT/'egorecover/e7_ablation.py',
                 ROOT/'egorecover/codec.py', ROOT/'egorecover/fk.py')},
            'checkpoint_sha256': sha(plan['checkpoint']),
            'smplx_sha256': sha(Path(os.environ['SMPLX_MODEL_PATH'])/'SMPLX_NEUTRAL.npz')}


def prepare(plan, output, ident):
    import torch
    from egorecover.data import open_dataset
    source_dir = Path(plan['input_packets']).parent
    report = json.loads((source_dir/'report.json').read_text())
    if sha(plan['input_packets']) != report['cache_sha256']:
        raise ValueError('Input packet identity changed.')
    packet = torch.load(plan['input_packets'], weights_only=True, map_location='cpu')
    split = packet['identity']['original_identity']
    dataset, _ = open_dataset(signal=Path(plan['data_signal']), spec_sha256=split['dataset_spec_sha256'])
    groups = {}
    for group, cases in packet['packs'].items():
        groups[group] = []
        for case in cases:
            if case['variant'] != 'clean':
                continue
            supervision = dataset.supervision(case['record']['variant_id'])
            supervision = {k: ({n: torch.as_tensor(v).clone() for n, v in value.items()}
                              if isinstance(value, dict) else torch.as_tensor(value).clone())
                           for k, value in supervision.items()}
            gt = case['truth']['joints'][..., :3, 3].clone()
            gt[..., 2] += case['startup']['floor_estimate_m']
            if not torch.allclose(gt, supervision['kp3d'][:, :22], atol=2e-5, rtol=0):
                raise ValueError('Original P targets do not recover source world coordinates.')
            groups[group].append({**case, 'supervision': supervision})
    if sorted(c['take'] for c in groups['dev']) != sorted(plan['takes']):
        raise ValueError('Dev take set changed.')
    if set(plan['takes']) & set(split['splits']['holdout']):
        raise ValueError('Holdout overlap.')
    path = output/'data/packets.pt'
    save(path, {'identity': ident, 'stats': packet['stats'], 'packs': groups})
    write_json(output/'data/report.json', {'completed': True, 'identity': ident,
        'packet_sha256': sha(path), 'source_packet_sha256': sha(plan['input_packets']),
        'holdout_used': False, 'counts': {k: len(v) for k, v in groups.items()}})


def load_data(output, ident):
    import torch
    report = json.loads((output/'data/report.json').read_text())
    # Data stores copied observations/labels, not model outputs. Preserve its
    # producer code hash, but do not reload the full source dataset when only
    # verification or inference code changes. Predictions still require exact
    # consumer code identity. Config/assets and packet integrity stay strict.
    expected = {k:v for k,v in ident.items() if k != 'code_sha256'}
    produced = {k:v for k,v in report['identity'].items() if k != 'code_sha256'}
    if produced != expected or sha(output/'data/packets.pt') != report['packet_sha256']:
        raise ValueError('Prepared data/config/asset identity changed; use a new output version.')
    payload = torch.load(output/'data/packets.pt', weights_only=True, map_location='cpu')
    if payload['identity'] != report['identity']:
        raise ValueError('Packet producer identity differs from manifest.')
    return payload


def verify(plan, output, ident):
    import torch
    from dataset.canonicalization import get_a_canonicalized_segment
    from dataset.representation_utils import saved_sequence_to_repre
    from dataset.smpl_utils import get_smpl
    from egorecover.codec import MotionCodec
    from egorecover.e7_ablation import conditions, matrix, decode
    payload = load_data(output, ident)
    codec = MotionCodec(payload['stats']); smpl = get_smpl().eval()
    base = next(r for r in matrix() if r['window']=='causal' and r['floor']=='source'
                and r['encoder']=='native' and r['beta']=='window' and r['decoder']=='native'
                and r['padding']=='pad80' and r['noise']=='independent')
    checks = []
    for case in payload['packs']['smoke']:
        for first, last in ((0, 21), (0, 80), (1, 81), (120, 200)):
            s = case['supervision']
            segment = get_a_canonicalized_segment(s['smpl_params'], case['observations']['aria_traj_obs'],
                                                 s['kp3d'], smpl, first, last-1)
            raw, traj = saved_sequence_to_repre('v4_beta', segment['can_aria_traj'],
                segment['can_smpl_params'], segment['can_kp3d'], float(s['floor_height']), smpl)
            y, anchor, floor = conditions(case, first, last, base, codec)
            traj_error = float((y['traj'][:last-first]-codec.normalize(traj, 'traj')).abs().max())
            joints, dense = decode(codec.normalize(raw), anchor, floor, base, codec, smpl)
            gt_error = float((joints-s['kp3d'][first:last, :22]).norm(dim=-1).max()*1000)
            dense_error = float((dense-s['kp3d'][first:last, :22]).norm(dim=-1).max()*1000)
            # Validate adaptation against independent upstream full representation and source GT.
            # The representation stores exact dense positions; SMPL FK can differ
            # slightly from saved dataset joints even with original GT rotations.
            # Gate world restoration on the dense roundtrip, report FK separately.
            if traj_error > 2e-4 or dense_error > .2 or not torch.isfinite(joints).all():
                raise ValueError(f'Native adapter parity failed: trajectory={traj_error}, dense roundtrip={dense_error} mm')
            checks.append({'take': case['take'], 'window': [first,last], 'native_input_max_abs': traj_error,
                           'source_GT_dense_roundtrip_max_mm': dense_error,
                           'source_GT_FK_roundtrip_max_mm': gt_error})
    write_json(output/'verification.json', {'passed': True, 'identity': ident, 'checks': checks,
        'meaning': 'Native preprocessing and source-world decoding only; no model accuracy claim.'})
    print(json.dumps(checks, indent=2), flush=True)


def load_model(plan, device):
    import torch
    from config.defaults import get_cfg_defaults
    from model.uniegomotion import UniEgoMotion
    from module.ema import apply_ema_weights_from_checkpoint
    from mydiffusion.flow_matching import FlowMatching
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(Path(plan['upstream'])/'ablation/configs/e7_x0_global_w8_u84k.yaml'))
    model = UniEgoMotion(cfg).to(device).eval()
    saved = torch.load(plan['checkpoint'], weights_only=True, map_location='cpu')
    state = saved.get('state_dict', saved)
    model.load_state_dict({k.removeprefix('model.'): v for k,v in state.items()}, strict=True)
    if not apply_ema_weights_from_checkpoint(model, saved):
        raise ValueError('E7 EMA missing.')
    # Use the pinned upstream implementation, not EgoRecover's Flow wrapper.
    flow = FlowMatching(num_steps=10, solver='euler', prediction_type='x0', global_weight=8.)
    return model.requires_grad_(False), flow


def evaluate(args, plan, output, ident):
    import torch
    from dataset.smpl_utils import get_smpl
    from egorecover.codec import MotionCodec
    from egorecover.e7_ablation import matrix, bounds, conditions, draw_noise, decode, metric_arrays
    checked = json.loads((output/'verification.json').read_text())
    if not checked['passed'] or checked['identity'] != ident:
        raise ValueError('Pass native parity verification with this code/data first.')
    payload = load_data(output, ident)
    group = 'smoke' if args.smoke else 'dev'
    takes = sorted(payload['packs'][group], key=lambda c:c['take'])[args.shard::args.shards]
    if not takes:
        raise ValueError('Empty shard.')
    rows = matrix(); routes = {r['inference_id']: r for r in rows}
    device = torch.device(args.device)
    model, flow = load_model(plan, device)
    # Keep native geometric processing on CPU, preserving upstream utilities.
    codec = MotionCodec(payload['stats']); smpl = get_smpl().eval().requires_grad_(False)
    times = torch.tensor([20,39,79,80,99,159,199] if args.smoke else list(range(20,200)))
    draws = plan['draws'][:1] if args.smoke else plan['draws']
    for case in takes:
        for draw in draws:
            case_dir = output/group/case['take']/str(draw)
            final = case_dir/'metrics.pt'
            if final.exists():
                old = torch.load(final, weights_only=True)
                if old['identity'] != ident or not torch.equal(old['times'], times):
                    raise ValueError('Completed case identity changed.')
                print('REUSE', final, flush=True); continue
            all_metrics = {}; all_joints = {}
            for route, row in routes.items():
                path = case_dir/'codes'/f'{route}.pt'
                windows = sorted({bounds(int(t), row['window']) for t in times})
                if path.exists():
                    cache = torch.load(path, weights_only=True)
                    if cache['identity'] != ident or cache['windows'] != windows:
                        raise ValueError('Prediction cache identity changed.')
                    codes = cache['codes']
                else:
                    codes = {}; by_length = {}
                    for first,last in windows:
                        y, _, _ = conditions(case, first, last, row, codec)
                        length = len(y['traj'])
                        noise = draw_noise(case['take'], draw, first, last, row['noise'], length)
                        by_length.setdefault(length, []).append(((first,last), y, noise))
                    for jobs in by_length.values():
                        for offset in range(0,len(jobs),args.batch_size):
                            batch = jobs[offset:offset+args.batch_size]
                            y = {k:torch.stack([j[1][k] for j in batch]).to(device) for k in batch[0][1]}
                            noise = torch.stack([j[2] for j in batch]).to(device)
                            pred = flow.sample_loop(model, noise.shape, {'y':y}, noise=noise).cpu()
                            for i, (window, _, _) in enumerate(batch):
                                codes[window] = pred[i,:window[1]-window[0]].clone()
                    save(path, {'identity':ident, 'windows':windows, 'codes':codes})
                for cell in (c for c in rows if c['inference_id']==route):
                    per_window = {}
                    for first,last in windows:
                        _, anchor, floor = conditions(case,first,last,cell,codec)
                        # Causal paths only need the terminal pose, but beta averages ALL valid codes.
                        # Native cumulative decoding still processes the full window.
                        per_window[first,last] = decode(codes[first,last],anchor,floor,cell,codec,smpl)
                    pred = torch.stack([per_window[bounds(int(t),cell['window'])][0][int(t)-bounds(int(t),cell['window'])[0]] for t in times])
                    dense = torch.stack([per_window[bounds(int(t),cell['window'])][1][int(t)-bounds(int(t),cell['window'])[0]] for t in times])
                    gt = case['supervision']['kp3d'][times,:22]
                    all_metrics[cell['id']] = metric_arrays(pred,dense,gt,times)
                    all_joints[cell['id']] = pred
                write_json(case_dir/'progress.json', {'route':route,'completed_cells':len(all_metrics),
                                                      'total_cells':len(rows),'take':case['take'],'draw':draw})
                print(group,case['take'],draw,f'{len(all_metrics)}/{len(rows)} cells',flush=True)
            save(final, {'identity':ident,'take':case['take'],'draw':draw,'times':times,'metrics':all_metrics,
                         'joints':all_joints,'truth':case['supervision']['kp3d'][times,:22]})
    write_json(output/group/f'shard{args.shard}.json', {'completed':True,'identity':ident,
        'takes':[c['take'] for c in takes],'draws':draws,'cells':len(rows),'smoke':args.smoke})


def summarize(plan, output, ident):
    import torch
    from egorecover.e7_ablation import matrix
    rows = matrix(); values = {}
    phases = {'all':(20,200),'startup':(20,79),'full_window':(79,200),'late':(100,200)}
    for take in sorted(plan['takes']):
        values[take] = []
        for draw in plan['draws']:
            path = output/'dev'/take/str(draw)/'metrics.pt'
            result = torch.load(path, weights_only=True)
            if result['identity'] != ident or not torch.equal(result['times'],torch.arange(20,200)):
                raise ValueError('Incomplete/mismatched result.')
            if set(result['metrics']) != {r['id'] for r in rows}:
                raise ValueError('Incomplete factor matrix.')
            values[take].append(result)
    scores = {}
    for row in rows:
        rid = row['id']; scores[rid] = {'factors':row,'phases':{}}
        for phase,(first,last) in phases.items():
            per_take = {}
            for take, draws in values.items():
                per_take[take] = {}
                for metric in draws[0]['metrics'][rid]:
                    arr = []
                    for d in draws:
                        t = d['times'][1:] if metric=='velocity_error_mmps' else d['times']
                        arr.append(d['metrics'][rid][metric][(t>=first)&(t<last)].double().mean())
                    per_take[take][metric] = float(torch.stack(arr).mean())
            scores[rid]['phases'][phase] = {'per_take':per_take,
                'mean':{k:sum(t[k] for t in per_take.values())/len(per_take) for k in next(iter(per_take.values()))}}
    # Every one-factor edge is paired by take. No independent-frame pseudo-replication.
    factors = ('window','floor')
    pairs=[]; resamples=torch.randint(len(values),(10000,len(values)),generator=torch.Generator().manual_seed(62))
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            changed=[k for k in factors if a[k]!=b[k]]
            if len(changed)!=1: continue
            for phase in phases:
                x=scores[a['id']]['phases'][phase]['per_take']; y=scores[b['id']]['phases'][phase]['per_take']
                delta=torch.tensor([y[t]['body_mpjpe_mm']-x[t]['body_mpjpe_mm'] for t in sorted(values)])
                ci=delta[resamples].mean(-1).quantile(torch.tensor([.025,.975])).tolist()
                pairs.append({'factor':changed[0],'a':a['id'],'b':b['id'],'phase':phase,
                              'b_minus_a_mm':float(delta.mean()),'take_bootstrap95':ci})
    interactions={}
    for phase in phases:
        delta=torch.tensor([scores['E3']['phases'][phase]['per_take'][t]['body_mpjpe_mm']
                            -scores['E1']['phases'][phase]['per_take'][t]['body_mpjpe_mm']
                            -scores['E2']['phases'][phase]['per_take'][t]['body_mpjpe_mm']
                            +scores['E0']['phases'][phase]['per_take'][t]['body_mpjpe_mm'] for t in sorted(values)])
        interactions[phase]={'difference_of_differences_mm':float(delta.mean()),
                            'take_bootstrap95':delta[resamples].mean(-1).quantile(torch.tensor([.025,.975])).tolist()}
    write_json(output/'summary.json',{'completed':True,'identity':ident,'scores':scores,'paired_edges':pairs,
        'interaction_E3_minus_E1_minus_E2_plus_E0':interactions,
        'holdout_used':False,'confidence_intervals':'exploratory, unadjusted for multiple comparisons'})
    lines=['# E7 最小适配因素对照结果','','同一12个dev take、3个采样seed；每个take内先平均帧和seed，再平均take。',
           '原地面组含标注/GT脚部回退，只作诊断。区间为探索性、未经多重比较校正。','',
           '| 配置 | Body mm | PA mm | Root mm | 速度误差 mm/s |','|---|---:|---:|---:|---:|']
    for row in rows:
        s=scores[row['id']]['phases']['all']['mean']
        lines.append(f"| {row['id']} | {s['body_mpjpe_mm']:.3f} | {s['pa_mpjpe_mm']:.3f} | {s['root_mm']:.3f} | {s['velocity_error_mmps']:.3f} |")
    lines += ['', '## 单因素结论（全时段，跨其余配置）','']
    for factor in factors:
        edges=[e for e in pairs if e['factor']==factor and e['phase']=='all']
        if not edges: continue
        deltas=[e['b_minus_a_mm'] for e in edges]
        lines.append(f'- {factor}：{len(edges)}个配对背景，B−A范围[{min(deltas):+.3f}, {max(deltas):+.3f}] mm；'
                     f"{'方向一致' if min(deltas)*max(deltas)>0 else '方向依赖其他设置，不能归为固定增益'}。具体方向、分阶段数值及区间见 [summary.json](summary.json)。")
    lines += [f"- 滑窗×地面交互：{interactions['all']['difference_of_differences_mm']:+.3f} mm，"
              f"95%区间{interactions['all']['take_bootstrap95']}。正值表示估计地面在滑窗条件下带来更大的误差增量。",
              '', '这些是冻结E7推理因素的结果；尚未评价P或重新训练G。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    marker=f'E7最小适配四组对照完成（{output.relative_to(ROOT) if output.is_absolute() else output}）'
    log=ROOT/'LOG.md'
    if marker not in log.read_text():
        from datetime import datetime
        from zoneinfo import ZoneInfo
        title=datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d--%H：%M')
        relative=os.path.relpath(output,ROOT)
        conclusion='\n'.join(lines[lines.index('## 单因素结论（全时段，跨其余配置）')+2:])
        with log.open('a') as f:
            f.write(f'\n## {title}：{marker}\n\n{conclusion}\n\n- 结果：[RESULTS.md]({relative}/RESULTS.md)；完整配对：[summary.json]({relative}/summary.json)。\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','prepare','verify','evaluate','summarize'])
    parser.add_argument('--plan',type=Path,default=ROOT/'config/e7_minimal_ablation_v1.json')
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards or args.batch_size<1:
        parser.error('Invalid shard or batch size.')
    os.chdir(ROOT);plan=json.loads(args.plan.read_text());output=Path(plan['output'])
    install_upstream(plan)
    import torch
    torch.set_num_threads(4)
    from egorecover.e7_ablation import matrix
    if args.command=='plan':
        rows=matrix();write_json(output/'matrix.json',rows)
        print(json.dumps({'cells':len(rows),'inference_routes':len({r['inference_id'] for r in rows}),
                          'matrix':str(output/'matrix.json')},indent=2));return
    ident=identity(plan,args.plan)
    if ident['checkpoint_sha256']!=plan['checkpoint_sha256'] or ident['smplx_sha256']!=plan['smplx_sha256']:
        raise ValueError('Model asset changed.')
    with torch.inference_mode():
        if args.command=='prepare': prepare(plan,output,ident)
        elif args.command=='verify': verify(plan,output,ident)
        elif args.command=='evaluate': evaluate(args,plan,output,ident)
        else: summarize(plan,output,ident)


if __name__=='__main__':main()
