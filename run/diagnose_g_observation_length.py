"""Frozen E7 21/80-frame diagnostic on preselected development frames only."""
import argparse
import json
from pathlib import Path
import time

import torch

from dataset.smpl_utils import get_smpl
from egorecover.bootstrap_shapes import load_e7_initializer
from egorecover.codec import MotionCodec
from egorecover.e7_ablation import conditions, matrix
from egorecover.evaluation_protocol import file_sha256
from egorecover.fk import FixedShapeFK
from egorecover.g_pretraining import PairedSequences, keyed_seed, native_states
from egorecover.prior_two_forward import state_map
from mydiffusion.flow_matching import FlowMatching
from run.e7_ablation import write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('exp/egorecover_g_pretraining/v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads((args.source / 'plan.json').read_text())
    names = json.loads((args.source / 'data/dev_monitor.json').read_text())['takes']
    data = PairedSequences(args.source, 'dev', take_names=names)
    codec = MotionCodec(data.stats)
    model = load_e7_initializer(cfg['e7_checkpoint'], device=args.device, weight_source='ema').eval()
    smpl = get_smpl().to(args.device).eval().requires_grad_(False)
    flow = FlowMatching()
    frames = [79, 109, 139, 169, 199]
    # A small, predeclared screen; the training queue must not inspect test scores.
    protocol = {'takes': data.names, 'frames': frames, 'lengths': [21, 80], 'draw': 1062,
                'nfe': 10, 'floor': 'cached known floor', 'beta': 'same cached startup beta',
                'noise': 'same noise per physical frame for the shared suffix',
                'decision': 'use80 if paired mean improves >2mm and a majority of takes improve',
                'manifest_sha256': data.manifest_sha,
                'e7_sha256': file_sha256(cfg['e7_checkpoint'])}
    write_json(args.output.with_suffix('.protocol.json'), protocol)
    rows = []
    begin = time.monotonic()
    for i, name in enumerate(data.names):
        case = {'observations': {'aria_traj_obs': data.head[i], 'img_feats': data.images[i],
                                'traj_available': torch.ones(200, dtype=torch.bool),
                                'img_available': torch.ones(200, dtype=torch.bool)},
                'supervision': {'floor_height': torch.tensor(0.)}}
        # The cache already subtracts the known floor. Use native encoding directly
        # to avoid floor_for's historical zero-floor fallback.
        from egorecover.e7_ablation import native_head_encoding
        jobs = []
        for t in frames:
            full_noise = torch.randn(80, 243, generator=torch.Generator().manual_seed(
                keyed_seed(data.ids[i], 1062, t)))
            for length in (21, 80):
                first = t - length + 1
                raw, anchor = native_head_encoding(data.head[i, first:t + 1], 0.)
                def pad(x):
                    return torch.cat((x, x.new_zeros(80 - length, *x.shape[1:])))
                valid = torch.arange(80) < length
                y = {'traj': pad(codec.normalize(raw, 'traj')),
                     'img_embs': pad(data.images[i, first:t + 1]),
                     'valid_frames': valid, 'valid_img_embs': valid,
                     'traj_mask': torch.zeros(80, dtype=torch.bool),
                     'img_mask': torch.zeros(80, dtype=torch.bool)}
                jobs.append((t, length, anchor, y, pad(full_noise[-length:])))
        y = {k: torch.stack([j[3][k] for j in jobs]).to(args.device) for k in jobs[0][3]}
        noise = torch.stack([j[4] for j in jobs]).to(args.device)
        codes = flow.sample_loop(model, noise.shape, {'y': y}, noise=noise, num_steps=10).float().cpu()
        fk = FixedShapeFK(smpl, data.beta[i].to(args.device))
        for j, (t, length, anchor, _, _) in enumerate(jobs):
            state = native_states(codec, codes[j, :length], anchor)
            last = state_map(state, lambda x: x[-1:].to(args.device))
            pred = fk(last)[0].cpu()
            gt = data.truth.joints[i, t, :, :3, 3]
            err = (pred - gt).norm(dim=-1) * 1000
            rows.append({'take': name, 'frame': t, 'length': length,
                         'body_mm': float(err.mean()), 'root_mm': float(err[0]), 'head_mm': float(err[15])})
        print(f'{i + 1}/{len(data.names)} {name} elapsed={time.monotonic() - begin:.1f}s', flush=True)
    per_take = {}
    for name in data.names:
        values = {length: sum(r['body_mm'] for r in rows if r['take'] == name and r['length'] == length) / len(frames)
                  for length in (21, 80)}
        per_take[name] = {'length21_mm': values[21], 'length80_mm': values[80],
                          'delta80_minus21_mm': values[80] - values[21]}
    diffs = torch.tensor([r['delta80_minus21_mm'] for r in per_take.values()])
    means = {length: sum(v[f'length{length}_mm'] for v in per_take.values()) / len(per_take) for length in (21, 80)}
    generator = torch.Generator().manual_seed(20260930)
    boot = diffs[torch.randint(len(diffs), (10000, len(diffs)), generator=generator)].mean(1)
    result = {'completed': True, 'protocol': protocol, 'means_mm': means, 'per_take': per_take,
              'delta80_minus21_mm': float(diffs.mean()), 'improved_takes': int((diffs < 0).sum()),
              'paired_take_bootstrap95_mm': torch.quantile(boot, torch.tensor([.025, .975])).tolist(),
              'use80': bool(diffs.mean() < -2 and (diffs < 0).sum() > len(diffs) / 2),
              'seconds': time.monotonic() - begin, 'rows': rows,
              'limitations': 'Sparse fixed dev frames and one draw; evidence about E7 observation length, not a G improvement claim.'}
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'per_take', 'protocol')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
