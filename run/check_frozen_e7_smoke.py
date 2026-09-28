"""Frozen E7 engineering rollouts on an audited small mismatch dataset.

This is not the official windowed E7 benchmark or trained P/Q evaluation.
Every action owns its predicted history; all siblings share a model startup.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from config.defaults import get_cfg_defaults, validate_e7
from dataset.smpl_utils import get_smpl
from egorecover.bootstrap_shapes import load_e7_initializer
from egorecover.checkpoint import load_e7_weights
from egorecover.codec import MotionCodec
from egorecover.data import open_dataset
from egorecover.evaluation_protocol import event_window, file_sha256, paired_fault_delta, phase_indices
from egorecover.history_flow import HistoryFlow
from egorecover.rollout import initialize_history, run_episode
from egorecover.smpl_evaluation import evaluate_saved_case, prepare_ground_truth
from model.history_uniegomotion import HistoryUniEgoMotion
from mydiffusion.flow_matching import FlowMatching


def save_report(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1062)
    take_selector = parser.add_mutually_exclusive_group(required=True)
    take_selector.add_argument('--take-index', type=int)
    take_selector.add_argument('--take-name')
    parser.add_argument('--actions', nargs='+', choices=('a11', 'a10', 'a01'), default=('a11', 'a10', 'a01'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = dict(completed=False, stage='loading', scope='frozen_E7_engineering_smoke_only',
                  history_source='model_predictions_only; no resets', results=[],
                  sampling_seed=args.seed, take_index=args.take_index, take_name=args.take_name,
                  interventions='constant action for each separate closed loop; not same-history counterfactual utility labels',
                  limitations='No trained P/Q or history branch; single-frame G differs from official 80-frame E7 evaluation.',
                  checkpoint=str(args.checkpoint.resolve()), dataset_spec_sha256=args.spec_sha256)
    save_report(report_path, report)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        dataset, _ = open_dataset(signal=args.signal, spec_sha256=args.spec_sha256)
        takes = sorted({r['base_take_name'] for r in dataset.records})
        take = args.take_name if args.take_name is not None else takes[args.take_index]
        if take not in takes:
            raise ValueError(f'Unknown pilot take: {take}')
        variants = ('clean', 'freeze_3s', 'drift_0p03mps')
        records = [next(r for r in dataset.records if r['base_take_name'] == take and
                        r['variant_name'] == name) for name in variants]
        if any(r['num_frames'] != 200 or r['bootstrap_frames'] != 20 for r in records):
            raise ValueError('Smoke protocol requires 200 frames with 20 clean startup frames.')
        stats_path = dataset.source.root / 'uniegomotion/v4_beta_ee_train_stats.pt'
        codec = MotionCodec(torch.load(stats_path, weights_only=False, map_location='cpu'),
                            reference_mode='planar').to(device)
        cfg = get_cfg_defaults()
        cfg.merge_from_file('config/e7.yaml')
        validate_e7(cfg)
        model = HistoryUniEgoMotion(cfg).to(device).eval()
        migration = load_e7_weights(model, args.checkpoint, weight_source='ema')
        model.requires_grad_(False)
        initializer = load_e7_initializer(args.checkpoint, device=device, weight_source='ema')
        prefix = {k: v.to(device) for k, v in dataset.observations(
            records[0]['variant_id'], as_of=19, history_frames=20).items()}
        buffer, floor, _ = initialize_history(initializer, FlowMatching(), codec, prefix, seed=args.seed)
        world = torch.stack([s.joints[..., :3, 3] for s in buffer.states]).cpu()
        world[..., 2] += floor
        startup = dict(normalized_motion=buffer.bootstrap_motion.cpu(),
                       initial_reference=buffer.initial_reference.cpu(), beta_boot=buffer.beta_boot.cpu(),
                       references=torch.stack([s.reference for s in buffer.states]).cpu(),
                       world_joints=world, floor_estimate_m=floor, sampling_seed=args.seed)
        torch.save(startup, args.output / 'startup.pt')
        del initializer, buffer
        report.update(stage='rollout', take=take, reference_mode='planar', weight_source='ema',
                      checkpoint_sha256=file_sha256(args.checkpoint), migration=asdict(migration),
                      stats_sha256=file_sha256(stats_path), startup_sha256=file_sha256(args.output/'startup.pt'),
                      source_mode='gaussian', sigma=1.0, nfe=10, motion_fps=10,
                      online_gt_body_shape_floor=False, generator_trainable_parameters=0)
        save_report(report_path, report)
        print(f'START {take}: shared E7 EMA startup ready', flush=True)
        flow = HistoryFlow(source_mode='gaussian', sigma=1.0)
        # Finish every inference case before loading labels or SMPL-X.
        for action in args.actions:
            for record in records:
                variant = record['variant_name']
                print(f'ROLLOUT {take}/{action}/{variant}', flush=True)
                requested = []

                def observations(**kwargs):
                    requested.append((kwargs['as_of'], kwargs['history_frames']))
                    return dataset.observations(record['variant_id'], **kwargs)

                started = time.monotonic()
                saved = run_episode(model, flow, None, None, codec, observations,
                                    num_frames=200, action=action, seed=args.seed, startup=startup)
                if requested != [(19, 20)] + [(t, 1) for t in range(20, 200)]:
                    raise ValueError('Online observation requests violate the causal protocol.')
                for key in ('dense_world_joints', 'normalized_motion', 'committed_motion', 'references'):
                    if not torch.isfinite(saved[key]).all():
                        raise ValueError(f'Nonfinite {key} in {action}/{variant}')
                path = args.output / f'{action}_{variant}.pt'
                torch.save(saved, path)
                report['results'].append(dict(action=action, variant=variant, trace=str(path.resolve()),
                    trace_sha256=file_sha256(path), finite=True, frames=180,
                    elapsed_seconds=time.monotonic()-started, causal_observation_requests_verified=True))
                save_report(report_path, report)
                print(f'SAVED {path}: 180 predicted frames', flush=True)
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        report['stage'] = 'offline_smplx'
        save_report(report_path, report)
        smpl = get_smpl().to(device).eval().requires_grad_(False)
        ground_truth = prepare_ground_truth(smpl, dataset.supervision(records[0]['variant_id']),
                                            list(range(20, 200)))
        report['gt_asset_audit'] = ground_truth['asset_audit']
        report['smplx_asset_sha256'] = file_sha256(Path('body_models/smplx/SMPLX_NEUTRAL.npz'))
        by_case = {}
        for row in report['results']:
            record = next(r for r in records if r['variant_name'] == row['variant'])
            window = event_window(record, records)
            saved = torch.load(row['trace'], weights_only=True, map_location='cpu')
            evaluated = evaluate_saved_case(smpl, codec, saved, ground_truth, fault_window=window)
            sections = phase_indices(saved['frame_indices'], window)
            error = evaluated['per_frame_smpl22_mm']
            row.update(fault_window=list(window) if window else None, **evaluated)
            row['phase_smpl22_mm'] = {
                name: sum(error[i] for i in indices)/len(indices) if indices else None
                for name, indices in sections.items()}
            by_case[row['action'], row['variant']] = row
            print(f"FK {row['action']}/{row['variant']}: {row['metrics']['mpjpe_body_m']*1000:.2f} mm", flush=True)
            save_report(report_path, report)
        for row in report['results']:
            if row['variant'] != 'clean':
                row['paired_fault_delta'] = paired_fault_delta(
                    row['per_frame_smpl22_mm'], by_case[row['action'], 'clean']['per_frame_smpl22_mm'],
                    list(range(20, 200)), row['fault_window'], fps=10)
        report.update(stage='complete', completed=True)
        save_report(report_path, report)
        print(f'COMPLETE {report_path}', flush=True)
    except Exception as error:
        report.update(stage='failed', error=f'{type(error).__name__}: {error}')
        save_report(report_path, report)
        raise


if __name__ == '__main__':
    main()
