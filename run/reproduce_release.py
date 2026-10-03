"""Evaluate published G or expanded P without changing frozen experiment identities."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path('/gaozt-test1/sxh/EgoRecover')


@contextmanager
def relocated_original_prior():
    """Resolve the one legacy data pointer used by ExpandedPriorSequences.

    Only this loader reference changes temporarily. Raw cache contents, identity
    comparisons, data hashes and model source files remain unchanged.
    """
    import egorecover.prior_expanded as expanded
    from egorecover.prior_two_forward import PriorSequences

    class RelocatedPriorSequences(PriorSequences):
        def __init__(self, directory, group):
            p = Path(directory)
            try:
                p = ROOT / p.relative_to(LEGACY_ROOT)
            except ValueError:
                pass
            super().__init__(p, group)

    previous = expanded.PriorSequences
    expanded.PriorSequences = RelocatedPriorSequences
    try:
        yield
    finally:
        expanded.PriorSequences = previous


def evaluate_g(args):
    from config.defaults import get_cfg_defaults
    from dataset.smpl_utils import get_smpl
    from egorecover.codec import MotionCodec
    from egorecover.evaluation_protocol import file_sha256
    from egorecover.g_pretraining import PairedSequences
    from egorecover.observation_reference import GROUPS
    from model.observation_reference_g import ObservationReferenceG
    from run.train_g_pretraining import load_prior
    from run.train_observation_reference import evaluate

    output = ROOT / 'exp/egorecover_observation_reference/v2'
    path = output / args.group / f'seed{args.seed}/best.pt'
    selected = json.loads((output / 'evaluation/selected_checkpoints.json').read_text())
    if file_sha256(path) != selected[f'{args.group}/seed{args.seed}']:
        raise ValueError('Selected G checkpoint differs from the recorded evaluation.')
    data = PairedSequences(output, 'test', limit=1 if args.smoke else None)
    codec = MotionCodec(data.stats).to(args.device)
    smpl = get_smpl().to(args.device).eval().requires_grad_(False)
    prior = load_prior(output / 'prior/prior.pt', args.device, manifest_sha=data.manifest_sha)
    length, anchored = GROUPS[args.group]
    model = ObservationReferenceG(get_cfg_defaults(), observation_length=length, anchored=anchored).to(args.device)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    summary, _ = evaluate(model, prior, data, codec, smpl, torch.device(args.device), max_frames=22 if args.smoke else 200)
    recorded = json.loads((output / args.group / f'seed{args.seed}/test/report.json').read_text())['body_mm']
    return {'model': 'G', 'group': args.group, 'seed': args.seed, 'smoke': args.smoke,
            'summary': summary, 'recorded_full_body_mm': recorded,
            'full_body_difference_mm': None if args.smoke else summary['scores']['body_mm']['mean'] - recorded,
            'checkpoint_sha256': file_sha256(path)}


def evaluate_p(args):
    from dataset.smpl_utils import get_smpl
    from egorecover.codec import MotionCodec
    from egorecover.prior import HistoryPrior
    from egorecover.prior_cv_residual import evaluate_single
    from egorecover.prior_expanded import ExpandedPriorSequences
    from egorecover.evaluation_protocol import file_sha256

    with relocated_original_prior():
        data = ExpandedPriorSequences(ROOT / 'exp/egorecover_prior_data_budget/v1/data', 'dev')
    path = ROOT / f'exp/egorecover_prior_data_budget/v1/train/D_s{args.seed}/prior.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt['original_cache_sha256'] != data.identity['original_cache_sha256']:
        raise ValueError('P and dev cache have different provenance.')
    model = HistoryPrior().to(args.device); model.base_mode = 'constant_velocity'
    model.load_state_dict(ckpt['state_dict'], strict=True)
    smpl = get_smpl().to(args.device).eval().requires_grad_(False)
    scores, _ = evaluate_single(model, data, MotionCodec(data.stats).to(args.device), smpl, torch.device(args.device), 32)
    recorded = json.loads((path.parent / 'report.json').read_text())['results']['prior']['single']['fk22_mm']['mean']
    return {'model': 'P', 'seed': args.seed, 'scores': scores, 'recorded_full_body_mm': recorded,
            'full_body_difference_mm': scores['fk22_mm']['mean'] - recorded,
            'checkpoint_sha256': file_sha256(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['G', 'P'], default='G')
    p.add_argument('--group', choices=['R0', 'R1', 'R2', 'R3'], default='R0')
    p.add_argument('--seed', type=int, choices=[62, 63, 64], default=62)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='G only: one take and two generated frames; not a full score')
    p.add_argument('--output', type=Path, default=ROOT / 'exp/reproduction/report.json')
    args = p.parse_args()
    if args.smoke and args.model != 'G': p.error('--smoke is available only for G')
    from egorecover.evaluation_protocol import file_sha256
    release = json.loads((ROOT / 'docs/release/artifacts.json').read_text())
    for asset in release.get('required_local_assets', []):
        directory = os.environ.get(asset['directory_environment_variable'])
        path = Path(directory) / asset['filename'] if directory else ROOT / asset['path']
        if not path.is_file() or path.stat().st_size != asset['bytes'] or file_sha256(path) != asset['sha256']:
            raise ValueError(f'Authorized local asset missing or different from the experiment: {asset["filename"]}')
    torch.set_num_threads(4)
    started = time.monotonic()
    result = evaluate_g(args) if args.model == 'G' else evaluate_p(args)
    result['seconds'] = time.monotonic() - started
    result['torch_version'] = torch.__version__; result['device'] = args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'summary', 'scores'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
