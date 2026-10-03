"""Prepare immutable HF payloads and a checksummed release index; no uploads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'sxhkk/EgoRecover'
DATASET = 'sxhkk/EgoRecover-data'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selection(row, test_ids):
    path = row['path']; p = Path(path); c = row['category']
    if row['kind'] != 'file':
        return None
    smoke = any(s in path for s in ('/verification/', '/smoke/', '_smoke', 'interrupted')) or '/egorecover_observation_reference/v1/' in path
    if c == 'baseline_checkpoint_review':
        return 'model', 'original_e7_ema_and_training_state', ['training']
    if c == 'hf_model_checkpoints' and not smoke and p.name != 'last.pt':
        core = bool(re.fullmatch(r'exp/egorecover_observation_reference/v2/R0/seed6[234]/best\.pt', path)) or path == 'exp/egorecover_g_pretraining/v1/prior/prior.pt'
        formal_g = bool(re.fullmatch(r'exp/(egorecover_g_pretraining/v1/T[0-3]|egorecover_observation_reference/v2/R[0-3])/seed6[234]/best\.pt', path))
        presets = ['training']
        if p.name == 'prior.pt': presets.append('p-evaluation')
        if formal_g or path == 'exp/egorecover_g_pretraining/v1/prior/prior.pt': presets.append('g-evaluation')
        if core: presets.append('core')
        return 'model', 'initial_for_paired_controls' if p.name == 'initial.pt' else 'dev_selected_model', presets
    if c == 'hf_training_resume' and (re.fullmatch(r'exp/egorecover_observation_reference/v2/R0/seed6[234]/resume\.pt', path) or re.fullmatch(r'exp/egorecover_prior_data_budget/v1/train/D_s6[234]/resume\.pt', path)):
        return 'model', 'current_training_resume', ['training']
    if c in {'external_dataset_reference', 'derived_dataset_review'}:
        presets = ['data', 'training']
        if p.name == 'v4_beta_ee_train_stats.pt': presets += ['core', 'p-evaluation', 'g-evaluation']
        return 'dataset', 'official_processed_source' if c == 'external_dataset_reference' else 'derived_mismatch_data', presets
    if c == 'hf_cache_review':
        if re.search(r'/histories/(?:train|dev)_shard\d+/', path) or '/jobs/' in path:
            return None
        presets = ['data', 'training']
        if '/egorecover_g_pretraining/v1/data/' in path:
            if '/episodes/' not in path or p.stem in test_ids: presets.append('g-evaluation')
        if '/egorecover_prior_two_forward/v1/data/' in path or '/egorecover_prior_data_budget/v1/data/' in path: presets.append('p-evaluation')
        return 'dataset', 'verified_prepared_cache', presets
    if c == 'hf_tensor_artifacts_review' and not smoke:
        final = '/test/best_nfe10_draw1062.pt' in path or p.name in {'dev_prior.pt', 'dev_hold.pt', 'dev_constant_velocity.pt', 'bootstrap.pt'} or path.startswith(('exp/e7_official_full/', 'exp/e7_official_key/'))
        if final:
            presets = ['data']
            if p.name == 'bootstrap.pt': presets.append('training')
            return 'dataset', 'final_prediction_or_metric_evidence' if p.name != 'bootstrap.pt' else 'model_startup_cache', presets
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=ROOT / 'exp/release_audit/20261002/files.csv')
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    staging = args.staging.resolve(); staging.mkdir(parents=True, exist_ok=True)
    if staging.is_relative_to(ROOT): raise ValueError('Staging must be outside the source workspace.')
    rows = list(csv.DictReader(args.inventory.open()))
    source_manifest = json.loads((ROOT / 'exp/egorecover_g_pretraining/v1/data/manifest.json').read_text())
    test_ids = {r['episode_id'] for r in source_manifest['records'] if r['group'] == 'test'}
    chosen = [(r, selection(r, test_ids)) for r in rows]
    chosen = [(r, selected) for r, selected in chosen if selected]
    print(f'Hashing {len(chosen)} files, {sum(int(r["size_bytes"]) for r, _ in chosen)/1024**3:.3f} GiB', flush=True)
    def prepare(pair):
        row, (kind, role, presets) = pair; path = row['path']; src = ROOT / path
        before = src.stat(); checksum = sha256(src); after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError(f'Source changed while hashing: {path}')
        target = staging / kind / path; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if before.st_size >= 1024**2:
                os.link(src, target)
            else:
                shutil.copy2(src, target)
        if target.stat().st_size != before.st_size: raise ValueError(f'Staging size differs: {path}')
        return {'path': path, 'repo_id': MODEL if kind == 'model' else DATASET, 'repo_type': kind,
                'bytes': before.st_size, 'sha256': checksum, 'role': role, 'presets': sorted(set(presets))}
    files = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, item in enumerate(pool.map(prepare, chosen), 1):
            files.append(item)
            if index % 1000 == 0: print(f'Hashed/staged {index}/{len(chosen)}', flush=True)
    aliases = json.loads((ROOT / 'exp/release_audit/20261002/symlink-restore.json').read_text())['aliases']
    manifest = {'schema_version': 'egorecover-release-v1', 'created_at': datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
                'github_repo': 'sxh-kk/EgoRecover', 'repositories': {
                    'model': {'repo_id': MODEL, 'revision': None, 'public': True},
                    'dataset': {'repo_id': DATASET, 'revision': None, 'public': True}},
                'aliases': [{'path': a['path'], 'target': a['canonical_target']} for a in aliases],
                'files': sorted(files, key=lambda x: x['path'])}
    (ROOT / 'docs/release/artifacts.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    selected_paths = {x['path'] for x in files}
    with (staging / 'excluded-files.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['path', 'category', 'bytes', 'reason'])
        for r in rows:
            if r['path'] not in selected_paths:
                c = r['category']; reason = ('GitHub lightweight record or source' if c.startswith('github_') or c in {'hf_source_snapshots', 'hf_logs_private'} else
                    'licensed SMPL-X asset: user download' if c == 'restricted_smplx' else
                    'initial/last/smoke/failed/interrupted or inactive optimizer state' if c in {'hf_model_checkpoints', 'hf_training_resume'} else
                    'intermediate, redundant shard, synthetic diagnostic or local-only file')
                w.writerow([r['path'], c, r['size_bytes'], reason])
    for kind in ['model', 'dataset']:
        subset = [x for x in files if x['repo_type'] == kind]
        print(kind, len(subset), round(sum(x['bytes'] for x in subset)/1024**3, 6), 'GiB', flush=True)
    print('Prepared locally. Nothing uploaded.', flush=True)


if __name__ == '__main__':
    main()
