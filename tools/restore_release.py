"""Restore published artifacts into a checkout and verify their SHA256 hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def destination(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'Unsafe artifact path: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes checkout: {relative}')
    return path


def restore_parts(root, item, manifest, path):
    """Join ordered transport parts without changing original checkpoint bytes."""
    from huggingface_hub import hf_hub_download
    kind = item.get('repo_type', 'dataset')
    revision = manifest['repositories'][kind]['revision']
    if not revision:
        raise ValueError('Parts repository revision is not pinned yet.')
    folder = root / 'data/.release-parts' / kind
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+'.release-', dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            for part in item['parts']:
                cached = destination(folder, part['path'])
                if not cached.exists():
                    hf_hub_download(repo_id=item['repo_id'], repo_type=kind,
                                    filename=part['path'], revision=revision, local_dir=folder)
                if cached.stat().st_size != part['bytes'] or digest(cached) != part['sha256']:
                    raise ValueError('Transport part size or SHA256 differs: '+part['path'])
                with cached.open('rb') as source:
                    shutil.copyfileobj(source, out, length=8*1024*1024)
        if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
            raise ValueError('Assembled artifact size or SHA256 differs: '+item['path'])
        if path.exists():
            raise ValueError('Destination appeared during assembly; refusing to overwrite it.')
        temporary.replace(path); temporary = None
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def restore_archive_member(root, item, manifest):
    from huggingface_hub import hf_hub_download
    archive = next(a for a in manifest['archives'] if a['path'] == item['archive']['path'])
    folder = root / 'data/.release-archives'
    cached = destination(folder, archive['path'])
    revision = manifest['repositories']['dataset']['revision']
    if not revision:
        raise ValueError('Archive repository revision is not pinned yet.')
    if cached.exists():
        if cached.stat().st_size != archive['bytes']:
            raise ValueError('Existing source archive has an unexpected size.')
    else:
        print('Downloading selected data archive once for restoration.', flush=True)
        if 'parts' in archive:
            restore_parts(root, archive, manifest, cached)
        else:
            hf_hub_download(repo_id=archive['repo_id'], repo_type='dataset', filename=archive['path'],
                            revision=revision, local_dir=folder)
    # Cache this process's verification; ZipFile also checks each member CRC.
    verified = getattr(restore_archive_member, '_verified', set())
    key = (str(cached), archive['sha256'], cached.stat().st_mtime_ns)
    if key not in verified:
        if digest(cached) != archive['sha256']:
            raise ValueError('Source archive SHA256 differs from the pinned official mirror.')
        verified.add(key); restore_archive_member._verified = verified
    path = destination(root, item['path']); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+'.release-', dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            with zipfile.ZipFile(cached) as archive_file:
                info = archive_file.getinfo(item['archive']['member'])
                if info.file_size != item['bytes']:
                    raise ValueError('Source member has an unexpected size.')
                with archive_file.open(info) as source:
                    shutil.copyfileobj(source, out, length=8*1024*1024)
        if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
            raise ValueError('Extracted source member SHA256 differs from the frozen local source.')
        if path.exists():
            raise ValueError('Destination appeared during source extraction; refusing to overwrite it.')
        temporary.replace(path); temporary = None
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'docs/release/artifacts.json')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--preset', choices=['core', 'g-smoke', 'g-evaluation', 'p-evaluation', 'training', 'data', 'all'], default='core')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text())
    files = [f for f in manifest['files'] if args.preset == 'all' or args.preset in f['presets']]
    selected_archives = {f['archive']['path'] for f in files if 'archive' in f}
    payload = sum(f['bytes'] for f in files if 'archive' not in f) + sum(a['bytes'] for a in manifest.get('archives', []) if a['path'] in selected_archives)
    print(f'{args.preset}: restore {len(files)} files / {sum(f["bytes"] for f in files) / 1024**3:.3f} GiB; download payload up to {payload / 1024**3:.3f} GiB', flush=True)
    if args.dry_run:
        for item in files:
            destination(root, item['path'])
        print('Paths validated; no downloads or changes.')
        return
    failures = []
    for index, item in enumerate(files, 1):
        path = destination(root, item['path'])
        valid = path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        if not valid and not args.verify_only:
            from huggingface_hub import hf_hub_download
            revision = manifest['repositories'][item['repo_type']]['revision']
            if not revision:
                raise ValueError('Release revision is not pinned yet.')
            if path.exists():
                raise ValueError(f'Refusing to overwrite a changed local artifact: {item["path"]}')
            if 'archive' in item:
                restore_archive_member(root, item, manifest)
            elif 'parts' in item:
                restore_parts(root, item, manifest, path)
            else:
                hf_hub_download(repo_id=item['repo_id'], repo_type=item['repo_type'],
                                filename=item['path'], revision=revision, local_dir=root)
            valid = path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        if not valid:
            failures.append(item['path'])
        if index % 100 == 0 or index == len(files):
            print(f'Checked {index}/{len(files)}; failures={len(failures)}', flush=True)
    if failures:
        raise SystemExit('Missing/changed artifacts:\n' + '\n'.join(failures))
    for alias in manifest['aliases']:
        path = destination(root, alias['path'])
        target = destination(root, alias['target'])
        if target.is_dir() and not args.verify_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                if path.resolve() != target.resolve():
                    raise ValueError(f'Unexpected existing alias: {alias["path"]}')
            elif path.exists():
                if path.resolve() != target.resolve():
                    raise ValueError(f'Expected a shared-directory symlink: {alias["path"]}')
            else:
                path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)
    print('All selected artifact sizes and hashes match.')


if __name__ == '__main__':
    main()
