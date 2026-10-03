"""Copy current source and lightweight records into an existing release clone.

No commits or pushes are performed. Markdown artifact links are rewritten only
in the staged public copies; immutable source records remain unchanged.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_inventory', ROOT / 'tools/audit_release_inventory.py')
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    args = parser.parse_args(); stage = args.staging.resolve()
    if stage == ROOT or stage.is_relative_to(ROOT) or not (stage / '.git').is_dir():
        raise ValueError('Use an existing independent Git clone outside the source workspace.')
    candidates = set()
    for folder, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', '.venv'} and not (Path(folder) / d).is_symlink()]
        for name in names:
            src = Path(folder) / name
            if not src.is_file() or src.is_symlink(): continue
            rel = src.relative_to(ROOT).as_posix(); c = AUDIT.category(rel)
            if c in {'github_source', 'github_verification', 'github_experiment_metadata', 'hf_source_snapshots', 'hf_logs_private'}:
                candidates.add(rel)
            elif rel == 'cv.md' or rel.startswith('图片和附件/'):
                candidates.add(rel)
            elif c == 'hf_cache_review' and src.suffix in AUDIT.TEXT and name in {'manifest.json', 'split.json', 'dev_monitor.json', 'report.json'} and '/episodes/' not in rel:
                candidates.add(rel)
            elif c in {'hf_other_artifacts_review', 'local_verification_binary'} and src.suffix in {'.png', '.pdf', '.xml'}:
                candidates.add(rel)
    audit_dir = ROOT / 'exp/release_audit/20261002'
    for src in audit_dir.iterdir():
        if src.suffix in {'.json', '.csv'}: candidates.add(src.relative_to(ROOT).as_posix())
    tracked = subprocess.check_output(['git', '-C', str(stage), 'ls-files', '-z']).decode().split('\0')
    for rel in filter(None, tracked):
        if not (ROOT / rel).exists() and rel not in candidates:
            p = stage / rel
            if p.is_file(): p.unlink()
    total = 0
    for rel in sorted(candidates):
        src = ROOT / rel; target = stage / rel
        if src.stat().st_size >= 50 * 1024**2:
            raise ValueError(f'Large file unexpectedly selected for GitHub: {rel}')
        if src.suffix in AUDIT.TEXT or src.suffix == '.log':
            content = src.read_text(errors='replace')
            patterns = [name for name, pat in AUDIT.KEY_PATTERNS.items() if pat.search(content)]
            if patterns: raise ValueError(f'Credential pattern in {rel}: {patterns}')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, target); total += src.stat().st_size
    manifest = json.loads((ROOT / 'docs/release/artifacts.json').read_text())
    artifacts = {f['path']: f for f in manifest['files']}
    archives = {a['path']: a for a in manifest.get('archives', [])}
    mapping = {'DATASET.md':'docs/data/dataset.md', 'EGORECOVER.md':'docs/design/interfaces.md',
               'EXPERIMENTS.md':'docs/archive/engineering-experiments.md', 'ICCV2027.md':'docs/archive/iccv2027-draft.md',
               'BLUE_PRINT.md':'docs/design/blueprint.md'}
    replacements=[]; broken=[]
    for rel in sorted(candidates):
        if not rel.endswith('.md'): continue
        p=stage/rel; omitted=[]
        def replace(match):
            label, raw = match.group(1), match.group(2)
            if raw.startswith(('http://','https://','mailto:','#')):return match.group(0)
            target, separator, anchor=raw.partition('#');target=unquote(target)
            prefixes=('/gaozt-test1/sxh/EgoRecover/','/home/ld666/projects/EgoRecover/')
            prefix=next((s for s in prefixes if target.startswith(s)),None)
            if prefix:
                target=target[len(prefix):]; absolute=stage/target
            else:absolute=Path(os.path.normpath(str(p.parent/target)))
            if 'UEM-update-original/' in str(absolute):
                suffix=str(absolute).split('UEM-update-original/',1)[1]
                link='https://github.com/sxh-kk/UEM-update/blob/156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08/'+suffix
            elif absolute.is_relative_to(stage):
                name=absolute.relative_to(stage).as_posix()
                if name in candidates:
                    if not prefix:return match.group(0)
                    link=os.path.relpath(absolute,p.parent)
                elif name in mapping and (stage/mapping[name]).exists():
                    link=os.path.relpath(stage/mapping[name],p.parent)
                else:
                    for alias in manifest['aliases']:
                        if name.startswith(alias['path']+'/'):name=alias['target']+name[len(alias['path']):]
                    item=artifacts.get(name)
                    if item:
                        prefix='https://huggingface.co/datasets/' if item['repo_type']=='dataset' else 'https://huggingface.co/'
                        revision=manifest['repositories'][item['repo_type']]['revision'] or 'main'
                        archived=archives.get(item.get('archive',{}).get('path'), {})
                        if 'parts' in item or 'parts' in archived:
                            link=prefix+item['repo_id']+'/blob/'+revision+'/release-index.json'
                        else:
                            link=prefix+item['repo_id']+'/resolve/'+revision+'/'+item.get('archive',{}).get('path',item['path'])
                    elif name.startswith(('exp/','data/','body_models/','data_pipeline/test_results')):
                        link=os.path.relpath(stage/'docs/release/migration.md',p.parent);anchor='为什么选择这些checkpoint';separator='#';omitted.append(name)
                    else:
                        broken.append({'document':rel,'link':raw});return match.group(0)
            else:
                broken.append({'document':rel,'link':raw});return match.group(0)
            replacements.append({'document':rel,'old':raw,'new':link+(separator+anchor if separator else '')})
            return f'[{label}]({link}{separator}{anchor})'
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',replace,p.read_text())
        if omitted:
            text+='\n> 迁移说明：本文部分原产物未选入发布包，相应链接指向迁移范围说明；具体原路径和原因见发布排除清单。\n'
        p.write_text(text)
    report={'source_files':len(candidates),'source_bytes_before_link_rewrite':total,'artifact_link_replacements':replacements,'remaining_missing_links':broken}
    (stage/'docs/release/github-staging.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (stage.parent/'github-selected-files.json').write_text(json.dumps(sorted(candidates),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files':len(candidates),'MiB':round(total/1024**2,3),'rewritten_links':len(replacements),'unresolved_links':len(broken)},ensure_ascii=False))
    for item in broken[:20]:print(json.dumps(item,ensure_ascii=False))


if __name__=='__main__':main()
