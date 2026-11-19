"""Reuse the finished 10-epoch runs on the full data as version v2_e10 (no new GPU time for them).

The archive phase1_old_lr0001.tgz (runs from 2026-11-06/07) has the full data with Antoine's stream
(Gene - causes_or_contributes_to_condition - Disease, cutoff 2010), Rules 1 and 2, 10 epochs, and every other
setting equal to today's preset 'phase1'. Their learning rate was PyKEEN's default for Adam (0.001), which is
the chosen learning rate (configs.TRAINING) of TransE, TransH, DistMult and RotatE, so those runs are exactly
version v2_e10. ComplEx's chosen rate is 0.01, so its old runs are NOT imported and are run again.

A run is imported only if every setting in its config.json matches, and its data counts (graph rows,
reference size, stream size) match today's code. Imported runs get 'imported_from' in summary.json.
The cases in that archive monitored 11 disease types (earlier setup), so they are not imported.

    python import_old_runs.py --archive ~/phase1_old_lr0001.tgz            # dry run: what would be imported
    python import_old_runs.py --archive ~/phase1_old_lr0001.tgz --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402

VERSION = 'v2_e10'
RULES = {'sphere1': 'sphere', 'always': 'always'}
MODELS = [m for m in configs.MODELS if configs.TRAINING[('full', m)]['learning_rate'] == 0.001]
EXPECTED = {'epochs': 10, 'real_dim': 64, 'batch_size': 2048, 'seed': 42, 'window_size': 250, 'nu': 0.1,
            'gamma': 'scale', 'alpha': 0.025, 'standardize': True, 'keep_fraction_after_drift': 0.1,
            'cutoff_year': 2010, 'target_type': configs.GENE_DISEASE, 'data_file': 'drifting_edges_timestamp.tsv'}


def check(config, summary, reference):
    problems = [f'{k}: {config.get(k)!r} != {v!r}' for k, v in EXPECTED.items() if config.get(k) != v]
    if config.get('learning_rate') not in (None, 0.001):
        problems.append(f"learning_rate {config.get('learning_rate')!r}")
    for k in ('initial_kg_rows', 'initial_reference_size', 'future_rows'):
        if reference and summary.get(k) != reference.get(k):
            problems.append(f'{k}: {summary.get(k)} != {reference.get(k)} (today)')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    out_dir = HERE / configs.VERSIONS[VERSION]['runs_dir']
    # Data counts of today's code, from any finished full-data run of today (v1).
    reference = None
    for f in sorted((HERE / configs.VERSIONS['v1']['runs_dir']).glob('full__*/summary.json')):
        reference = json.loads(f.read_text(encoding='utf-8'))
        break
    if reference is None:
        print('note: no finished v1 full run here to compare the data counts with; only the settings are checked')

    wanted = {f'runs/full__{m}__{c}__{tag}__phase1' for m in MODELS for c in configs.COMPOSITIONS for tag in RULES}
    with tarfile.open(args.archive) as tar, tempfile.TemporaryDirectory() as tmp:
        members = [m for m in tar.getmembers() if '/'.join(Path(m.name).parts[:2]) in wanted]
        tar.extractall(tmp, members=members, filter='data')
        imported, skipped = 0, 0
        for old_name in sorted(wanted):
            src = Path(tmp) / old_name
            parts = src.name.split('__')
            new_name = configs.run_id('full', parts[1], parts[2], 'phase1', RULES[parts[3]], 1.0, version=VERSION)
            if not (src / 'summary.json').exists():
                print(f'[missing] {src.name}: not finished in the archive; it will be run')
                skipped += 1
                continue
            config = json.loads((src / 'config.json').read_text(encoding='utf-8'))
            summary = json.loads((src / 'summary.json').read_text(encoding='utf-8'))
            problems = check(config, summary, reference)
            if problems:
                print(f'[differs] {src.name}: ' + '; '.join(problems) + ' -> not imported, it will be run')
                skipped += 1
                continue
            if (out_dir / new_name / 'summary.json').exists():
                print(f'[exists] {new_name}')
                continue
            print(f'[import] {src.name} -> {configs.VERSIONS[VERSION]["runs_dir"]}/{new_name}')
            imported += 1
            if not args.apply:
                continue
            dst = out_dir / new_name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            note = {'imported_from': f'{args.archive.name}:{old_name}', 'run': new_name, 'version': VERSION,
                    'embed_update': 'scratch', 'learning_rate': 0.001}
            summary.update(note)
            config.update({k: v for k, v in note.items() if k != 'run'})
            (dst / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
            (dst / 'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    print(f'{imported} run(s) {"imported" if args.apply else "to import (dry run; add --apply)"}, {skipped} not importable')
    return 0


if __name__ == '__main__':
    sys.exit(main())
