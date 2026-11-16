"""Run a grid of experiments for one version (meant for tmux on the server).

    python run_grid.py --version v1 --grid sphere always --parallel 3
    python run_grid.py --version v3 --grid sphere always --datasets case1 case2 --parallel 3
    python run_grid.py --version v3 --grid sweep every --parallel 3   # sweep needs the version's always runs
    python run_grid.py --version v2 --grid sphere --dry-run

Version = how the KG is re-embedded (configs.VERSIONS). Grid = retraining rule (configs.RULES):
sphere (Rule 1), always (Rule 2), sweep (Rule 3, smaller sphere sizes), every (Rule 4, every X triples).
Without --datasets each rule runs on the datasets planned for the version (configs.PLAN).

Finished runs (summary.json present) are skipped, so the command can be restarted after a
crash or a lost connection. Each run's console output goes to <out-dir>/_logs/<run>.log.

Sphere sweep: for every dataset and composition, the center-radius ratios logged by the version's
five always-retrain runs (one ratio per new-entity re-embed, five models pooled) give the k values
at which the rule would have retrained on ~10%, ~50%, ~90% of new entities. The chosen k values are
saved to <out-dir>/_sweep_plan.csv.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402

MIN_RATIOS_FOR_SWEEP = 5
PRESET = 'phase1'


def entry_name(entry, version):
    return configs.run_id(entry['dataset'], entry['model'], entry['composition'], PRESET, entry['policy'],
                          entry['k'], entry['tag'], version=version, retrain_every=entry['retrain_every'])


def sweep_entries(version, datasets, out_dir):
    entries, plan = [], []
    for dataset in datasets:
        for composition in configs.COMPOSITIONS:
            ratios, missing = [], []
            for model in configs.MODELS:
                base = configs.run_id(dataset, model, composition, PRESET, 'always', 1.0, version=version)
                shift_path = out_dir / base / 'embedding_shift_metrics.tsv'
                if not (out_dir / base / 'summary.json').exists():
                    missing.append(base)
                    continue
                shifts = pd.read_csv(shift_path, sep='\t') if shift_path.stat().st_size > 1 else pd.DataFrame()
                if not shifts.empty:
                    unseen = shifts[shifts['trigger_reason'].eq('unseen_element')]
                    ratios += unseen['center_radius_ratio'].replace([np.inf, -np.inf], np.nan).dropna().tolist()
            if missing:
                print(f'[sweep] {dataset} {composition}: always runs not finished ({len(missing)} of '
                      f'{len(configs.MODELS)}), e.g. {missing[0]}; skipped')
                continue
            ratios = np.asarray(ratios)
            if len(ratios) < MIN_RATIOS_FOR_SWEEP:
                print(f'[sweep] {dataset} {composition}: only {len(ratios)} re-embed ratios; skipped')
                continue
            for tag, share in configs.SWEEP_TARGETS.items():
                k = float(np.quantile(ratios, 1.0 - share))
                entries += [{'dataset': dataset, 'model': model, 'composition': composition, 'policy': 'sphere',
                             'k': k, 'tag': tag, 'retrain_every': None} for model in configs.MODELS]
                plan.append({'version': version, 'dataset': dataset, 'composition': composition, 'tag': tag,
                             'target_retrain_share': share, 'sphere_k': k, 'n_ratios': len(ratios),
                             'share_in_always_runs': float((ratios > k).mean())})
    if plan:
        path = out_dir / '_sweep_plan.csv'
        new = pd.DataFrame(plan)
        if path.exists():
            old = pd.read_csv(path)
            keys = ['dataset', 'composition', 'tag']
            if set(keys) <= set(old.columns):
                keep = ~old.set_index(keys).index.isin(new.set_index(keys).index)
                new = pd.concat([old[keep], new], ignore_index=True)
        new.to_csv(path, index=False)
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', default='v1', choices=list(configs.VERSIONS))
    p.add_argument('--grid', required=True, nargs='+', choices=configs.RULES)
    p.add_argument('--datasets', nargs='+', default=None, choices=list(configs.DATASETS),
                   help='only these datasets (default: the datasets planned for the version and rule)')
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--threads', type=int, default=None, help='CPU threads per run (default: 8 // parallel)')
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=None, help="default: the version's folder (runs, runs_v2, runs_v3)")
    p.add_argument('--adazor-src', type=Path, default=HERE.parent / 'adazor' / 'src')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER, help='passed to drift_sphere.py after --')
    args = p.parse_args()
    extra = [a for a in args.extra if a != '--']
    out_dir = args.out_dir or HERE / configs.VERSIONS[args.version]['runs_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rule in args.grid:
        datasets = args.datasets or configs.PLAN[args.version][rule]
        entries = (sweep_entries(args.version, datasets, out_dir) if rule == 'sweep'
                   else configs.static_entries(rule, datasets))
        for entry in entries:
            if entry not in jobs:
                jobs.append(entry)

    todo = []
    for entry in jobs:
        name = entry_name(entry, args.version)
        if (out_dir / name / 'summary.json').exists():
            print(f'[skip] {name}')
            continue
        cmd = [sys.executable, str(HERE / 'drift_sphere.py'), '--dataset', entry['dataset'], '--model', entry['model'],
               '--composition', entry['composition'], '--preset', PRESET, '--version', args.version,
               '--retrain-policy', entry['policy'], '--sphere-k', repr(float(entry['k'])), '--run-name', name,
               '--data-dir', str(args.data_dir), '--out-dir', str(out_dir), '--adazor-src', str(args.adazor_src)]
        if entry['retrain_every'] is not None:
            cmd += ['--retrain-every', str(entry['retrain_every'])]
        todo.append((name, cmd + extra))

    print(f'{len(todo)} run(s) to do, {args.parallel} at a time', flush=True)
    if args.dry_run:
        for name, cmd in todo:
            print(name, '\n   ', ' '.join(cmd))
        return 0

    threads = str(args.threads or max(1, 8 // args.parallel))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    log_dir = out_dir / '_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    running, failed = [], []
    while todo or running:
        while todo and len(running) < args.parallel:
            name, cmd = todo.pop(0)
            log = open(log_dir / f'{name}.log', 'a', encoding='utf-8')
            print(f'[start] {name}  ({time.strftime("%H:%M:%S")})', flush=True)
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, time.time()))
        time.sleep(5)
        for item in list(running):
            name, proc, log, t0 = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            status = 'done' if proc.returncode == 0 else f'FAILED (exit {proc.returncode}, see _logs/{name}.log)'
            if proc.returncode != 0:
                failed.append(name)
            print(f'[{status}] {name}  {(time.time() - t0) / 60:.1f} min', flush=True)

    print(f'finished; {len(failed)} failed: {failed}' if failed else 'finished; all runs ok', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
