"""Run a grid of Phase 1 experiments (meant for tmux on the server).

    python run_grid.py --grid sphere always --parallel 3
    python run_grid.py --grid sphere always --datasets full --parallel 3
    python run_grid.py --grid sweep --parallel 3      # sphere size on case1, needs the case1 sphere runs
    python run_grid.py --grid sphere --dry-run

Finished runs (summary.json present) are skipped, so the command can be restarted after a
crash or a lost connection. Each run's console output goes to <out-dir>/_logs/<run>.log.

Sphere sweep: for every model/composition on case1, the sphere run's logged
center-radius ratios (one per new-entity re-embed) give k values at which the rule would
have retrained on ~10%, ~50%, ~90% of new entities. Chosen k values are saved to
<out-dir>/_sweep_plan.csv.
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


def sweep_entries(sweep_name, out_dir):
    spec = configs.DYNAMIC_SWEEPS[sweep_name]
    entries, plan = [], []
    for dataset in spec['datasets']:
        for model in spec['models']:
            for composition in configs.COMPOSITIONS:
                base = configs.run_id(dataset, model, composition, 'phase1', 'sphere', 1.0)
                shift_path = out_dir / base / 'embedding_shift_metrics.tsv'
                if not (out_dir / base / 'summary.json').exists():
                    print(f'[sweep] no finished sphere run {base}; skipped')
                    continue
                shifts = pd.read_csv(shift_path, sep='\t') if shift_path.stat().st_size > 1 else pd.DataFrame()
                ratios = np.array([])
                if not shifts.empty:
                    unseen = shifts[shifts['trigger_reason'].eq('unseen_element')]
                    ratios = unseen['center_radius_ratio'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(ratios) < MIN_RATIOS_FOR_SWEEP:
                    print(f'[sweep] {base}: only {len(ratios)} re-embed ratios; skipped')
                    continue
                for tag, share in configs.SWEEP_TARGETS.items():
                    k = float(np.quantile(ratios, 1.0 - share))
                    entries.append((dataset, model, composition, 'phase1', 'sphere', k, tag))
                    plan.append({'dataset': dataset, 'model': model, 'composition': composition, 'tag': tag,
                                 'target_retrain_share': share, 'sphere_k': k, 'n_ratios': len(ratios),
                                 'share_in_sphere_run': float((ratios > k).mean())})
    if plan:
        path = out_dir / '_sweep_plan.csv'
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        new = pd.DataFrame(plan)
        if not old.empty:
            keep = ~old.set_index(['dataset', 'model', 'composition', 'tag']).index.isin(
                new.set_index(['dataset', 'model', 'composition', 'tag']).index)
            new = pd.concat([old[keep], new], ignore_index=True)
        new.to_csv(path, index=False)
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', required=True, nargs='+', choices=list(configs.GRIDS) + list(configs.DYNAMIC_SWEEPS))
    p.add_argument('--datasets', nargs='+', default=None, help='only runs on these datasets')
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--threads', type=int, default=None, help='CPU threads per run (default: 8 // parallel)')
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=HERE / 'runs')
    p.add_argument('--adazor-src', type=Path, default=HERE.parent / 'adazor' / 'src')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('extra', nargs=argparse.REMAINDER, help='passed to drift_sphere.py after --')
    args = p.parse_args()
    extra = [a for a in args.extra if a != '--']
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for grid in args.grid:
        entries = sweep_entries(grid, args.out_dir) if grid in configs.DYNAMIC_SWEEPS else configs.GRIDS[grid]
        for entry in entries:
            if args.datasets and entry[0] not in args.datasets:
                continue
            if entry not in jobs:
                jobs.append(entry)

    todo = []
    for dataset, model, composition, preset, policy, k, tag in jobs:
        name = configs.run_id(dataset, model, composition, preset, policy, k, tag)
        if (args.out_dir / name / 'summary.json').exists():
            print(f'[skip] {name}')
            continue
        cmd = [sys.executable, str(HERE / 'drift_sphere.py'), '--dataset', dataset, '--model', model,
               '--composition', composition, '--preset', preset, '--retrain-policy', policy,
               '--sphere-k', repr(float(k)), '--run-name', name, '--data-dir', str(args.data_dir),
               '--out-dir', str(args.out_dir), '--adazor-src', str(args.adazor_src), *extra]
        todo.append((name, cmd))

    print(f'{len(todo)} run(s) to do, {args.parallel} at a time', flush=True)
    if args.dry_run:
        for name, cmd in todo:
            print(name, '\n   ', ' '.join(cmd))
        return 0

    threads = str(args.threads or max(1, 8 // args.parallel))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    log_dir = args.out_dir / '_logs'
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
