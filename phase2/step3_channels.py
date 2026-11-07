"""Step 3 (CPU): every monitoring channel on real, no-drift and injected-switch streams.

Experiments 1, 5-9, 11 and 12. Uses the holdout embedding from step 1 (reference and stream are both
unseen by the embedding). Scenarios:
  real     the real stream in time order; reference = the 500 most recent pre-cutoff triples
  control  NO drift: the pre-switch triples shuffled; reference = first 500, stream = the rest
           (repeated MONTE_CARLO['control'] times)                                     -> Exp 1
  inject   the switch at a random position: shuffled pre-switch triples, then post-switch triples
           (repeated MONTE_CARLO['inject'] times)                                      -> Exp 5-9, 11, 12

Writes outputs/channels/<dataset>/:
  runs/<scenario>_<repeat>.csv   one row per channel x model x composition (resumable)
  runs.csv                       all rows together
  series_real.csv.gz             the statistic and threshold at every tested position (real order)
  meta.json                      settings, MONDO release

  python phase2/step3_channels.py --dataset case1 --workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402
import lib_detect  # noqa: E402

SCENARIO_ID = {'real': 0, 'control': 1, 'inject': 2}
_DATA = {}


def load_inputs(dataset, models):
    pool = lib_data.load_pool(dataset, 'holdout')
    pool_data = {'type_codes': pd.factorize(pool['type'].astype(str), sort=True)[0],
                 'new_entity': pool['new_entity'].to_numpy().astype(np.int8), 'categories': None}
    meta = {'mondo_version': None}
    cpath = lib_data.categories_path(dataset)
    if cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        if np.array_equal(z['row_number'], pool['row_number'].to_numpy()):
            pool_data['categories'] = z['matrix']
            meta.update({'mondo_version': str(z['version']), 'mondo_categories': z['labels'].tolist()})
    model_data = {}
    for m in models:
        path = lib_data.kge_features_path(dataset, m, 'holdout')
        if not path.exists():
            print(f'  [missing] {path.name}: run step1_prepare.py first; {m} skipped')
            continue
        z = np.load(path, allow_pickle=False)
        if not np.array_equal(z['row_number'], pool['row_number'].to_numpy()):
            raise RuntimeError(f'{path.name} does not match the pool (rerun step 1 with --overwrite)')
        model_data[m] = {k: z[k] for k in ('hadamard', 'concatenation', 'score', 'folded')}
        meta.setdefault('embedder', str(z['embedder']))
    return pool, pool_data, model_data, meta


def _init_worker(dataset, models):
    _DATA['pool'], _DATA['pool_data'], _DATA['model_data'], _ = load_inputs(dataset, models)


def run_one(job):
    """One scenario repeat: split, fit every channel, compute statistics, thresholds, alarms."""
    dataset, scenario, rep, channels, compositions, out_file = job
    pool, pool_data, model_data = _DATA['pool'], _DATA['pool_data'], _DATA['model_data']
    rng = np.random.default_rng([settings.MONTE_CARLO['seed'], SCENARIO_ID[scenario], rep])
    split = {'real': lambda: lib_data.real_order_split(pool, 'holdout', rng),
             'control': lambda: lib_data.control_split(pool, rng),
             'inject': lambda: lib_data.inject_split(pool, rng)}[scenario]
    ref_fit, ref_cal, stream, switch = split()
    instances = lib_detect.build_channels(pool_data, model_data, ref_fit, ref_cal, list(model_data),
                                          compositions, channels)
    rows, series = [], []
    for ch in instances:
        t0 = time.time()
        pos, stat = ch.stat(ch.values[stream], ch.stride)
        thr = ch.fixed_threshold if ch.fixed_threshold is not None else lib_detect.calibrate(ch, rng)
        tested, alarms = lib_detect.run_alarms(pos, stat, thr)
        name, model, comp = (ch.name.split('|') + ['-', '-'])[:3]
        row = {'dataset': dataset, 'scenario': scenario, 'repeat': rep, 'channel': name, 'model': model,
               'composition': comp, 'threshold': thr,
               'threshold_kind': 'fixed' if ch.fixed_threshold is not None else 'calibrated',
               'stat_max': float(stat.max()) if len(stat) else np.nan,
               'alarm_positions': ' '.join(map(str, alarms)), 'seconds': time.time() - t0, **ch.info}
        row.update(lib_detect.score_run(alarms, tested, len(stream), switch))
        if scenario == 'real':
            row['first_alarm_year'] = float(pool['event_year'].iloc[stream[alarms[0]]]) if alarms else np.nan
            series.append(pd.DataFrame({'channel': name, 'model': model, 'composition': comp, 'position': pos,
                                        'row_number': pool['row_number'].to_numpy()[stream[pos]],
                                        'event_year': pool['event_year'].to_numpy()[stream[pos]],
                                        'stat': stat, 'threshold': thr,
                                        'alarm': np.isin(pos, alarms)}))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_file, index=False)
    if series:
        pd.concat(series, ignore_index=True).to_csv(out_file.parent.parent / 'series_real.csv.gz', index=False)
    return out_file.name, len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(settings.DATASETS))
    p.add_argument('--models', nargs='+', default=settings.MODELS)
    p.add_argument('--compositions', nargs='+', default=settings.COMPOSITIONS)
    p.add_argument('--channels', nargs='+', default=list(settings.CHANNELS))
    p.add_argument('--scenarios', nargs='+', default=['real', 'control', 'inject'], choices=list(SCENARIO_ID))
    p.add_argument('--control', type=int, default=settings.MONTE_CARLO['control'], help='number of control streams')
    p.add_argument('--inject', type=int, default=settings.MONTE_CARLO['inject'], help='number of injected streams')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(argv)

    out_dir = settings.OUT / 'channels' / args.dataset
    (out_dir / 'runs').mkdir(parents=True, exist_ok=True)
    pool, _, model_data, meta = load_inputs(args.dataset, args.models)
    has_switch = settings.DATASETS[args.dataset]['switch_year'] is not None
    jobs = []
    for sc in args.scenarios:
        if sc != 'real' and not has_switch:
            print(f'  {sc}: skipped ({args.dataset} has no known switch)')
            continue
        n = {'real': 1, 'control': args.control, 'inject': args.inject}[sc]
        for rep in range(n):
            f = out_dir / 'runs' / f'{sc}_{rep:03d}.csv'
            if not f.exists():
                jobs.append((args.dataset, sc, rep, args.channels, args.compositions, f))
    print(f'[{args.dataset}] {len(jobs)} scenario runs to do; models: {list(model_data)}; '
          f'pool {len(pool):,} triples; MONDO: {meta.get("mondo_version")}', flush=True)
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args.dataset, list(model_data))) as ex:
            for k, (name, n_rows) in enumerate(ex.map(run_one, jobs), 1):
                if k % 10 == 0 or k == len(jobs):
                    print(f'  {k}/{len(jobs)} done ({(time.time() - t0) / 60:.1f} min)', flush=True)
    files = sorted((out_dir / 'runs').glob('*.csv'))
    if files:
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(out_dir / 'runs.csv', index=False)
    lib_data.write_json(out_dir / 'meta.json', {
        **meta, 'dataset': args.dataset, 'window': settings.WINDOW, 'reference': settings.REFERENCE,
        'calibration': settings.CALIBRATION, 'monte_carlo': settings.MONTE_CARLO, 'channels': settings.CHANNELS,
        'ocsvm': settings.OCSVM, 'alpha': settings.ALPHA, 'kge': settings.KGE, 'foldin': settings.FOLDIN})
    print(f'[{args.dataset}] finished: {out_dir / "runs.csv"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
