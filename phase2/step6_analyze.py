"""Step 6: collect every Phase 1 and Phase 2 output into tidy tables for the two notebooks.

Writes outputs/results/:
  Part 1 (evaluating Antoine's method)
    phase1_runs.csv              every Phase 1 run (sphere k=1, always, sweep): alarms, false alarms, delay, peak, MRR
    phase1_sphere_ratios.csv     the sphere-rule ratio at every Phase 1 re-embed (fires only if > k)
    seed_pairs.csv               comparability test: pairs of embeddings (step 4)
  Both parts (Antoine's loop with three embedding strategies, step 2)
    stream_runs.csv              one row per run: method x scenario x model x composition x repeat
    stream_alignment.csv         retrain + align: how far each new embedding was from the previous one
  Part 2 (other methods)
    channel_runs.csv             every channel on real / no-drift / injected streams (step 3)
    channel_summary.csv          per channel: false alarms with no drift, detection rate, delay, real stream
    channel_separation.csv       threshold-free: how well each channel's statistic separates drift from no drift
    channel_curves.csv           false alarms vs detection delay over all thresholds
    channels_real_series.csv.gz  statistic and threshold over the real streams
    graph_series.csv, graph_changepoints.csv   yearly graph distances (step 5)
    quality.csv                  MRR vs training epochs (step 7) + Phase 1 MRR at 10 epochs
  meta.json                      settings and the MONDO release used

  python phase2/step6_analyze.py --phase1 phase1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402

sys.path.insert(0, str(settings.REPO / 'phase1'))
import analyze as p1a  # noqa: E402  (phase1/analyze.py: alarm and peak metrics)

_SWITCH_CACHE = {}


def true_switch_index(dataset):
    """1-based position of the first switch-year triple in the real future stream (None if unknown)."""
    if dataset not in _SWITCH_CACHE:
        sy = settings.DATASETS.get(dataset, {}).get('switch_year')
        value = None
        try:
            pool = lib_data.load_pool(dataset, 'full')
            years = pool.loc[pool['part'].eq('stream'), 'event_year'].to_numpy()
            after = np.flatnonzero(years >= sy) if sy is not None else []
            value = int(after[0]) + 1 if len(after) else None
        except FileNotFoundError:
            pass
        _SWITCH_CACHE[dataset] = value
    return _SWITCH_CACHE[dataset]


def run_metrics(run_dir, scenario=None):
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    history = pd.read_csv(run_dir / 'window_history.tsv.gz', sep='\t')
    try:
        shifts = pd.read_csv(run_dir / 'embedding_shift_metrics.tsv', sep='\t')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        shifts = pd.DataFrame()
    scenario = scenario or summary.get('scenario', 'real')
    switch_index = summary.get('switch_stream_index')
    if switch_index is None and scenario == 'real' and config.get('switch_year') is not None:
        switch_index = true_switch_index(summary['dataset'])
        if switch_index is None and not history.empty:   # fallback: first tested switch-year triple
            after = history.loc[history['event_year'] >= config['switch_year'], 'stream_index']
            switch_index = int(after.iloc[0]) if len(after) else None
    row = {k: summary.get(k) for k in ('run', 'dataset', 'model', 'composition', 'method', 'repeat', 'epochs',
                                       'retrain_policy', 'sphere_k', 'future_rows', 'tested_steps',
                                       'processed_future_triples', 'num_detected_drifts', 'num_unseen_refresh_events',
                                       'num_threshold_retrains', 'unseen_retrain_ratio',
                                       'initial_reference_outlier_fraction', 'runtime_seconds')}
    row['scenario'] = scenario
    row['switch_index'] = switch_index
    row['tested_share'] = (row['tested_steps'] or 0) / max(1, row['future_rows'] or summary.get('processed_future_triples') or 1)
    row['mean_reference_outlier_rate'] = float(history['reference_outlier_fraction'].mean()) if len(history) else np.nan
    row['mean_current_outlier_rate'] = float(history['current_outlier_fraction'].mean()) if len(history) else np.nan
    row['alarms_per_1000_tested'] = 1000 * (row['num_detected_drifts'] or 0) / max(1, row['tested_steps'] or 0)
    if len(history):
        row.update(p1a.alarm_metrics(history, shifts, switch_index, int(config['window_size'])))
        row.update(p1a.peak_metrics(history, switch_index, int(config['window_size'])))
        row.update(p1a.sphere_metrics(shifts))
    return row, shifts


def phase1_tables(phase1, out):
    runs_dir = Path(phase1) / 'runs'
    if not runs_dir.exists():
        print(f'  no Phase 1 runs in {runs_dir}')
        return 0
    rows, ratios = [], []
    for d in sorted(runs_dir.iterdir()):
        if not (d / 'summary.json').exists():
            continue
        r, shifts = run_metrics(d, scenario='real')
        parts = d.name.split('__')
        r['policy_tag'] = parts[3] if len(parts) >= 5 else (r.get('retrain_policy') or d.name)
        rows.append(r)
        if not shifts.empty and 'center_radius_ratio' in shifts:
            s = shifts[shifts['trigger_reason'].eq('unseen_element')]
            ratios.append(pd.DataFrame({'run': d.name, 'dataset': r['dataset'], 'model': r['model'],
                                        'composition': r['composition'], 'policy_tag': r['policy_tag'],
                                        'sphere_k': r['sphere_k'], 'ratio': s['center_radius_ratio'].to_numpy(),
                                        'retrained': s['retrained_ocsvm'].astype(bool).to_numpy()}))
    table = pd.DataFrame(rows)
    q = p1a.load_quality(Path(phase1) / 'quality')
    if not q.empty and not table.empty:
        table['qd'] = table['dataset'].replace({'full_short': 'full'})
        table = table.merge(q, how='left', left_on=['qd', 'model'], right_on=['dataset', 'model'],
                            suffixes=('', '_q')).drop(columns=['qd', 'dataset_q'], errors='ignore')
    table.to_csv(out / 'phase1_runs.csv', index=False)
    if ratios:
        pd.concat(ratios, ignore_index=True).to_csv(out / 'phase1_sphere_ratios.csv', index=False)
    return len(table)


def stream_tables(phase1, out):
    rows, align = [], []
    for ds in settings.DATASETS:
        root = settings.OUT / 'stream' / ds
        if root.exists():
            for d in sorted(root.iterdir()):
                if (d / 'summary.json').exists():
                    r, _ = run_metrics(d)
                    rows.append(r)
                    if (d / 'alignment.tsv').exists():
                        try:
                            align.append(pd.read_csv(d / 'alignment.tsv', sep='\t').assign(run=d.name, dataset=ds))
                        except pd.errors.EmptyDataError:
                            pass
        # 'original' on the real stream = Phase 1 block A (sphere rule k = 1)
        for model in settings.MODELS:
            for comp in settings.COMPOSITIONS:
                a = Path(phase1) / 'runs' / f'{ds}__{model}__{comp}__sphere1__phase1'
                if (a / 'summary.json').exists():
                    r, _ = run_metrics(a, scenario='real')
                    r.update({'method': 'original', 'repeat': 0, 'epochs': settings.KGE['epochs']})
                    rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'stream_runs.csv', index=False)
    if align:
        pd.concat(align, ignore_index=True).to_csv(out / 'stream_alignment.csv', index=False)
    return len(rows)


def separation_and_curves(runs):
    """Threshold-free comparison from the saved maxima (no-drift) and post-switch running maxima (injected)."""
    grid = np.asarray(settings.MONTE_CARLO['delay_grid'])
    keys = ['dataset', 'channel', 'model', 'composition']
    sep, curves = [], []
    for key, g in runs.groupby(keys):
        ctrl = g.loc[g['scenario'].eq('control'), 'stat_max'].dropna().to_numpy()
        inj = g[g['scenario'].eq('inject') & g['post_runmax'].notna()]
        if len(ctrl) < 2 or inj.empty:
            continue
        R = np.array([[float(v) for v in s.split()] for s in inj['post_runmax']])
        R = np.where(np.isnan(R), -np.inf, R)
        row = dict(zip(keys, key))
        for D in (250, 500, 1000):
            j = min(int(np.searchsorted(grid, D)), len(grid) - 1)
            post = R[:, j]
            # P(max within D triples after the switch > max over a whole no-drift stream), ties count half
            gt = (post[:, None] > ctrl[None, :]).mean()
            eq = (post[:, None] == ctrl[None, :]).mean()
            row[f'separation_{D}'] = float(gt + 0.5 * eq)
        sep.append(row)
        thresholds = np.unique(np.quantile(np.concatenate([ctrl, R[np.isfinite(R)]]), np.linspace(0, 1, 60)))
        for t in thresholds:
            hit = R > t
            first = np.where(hit.any(1), grid[np.argmax(hit, axis=1)], np.inf)
            curves.append({**dict(zip(keys, key)), 'threshold': float(t),
                           'no_drift_false_alarm_share': float((ctrl > t).mean()),
                           'detected_within_500': float((first <= 500).mean()),
                           'median_delay': float(np.median(first)) if np.isfinite(np.median(first)) else np.nan})
    return pd.DataFrame(sep), pd.DataFrame(curves)


def channel_tables(out):
    frames, series, meta = [], [], {}
    for ds in settings.DATASETS:
        d = settings.OUT / 'channels' / ds
        if (d / 'runs.csv').exists():
            frames.append(pd.read_csv(d / 'runs.csv'))
        if (d / 'series_real.csv.gz').exists():
            series.append(pd.read_csv(d / 'series_real.csv.gz').assign(dataset=ds))
        if (d / 'meta.json').exists():
            meta[ds] = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    lib_data.write_json(out / 'meta.json', meta)
    if not frames:
        return 0
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(out / 'channel_runs.csv', index=False)
    if series:
        pd.concat(series, ignore_index=True).to_csv(out / 'channels_real_series.csv.gz', index=False)
    keys = ['dataset', 'channel', 'model', 'composition']
    ctrl = runs[runs['scenario'].eq('control')].groupby(keys).agg(
        control_runs=('repeat', 'count'), control_alarms_per_1000=('alarms_per_1000', 'mean'),
        control_any_alarm=('any_alarm', 'mean'), control_stream_length=('stream_length', 'mean'))
    inj = runs[runs['scenario'].eq('inject')].groupby(keys).agg(
        inject_runs=('repeat', 'count'), detection_rate=('detected', 'mean'), median_delay=('delay', 'median'),
        false_alarms_per_1000_before_switch=('false_alarms_per_1000', 'mean'))
    real = runs[runs['scenario'].eq('real')].set_index(keys)[
        [c for c in ('alarms', 'false_alarms', 'detected', 'delay', 'first_alarm_year') if c in runs]].add_prefix('real_')
    ctrl.join(inj, how='outer').join(real, how='outer').reset_index().to_csv(out / 'channel_summary.csv', index=False)
    if 'post_runmax' in runs:
        sep, curves = separation_and_curves(runs)
        sep.to_csv(out / 'channel_separation.csv', index=False)
        curves.to_csv(out / 'channel_curves.csv', index=False)
    return len(runs)


def other_tables(phase1, out):
    n = 0
    seeds = [pd.read_csv(p) for ds in settings.DATASETS
             for p in [settings.OUT / 'seed_test' / ds / 'pairs.csv'] if p.exists()]
    if seeds:
        pd.concat(seeds, ignore_index=True).to_csv(out / 'seed_pairs.csv', index=False)
        n += 1
    s, c = [], []
    for ds in settings.DATASETS:
        d = settings.OUT / 'graph_distance' / ds
        if (d / 'series.csv').exists():
            s.append(pd.read_csv(d / 'series.csv').assign(dataset=ds))
            c.append(pd.read_csv(d / 'changepoints.csv'))
    if s:
        pd.concat(s).to_csv(out / 'graph_series.csv', index=False)
        pd.concat(c).to_csv(out / 'graph_changepoints.csv', index=False)
        n += 1
    q = [pd.read_csv(p) for p in sorted((settings.OUT / 'quality').glob('*.csv'))] if (settings.OUT / 'quality').exists() else []
    p1q = p1a.load_quality(Path(phase1) / 'quality')
    if not p1q.empty:
        q.append(p1q.assign(epochs=settings.KGE['epochs'], source='phase1'))
    if q:
        pd.concat(q, ignore_index=True).to_csv(out / 'quality.csv', index=False)
        n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--phase1', type=Path, default=settings.REPO / 'phase1', help='folder with runs/ and quality/')
    args = p.parse_args(argv)
    out = settings.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    print(f'Phase 1 runs: {phase1_tables(args.phase1, out)}; stream runs: {stream_tables(args.phase1, out)}; '
          f'channel rows: {channel_tables(out)}; other tables: {other_tables(args.phase1, out)}\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
