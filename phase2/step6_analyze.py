"""Step 6: collect every Phase 2 output into tidy tables for the results notebook.

Reads outputs/{frozen,channels,seed_test,graph_distance}/ and the Phase 1 block A runs, and writes
outputs/results/:
  exp2_a0_vs_a1_real.csv       A0 (Phase 1) vs A1 (frozen) on the real stream, per model x composition
  exp1_a0_a1_control.csv       A0 vs A1 on the no-drift control streams (one row per run)
  exp1_channels_control.csv    every channel on the no-drift control streams (one row per run)
  channels_inject.csv          every channel on the injected-switch streams (one row per run)
  channels_real.csv            every channel on the real stream
  channels_real_series.csv.gz  statistic and threshold over the real stream
  channel_summary.csv          one row per dataset x channel x model x composition: false alarms
                               with no drift, detection rate, median delay
  exp3_seed_pairs.csv          seed / alignment test, one row per pair of embeddings
  exp10_graph_series.csv, exp10_graph_changepoints.csv
  meta.json                    settings and the MONDO release used

  python phase2/step6_analyze.py --phase1-runs phase1/runs
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


def run_metrics(run_dir):
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    history = pd.read_csv(run_dir / 'window_history.tsv.gz', sep='\t')
    shifts_path = run_dir / 'embedding_shift_metrics.tsv'
    try:
        shifts = pd.read_csv(shifts_path, sep='\t')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        shifts = pd.DataFrame()
    switch_year = config.get('switch_year')
    switch_index = None
    if switch_year is not None and not history.empty:
        after = history.loc[history['event_year'] >= switch_year, 'stream_index']
        switch_index = int(after.iloc[0]) if len(after) else None
    row = {k: summary.get(k) for k in ('run', 'dataset', 'model', 'composition', 'method', 'scenario', 'repeat',
                                       'tested_steps', 'processed_future_triples', 'num_detected_drifts',
                                       'num_unseen_refresh_events', 'initial_reference_outlier_fraction',
                                       'runtime_seconds')}
    row['switch_index'] = switch_index
    row['mean_reference_outlier_rate'] = float(history['reference_outlier_fraction'].mean()) if len(history) else np.nan
    row['mean_current_outlier_rate'] = float(history['current_outlier_fraction'].mean()) if len(history) else np.nan
    row['alarms_per_1000_tested'] = 1000 * row['num_detected_drifts'] / max(1, row['tested_steps'] or 0)
    if len(history):
        row.update(p1a.alarm_metrics(history, shifts, switch_index, int(config['window_size'])))
        row.update(p1a.peak_metrics(history, switch_index, int(config['window_size'])))
    return row


def exp2_tables(phase1_runs, out):
    real, control = [], []
    for ds in settings.DATASETS:
        root = settings.OUT / 'frozen' / ds
        for model in settings.MODELS:
            for comp in settings.COMPOSITIONS:
                a0 = Path(phase1_runs) / f'{ds}__{model}__{comp}__sphere1__phase1'
                if (a0 / 'summary.json').exists():
                    real.append({**run_metrics(a0), 'method': 'A0', 'scenario': 'real'})
        if root.exists():
            for d in sorted(root.iterdir()):
                if (d / 'summary.json').exists():
                    r = run_metrics(d)
                    (real if r['scenario'] == 'real' else control).append(r)
    if real:
        pd.DataFrame(real).to_csv(out / 'exp2_a0_vs_a1_real.csv', index=False)
    if control:
        pd.DataFrame(control).to_csv(out / 'exp1_a0_a1_control.csv', index=False)
    return len(real), len(control)


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
    if not frames:
        return 0
    runs = pd.concat(frames, ignore_index=True)
    for sc, name in (('control', 'exp1_channels_control.csv'), ('inject', 'channels_inject.csv'), ('real', 'channels_real.csv')):
        runs[runs['scenario'].eq(sc)].to_csv(out / name, index=False)
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
    summary = ctrl.join(inj, how='outer').join(real, how='outer').reset_index()
    summary.to_csv(out / 'channel_summary.csv', index=False)
    lib_data.write_json(out / 'meta.json', meta)
    return len(runs)


def other_tables(out):
    n = 0
    for ds in settings.DATASETS:
        p = settings.OUT / 'seed_test' / ds / 'pairs.csv'
        if p.exists():
            pd.read_csv(p).to_csv(out / 'exp3_seed_pairs.csv', index=False)
            n += 1
    s, c = [], []
    for ds in settings.DATASETS:
        d = settings.OUT / 'graph_distance' / ds
        if (d / 'series.csv').exists():
            s.append(pd.read_csv(d / 'series.csv').assign(dataset=ds))
            c.append(pd.read_csv(d / 'changepoints.csv'))
    if s:
        pd.concat(s).to_csv(out / 'exp10_graph_series.csv', index=False)
        pd.concat(c).to_csv(out / 'exp10_graph_changepoints.csv', index=False)
    return n + len(s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--phase1-runs', type=Path, default=settings.REPO / 'phase1' / 'runs')
    args = p.parse_args(argv)
    out = settings.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    n_real, n_ctrl = exp2_tables(args.phase1_runs, out)
    n_ch = channel_tables(out)
    n_other = other_tables(out)
    print(f'A0/A1 real runs: {n_real}; A0/A1 control runs: {n_ctrl}; channel rows: {n_ch}; '
          f'seed/graph tables: {n_other}\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
