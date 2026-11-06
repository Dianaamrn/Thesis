"""Phase 1 analysis: one table row per run, peak diagnosis on the known-drift cases, plots.

    python analyze.py --runs runs --out analysis

Writes analysis/summary_table.csv (+ .md), analysis/figures/*.png.

Definitions (stream_index = position in the target-type stream after the cutoff; refill rows
after an alarm count too, so distances are in stream triples):
  switch_index        first tested triple with event_year >= the case's switch year
  alarms_pre/post     alarms before / from the switch; false_alarms_per_1000 uses pre-switch tested steps
  detection_delay     first alarm at or after the switch, minus switch_index
  peak_*              the outlier-rate curve (current window outlier fraction per tested step):
                      peak = its global maximum; switch_to_peak = peak index - switch_index;
                      peaks are found with scipy find_peaks (min distance = one window);
                      peak_prominence = scipy prominence of the highest peak;
                      peak_vs_second = highest peak height / second-highest peak height;
                      post_switch_peak = maximum in [switch, switch + 2 windows), and its
                      distance from the switch
  density_pre/post    alarms per 100 tested triples in the 1,000 stream triples before / after the
                      switch (provisional; Antoine's definition to be confirmed)
  alarms_after_reembed share of alarms with an unseen-entity re-embed that did NOT retrain the OCSVM
                      within the preceding window (250 stream triples)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover
    find_peaks = None


def load_run(run_dir):
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    history = pd.read_csv(run_dir / 'window_history.tsv.gz', sep='\t')
    shifts_path = run_dir / 'embedding_shift_metrics.tsv'
    shifts = pd.read_csv(shifts_path, sep='\t') if shifts_path.stat().st_size > 1 else pd.DataFrame()
    return summary, config, history, shifts


def peak_metrics(history, switch_index, window):
    out = {}
    if history.empty:
        return out
    idx = history['stream_index'].to_numpy()
    curve = history['current_outlier_fraction'].to_numpy()
    top = int(np.argmax(curve))
    out['peak_value'] = float(curve[top])
    out['peak_stream_index'] = int(idx[top])
    if switch_index is not None:
        out['switch_to_peak'] = int(idx[top] - switch_index)
    if find_peaks is not None and len(curve) > 2:
        peaks, props = find_peaks(curve, distance=max(1, window), prominence=0)
        if len(peaks):
            order = np.argsort(curve[peaks])[::-1]
            p1 = peaks[order[0]]
            out['peak_prominence'] = float(props['prominences'][order[0]])
            if len(order) > 1:
                p2 = peaks[order[1]]
                out['second_peak_value'] = float(curve[p2])
                out['peak_vs_second'] = float(curve[p1] / curve[p2]) if curve[p2] > 0 else np.inf
    if switch_index is not None:
        post = (idx >= switch_index) & (idx < switch_index + 2 * window)
        if post.any():
            j = np.flatnonzero(post)[np.argmax(curve[post])]
            out['post_switch_peak_value'] = float(curve[j])
            out['switch_to_post_switch_peak'] = int(idx[j] - switch_index)
        pre = idx < switch_index
        if pre.any():
            out['pre_switch_max_outlier_rate'] = float(curve[pre].max())
            out['pre_switch_mean_outlier_rate'] = float(curve[pre].mean())
        if (idx >= switch_index).any():
            out['post_switch_mean_outlier_rate'] = float(curve[idx >= switch_index].mean())
    return out


def alarm_metrics(history, shifts, switch_index, window, local=1000):
    out = {}
    idx = history['stream_index'].to_numpy()
    alarms = history.loc[history['drift_detected'].astype(bool), 'stream_index'].to_numpy()
    out['alarms_total'] = int(len(alarms))
    if switch_index is None:
        out['alarms_per_1000_tested'] = 1000 * len(alarms) / max(1, len(idx))
    else:
        pre_tested = int((idx < switch_index).sum())
        post_tested = int((idx >= switch_index).sum())
        a_pre = int((alarms < switch_index).sum())
        a_post = int((alarms >= switch_index).sum())
        out.update({'tested_pre': pre_tested, 'tested_post': post_tested, 'alarms_pre': a_pre, 'alarms_post': a_post,
                    'false_alarms_per_1000': 1000 * a_pre / pre_tested if pre_tested else np.nan})
        after = alarms[alarms >= switch_index]
        before = alarms[alarms < switch_index]
        out['detection_delay'] = int(after[0] - switch_index) if len(after) else np.nan
        out['last_false_alarm_before_switch'] = int(switch_index - before[-1]) if len(before) else np.nan
        out['median_false_alarm_distance'] = float(np.median(switch_index - before)) if len(before) else np.nan
        before_local = (idx >= switch_index - local) & (idx < switch_index)
        after_local = (idx >= switch_index) & (idx < switch_index + local)
        out['density_pre'] = 100 * ((alarms >= switch_index - local) & (alarms < switch_index)).sum() / max(1, before_local.sum())
        out['density_post'] = 100 * ((alarms >= switch_index) & (alarms < switch_index + local)).sum() / max(1, after_local.sum())
    if len(alarms) and not shifts.empty:
        unseen = shifts[shifts['trigger_reason'].eq('unseen_element') & ~shifts['retrained_ocsvm'].astype(bool)]
        ev = unseen['stream_index'].to_numpy()
        near = [bool(((ev <= a) & (ev > a - window)).any()) for a in alarms]
        out['alarms_after_reembed_share'] = float(np.mean(near))
    return out


def sphere_metrics(shifts):
    out = {}
    if shifts.empty:
        return out
    unseen = shifts[shifts['trigger_reason'].eq('unseen_element')]
    r = unseen['center_radius_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(r):
        out.update({'sphere_ratio_median': float(r.median()), 'sphere_ratio_p95': float(r.quantile(0.95)),
                    'sphere_ratio_max': float(r.max())})
    if 'train_seconds' in unseen and len(unseen):
        out['mean_reembed_seconds'] = float(unseen['train_seconds'].mean())
    return out


def plot_run(name, history, shifts, switch_index, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history['stream_index'], history['current_outlier_fraction'], lw=1, color='#3b6ea5', label='current outlier fraction')
    ax.plot(history['stream_index'], history['reference_outlier_fraction'], lw=1, color='#999999', label='reference outlier fraction')
    alarms = history[history['drift_detected'].astype(bool)]
    ax.scatter(alarms['stream_index'], alarms['current_outlier_fraction'], color='#c0392b', s=14, zorder=3, label='alarm')
    if not shifts.empty:
        unseen = shifts[shifts['trigger_reason'].eq('unseen_element')]
        ymin = -0.03
        ax.scatter(unseen['stream_index'], np.full(len(unseen), ymin), marker='|', s=40,
                   c=np.where(unseen['retrained_ocsvm'].astype(bool), '#2e7d32', '#bbbbbb'), label='re-embed (green = retrained)')
    if switch_index is not None:
        ax.axvline(switch_index, color='black', ls='--', lw=1, label='switch')
    ax.set_xlabel('stream triple')
    ax.set_ylabel('outlier fraction')
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_alarm_density(offsets_df, fig_dir, bin_width=250):
    """Where alarms fall relative to the switch (negative = before = false alarm), per dataset.
    One panel per composition, one line per model; bins of one window (250 stream triples)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (dataset, policy_tag), d in offsets_df.groupby(['dataset', 'policy_tag']):
        comps = sorted(d['composition'].unique())
        fig, axes = plt.subplots(1, len(comps), figsize=(6 * len(comps), 3.8), sharey=True, squeeze=False)
        lo = int(np.floor(d['offset_from_switch'].min() / bin_width) * bin_width)
        hi = int(np.ceil((d['offset_from_switch'].max() + 1) / bin_width) * bin_width)
        bins = np.arange(lo, hi + bin_width, bin_width)
        for ax, comp in zip(axes[0], comps):
            for model, m in d[d['composition'].eq(comp)].groupby('model'):
                counts, edges = np.histogram(m['offset_from_switch'], bins=bins)
                ax.step(edges[:-1], counts, where='post', label=model, lw=1.3)
            ax.axvline(0, color='black', ls='--', lw=1)
            ax.set_title(f'{dataset} - {comp} - {policy_tag}', fontsize=9)
            ax.set_xlabel('stream triples from switch (negative = before)')
        axes[0][0].set_ylabel(f'alarms per {bin_width} triples')
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f'alarm_density__{dataset}__{policy_tag}.png', dpi=140)
        plt.close(fig)


def load_quality(quality_dir):
    if quality_dir is None or not Path(quality_dir).exists():
        return pd.DataFrame()
    rows = [json.loads(f.read_text()) for f in Path(quality_dir).glob('*__phase1.json')]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[['dataset', 'model', 'mrr', 'hits_at_10']]


def plot_policy_tradeoff(table, fig_dir):
    """Retrain share (cost) vs alarm quality, per dataset and composition, one line per model,
    points = policies (sphere sweep, Antoine k=1, always)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset, d in table.groupby('dataset'):
        if d['policy_tag'].nunique() < 2:
            continue
        has_switch = d['switch_index'].notna().any()
        metrics = (['false_alarms_per_1000', 'detection_delay'] if has_switch else ['alarms_per_1000_tested'])
        metrics = [m for m in metrics if m in d]
        comps = sorted(d['composition'].unique())
        fig, axes = plt.subplots(len(metrics), len(comps), figsize=(5.5 * len(comps), 3.6 * len(metrics)), squeeze=False)
        for j, comp in enumerate(comps):
            for i, metric in enumerate(metrics):
                ax = axes[i][j]
                for model, m in d[d['composition'].eq(comp)].groupby('model'):
                    m = m.sort_values('unseen_retrain_ratio')
                    ax.plot(m['unseen_retrain_ratio'], m[metric], marker='o', ms=4, lw=1, label=model)
                ax.set_title(f'{dataset} - {comp}', fontsize=9)
                ax.set_xlabel('share of new-entity re-embeds that retrained the OCSVM')
                ax.set_ylabel(metric)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f'policy_tradeoff__{dataset}.png', dpi=140)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, default=Path('runs'))
    p.add_argument('--out', type=Path, default=Path('analysis'))
    p.add_argument('--quality', type=Path, default=Path('quality'), help='folder with embedding_quality.py outputs')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    (args.out / 'figures').mkdir(parents=True, exist_ok=True)

    rows, offsets = [], []
    for run_dir in sorted(d for d in args.runs.iterdir() if (d / 'summary.json').exists()):
        summary, config, history, shifts = load_run(run_dir)
        window = int(config['window_size'])
        switch_year = config.get('switch_year')
        switch_index = None
        if switch_year is not None and not history.empty:
            after = history.loc[history['event_year'] >= switch_year, 'stream_index']
            switch_index = int(after.iloc[0]) if len(after) else None
        row = {k: summary.get(k) for k in ('run', 'dataset', 'model', 'composition', 'preset', 'retrain_policy',
                                           'sphere_k', 'feature_dim', 'future_rows', 'processed_future_triples',
                                           'tested_steps', 'num_detected_drifts', 'num_unseen_refresh_events',
                                           'num_drift_refresh_events', 'num_threshold_retrains', 'unseen_retrain_ratio',
                                           'initial_reference_outlier_fraction', 'runtime_seconds')}
        row['policy_tag'] = summary['run'].split('__')[3]
        row['switch_index'] = switch_index
        row['mean_reference_outlier_rate'] = float(history['reference_outlier_fraction'].mean()) if len(history) else np.nan
        row['mean_current_outlier_rate'] = float(history['current_outlier_fraction'].mean()) if len(history) else np.nan
        row.update(alarm_metrics(history, shifts, switch_index, window))
        row.update(peak_metrics(history, switch_index, window))
        row.update(sphere_metrics(shifts))
        rows.append(row)
        if switch_index is not None:
            for a in history.loc[history['drift_detected'].astype(bool), 'stream_index']:
                offsets.append({'run': summary['run'], 'dataset': summary['dataset'], 'model': summary['model'],
                                'composition': summary['composition'], 'policy_tag': row['policy_tag'],
                                'alarm_stream_index': int(a),
                                'offset_from_switch': int(a - switch_index)})
        if not args.no_plots:
            plot_run(summary['run'], history, shifts, switch_index, args.out / 'figures' / f'{summary["run"]}.png')

    table = pd.DataFrame(rows)
    if table.empty:
        print('no finished runs found in', args.runs)
        return
    table['runtime_min'] = (table.pop('runtime_seconds') / 60).round(1)
    quality = load_quality(args.quality)
    if not quality.empty:
        table['quality_dataset'] = table['dataset'].replace({'full_short': 'full'})
        table = table.merge(quality, how='left', left_on=['quality_dataset', 'model'], right_on=['dataset', 'model'],
                            suffixes=('', '_q')).drop(columns=['quality_dataset', 'dataset_q'])
    table = table.sort_values(['dataset', 'model', 'composition', 'policy_tag']).reset_index(drop=True)
    table.to_csv(args.out / 'summary_table.csv', index=False)
    offsets_df = pd.DataFrame(offsets)
    if not offsets_df.empty:
        offsets_df.to_csv(args.out / 'alarm_offsets.csv', index=False)
        if not args.no_plots:
            plot_alarm_density(offsets_df, args.out / 'figures')
    if not args.no_plots:
        plot_policy_tradeoff(table, args.out / 'figures')
    key_cols = [c for c in ['dataset', 'model', 'composition', 'policy_tag', 'sphere_k', 'mrr', 'num_detected_drifts',
                            'alarms_pre', 'alarms_post', 'false_alarms_per_1000', 'last_false_alarm_before_switch',
                            'detection_delay', 'switch_to_peak',
                            'peak_vs_second', 'density_pre', 'density_post', 'num_unseen_refresh_events',
                            'num_threshold_retrains', 'unseen_retrain_ratio', 'alarms_after_reembed_share',
                            'mean_reference_outlier_rate', 'mean_current_outlier_rate', 'runtime_min'] if c in table]
    try:
        md = table[key_cols].round(3).to_markdown(index=False)
    except ImportError:  # tabulate not installed
        md = table[key_cols].round(3).to_string(index=False)
    (args.out / 'summary_table.md').write_text(md, encoding='utf-8')
    print(table[key_cols].round(3).to_string(index=False))
    print('\nwrote', args.out / 'summary_table.csv')


if __name__ == '__main__':
    main()
