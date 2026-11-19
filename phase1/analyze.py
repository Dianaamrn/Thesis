"""Analysis: one table row per run, peak analysis, plots, comparison of the versions.

    python analyze.py                                  # every version folder that exists
    python analyze.py --runs runs runs_v3 --out analysis

Writes analysis/summary_table.csv (+ .md), analysis/version_comparison.csv (+ .md),
analysis/peaks_all.csv and analysis/peaks/<run>.csv (one row per peak), analysis/figures/*.png.

Peak table (one row per peak of the outlier-rate curve; peaks = scipy find_peaks with a minimum
distance of one window, plus the curve's maximum if it lies at an end):
  rank                  1 = highest peak
  date, row_number      the stream triple at the peak: its date and its row in the data file
  subject, object       the triple; disease_name / disease_group (cancer, infectious, other) from MONDO
  triples_same_date     stream triples of the monitored type with the same date
  alarm_here            the detector raised an alarm at this triple
  triples_from_switch, days_from_switch        distance to the switch (cases only; negative = before)
  triples_from_highest, days_from_highest      distance to the highest peak
The curve only has points at tested triples (the 225 refill triples after an alarm are not tested).

Refill triples (shadow logging in drift_sphere.py, runs from 2026-11-16 on): the plot shows, in light
blue, what the OCSVM in use at the alarm says about them; the table has
  refill_scorable_share   share of refill triples whose entities all had a vector
  refill_outlier_share    share of the scorable refill triples that this OCSVM calls outliers

Definitions (stream_index = position in the target-type stream after the cutoff; refill rows
after an alarm count too, so distances are in stream triples):
  switch_index        the true change: the first stream triple with event_year >= the case's switch year,
                      read from the data file, whether or not the run tested it (decided 2026-11-19)
  switch_index_first_tested   the first TESTED triple from the switch year (the definition used before)
  switch_tested       whether the run tested the true switch triple (False = it fell inside a refill)
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover
    find_peaks = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402

VERSION_FOLDERS = list(configs.VERSIONS.values())


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


def to_date(timestamp):
    """Stream timestamps are seconds since 1970; returns a pandas Timestamp (NaT if missing)."""
    return pd.to_datetime(pd.to_numeric(timestamp, errors='coerce'), unit='s', errors='coerce')


class Mondo:
    """Disease names and the two groups of the cases (cancer, infectious) from data/mondo.obo."""

    GROUPS = {'cancer or benign tumor': 'cancer', 'infectious disease': 'infectious'}

    def __init__(self, path):
        self.parents, self.names, self._memo = {}, {}, {}
        term = None
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line == '[Term]':
                    term = ''
                elif line.startswith('['):
                    term = None
                elif term == '' and line.startswith('id: '):
                    term = line[4:].strip()
                    self.parents.setdefault(term, [])
                elif term:
                    if line.startswith('name: '):
                        self.names[term] = line[6:].strip()
                    elif line.startswith('is_a: '):
                        self.parents[term].append(line[6:].split('!')[0].strip().split(' ')[0])
        self.group_terms = {t: self.GROUPS[n] for t, n in self.names.items() if n in self.GROUPS}

    def ancestors(self, term):
        if term not in self._memo:
            seen, stack = set(), [term]
            while stack:
                x = stack.pop()
                if x not in seen:
                    seen.add(x)
                    stack.extend(self.parents.get(x, []))
            self._memo[term] = seen
        return self._memo[term]

    def describe(self, subject, obj):
        """(disease id, name, group) of a triple; the disease is the MONDO term among object / subject."""
        for x in (obj, subject):
            x = str(x)
            if 'MONDO_' in x:
                term = 'MONDO:' + x.rsplit('MONDO_', 1)[1]
                groups = sorted({g for t, g in self.group_terms.items() if t in self.ancestors(term)})
                return term, self.names.get(term, ''), '+'.join(groups) if groups else 'other'
        return '', '', ''


_SAME_DATE_CACHE = {}
_STREAM_CACHE = {}


def monitored_stream(data_dir, config):
    """The run's monitored stream after the cutoff, in the order the run reads it (stream_index = row + 1),
    with subject, object, timestamp and event_year. Read with the run code's own loader. None if no data."""
    key = (config.get('data_file'), str(config.get('target_type')), config.get('cutoff_year'))
    if key not in _STREAM_CACHE:
        path = Path(data_dir) / str(config.get('data_file')) if data_dir is not None else None
        stream = None
        if path is not None and path.exists():
            import drift_sphere as ds
            triples = ds.load_temporal_triples(path)
            _, _, future, _ = ds.build_initial_knowledge_graph(triples, cutoff_year=config['cutoff_year'],
                                                               target_type=config['target_type'])
            stream = future[['subject', 'object', 'timestamp', 'event_year', 'row_number']].copy()
            stream.insert(0, 'stream_index', np.arange(1, len(stream) + 1))
        _STREAM_CACHE[key] = stream
    return _STREAM_CACHE[key]


def true_switch_index(data_dir, config):
    """First stream triple with event_year >= switch_year (None if no switch, no data, or no such triple)."""
    switch_year = config.get('switch_year')
    if switch_year is None:
        return None
    stream = monitored_stream(data_dir, config)
    if stream is None:
        return None
    after = stream.loc[stream['event_year'] >= switch_year, 'stream_index']
    return int(after.iloc[0]) if len(after) else None


def same_date_counts(data_dir, config):
    """{timestamp (float): number of stream triples of the monitored type with that timestamp}."""
    key = (config.get('data_file'), str(config.get('target_type')))
    if key not in _SAME_DATE_CACHE:
        counts = {}
        path = Path(data_dir) / str(config.get('data_file'))
        if data_dir is not None and path.exists():
            df = pd.read_csv(path, sep='\t', usecols=['timestamp', 'type'], low_memory=False)
            target = config['target_type']
            mask = df['type'].isin(target) if isinstance(target, list) else df['type'].eq(target)
            counts = pd.to_numeric(df.loc[mask, 'timestamp'], errors='coerce').dropna().value_counts().to_dict()
        _SAME_DATE_CACHE[key] = counts
    return _SAME_DATE_CACHE[key]


def peak_table(trace, switch_index, window, mondo=None, date_counts=None, switch_timestamp=None):
    """One row per peak of the outlier-rate curve, highest first. trace = tested steps of step_trace."""
    if trace.empty or find_peaks is None:
        return pd.DataFrame()
    trace = trace.reset_index(drop=True)
    curve = trace['current_outlier_fraction'].to_numpy()
    peaks, props = find_peaks(curve, distance=max(1, window), prominence=0) if len(curve) > 2 else (np.array([], int), {'prominences': np.array([])})
    prominence = dict(zip(peaks.tolist(), props['prominences'].tolist()))
    top = int(np.argmax(curve))
    positions = sorted(set(peaks.tolist()) | {top}, key=lambda j: (-curve[j], j))
    rows = trace.iloc[positions].copy()
    out = pd.DataFrame({
        'rank': np.arange(1, len(positions) + 1),
        'outlier_rate': curve[positions],
        'prominence': [prominence.get(j, np.nan) for j in positions],
        'stream_index': rows['stream_index'].to_numpy(),
        'row_number': rows['row_number'].to_numpy(),
        'date': to_date(rows['timestamp']).dt.strftime('%Y-%m-%d').to_numpy(),
        'event_year': rows['event_year'].to_numpy(),
        'subject': rows['subject'].to_numpy(),
        'object': rows['object'].to_numpy(),
        'alarm_here': rows['drift_detected'].astype(bool).to_numpy(),
    })
    if mondo is not None:
        described = [mondo.describe(s, o) for s, o in zip(out['subject'], out['object'])]
        out['disease_id'] = [d[0] for d in described]
        out['disease_name'] = [d[1] for d in described]
        out['disease_group'] = [d[2] for d in described]
    if date_counts:
        ts = pd.to_numeric(rows['timestamp'], errors='coerce').to_numpy()
        out['triples_same_date'] = [date_counts.get(t, np.nan) for t in ts]
    dates = to_date(rows['timestamp']).to_numpy()
    if switch_index is not None:
        at_switch = trace[trace['stream_index'].eq(switch_index)]
        switch_date = to_date(at_switch['timestamp']).iloc[0] if len(at_switch) else pd.NaT
        if pd.isna(switch_date) and switch_timestamp is not None:   # the switch triple was not tested
            switch_date = to_date(pd.Series([switch_timestamp])).iloc[0]
        out['triples_from_switch'] = out['stream_index'] - switch_index
        out['days_from_switch'] = [(d - switch_date).days if pd.notna(d) and pd.notna(switch_date) else np.nan
                                   for d in pd.to_datetime(dates)]
    out['triples_from_highest'] = out['stream_index'] - out['stream_index'].iloc[0]
    first_date = pd.to_datetime(dates[0])
    out['days_from_highest'] = [(d - first_date).days if pd.notna(d) and pd.notna(first_date) else np.nan
                                for d in pd.to_datetime(dates)]
    return out


def peak_summary(peaks):
    """Columns for the run table: the highest and second-highest peak and the distance between them."""
    out = {'n_peaks': int(len(peaks))}
    if peaks.empty:
        return out
    out['highest_peak_date'] = peaks['date'].iloc[0]
    if len(peaks) > 1:
        second = peaks.iloc[1]
        out.update({'second_peak_stream_index': int(second['stream_index']), 'second_peak_date': second['date'],
                    'highest_to_second_triples': int(second['triples_from_highest']),
                    'highest_to_second_days': second['days_from_highest']})
        if 'triples_from_switch' in peaks:
            out.update({'switch_to_second_peak': int(second['triples_from_switch']),
                        'switch_to_second_peak_days': second['days_from_switch']})
    if 'days_from_switch' in peaks:
        out['switch_to_peak_days'] = peaks['days_from_switch'].iloc[0]
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


def plot_run(name, history, shifts, switch_index, out_path, peaks=None, label_top=10, refill=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    if refill is not None and not refill.empty:
        # Shadow logging: the untested refill triples, seen by the OCSVM that was in use at the alarm.
        ax.scatter(refill['stream_index'], refill['shadow_outlier_fraction'], s=2, color='#9ecae1',
                   label='not tested (refill), old OCSVM')
    # The lines are broken where triples were not tested, so a gap is not drawn as a straight line.
    x = history['stream_index'].to_numpy(dtype=float)
    gap = np.flatnonzero(np.diff(x) > 1) + 1
    x = np.insert(x, gap, np.nan)
    ax.plot(x, np.insert(history['current_outlier_fraction'].to_numpy(dtype=float), gap, np.nan),
            lw=1, color='#3b6ea5', label='current outlier fraction')
    ax.plot(x, np.insert(history['reference_outlier_fraction'].to_numpy(dtype=float), gap, np.nan),
            lw=1, color='#999999', label='reference outlier fraction')
    alarms = history[history['drift_detected'].astype(bool)]
    ax.scatter(alarms['stream_index'], alarms['current_outlier_fraction'], color='#c0392b', s=14, zorder=3, label='alarm')
    if peaks is not None and not peaks.empty:
        ax.scatter(peaks['stream_index'], peaks['outlier_rate'], marker='v', s=34, facecolors='none',
                   edgecolors='#e67e22', linewidths=1.2, zorder=4, label='peak (number = rank by height)')
        for _, pk in peaks.head(label_top).iterrows():
            ax.annotate(f"{int(pk['rank'])}: {pk['date']}", (pk['stream_index'], pk['outlier_rate']),
                        textcoords='offset points', xytext=(0, 7), ha='center', fontsize=6, color='#a04000')
        highest = float(history['current_outlier_fraction'].max())
        if refill is not None and not refill.empty:
            highest = max(highest, float(refill['shadow_outlier_fraction'].max()))
        ax.set_ylim(top=highest * 1.25 + 0.02)   # room for the labels
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
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)   # beside the plot
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_alarm_density(offsets_df, fig_dir, bin_width=250):
    """Where alarms fall relative to the switch (negative = before = false alarm), per dataset.
    One panel per composition, one line per model; bins of one window (250 stream triples)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (dataset, version, policy_tag), d in offsets_df.groupby(['dataset', 'version', 'policy_tag']):
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
            ax.set_title(f'{dataset} - {version} - {comp} - {policy_tag}', fontsize=9)
            ax.set_xlabel('stream triples from switch (negative = before)')
        axes[0][0].set_ylabel(f'alarms per {bin_width} triples')
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f'alarm_density__{dataset}__{version}__{policy_tag}.png', dpi=140)
        plt.close(fig)


def load_quality(quality_dir):
    if quality_dir is None or not Path(quality_dir).exists():
        return pd.DataFrame()
    rows = []
    for f in Path(quality_dir).glob('*.json'):
        row = json.loads(f.read_text())
        row['quality_of'] = f.stem.split('__')[-1]   # 'phase1' (v1 settings) or the version tag (v2e2, v2e10, v2e20)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[['dataset', 'model', 'quality_of', 'mrr', 'hits_at_10']]


def plot_policy_tradeoff(table, fig_dir):
    """Retrain share (cost) vs alarm quality, per dataset and composition, one line per model,
    points = policies (sphere sweep, Antoine k=1, always)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (dataset, version), d in table.groupby(['dataset', 'version']):
        d = d[d['setting'].eq('base')] if 'setting' in d else d
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
                ax.set_title(f'{dataset} - {version} - {comp}', fontsize=9)
                ax.set_xlabel('share of new-entity re-embeds that retrained the OCSVM')
                ax.set_ylabel(metric)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f'policy_tradeoff__{dataset}__{version}.png', dpi=140)
        plt.close(fig)


def version_comparison(table):
    """Rows = dataset, model, composition, rule; columns = the versions, for the alarm counts, the delay, the
    share of the stream that is tested and the run time. One run per cell, nothing is averaged."""
    values = [c for c in ('num_detected_drifts', 'alarms_pre', 'alarms_post', 'detection_delay', 'tested_share',
                          'runtime_min') if c in table]
    if 'setting' in table:
        table = table[table['setting'].eq('base')]
    wide = table.pivot_table(index=['dataset', 'model', 'composition', 'policy_tag'], columns='version',
                             values=values, aggfunc='first')
    wide.columns = [f'{value}__{version}' for value, version in wide.columns]
    return wide.reset_index()


def literature_tables(peaks_all, run_dirs, data_dir, mondo, out):
    """For finding the papers behind the peaks (by hand): per dataset, one row per peak DATE over all runs
    (how many runs have a peak there, how many have their highest peak or an alarm there), and every
    monitored triple of the data file with that date (gene or miRNA, disease, file row), since a date
    usually carries several triples from several genes.
      literature/<dataset>__peak_dates.csv        one row per date
      literature/<dataset>__peak_date_triples.csv one row per (date, triple)"""
    out.mkdir(parents=True, exist_ok=True)
    configs_by_dataset = {}
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        configs_by_dataset.setdefault(cfg.get('dataset'), cfg)
    base = peaks_all[peaks_all['setting'].eq('base')] if 'setting' in peaks_all else peaks_all
    for dataset, d in base.groupby('dataset'):
        d = d.dropna(subset=['date'])
        dates = d.groupby('date').agg(
            runs_with_peak=('run', 'nunique'),
            runs_with_highest_peak=('rank', lambda r: int((r == 1).sum())),
            runs_with_alarm_at_peak=('alarm_here', 'sum'),
            versions=('version', lambda v: ' '.join(sorted(set(v)))),
            rules=('policy_tag', lambda v: ' '.join(sorted(set(v)))),
            highest_outlier_rate=('outlier_rate', 'max'),
            first_stream_index=('stream_index', 'min')).reset_index()
        if 'triples_from_switch' in d:
            dates = dates.merge(d.groupby('date')['triples_from_switch'].min().rename('triples_from_switch').reset_index(),
                                on='date', how='left')
        stream = monitored_stream(data_dir, configs_by_dataset.get(dataset, {})) if dataset in configs_by_dataset else None
        if stream is not None:
            s = stream.assign(date=to_date(stream['timestamp']).dt.strftime('%Y-%m-%d'))
            s = s[s['date'].isin(set(dates['date']))].copy()
            if mondo is not None:
                described = [mondo.describe(a, b) for a, b in zip(s['subject'], s['object'])]
                s['disease_id'] = [x[0] for x in described]
                s['disease_name'] = [x[1] for x in described]
                s['disease_group'] = [x[2] for x in described]
            counts = s.groupby('date').agg(triples_on_date=('subject', 'size'),
                                           subjects_on_date=('subject', 'nunique')).reset_index()
            dates = dates.merge(counts, on='date', how='left')
            s.sort_values(['date', 'row_number']).to_csv(out / f'{dataset}__peak_date_triples.csv', index=False)
        dates.sort_values(['runs_with_peak', 'date'], ascending=[False, True]).to_csv(
            out / f'{dataset}__peak_dates.csv', index=False)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', default=None,
                   help="run folders (default: every version's folder next to this script that exists; "
                        'the correctness check and the sensitivity runs are analysed with --runs)')
    p.add_argument('--out', type=Path, default=Path('analysis'))
    p.add_argument('--quality', type=Path, default=Path('quality'), help='folder with embedding_quality.py outputs')
    p.add_argument('--data-dir', type=Path, default=here.parent / 'data',
                   help='data files, for the number of stream triples that share a date (skipped if missing)')
    p.add_argument('--mondo', type=Path, default=here.parent / 'data' / 'mondo.obo',
                   help='MONDO ontology, for disease names and groups (skipped if missing)')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    (args.out / 'figures').mkdir(parents=True, exist_ok=True)
    (args.out / 'peaks').mkdir(parents=True, exist_ok=True)
    run_folders = args.runs or [here / v['runs_dir'] for v in VERSION_FOLDERS if (here / v['runs_dir']).exists()]
    mondo = Mondo(args.mondo) if args.mondo.exists() else None
    if mondo is None:
        print(f'no MONDO file at {args.mondo}: peak tables have no disease names')

    rows, offsets, all_peaks = [], [], []
    run_dirs = sorted((d for folder in run_folders for d in folder.iterdir() if (d / 'summary.json').exists()),
                      key=lambda d: d.name)
    for run_dir in run_dirs:
        summary, config, history, shifts = load_run(run_dir)
        window = int(config['window_size'])
        switch_year = config.get('switch_year')
        first_tested = None
        if switch_year is not None and not history.empty:
            after = history.loc[history['event_year'] >= switch_year, 'stream_index']
            first_tested = int(after.iloc[0]) if len(after) else None
        switch_index = true_switch_index(args.data_dir, config)
        if switch_year is not None and switch_index is None:
            print(f'[{run_dir.name}] data file not found: the switch is placed at the first tested triple')
            switch_index = first_tested
        switch_timestamp = None
        if switch_index is not None and monitored_stream(args.data_dir, config) is not None:
            switch_timestamp = monitored_stream(args.data_dir, config)['timestamp'].iloc[switch_index - 1]
        row = {k: summary.get(k) for k in ('run', 'dataset', 'model', 'composition', 'preset', 'retrain_policy',
                                           'sphere_k', 'retrain_every', 'learning_rate', 'epochs', 'warm_epochs',
                                           'warm_learning_rate', 'seed',
                                           'feature_dim', 'future_rows', 'processed_future_triples',
                                           'tested_steps', 'num_detected_drifts', 'num_unseen_refresh_events',
                                           'num_drift_refresh_events', 'num_threshold_retrains',
                                           'num_scheduled_retrains', 'num_scheduled_refits',
                                           'num_refill_triples', 'num_refill_scorable', 'unseen_retrain_ratio',
                                           'initial_reference_outlier_fraction', 'total_embedding_seconds',
                                           'runtime_seconds', 'imported_from')}
        for k in ('nu', 'window_size', 'alpha', 'keep_fraction_after_drift', 'seed'):
            row[k] = config.get(k)
        row['version'] = summary.get('version') or 'v1'   # runs made before the versions existed are v1
        parts = summary['run'].split('__')
        row['policy_tag'] = parts[3]
        row['setting'] = parts[5] if len(parts) > 5 else 'base'   # sensitivity runs: the changed setting
        row['switch_index'] = switch_index
        row['switch_index_first_tested'] = first_tested
        row['switch_tested'] = bool(switch_index is not None and (history['stream_index'] == switch_index).any())
        if row.get('processed_future_triples'):
            row['tested_share'] = float(len(history) / row['processed_future_triples'])
        row['mean_reference_outlier_rate'] = float(history['reference_outlier_fraction'].mean()) if len(history) else np.nan
        row['mean_current_outlier_rate'] = float(history['current_outlier_fraction'].mean()) if len(history) else np.nan
        row.update(alarm_metrics(history, shifts, switch_index, window))
        row.update(peak_metrics(history, switch_index, window))
        row.update(sphere_metrics(shifts))
        trace = pd.read_csv(run_dir / 'step_trace.tsv.gz', sep='\t')
        # Refill triples with shadow logging (runs made before 2026-11-16 do not have them).
        refill = trace[trace['refill'].fillna(False).astype(bool)] if 'refill' in trace else pd.DataFrame()
        if not refill.empty:
            scored = refill[refill['shadow_scorable'].astype(bool)]
            row['refill_scorable_share'] = float(len(scored) / len(refill))
            row['refill_outlier_share'] = float(scored['shadow_outlier'].mean()) if len(scored) else np.nan
            if switch_index is not None and switch_year is not None:
                row['refill_triples_from_switch_year'] = int((refill['event_year'] >= switch_year).sum())
        trace = trace[trace['drift_tested'].astype(bool)]
        peaks = peak_table(trace, switch_index, window, mondo, same_date_counts(args.data_dir, config),
                           switch_timestamp=switch_timestamp)
        row.update(peak_summary(peaks))
        if not peaks.empty:
            peaks.to_csv(args.out / 'peaks' / f'{summary["run"]}.csv', index=False)
            ident = {k: row[k] for k in ('run', 'dataset', 'version', 'model', 'composition', 'policy_tag', 'setting')}
            all_peaks.append(peaks.assign(**ident)[list(ident) + list(peaks.columns)])
        rows.append(row)
        if switch_index is not None:
            for a in history.loc[history['drift_detected'].astype(bool), 'stream_index']:
                offsets.append({'run': summary['run'], 'dataset': summary['dataset'], 'version': row['version'],
                                'model': summary['model'], 'composition': summary['composition'],
                                'policy_tag': row['policy_tag'], 'setting': row['setting'],
                                'alarm_stream_index': int(a), 'offset_from_switch': int(a - switch_index)})
        if not args.no_plots:
            plot_run(summary['run'], history, shifts, switch_index, args.out / 'figures' / f'{summary["run"]}.png',
                     peaks=peaks, refill=refill)

    table = pd.DataFrame(rows)
    if table.empty:
        print('no finished runs found in', [str(f) for f in run_folders])
        return
    if all_peaks:
        peaks_all = pd.concat(all_peaks, ignore_index=True)
        peaks_all.to_csv(args.out / 'peaks_all.csv', index=False)
        literature_tables(peaks_all, run_dirs, args.data_dir, mondo, args.out / 'literature')
    table['runtime_min'] = (table.pop('runtime_seconds') / 60).round(1)
    quality = load_quality(args.quality)
    if not quality.empty:
        # MRR of the embedding the run starts from: each from-scratch version has its own (its epochs);
        # v1 and v3 share v1's (v3's first embedding is trained as in v1).
        table['quality_dataset'] = table['dataset'].replace({'full_short': 'full'})
        table['quality_of'] = [configs.VERSIONS[v]['tag'] if v in configs.VERSIONS and v not in ('v1', 'v3')
                               else 'phase1' for v in table['version']]
        table = table.merge(quality, how='left', left_on=['quality_dataset', 'model', 'quality_of'],
                            right_on=['dataset', 'model', 'quality_of'],
                            suffixes=('', '_q')).drop(columns=['quality_dataset', 'dataset_q', 'quality_of'])
    table = table.sort_values(['dataset', 'version', 'model', 'composition', 'policy_tag', 'setting']).reset_index(drop=True)
    table.to_csv(args.out / 'summary_table.csv', index=False)
    if table['version'].nunique() > 1:
        comparison = version_comparison(table)
        comparison.to_csv(args.out / 'version_comparison.csv', index=False)
        try:
            (args.out / 'version_comparison.md').write_text(comparison.round(3).to_markdown(index=False), encoding='utf-8')
        except ImportError:  # tabulate not installed
            (args.out / 'version_comparison.md').write_text(comparison.round(3).to_string(index=False), encoding='utf-8')
    offsets_df = pd.DataFrame(offsets)
    if not offsets_df.empty:
        offsets_df.to_csv(args.out / 'alarm_offsets.csv', index=False)
        if not args.no_plots:
            plot_alarm_density(offsets_df, args.out / 'figures')
    if not args.no_plots:
        plot_policy_tradeoff(table, args.out / 'figures')
    key_cols = [c for c in ['dataset', 'version', 'model', 'composition', 'policy_tag', 'setting', 'sphere_k', 'mrr',
                            'num_detected_drifts', 'tested_share', 'switch_tested',
                            'alarms_pre', 'alarms_post', 'false_alarms_per_1000', 'last_false_alarm_before_switch',
                            'detection_delay', 'switch_to_peak', 'switch_to_second_peak', 'highest_to_second_triples',
                            'peak_vs_second', 'n_peaks', 'density_pre', 'density_post', 'num_unseen_refresh_events',
                            'num_threshold_retrains', 'num_scheduled_refits', 'unseen_retrain_ratio',
                            'alarms_after_reembed_share',
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
