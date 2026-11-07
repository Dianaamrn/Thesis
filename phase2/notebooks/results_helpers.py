"""Tables and figures for phase2_results.ipynb. Reads phase2/outputs/results/ (written by step6_analyze.py).

Kept out of the notebook so the notebook shows only the questions, the results and how to read them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'phase2' / 'settings.py').exists())
RESULTS = Path(os.environ.get('PHASE2_RESULTS', REPO / 'phase2' / 'outputs' / 'results'))

# Categorical colours in a fixed order (validated default palette, light mode). Colour follows the
# entity: a model always has the same colour in every figure.
MODEL_COLORS = {'TransE': '#2a78d6', 'TransH': '#eb6834', 'DistMult': '#1baf7a', 'ComplEx': '#eda100',
                'RotatE': '#e87ba4'}
METHOD_COLORS = {'A0': '#eb6834', 'A1': '#2a78d6'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
MODELS = list(MODEL_COLORS)
SWITCH = {'case1': 2005, 'case2': 2022, 'full': None}

CHANNEL_LABELS = {
    'C1_types': 'C1 triple types',
    'C2_novelty': 'C2 new entities',
    'C3_ontology': 'C3 MONDO categories',
    'C4_plausibility': 'C4 KGE plausibility',
    'C4_mmd': 'C4 MMD on vectors',
    'C5_insample_fixed': 'C5 OCSVM, in-sample ref., fixed thr. (as A1)',
    'C5_conformal_fixed': 'C5 OCSVM, conformal ref., fixed thr.',
    'C5_conformal_calibrated': 'C5 OCSVM, conformal ref., calibrated thr. (A2)',
}


def style():
    plt.rcParams.update({
        'figure.dpi': 110, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 2, 'legend.frameon': False,
    })


style()


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f'-- {name} not found in {RESULTS}: run the corresponding step on the server, then step6_analyze.py')
        return None
    return pd.read_csv(path)


def meta():
    p = RESULTS / 'meta.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def _empty(df):
    return df is None or len(df) == 0


# ---------------------------------------------------------------------------
# Experiment 2: A0 vs A1 on the real stream
# ---------------------------------------------------------------------------

def exp2_table():
    df = load('exp2_a0_vs_a1_real.csv')
    if _empty(df):
        return None
    cols = ['dataset', 'model', 'composition', 'method', 'num_detected_drifts', 'alarms_pre', 'alarms_post',
            'false_alarms_per_1000', 'detection_delay', 'num_unseen_refresh_events', 'alarms_after_reembed_share',
            'tested_steps', 'mean_reference_outlier_rate', 'mean_current_outlier_rate']
    out = df[[c for c in cols if c in df]].rename(columns={
        'num_detected_drifts': 'alarms', 'num_unseen_refresh_events': 're-embeds',
        'alarms_after_reembed_share': 'alarms within a window of a re-embed', 'tested_steps': 'triples tested'})
    return out.sort_values(['dataset', 'composition', 'model', 'method']).reset_index(drop=True).round(3)


def plot_exp2():
    df = load('exp2_a0_vs_a1_real.csv')
    if _empty(df):
        return
    for ds, d in df.groupby('dataset'):
        comps = sorted(d['composition'].unique())
        fig, axes = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 3.0), sharey=True, squeeze=False)
        for ax, comp in zip(axes[0], comps):
            sub = d[d['composition'] == comp]
            x = np.arange(len(MODELS))
            for k, m in enumerate(['A0', 'A1']):
                vals = [sub.loc[(sub['model'] == mod) & (sub['method'] == m), 'num_detected_drifts'].sum()
                        if ((sub['model'] == mod) & (sub['method'] == m)).any() else np.nan for mod in MODELS]
                ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=METHOD_COLORS[m],
                       label={'A0': 'A0 re-embed (Phase 1)', 'A1': 'A1 frozen'}[m], edgecolor='white', linewidth=1)
            ax.set_xticks(x, MODELS)
            ax.set_title(f'{ds}, {comp}')
            ax.grid(axis='x', visible=False)
        axes[0][0].set_ylabel('drift alarms on the real stream')
        axes[0][-1].legend(loc='upper right')
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Experiment 1: no-drift controls
# ---------------------------------------------------------------------------

def exp1_frozen_table():
    df = load('exp1_a0_a1_control.csv')
    if _empty(df):
        return None
    g = df.groupby(['dataset', 'method', 'model', 'composition']).agg(
        runs=('run', 'count'), alarms_mean=('num_detected_drifts', 'mean'),
        alarms_per_1000_tested=('alarms_per_1000_tested', 'mean'),
        runs_with_any_alarm=('num_detected_drifts', lambda s: float((s > 0).mean())),
        triples_tested=('tested_steps', 'mean'), re_embeds=('num_unseen_refresh_events', 'mean'))
    return g.reset_index().round(3)


def plot_exp1_frozen():
    t = exp1_frozen_table()
    if _empty(t):
        return
    for ds, d in t.groupby('dataset'):
        comps = sorted(d['composition'].unique())
        fig, axes = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 3.0), sharey=True, squeeze=False)
        for ax, comp in zip(axes[0], comps):
            sub = d[d['composition'] == comp]
            x = np.arange(len(MODELS))
            for k, m in enumerate(['A0', 'A1']):
                vals = [sub.loc[(sub['model'] == mod) & (sub['method'] == m), 'alarms_per_1000_tested'].mean()
                        for mod in MODELS]
                ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=METHOD_COLORS[m], edgecolor='white',
                       linewidth=1, label={'A0': 'A0 re-embed', 'A1': 'A1 frozen'}[m])
            ax.set_xticks(x, MODELS)
            ax.set_title(f'{ds}, {comp}: NO drift')
            ax.grid(axis='x', visible=False)
        axes[0][0].set_ylabel('false alarms per 1,000 tested triples')
        axes[0][-1].legend(loc='upper right')
        fig.tight_layout()
        plt.show()


def _channel_summary():
    df = load('channel_summary.csv')
    if _empty(df):
        return None
    df['label'] = df['channel'].map(CHANNEL_LABELS).fillna(df['channel'])
    return df


def exp1_channel_table():
    df = _channel_summary()
    if _empty(df):
        return None
    cols = ['dataset', 'label', 'model', 'composition', 'control_runs', 'control_stream_length',
            'control_any_alarm', 'control_alarms_per_1000']
    return df[cols].sort_values(['dataset', 'label', 'model', 'composition']).reset_index(drop=True).round(3)


def target_any_alarm(stream_length):
    """Expected share of no-drift streams with any alarm if calibration holds exactly."""
    c = meta_value('calibration') or {'horizon': 1000, 'beta': 0.1}
    return 1 - (1 - c['beta']) ** (stream_length / c['horizon'])


def meta_value(key):
    for ds_meta in meta().values():
        if key in ds_meta:
            return ds_meta[key]
    return None


def _dot_panel(ax, d, col, xlabel, ref=None):
    order = [lab for lab in CHANNEL_LABELS.values() if lab in set(d['label'])]
    for i, lab in enumerate(order):
        v = d.loc[d['label'] == lab, col].dropna()
        if v.empty:
            continue
        ax.plot([v.min(), v.max()], [i, i], color=MUTED, linewidth=1.5, solid_capstyle='round')
        ax.plot(v.median(), i, 'o', color='#2a78d6', markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    if ref is not None:
        ax.axvline(ref, color=INK, linewidth=1, linestyle='--')


def plot_exp1_channels():
    df = _channel_summary()
    if _empty(df):
        return
    for ds, d in df.groupby('dataset'):
        if d['control_runs'].isna().all():
            continue
        L = d['control_stream_length'].median()
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        _dot_panel(ax, d, 'control_any_alarm', 'share of no-drift streams with at least one false alarm',
                   ref=target_any_alarm(L))
        ax.set_title(f'{ds}: no drift (dot = median over models/compositions, line = range; '
                     f'dashed = calibration target for {L:.0f} triples)', loc='left')
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Experiment 3: seed / alignment test
# ---------------------------------------------------------------------------

PAIR_LABELS = {'seed_only': 'same graph, other seed', 'same_seed_grown': 'grown graph, same seed (= A0 re-embed)',
               'cross': 'grown graph, other seed'}


def exp3_table():
    df = load('exp3_seed_pairs.csv')
    if _empty(df):
        return None
    cols = ['raw_cosine', 'aligned_cosine', 'procrustes_resid', 'linear_cka', 'knn_overlap',
            'ocsvm_hadamard_reference_insample', 'ocsvm_hadamard_same_frame', 'ocsvm_hadamard_other_frame_raw',
            'ocsvm_hadamard_other_frame_aligned', 'ocsvm_concatenation_reference_insample',
            'ocsvm_concatenation_same_frame', 'ocsvm_concatenation_other_frame_raw',
            'ocsvm_concatenation_other_frame_aligned']
    g = df.groupby(['model', 'pair'])[[c for c in cols if c in df]].mean().reset_index()
    g['pair'] = g['pair'].map(PAIR_LABELS).fillna(g['pair'])
    return g.round(3)


def plot_exp3():
    df = load('exp3_seed_pairs.csv')
    if _empty(df):
        return
    pairs = [p for p in PAIR_LABELS if p in set(df['pair'])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (col, title) in zip(axes, [('raw_cosine', 'same entity, raw coordinates (cosine)'),
                                       ('linear_cka', 'linear CKA (ignores rotation)'),
                                       ('knn_overlap', '10-nearest-neighbour overlap')]):
        x = np.arange(len(pairs))
        for k, m in enumerate(MODELS):
            vals = [df.loc[(df['model'] == m) & (df['pair'] == p), col].mean() for p in pairs]
            ax.bar(x + (k - 2) * 0.16, vals, width=0.15, color=MODEL_COLORS[m], label=m, edgecolor='white', linewidth=1)
        ax.set_xticks(x, [PAIR_LABELS[p].replace(' (', '\n(').replace(', ', ',\n', 1) for p in pairs], fontsize=7)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('agreement between the two embeddings')
    axes[-1].legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
    for ax, comp in zip(axes, ['hadamard', 'concatenation']):
        cols = [f'ocsvm_{comp}_reference_insample', f'ocsvm_{comp}_same_frame', f'ocsvm_{comp}_other_frame_raw',
                f'ocsvm_{comp}_other_frame_aligned']
        labels = ['reference\n(in-sample)', 'new triples,\nsame embedding', 'new triples,\nre-trained (raw)',
                  'new triples,\nre-trained + aligned']
        if not all(c in df for c in cols):
            continue
        sub = df[df['pair'] == 'same_seed_grown']
        x = np.arange(len(cols))
        for k, m in enumerate(MODELS):
            vals = [sub.loc[sub['model'] == m, c].mean() for c in cols]
            ax.bar(x + (k - 2) * 0.16, vals, width=0.15, color=MODEL_COLORS[m], label=m, edgecolor='white', linewidth=1)
        ax.axhline(0.1, color=INK, linewidth=1, linestyle='--')
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(f'OCSVM outlier rate, {comp} (dashed = nu = 0.1)')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('share flagged as outlier')
    axes[-1].legend(loc='upper left', fontsize=7)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Experiments 5-9, 11, 12: channels on the known switch
# ---------------------------------------------------------------------------

def channel_table(dataset=None):
    df = _channel_summary()
    if _empty(df):
        return None
    if dataset:
        df = df[df['dataset'] == dataset]
    cols = ['dataset', 'label', 'model', 'composition', 'control_any_alarm', 'control_alarms_per_1000',
            'detection_rate', 'median_delay', 'false_alarms_per_1000_before_switch',
            'real_alarms', 'real_false_alarms', 'real_delay', 'real_first_alarm_year']
    return df[[c for c in cols if c in df]].reset_index(drop=True).round(3)


def calibration_table():
    df = _channel_summary()
    if _empty(df):
        return None
    d = df[df['channel'].str.startswith('C5_')]
    return d.groupby(['dataset', 'label']).agg(
        no_drift_any_alarm=('control_any_alarm', 'median'), no_drift_alarms_per_1000=('control_alarms_per_1000', 'median'),
        detection_rate=('detection_rate', 'median'), median_delay=('median_delay', 'median'),
        false_alarms_per_1000_before_switch=('false_alarms_per_1000_before_switch', 'median')).reset_index().round(3)


def plot_channels():
    df = _channel_summary()
    if _empty(df):
        return
    for ds, d in df.groupby('dataset'):
        if d['inject_runs'].isna().all():
            continue
        L = d['control_stream_length'].median()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
        _dot_panel(axes[0], d, 'control_any_alarm', 'no drift: share of streams with a false alarm',
                   ref=target_any_alarm(L))
        _dot_panel(axes[1], d, 'detection_rate', 'switch injected: share detected')
        _dot_panel(axes[2], d, 'median_delay', 'switch injected: median delay (triples)')
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        fig.suptitle(f'{ds}: every channel (dot = median over models/compositions, line = range)', x=0.01,
                     ha='left', fontsize=10)
        fig.tight_layout()
        plt.show()


def plot_real_series(dataset, model='TransE', composition='hadamard'):
    df = load('channels_real_series.csv.gz')
    if _empty(df):
        return
    d = df[(df['dataset'] == dataset) & (((df['model'] == '-') & (df['composition'] == '-'))
                                         | ((df['model'] == model) & (df['composition'].isin([composition, '-']))))]
    chans = [c for c in CHANNEL_LABELS if c in set(d['channel'])]
    if not chans:
        return
    n = len(chans)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 1.9 * ((n + 1) // 2)), sharex=True, squeeze=False)
    sw = SWITCH.get(dataset)
    for ax, ch in zip(axes.ravel(), chans):
        s = d[d['channel'] == ch].sort_values('position')
        ax.plot(s['position'], s['stat'], color='#2a78d6', linewidth=1.2)
        ax.axhline(s['threshold'].iloc[0], color=INK, linewidth=1, linestyle='--')
        a = s[s['alarm'].astype(bool)]
        ax.plot(a['position'], a['stat'], 'o', color='#eb6834', markersize=6, markeredgecolor='white', markeredgewidth=1.2)
        if sw is not None:
            post = s[s['event_year'] >= sw]
            if len(post):
                ax.axvline(post['position'].iloc[0], color=MUTED, linewidth=1)
        ax.set_title(CHANNEL_LABELS[ch], loc='left', fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('position in the stream (triples)')
    fig.suptitle(f'{dataset}, real order ({model}, {composition} for model channels). Line = statistic, '
                 'dashed = threshold, dots = alarms, grey = switch', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Experiment 10: graph distances between years
# ---------------------------------------------------------------------------

def graph_tables():
    return load('exp10_graph_series.csv'), load('exp10_graph_changepoints.csv')


def plot_graph():
    s, c = graph_tables()
    if _empty(s):
        return
    measures = [m for m in ('disease_jaccard', 'wl_distance', 'deltacon') if m in s]
    titles = {'disease_jaccard': 'diseases studied: 1 - Jaccard (year vs previous year)',
              'wl_distance': 'typed WL histogram distance (year vs previous year)',
              'deltacon': 'DeltaCon distance (cumulative graph, year vs previous year)'}
    for ds, d in s.groupby('dataset'):
        fig, axes = plt.subplots(1, len(measures), figsize=(4.3 * len(measures), 2.8), squeeze=False)
        for ax, m in zip(axes[0], measures):
            ax.plot(d['year'], d[m], color='#2a78d6', marker='o', markersize=4)
            if SWITCH.get(ds):
                ax.axvline(SWITCH[ds] - 0.5, color=MUTED, linewidth=1)
            if c is not None:
                cp = c[(c['dataset'] == ds) & (c['measure'] == m)]
                if len(cp) and cp['changepoint_year'].notna().all():
                    ax.axvline(cp['changepoint_year'].iloc[0] - 0.5, color='#eb6834', linewidth=1, linestyle='--')
            ax.set_title(titles[m], fontsize=8, loc='left')
            ax.set_xlabel('year')
        fig.suptitle(f'{ds}: grey = known switch, orange dashed = detected change point', x=0.01, ha='left', fontsize=10)
        fig.tight_layout()
        plt.show()
