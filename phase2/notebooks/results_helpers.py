"""Tables and figures for the Phase 2 notebooks. Reads phase2/outputs/results/ (written by step6_analyze.py).

Kept out of the notebooks so they show only the question, the result and how to read it.
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

# Fixed categorical order (validated default palette, light mode). Colour follows the entity.
MODEL_COLORS = {'TransE': '#2a78d6', 'TransH': '#eb6834', 'DistMult': '#1baf7a', 'ComplEx': '#eda100',
                'RotatE': '#e87ba4'}
METHOD_COLORS = {'original': '#eb6834', 'aligned': '#1baf7a', 'frozen': '#2a78d6'}
METHOD_LABELS = {'original': "Antoine (retrain from scratch)", 'aligned': 'retrain + align',
                 'frozen': 'fixed embedding + fold-in'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
MODELS = list(MODEL_COLORS)
SWITCH = {'case1': 2005, 'case2': 2022, 'full': None}
CUTOFF = {'case1': 2002, 'case2': 2018, 'full': 2010}
CHANNEL_LABELS = {
    'C1_types': 'triple types',
    'C2_novelty': 'new entities',
    'C3_ontology': 'MONDO disease categories',
    'C4_plausibility': 'KGE plausibility (KS)',
    'C4_mmd': 'MMD on triple vectors',
    'C5_insample_fixed': "OCSVM, Antoine's statistics",
    'C5_conformal_fixed': 'OCSVM, fair reference',
    'C5_conformal_calibrated': 'OCSVM, fair reference + calibrated threshold',
}
PAIR_LABELS = {'identical_rerun': 'same graph\nsame seed\ntrained twice',
               'seed_only': 'same graph\nother seed',
               'same_seed_grown': 'grown graph\nfrom scratch\n(= Antoine)',
               'warm_start_grown': 'grown graph\nwarm start',
               'cross': 'grown graph\nother seed'}

plt.rcParams.update({
    'figure.dpi': 110, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load(name):
    path = RESULTS / name
    if not path.exists():
        print(f'-- {name}: not available yet (run the corresponding step on the server, then step6_analyze.py)')
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def meta():
    p = RESULTS / 'meta.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def settings_summary():
    for ds, m in meta().items():
        print(f"{ds}: MONDO {m.get('mondo_version')}; embedder {m.get('embedder')}; KGE {m.get('kge')}; "
              f"window {m.get('window')}; calibration {m.get('calibration')}")


def _bars(ax, groups, series, values, colors, labels=None, width=0.8):
    """Grouped bars: values[(group, series)]. Thin bars, white gap between neighbours."""
    n = len(series)
    w = width / max(n, 1)
    x = np.arange(len(groups))
    for k, s in enumerate(series):
        v = [values.get((g, s), np.nan) for g in groups]
        ax.bar(x + (k - (n - 1) / 2) * w, v, width=w * 0.92, color=colors[s], label=(labels or {}).get(s, s),
               edgecolor='white', linewidth=0.8)
    ax.set_xticks(x, groups)
    ax.grid(axis='x', visible=False)


def _dots(ax, d, col, label_col, order, xlabel, ref=None):
    """One row per item: dot = median, line = range (over models / compositions / repeats)."""
    order = [o for o in order if o in set(d[label_col])]
    for i, lab in enumerate(order):
        v = d.loc[d[label_col] == lab, col].dropna()
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


# ===========================================================================
# PART 1
# ===========================================================================

def sphere_ratios():
    """1.1 Every sphere-rule ratio logged at a Phase 1 re-embed (Antoine's rule, k = 1)."""
    r = load('phase1_sphere_ratios.csv')
    if r is None:
        return None
    a = r[r['policy_tag'] == 'sphere1']
    if a.empty:
        a = r
    datasets = list(dict.fromkeys(a['dataset']))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 2.8), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], datasets):
        v = a.loc[a['dataset'] == ds, 'ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(v, bins=np.linspace(0, max(1.2, v.max() * 1.05), 40), color='#2a78d6', edgecolor='white', linewidth=0.5)
        ax.axvline(1.0, color='#eb6834', linewidth=1.5)
        ax.text(1.0, ax.get_ylim()[1] * 0.92, '  fires only beyond here', color='#eb6834', fontsize=8, va='top')
        ax.set_title(f'{ds}: {len(v)} re-embeds, largest ratio {v.max():.2f}', loc='left')
        ax.set_xlabel('distance to box centre / half diagonal')
    axes[0][0].set_ylabel('re-embeds')
    fig.tight_layout()
    plt.show()
    return a.groupby('dataset').agg(re_embeds=('ratio', 'size'), times_rule_fired=('retrained', 'sum'),
                                    median_ratio=('ratio', 'median'), p95_ratio=('ratio', lambda s: s.quantile(.95)),
                                    max_ratio=('ratio', 'max')).round(3)


def sphere_sweep():
    """1.1 The sphere-size sweep: how often the rule retrains vs false alarms and re-embed-driven alarms."""
    t = load('phase1_runs.csv')
    if t is None:
        return None
    order = ['sphere1', 'sphereR10', 'sphereR50', 'sphereR90', 'always']
    t = t[t['policy_tag'].isin(order)].copy()
    t['policy_tag'] = pd.Categorical(t['policy_tag'], order, ordered=True)
    datasets = [d for d in ('case1', 'case2', 'full_short') if d in set(t['dataset'])]
    if not datasets:
        print('-- no sphere-sweep runs in phase1_runs.csv yet')
        return None
    cols = [('unseen_retrain_ratio', 'share of re-embeds where the SVM was retrained'),
            ('false_alarms_per_1000', 'false alarms per 1,000 tested pre-switch triples'),
            ('alarms_after_reembed_share', 'share of alarms right after a re-embed')]
    fig, axes = plt.subplots(len(datasets), len(cols), figsize=(12.5, 2.6 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        d = t[t['dataset'] == ds]
        for j, (col, lab) in enumerate(cols):
            ax = axes[i][j]
            if col not in d or d[col].isna().all():
                ax.set_visible(False)
                continue
            for m in MODELS:
                for comp, mk in (('hadamard', 'o'), ('concatenation', 's')):
                    s = d[(d['model'] == m) & (d['composition'] == comp)].sort_values('policy_tag')
                    if len(s):
                        ax.plot(s['policy_tag'].astype(str), s[col], marker=mk, color=MODEL_COLORS[m], linewidth=1,
                                markersize=4, label=f'{m} {comp[:3]}' if (i == 0 and j == 2) else None)
            ax.set_title(f'{ds}: {lab}', loc='left', fontsize=8)
            ax.tick_params(axis='x', labelsize=7)
    handles, labels = axes[0][2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Sphere size: Antoine (k = 1) -> smaller spheres (retrain ~10/50/90%) -> always retrain '
                 '(circle = Hadamard, square = concatenation)', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    plt.show()
    keep = ['dataset', 'policy_tag', 'unseen_retrain_ratio', 'num_detected_drifts', 'false_alarms_per_1000',
            'alarms_after_reembed_share', 'detection_delay', 'tested_steps']
    return (t[[c for c in keep if c in t]].groupby(['dataset', 'policy_tag'], observed=True).median(numeric_only=True)
            .round(3).rename(columns=lambda c: c + ' (median)'))


QUOTES = [
    ('Mahdavi et al. 2018 (dynnode2vec)', '"First, embedding vectors for each timestamp are in different spaces."', True),
    ('Singer, Guy & Radinsky, IJCAI 2019 (tNodeEmbed)',
     '"the coordinate axes of the embedded graph nodes are not guaranteed to align"', True),
    ('Gursoy, Haddad & Bothorel 2021',
     '"even dynamic methods in the literature do not always guarantee embedding alignment between timesteps."', True),
    ('Meroue, Gandon & Monnin 2026 (TransE, RotatE, DistMult, ComplEx, ...)',
     'models with similar MRR had "highly variable embedding spaces"; for DistMult two runs share only '
     '~5.6 of each entity\'s 10 nearest neighbours', False),
    ('Polleres et al. 2023 (TGDK survey, summarising Pernisch et al.)', '"comparing two embeddings to each other is complex"', False),
    ('Hamilton, Leskovec & Jurafsky, ACL 2016', 'train one embedding per decade, then align consecutive decades with '
     'orthogonal Procrustes before comparing them (method, paraphrased)', False),
    ('Hu et al., KDD 2022', 'feeding retrained embeddings to a model built on the old ones gives arbitrary predictions; '
     'learning an alignment cut that model\'s degradation from -26.45% to -0.65% (paraphrased)', False),
    ('Wewer, Lemmerich & Cochez 2021', 'recomputing embeddings after small graph changes forces retraining every '
     'downstream model (paraphrased)', False),
    ('Dubossarsky, Weinshall & Grossman, EMNLP 2017', 'reported semantic-change "laws" largely disappeared in a '
     'control with no real change: they were artefacts of the training (paraphrased)', False),
]


def literature():
    """1.2 What other researchers say about comparing embeddings trained separately."""
    for src, text, dagger in QUOTES:
        print(f'- {src}{" [check against PDF]" if dagger else ""}\n    {text}')


def seed_pairs():
    """1.2 Comparability: how much two trainings agree, raw and after alignment."""
    df = load('seed_pairs.csv')
    if df is None:
        return None
    kinds = [k for k in PAIR_LABELS if k in set(df['pair'])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [('raw_cosine', 'same entity, raw coordinates (cosine)'),
                                       ('aligned_cosine', 'same entity after alignment (cosine)'),
                                       ('knn_overlap', 'shared 10 nearest neighbours')]):
        vals = df.groupby(['pair', 'model'])[col].mean().to_dict()
        _bars(ax, kinds, [m for m in MODELS if m in set(df['model'])], vals, MODEL_COLORS)
        ax.set_xticks(range(len(kinds)), [PAIR_LABELS[k] for k in kinds], fontsize=6.5)
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('agreement (1 = identical)')
    axes[-1].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    plt.show()
    cols = ['raw_cosine', 'aligned_cosine', 'procrustes_resid', 'linear_cka', 'knn_overlap']
    t = df.groupby(['pair', 'model'])[cols].mean().round(3).reset_index()
    t['pair'] = pd.Categorical(t['pair'], kinds, ordered=True)
    return t.sort_values(['pair', 'model']).reset_index(drop=True)


def seed_ocsvm():
    """1.2 What a retrained embedding does to Antoine's OCSVM (no drift at all)."""
    df = load('seed_pairs.csv')
    if df is None:
        return None
    cases = [('same_seed_grown', 'retrained from scratch\n(Antoine)'), ('warm_start_grown', 'warm start')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), sharey=True)
    for ax, comp in zip(axes, ['hadamard', 'concatenation']):
        cols = [f'ocsvm_{comp}_reference_insample', f'ocsvm_{comp}_same_frame', f'ocsvm_{comp}_other_frame_raw',
                f'ocsvm_{comp}_other_frame_aligned']
        if not all(c in df for c in cols):
            continue
        labels = ['reference\n(own training data)', 'new triples,\nsame embedding', 'new triples,\nnew embedding',
                  'new triples,\nnew embedding aligned']
        vals = {}
        for kind, _ in cases:
            sub = df[df['pair'] == kind]
            for c, lab in zip(cols, labels):
                vals[(lab, kind)] = sub[c].mean()
        _bars(ax, labels, [k for k, _ in cases if k in set(df['pair'])], vals,
              {'same_seed_grown': '#eb6834', 'warm_start_grown': '#2a78d6'}, labels=dict(cases))
        ax.axhline(0.1, color=INK, linewidth=1, linestyle='--')
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(f'OCSVM outlier rate, {comp} (dashed = nu = 0.1); mean over models', loc='left', fontsize=9)
    axes[0].set_ylabel('share of triples flagged as outliers')
    axes[-1].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    plt.show()


def _stream(scenario=None, epochs=None):
    df = load('stream_runs.csv')
    if df is None:
        return None
    if 'epochs' in df:
        default = df['epochs'].mode().iloc[0] if df['epochs'].notna().any() else None
        df = df[df['epochs'].fillna(default) == (epochs or 10)]
    return df if scenario is None else df[df['scenario'] == scenario]


def stream_real(epochs=None):
    """1.3 / 2.1 / 2.2 The real streams: Antoine vs retrain + align vs fixed embedding, every model."""
    df = _stream('real', epochs)
    if df is None or df.empty:
        return None
    for ds, d in df.groupby('dataset'):
        comps = sorted(d['composition'].unique())
        fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 3), squeeze=False, sharey=True)
        for ax, comp in zip(axes[0], comps):
            s = d[d['composition'] == comp]
            vals = s.groupby(['model', 'method'])['num_detected_drifts'].mean().to_dict()
            _bars(ax, [m for m in MODELS if m in set(s['model'])],
                  [m for m in METHOD_COLORS if m in set(s['method'])], vals, METHOD_COLORS, METHOD_LABELS)
            ax.set_title(f'{ds}, {comp}: alarms on the real stream', loc='left')
        axes[0][0].set_ylabel('drift alarms')
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        plt.show()
    cols = ['dataset', 'model', 'composition', 'method', 'num_detected_drifts', 'alarms_pre', 'alarms_post',
            'num_unseen_refresh_events', 'alarms_after_reembed_share', 'detection_delay', 'tested_steps',
            'peak_vs_second', 'switch_to_peak']
    return df[[c for c in cols if c in df]].sort_values(['dataset', 'composition', 'model', 'method']).round(3) \
        .rename(columns={'num_detected_drifts': 'alarms', 'num_unseen_refresh_events': 're-embeds',
                         'alarms_after_reembed_share': 'share of alarms right after a re-embed',
                         'tested_steps': 'triples tested'}).reset_index(drop=True)


def stream_no_drift(epochs=None):
    """1.4 No-drift streams: every alarm is false."""
    df = _stream('control', epochs)
    if df is None or df.empty:
        return None
    df = df.assign(label=df['method'].map(METHOD_LABELS))
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 2.3), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        _dots(ax, df[df['dataset'] == ds], 'alarms_per_1000_tested', 'label', list(METHOD_LABELS.values()),
              'false alarms per 1,000 tested triples (no drift)')
        ax.set_title(f'{ds}: dot = median over streams and models, line = range', loc='left', fontsize=9)
    fig.tight_layout()
    plt.show()
    return df.groupby(['dataset', 'method', 'model', 'composition']).agg(
        streams=('repeat', 'count'), mean_alarms=('num_detected_drifts', 'mean'),
        alarms_per_1000_tested=('alarms_per_1000_tested', 'mean'),
        share_with_any_alarm=('num_detected_drifts', lambda s: float((s > 0).mean())),
        tested_share=('tested_share', 'mean')).round(3).reset_index()


def insample_bias():
    """1.5 In every Phase 1 run, is the outlier rate on new triples above the reference's own rate?"""
    t = load('phase1_runs.csv')
    if t is None:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for m in MODELS:
        s = t[t['model'] == m]
        ax.plot(s['mean_reference_outlier_rate'], s['mean_current_outlier_rate'], 'o', color=MODEL_COLORS[m],
                markersize=5, markeredgecolor='white', markeredgewidth=0.8, label=m)
    lim = [0, max(0.5, float(np.nanmax(t[['mean_reference_outlier_rate', 'mean_current_outlier_rate']].to_numpy())) * 1.05)]
    ax.plot(lim, lim, color=INK, linewidth=1, linestyle='--')
    ax.set_xlabel("reference outlier rate (SVM's own training data)")
    ax.set_ylabel('outlier rate on new triples')
    ax.set_title('every Phase 1 run (dashed = equal)', loc='left')
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.show()
    above = (t['mean_current_outlier_rate'] > t['mean_reference_outlier_rate'])
    print(f'{int(above.sum())} of {len(t)} Phase 1 runs have a higher outlier rate on new triples than on the reference.')


def statistics_steps():
    """1.5 / 2.3 Antoine's statistics -> fair reference -> + calibrated threshold, same fixed embedding."""
    df = _channel_summary()
    if df is None:
        return None
    d = df[df['channel'].str.startswith('C5_')]
    return d.groupby(['dataset', 'label']).agg(
        no_drift_share_with_false_alarm=('control_any_alarm', 'median'),
        no_drift_alarms_per_1000=('control_alarms_per_1000', 'median'),
        switch_detected=('detection_rate', 'median'), median_delay=('median_delay', 'median')).round(3).reset_index()


# ===========================================================================
# PART 2
# ===========================================================================

def stream_inject(epochs=None):
    """2.1 / 2.2 Injected switch: detection, delay and false alarms for the three ways of handling the embedding."""
    df = _stream('inject', epochs)
    if df is None or df.empty:
        return None
    df = df.assign(detected=df['alarms_post'] > 0)
    return df.groupby(['dataset', 'method', 'model', 'composition']).agg(
        streams=('repeat', 'count'), switch_detected=('detected', 'mean'),
        median_delay=('detection_delay', 'median'), false_alarms_before_switch=('alarms_pre', 'mean'),
        tested_share=('tested_share', 'mean')).round(3).reset_index()


def alignment_log():
    """2.1 Retrain + align: how different each new embedding was before and after rotating it back."""
    a = load('stream_alignment.csv')
    if a is None:
        return None
    return a.groupby('dataset').agg(re_embeds=('kg_rows', 'size'), raw_cosine_before=('raw_cosine', 'median'),
                                    residual_before=('resid_before', 'median'),
                                    residual_after=('resid_after', 'median')).round(3)


def _channel_summary():
    df = load('channel_summary.csv')
    if df is None:
        return None
    return df.assign(label=df['channel'].map(CHANNEL_LABELS).fillna(df['channel']))


def target_any_alarm(stream_length):
    c = next((m['calibration'] for m in meta().values() if 'calibration' in m), {'horizon': 1000, 'beta': 0.1})
    return 1 - (1 - c['beta']) ** (stream_length / c['horizon'])


def channels():
    """2.4 Every measure on no-drift and injected-switch streams."""
    df = _channel_summary()
    if df is None:
        return None
    for ds, d in df.groupby('dataset'):
        if d['inject_runs'].isna().all():
            continue
        L = d['control_stream_length'].median()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
        order = list(CHANNEL_LABELS.values())
        _dots(axes[0], d, 'control_any_alarm', 'label', order, 'no drift: share of streams with a false alarm',
              ref=target_any_alarm(L))
        _dots(axes[1], d, 'detection_rate', 'label', order, 'switch injected: share detected')
        _dots(axes[2], d, 'median_delay', 'label', order, 'switch injected: median delay (triples)')
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        fig.suptitle(f'{ds}: dot = median over models/compositions, line = range; dashed = calibration target',
                     x=0.01, ha='left', fontsize=10)
        fig.tight_layout()
        plt.show()
    cols = ['dataset', 'label', 'model', 'composition', 'control_any_alarm', 'control_alarms_per_1000',
            'detection_rate', 'median_delay', 'false_alarms_per_1000_before_switch', 'real_alarms',
            'real_false_alarms', 'real_delay', 'real_first_alarm_year']
    return df[[c for c in cols if c in df]].round(3)


def separation():
    """2.6 Threshold-free: P(statistic within D triples after the switch exceeds its max over a no-drift stream)."""
    s = load('channel_separation.csv')
    if s is None:
        return None
    s = s.assign(label=s['channel'].map(CHANNEL_LABELS).fillna(s['channel']))
    for ds, d in s.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(7, 3))
        _dots(ax, d, 'separation_500', 'label', list(CHANNEL_LABELS.values()),
              'separation within 500 triples (1 = perfect, 0.5 = no better than chance)', ref=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f'{ds}: threshold-free comparison', loc='left')
        fig.tight_layout()
        plt.show()
    return s.round(3)


def curves(model='TransE', composition='hadamard'):
    """2.6 False alarms vs detection over all thresholds (one line per measure)."""
    c = load('channel_curves.csv')
    if c is None:
        return None
    c = c[((c['model'] == '-') | (c['model'] == model)) & (c['composition'].isin(['-', composition]))]
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    for ds, d in c.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, ch in enumerate([k for k in CHANNEL_LABELS if k in set(d['channel'])]):
            s = d[d['channel'] == ch].sort_values('no_drift_false_alarm_share')
            ax.plot(s['no_drift_false_alarm_share'], s['detected_within_500'], color=colors[i], linewidth=1.5,
                    label=CHANNEL_LABELS[ch])
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=':')
        ax.set_xlabel('share of no-drift streams with a false alarm')
        ax.set_ylabel('share of switches detected within 500 triples')
        ax.set_title(f'{ds} ({model}, {composition} for embedding measures): up and left is better', loc='left')
        ax.legend(fontsize=7, loc='lower right')
        fig.tight_layout()
        plt.show()


def real_series(dataset, model='TransE', composition='hadamard'):
    """2.4 Each measure along the real stream: statistic, threshold, alarms, switch."""
    df = load('channels_real_series.csv.gz')
    if df is None:
        return
    d = df[(df['dataset'] == dataset) & (((df['model'] == '-') & (df['composition'] == '-'))
                                         | ((df['model'] == model) & (df['composition'].isin([composition, '-']))))]
    chans = [c for c in CHANNEL_LABELS if c in set(d['channel'])]
    if not chans:
        print(f'-- no real-stream series for {dataset}')
        return
    rows = (len(chans) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 1.9 * rows), sharex=True, squeeze=False)
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
    for ax in axes.ravel()[len(chans):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('position in the stream (triples)')
    fig.suptitle(f'{dataset}, real order: line = statistic, dashed = threshold, dots = alarms, grey = switch',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    plt.show()


def graph():
    """2.5 Year-to-year graph distances and the detected change point."""
    s, c = load('graph_series.csv'), load('graph_changepoints.csv')
    if s is None:
        return None
    measures = [m for m in ('disease_jaccard', 'wl_distance', 'deltacon') if m in s]
    titles = {'disease_jaccard': 'diseases studied (1 - Jaccard)', 'wl_distance': 'typed WL patterns (TV distance)',
              'deltacon': 'DeltaCon (cumulative graph)'}
    for ds, d in s.groupby('dataset'):
        fig, axes = plt.subplots(1, len(measures), figsize=(4.3 * len(measures), 2.6), squeeze=False)
        for ax, m in zip(axes[0], measures):
            ax.plot(d['year'], d[m], color='#2a78d6', marker='o', markersize=4)
            if SWITCH.get(ds):
                ax.axvline(SWITCH[ds] - 0.5, color=MUTED, linewidth=1)
            if CUTOFF.get(ds):
                ax.axvline(CUTOFF[ds] + 0.5, color=MUTED, linewidth=1, linestyle=':')
            if c is not None:
                cp = c[(c['dataset'] == ds) & (c['measure'] == m)]
                if len(cp) and cp['changepoint_year'].notna().all():
                    ax.axvline(cp['changepoint_year'].iloc[0] - 0.5, color='#eb6834', linewidth=1, linestyle='--')
            ax.set_title(titles[m], loc='left', fontsize=8)
            ax.set_xlabel('year')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        fig.suptitle(f'{ds}: dotted = cutoff (reference period | stream), grey = known switch, '
                     'orange dashed = detected change point', x=0.01, ha='left', fontsize=10)
        fig.tight_layout()
        plt.show()
    return c


def quality():
    """2.6 Embedding quality: MRR vs training length, and Phase 1's MRR at 10 epochs."""
    q = load('quality.csv')
    if q is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.2))
    curve = q[q.get('source', pd.Series(index=q.index, dtype=object)).ne('phase1')] if 'source' in q else q
    for m, s in curve.groupby('model'):
        s = s.sort_values('epochs')
        ax.plot(s['epochs'], s['mrr'], marker='o', color=MODEL_COLORS.get(m, INK), label=f'{m} (curve)')
    if 'source' in q:
        p1 = q[q['source'] == 'phase1']
        for m, s in p1.groupby('model'):
            ax.plot(s['epochs'], s['mrr'], 's', color=MODEL_COLORS.get(m, INK), markersize=6, markeredgecolor='white')
    if 'random_mrr_approx' in q:
        ax.axhline(q['random_mrr_approx'].dropna().median(), color=MUTED, linewidth=1, linestyle=':')
    ax.set_xlabel('training epochs')
    ax.set_ylabel('MRR (held-out triples)')
    ax.set_title('squares = Phase 1 models at 10 epochs; dotted = random guessing', loc='left', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plt.show()
    return q.round(4)
