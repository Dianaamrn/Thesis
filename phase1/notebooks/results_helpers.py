"""Tables and figures for the results notebook (phase1/notebooks/results.ipynb).

Everything here only reads the outputs of the runs and of analyze.py; nothing is trained.
Numbers shown are taken from those outputs; where runs are summarised over models and compositions,
the table says so (counts of runs, or median with min and max).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE1))
import configs  # noqa: E402

VERSION_LABEL = {
    'v1': 'scratch, MRR-chosen epochs',
    'v2_e2': 'scratch, 2 epochs',
    'v2_e10': 'scratch, 10 epochs',
    'v2_e20': 'scratch, 20 epochs',
    'v3': 'warm start',
}
VERSION_ORDER = ['v2_e2', 'v2_e10', 'v2_e20', 'v1', 'v3']
RULE_ORDER = ['sphere1', 'sphereR10', 'sphereR50', 'sphereR90', 'always', 'every250', 'every50', 'every10']
RULE_LABEL = {'sphere1': 'R1 sphere k=1', 'sphereR10': 'R3 sphere ~10%', 'sphereR50': 'R3 sphere ~50%',
              'sphereR90': 'R3 sphere ~90%', 'always': 'R2 always', 'every250': 'R4 every 250',
              'every50': 'R4 every 50', 'every10': 'R4 every 10'}
MODEL_COLORS = {'TransE': '#2a78d6', 'TransH': '#eb6834', 'DistMult': '#1baf7a', 'ComplEx': '#eda100',
                'RotatE': '#e87ba4'}


def paths(analysis='analysis'):
    a = PHASE1 / analysis
    return {'analysis': a, 'summary': a / 'summary_table.csv', 'peaks': a / 'peaks_all.csv',
            'offsets': a / 'alarm_offsets.csv', 'figures': a / 'figures', 'literature': a / 'literature',
            'quality': PHASE1 / 'quality', 'warm': PHASE1 / 'warm_check', 'settings': PHASE1 / 'settings_check',
            'checks': PHASE1 / 'checks' / 'checks.csv'}


def read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def load(analysis='analysis'):
    p = paths(analysis)
    t = read_csv(p['summary'])
    if not t.empty:
        t = t[t['setting'].eq('base')] if 'setting' in t else t
        t['rule'] = t['policy_tag']
        t['version_label'] = t['version'].map(VERSION_LABEL).fillna(t['version'])
    return t


# ---------------------------------------------------------------------------------------------
# What was run
# ---------------------------------------------------------------------------------------------
def expected_runs():
    rows = []
    per_rule = {'sphere': ['sphere1'], 'always': ['always'], 'sweep': list(configs.SWEEP_TARGETS),
                'every': [f'every{x}' for x in configs.EVERY_X]}
    for version, plan in configs.PLAN.items():
        for rule, datasets in plan.items():
            for d in datasets:
                for tag in per_rule[rule]:
                    rows.append({'dataset': d, 'version': version, 'rule': tag,
                                 'planned': len(configs.MODELS) * len(configs.COMPOSITIONS)})
    return pd.DataFrame(rows)


def run_overview(t):
    exp = expected_runs()
    done = t.groupby(['dataset', 'version', 'rule']).size().rename('finished').reset_index() if len(t) else \
        pd.DataFrame(columns=['dataset', 'version', 'rule', 'finished'])
    m = exp.merge(done, how='outer', on=['dataset', 'version', 'rule']).fillna({'finished': 0, 'planned': 0})
    m['cell'] = m['finished'].astype(int).astype(str) + ' / ' + m['planned'].astype(int).astype(str)
    out = m.pivot_table(index=['dataset', 'rule'], columns='version', values='cell', aggfunc='first').fillna('')
    out = out.reindex(columns=[v for v in VERSION_ORDER if v in out.columns])
    out = out.reindex(sorted(out.index, key=lambda x: (x[0], RULE_ORDER.index(x[1]) if x[1] in RULE_ORDER else 99)))
    return out


# ---------------------------------------------------------------------------------------------
# Embedding quality
# ---------------------------------------------------------------------------------------------
def quality_table():
    rows = []
    for f in sorted(paths()['quality'].glob('*.json')):
        r = json.loads(f.read_text())
        tag = f.stem.split('__')[-1]
        version = {'phase1': 'v1', 'v2e2': 'v2_e2', 'v2e10': 'v2_e10', 'v2e20': 'v2_e20'}.get(tag, tag)
        rows.append({'dataset': r['dataset'], 'model': r['model'], 'version': version, 'epochs': r['epochs'],
                     'learning_rate': r['learning_rate'], 'test MRR': r['mrr'], 'Hits@10': r['hits_at_10'],
                     'random MRR': r['random_mrr_approx']})
    q = pd.DataFrame(rows)
    if q.empty:
        return q
    return q.pivot_table(index=['dataset', 'model'], columns='version', values='test MRR').reindex(
        columns=[v for v in VERSION_ORDER if v in set(q['version'])]).round(4)


def warm_tables():
    out = {}
    for f in sorted(paths()['warm'].glob('*__chosen.csv')):
        out[f.name.split('__')[0]] = pd.read_csv(f)
    return out


# ---------------------------------------------------------------------------------------------
# Cases: alarms before / after the change
# ---------------------------------------------------------------------------------------------
def per_run_table(t, dataset, value):
    d = t[t['dataset'].eq(dataset)]
    if d.empty or value not in d:
        return pd.DataFrame()
    d = d.assign(run_col=d['model'] + ' ' + d['composition'].str[:4])
    out = d.pivot_table(index=['version', 'rule'], columns='run_col', values=value, aggfunc='first')
    order = [(v, r) for v in VERSION_ORDER for r in RULE_ORDER if (v, r) in out.index]
    return out.reindex(order)


def case_summary(t, dataset):
    """One row per version and rule; counts over the 10 runs (5 models x 2 compositions)."""
    d = t[t['dataset'].eq(dataset)]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for (v, r), g in d.groupby(['version', 'rule']):
        rows.append({
            'version': v, 'rule': RULE_LABEL.get(r, r), 'runs': len(g),
            'runs with 0 alarms before the change': int((g['alarms_pre'] == 0).sum()),
            'runs with >= 1 alarm after the change': int((g['alarms_post'] >= 1).sum()),
            'runs that tested the change triple': int(g['switch_tested'].astype(bool).sum()),
            'alarms before (median, min-max)': fmt_range(g['alarms_pre']),
            'alarms after (median, min-max)': fmt_range(g['alarms_post']),
            'delay in triples (median, min-max)': fmt_range(g['detection_delay']),
            'share of stream tested (median)': round(float(g['tested_share'].median()), 3) if 'tested_share' in g else np.nan,
            'OCSVM retrains (median)': float(np.nanmedian(g['num_threshold_retrains'].fillna(0) + g['num_scheduled_refits'].fillna(0))),
            'run time, hours (median)': round(float(g['runtime_min'].median()) / 60, 2),
        })
    out = pd.DataFrame(rows)
    out['_v'] = out['version'].map({v: i for i, v in enumerate(VERSION_ORDER)})
    out['_r'] = out['rule'].map({RULE_LABEL[r]: i for i, r in enumerate(RULE_ORDER)})
    out['version'] = out['version'].map(VERSION_LABEL).fillna(out['version'])
    return out.sort_values(['_v', '_r']).drop(columns=['_v', '_r']).reset_index(drop=True)


def fmt_range(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if s.empty:
        return 'none'
    return f'{s.median():g} ({s.min():g}-{s.max():g})'


def heatmap(t, dataset, value, title, ax=None, cmap='Reds'):
    import matplotlib.pyplot as plt
    m = per_run_table(t, dataset, value)
    if m.empty:
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * m.shape[1] + 3), 0.32 * m.shape[0] + 1.6))
    vals = m.to_numpy(dtype=float)
    ax.imshow(np.where(np.isnan(vals), np.nan, vals), aspect='auto', cmap=cmap)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f'{vals[i, j]:g}', ha='center', va='center', fontsize=7)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(m.shape[0]), [f'{VERSION_LABEL.get(v, v)} | {RULE_LABEL.get(r, r)}' for v, r in m.index], fontsize=8)
    ax.set_title(f'{dataset}: {title} (one cell = one run; empty = not run or no value)', fontsize=10, loc='left')
    plt.tight_layout()
    plt.show()


def alarm_offsets_grid(dataset, bin_width=100):
    """Where the alarms fall relative to the change: one panel per version (rows) and rule (columns),
    all 10 runs of a cell stacked (counts of alarms per bin of stream triples)."""
    import matplotlib.pyplot as plt
    o = read_csv(paths()['offsets'])
    if o.empty:
        print('no alarm offsets yet')
        return
    o = o[o['dataset'].eq(dataset) & o.get('setting', pd.Series('base', index=o.index)).fillna('base').eq('base')]
    versions = [v for v in VERSION_ORDER if v in set(o['version'])]
    rules = [r for r in RULE_ORDER if r in set(o['policy_tag'])]
    if not versions or not rules:
        print('no alarms')
        return
    lo = int(np.floor(o['offset_from_switch'].min() / bin_width) * bin_width)
    hi = int(np.ceil((o['offset_from_switch'].max() + 1) / bin_width) * bin_width)
    bins = np.arange(lo, hi + bin_width, bin_width)
    fig, axes = plt.subplots(len(versions), len(rules), figsize=(2.2 * len(rules), 1.6 * len(versions)),
                             sharex=True, sharey=True, squeeze=False)
    for i, v in enumerate(versions):
        for j, r in enumerate(rules):
            ax = axes[i][j]
            g = o[o['version'].eq(v) & o['policy_tag'].eq(r)]
            ax.hist(g['offset_from_switch'], bins=bins, color='#3b6ea5')
            ax.axvline(0, color='black', ls='--', lw=0.8)
            if i == 0:
                ax.set_title(RULE_LABEL.get(r, r), fontsize=8)
            if j == 0:
                ax.set_ylabel(VERSION_LABEL.get(v, v), fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f'{dataset}: alarms per {bin_width} stream triples, relative to the change (0 = first triple of the '
                 f'change year; left = before = false alarms). All 10 runs of a cell together.', fontsize=9)
    plt.tight_layout()
    plt.show()


def show_run_figures(dataset, model='TransE', composition='hadamard', rules=('sphere1', 'always'), versions=None):
    from IPython.display import Image, display
    fig_dir = paths()['figures']
    for v in versions or VERSION_ORDER:
        tag = configs.VERSIONS[v]['tag'] if v in configs.VERSIONS else v
        for r in rules:
            f = fig_dir / f'{dataset}__{model}__{composition}__{r}__{tag}.png'
            if f.exists():
                print(f'{VERSION_LABEL.get(v, v)} | {RULE_LABEL.get(r, r)}')
                display(Image(filename=str(f)))


# ---------------------------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------------------------
def peak_distance_table(t, dataset):
    cols = ['version', 'rule', 'model', 'composition', 'n_peaks', 'highest_peak_date', 'switch_to_peak',
            'switch_to_peak_days', 'second_peak_date', 'switch_to_second_peak', 'highest_to_second_triples',
            'highest_to_second_days']
    d = t[t['dataset'].eq(dataset)]
    d = d[[c for c in cols if c in d]].copy()
    d['_v'] = d['version'].map({v: i for i, v in enumerate(VERSION_ORDER)})
    d['_r'] = d['rule'].map({r: i for i, r in enumerate(RULE_ORDER)})
    return d.sort_values(['_v', '_r', 'model', 'composition']).drop(columns=['_v', '_r']).reset_index(drop=True)


def peak_dates(dataset, top=30):
    f = paths()['literature'] / f'{dataset}__peak_dates.csv'
    return read_csv(f).head(top)


# ---------------------------------------------------------------------------------------------
# Full data: alarms over time
# ---------------------------------------------------------------------------------------------
def full_alarms_by_year(t, dataset='full'):
    """Alarms per year of the alarm triple, one line per version and rule (all 10 runs of a cell together)."""
    import matplotlib.pyplot as plt
    rows = []
    for v in VERSION_ORDER:
        folder = PHASE1 / configs.VERSIONS[v]['runs_dir']
        if not folder.exists():
            continue
        for run in folder.glob(f'{dataset}__*'):
            f = run / 'drifts.tsv'
            if (run / 'summary.json').exists() and f.exists() and f.stat().st_size > 1:
                d = pd.read_csv(f, sep='\t')
                rule = run.name.split('__')[3]
                rows += [{'version': v, 'rule': rule, 'year': y} for y in d['event_year']]
    a = pd.DataFrame(rows)
    if a.empty:
        print('no alarms yet')
        return a
    counts = a.groupby(['version', 'rule', 'year']).size().rename('alarms').reset_index()
    rules = [r for r in RULE_ORDER if r in set(counts['rule'])]
    fig, axes = plt.subplots(1, len(rules), figsize=(3.2 * len(rules), 3), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], rules):
        for v in VERSION_ORDER:
            g = counts[counts['version'].eq(v) & counts['rule'].eq(r)]
            if len(g):
                ax.plot(g['year'], g['alarms'], marker='o', ms=3, lw=1, label=VERSION_LABEL[v])
        ax.set_title(RULE_LABEL.get(r, r), fontsize=9)
        ax.set_xlabel('year')
    axes[0][0].set_ylabel('alarms (10 runs together)')
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    return counts.pivot_table(index=['version', 'rule'], columns='year', values='alarms', fill_value=0)


# ---------------------------------------------------------------------------------------------
# Sensitivity and cost
# ---------------------------------------------------------------------------------------------
def sensitivity_table(analysis='analysis_sensitivity'):
    t = read_csv(paths(analysis)['summary'])
    if t.empty:
        return t
    base = t['version'].eq(configs.SENSITIVITY_VERSION) & t['policy_tag'].eq(configs.SENSITIVITY_RULE)
    t = t[base & t['dataset'].isin(configs.CASES)]
    rows = []
    for (d, s), g in t.groupby(['dataset', 'setting']):
        rows.append({'dataset': d, 'setting': s, 'runs': len(g),
                     'runs with 0 alarms before': int((g['alarms_pre'] == 0).sum()),
                     'runs with >= 1 alarm after': int((g['alarms_post'] >= 1).sum()),
                     'alarms before (median, min-max)': fmt_range(g['alarms_pre']),
                     'alarms after (median, min-max)': fmt_range(g['alarms_post']),
                     'delay (median, min-max)': fmt_range(g['detection_delay']),
                     'share tested (median)': round(float(g['tested_share'].median()), 3)})
    return pd.DataFrame(rows)


def cost_table(t):
    if t.empty:
        return t
    g = t.groupby(['dataset', 'version'])
    out = pd.DataFrame({'runs': g.size(), 'GPU hours (sum of run times)': (g['runtime_min'].sum() / 60).round(1),
                        'hours per run (median)': (g['runtime_min'].median() / 60).round(2),
                        're-embeds per run (median)': g['num_unseen_refresh_events'].median()})
    return out.reset_index()
