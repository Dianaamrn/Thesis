"""Writes checks/decision_checks.ipynb: one notebook with the evidence behind every pipeline decision.

Each decision has the same five parts: Decision, Observation, Literature, Check, Conclusion.
Small checks are computed in the notebook (data counts; two TransE trainings on CASE1 for the baseline
checks). Big checks (settings check, warm-start check, correctness checks, the runs, the sensitivity runs)
are run by their own scripts; the notebook only reads their saved outputs. A missing output is reported,
not an error. Nothing in phase1/ is changed.

    python checks/build_decision_checks.py
Run on the server (inside tmux), from the repository root:
    jupyter nbconvert --to notebook --execute checks/decision_checks.ipynb --output-dir checks/executed \
        --ExecutePreprocessor.timeout=-1
Environment variables (all optional):
    CHECKS_DATA_DIR    data folder              (default: <repo>/data)
    CHECKS_PHASE1_DIR  folder with the runs     (default: <repo>/phase1)
    CHECKS_FULL=1      also read the full-data step traces (slow; default: cases only for step traces)
    CHECKS_TEST=1      smoke test: random stand-in embedding, 10 repeats (no real training)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n')}


CELLS = [
    md("""
# Decision checks

Every decision in the pipeline, with the evidence behind it. Each section has the same five parts:

1. **Decision** — what is chosen
2. **Observation** — what in the runs or the data raised the question
3. **Literature** — what published work does (quotes checked against the PDFs in `literature/papers/`)
4. **Check** — a test whose result is in the output below the cell
5. **Conclusion**

Small checks are computed here. Big checks are run by their own scripts (listed in `checks/README.md`) and this
notebook reads their saved outputs; if an output is not available on this machine, the cell says so.

| Section | Decision |
|---|---|
| 1 | The code implements the method (correctness checks) |
| 2 | Data: sizes, undated triples, reverse pairs |
| 3 | Embedding: learning rate and epochs per model; warm-start updates |
| 4 | Re-embedding from scratch changes the coordinate system |
| 5 | Baseline outlier rate: in-sample vs out-of-sample |
| 6 | Alarms at the start of the stream |
| 7 | Sphere rule (Rule 1) |
| 8 | Direction of alarms (two-sided test) |
| 9 | Detector settings: sensitivity |
| 10 | Untested refill triples after an alarm |
| 11 | Checks still to be run |
"""),
    code("""
import json, os, pickle, sys, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
logging.getLogger('pykeen').setLevel(logging.ERROR)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)

ROOT = Path.cwd()
while not (ROOT / 'phase1' / 'drift_sphere.py').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
PHASE1 = Path(os.environ.get('CHECKS_PHASE1_DIR', ROOT / 'phase1'))
DATA = Path(os.environ.get('CHECKS_DATA_DIR', ROOT / 'data'))
CACHE = ROOT / 'checks' / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
TEST = os.environ.get('CHECKS_TEST') == '1'
READ_FULL_TRACES = os.environ.get('CHECKS_FULL') == '1'
sys.path.insert(0, str(PHASE1))
import configs
import drift_sphere as ds
ADAZOR = ROOT / 'adazor' / 'src'
P = configs.PRESETS['phase1']
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'
show = display if 'display' in dir() else (lambda x: print(x.to_string() if hasattr(x, 'to_string') else x))


def missing(what, how):
    print(f'NOT AVAILABLE HERE: {what}.\\n  How to get it: {how}')


def run_dirs():
    \"\"\"Every finished run folder under phase1/runs*/ (name: dataset__model__composition__rule__version[__setting]).\"\"\"
    out = []
    for folder in sorted(PHASE1.glob('runs*')):
        for d in sorted(folder.glob('*__*__*__*__*')):
            if (d / 'summary.json').exists():
                p = d.name.split('__')
                out.append({'folder': folder.name, 'run': d.name, 'path': d, 'dataset': p[0], 'model': p[1],
                            'composition': p[2], 'rule': p[3], 'version': p[4],
                            'setting': p[5] if len(p) > 5 else 'base'})
    return pd.DataFrame(out)


def read_tsv(path):
    try:
        return pd.read_csv(path, sep='\\t')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


RUNS = run_dirs()
MAIN_FOLDERS = [v['runs_dir'] for v in configs.VERSIONS.values()]
MAIN = RUNS[RUNS.folder.isin(MAIN_FOLDERS) & RUNS.setting.eq('base')] if len(RUNS) else RUNS
print('device:', DEVICE, '| test mode:', TEST)
print('finished runs found:', len(RUNS), '| folders:', dict(RUNS.folder.value_counts()) if len(RUNS) else {})
"""),

    # ------------------------------------------------------------------ 1
    md("""
## 1. The code implements the method

**Decision.** Use `phase1/drift_sphere.py` (a port of Antoine's notebook) for every run.

**Observation.** The code was rewritten from a notebook into a script and extended (5 models, versions, rules); every
step must still do what the method says.

**Literature.** The method: ADAZOR (Zanaboni, Malchiodi, Nasra 2026), Algorithms 1 and 2; Antoine's notebook (sphere rule).

**Check.** Big check `phase1/checks.py` (48 checks: alarm threshold, data counts, stream loop, re-embedding trigger,
sphere geometry, compositions, warm start, Antoine's own alarms on CPU). Its output `phase1/checks/checks.csv` is shown here.
"""),
    code("""
f = PHASE1 / 'checks' / 'checks.csv'
if f.exists():
    c = pd.read_csv(f)
    status = c['passed'].map({True: 'pass', False: 'FAIL'}) if 'passed' in c else None
    if 'info' in c:
        status = np.where(c['info'].fillna(False).astype(bool), 'finding', status)
    c['status'] = status
    print(c['status'].value_counts().to_string())
    show(c)
else:
    missing('phase1/checks/checks.csv', 'on the server: cd ~/Thesis/phase1 && python checks.py (or bash run_all.sh checks)')
"""),
    md("""
**Conclusion.** The pipeline is accepted only if no check fails. Check 7 reproduces Antoine's own alarms on a CPU;
the GPU difference for Hadamard is reported as a finding (hardware non-determinism; also reported by Méroué 2026).
"""),

    # ------------------------------------------------------------------ 2
    md("""
## 2. Data

**Decision.** Starting graph = every triple dated up to the cutoff plus every undated triple, all types; stream = one
monitored triple type after the cutoff, in time order. Undated triples and reverse pairs are kept.

**Observation.** Most of the starting graph has no date; many facts are stored in both directions.

**Literature.**
- TGB 2.0 (Gastinger et al. 2024, p. 4) provides untimed *static relations* next to the dated facts, "that can be used to enhance model performance".
- Reverse pairs are a known cause of inflated link-prediction scores (*test leakage*): in FB15k about 81% of the test triples are inverses of training triples (Ali et al. 2020; Rossi et al. 2021).
- For training, reverse triples are not harmful: Lacroix et al. 2018 add an inverse triple for every training triple on purpose.

**Check.** Sizes, undated share, and reverse pairs, computed from the data files.
"""),
    code("""
rows, GRAPHS = [], {}
for name, dcfg in configs.DATASETS.items():
    f = DATA / dcfg['file']
    if not f.exists():
        missing(f'data file {f.name}', 'copy it into the data folder'); continue
    df = ds.load_temporal_triples(f)
    kg, init, fut, _ = ds.build_initial_knowledge_graph(df, dcfg['cutoff_year'], dcfg['target_type'])
    GRAPHS[name] = (kg, init, fut)
    n_win = min(P['window_size'], len(init) - 1)
    und = kg['event_year'].isna()
    rows.append({'dataset': name, 'cutoff': dcfg['cutoff_year'], 'file rows': len(df), 'starting graph': len(kg),
                 'undated in graph': int(und.sum()), 'undated share': round(und.mean(), 3),
                 'types in graph': kg['type'].nunique(), 'monitored up to cutoff': len(init),
                 'reference': len(init) - n_win, 'first window': n_win, 'stream': len(fut)})
show(pd.DataFrame(rows))
"""),
    code("""
rows = []
for name, (kg, init, fut) in GRAPHS.items():
    und = kg[kg['event_year'].isna()]; dat = kg[kg['event_year'].notna()]
    nodes_und = set(und.subject) | set(und.object); nodes_dat = set(dat.subject) | set(dat.object)
    stream_nodes = set(fut.subject) | set(fut.object)
    rows.append({'dataset': name, 'stream entities': len(stream_nodes),
                 'already in starting graph': len(stream_nodes & (nodes_und | nodes_dat)),
                 'known only through undated triples': len(stream_nodes & nodes_und - nodes_dat),
                 'new (cause a re-embed)': len(stream_nodes - nodes_und - nodes_dat)})
print('Stream entities and how the starting graph knows them:')
show(pd.DataFrame(rows))
"""),
    code("""
rows = []
for name, (kg, init, fut) in GRAPHS.items():
    s, o = kg['subject'].astype(str).to_numpy(), kg['object'].astype(str).to_numpy()
    back = set(zip(o, s))
    has_rev = np.fromiter(((a, b) in back and a != b for a, b in zip(s, o)), dtype=bool, count=len(kg))
    monitored = kg['type'].eq(configs.DATASETS[name]['target_type']).to_numpy()
    rows.append({'dataset': name, 'graph rows': len(kg), 'rows with a reverse pair': int(has_rev.sum()),
                 'share': round(has_rev.mean(), 3), 'of which monitored type': int((has_rev & monitored).sum())})
    if name == 'full':
        top = kg.loc[has_rev, 'type'].value_counts().head(8)
show(pd.DataFrame(rows))
if 'top' in dir():
    print('\\nFull data, types with the most reverse pairs:'); print(top.to_string())
"""),
    md("""
**Conclusion.** Undated triples are background knowledge (ontology and database facts) and stay in the starting graph,
as in Antoine's design and TGB 2.0; their possible later origin (temporal leakage) is reported as a limitation, with the
number of stream entities known only through them. Reverse pairs do not touch the monitored type; they stay (harmless
for training, Lacroix 2018) but make the reported link-prediction MRR optimistic.
"""),

    # ------------------------------------------------------------------ 3
    md("""
## 3. Embedding: learning rate, epochs, warm-start updates

**Decision.** Each model is trained with the learning rate (0.001, 0.01, 0.05) and number of epochs (up to 100) that give
the best validation MRR on the cutoff graph (early stopping). Fixed-epoch versions (2, 10, 20) are run as alternatives.
Warm-start updates use a few epochs (1–10) at the first learning rate or one fifth of it.

**Observation.** With 0.001 for every model, ComplEx did not learn (MRR at the level of random guessing).

**Literature.**
- Early stopping on validation MRR is standard: Ruffinelli et al. 2020 (max 400 epochs, MRR every 5 epochs, patience 50); Ali et al. 2020 ("often used for early stopping since it is a smooth measure").
- Tuned biomedical benchmarks find a different best learning rate per model (Bonner et al. 2021, Table 4).
- MRR does not measure stability: "performance and stability are not aligned criteria" (Méroué et al. 2026).
- Warm updates: about one fifth of the first learning rate (Wewer et al. 2021); fine-tuning on new facts "would quickly disrupt" old knowledge (Cui et al. 2022); multiplicative models forget most (Pons et al. 2026).

**Check.** Big checks `phase1/settings_check.py` (MRR per model and learning rate) and `phase1/warm_check.py`
(warm updates of 1–10 epochs); their saved outputs are shown here.
"""),
    code("""
rows = []
for f in sorted((PHASE1 / 'settings_check').glob('*__chosen.csv')):
    t = pd.read_csv(f)
    for _, r in t.iterrows():
        rec = {'dataset': r['dataset'], 'model': r['model'], 'chosen lr': r['learning_rate'], 'chosen epochs': r['epochs'],
               'validation MRR': round(r['mrr'], 4), 'x random': round(r['times_random']), 'still improving at 100': r['epochs'] == 100}
        for part in str(r['mrr_by_learning_rate']).split(';'):
            lr, rest = part.split(':', 1); rec[f'MRR @ lr {lr.strip()}'] = rest.strip()
        rows.append(rec)
if rows:
    show(pd.DataFrame(rows))
else:
    missing('phase1/settings_check/*__chosen.csv', 'on the server: python settings_check.py --dataset <d>')
"""),
    code("""
found = False
for name in configs.DATASETS:
    f = PHASE1 / 'warm_check' / f'{name}__table.csv'
    if f.exists():
        found = True
        print(f'== {name}: warm update after one window of stream triples'); show(pd.read_csv(f))
if not found:
    missing('phase1/warm_check/<dataset>__table.csv', 'on the server: python warm_check.py --dataset <d> (or bash run_all.sh warm), then copy warm_check/')
"""),
    md("""
**Conclusion.** The learning rate is chosen per model by validation MRR (ComplEx needs 0.01; the others 0.001); several
models were still improving at 100 epochs, so the first embedding is a budget-limited one (limitation). Warm updates are
judged on two things the drift test needs: the old graph's MRR is kept, and the old OCSVM still recognises the reference.
"""),

    # ------------------------------------------------------------------ 4 and 5: two trainings
    md("""
## 4. Re-embedding from scratch changes the coordinate system

**Decision.** After a from-scratch re-embed the OCSVM must be refitted (Rule 2), or the re-embed must keep the old
coordinates (warm start).

**Observation.** With the rules that do not refit at every re-embed, most alarms happen exactly at a re-embed.

**Literature.** Independently trained embeddings differ by more than a rotation; alignment methods (orthogonal
Procrustes, backward-compatible embeddings, Hu et al. 2022; Gursoy et al. 2021) exist because of it; the random seed
alone changes KG embeddings (Méroué et al. 2026).

**Check.** Train TransE twice on the same CASE1 graph (seeds 42 and 43), nothing else changed. Fit the OCSVM on the
reference under the first embedding, then score the same triples under the second.
"""),
    code("""
dcfg = configs.DATASETS['case1']
if 'case1' in GRAPHS:
    kg1, init1, _ = GRAPHS['case1']
else:
    df1 = ds.load_temporal_triples(DATA / dcfg['file'])
    kg1, init1, _, _ = ds.build_initial_knowledge_graph(df1, dcfg['cutoff_year'], dcfg['target_type'])
KG = ds.KnowledgeGraph(kg1)
INIT = init1.to_dict('records')
N_WIN = min(P['window_size'], len(INIT) - 1)
REF_ROWS, WIN_ROWS = INIT[:-N_WIN], INIT[-N_WIN:]
TR = configs.TRAINING[('case1', 'TransE')]


def embed(seed):
    cfg = {'embedder': 'random' if TEST else 'pykeen', 'model': 'TransE', 'real_dim': P['real_dim'], 'epochs': TR['epochs'],
           'learning_rate': TR['learning_rate'], 'batch_size': P['batch_size'], 'seed': seed, 'device': DEVICE}
    f = CACHE / f"case1_TransE_seed{seed}_e{TR['epochs']}{'_TEST' if TEST else ''}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.time(); st = ds.generate_embeddings_from_kg(KG, cfg)
    print(f'trained seed {seed} in {(time.time() - t0) / 60:.1f} min'); f.write_bytes(pickle.dumps(st)); return st


E1, E2 = embed(42), embed(43)
shared = [e for e in E1.entity_index if e in E2.entity_index]
cos = np.sum(E1.entity_matrix[[E1.entity_index[e] for e in shared]] * E2.entity_matrix[[E2.entity_index[e] for e in shared]], axis=1)
print(f'reference {len(REF_ROWS)} | first window {len(WIN_ROWS)} | same entity in the two trainings: mean cosine {cos.mean():.3f}')


def detector(flags):
    det, _ = ds.make_detector('ztest', P['alpha'], ADAZOR); det.reset(flags); return det


rows = []
for comp in configs.COMPOSITIONS:
    feats = lambda r, E: ds.rows_to_features(r, E, comp)
    oc, sc, own = ds.train_classifier(feats(REF_ROWS, E1), P)
    rate = lambda X: round(float(ds.outlier_flags(oc, sc, X).mean()), 3)
    for label, X in [('reference, first training (its own training data)', feats(REF_ROWS, E1)),
                     ('first window, first training (same coordinates)', feats(WIN_ROWS, E1)),
                     ('reference, second training (same triples, new coordinates)', feats(REF_ROWS, E2)),
                     ('first window, second training', feats(WIN_ROWS, E2))]:
        rows.append({'composition': comp, 'triples scored by the OCSVM of the first training': label, 'outlier share': rate(X)})
show(pd.DataFrame(rows))
"""),
    md("""
**Conclusion.** Re-embedding the same graph from scratch gives a new coordinate system: an OCSVM fitted before it flags
most of the same triples as outliers. Hence Rule 2 (refit at every re-embed) and the warm start (v3).

## 5. Baseline outlier rate: in-sample vs out-of-sample

**Decision.** Phase 1 follows ADAZOR (baseline rate measured on the OCSVM's training data). Proposed extension: measure
it on held-out data (calibration split).

**Observation.** Every CASE1 run alarms at the first stream triple (section 6); in section 4 the OCSVM's own training
triples have a much lower outlier share than unseen triples of the same period, with the same coordinates.

**Literature.**
- ADAZOR, Algorithm 1: `outlier_detector ← fit_detector(window.x)`; `baseline ← outlier_detector(window.l)` (same data). Nasra's thesis, Algorithm 1: `nodrift_sample ← detector.predict(Xtrain)`.
- ν bounds the outliers among the **training** points only: "an upper bound on the fraction of outliers, that is, training points outside the estimated region" (Schölkopf et al. 2001).
- Held-out estimation is the standard: Bates et al. 2023 evaluate scores "on the remaining n hold-out samples" (calibration set); Vovk et al. 2021 use three folds, each calibrated by a model trained on the other two; Lopez-Paz & Oquab 2016 "split it into the disjoint training and testing subsets"; Rabanser et al. 2018 measure their classifier "on held-out" data.
- The Z-test assumes both rates estimate the same quantity: "two samples drawn from Bernoulli populations … have the same mean" (ADAZOR, p. 7).

**Check.** No-drift test: the monitored CASE1 triples before the cutoff, one embedding, no re-embedding. Shuffle them,
split into reference and window, run the Z-test with (a) the in-sample baseline and (b) an out-of-sample baseline
(OCSVM fitted on half of the reference, rate measured on the other half). Repeated 100 times; every alarm is false.
"""),
    code("""
N_REP = 10 if TEST else 100
rows = []
for comp in configs.COMPOSITIONS:
    X_all = ds.rows_to_features(INIT, E1, comp); n_ref = len(REF_ROWS); half = n_ref // 2
    for rep in range(N_REP):
        perm = np.random.default_rng(rep).permutation(len(X_all))
        Xr, Xw = X_all[perm[:n_ref]], X_all[perm[n_ref:]]
        oc, sc, own = ds.train_classifier(Xr, P)
        win = ds.outlier_flags(oc, sc, Xw)
        oc2, sc2, _ = ds.train_classifier(Xr[:half], P)
        cal = ds.outlier_flags(oc2, sc2, Xr[half:])
        rows.append({'composition': comp, 'in-sample baseline rate': own.mean(), 'window rate': win.mean(),
                     'alarm, in-sample baseline': bool(detector(own).detect(win)),
                     'out-of-sample baseline rate': cal.mean(),
                     'alarm, out-of-sample baseline': bool(detector(cal).detect(ds.outlier_flags(oc2, sc2, Xw)))})
B = pd.DataFrame(rows)
show(B.groupby('composition').agg(**{
    'repeats': ('window rate', 'size'),
    'in-sample baseline rate': ('in-sample baseline rate', 'mean'),
    'window rate (unseen)': ('window rate', 'mean'),
    'false alarms, in-sample (%)': ('alarm, in-sample baseline', lambda x: 100 * x.mean()),
    'out-of-sample baseline rate': ('out-of-sample baseline rate', 'mean'),
    'false alarms, out-of-sample (%)': ('alarm, out-of-sample baseline', lambda x: 100 * x.mean())}).round(3))
"""),
    md("""
**Conclusion.** With no drift, the in-sample baseline produces false alarms in most repeats; the out-of-sample baseline
brings them close to the test's nominal level. Phase 1 keeps ADAZOR as published (the method under test); the
out-of-sample baseline is proposed as an extension, to be run next to the original.
"""),

    # ------------------------------------------------------------------ 6
    md("""
## 6. Alarms at the start of the stream

**Decision.** Report the first alarm separately; it is explained by sections 5 and the first-window design.

**Observation.** In Antoine's KG version the first test window (the newest triples before the cutoff) is separate from
the reference and has never been seen by the OCSVM.

**Literature.** In ADAZOR (Algorithm 1, line 1) and Nasra's thesis, the first window **is** the training window, so
testing starts from data the OCSVM has seen.

**Check.** From the finished runs: position of the first alarm.
"""),
    code("""
if len(MAIN):
    rows = []
    for _, r in MAIN.iterrows():
        s = json.loads((r.path / 'summary.json').read_text())
        rows.append({'dataset': r.dataset, 'version': r.version, 'rule': r.rule,
                     'first alarm at stream triple': s.get('first_drift_stream_index')})
    t = pd.DataFrame(rows)
    show(t.groupby(['dataset', 'version']).agg(runs=('rule', 'size'),
         first_alarm_at_triple_1=('first alarm at stream triple', lambda x: int((x == 1).sum())),
         median_first_alarm=('first alarm at stream triple', 'median'),
         max_first_alarm=('first alarm at stream triple', 'max')))
else:
    missing('run folders', 'copy the run folders (results_phase1.tgz) into phase1/')
"""),
    md("""
**Conclusion.** The first alarm marks the difference between the OCSVM's own training data and the first unseen window,
not a change in the stream; it is reported apart from the alarms that follow.

## 7. Sphere rule (Rule 1)

**Decision.** Keep Antoine's sphere rule as the baseline rule; Rules 2–4 are the alternatives.

**Observation.** The rule refits the OCSVM only if a new triple lies outside the sphere around the box of **all** graph
triples (all types).

**Literature.** No published rule uses a bounding box over the whole graph; the closest are cost- or threshold-based
retraining triggers (Mahadevan & Mathioudakis 2023).

**Check.** The logged sphere ratios of every re-embed (Rule 1 refits only if the ratio is above 1).
"""),
    code("""
rows = []
for _, r in MAIN.iterrows():
    sh = read_tsv(r.path / 'embedding_shift_metrics.tsv')
    if len(sh) and 'center_radius_ratio' in sh:
        v = sh.loc[sh['trigger_reason'].eq('unseen_element'), 'center_radius_ratio'].dropna()
        rows.append({'dataset': r.dataset, 'version': r.version, 'ratios': len(v), 'largest': v.max(), 'above 1': int((v > 1).sum())})
if rows:
    t = pd.DataFrame(rows)
    show(t.groupby(['dataset', 'version']).agg(runs=('ratios', 'size'), ratios=('ratios', 'sum'),
                                               largest_ratio=('largest', 'max'), ratios_above_1=('above 1', 'sum')).round(3))
else:
    missing('embedding_shift_metrics.tsv in the run folders', 'copy the run folders into phase1/')
"""),
    md("""
**Conclusion.** The ratio stays far below 1 for from-scratch and warm-start versions alike, because the box spans the
whole graph; Rule 1 practically never refits the OCSVM. Rule 3 (smaller spheres) tests whether a geometric trigger
works at all.

## 8. Direction of alarms

**Decision.** Keep ADAZOR's two-sided test; report the two directions separately.

**Literature.** ADAZOR (p. 7): a drift is when the window's outlier rate differs "either higher or lower" from the baseline.

**Check.** For every alarm in the finished runs: was the window's outlier share above or below the baseline?
"""),
    code("""
rows = []
for _, r in MAIN.iterrows():
    d = read_tsv(r.path / 'drifts.tsv')
    if len(d) and 'current_outlier_fraction' in d:
        up = d['current_outlier_fraction'] > d['reference_outlier_fraction']
        rows.append({'dataset': r.dataset, 'version': r.version, 'alarms': len(d), 'more outliers': int(up.sum()), 'fewer outliers': int((~up).sum())})
if rows:
    show(pd.DataFrame(rows).groupby(['dataset', 'version']).sum(numeric_only=True))
else:
    missing('drifts.tsv in the run folders', 'copy the run folders into phase1/')
"""),

    # ------------------------------------------------------------------ 9
    md("""
## 9. Detector settings: sensitivity

**Decision.** Keep ADAZOR's settings (ν 0.1, window 250, α 0.025, keep 10%) and seed 42. They are not tuned; the check
is whether conclusions change when one of them changes.

**Literature.** ADAZOR and Nasra's thesis: window 250, α 0.025, keep 10%, ν 0.1; Nasra's thesis tried windows 200–500 on
tabular data and the number of drifts changed with them. ν is an upper bound on training outliers (Schölkopf et al. 2001).
Seeds alone change KG embeddings (Méroué et al. 2026).

**Check.** Big check: the sensitivity runs (`run_grid.py --grid sensitivity`: cases, version v2_e20, Rule 2, one
setting changed at a time). Alarms per run, per setting, next to the base setting.
"""),
    code("""
sens = RUNS[RUNS.folder.eq(configs.SENSITIVITY_RUNS_DIR)] if len(RUNS) else pd.DataFrame()
base_tag = configs.VERSIONS[configs.SENSITIVITY_VERSION]['tag']
base = RUNS[RUNS.version.eq(base_tag) & RUNS.rule.eq(configs.SENSITIVITY_RULE) & RUNS.setting.eq('base') &
            RUNS.dataset.isin(configs.CASES)] if len(RUNS) else pd.DataFrame()
if len(sens):
    rows = []
    for _, r in pd.concat([base, sens]).iterrows():
        s = json.loads((r.path / 'summary.json').read_text())
        rows.append({'dataset': r.dataset, 'setting': r.setting, 'alarms': s['num_detected_drifts'],
                     'tested triples': s['tested_steps'], 'stream': s['future_rows']})
    t = pd.DataFrame(rows)
    show(t.groupby(['dataset', 'setting']).agg(runs=('alarms', 'size'), mean_alarms=('alarms', 'mean'),
                                               mean_tested=('tested triples', 'mean')).round(1))
else:
    missing('phase1/runs_sensitivity', 'running on the server (run_grid.py --grid sensitivity); copy the folder when done')
"""),

    # ------------------------------------------------------------------ 10
    md("""
## 10. Untested refill triples after an alarm

**Decision.** Keep ADAZOR's procedure (keep 10% of the window, refill with untested triples); measure what is missed.

**Observation.** After every alarm 225 stream triples enter the new reference without being tested.

**Literature.** ADAZOR (Algorithm 1, lines 10–15) and Nasra's thesis use this procedure; the period without testing is
not discussed in the sources.

**Check.** From the runs: share of the stream that was tested; and, from the shadow logging (the OCSVM in use at the
alarm scores the skipped triples, logging only), how often the detector would have alarmed inside a skipped stretch.
Step traces of the full data are large; they are read only with `CHECKS_FULL=1`.
"""),
    code("""
rows = []
for _, r in MAIN.iterrows():
    s = json.loads((r.path / 'summary.json').read_text())
    rec = {'dataset': r.dataset, 'version': r.version, 'runs': 1, 'stream triples': s['future_rows'], 'tested': s['tested_steps']}
    if r.dataset != 'full' or READ_FULL_TRACES:
        t = read_tsv(r.path / 'step_trace.tsv.gz')
        if 'shadow_would_alarm' in t:
            refill = t[t['refill'].astype(bool)]
            rec.update({'refill triples': len(refill), 'scorable by the old embedding': int(refill['shadow_scorable'].astype(bool).sum()),
                        'would have alarmed': int(refill['shadow_would_alarm'].astype(bool).sum())})
    rows.append(rec)
if rows:
    t = pd.DataFrame(rows).groupby(['dataset', 'version']).sum(numeric_only=True)
    t['tested share'] = (t['tested'] / t['stream triples']).round(3)
    show(t)
    print('(refill columns appear only for runs made after the shadow logging was added; full data only with CHECKS_FULL=1)')
else:
    missing('run folders', 'copy the run folders into phase1/')
"""),

    # ------------------------------------------------------------------ 11
    md("""
## 11. Checks still to be run

These need either new code or new runs; they are listed so the notebook shows the complete plan.

| Check | Question | Needs | How |
|---|---|---|---|
| No-drift control on the full data | How many alarms does the method give on real data when nothing changes (false-alarm level, cf. ARL₀)? | a `--shuffle-stream` option | full stream with its time order shuffled; a few runs (Rabanser et al. 2018 include "the no-shift case to check against false positives") |
| Out-of-sample baseline in the stream | Does the held-out baseline change the alarms of real runs? | a code option | cases, next to the original runs |
| First window = training window | Does the ADAZOR-style start remove the first alarm? | a code option | cases |
| Embedding dimension | Do conclusions change with 128 instead of 64? | no new code | `drift_sphere.py --real-dim 128` on a few case runs (benchmarks search 64–512: Ali 2020, Ruffinelli 2020; biomedical best ≈ 200–500, Bonner 2021) |
| Sensitivity on the full data | Do the detector settings matter on the full data? | no new code | `run_grid.py --grid sensitivity --datasets full` |
"""),
]


def main():
    nb = {'cells': CELLS,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = HERE / 'decision_checks.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
