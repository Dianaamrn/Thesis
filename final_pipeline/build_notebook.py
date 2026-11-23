"""Writes final_pipeline/final_pipeline.ipynb: the whole final pipeline in one self-contained notebook,
in the style of Antoine Nasra's notebook (all code in the cells, a text cell before every code cell).

    python final_pipeline/build_notebook.py

Run on the server, inside tmux, once per dataset (outputs are saved under every cell):
    cd ~/Thesis && source ~/drift/.venv/bin/activate
    NB_DATASET=case1 jupyter nbconvert --to notebook --execute final_pipeline/final_pipeline.ipynb \
        --output final_pipeline_case1 --output-dir final_pipeline/executed --ExecutePreprocessor.timeout=-1
(the same with NB_DATASET=case2 and NB_DATASET=full).

Environment variables:
    NB_DATASET    case1 | case2 | full                       (default case1)
    NB_DATA_DIR   folder with the data files                 (default <repo>/data)
    NB_OUT        folder for everything the notebook saves   (default final_pipeline/output/<dataset>)
    NB_HELDOUT    1 = also run the held-out-baseline comparison (default: 1 on the cases, 0 on full)
    NB_REUSE_SETTINGS 1 = Part 3 loads phase1/settings_check results when they exist (default 1; same rule)
    NB_PART       i/n = this execution runs only its share (every n-th) of the 40 runs and stops after Part 5

Several runs at a time (e.g. 3): start three executions with NB_PART=1/3, 2/3, 3/3 (each in its own tmux window and
with its own --output name). Part 1/3 also does Part 4 and the first embeddings; 2/3 and 3/3 wait for it. When all three
have finished, run the notebook once more without NB_PART: every run is then loaded, and Parts 6-9 analyse them.
    NB_TEST       1 = smoke test: tiny trainings, random stand-in embedding in the stream, short stream
Every expensive step saves its result and is skipped when the result exists, so a stopped run can be restarted.
"""
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = []
WRAP = False   # True for the cells after Part 5: skipped when the notebook only runs its share of the runs (NB_PART)


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')})


def code(text):
    text = text.strip('\n')
    if WRAP:
        text = 'if not RUNS_ONLY:   # skipped in the executions that only run their share of the runs (NB_PART)\n' + textwrap.indent(text, '    ')
    CELLS.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text})


# =====================================================================================================
md(r"""
# Concept drift detection on a biomedical knowledge-graph stream — final pipeline

A stream of biomedical triples arrives in time order. The knowledge graph known so far is embedded with a
knowledge-graph embedding model; every monitored triple becomes a feature vector; a one-class SVM fitted on a reference
window flags outliers; the ADAZOR Z-test (Zanaboni, Malchiodi, Nasra 2026) compares the outlier share of a sliding window
with the baseline outlier rate of the reference. This notebook extends Antoine Nasra's notebook with five embedding
models, two compositions, four retraining rules and a warm-start re-embedding.

**Final configuration**

| Part | Choice |
|---|---|
| Re-embedding | warm start: the first embedding with the learning rate and epochs chosen by validation MRR (Part 2); every later re-embed continues from the previous vectors for a few epochs (Part 3) |
| Retraining rule (when the OCSVM is refitted) | R1 sphere (Antoine, k = 1) · R2 always · R3 smaller sphere (refits at about 50% of re-embeds) · R4 every 250 triples |
| Embedding model | TransE, TransH, DistMult, ComplEx, RotatE (64 real numbers per vector) |
| Composition | Hadamard (s ⊙ p ⊙ o), concatenation ([s, p, o]) |
| Detector | StandardScaler + OCSVM (RBF, ν = 0.1, γ = 'scale'); window 250; α = 0.025; after an alarm keep the newest 10% of the window |

5 models × 2 compositions × 4 rules = **40 runs** per dataset.

**Contents**

| Part | What |
|---|---|
| 0 | Settings |
| 1 | The pipeline, function by function |
| 2 | Data |
| 3 | First embedding: learning rate and epochs (validation MRR) |
| 4 | Every later re-embed: how many epochs (catastrophic forgetting over many updates) |
| 5 | The 40 runs |
| 6 | Results: how many alarms and where |
| 7 | Analysis: alarm density and peaks, with the triples behind them |
| 8 | Comparison: baseline outlier rate measured in-sample (ADAZOR) vs on held-out data (cases) |
| 9 | Summary |

The checks of the method itself (where the baseline outlier rate is measured, re-embedding and coordinate systems, data
facts) are in a separate notebook: `checks/decision_checks.ipynb`.

The notebook is run once per dataset (`NB_DATASET=case1 | case2 | full`). Every expensive step saves its result in
`final_pipeline/output/<dataset>/` and is skipped when the result exists.
""")

# =====================================================================================================
md(r"""
# Part 0 — Settings

The next cell imports the libraries, finds the repository folders and reads the run options (dataset, test mode).
""")
code(r"""
import os, sys, json, time, math, pickle, warnings, logging, inspect
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')
logging.getLogger('pykeen').setLevel(logging.ERROR)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)

ROOT = Path.cwd()
while not (ROOT / 'final_pipeline').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATASET = os.environ.get('NB_DATASET', 'case1')
TEST = os.environ.get('NB_TEST') == '1'
DATA_DIR = Path(os.environ.get('NB_DATA_DIR', ROOT / 'data'))
OUT = Path(os.environ.get('NB_OUT', ROOT / 'final_pipeline' / 'output' / (DATASET + ('_TEST' if TEST else ''))))
OUT.mkdir(parents=True, exist_ok=True)
ADAZOR_SRC = ROOT / 'adazor' / 'src'
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'
show = display if 'display' in dir() else (lambda x: print(x.to_string() if hasattr(x, 'to_string') else x))
print(f'dataset {DATASET} | test mode {TEST} | device {DEVICE}\ndata {DATA_DIR}\noutput {OUT}')
""")

md(r"""
The next cell sets every parameter of the pipeline in one place: the datasets, the models and compositions, the detector
settings, the four retraining rules, and the settings of the two embedding checks (Parts 3 and 4).
""")
code(r"""
GENE_DISEASE = 'Gene - causes_or_contributes_to_condition - Disease'
MIRNA_DISEASE = 'miRNA - causes_or_contributes_to_condition - Disease'
DATASETS = {
    'case1': {'file': 'filtered_drifting_edgesCASE1.tsv', 'cutoff': 2002, 'type': GENE_DISEASE, 'switch_year': 2005},
    'case2': {'file': 'filtered_drifting_edgesCASE2.tsv', 'cutoff': 2018, 'type': MIRNA_DISEASE, 'switch_year': 2022},
    'full':  {'file': 'drifting_edges_timestamp.tsv',     'cutoff': 2010, 'type': GENE_DISEASE, 'switch_year': None},
}
D = DATASETS[DATASET]

MODELS = ['TransE', 'TransH', 'DistMult', 'ComplEx', 'RotatE']
COMPLEX_MODELS = {'ComplEx', 'RotatE'}          # 32 complex numbers = 64 real numbers
COMPOSITIONS = ['hadamard', 'concatenation']

S = {                      # detector and embedding settings (Antoine / ADAZOR values; Phase 1 adds the scaler)
    'real_dim': 64, 'batch_size': 2048, 'seed': 42,
    'window': 250, 'alpha': 0.025, 'keep': 0.10,
    'nu': 0.1, 'gamma': 'scale', 'standardize': True,
}
# The four retraining rules (TO CONFIRM: the size of R3 and the X of R4).
RULES = {
    'R2_always':     {'kind': 'always'},
    'R1_sphere':     {'kind': 'sphere', 'k': 1.0},
    'R4_every250':   {'kind': 'every', 'x': 250},
    'R3_sphere50':   {'kind': 'sphere', 'share': 0.50},   # k chosen from R2's ratios so that it refits at ~50% of re-embeds
}

# Part 3: first embedding (PyKEEN early stopping on validation MRR, one rule for all models)
MRR = {'learning_rates': [0.001, 0.01, 0.05], 'max_epochs': 100, 'every': 10, 'patience': 2, 'min_gain': 0.01,
       'eval_sample': 2000, 'split': [0.96, 0.02, 0.02]}
# Part 4: later re-embeds (warm updates); limits fixed before measuring
UPD = {'epochs': [1, 2, 3, 5, 10], 'lr_factors': [1.0],      # 1.0 = the first embedding's learning rate (0.2 = Wewer 2021)
       'max_updates': 30, 'mrr_at': [1, 10, 30], 'keep_old_mrr': 0.95, 'max_reference_rise': 0.05, 'tie': 0.01}
PEAK = {'distance': S['window'], 'prominence': 0.05}          # Part 7: a peak = local maximum of the window outlier share

RUN_HELDOUT = os.environ.get('NB_HELDOUT', '0' if DATASET == 'full' else '1') == '1'
REUSE_SAVED = os.environ.get('NB_REUSE_SETTINGS', '1') == '1'   # Part 3: load phase1/settings_check results if present
PART = os.environ.get('NB_PART')            # 'i/n': this execution runs only its share of the 40 runs (Part 5), then stops
PART_I, PART_N = (int(x) for x in PART.split('/')) if PART else (1, 1)
RUNS_ONLY = PART is not None
EMBEDDER = 'pykeen'
MAX_STREAM = None
if TEST:                     # smoke test: tiny trainings, random stand-in vectors in the stream, short stream
    MODELS = ['TransE', 'ComplEx']
    MRR.update({'learning_rates': [0.01], 'max_epochs': 2, 'every': 1, 'eval_sample': 200})
    UPD.update({'epochs': [1, 2], 'max_updates': 3, 'mrr_at': [1, 3]})
    EMBEDDER, MAX_STREAM = 'random', 400
print(json.dumps({'dataset': D, 'settings': S, 'rules': RULES}, indent=1))
""")

# =====================================================================================================
md(r"""
# Part 1 — The pipeline, function by function

The same steps as Antoine's notebook. Each cell defines functions only; they are used from Part 2 on.

The next cell reads a data file and sorts it in time: undated rows first, then by year, date and file row.
""")
code(r"""
def load_triples(path):
    df = pd.read_csv(path, sep='\t')
    rel = 'edge_type' if 'edge_type' in df.columns else 'predicate'
    df = df.rename(columns={rel: 'edge_type'})
    df['row_number'] = np.arange(1, len(df) + 1)
    df['timestamp'] = df['timestamp'].fillna('').astype(str)
    if 'type' not in df.columns:
        df['type'] = ''
    year = pd.to_numeric(df['year'], errors='coerce') if 'year' in df.columns else pd.Series(np.nan, index=df.index)
    ts_year = pd.to_numeric(df['timestamp'].str.extract(r'(\d{4})', expand=False), errors='coerce')
    df['event_year'] = year.fillna(ts_year)
    numeric_ts = pd.to_numeric(df['timestamp'].where(df['timestamp'].ne(''), '0'), errors='coerce').fillna(0)
    df = df.assign(_undated=df['timestamp'].eq(''), _ts=numeric_ts).sort_values(
        ['_undated', 'event_year', '_ts', 'row_number'], ascending=[False, True, True, True], kind='stable')
    df = df.drop(columns=['_undated', '_ts']).reset_index(drop=True)
    df['stream_position'] = np.arange(1, len(df) + 1)
    return df
""")

md(r"""
The next cell builds the starting graph (every triple dated up to the cutoff plus every undated triple, all types), the
monitored triples up to the cutoff, the stream (the monitored type after the cutoff), and the list of known nodes and
relations used to decide whether a triple is new. `KnowledgeGraph` keeps the starting graph plus every streamed triple.
""")
code(r"""
def split_graph(df, cutoff, target_type):
    known = df['event_year'].isna() | (df['event_year'] <= cutoff)
    start, future = df[known].reset_index(drop=True), df[~known].reset_index(drop=True)
    monitored_before = start[start['type'].eq(target_type)].reset_index(drop=True)
    stream = future[future['type'].eq(target_type)].reset_index(drop=True)
    lookup = {'nodes': set(start['subject']) | set(start['object']), 'relations': set(start['edge_type'])}
    return start, monitored_before, stream, lookup


def is_new(row, lookup):
    # A triple is new if its subject, its object or its relation is not in the graph yet (each part on its own).
    return (row['subject'] not in lookup['nodes'] or row['object'] not in lookup['nodes']
            or row['edge_type'] not in lookup['relations'])


class KnowledgeGraph:
    def __init__(self, start_df):
        self.base = start_df[['subject', 'edge_type', 'object']].dropna().astype(str).to_numpy(dtype=object)
        self.added = []

    def add(self, row, lookup):
        values = (row['subject'], row['edge_type'], row['object'])
        if not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
            self.added.append(tuple(str(v) for v in values))
        lookup['nodes'].update([row['subject'], row['object']]); lookup['relations'].add(row['edge_type'])

    def __len__(self):
        return len(self.base) + len(self.added)

    def triples(self):
        return np.vstack([self.base, np.asarray(self.added, dtype=object)]) if self.added else self.base
""")

md(r"""
The next cell trains a PyKEEN model on a graph and returns its vectors (one row per entity and relation, scaled to
length 1). **Cold** = a new model from random numbers. **Warm** = the previous model's numbers are copied in (a new entity
keeps PyKEEN's random start) and training continues for a few epochs over the whole graph. `random` is a stand-in used
only in test mode.
""")
code(r"""
class Embedding:
    def __init__(self, entity_index, entity_matrix, relation_index, relation_matrix, kg_rows, seconds, raw):
        self.entity_index, self.entity_matrix = entity_index, entity_matrix
        self.relation_index, self.relation_matrix = relation_index, relation_matrix
        self.kg_rows, self.seconds, self.raw = kg_rows, seconds, raw


def unit_rows(m):
    m = np.asarray(m, dtype=np.float64); n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1.0
    return m / n


def to_numpy(representations):
    rep = representations
    if not hasattr(rep, '_embeddings') and hasattr(rep, '__getitem__'):
        rep = rep[0]
    with torch.no_grad():
        x = rep(indices=None)
    if torch.is_complex(x):
        x = torch.cat([x.real, x.imag], dim=-1)          # complex -> [real, imaginary]
    return x.reshape(x.shape[0], -1).detach().cpu().numpy().astype(np.float64)


def model_weights(model, tf):
    tables = lambda reps: [r._embeddings.weight.detach().cpu().numpy().copy() for r in reps]
    return {'entity_labels': sorted(tf.entity_to_id, key=tf.entity_to_id.get),
            'relation_labels': sorted(tf.relation_to_id, key=tf.relation_to_id.get),
            'entity': tables(model.entity_representations), 'relation': tables(model.relation_representations)}


def copy_weights(model, tf, raw):
    for reps, labels, new_ids, old_tables in ((model.entity_representations, raw['entity_labels'], tf.entity_to_id, raw['entity']),
                                              (model.relation_representations, raw['relation_labels'], tf.relation_to_id, raw['relation'])):
        old_rows = [i for i, label in enumerate(labels) if label in new_ids]
        new_rows = torch.tensor([new_ids[labels[i]] for i in old_rows], dtype=torch.long)
        for rep, old in zip(reps, old_tables):
            w = rep._embeddings.weight
            with torch.no_grad():
                w.data[new_rows.to(w.device)] = torch.as_tensor(old[old_rows]).to(w.device, w.dtype)


def new_model(name, tf, seed=None):
    import pykeen.models as pm
    dim = S['real_dim'] // 2 if name in COMPLEX_MODELS else S['real_dim']
    return getattr(pm, name)(triples_factory=tf, embedding_dim=dim, random_seed=S['seed'] if seed is None else seed).to(DEVICE)


def train_epochs(model, tf, lr, epochs, loop=None, done=0):
    from pykeen.training import SLCWATrainingLoop
    loop = loop or SLCWATrainingLoop(model=model, triples_factory=tf, optimizer='Adam', optimizer_kwargs={'lr': lr})
    loop.train(triples_factory=tf, num_epochs=done + epochs, batch_size=min(S['batch_size'], max(32, tf.num_triples)),
               continue_training=done > 0, use_tqdm=False, use_tqdm_batch=False, pin_memory=False, num_workers=0)
    return loop


def state_from_model(model, tf, kg_rows, seconds, keep_raw):
    ents = list(tf.entity_to_id); rels = list(tf.relation_to_id)
    em = unit_rows(to_numpy(model.entity_representations)[[tf.entity_to_id[e] for e in ents]])
    rm = unit_rows(to_numpy(model.relation_representations)[[tf.relation_to_id[r] for r in rels]])
    return Embedding({e: i for i, e in enumerate(ents)}, em, {r: i for i, r in enumerate(rels)}, rm,
                     kg_rows, seconds, model_weights(model, tf) if keep_raw else None)


def random_embedding(triples, kg_rows, previous=None):
    rng = np.random.default_rng(S['seed'] + kg_rows)
    ents = sorted(set(triples[:, 0]).union(triples[:, 2])); rels = sorted(set(triples[:, 1]))
    em = unit_rows(rng.normal(size=(len(ents), S['real_dim']))); rm = unit_rows(rng.normal(size=(len(rels), S['real_dim'])))
    if previous is not None:
        for labels, m, oi, om in ((ents, em, previous.entity_index, previous.entity_matrix), (rels, rm, previous.relation_index, previous.relation_matrix)):
            known = [i for i, l in enumerate(labels) if l in oi]
            m[known] = om[[oi[labels[i]] for i in known]]
    return Embedding({e: i for i, e in enumerate(ents)}, em, {r: i for i, r in enumerate(rels)}, rm, kg_rows, 0.0, {})


def embed(kg, model_name, lr, epochs, previous=None, warm_epochs=None, warm_lr=None):
    # Cold if previous is None, warm otherwise. Returns an Embedding.
    triples = kg.triples().astype(str)
    if EMBEDDER == 'random':
        return random_embedding(triples, len(kg), previous)
    from pykeen.triples import TriplesFactory
    t0 = time.time()
    tf = TriplesFactory.from_labeled_triples(triples)
    model = new_model(model_name, tf)
    if previous is not None:
        copy_weights(model, tf, previous.raw)
        model.reset_parameters_ = lambda *a, **k: model      # keep the copied numbers (PyKEEN would re-initialise them)
        train_epochs(model, tf, warm_lr, warm_epochs)
    else:
        train_epochs(model, tf, lr, epochs)
    state = state_from_model(model, tf, len(kg), time.time() - t0, keep_raw=True)
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return state
""")

md(r"""
The next cell turns triples into feature vectors (Hadamard: s ⊙ p ⊙ o, 64 numbers; concatenation: [s, p, o], 192
numbers), fits the scaler and the one-class SVM, and flags outliers.

`fit_reference` also decides **where the baseline outlier rate is measured**: `in_sample` = on the OCSVM's own training
triples (ADAZOR, Algorithm 1); `held_out` = the OCSVM is fitted on one half of the reference and the rate is measured on
the other half (used only in Part 8).
""")
code(r"""
def features_of(subjects, relations, objects, emb, comp):
    s = emb.entity_matrix[[emb.entity_index[str(x)] for x in subjects]]
    p = emb.relation_matrix[[emb.relation_index[str(x)] for x in relations]]
    o = emb.entity_matrix[[emb.entity_index[str(x)] for x in objects]]
    return s * p * o if comp == 'hadamard' else np.concatenate([s, p, o], axis=1)


def features(rows, emb, comp):
    return features_of([r['subject'] for r in rows], [r['edge_type'] for r in rows], [r['object'] for r in rows], emb, comp)


def fit_ocsvm(X):
    scaler = StandardScaler().fit(X) if S['standardize'] else None
    svm = OneClassSVM(kernel='rbf', nu=S['nu'], gamma=S['gamma']).fit(scaler.transform(X) if scaler else X)
    return scaler, svm


def outliers(model, X):
    scaler, svm = model
    return svm.predict(scaler.transform(X) if scaler else X) == -1


def fit_reference(X_ref, baseline='in_sample', split=None):
    # Returns (OCSVM used to score the window, outlier flags whose mean is the baseline outlier rate).
    if baseline == 'in_sample':
        model = fit_ocsvm(X_ref)
        return model, outliers(model, X_ref)
    fit_idx, cal_idx = split
    model = fit_ocsvm(X_ref[fit_idx])
    return model, outliers(model, X_ref[cal_idx])


def make_split(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: n // 2], perm[n // 2:]
""")

md(r"""
The next cell creates the ADAZOR Z-test: an alarm when |window outlier share − baseline outlier rate| >
z(1 − α/2) / √(2n). It uses the `adazor` package (Malchiodi); the same class is written out below as a fallback.
""")
code(r"""
class ZTest:
    def __init__(self, alpha):
        self.alpha = alpha
    def reset(self, flags):
        flags = np.asarray(flags, dtype=bool); self.tot, self.n = int(flags.sum()), len(flags)
    def detect(self, flags):
        import scipy.stats as st
        flags = np.asarray(flags, dtype=bool); n = len(flags)
        return bool(abs(flags.mean() - self.tot / self.n) > st.norm().ppf(1 - self.alpha / 2) / math.sqrt(2 * n))


def make_detector():
    if str(ADAZOR_SRC) not in sys.path:
        sys.path.append(str(ADAZOR_SRC))
    try:
        from adazor.detectors import ZTestDriftDetector
        return ZTestDriftDetector(alpha=S['alpha'])
    except Exception:
        return ZTest(S['alpha'])


import scipy.stats as st
print('alarm threshold for a window of 250:', round(st.norm().ppf(1 - S['alpha'] / 2) / math.sqrt(2 * S['window']), 4))
""")

md(r"""
The next cell is Antoine's sphere: the box around the feature vectors of all graph triples (smallest and largest value
of every number), its centre and half its diagonal (the radius). At a re-embed the new triple's distance from the old
centre is divided by the old radius; Rule 1 refits the OCSVM if that ratio is above k.
""")
code(r"""
def graph_box(kg, emb, comp, chunk=200_000):
    t = kg.triples(); lo = hi = None
    for i in range(0, len(t), chunk):
        part = t[i:i + chunk]
        X = features_of(part[:, 0], part[:, 1], part[:, 2], emb, comp)
        lo = X.min(0) if lo is None else np.minimum(lo, X.min(0)); hi = X.max(0) if hi is None else np.maximum(hi, X.max(0))
    return {'lo': lo, 'hi': hi, 'centre': (lo + hi) / 2, 'radius': float(np.linalg.norm(hi - lo)) / 2}


def stretch_box(box, x):
    lo, hi = np.minimum(box['lo'], x), np.maximum(box['hi'], x)
    if np.array_equal(lo, box['lo']) and np.array_equal(hi, box['hi']):
        return box
    return {'lo': lo, 'hi': hi, 'centre': (lo + hi) / 2, 'radius': float(np.linalg.norm(hi - lo)) / 2}


def sphere_ratio(x, box):
    d = float(np.linalg.norm(x - box['centre']))
    return (np.inf if d > 0 else 0.0) if box['radius'] == 0 else d / box['radius']
""")

md(r"""
The next cell is the stream loop, the heart of the method. For every stream triple:

1. **New subject, object or relation?** Add the triple to the graph; if it is new, re-embed the whole graph (warm start)
   and compute the sphere ratio; otherwise keep the vectors and stretch the box.
2. **Retraining rule:** R2 always refits the OCSVM at a re-embed; R1 and R3 only if the ratio is above k; R4 refits after
   every X stream triples (only if the vectors changed since the last fit). A refit uses the same reference triples with
   the current vectors.
3. **Slide the window** (250 triples) and **test**: the OCSVM flags the window's triples; Z-test against the baseline.
4. **After an alarm:** keep the newest 10% of the window, add the next triples to the graph without testing them
   (their outlier flags under the OCSVM in use are logged: "shadow logging"), re-embed, make these triples the new
   reference and refit the OCSVM.
""")
code(r"""
def run_stream(data, model_name, comp, rule, first, lr, epochs, warm_epochs, warm_lr, baseline='in_sample', max_stream=None):
    W, keep = S['window'], S['keep']
    kg = KnowledgeGraph(data['start']); lookup = {'nodes': set(data['lookup']['nodes']), 'relations': set(data['lookup']['relations'])}
    stream = data['stream'].to_dict('records')[:max_stream] if max_stream else data['stream'].to_dict('records')
    init = data['monitored_before'].to_dict('records')
    n_win = min(W, len(init) - 1)
    reference, window = init[:-n_win], deque(init[-n_win:], maxlen=W)
    emb = first
    box = graph_box(kg, emb, comp)
    splits = 0
    split = make_split(len(reference), S['seed'] + splits) if baseline == 'held_out' else None
    fitted = {id(reference[j]) for j in split[0]} if split else set()   # held-out: triples the OCSVM was fitted on
    ocsvm, ref_flags = fit_reference(features(reference, emb, comp), baseline, split)
    det = make_detector(); det.reset(ref_flags)
    k = rule.get('k', 1.0)
    history, alarms, reembeds, refills = [], [], [], []
    since_fit, changed, i, t_start = 0, False, 0, time.time()

    def refit():
        nonlocal ocsvm, ref_flags, since_fit, changed
        ocsvm, ref_flags = fit_reference(features(reference, emb, comp), baseline, split)
        det.reset(ref_flags); since_fit, changed = 0, False

    while i < len(stream):
        row = stream[i]; i += 1
        new = is_new(row, lookup)
        if new:
            kg.add(row, lookup)
            emb = embed(kg, model_name, lr, epochs, previous=emb, warm_epochs=warm_epochs, warm_lr=warm_lr)
            x = features([row], emb, comp)[0]
            ratio = sphere_ratio(x, box)
            box = graph_box(kg, emb, comp)
            refitted = rule['kind'] == 'always' or (rule['kind'] == 'sphere' and np.isfinite(ratio) and ratio > k)
            if refitted:
                refit()
            else:
                changed = True
            reembeds.append({'stream_index': i, 'trigger': 'new entity', 'ratio': ratio, 'refit': refitted, 'seconds': emb.seconds})
        else:
            x = features([row], emb, comp)[0]
            kg.add(row, lookup); box = stretch_box(box, x); ratio, refitted = np.nan, False
        scheduled = False
        since_fit += 1
        if rule['kind'] == 'every' and since_fit >= rule['x']:
            scheduled, since_fit = True, 0
            if changed:
                refit(); refitted = True
        window.append(row)
        rows_w = list(window)
        flags = outliers(ocsvm, features(rows_w, emb, comp))
        if fitted:   # held-out: the window's share is measured only on triples the OCSVM was not fitted on
            flags = flags[np.fromiter((id(r) not in fitted for r in rows_w), dtype=bool, count=len(rows_w))]
        alarm = bool(det.detect(flags)) if len(flags) else False
        history.append({'stream_index': i, 'timestamp': row['timestamp'], 'event_year': row['event_year'], 'row_number': row['row_number'],
                        'window_outlier_share': float(flags.mean()) if len(flags) else np.nan, 'baseline_outlier_rate': float(np.mean(ref_flags)),
                        'reembedded': new, 'refitted': refitted, 'alarm': alarm})
        if not alarm:
            continue
        alarms.append(history[-1])
        rows = list(window)[-max(1, int(round(W * keep))):]
        refill = []
        while len(rows) < W and i < len(stream):
            r = stream[i]; i += 1
            kg.add(r, lookup); rows.append(r); refill.append((i, r))
        scorable = [(j, r) for j, r in refill if str(r['subject']) in emb.entity_index and str(r['object']) in emb.entity_index
                    and str(r['edge_type']) in emb.relation_index]
        shadow = outliers(ocsvm, features([r for _, r in scorable], emb, comp)) if scorable else np.array([], bool)
        sw = deque(flags, maxlen=W)
        for (j, r), f in zip(scorable, shadow):
            sw.append(bool(f))
            refills.append({'stream_index': j, 'timestamp': r['timestamp'], 'shadow_outlier': bool(f),
                            'shadow_window_share': float(np.mean(sw)), 'shadow_would_alarm': bool(det.detect(list(sw)))})
        refills.extend({'stream_index': j, 'timestamp': r['timestamp'], 'shadow_outlier': None, 'shadow_window_share': None,
                        'shadow_would_alarm': None} for j, r in refill if (j, r) not in scorable)
        reference = rows
        emb = embed(kg, model_name, lr, epochs, previous=emb, warm_epochs=warm_epochs, warm_lr=warm_lr)
        box = graph_box(kg, emb, comp)
        reembeds.append({'stream_index': i, 'trigger': 'alarm', 'ratio': np.nan, 'refit': True, 'seconds': emb.seconds})
        window = deque(reference[-W:], maxlen=W)
        splits += 1
        split = make_split(len(reference), S['seed'] + splits) if baseline == 'held_out' else None
        fitted = {id(reference[j]) for j in split[0]} if split else set()
        refit()
    summary = {'stream_triples': len(stream), 'tested': len(history), 'alarms': len(alarms), 'reembeds': len(reembeds),
               'refits_by_rule': int(sum(h['refitted'] for h in history)),   # between alarms, all rules (R4 included)
               'refill_triples': len(refills), 'minutes': round((time.time() - t_start) / 60, 2)}
    return {'summary': summary, 'history': pd.DataFrame(history), 'alarms': pd.DataFrame(alarms),
            'reembeds': pd.DataFrame(reembeds), 'refill': pd.DataFrame(refills)}
""")

md(r"""
The next cell checks that the notebook's pipeline is the same as the code that produced the earlier runs
(`phase1/drift_sphere.py`): one CASE1 run per rule, with the deterministic random stand-in embedding (so both are
compared on identical vectors) on the first 400 stream triples. The alarm positions must be identical.
""")
code(r"""
def equivalence_check():
    p1 = ROOT / 'phase1'
    if not (p1 / 'drift_sphere.py').exists():
        print('phase1/drift_sphere.py not found: check skipped'); return
    sys.path.insert(0, str(p1)); import drift_sphere as ds
    global EMBEDDER
    saved, EMBEDDER = EMBEDDER, 'random'
    c = DATASETS['case1']; df = load_triples(DATA_DIR / c['file'])
    start, mon, stream, lookup = split_graph(df, c['cutoff'], c['type'])
    data = {'start': start, 'monitored_before': mon, 'stream': stream, 'lookup': lookup}
    first = random_embedding(KnowledgeGraph(start).triples().astype(str), len(KnowledgeGraph(start)))
    i_kg, i_st, f_st, lk = ds.build_initial_knowledge_graph(ds.load_temporal_triples(DATA_DIR / c['file']), c['cutoff'], c['type'])
    rows = []
    for policy, rule in [('always', {'kind': 'always'}), ('sphere', {'kind': 'sphere', 'k': 1.0}), ('every', {'kind': 'every', 'x': 50})]:
        for comp in COMPOSITIONS:
            ours = run_stream(data, 'TransE', comp, rule, first, 0, 0, 0, 0, max_stream=400)
            cfg = {'window_size': S['window'], 'max_stream': 400, 'composition': comp, 'embedder': 'random', 'model': 'TransE',
                   'real_dim': S['real_dim'], 'seed': S['seed'], 'batch_size': S['batch_size'], 'device': 'cpu',
                   'embed_update': 'warm', 'epochs': 0, 'learning_rate': 0, 'warm_epochs': 0, 'warm_learning_rate': 0,
                   'sphere_k': 1.0, 'retrain_policy': policy, 'retrain_every': 50, 'standardize': S['standardize'],
                   'nu': S['nu'], 'gamma': S['gamma'], 'keep_fraction_after_drift': S['keep'], 'save_drift_windows': False}
            det, _ = ds.make_detector('ztest', S['alpha'], ADAZOR_SRC)
            run_dir = OUT / 'equivalence' / f'{policy}_{comp}'; run_dir.mkdir(parents=True, exist_ok=True)
            theirs = ds.run_stream(i_kg, i_st, f_st, lk, cfg, run_dir, det)
            a = list(ours['alarms']['stream_index']) if len(ours['alarms']) else []
            b = list(theirs['drifts']['stream_index']) if len(theirs['drifts']) else []
            rows.append({'rule': policy, 'composition': comp, 'alarms (notebook)': a, 'alarms (drift_sphere.py)': b, 'identical': a == b})
    EMBEDDER = saved
    t = pd.DataFrame(rows); show(t)
    print('ALL IDENTICAL' if t['identical'].all() else 'DIFFERENCES FOUND')


if DATASET == 'case1' and not RUNS_ONLY:
    equivalence_check()
else:
    print('the equivalence check runs in the CASE1 execution (without NB_PART)')
""")

# =====================================================================================================
md(r"""
# Part 2 — Data

The next cell loads the dataset, builds the starting graph and the stream, and shows their sizes. On the cases, the
*switch* is the first stream triple of the year in which the topic changes.
""")
code(r"""
df = load_triples(DATA_DIR / D['file'])
start, monitored_before, stream, lookup = split_graph(df, D['cutoff'], D['type'])
DATA = {'start': start, 'monitored_before': monitored_before, 'stream': stream, 'lookup': lookup}
n_win = min(S['window'], len(monitored_before) - 1)
SWITCH = int(stream.index[stream['event_year'] >= D['switch_year']][0]) + 1 if D['switch_year'] else None
show(pd.DataFrame([{'file rows': len(df), 'starting graph': len(start), 'undated in starting graph': int(start['event_year'].isna().sum()),
                    'relation types in starting graph': start['type'].nunique(), 'monitored type': D['type'],
                    'monitored up to the cutoff': len(monitored_before), 'reference': len(monitored_before) - n_win,
                    'first window': n_win, 'stream': len(stream), 'switch at stream triple': SWITCH}]).T.rename(columns={0: DATASET}))
""")

md(r"""
The next cell turns timestamps into dates, used by every plot and table of Parts 6–8.
""")
code(r"""
def to_date(ts):
    v = pd.to_numeric(pd.Series(ts), errors='coerce')
    return pd.to_datetime(v, unit='s', errors='coerce')

stream_dates = to_date(stream['timestamp']).reset_index(drop=True)
print('stream from', stream_dates.min().date(), 'to', stream_dates.max().date())
""")

# =====================================================================================================
md(r"""
# Part 3 — First embedding: learning rate and epochs

**Rule (the same for every model).** The starting graph is split 96 / 2 / 2 (training / validation / test). Each model is
trained at learning rates 0.001, 0.01 and 0.05; every 10 epochs the validation MRR (filtered ranking, on up to 2,000
validation triples) is computed; training stops after 2 checks without a 1% improvement, or at 100 epochs. Per model the
learning rate with the best MRR is kept, with the epochs at which it was reached.

*Literature:* early stopping on validation MRR (Ruffinelli et al. 2020; Ali et al. 2020); the best learning rate differs
per model (Bonner et al. 2021); MRR does not measure stability (Méroué et al. 2026).

The next cell prepares the split and the evaluation triples (on the CASE1 graph this takes seconds; on the full graph
a few minutes).
""")
code(r"""
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator

START_KG = KnowledgeGraph(start)
start_triples = START_KG.triples().astype(str)
if TEST:   # tiny graph: the monitored triples before the cutoff plus 3,000 random others
    mon = monitored_before[['subject', 'edge_type', 'object']].astype(str).to_numpy()
    start_triples = np.vstack([mon, start_triples[np.random.default_rng(0).choice(len(start_triples), 3000, replace=False)]])
TF = TriplesFactory.from_labeled_triples(start_triples)
TRAIN, VALID, TESTSET = TF.split(MRR['split'], random_state=S['seed'])
rng = np.random.default_rng(S['seed'])
pick = lambda m, n: m[rng.choice(len(m), n, replace=False)] if len(m) > n else m
VALID_SAMPLE = pick(VALID.mapped_triples, MRR['eval_sample'])
TRAIN_SAMPLE = pick(TRAIN.mapped_triples, MRR['eval_sample'])
KNOWN = [TRAIN.mapped_triples, VALID.mapped_triples, TESTSET.mapped_triples]
EVALUATOR = RankBasedEvaluator(filtered=True)
mrr_of = lambda model, triples, known: float(EVALUATOR.evaluate(model, triples, additional_filter_triples=known, use_tqdm=False, batch_size=256)
                                             .get_metric('both.realistic.inverse_harmonic_mean_rank'))
print(f'starting graph {TF.num_triples} triples, {TF.num_entities} entities; validation sample {len(VALID_SAMPLE)}; '
      f'random-guess MRR {np.log(TF.num_entities) / TF.num_entities:.5f}')
""")

md(r"""
The next cell applies the rule for every model and learning rate and shows the validation MRR per learning rate and the
chosen setting. The same rule was already run by `phase1/settings_check.py` (same split, seed, validation triples,
checks every 10 epochs, patience 2, 1% gain, at most 100 epochs, choice by best MRR); when its saved result exists it is
loaded (marked "loaded"), otherwise it is computed here (marked "computed here"). `NB_REUSE_SETTINGS=0` forces computing.
""")
code(r"""
SAVED = ROOT / 'phase1' / 'settings_check'      # results of phase1/settings_check.py (same rule, same split)


def mrr_curve(model_name, lr):
    f = OUT / 'first_embedding' / f'{model_name}__lr{lr:g}.json'
    if f.exists():
        return json.loads(f.read_text())
    saved = SAVED / f'{DATASET}__{model_name}__lr{lr:g}.json'
    if REUSE_SAVED and saved.exists():
        r = json.loads(saved.read_text())
        return {'model': model_name, 'lr': lr, 'best_mrr': r['best_mrr'], 'best_epoch': r['best_epoch'],
                'stopped_at': r['stopped_at_epoch'], 'seconds_per_epoch': r['seconds_per_epoch'], 'source': f'loaded: {saved.name}',
                'curve': [{'epoch': c['epoch'], 'validation_mrr': c['mrr']} for c in r['curve']]}
    f.parent.mkdir(parents=True, exist_ok=True)
    model = new_model(model_name, TRAIN); loop, done, curve, best, best_ep, bad = None, 0, [], -1.0, 0, 0
    t0 = time.time()
    while done < MRR['max_epochs']:
        step = min(MRR['every'], MRR['max_epochs'] - done)
        loop = train_epochs(model, TRAIN, lr, step, loop, done); done += step
        m = mrr_of(model, VALID_SAMPLE, KNOWN)
        curve.append({'epoch': done, 'validation_mrr': m})
        if m > best * (1 + MRR['min_gain']):
            best, best_ep, bad = m, done, 0
        else:
            bad += 1
            if bad >= MRR['patience']:
                break
    res = {'model': model_name, 'lr': lr, 'best_mrr': best, 'best_epoch': best_ep, 'stopped_at': done,
           'seconds_per_epoch': (time.time() - t0) / max(done, 1), 'source': 'computed here', 'curve': curve}
    f.write_text(json.dumps(res)); del model
    return res


rows = []
for m in MODELS:
    for lr in MRR['learning_rates']:
        r = mrr_curve(m, lr)
        print(f"{m:9s} lr {lr:<6g} best validation MRR {r['best_mrr']:.4f} at epoch {r['best_epoch']}  ({r.get('source', '')})")
        rows.append(r)
first_table = pd.DataFrame(rows)
CHOSEN = {}
for m, d in first_table.groupby('model', sort=False):
    b = d.sort_values('best_mrr', ascending=False).iloc[0]
    CHOSEN[m] = {'lr': float(b['lr']), 'epochs': int(max(b['best_epoch'], 1))}
pivot = first_table.pivot(index='model', columns='lr', values='best_mrr').round(4)
pivot['chosen lr'] = [CHOSEN[m]['lr'] for m in pivot.index]; pivot['chosen epochs'] = [CHOSEN[m]['epochs'] for m in pivot.index]
show(pivot)
""")

md(r"""
The next cell plots the validation MRR during training, per model and learning rate (the curve the rule stops on).
""")
code(r"""
fig, axes = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 2.8), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    for _, r in first_table[first_table.model.eq(m)].iterrows():
        c = pd.DataFrame(r['curve']); ax.plot(c['epoch'], c['validation_mrr'], marker='o', ms=3, label=f"lr {r['lr']:g}")
    ax.set_title(m, fontsize=10); ax.set_xlabel('epoch')
np.atleast_1d(axes)[0].set_ylabel('validation MRR'); np.atleast_1d(axes)[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
The next cell defines the first embedding used by the runs: the model trained on the **whole** starting graph with the
chosen learning rate and epochs. It is trained once per model, saved, and shared by all runs of that model.
""")
code(r"""
def first_embedding(model_name, seed=S['seed']):
    f = OUT / 'first_embedding' / f'{model_name}__seed{seed}__state.pkl'
    if f.exists():
        return Embedding(**pickle.loads(f.read_bytes()))
    ch = CHOSEN.get(model_name, {'lr': 0.001, 'epochs': 2})
    if EMBEDDER == 'random':
        st_ = random_embedding(START_KG.triples().astype(str), len(START_KG))
    else:
        tf = TriplesFactory.from_labeled_triples(START_KG.triples().astype(str)); t0 = time.time()
        model = new_model(model_name, tf, seed); train_epochs(model, tf, ch['lr'], ch['epochs'])
        st_ = state_from_model(model, tf, len(START_KG), time.time() - t0, keep_raw=True); del model
    f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(pickle.dumps(vars(st_)))   # plain data, not the class
    return st_
""")

# =====================================================================================================
md(r"""
# Part 4 — Every later re-embed: how many epochs?

A warm re-embed continues training the previous vectors. Too few epochs: the new triples are not learned. Too many: the
vectors drift away from where the OCSVM was fitted (the coordinates change), and the old knowledge is overwritten
(*catastrophic forgetting*; Cui et al. 2022; Pons et al. 2026). One update can look harmless while hundreds in a row add
up, so the check follows **a sequence of updates**, exactly as in the stream: walk through the stream, and at every
triple with a new entity do a warm update of *e* epochs.

**Measured after every update** (for e = 1, 2, 3, 5, 10 epochs, learning rate = the first embedding's; Wewer et al. 2021
recommend about one fifth, which can be added in `UPD['lr_factors']`):
1. **coordinates kept**: outlier share of the reference under the OCSVM fitted on the *first* embedding (per composition);
2. **old knowledge kept**: validation MRR of the starting graph (at updates 1, 10, 30);
3. **new knowledge**: MRR of the stream triples added so far (at updates 1, 10, 30);
4. mean cosine between every original entity vector and its current vector.

**Choice (limits fixed before measuring), per model and composition, at the last update:** old MRR ≥ 95% of the first
embedding's; reference outlier share at most 0.05 above the first embedding's; among those, the best new-triple MRR
(within 1% counts as equal), then the fewest epochs. If no setting passes, the gentlest (fewest epochs) is taken and
marked.

The next cell runs the sequence for every model and setting (saved per model and setting).
""")
code(r"""
BASES = {}


def base_for(model_name):
    # The embedding the updates start from: trained once per model on the training part of the starting graph.
    if model_name not in BASES:
        ch = CHOSEN[model_name]
        base = new_model(model_name, TRAIN); train_epochs(base, TRAIN, ch['lr'], ch['epochs'])
        BASES[model_name] = (state_from_model(base, TRAIN, TRAIN.num_triples, 0.0, keep_raw=True), mrr_of(base, VALID_SAMPLE, KNOWN))
        del base
    return BASES[model_name]


def update_sequence(model_name, e, lr):
    f = OUT / 'update_epochs' / f'{model_name}__e{e}__lr{lr:g}.json'
    if f.exists():
        return json.loads(f.read_text())
    f.parent.mkdir(parents=True, exist_ok=True)
    base_state, base_mrr = base_for(model_name)
    ref_rows = [r for r in monitored_before.to_dict('records')[:-n_win]
                if str(r['subject']) in base_state.entity_index and str(r['object']) in base_state.entity_index]
    if len(ref_rows) < 20:
        print(f'{model_name}: only {len(ref_rows)} reference triples have vectors in this graph; check skipped'); return
    ref_models = {c: fit_ocsvm(features(ref_rows, base_state, c)) for c in COMPOSITIONS}
    ref_rate0 = {c: float(outliers(ref_models[c], features(ref_rows, base_state, c)).mean()) for c in COMPOSITIONS}
    labels = list(base_state.entity_index); old_vecs = base_state.entity_matrix
    triples = [tuple(t) for t in TF.label_triples(TRAIN.mapped_triples)]
    nodes = set(base_state.entity_index); rels = set(base_state.relation_index)
    added, prev, rows, n_upd = [], base_state, [], 0
    for r in stream.to_dict('records'):
        t = (str(r['subject']), str(r['edge_type']), str(r['object']))
        new = t[0] not in nodes or t[2] not in nodes or t[1] not in rels
        triples.append(t); added.append(t); nodes.update([t[0], t[2]]); rels.add(t[1])
        if not new:
            continue
        n_upd += 1
        tf = TriplesFactory.from_labeled_triples(np.asarray(triples, dtype=str))
        model = new_model(model_name, tf); copy_weights(model, tf, prev.raw)
        model.reset_parameters_ = lambda *a, _m=model, **k: _m
        t0 = time.time(); train_epochs(model, tf, lr, e); secs = time.time() - t0
        cur = state_from_model(model, tf, tf.num_triples, secs, keep_raw=True)
        rec = {'update': n_upd, 'stream_triples_added': len(added), 'seconds': secs,
               'old_cosine': float(np.mean(np.sum(old_vecs * cur.entity_matrix[[cur.entity_index[l] for l in labels]], axis=1)))}
        for c in COMPOSITIONS:
            rec[f'reference_outliers_{c}'] = float(outliers(ref_models[c], features(ref_rows, cur, c)).mean())
        if n_upd in UPD['mrr_at']:
            ids = lambda lab: TriplesFactory.from_labeled_triples(np.asarray(lab, dtype=str), entity_to_id=tf.entity_to_id,
                                                                  relation_to_id=tf.relation_to_id).mapped_triples
            known = [tf.mapped_triples, ids(TF.label_triples(VALID.mapped_triples)), ids(TF.label_triples(TESTSET.mapped_triples))]
            rec['old_mrr'] = mrr_of(model, ids(TF.label_triples(VALID_SAMPLE)), known)
            rec['new_mrr'] = mrr_of(model, pick(ids(added), MRR['eval_sample']), known)
        rows.append(rec); prev = cur; del model
        if n_upd >= UPD['max_updates']:
            break
    res = {'model': model_name, 'epochs': e, 'lr': lr, 'base_mrr': base_mrr, 'base_reference_outliers': ref_rate0, 'updates': rows}
    f.write_text(json.dumps(res))
    return res


DONE = OUT / 'update_epochs' / 'DONE'
if RUNS_ONLY and PART_I != 1:
    print('waiting for part 1 to finish Part 4 and the first embeddings ...', flush=True)
    while not DONE.exists():
        time.sleep(60)
UPDATE_RESULTS = []
for m in MODELS:
    for fac in UPD['lr_factors']:
        for e in UPD['epochs']:
            r = update_sequence(m, e, CHOSEN[m]['lr'] * fac)
            if r is None:
                continue
            UPDATE_RESULTS.append(r)
            last = r['updates'][-1] if r['updates'] else {}
            print(f"{m:9s} {e:2d} epochs lr {r['lr']:<7g}: {len(r['updates'])} updates; last: old MRR {last.get('old_mrr', float('nan')):.4f} "
                  f"(first {r['base_mrr']:.4f}); reference outliers had {last.get('reference_outliers_hadamard', float('nan')):.2f} "
                  f"conc {last.get('reference_outliers_concatenation', float('nan')):.2f}")
if not DONE.exists():   # train every first embedding now, so parallel executions can share them
    for m in MODELS:
        first_embedding(m)
    DONE.parent.mkdir(parents=True, exist_ok=True); DONE.write_text('done')
""")

md(r"""
The next cell plots, per model, how the reference outlier share (coordinates kept?) and the old-graph MRR (old knowledge
kept?) evolve over the sequence of updates, for each number of epochs. Flat lines = the coordinates stay where the OCSVM
was fitted.
""")
code(r"""
for comp in (COMPOSITIONS if UPDATE_RESULTS else []):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 2.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        if not any(x['model'] == m for x in UPDATE_RESULTS):
            continue
        for r in [x for x in UPDATE_RESULTS if x['model'] == m]:
            u = pd.DataFrame(r['updates'])
            if len(u):
                ax.plot(u['update'], u[f'reference_outliers_{comp}'], marker='.', label=f"{r['epochs']} ep, lr {r['lr']:g}")
        base = next(x['base_reference_outliers'][comp] for x in UPDATE_RESULTS if x['model'] == m)
        ax.axhline(base, color='k', lw=0.8); ax.axhline(base + UPD['max_reference_rise'], color='k', lw=0.8, ls='--')
        ax.set_title(m, fontsize=10); ax.set_xlabel('update')
    np.atleast_1d(axes)[0].set_ylabel(f'reference outlier share ({comp})'); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f'{comp}: coordinates kept? (solid = first embedding, dashed = allowed rise)', x=0.01, ha='left', fontsize=10)
    plt.tight_layout(); plt.show()
""")

md(r"""
The next cell applies the choice rule and shows the chosen number of epochs per model and composition, used by the runs.
""")
code(r"""
rows = []
for r in UPDATE_RESULTS:
    u = pd.DataFrame(r['updates'])
    if not len(u):
        continue
    with_mrr = u.dropna(subset=['old_mrr']) if 'old_mrr' in u else u.iloc[0:0]
    last = with_mrr.iloc[-1] if len(with_mrr) else u.iloc[-1]
    for c in COMPOSITIONS:
        rows.append({'model': r['model'], 'composition': c, 'epochs': r['epochs'], 'lr': r['lr'], 'updates': int(u['update'].max()),
                     'old_mrr_kept': float(last.get('old_mrr', np.nan)) / r['base_mrr'] if r['base_mrr'] > 0 else np.nan,
                     'reference_rise': float(u[f'reference_outliers_{c}'].iloc[-1]) - r['base_reference_outliers'][c],
                     'new_mrr': float(last.get('new_mrr', np.nan)), 'old_cosine': float(u['old_cosine'].iloc[-1])})
UPDATE_TABLE = pd.DataFrame(rows)
WARM = {}
for (m, c), d in UPDATE_TABLE.groupby(['model', 'composition'], sort=False):
    ok = d[(d['old_mrr_kept'] >= UPD['keep_old_mrr']) & (d['reference_rise'] <= UPD['max_reference_rise'])]
    if len(ok):
        best = ok['new_mrr'].max()
        p = ok[ok['new_mrr'] >= (1 - UPD['tie']) * best].sort_values(['epochs', 'lr']).iloc[0]; passes = True
    else:
        p = d.sort_values(['epochs', 'lr']).iloc[0]; passes = False
    WARM[(m, c)] = {'epochs': int(p['epochs']), 'lr': float(p['lr']), 'passes': passes}
show(UPDATE_TABLE.round(3))
show(pd.DataFrame([{'model': m, 'composition': c, **v} for (m, c), v in WARM.items()]))
""")

# =====================================================================================================
md(r"""
# Part 5 — The 40 runs

The next cell runs the stream for every model, composition and rule, with the first embedding of Part 3 (trained once per
model and shared by its runs) and the warm-update epochs of Part 4. R2 runs first: R3's sphere size k is set from R2's
logged ratios (per composition, the 5 models pooled), so that R3 refits at about 50% of re-embeds. Each finished run is
saved in `output/<dataset>/runs/` and skipped on a restart. With `NB_PART=i/n` this execution runs only every n-th run
(several executions in parallel), and R3 waits until all R2 runs exist.
""")
code(r"""
def run_name(model, comp, rule, baseline):
    return f'{model}__{comp}__{rule}' + ('' if baseline == 'in_sample' else '__heldout')


def run_one(model, comp, rule_name, baseline='in_sample'):
    d = OUT / ('runs' if baseline == 'in_sample' else 'runs_heldout') / run_name(model, comp, rule_name, baseline)
    if (d / 'summary.json').exists():
        return
    rule = dict(RULES[rule_name])
    if 'share' in rule:
        need = [OUT / 'runs' / run_name(m, comp, 'R2_always', 'in_sample') / 'summary.json' for m in MODELS]
        while not all(x.exists() for x in need):   # R3's sphere size comes from the R2 runs (maybe running elsewhere)
            time.sleep(60)
        rule['k'] = sphere_k(comp, rule['share'])
    ch = CHOSEN.get(model, {'lr': 0.001, 'epochs': 2}); wm = WARM.get((model, comp), {'epochs': 1, 'lr': ch['lr']})
    res = run_stream(DATA, model, comp, rule, first_embedding(model), ch['lr'], ch['epochs'], wm['epochs'], wm['lr'],
                     baseline=baseline, max_stream=MAX_STREAM)
    d.mkdir(parents=True, exist_ok=True)
    for key in ('history', 'alarms', 'reembeds', 'refill'):
        res[key].to_csv(d / f'{key}.tsv.gz', sep='\t', index=False)
    (d / 'summary.json').write_text(json.dumps({**res['summary'], 'model': model, 'composition': comp, 'rule': rule_name,
                                                'k': rule.get('k'), 'baseline': baseline, 'warm_epochs': wm['epochs'], 'warm_lr': wm['lr']}))
    print(f"{run_name(model, comp, rule_name, baseline):45s} alarms {res['summary']['alarms']:4d}  re-embeds {res['summary']['reembeds']:4d}  "
          f"{res['summary']['minutes']:.1f} min", flush=True)


def sphere_k(comp, share):
    ratios = []
    for f in (OUT / 'runs').glob(f'*__{comp}__R2_always/reembeds.tsv.gz'):
        r = pd.read_csv(f, sep='\t'); ratios += list(r.loc[r['trigger'].eq('new entity'), 'ratio'].dropna())
    return float(np.quantile(ratios, 1 - share)) if ratios else 1.0


def run_grid(baseline='in_sample'):
    grid = [(m, c, r) for r in RULES for m in MODELS for c in COMPOSITIONS]
    for n, (model, comp, rule_name) in enumerate(grid):
        if RUNS_ONLY and n % PART_N != PART_I - 1:
            continue
        run_one(model, comp, rule_name, baseline)


t0 = time.time(); run_grid()
print(f"{'part ' + PART + ': my' if RUNS_ONLY else 'all'} runs done ({(time.time() - t0) / 60:.1f} min in this execution)")
""")
WRAP = True   # every code cell from here on is skipped in the NB_PART executions

# =====================================================================================================
md(r"""
# Part 6 — Results: how many alarms and where

The next cell loads every run and shows, per run: alarms, re-embeds, refits by the rule (between alarms; every rule also
refits after each alarm), the share of the stream tested,
the run time, and on the cases the alarms before and after the switch and the delay of the first alarm after it.
""")
code(r"""
def load_runs(folder='runs'):
    out = {}
    for d in sorted((OUT / folder).glob('*__*__*')):
        if (d / 'summary.json').exists():
            r = {}
            for k in ('history', 'alarms', 'reembeds', 'refill'):
                try:
                    r[k] = pd.read_csv(d / f'{k}.tsv.gz', sep='\t')
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    r[k] = pd.DataFrame()
            r['summary'] = json.loads((d / 'summary.json').read_text()); out[d.name] = r
    return out


RUNS = load_runs()
rows = []
for name, r in RUNS.items():
    s, a = r['summary'], r['alarms']; idx = list(a['stream_index']) if len(a) else []
    rec = {'model': s['model'], 'composition': s['composition'], 'rule': s['rule'], 'alarms': s['alarms'],
           'first alarm at': idx[0] if idx else None, 're-embeds': s['reembeds'],
           # refits by the rule between alarms (R1/R2/R3 at a new entity, R4 on its schedule), counted from the history
           'refits by rule': int(r['history']['refitted'].sum()) if len(r['history']) else 0,
           'tested share': round(s['tested'] / s['stream_triples'], 3), 'warm epochs': s['warm_epochs'], 'minutes': s['minutes']}
    if SWITCH:
        rec.update({'alarms before switch': sum(i < SWITCH for i in idx), 'alarms after switch': sum(i >= SWITCH for i in idx),
                    'delay': next((i - SWITCH for i in idx if i >= SWITCH), None)})
    rows.append(rec)
RESULTS = pd.DataFrame(rows).sort_values(['rule', 'model', 'composition']).reset_index(drop=True)
show(RESULTS)
""")

md(r"""
The next cell lists where every run alarmed, as dates (year-month).
""")
code(r"""
rows = []
for name, r in RUNS.items():
    a = r['alarms']
    rows.append({'run': name, 'alarm dates': ', '.join(to_date(a['timestamp']).dt.strftime('%Y-%m').fillna('?')) if len(a) else '—'})
show(pd.DataFrame(rows))
""")

md(r"""
The next cell draws, for every run, the window outlier share at every tested triple (blue), the baseline outlier rate
(orange), the alarms (black ticks), the re-embeds (grey ticks below) and, on the cases, the switch (red line). One figure
per rule, one panel per model and composition.
""")
code(r"""
def plot_run(ax, r, title):
    h = r['history']
    if not len(h):
        return
    ax.plot(h['stream_index'], h['window_outlier_share'], lw=0.9, color='#2a78d6')
    ax.plot(h['stream_index'], h['baseline_outlier_rate'], lw=0.9, color='#eb6834')
    for i in r['alarms']['stream_index'] if len(r['alarms']) else []:
        ax.axvline(i, color='k', lw=0.6, ymin=0.85)
    for i in r['reembeds']['stream_index'] if len(r['reembeds']) else []:
        ax.axvline(i, color='0.6', lw=0.4, ymax=0.06)
    if SWITCH:
        ax.axvline(SWITCH, color='#b4452f', lw=1.2, ls='--')
    ax.set_ylim(0, 1.02); ax.set_title(title, fontsize=8)


for rule_name in RULES:
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(12, 1.7 * len(MODELS)), sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(MODELS):
        for j, c in enumerate(COMPOSITIONS):
            name = run_name(m, c, rule_name, 'in_sample')
            if name in RUNS:
                plot_run(axes[i, j], RUNS[name], f'{m} · {c}')
    fig.suptitle(f'{rule_name}: window outlier share (blue), baseline (orange), alarms (black), re-embeds (grey)', x=0.01, ha='left')
    for ax in axes[-1]:
        ax.set_xlabel('stream triple')
    plt.tight_layout(); plt.show()
""")

# =====================================================================================================
md(r"""
# Part 7 — Analysis: alarm density and peaks

## 8.1 Where the alarms pile up

The next cell draws the density of alarms over time: all 40 runs together (top) and per rule (bottom). Bars are months
on the full data and quarters on the cases.
""")
code(r"""
freq = 'M' if DATASET == 'full' else 'Q'
alarm_dates = pd.concat([to_date(r['alarms']['timestamp']).to_frame('date').assign(rule=r['summary']['rule'])
                         for r in RUNS.values() if len(r['alarms'])], ignore_index=True) if RUNS else pd.DataFrame(columns=['date', 'rule'])
if len(alarm_dates):
    period = alarm_dates['date'].dt.to_period(freq).dt.to_timestamp()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    counts = period.value_counts().sort_index(); a1.bar(counts.index, counts.values, width=20 if freq == 'M' else 60, color='#1b2422')
    a1.set_ylabel('alarms (40 runs)')
    for rule_name in RULES:
        c = period[alarm_dates['rule'].eq(rule_name)].value_counts().sort_index(); a2.plot(c.index, c.values, marker='.', label=rule_name)
    a2.set_ylabel('alarms per rule'); a2.legend(fontsize=8)
    if SWITCH:
        for ax in (a1, a2):
            ax.axvline(stream_dates.iloc[SWITCH - 1], color='#b4452f', ls='--')
    plt.tight_layout(); plt.show()
""")

md(r"""
## 8.2 Peaks of the outlier share

A peak is a local maximum of a run's window outlier share, at least one window (250 triples) away from the next peak and
standing out by at least 0.05 (scipy `find_peaks`, distance and prominence fixed in Part 0). Alarms show where the test
fired; peaks show where the outlier share rose, also between alarms.

The next cell finds the peaks of every run and draws their density over time, like 8.1.
""")
code(r"""
rows = []
for name, r in RUNS.items():
    h = r['history']
    if len(h) < 3:
        continue
    idx, props = find_peaks(h['window_outlier_share'].to_numpy(), distance=PEAK['distance'], prominence=PEAK['prominence'])
    for i, prom in zip(idx, props['prominences']):
        rows.append({'run': name, 'rule': r['summary']['rule'], 'stream_index': int(h['stream_index'].iloc[i]),
                     'date': to_date([h['timestamp'].iloc[i]]).iloc[0], 'height': float(h['window_outlier_share'].iloc[i]),
                     'prominence': float(prom)})
PEAKS = pd.DataFrame(rows)
print(len(PEAKS), 'peaks in', PEAKS['run'].nunique() if len(PEAKS) else 0, 'runs')
if len(PEAKS):
    period = PEAKS['date'].dt.to_period(freq).dt.to_timestamp(); counts = period.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 2.8)); ax.bar(counts.index, counts.values, width=20 if freq == 'M' else 60, color='#17675a')
    ax.set_ylabel('peaks (40 runs)')
    if SWITCH:
        ax.axvline(stream_dates.iloc[SWITCH - 1], color='#b4452f', ls='--')
    plt.tight_layout(); plt.show()
""")

md(r"""
## 8.3 What is behind the peaks

The next cell lists the peak dates shared by the most runs and, for each, the monitored triples of that date with the
disease name (from the MONDO ontology, `data/mondo.obo`), to trace them to the papers.
""")
code(r"""
def mondo_names(path=DATA_DIR / 'mondo.obo'):
    names, cur = {}, None
    if not path.exists():
        return names
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if line.startswith('id: MONDO:'):
            cur = line[4:].strip()
        elif line.startswith('name: ') and cur:
            names[cur] = line[6:].strip(); cur = None
    return names


NAMES = mondo_names()
disease = lambda iri: NAMES.get(str(iri).rsplit('/', 1)[-1].replace('_', ':'), str(iri).rsplit('/', 1)[-1])
if len(PEAKS):
    PEAKS['day'] = PEAKS['date'].dt.date
    top = PEAKS.groupby('day').agg(runs=('run', 'nunique'), mean_height=('height', 'mean')).sort_values('runs', ascending=False).head(10)
    show(top.round(3))
    sd = stream.assign(day=stream_dates.dt.date)
    for day in top.index[:5]:
        t = sd[sd['day'].eq(day)]
        print(f'\n{day}: {len(t)} monitored triples, {t.subject.nunique()} subjects')
        show(pd.DataFrame({'subject': t['subject'].str.rsplit('/', n=1).str[-1], 'disease': t['object'].map(disease)}).head(25))
""")

# =====================================================================================================
md(r"""
# Part 8 — Comparison: baseline outlier rate in-sample (ADAZOR) vs held-out

Only on the cases (they have a known change). The same 40 runs again, with one difference: the OCSVM is fitted on a
random half of the reference and the baseline outlier rate is measured on the other half (also after every alarm).
Triples the OCSVM was fitted on are left out of the window's outlier share until they slide out of the window
(right after an alarm the window starts as the new reference), so both sides of the test use only triples the OCSVM has
not seen. Everything else is identical. Compared per run: alarm at the first stream triple, alarms before the switch (false
alarms), delay after the switch, alarms after it, share of the stream tested.

The next cell runs the held-out version (skipped on the full data, or with `NB_HELDOUT=0`) and shows the comparison.
""")
code(r"""
if RUN_HELDOUT and SWITCH:
    run_grid('held_out')
    HELD = load_runs('runs_heldout')
    rows = []
    for name, r in RUNS.items():
        h = HELD.get(name + '__heldout')
        if not h:
            continue
        for label, x in (('in-sample', r), ('held-out', h)):
            idx = list(x['alarms']['stream_index']) if len(x['alarms']) else []
            rows.append({'run': name, 'baseline': label, 'alarm at triple 1': 1 in idx,
                         'alarms before switch': sum(i < SWITCH for i in idx), 'alarms after switch': sum(i >= SWITCH for i in idx),
                         'delay': next((i - SWITCH for i in idx if i >= SWITCH), None),
                         'tested share': round(x['summary']['tested'] / x['summary']['stream_triples'], 3)})
    COMPARE = pd.DataFrame(rows)
    show(COMPARE.groupby('baseline').agg(runs=('run', 'size'), first_triple_alarms=('alarm at triple 1', 'sum'),
         mean_alarms_before=('alarms before switch', 'mean'), mean_alarms_after=('alarms after switch', 'mean'),
         runs_detecting=('delay', lambda x: int(x.notna().sum())), median_delay=('delay', 'median'),
         mean_tested_share=('tested share', 'mean')).round(2))
    show(COMPARE)
else:
    print('comparison not run here (full data, or NB_HELDOUT=0)')
""")

md(r"""
The next cell draws the same run with both baselines side by side, for one model per rule.
""")
code(r"""
if RUN_HELDOUT and SWITCH:
    for rule_name in RULES:
        name = run_name(MODELS[0], COMPOSITIONS[0], rule_name, 'in_sample')
        if name in RUNS and name + '__heldout' in HELD:
            fig, axes = plt.subplots(1, 2, figsize=(12, 2.4), sharey=True)
            plot_run(axes[0], RUNS[name], f'{name}: baseline in-sample (ADAZOR)')
            plot_run(axes[1], HELD[name + '__heldout'], f'{name}: baseline held-out')
            plt.tight_layout(); plt.show()
""")

# =====================================================================================================
md(r"""
# Part 9 — Summary

The next cell collects the main numbers of this execution in one table: data sizes, chosen first-embedding settings,
chosen warm-update epochs, and the alarms per rule.
""")
code(r"""
print('dataset:', DATASET, '| stream:', len(stream), '| switch:', SWITCH)
print('\nfirst embedding (Part 3):'); show(pd.DataFrame(CHOSEN).T)
print('\nwarm-update epochs (Part 4):'); show(pd.DataFrame([{'model': m, 'composition': c, **v} for (m, c), v in WARM.items()]))
if len(RESULTS):
    print('\nalarms per rule (Part 6):')
    cols = ['alarms'] + (['alarms before switch', 'alarms after switch'] if SWITCH else [])
    show(RESULTS.groupby('rule')[cols].agg(['mean', 'min', 'max']).round(1))
""")


def main():
    nb = {'cells': CELLS,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = HERE / 'final_pipeline.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', out, f'({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
