"""Data for Phase 2: the stream, the reference pool, the frozen training graph, MONDO categories.

Uses Phase 1's loader (phase1/drift_sphere.py), so the stream, its order, the cutoff and the
target types are exactly the Phase 1 ones.

Two "variants" of the frozen setup:
  holdout  the embedding is trained on the cutoff graph MINUS the most recent `holdout` stream
           triples; those triples are the reference pool. Used by the channels (C1-C5) and A2.
  full     the embedding is trained on the whole cutoff graph, the reference is Antoine's
           (initial stream minus the last window). Used by A1, the "A0 without re-embedding" ablation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import settings
import drift_sphere as p1  # phase1/drift_sphere.py (configs puts phase1 on sys.path)

POOL_COLUMNS = ['row_number', 'subject', 'edge_type', 'object', 'type', 'event_year', 'part', 'segment']


def load_dataset(name):
    """Phase 1 loading: returns (initial_kg, initial_stream, future_stream, dataset settings)."""
    d = settings.DATASETS[name]
    df = p1.load_temporal_triples(settings.DATA_DIR / d['file'])
    initial_kg, initial_stream, future_stream, _ = p1.build_initial_knowledge_graph(
        df, cutoff_year=d['cutoff_year'], target_type=d['target_type'])
    return initial_kg, initial_stream, future_stream, d


def build_variant(name, variant):
    """Training triples for the frozen embedding and the table of triples to monitor.

    Returns (train_triples: (n, 3) str array, pool: DataFrame). pool has one row per triple that
    can be a reference or stream triple, in stream order, with:
      part     'holdout' / 'initial' (reference side) or 'stream'
      segment  'pre' (before the switch year) or 'post' (from the switch year)
      novel       1 if the subject, object or relation is not in the training graph
      new_entity  1 if the triple brings an entity for the first time (A0's re-embed trigger)
    """
    initial_kg, initial_stream, future_stream, d = load_dataset(name)
    if variant == 'holdout':
        held = initial_stream.tail(min(d['holdout'], len(initial_stream)))
        keep = ~initial_kg['row_number'].isin(set(held['row_number']))
        train_df = initial_kg.loc[keep]
        ref_side = held.assign(part='holdout')
    elif variant == 'full':
        train_df = initial_kg
        ref_side = initial_stream.assign(part='initial')
    else:
        raise ValueError(variant)
    pool = pd.concat([ref_side, future_stream.assign(part='stream')], ignore_index=True)
    switch = d['switch_year']
    post = pool['part'].eq('stream') & (switch is not None) & (pool['event_year'] >= (switch or 0))
    pool['segment'] = np.where(post, 'post', 'pre')
    pool = pool[POOL_COLUMNS].reset_index(drop=True)

    train = train_df[['subject', 'edge_type', 'object']].dropna().astype(str).to_numpy(dtype=object)
    entities = set(train[:, 0]).union(train[:, 2])
    relations = set(train[:, 1])
    pool['novel'] = (~pool['subject'].astype(str).isin(entities) | ~pool['object'].astype(str).isin(entities)
                     | ~pool['edge_type'].astype(str).isin(relations)).astype(int)
    # new_entity: the triple brings an entity for the FIRST time (not in the training graph, not in any
    # earlier pool triple). This is the event that makes A0 re-embed.
    seen, first = set(entities), []
    for s_, o_ in zip(pool['subject'].astype(str), pool['object'].astype(str)):
        first.append(int(s_ not in seen or o_ not in seen))
        seen.update((s_, o_))
    pool['new_entity'] = first
    pool['disease'] = [disease_of(s, o) for s, o in zip(pool['subject'], pool['object'])]
    return train, pool


def disease_of(subject, obj):
    """The MONDO term of a triple (object first, as in 'Gene - causes... - Disease'), as MONDO:nnnnnnn."""
    for x in (obj, subject):
        m = re.search(r'MONDO_(\d+)', str(x))
        if m:
            return f'MONDO:{m.group(1)}'
    return ''


# ---------------------------------------------------------------------------
# Reference / stream splits (index arrays into the pool)
# ---------------------------------------------------------------------------

def real_order_split(pool, variant, rng):
    """Real stream order. Reference = last REFERENCE['size'] reference-side triples, shuffled before
    the fit/calibration split (triples from one paper arrive together). Stream = the future stream."""
    ref_side = np.flatnonzero(pool['part'].ne('stream').to_numpy())
    ref = ref_side[-settings.REFERENCE['size']:]
    ref = rng.permutation(ref)
    stream = np.flatnonzero(pool['part'].eq('stream').to_numpy())
    return ref[:settings.REFERENCE['fit']], ref[settings.REFERENCE['fit']:], stream, switch_index(pool, stream)


def switch_index(pool, stream):
    seg = pool['segment'].to_numpy()[stream]
    post = np.flatnonzero(seg == 'post')
    return int(post[0]) if len(post) else None


def pre_pool(pool):
    """Triples of the pre-switch topic that the holdout embedding never saw."""
    return np.flatnonzero(pool['segment'].eq('pre').to_numpy())


def control_split(pool, rng):
    """No-drift control: shuffle the pre-switch pool; first 500 = reference, the rest = stream."""
    p = rng.permutation(pre_pool(pool))
    n_ref, n_fit = settings.REFERENCE['size'], settings.REFERENCE['fit']
    return p[:n_fit], p[n_fit:n_ref], p[n_ref:], None


def inject_split(pool, rng):
    """Known switch at a random position: reference + k pre-switch triples, then post-switch triples."""
    p = rng.permutation(pre_pool(pool))
    n_ref, n_fit = settings.REFERENCE['size'], settings.REFERENCE['fit']
    rest = p[n_ref:]
    k = int(rng.integers(settings.WINDOW, max(settings.WINDOW + 1, len(rest) + 1)))
    post = rng.permutation(np.flatnonzero(pool['segment'].eq('post').to_numpy()))[:settings.MONTE_CARLO['post_len']]
    return p[:n_fit], p[n_fit:n_ref], np.concatenate([rest[:k], post]), k


# ---------------------------------------------------------------------------
# MONDO categories (C3)
# ---------------------------------------------------------------------------

def read_mondo(path):
    """Minimal OBO reader: {term: [is_a parents]}, {term: name}, data-version."""
    parents, names, version, term = {}, {}, None, None
    obsolete = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('data-version:') and version is None:
                version = line.split(':', 1)[1].strip()
            elif line == '[Term]':
                term = None
            elif line.startswith('[') and line.endswith(']'):
                term = '__skip__'
            elif line.startswith('id: ') and term is None:
                term = line[4:].strip()
                parents.setdefault(term, [])
            elif term and term != '__skip__':
                if line.startswith('name: '):
                    names[term] = line[6:].strip()
                elif line.startswith('is_a: '):
                    parents[term].append(line[6:].split('!')[0].strip().split(' ')[0])
                elif line.startswith('is_obsolete: true'):
                    obsolete.add(term)
    return parents, names, version, obsolete


def mondo_categories(diseases):
    """Multi-hot matrix (n, K) of top-level MONDO categories for each disease, plus names and version.

    Categories = terms MONDO['depth'] levels below MONDO['root'] ("human disease"); at depth 2 these
    are groups such as 'cancer or benign tumor' or 'disease of primarily extrinsic mechanism'
    (infections). A disease belongs to every category among its ancestors (MONDO allows several
    parents). 'unmapped' = not found / obsolete. Categories no triple belongs to are dropped.
    Returns None if the ontology file is missing.
    """
    path = Path(settings.MONDO['file'])
    if not path.exists():
        return None
    parents, names, version, obsolete = read_mondo(path)
    root = settings.MONDO['root'] if settings.MONDO['root'] in parents else settings.MONDO['fallback_root']
    level = {root}
    for _ in range(settings.MONDO['depth']):
        level = {t for t, ps in parents.items() if t not in obsolete and any(p in level for p in ps)}
    cats = sorted(level)
    cat_pos = {c: i for i, c in enumerate(cats)}
    memo = {}

    def ancestors(t):
        if t in memo:
            return memo[t]
        seen, stack = set(), [t]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents.get(x, []))
        memo[t] = seen
        return seen

    unique = sorted(set(diseases))
    rows = {}
    for dis in unique:
        hits = [cat_pos[c] for c in ancestors(dis) if c in cat_pos] if dis in parents and dis not in obsolete else []
        rows[dis] = hits
    K = len(cats) + 1  # last column = unmapped
    M = np.zeros((len(diseases), K), dtype=np.int8)
    for i, dis in enumerate(diseases):
        hits = rows.get(dis, [])
        if hits:
            M[i, hits] = 1
        else:
            M[i, -1] = 1
    labels = [f'{c} {names.get(c, "")}'.strip() for c in cats] + ['unmapped']
    used = M.any(axis=0)
    return {'matrix': M[:, used], 'labels': [lab for lab, u in zip(labels, used) if u],
            'version': version, 'root': root}


# ---------------------------------------------------------------------------
# Paths and small IO helpers
# ---------------------------------------------------------------------------

def feature_dir(dataset):
    p = settings.FEATURES_DIR / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def pool_path(dataset, variant):
    return feature_dir(dataset) / f'pool__{variant}.tsv.gz'


def kge_features_path(dataset, model, variant, seed=None):
    seed = settings.KGE['seed'] if seed is None else seed
    return feature_dir(dataset) / f'{model}__{variant}__seed{seed}.npz'


def categories_path(dataset):
    return feature_dir(dataset) / 'categories__holdout.npz'


def load_pool(dataset, variant):
    return pd.read_csv(pool_path(dataset, variant), sep='\t', keep_default_na=False,
                       na_values={'event_year': ['']})


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
