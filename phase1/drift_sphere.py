"""Phase 1 experiment: Antoine's sphere-rule OCSVM + Z-test drift detector, for 5 PyKEEN models.

This is a port of `biomedical_ocsvm_ztest_experiment (6).ipynb` (Antoine). Function names,
the stream loop, the sphere (center-radius) retrain rule, keep-10% after drift and the
Z-test are the notebook's. What changed, and why:

  1. Model: TransE -> any of TransE, TransH, DistMult, ComplEx, RotatE (`--model`).
     Complex models (ComplEx, RotatE) are turned into real vectors [real parts, imag parts].
  2. Settings come from a preset (configs.PRESETS): 'antoine' = the notebook's constants,
     'phase1' = 64 real dims, 10 epochs, StandardScaler + gamma='scale'.
     Retrain policy on a new entity: 'sphere' (Antoine, ratio > k) or 'always' (Diana's reference).
  3. Speed, same numbers: the KG is appended to instead of re-concatenated every step,
     features and the KG bounding box are computed as numpy arrays instead of row loops,
     and PyKEEN trains on the GPU when there is one.
  4. Input: case files call the relation column `predicate`; it is renamed to `edge_type`.
     `target_type` may be a list of types.
  5. Extra logging for the analysis (stream index, event year, re-embed / retrain flags,
     embedding time). The final KG is not saved (it is the input file up to the end).

Retrain rule (notebook): when a triple brings an unseen entity or relation, the KG is
re-embedded from scratch; the new triple's feature vector is compared with the box of all
KG triple features under the previous embedding; the OCSVM is retrained if
distance_to_center / radius > sphere_k (notebook: sphere_k = 1).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402


# ---------------------------------------------------------------------------
# Detector (notebook cell 1): adazor's ZTestDriftDetector, with the notebook's fallback.
# ---------------------------------------------------------------------------

class FallbackZTestDriftDetector:
    """Same reset/detect as adazor.detectors.ZTestDriftDetector (copied from the notebook)."""

    def __init__(self, alpha=0.05):
        self.alpha = float(alpha)
        self.is_built = False
        self.tot_orig = 0
        self.n_orig = 0

    def reset(self, outliers):
        outs = np.fromiter((bool(o) for o in outliers), dtype=bool)
        self.tot_orig = int(outs.sum())
        self.n_orig = int(len(outs))
        if self.n_orig == 0:
            raise ValueError('reset requires a non-empty baseline window')
        self.is_built = True

    def detect(self, outliers):
        import scipy.stats as st
        if not self.is_built:
            raise RuntimeError('ZTestDriftDetector has not been reset.')
        outs = np.fromiter((bool(o) for o in outliers), dtype=bool)
        n = int(len(outs))
        if n == 0:
            return False
        p_outliers = int(outs.sum()) / n
        p_orig = self.tot_orig / self.n_orig
        threshold = st.norm().ppf(1 - self.alpha / 2) / math.sqrt(2 * n)
        return bool(abs(p_outliers - p_orig) > threshold)


def make_detector(kind, alpha, adazor_src):
    if adazor_src and str(adazor_src) not in sys.path:
        sys.path.append(str(adazor_src))
    try:
        from adazor.detectors import BiomedicalTwoSampleZTestDriftDetector, ZTestDriftDetector
        source = 'adazor'
    except Exception:
        ZTestDriftDetector, BiomedicalTwoSampleZTestDriftDetector = FallbackZTestDriftDetector, None
        source = 'fallback'
    if kind == 'ztest':
        return ZTestDriftDetector(alpha=alpha), source
    if kind == 'biomedical':
        if BiomedicalTwoSampleZTestDriftDetector is None:
            raise RuntimeError('BiomedicalTwoSampleZTestDriftDetector needs the adazor package (--adazor-src).')
        return BiomedicalTwoSampleZTestDriftDetector(alpha=alpha), source
    raise ValueError(f'unknown detector {kind}')


# ---------------------------------------------------------------------------
# Progress file (notebook's write_progress), throttled so it does not slow the loop.
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self, run_dir, total, label):
        self.json_path = run_dir / 'progress.json'
        self.log_path = run_dir / 'progress.log'
        self.total = total
        self.label = label
        self._last_json = 0.0
        self._counts = (0, 0, 0)

    def write(self, stage, iteration, drifts=None, refreshes=None, unseen_all=None, detail='', force=False):
        # Calls without counts (e.g. 're-embedding started') reuse the last known counts.
        if drifts is None:
            drifts, refreshes, unseen_all = self._counts
        else:
            self._counts = (drifts, refreshes, unseen_all)
        now = time.time()
        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'run': self.label,
            'stage': stage,
            'detail': detail,
            'iteration': int(iteration),
            'total': int(self.total),
            'percent': round(100.0 * iteration / self.total, 2) if self.total else None,
            'drifts': int(drifts),
            'refreshes': int(refreshes),
            'unseen_all': int(unseen_all),
        }
        routine = stage in ('monitor', 'warmup', 'rebuild')
        if force or not routine or now - self._last_json > 5:
            self.json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self._last_json = now
        if force or not routine or iteration % 500 == 0:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(payload) + '\n')


# ---------------------------------------------------------------------------
# Data (notebook cell 2)
# ---------------------------------------------------------------------------

def read_table_auto(path, usecols=None, nrows=None):
    sep = '\t' if path.suffix.lower() == '.tsv' else ','
    return pd.read_csv(path, sep=sep, usecols=usecols, nrows=nrows)


def infer_event_year(df):
    # Prefer explicit year, otherwise derive a year from timestamp text (notebook logic).
    if 'year' in df.columns:
        year = pd.to_numeric(df['year'], errors='coerce')
    else:
        year = pd.Series(np.nan, index=df.index, dtype='float64')
    timestamp_year = pd.to_numeric(df['timestamp'].astype(str).str.extract(r'(\d{4})', expand=False), errors='coerce')
    return year.fillna(timestamp_year)


def sort_temporal_triples(df):
    # Static rows first, then future rows by event time and row order (notebook logic).
    numeric_ts = pd.to_numeric(df['timestamp'].where(df['timestamp'].ne(''), '0'), errors='coerce').fillna(0)
    sorted_df = df.assign(_missing_ts=df['timestamp'].eq(''), _numeric_ts=numeric_ts).sort_values(
        by=['_missing_ts', 'event_year', '_numeric_ts', 'row_number'],
        ascending=[False, True, True, True],
        kind='stable',
    )
    sorted_df = sorted_df.drop(columns=['_missing_ts', '_numeric_ts']).reset_index(drop=True)
    sorted_df['stream_position'] = np.arange(1, len(sorted_df) + 1)
    return sorted_df


def load_temporal_triples(path):
    sample = read_table_auto(path, nrows=0)
    relation_col = 'edge_type' if 'edge_type' in sample.columns else 'predicate'
    required_cols = ['subject', 'object', relation_col, 'timestamp']
    optional_cols = [c for c in ['year', 'month', 'day', 'type'] if c in sample.columns]
    df = read_table_auto(path, usecols=required_cols + optional_cols)
    if relation_col != 'edge_type':
        df = df.rename(columns={relation_col: 'edge_type'})
    df['row_number'] = np.arange(1, len(df) + 1)
    df['timestamp'] = df['timestamp'].fillna('').astype(str)
    if 'type' not in df.columns:
        df['type'] = ''
    df['event_year'] = infer_event_year(df)
    return sort_temporal_triples(df)


def _type_mask(df, target_type):
    if isinstance(target_type, (list, tuple, set)):
        return df['type'].isin(list(target_type))
    return df['type'].eq(target_type)


def build_initial_knowledge_graph(triples_df, cutoff_year, target_type=None):
    # Initial KG uses only triples known through the cutoff year (undated rows included).
    initial_mask = triples_df['event_year'].isna() | (triples_df['event_year'] <= cutoff_year)
    initial_kg = triples_df.loc[initial_mask].copy().reset_index(drop=True)
    future_df = triples_df.loc[~initial_mask].copy().reset_index(drop=True)
    if target_type is None:
        initial_stream = initial_kg.copy().reset_index(drop=True)
        future_stream = future_df.copy().reset_index(drop=True)
    else:
        initial_stream = initial_kg[_type_mask(initial_kg, target_type)].copy().reset_index(drop=True)
        future_stream = future_df[_type_mask(future_df, target_type)].copy().reset_index(drop=True)
    lookup = build_lookup_sets(initial_kg)
    return initial_kg, initial_stream, future_stream, lookup


def build_lookup_sets(kg_df):
    return {
        'nodes': set(kg_df['subject']).union(set(kg_df['object'])),
        'edge_types': set(kg_df['edge_type']),
    }


def triple_has_unseen_elements(row, lookup):
    return (
        row['subject'] not in lookup['nodes']
        or row['object'] not in lookup['nodes']
        or row['edge_type'] not in lookup['edge_types']
    )


class KnowledgeGraph:
    """The notebook's kg_df (initial KG + every streamed triple, in order), kept as arrays.

    The notebook re-concatenated a DataFrame on every triple; this keeps the same rows in
    the same order and only builds the array when an embedding is trained.
    """

    def __init__(self, initial_kg_df):
        base = initial_kg_df[['subject', 'edge_type', 'object']].dropna().astype(str)
        self._base = base.to_numpy(dtype=object)
        self._added = []

    def add(self, row):
        values = (row['subject'], row['edge_type'], row['object'])
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
            return
        self._added.append(tuple(str(v) for v in values))

    def __len__(self):
        return len(self._base) + len(self._added)

    def triples(self):
        if self._added:
            return np.vstack([self._base, np.asarray(self._added, dtype=object)])
        return self._base


def update_knowledge_graph(kg, row, lookup):
    # Add one incoming temporal triple to the current KG (notebook logic, one row at a time).
    kg.add(row)
    lookup['nodes'].add(row['subject'])
    lookup['nodes'].add(row['object'])
    lookup['edge_types'].add(row['edge_type'])
    return kg, lookup


# ---------------------------------------------------------------------------
# Embeddings (notebook cell 3, generalised to 5 models)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingState:
    entity_index: dict
    entity_matrix: np.ndarray      # (n_entities, d), rows L2-normalised
    relation_index: dict
    relation_matrix: np.ndarray    # (n_relations, d), rows L2-normalised
    dim: int
    kg_rows: int
    train_seconds: float


def normalize_rows(matrix):
    # Same as the notebook's normalize_vector, applied to every row (zero rows stay zero).
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _representation_to_numpy(representations):
    import torch
    representation = representations
    if not hasattr(representation, '_embeddings') and hasattr(representation, '__getitem__'):
        representation = representation[0]
    with torch.no_grad():
        x = representation(indices=None)
    if torch.is_complex(x):
        x = torch.cat([x.real, x.imag], dim=-1)
    return x.reshape(x.shape[0], -1).detach().cpu().numpy().astype(np.float64)


def _predicate_embeddings_from_translation(triples, entity_index, entity_matrix, relation_labels):
    # Notebook fallback: mean (object - subject) per relation, normalised.
    out = np.zeros((len(relation_labels), entity_matrix.shape[1]), dtype=np.float64)
    rel_pos = {r: i for i, r in enumerate(relation_labels)}
    s = entity_matrix[[entity_index[x] for x in triples[:, 0]]]
    o = entity_matrix[[entity_index[x] for x in triples[:, 2]]]
    diffs = pd.DataFrame(o - s)
    diffs['rel'] = triples[:, 1]
    means = diffs.groupby('rel').mean()
    for rel, vec in means.iterrows():
        out[rel_pos[rel]] = vec.to_numpy()
    return normalize_rows(out)


def generate_embeddings_from_kg(kg, cfg):
    """Train a PyKEEN model from scratch on the current KG (notebook: generate_transe_embeddings_from_kg)."""
    if cfg['embedder'] == 'random':
        return _random_embeddings(kg, cfg)

    import pykeen.models as pykeen_models
    import torch
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    started = time.time()
    triples = kg.triples()
    if len(triples) == 0:
        raise ValueError('Cannot generate embeddings from an empty KG.')
    triples_factory = TriplesFactory.from_labeled_triples(triples.astype(str))

    model_name = cfg['model']
    dim = cfg['real_dim'] // 2 if model_name in configs.COMPLEX_MODELS else cfg['real_dim']
    model_cls = getattr(pykeen_models, model_name)
    model = model_cls(triples_factory=triples_factory, embedding_dim=dim, random_seed=cfg['seed'])
    model = model.to(torch.device(cfg['device']))
    training_loop = SLCWATrainingLoop(model=model, triples_factory=triples_factory)
    training_loop.train(
        triples_factory=triples_factory,
        num_epochs=cfg['epochs'],
        batch_size=min(cfg['batch_size'], max(32, len(triples))),
        use_tqdm=False,
        use_tqdm_batch=False,
        pin_memory=False,
        num_workers=0,
    )

    entity_labels = list(triples_factory.entity_to_id.keys())
    relation_labels = list(triples_factory.relation_to_id.keys())
    entity_ids = [triples_factory.entity_to_id[x] for x in entity_labels]
    relation_ids = [triples_factory.relation_to_id[x] for x in relation_labels]
    entity_matrix = normalize_rows(_representation_to_numpy(model.entity_representations)[entity_ids])
    entity_index = {label: i for i, label in enumerate(entity_labels)}
    try:
        relation_matrix = normalize_rows(_representation_to_numpy(model.relation_representations)[relation_ids])
        if relation_matrix.shape[1] != entity_matrix.shape[1]:
            raise ValueError('relation dim differs from entity dim')
    except Exception:
        relation_matrix = _predicate_embeddings_from_translation(triples.astype(str), entity_index, entity_matrix, relation_labels)

    del model, training_loop
    if cfg['device'].startswith('cuda'):
        torch.cuda.empty_cache()
    return EmbeddingState(
        entity_index=entity_index,
        entity_matrix=entity_matrix,
        relation_index={label: i for i, label in enumerate(relation_labels)},
        relation_matrix=relation_matrix,
        dim=entity_matrix.shape[1],
        kg_rows=len(kg),
        train_seconds=time.time() - started,
    )


def _random_embeddings(kg, cfg):
    """Smoke-test embedder: random unit vectors, new ones on every call (like a from-scratch re-embed)."""
    triples = kg.triples().astype(str)
    rng = np.random.default_rng(cfg['seed'] + len(kg))
    entities = sorted(set(triples[:, 0]).union(triples[:, 2]))
    relations = sorted(set(triples[:, 1]))
    d = cfg['real_dim']
    return EmbeddingState(
        entity_index={e: i for i, e in enumerate(entities)},
        entity_matrix=normalize_rows(rng.normal(size=(len(entities), d))),
        relation_index={r: i for i, r in enumerate(relations)},
        relation_matrix=normalize_rows(rng.normal(size=(len(relations), d))),
        dim=d,
        kg_rows=len(kg),
        train_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# Features, OCSVM, sphere boundary (notebook cell 4)
# ---------------------------------------------------------------------------

def compose(subjects, edge_types, objects, state, composition):
    s = state.entity_matrix[[state.entity_index[str(x)] for x in subjects]]
    p = state.relation_matrix[[state.relation_index[str(x)] for x in edge_types]]
    o = state.entity_matrix[[state.entity_index[str(x)] for x in objects]]
    if composition == 'hadamard':
        return s * p * o
    if composition == 'concatenation':
        return np.concatenate([s, p, o], axis=1)
    raise ValueError(composition)


def rows_to_features(rows, state, composition):
    # Notebook: rows_to_feature_frame (one triple -> one feature vector), as a matrix.
    return compose([r['subject'] for r in rows], [r['edge_type'] for r in rows], [r['object'] for r in rows], state, composition)


def feature_dim(cfg):
    return cfg['real_dim'] * (3 if cfg['composition'] == 'concatenation' else 1)


def train_classifier(X_ref, cfg):
    scaler = StandardScaler() if cfg['standardize'] else None
    X = scaler.fit_transform(X_ref) if scaler is not None else X_ref
    ocsvm = OneClassSVM(kernel='rbf', nu=cfg['nu'], gamma=cfg['gamma'])
    ocsvm.fit(X)
    flags = ocsvm.predict(X) == -1  # OCSVM predicts -1 for outliers
    return ocsvm, scaler, flags


def outlier_flags(ocsvm, scaler, X):
    if scaler is not None:
        X = scaler.transform(X)
    return ocsvm.predict(X) == -1


def retrain_reference_model(reference_rows, state, cfg, detector):
    X_ref = rows_to_features(reference_rows, state, cfg['composition'])
    ocsvm, scaler, flags = train_classifier(X_ref, cfg)
    detector.reset(flags)
    return X_ref, ocsvm, scaler, flags


def boundary_from_min_max(min_point, max_point):
    if min_point is None or max_point is None:
        return None
    center = 0.5 * (min_point + max_point)
    diagonal_length = float(np.linalg.norm(max_point - min_point))
    return {
        'min_point': min_point.copy(),
        'max_point': max_point.copy(),
        'center': center,
        'radius': diagonal_length / 2.0,
        'diagonal_length': diagonal_length,
    }


def compute_feature_space_boundary_from_kg(kg, state, composition, chunk_size=200_000):
    # Min/max box of all KG triple features under the current embedding (notebook logic, chunked).
    triples = kg.triples()
    min_point = max_point = None
    for start in range(0, len(triples), chunk_size):
        part = triples[start:start + chunk_size]
        X = compose(part[:, 0], part[:, 1], part[:, 2], state, composition)
        lo, hi = X.min(axis=0), X.max(axis=0)
        min_point = lo if min_point is None else np.minimum(min_point, lo)
        max_point = hi if max_point is None else np.maximum(max_point, hi)
    return boundary_from_min_max(min_point, max_point)


def update_boundary_incrementally(boundary, point_vec):
    if boundary is None:
        return boundary_from_min_max(point_vec, point_vec)
    new_min = np.minimum(boundary['min_point'], point_vec)
    new_max = np.maximum(boundary['max_point'], point_vec)
    if np.array_equal(new_min, boundary['min_point']) and np.array_equal(new_max, boundary['max_point']):
        return boundary
    return boundary_from_min_max(new_min, new_max)


def compute_center_radius_metrics(point_vec, boundary, sphere_k):
    if boundary is None:
        return {'distance_to_center': np.nan, 'center_radius': np.nan, 'center_diagonal_length': np.nan,
                'center_radius_ratio': np.nan, 'threshold_exceeded_center_radius': False}
    distance_to_center = float(np.linalg.norm(point_vec - boundary['center']))
    radius = float(boundary['radius'])
    if radius == 0.0:
        ratio = np.inf if distance_to_center > 0 else 0.0
    else:
        ratio = distance_to_center / radius
    return {
        'distance_to_center': distance_to_center,
        'center_radius': radius,
        'center_diagonal_length': float(boundary['diagonal_length']),
        'center_radius_ratio': float(ratio),
        'threshold_exceeded_center_radius': bool(np.isfinite(ratio) and ratio > sphere_k),
    }


# ---------------------------------------------------------------------------
# Stream (notebook cell 5)
# ---------------------------------------------------------------------------

def _row_meta(row, i, iteration):
    return {
        'iteration': int(iteration),
        'stream_index': int(i),
        'stream_position': int(row['stream_position']),
        'row_number': int(row['row_number']),
        'timestamp': row['timestamp'],
        'event_year': float(row['event_year']) if pd.notna(row['event_year']) else np.nan,
        'type': row['type'],
    }


def process_triple_for_features(row, kg, lookup, state, boundary, cfg, progress, iteration):
    """Notebook logic: keep the boundary live while embeddings stay fixed; on an unseen element,
    re-embed, compare the new triple (new embedding) with the saved boundary (old embedding),
    then rebuild the boundary from the whole KG under the new embedding."""
    has_unseen = triple_has_unseen_elements(row, lookup)
    if not has_unseen:
        vec = rows_to_features([row], state, cfg['composition'])[0]
        kg, lookup = update_knowledge_graph(kg, row, lookup)
        boundary = update_boundary_incrementally(boundary, vec)
        return vec, kg, lookup, state, boundary, None, False

    pre_unseen_boundary = boundary
    kg, lookup = update_knowledge_graph(kg, row, lookup)
    progress.write('reembedding_unseen_start', iteration, detail='training on updated KG')
    state = generate_embeddings_from_kg(kg, cfg)
    vec = rows_to_features([row], state, cfg['composition'])[0]
    shift_record = {'trigger_reason': 'unseen_element', 'is_drift': False, 'retrained_ocsvm': False,
                    'embedding_kg_rows': int(len(kg)), 'train_seconds': state.train_seconds}
    shift_record.update(compute_center_radius_metrics(vec, pre_unseen_boundary, cfg['sphere_k']))
    t0 = time.time()
    boundary = compute_feature_space_boundary_from_kg(kg, state, cfg['composition'])
    shift_record['boundary_seconds'] = time.time() - t0
    return vec, kg, lookup, state, boundary, shift_record, True


def rebuild_reference_after_drift(reference_rows, kg, state, cfg, drift_row, progress, iteration):
    progress.write('reembedding_drift_start', iteration, detail='training after drift')
    state = generate_embeddings_from_kg(kg, cfg)
    shift_record = {'trigger_reason': 'drift', 'is_drift': True, 'retrained_ocsvm': True,
                    'embedding_kg_rows': int(len(kg)), 'train_seconds': state.train_seconds}
    drift_vec = rows_to_features([drift_row], state, cfg['composition'])[0]
    boundary = compute_feature_space_boundary_from_kg(kg, state, cfg['composition'])
    shift_record.update(compute_center_radius_metrics(drift_vec, boundary, cfg['sphere_k']))
    return state, boundary, shift_record


def should_retrain(shift_record, cfg):
    policy = cfg['retrain_policy']
    if policy == 'sphere':
        return bool(shift_record['threshold_exceeded_center_radius'])
    if policy == 'always':
        return True
    if policy == 'never':
        return False
    raise ValueError(policy)


def run_stream(initial_kg, initial_stream, future_stream, lookup, cfg, run_dir, detector):
    if initial_stream.empty:
        raise ValueError('No initial training triples for the target type(s).')
    if future_stream.empty:
        raise ValueError('No future stream triples for the target type(s).')
    if len(initial_stream) < 2:
        raise ValueError('Need at least 2 initial triples.')

    window_size = cfg['window_size']
    future_records = future_stream.to_dict('records')
    if cfg.get('max_stream'):
        future_records = future_records[:cfg['max_stream']]
    total = len(future_records)
    progress = Progress(run_dir, total, run_dir.name)

    kg = KnowledgeGraph(initial_kg)
    lookup = {'nodes': set(lookup['nodes']), 'edge_types': set(lookup['edge_types'])}
    progress.write('initial_embedding_start', 0, detail='training initial embedding from cutoff KG', force=True)
    state = generate_embeddings_from_kg(kg, cfg)
    initial_embedding_seconds = state.train_seconds
    boundary = compute_feature_space_boundary_from_kg(kg, state, cfg['composition'])
    progress.write('initial_embedding_done', 0, detail=f'{initial_embedding_seconds:.1f}s', force=True)

    seed_window_size = min(window_size, len(initial_stream) - 1)
    initial_rows = initial_stream.to_dict('records')
    reference_rows = initial_rows[:-seed_window_size]
    initial_window_rows = initial_rows[-seed_window_size:]
    X_init_ref = rows_to_features(reference_rows, state, cfg['composition'])
    ocsvm, scaler, reference_flags = train_classifier(X_init_ref, cfg)
    initial_reference_flags = reference_flags.copy()
    detector.reset(reference_flags)

    current_window = deque(initial_window_rows, maxlen=window_size)
    shift_records, step_records, window_history, drift_records, drift_windows = [], [], [], [], []
    threshold_retrain_count = 0
    unseen_elements_count_all = 0
    i = 0
    iteration = 0

    def handle_shift(row, shift_record):
        nonlocal ocsvm, scaler, reference_flags, threshold_retrain_count
        retrained = False
        if shift_record is not None and should_retrain(shift_record, cfg):
            _, ocsvm, scaler, reference_flags = retrain_reference_model(reference_rows, state, cfg, detector)
            shift_record['retrained_ocsvm'] = True
            threshold_retrain_count += 1
            retrained = True
        if shift_record is not None:
            shift_record.update(_row_meta(row, i, iteration))
            shift_record.update({'subject': row['subject'], 'edge_type': row['edge_type'], 'object': row['object']})
            shift_records.append(shift_record)
        return retrained

    # Warm-up (only runs if the initial stream was shorter than the window).
    while len(current_window) < window_size and i < total:
        row = future_records[i]
        i += 1
        iteration += 1
        _, kg, lookup, state, boundary, shift_record, has_unseen = process_triple_for_features(
            row, kg, lookup, state, boundary, cfg, progress, iteration)
        unseen_elements_count_all += int(has_unseen)
        retrained = handle_shift(row, shift_record)
        current_window.append(row)
        step_records.append({**_row_meta(row, i, iteration), 'drift_tested': False, 'drift_detected': False,
                             'reembedded': has_unseen, 'sphere_retrained': retrained,
                             'reference_outlier_fraction': np.nan, 'current_outlier_fraction': np.nan,
                             'embedding_kg_rows': int(state.kg_rows)})
        progress.write('warmup', i, 0, len(shift_records), unseen_elements_count_all)

    first_drift_detected = False
    while i < total:
        row = future_records[i]
        i += 1
        iteration += 1
        _, kg, lookup, state, boundary, shift_record, has_unseen = process_triple_for_features(
            row, kg, lookup, state, boundary, cfg, progress, iteration)
        unseen_elements_count_all += int(has_unseen)
        retrained = handle_shift(row, shift_record)
        current_window.append(row)

        window_rows = list(current_window)
        X_curr = rows_to_features(window_rows, state, cfg['composition'])
        current_flags = outlier_flags(ocsvm, scaler, X_curr)
        is_drift = bool(detector.detect(current_flags))
        current_fraction = float(np.mean(current_flags)) if len(current_flags) else 0.0
        reference_fraction = float(reference_flags.mean()) if len(reference_flags) else 0.0

        meta = _row_meta(row, i, iteration)
        history = {**meta, 'window_size': len(window_rows), 'reference_size': int(len(reference_flags)),
                   'reference_outlier_fraction': reference_fraction, 'current_outlier_fraction': current_fraction,
                   'drift_detected': is_drift, 'reembedded': has_unseen, 'sphere_retrained': retrained,
                   'center_radius_ratio': shift_record['center_radius_ratio'] if shift_record else np.nan,
                   'embedding_kg_rows': int(state.kg_rows)}
        window_history.append(history)
        step_records.append({**history, 'subject': row['subject'], 'edge_type': row['edge_type'],
                             'object': row['object'], 'drift_tested': True})
        progress.write('monitor', i, len(drift_records), len(shift_records), unseen_elements_count_all)

        if not is_drift:
            continue

        drift_event_id = len(drift_records) + 1
        if cfg['save_drift_windows']:
            snap = pd.DataFrame(X_curr, columns=[f'f{k}' for k in range(X_curr.shape[1])])
            snap.insert(0, 'row_number', [r['row_number'] for r in window_rows])
            snap['outlier_flag'] = current_flags.astype(int)
            snap['drift_event_id'] = drift_event_id
            drift_windows.append(snap)
        drift_records.append({**meta, 'drift_event_id': drift_event_id, 'window_size': len(window_rows),
                              'reference_size': int(len(reference_flags)),
                              'reference_outlier_fraction': reference_fraction,
                              'current_outlier_fraction': current_fraction,
                              'first_drift_phase': not first_drift_detected,
                              'embedding_kg_rows': int(state.kg_rows)})
        first_drift_detected = True

        # Keep 10% of the drifted window, refill from the stream (no re-embed during refill).
        keep_count = max(1, int(round(window_size * cfg['keep_fraction_after_drift'])))
        rebuilt_rows = window_rows[-min(keep_count, len(window_rows)):]
        while len(rebuilt_rows) < window_size and i < total:
            next_row = future_records[i]
            i += 1
            unseen_elements_count_all += int(triple_has_unseen_elements(next_row, lookup))
            kg, lookup = update_knowledge_graph(kg, next_row, lookup)
            rebuilt_rows.append(next_row)
            progress.write('rebuild', i, len(drift_records), len(shift_records), unseen_elements_count_all)

        reference_rows = rebuilt_rows
        state, boundary, drift_shift = rebuild_reference_after_drift(reference_rows, kg, state, cfg, row, progress, iteration)
        drift_shift.update(_row_meta(row, i, iteration))
        drift_shift.update({'subject': row['subject'], 'edge_type': row['edge_type'], 'object': row['object']})
        shift_records.append(drift_shift)
        current_window = deque(reference_rows[-min(window_size, len(reference_rows)):], maxlen=window_size)
        _, ocsvm, scaler, reference_flags = retrain_reference_model(reference_rows, state, cfg, detector)

    progress.write('finished', i, len(drift_records), len(shift_records), unseen_elements_count_all, force=True)

    shift_df = pd.DataFrame(shift_records)
    unseen = shift_df[shift_df['trigger_reason'].eq('unseen_element')] if not shift_df.empty else shift_df
    n_unseen_refresh = int(len(unseen))
    n_unseen_retrain = int(unseen['retrained_ocsvm'].sum()) if n_unseen_refresh else 0
    summary = {
        'initial_kg_rows': int(len(initial_kg)),
        'initial_stream_rows': int(len(initial_stream)),
        'initial_reference_size': int(len(X_init_ref)),
        'initial_reference_outlier_fraction': float(initial_reference_flags.mean()),
        'initial_window_seed_size': int(len(initial_window_rows)),
        'future_rows': int(len(future_stream)),
        'processed_future_triples': int(i),
        'tested_steps': int(len(window_history)),
        'num_detected_drifts': int(len(drift_records)),
        'first_drift_stream_index': int(drift_records[0]['stream_index']) if drift_records else None,
        'first_drift_year': drift_records[0]['event_year'] if drift_records else None,
        'unseen_elements_count_all': int(unseen_elements_count_all),
        'num_embedding_refresh_events': int(len(shift_df)),
        'num_unseen_refresh_events': n_unseen_refresh,
        'num_drift_refresh_events': int(len(shift_df) - n_unseen_refresh),
        'num_threshold_retrains': int(threshold_retrain_count),
        'unseen_retrain_ratio': float(n_unseen_retrain / n_unseen_refresh) if n_unseen_refresh else 0.0,
        'num_center_radius_outside_boundary': int(unseen['threshold_exceeded_center_radius'].sum()) if n_unseen_refresh else 0,
        'initial_embedding_seconds': float(initial_embedding_seconds),
        'total_embedding_seconds': float(initial_embedding_seconds + (shift_df['train_seconds'].sum() if not shift_df.empty else 0.0)),
    }
    return {
        'summary': summary,
        'drifts': pd.DataFrame(drift_records),
        'window_history': pd.DataFrame(window_history),
        'step_trace': pd.DataFrame(step_records),
        'embedding_shift_metrics': shift_df,
        'drift_windows': pd.concat(drift_windows, ignore_index=True) if drift_windows else pd.DataFrame(),
        'initial_reference_flags': initial_reference_flags,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(args):
    cfg = dict(configs.PRESETS[args.preset])
    for key in ('real_dim', 'epochs', 'batch_size', 'seed', 'window_size', 'nu', 'alpha', 'keep_fraction_after_drift'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.gamma is not None:
        cfg['gamma'] = args.gamma if args.gamma in ('scale', 'auto') else float(args.gamma)
    if args.standardize is not None:
        cfg['standardize'] = args.standardize == 'yes'
    dataset = configs.DATASETS[args.dataset]
    if args.model in configs.COMPLEX_MODELS and cfg['real_dim'] % 2:
        raise ValueError('real_dim must be even for complex models')
    if args.device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    else:
        device = args.device
    cfg.update({
        'dataset': args.dataset,
        'data_file': dataset['file'],
        'cutoff_year': args.cutoff_year if args.cutoff_year is not None else dataset['cutoff_year'],
        'target_type': dataset['target_type'],
        'switch_year': dataset['switch_year'],
        'model': args.model,
        'composition': args.composition,
        'preset': args.preset,
        'retrain_policy': args.retrain_policy,
        'sphere_k': args.sphere_k,
        'detector': args.detector,
        'embedder': args.embedder,
        'device': device,
        'max_stream': args.max_stream if args.max_stream is not None else dataset.get('max_stream'),
        'save_drift_windows': not args.no_drift_windows,
    })
    return cfg


def versions():
    out = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'sklearn', 'scipy', 'torch', 'pykeen'):
        try:
            import importlib.metadata as md
            out[name] = md.version('scikit-learn' if name == 'sklearn' else name)
        except Exception:
            out[name] = None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(configs.DATASETS))
    p.add_argument('--model', default='TransE', choices=configs.MODELS)
    p.add_argument('--composition', default='hadamard', choices=configs.COMPOSITIONS)
    p.add_argument('--preset', default='phase1', choices=list(configs.PRESETS))
    p.add_argument('--retrain-policy', default='sphere', choices=['sphere', 'always', 'never'])
    p.add_argument('--sphere-k', type=float, default=1.0)
    p.add_argument('--detector', default='ztest', choices=['ztest', 'biomedical'])
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'runs')
    p.add_argument('--adazor-src', type=Path, default=Path(__file__).resolve().parent.parent / 'adazor' / 'src')
    p.add_argument('--device', default='auto')
    p.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'], help='random = smoke test only')
    p.add_argument('--max-stream', type=int, default=None, help='stop after this many stream triples (pilot/smoke)')
    p.add_argument('--run-name', default=None)
    p.add_argument('--no-drift-windows', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    # Optional overrides of the preset.
    for name, typ in (('real_dim', int), ('epochs', int), ('batch_size', int), ('seed', int), ('window_size', int),
                      ('nu', float), ('alpha', float), ('keep_fraction_after_drift', float), ('cutoff_year', int)):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=typ, default=None)
    p.add_argument('--gamma', default=None)
    p.add_argument('--standardize', choices=['yes', 'no'], default=None)
    args = p.parse_args(argv)

    cfg = build_config(args)
    name = args.run_name or configs.run_id(args.dataset, args.model, args.composition, args.preset,
                                           args.retrain_policy, args.sphere_k)
    run_dir = args.out_dir / name
    if (run_dir / 'summary.json').exists() and not args.overwrite:
        print(f'[skip] {name} already finished')
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2, default=str), encoding='utf-8')

    started = time.time()
    data_path = args.data_dir / cfg['data_file']
    print(f'[{name}] loading {data_path}', flush=True)
    triples_df = load_temporal_triples(data_path)
    initial_kg, initial_stream, future_stream, lookup = build_initial_knowledge_graph(
        triples_df, cutoff_year=cfg['cutoff_year'], target_type=cfg['target_type'])
    print(f'[{name}] all={len(triples_df)} initial_kg={len(initial_kg)} initial_stream={len(initial_stream)} '
          f'future_stream={len(future_stream)} device={cfg["device"]}', flush=True)
    del triples_df

    detector, detector_source = make_detector(cfg['detector'], cfg['alpha'], args.adazor_src)
    result = run_stream(initial_kg, initial_stream, future_stream, lookup, cfg, run_dir, detector)

    summary = {'run': name, **{k: cfg[k] for k in ('dataset', 'model', 'composition', 'preset', 'retrain_policy',
                                                  'sphere_k', 'detector', 'cutoff_year', 'switch_year',
                                                  'epochs', 'real_dim', 'max_stream')},
               'detector_source': detector_source, 'feature_dim': feature_dim(cfg), **result['summary'],
               'runtime_seconds': time.time() - started, 'versions': versions(),
               'finished_utc': datetime.now(timezone.utc).isoformat()}
    result['drifts'].to_csv(run_dir / 'drifts.tsv', sep='\t', index=False)
    result['window_history'].to_csv(run_dir / 'window_history.tsv.gz', sep='\t', index=False)
    result['step_trace'].to_csv(run_dir / 'step_trace.tsv.gz', sep='\t', index=False)
    result['embedding_shift_metrics'].to_csv(run_dir / 'embedding_shift_metrics.tsv', sep='\t', index=False)
    if not result['drift_windows'].empty:
        result['drift_windows'].to_csv(run_dir / 'drift_windows.tsv.gz', sep='\t', index=False)
    np.save(run_dir / 'initial_reference_flags.npy', result['initial_reference_flags'])
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(f'[{name}] done: drifts={summary["num_detected_drifts"]} re-embeds={summary["num_embedding_refresh_events"]} '
          f'sphere_retrains={summary["num_threshold_retrains"]} runtime={summary["runtime_seconds"]/60:.1f} min', flush=True)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('PYSTOW_HOME', str(Path(__file__).resolve().parent / '.pystow'))
    sys.exit(main())
