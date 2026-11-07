"""Step 2: Antoine's stream loop with three ways of handling the embedding (Part 1.3, 1.4 and Part 2.1, 2.2).

Everything except the embedding is Antoine's method exactly as in Phase 1 (same reference, OCSVM,
ADAZOR Z-test, sphere rule k = 1, keep 10% and refill after an alarm):
  original  re-embed from scratch on every new entity                         (Antoine; Phase 1 code)
  aligned   re-embed from scratch, then rotate the new embedding onto the previous one with
            orthogonal Procrustes on the shared entities, before anything is computed with it
            (keeps the professor's retraining, adds the step the literature uses: Hamilton et al. 2016)
  frozen    embedding trained once; a new entity gets a folded-in vector (step 1, variant 'full')

Streams (the same ones for every method, so results can be compared pairwise):
  real     the real stream in time order
  control  NO drift: the pre-switch triples in random order                 (every alarm is false)
  inject   the switch at a random position: shuffled pre-switch triples, then post-switch triples

Output: outputs/stream/<dataset>/<run>/ in Phase 1's format (summary.json, config.json,
window_history.tsv.gz, drifts.tsv, embedding_shift_metrics.tsv; aligned runs also alignment.tsv).
The real-order 'original' runs are Phase 1's block A (phase1/runs) unless --epochs differs from Phase 1.

  python phase2/step2_stream_methods.py --dataset case1 --methods frozen --scenarios real control inject --repeats 20
  python phase2/step2_stream_methods.py --dataset case1 --methods original aligned --models TransE --repeats 10 --parallel 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402
import lib_kge  # noqa: E402
import drift_sphere as p1  # noqa: E402  (phase1)

ADAZOR_SRC = settings.REPO / 'adazor' / 'src'
_ORIGINAL_EMBED = p1.generate_embeddings_from_kg
_DATA = {}


def phase1_cfg(dataset, model, composition, device='cpu', embedder='pykeen', epochs=None):
    """The Phase 1 run configuration (preset 'phase1', sphere rule k = 1), as phase1/drift_sphere.py builds it."""
    d = settings.DATASETS[dataset]
    cfg = dict(p1.configs.PRESETS['phase1'])
    if epochs:
        cfg['epochs'] = epochs
    cfg.update({'dataset': dataset, 'data_file': d['file'], 'cutoff_year': d['cutoff_year'],
                'target_type': d['target_type'], 'switch_year': d['switch_year'], 'model': model,
                'composition': composition, 'preset': 'phase1', 'retrain_policy': 'sphere', 'sphere_k': 1.0,
                'detector': 'ztest', 'embedder': embedder, 'device': device, 'max_stream': None,
                'save_drift_windows': False, 'cache_dir': None, 'cache_max_gb': 0})
    return cfg


# ---------------------------------------------------------------------------
# Stream orders (positions in the future stream), identical for every method
# ---------------------------------------------------------------------------

def stream_order(years, switch_year, scenario, rep):
    """Positions of the future stream to feed, in order, and the 0-based position of the switch."""
    n = len(years)
    if scenario == 'real':
        order = np.arange(n)
    else:
        pre = np.flatnonzero(years < switch_year)
        post = np.flatnonzero(years >= switch_year)
        if scenario == 'control':
            order = np.random.default_rng([settings.MONTE_CARLO['seed'], 7, rep]).permutation(pre)
        elif scenario == 'inject':
            rng = np.random.default_rng([settings.MONTE_CARLO['seed'], 9, rep])
            W = settings.WINDOW
            k = int(rng.integers(W, max(W + 1, len(pre) + 1)))
            order = np.concatenate([rng.permutation(pre)[:k],
                                    rng.permutation(post)[:settings.MONTE_CARLO['post_len']]])
        else:
            raise ValueError(scenario)
    after = np.flatnonzero(years[order] >= switch_year) if (switch_year is not None and scenario != 'control') else []
    return order, (int(after[0]) if len(after) else None)


# ---------------------------------------------------------------------------
# aligned: wrap Phase 1's embedding function so every new embedding is rotated onto the previous one
# ---------------------------------------------------------------------------

class AlignedEmbedder:
    def __init__(self):
        self.prev, self.log = None, []

    def __call__(self, kg, cfg):
        state = _ORIGINAL_EMBED(kg, cfg)
        if self.prev is not None:
            shared = [e for e in state.entity_index if e in self.prev.entity_index]
            A = state.entity_matrix[[state.entity_index[e] for e in shared]]
            B = self.prev.entity_matrix[[self.prev.entity_index[e] for e in shared]]
            R = lib_kge.procrustes(A, B)
            nb = np.linalg.norm(B)
            self.log.append({'kg_rows': len(kg), 'shared_entities': len(shared),
                             'raw_cosine': float(np.mean(np.sum(A * B, 1))),
                             'resid_before': float(np.linalg.norm(A - B) / nb),
                             'resid_after': float(np.linalg.norm(A @ R - B) / nb)})
            state.entity_matrix = state.entity_matrix @ R
            state.relation_matrix = state.relation_matrix @ R
        self.prev = state
        return state


# ---------------------------------------------------------------------------
# frozen: Antoine's loop without re-embedding
# ---------------------------------------------------------------------------

def run_frozen(X, init_idx, stream_idx, meta, cfg, detector):
    """Phase 1's run_stream (phase1/drift_sphere.py) minus re-embedding. With a frozen embedding a
    triple's outlier flag only changes when the OCSVM is retrained, so flags are computed once per fit."""
    W = cfg['window_size']
    seed = min(W, len(init_idx) - 1)
    reference = list(init_idx[:-seed])
    window = deque(init_idx[-seed:], maxlen=W)

    def fit(ref):
        ocsvm, scaler, ref_flags = p1.train_classifier(X[ref], cfg)
        detector.reset(ref_flags)
        return p1.outlier_flags(ocsvm, scaler, X), ref_flags

    flags, ref_flags = fit(reference)
    initial_ref_rate = float(ref_flags.mean())
    history, drifts, i, total = [], [], 0, len(stream_idx)
    while len(window) < W and i < total:   # warm-up (only if the initial stream is short)
        window.append(stream_idx[i])
        i += 1
    while i < total:
        r = stream_idx[i]
        i += 1
        window.append(r)
        cur = flags[list(window)]
        is_drift = bool(detector.detect(cur))
        rec = {'stream_index': i, 'row_number': int(meta['row_number'][r]),
               'event_year': float(meta['event_year'][r]), 'window_size': len(window),
               'reference_size': len(reference), 'reference_outlier_fraction': float(ref_flags.mean()),
               'current_outlier_fraction': float(cur.mean()), 'drift_detected': is_drift,
               'reembedded': False, 'new_entity': bool(meta['new_entity'][r]), 'folded': bool(meta['folded'][r]),
               'sphere_retrained': False, 'center_radius_ratio': np.nan}
        history.append(rec)
        if not is_drift:
            continue
        drifts.append({**rec, 'drift_event_id': len(drifts) + 1})
        keep = max(1, int(round(W * cfg['keep_fraction_after_drift'])))
        rebuilt = list(window)[-keep:]
        while len(rebuilt) < W and i < total:   # refill from the stream (these triples are not tested)
            rebuilt.append(stream_idx[i])
            i += 1
        reference = rebuilt
        flags, ref_flags = fit(reference)
        window = deque(reference[-W:], maxlen=W)
    summary = {'initial_reference_size': len(init_idx) - seed, 'initial_reference_outlier_fraction': initial_ref_rate,
               'processed_future_triples': i, 'tested_steps': len(history), 'num_detected_drifts': len(drifts),
               'num_unseen_refresh_events': 0, 'num_drift_refresh_events': 0, 'num_threshold_retrains': 0,
               'unseen_retrain_ratio': 0.0, 'new_entity_triples': int(sum(h['new_entity'] for h in history))}
    return summary, pd.DataFrame(history), pd.DataFrame(drifts)


def save_run(run_dir, cfg, summary, history, drifts, shifts=None, extra=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2, default=str), encoding='utf-8')
    history.to_csv(run_dir / 'window_history.tsv.gz', sep='\t', index=False)
    drifts.to_csv(run_dir / 'drifts.tsv', sep='\t', index=False)
    (shifts if shifts is not None else pd.DataFrame()).to_csv(run_dir / 'embedding_shift_metrics.tsv', sep='\t', index=False)
    for name, df in (extra or {}).items():
        df.to_csv(run_dir / name, sep='\t', index=False)
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')


def run_name(dataset, model, comp, method, scenario, rep, epochs):
    return f'{dataset}__{model}__{comp}__{method}__{scenario}{rep:02d}{settings.epochs_suffix(epochs)}'


def base_summary(name, dataset, model, comp, method, scenario, rep, switch_pos, order_len, epochs, cfg):
    return {'run': name, 'dataset': dataset, 'model': model, 'composition': comp, 'preset': 'phase1',
            'method': method, 'scenario': scenario, 'repeat': rep, 'epochs': cfg['epochs'],
            'retrain_policy': {'original': 'sphere', 'aligned': 'sphere+align', 'frozen': 'frozen'}[method],
            'sphere_k': 1.0, 'switch_year': cfg['switch_year'],
            'switch_stream_index': None if switch_pos is None else switch_pos + 1,   # 1-based, like stream_index
            'future_rows': int(order_len), 'feature_dim': settings.feature_dim(comp)}


# ---------------------------------------------------------------------------
# Retraining methods (original, aligned): Phase 1's run_stream, one job per run, can run in parallel
# ---------------------------------------------------------------------------

def _init_worker(dataset):
    _DATA['initial_kg'], _DATA['initial_stream'], _DATA['future_stream'], _ = lib_data.load_dataset(dataset)
    _DATA['lookup'] = p1.build_lookup_sets(_DATA['initial_kg'])


def retrain_job(job):
    dataset, model, comp, method, scenario, rep, epochs, device, embedder, out_root = job
    fs = _DATA['future_stream']
    d = settings.DATASETS[dataset]
    order, switch_pos = stream_order(fs['event_year'].to_numpy(), d['switch_year'], scenario, rep)
    name = run_name(dataset, model, comp, method, scenario, rep, epochs)
    run_dir = Path(out_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = phase1_cfg(dataset, model, comp, device, embedder, epochs)
    cfg.update({'method': method, 'scenario': scenario, 'repeat': rep,
                'switch_year': None if scenario == 'control' else d['switch_year']})
    detector, source = p1.make_detector('ztest', cfg['alpha'], ADAZOR_SRC)
    aligner = AlignedEmbedder() if method == 'aligned' else None
    p1.generate_embeddings_from_kg = aligner if aligner else _ORIGINAL_EMBED
    try:
        res = p1.run_stream(_DATA['initial_kg'], _DATA['initial_stream'], fs.iloc[order].reset_index(drop=True),
                            _DATA['lookup'], cfg, run_dir, detector)
    finally:
        p1.generate_embeddings_from_kg = _ORIGINAL_EMBED
    summary = {**base_summary(name, dataset, model, comp, method, scenario, rep, switch_pos, len(order), epochs, cfg),
               'detector_source': source, **res['summary'], 'runtime_seconds': time.time() - t0}
    extra = {'alignment.tsv': pd.DataFrame(aligner.log)} if aligner else None
    save_run(run_dir, cfg, summary, res['window_history'], res['drifts'], res['embedding_shift_metrics'], extra)
    return f'{name}: alarms={summary["num_detected_drifts"]} re-embeds={summary["num_unseen_refresh_events"]} ' \
           f'({(time.time() - t0) / 60:.1f} min)'


# ---------------------------------------------------------------------------
# Frozen method: fast, runs in the main process
# ---------------------------------------------------------------------------

def frozen_jobs(dataset, models, compositions, scenarios, repeats, epochs, out_root):
    pool = lib_data.load_pool(dataset, 'full')
    d = settings.DATASETS[dataset]
    init_idx = np.flatnonzero(pool['part'].eq('initial').to_numpy())
    stream_all = np.flatnonzero(pool['part'].eq('stream').to_numpy())
    years = pool['event_year'].to_numpy()[stream_all]
    meta = {'row_number': pool['row_number'].to_numpy(), 'event_year': pool['event_year'].to_numpy(),
            'new_entity': pool['new_entity'].to_numpy()}
    for model in models:
        path = lib_data.kge_features_path(dataset, model, 'full', epochs=epochs)
        if not path.exists():
            print(f'  [missing] {path.name}: run step1_prepare.py --variant full first')
            continue
        z = np.load(path, allow_pickle=False)
        meta['folded'] = z['folded']
        for comp in compositions:
            X = z[comp].astype(np.float64)
            for scenario in scenarios:
                if scenario != 'real' and d['switch_year'] is None:
                    continue
                for rep in range(1 if scenario == 'real' else repeats):
                    name = run_name(dataset, model, comp, 'frozen', scenario, rep, epochs)
                    run_dir = out_root / name
                    if (run_dir / 'summary.json').exists():
                        continue
                    t0 = time.time()
                    order, switch_pos = stream_order(years, d['switch_year'], scenario, rep)
                    cfg = phase1_cfg(dataset, model, comp, epochs=epochs)
                    cfg.update({'method': 'frozen', 'scenario': scenario, 'repeat': rep,
                                'switch_year': None if scenario == 'control' else d['switch_year']})
                    detector, source = p1.make_detector('ztest', cfg['alpha'], ADAZOR_SRC)
                    summary, history, drifts = run_frozen(X, init_idx, stream_all[order], meta, cfg, detector)
                    summary = {**base_summary(name, dataset, model, comp, 'frozen', scenario, rep, switch_pos,
                                              len(order), epochs, cfg),
                               'detector_source': source, **summary, 'runtime_seconds': time.time() - t0}
                    save_run(run_dir, cfg, summary, history, drifts)
                    print(f'  {name}: alarms={summary["num_detected_drifts"]} tested={summary["tested_steps"]}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(settings.DATASETS))
    p.add_argument('--methods', nargs='+', default=settings.STREAM['methods'], choices=settings.STREAM['methods'])
    p.add_argument('--scenarios', nargs='+', default=['real', 'control', 'inject'], choices=['real', 'control', 'inject'])
    p.add_argument('--models', nargs='+', default=settings.MODELS)
    p.add_argument('--compositions', nargs='+', default=settings.COMPOSITIONS)
    p.add_argument('--repeats', type=int, default=None,
                   help='no-drift and injected streams each (default: settings.STREAM)')
    p.add_argument('--epochs', type=int, default=None, help='training epochs (default: Phase 1 setting, 10)')
    p.add_argument('--parallel', type=int, default=1, help='retraining runs side by side (GPU)')
    p.add_argument('--device', default='auto')
    p.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'], help='random = smoke test only')
    args = p.parse_args(argv)
    d = settings.DATASETS[args.dataset]
    out_root = settings.OUT / 'stream' / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    if 'frozen' in args.methods:
        reps = args.repeats if args.repeats is not None else settings.STREAM['frozen_repeats']
        frozen_jobs(args.dataset, args.models, args.compositions, args.scenarios, reps, args.epochs, out_root)

    retrain = [m for m in args.methods if m != 'frozen']
    if not retrain:
        return 0
    reps = args.repeats if args.repeats is not None else settings.STREAM['retrain_repeats']
    device = lib_kge.pick_device(args.device)
    jobs = []
    for method in retrain:
        for scenario in args.scenarios:
            if scenario != 'real' and d['switch_year'] is None:
                continue
            if method == 'original' and scenario == 'real' and settings.epochs_suffix(args.epochs) == '':
                print('  original / real: = Phase 1 block A runs (phase1/runs), not rerun')
                continue
            for rep in range(1 if scenario == 'real' else reps):
                for model in args.models:
                    for comp in args.compositions:
                        name = run_name(args.dataset, model, comp, method, scenario, rep, args.epochs)
                        if not (out_root / name / 'summary.json').exists():
                            jobs.append((args.dataset, model, comp, method, scenario, rep, args.epochs, device,
                                         args.embedder, str(out_root)))
    print(f'[{args.dataset}] {len(jobs)} retraining runs ({", ".join(retrain)}), {args.parallel} at a time', flush=True)
    if not jobs:
        return 0
    if args.parallel <= 1:
        _init_worker(args.dataset)
        for job in jobs:
            print('  ' + retrain_job(job), flush=True)
    else:
        # 'spawn': each worker starts clean, so CUDA is never shared across a fork
        with ProcessPoolExecutor(max_workers=args.parallel, initializer=_init_worker, initargs=(args.dataset,),
                                 mp_context=multiprocessing.get_context('spawn')) as ex:
            for msg in ex.map(retrain_job, jobs):
                print('  ' + msg, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
