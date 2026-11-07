"""Step 2: A0 vs A1 -- does freezing the embedding remove the alarms? (Experiment 2, and A0/A1 in Exp 1)

A0  Antoine's method exactly as in Phase 1 (re-embed from scratch on every new entity, sphere rule).
A1  the same method with ONE change: the embedding is trained once and frozen; a new entity gets a
    folded-in vector (step 1, variant 'full') instead of a re-embed. Same reference, same OCSVM,
    same Z-test, same keep-10%-and-refill after an alarm. Any difference A0 -> A1 is caused by
    re-embedding alone.

Runs:
  A1 real      real stream order, every model x composition (A0 real = the Phase 1 block A runs)
  A1 control   no drift: the future stream replaced by the pre-switch triples in random order
  A0 control   the same control streams through Phase 1's own code (re-embeds; uses the GPU)
The control streams are identical for A0 and A1 (same seed per repeat), so they can be compared pairwise.

Output: outputs/frozen/<dataset>/<run>/ in Phase 1's format (summary.json, config.json,
window_history.tsv.gz, drifts.tsv, embedding_shift_metrics.tsv), so phase1/analyze.py functions apply.

  python phase2/step2_frozen.py --dataset case1 --what A1_real A1_control
  python phase2/step2_frozen.py --dataset case1 --what A0_control --a0-repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
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


def phase1_cfg(dataset, model, composition, device='cpu', embedder='pykeen'):
    """The Phase 1 run configuration (preset 'phase1', sphere rule k = 1), as phase1/drift_sphere.py builds it."""
    d = settings.DATASETS[dataset]
    cfg = dict(p1.configs.PRESETS['phase1'])
    cfg.update({'dataset': dataset, 'data_file': d['file'], 'cutoff_year': d['cutoff_year'],
                'target_type': d['target_type'], 'switch_year': d['switch_year'], 'model': model,
                'composition': composition, 'preset': 'phase1', 'retrain_policy': 'sphere', 'sphere_k': 1.0,
                'detector': 'ztest', 'embedder': embedder, 'device': device, 'max_stream': None,
                'save_drift_windows': False, 'cache_dir': None, 'cache_max_gb': 0})
    return cfg


def control_order(stream_df, switch_year, rep):
    """Pre-switch triples of the future stream in random order (seeded per repeat; same for A0 and A1)."""
    pre = np.flatnonzero((stream_df['event_year'] < switch_year).to_numpy())
    return np.random.default_rng([settings.MONTE_CARLO['seed'], 7, rep]).permutation(pre)


# ---------------------------------------------------------------------------
# A1: Antoine's loop with a frozen embedding
# ---------------------------------------------------------------------------

def run_a1(X, init_idx, stream_idx, meta, cfg, detector):
    """Phase 1's run_stream (phase1/drift_sphere.py) minus re-embedding.

    X: features of every pool row (frozen embedding + folded-in vectors). With a frozen embedding a
    triple's outlier flag only changes when the OCSVM is retrained, so flags are computed once per fit.
    """
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
    folded = meta['folded']
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
               'reembedded': False, 'new_entity': bool(meta['new_entity'][r]), 'folded': bool(folded[r]),
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


def save_run(run_dir, cfg, summary, history, drifts, shifts=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2, default=str), encoding='utf-8')
    history.to_csv(run_dir / 'window_history.tsv.gz', sep='\t', index=False)
    drifts.to_csv(run_dir / 'drifts.tsv', sep='\t', index=False)
    (shifts if shifts is not None else pd.DataFrame()).to_csv(run_dir / 'embedding_shift_metrics.tsv', sep='\t', index=False)
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')


def a1_jobs(dataset, models, compositions, scenarios, repeats, out_root):
    pool = lib_data.load_pool(dataset, 'full')
    d = settings.DATASETS[dataset]
    init_idx = np.flatnonzero(pool['part'].eq('initial').to_numpy())
    stream_all = np.flatnonzero(pool['part'].eq('stream').to_numpy())
    meta = {'row_number': pool['row_number'].to_numpy(), 'event_year': pool['event_year'].to_numpy(),
            'new_entity': pool['new_entity'].to_numpy()}
    for model in models:
        path = lib_data.kge_features_path(dataset, model, 'full')
        if not path.exists():
            print(f'  [missing] {path.name}: run step1_prepare.py --variant full first')
            continue
        z = np.load(path, allow_pickle=False)
        meta['folded'] = z['folded']
        for comp in compositions:
            X = z[comp].astype(np.float64)
            runs = []
            if 'real' in scenarios:
                runs.append(('real', 0, stream_all))
            if 'control' in scenarios and d['switch_year'] is not None:
                stream_df = pool.iloc[stream_all]
                runs += [('control', k, stream_all[control_order(stream_df, d['switch_year'], k)]) for k in range(repeats)]
            for scenario, rep, order in runs:
                name = f'{dataset}__{model}__{comp}__A1frozen__{scenario}{rep:02d}'
                run_dir = out_root / name
                if (run_dir / 'summary.json').exists():
                    continue
                t0 = time.time()
                cfg = phase1_cfg(dataset, model, comp)
                cfg.update({'method': 'A1', 'scenario': scenario, 'repeat': rep,
                            'switch_year': d['switch_year'] if scenario == 'real' else None})
                detector, source = p1.make_detector('ztest', cfg['alpha'], ADAZOR_SRC)
                summary, history, drifts = run_a1(X, init_idx, order, meta, cfg, detector)
                summary.update({'run': name, 'dataset': dataset, 'model': model, 'composition': comp,
                                'preset': 'phase1', 'retrain_policy': 'frozen', 'sphere_k': None, 'method': 'A1',
                                'scenario': scenario, 'repeat': rep, 'detector_source': source,
                                'future_rows': len(order), 'feature_dim': settings.feature_dim(comp),
                                'runtime_seconds': time.time() - t0, 'switch_year': cfg['switch_year']})
                save_run(run_dir, cfg, summary, history, drifts)
                print(f'  A1 {name}: alarms={summary["num_detected_drifts"]} tested={summary["tested_steps"]}', flush=True)


def a0_control_jobs(dataset, models, compositions, repeats, out_root, device, embedder):
    """A0 on the same control streams, through Phase 1's own run_stream (re-embeds on new entities)."""
    initial_kg, initial_stream, future_stream, d = lib_data.load_dataset(dataset)
    lookup = p1.build_lookup_sets(initial_kg)
    for rep in range(repeats):
        order = control_order(future_stream, d['switch_year'], rep)
        control = future_stream.iloc[order].reset_index(drop=True)
        for model in models:
            for comp in compositions:
                name = f'{dataset}__{model}__{comp}__A0sphere1__control{rep:02d}'
                run_dir = out_root / name
                if (run_dir / 'summary.json').exists():
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                cfg = phase1_cfg(dataset, model, comp, device, embedder)
                cfg.update({'method': 'A0', 'scenario': 'control', 'repeat': rep, 'switch_year': None})
                detector, source = p1.make_detector('ztest', cfg['alpha'], ADAZOR_SRC)
                res = p1.run_stream(initial_kg, initial_stream, control, lookup, cfg, run_dir, detector)
                summary = {'run': name, 'dataset': dataset, 'model': model, 'composition': comp, 'preset': 'phase1',
                           'retrain_policy': 'sphere', 'sphere_k': 1.0, 'method': 'A0', 'scenario': 'control',
                           'repeat': rep, 'detector_source': source, 'feature_dim': settings.feature_dim(comp),
                           'switch_year': None, **res['summary'], 'runtime_seconds': time.time() - t0}
                save_run(run_dir, cfg, summary, res['window_history'], res['drifts'], res['embedding_shift_metrics'])
                print(f'  A0 {name}: alarms={summary["num_detected_drifts"]} re-embeds='
                      f'{summary["num_unseen_refresh_events"]} ({(time.time() - t0) / 60:.1f} min)', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(settings.DATASETS))
    p.add_argument('--what', nargs='+', default=['A1_real', 'A1_control'],
                   choices=['A1_real', 'A1_control', 'A0_control'])
    p.add_argument('--models', nargs='+', default=settings.MODELS)
    p.add_argument('--compositions', nargs='+', default=settings.COMPOSITIONS)
    p.add_argument('--a1-repeats', type=int, default=20, help='A1 control streams per model x composition')
    p.add_argument('--a0-repeats', type=int, default=3, help='A0 control streams per model x composition (GPU)')
    p.add_argument('--device', default='auto')
    p.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'], help='random = smoke test (A0 only)')
    args = p.parse_args(argv)
    out_root = settings.OUT / 'frozen' / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    scen = [w.split('_')[1] for w in args.what if w.startswith('A1')]
    if scen:
        a1_jobs(args.dataset, args.models, args.compositions, scen, args.a1_repeats, out_root)
    if 'A0_control' in args.what:
        if settings.DATASETS[args.dataset]['switch_year'] is None:
            print('  A0 control needs a dataset with a known switch; skipped')
        else:
            a0_control_jobs(args.dataset, args.models, args.compositions, args.a0_repeats, out_root,
                            lib_kge.pick_device(args.device), args.embedder)
    return 0


if __name__ == '__main__':
    sys.exit(main())
