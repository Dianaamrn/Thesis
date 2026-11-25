"""CASE2 bench: can the detector's statistic (the one-class SVM outlier share) see the 2022 topic change, in the best
case, with and without a PCA step before the SVM?

Plan: reports/Plan 2026-11-25 best-case detector check.md (rule revised in the discussion of 25 Nov).

Setup (nothing of the pipeline changes):
  - Vectors: one embedding of the starting graph plus every monitored stream triple, Phase 1 settings (64 real
    dimensions, learning rate and epochs from configs.TRAINING) - the same vectors as the separability check
    (checks/separability_check.py, AUC 0.93 for TransH). Every stream triple has a fully trained vector.
  - Reference and SVM as the pipeline: reference = monitored triples up to 2018 minus the newest 250 (226 triples);
    StandardScaler + OneClassSVM (RBF, nu 0.1, gamma 'scale'), fitted once, never refitted.
  - Variant (idea 1): before the pipeline's scaler + SVM, the features are standardised and projected on their first
    k principal components (PCA fitted on the reference only), k = 8, 16, 32. 'none' = the pipeline as it is.
  - Stream scored in consecutive blocks of 250 triples from its start; the block that contains the change (first
    triple of 2022) is left out. Blocks of 100 are reported next to it (line 0.1585), not used for the decision.

Decision rule (fixed before measuring; 0.1002 = ADAZOR's alarm line for 250 triples, alpha 0.025):
  unstable           some pre-change block is more than 0.1002 from the mean of the pre-change blocks
  sees the change    stable, and at least half of the post-change blocks are more than 0.1002 from that mean
  does not see it    stable, and fewer than half are
Control: random unit vectors (no meaning), same blocks, same rule. Added before the real run (25 Nov), after the
smoke test showed the random control itself 'sees the change' with Hadamard (no PCA and k = 32): after 2022 no triple
shares a disease with the reference, so a detector can react to WHICH entities appear rather than to their meaning.
A model therefore counts as seeing the topic change only for a composition and PCA variant where the random control
does NOT see the change.

Run (server, inside tmux, from the repository root):
    cd ~/Thesis && source ~/drift/.venv/bin/activate
    python checks/case2_bench_check.py
    python checks/case2_bench_check.py --embedder random --models TransE      (control)
Output: checks/case2_bench/ (table, blocks, features per model in features/).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'phase1'))
sys.path.insert(0, str(ROOT / 'checks'))
import configs  # noqa: E402
import drift_sphere as ds  # noqa: E402
from separability_check import training_cfg  # noqa: E402

SEED = 42
ALPHA = 0.025
line = lambda n: st.norm().ppf(1 - ALPHA / 2) / math.sqrt(2 * n)   # ADAZOR's alarm line for a window of n
MODELS = ['TransE', 'TransH', 'DistMult', 'ComplEx', 'RotatE']
COMPOSITIONS = ['hadamard', 'concatenation']
PCA_K = ['none', 8, 16, 32]
OUT = ROOT / 'checks' / 'case2_bench'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def make_svm(X_ref, k):
    """The pipeline's StandardScaler + OCSVM, with an optional PCA step (fitted on the reference) before it."""
    steps = []
    if k != 'none':
        sc = StandardScaler().fit(X_ref)
        pca = PCA(n_components=int(k), random_state=SEED).fit(sc.transform(X_ref))
        steps = [sc, pca]
    pre = lambda X: _apply(steps, X)
    Z = pre(X_ref)
    scaler = StandardScaler().fit(Z)
    svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale').fit(scaler.transform(Z))
    return lambda X: svm.predict(scaler.transform(pre(X))) == -1


def _apply(steps, X):
    for s in steps:
        X = s.transform(X)
    return X


def embed(initial_kg, stream, model, embedder, device):
    graph = ds.KnowledgeGraph(initial_kg)
    for row in stream.to_dict('records'):
        graph.add(row)
    try:
        return ds.generate_embeddings_from_kg(graph, training_cfg('case2', model, embedder, device)), device
    except RuntimeError as e:          # GPU out of memory (the GPU is shared): train on the CPU instead
        if device == 'cpu' or 'out of memory' not in str(e).lower():
            raise
        log(f'{model}: GPU out of memory, training on the CPU')
        import torch
        torch.cuda.empty_cache()
        return ds.generate_embeddings_from_kg(graph, training_cfg('case2', model, embedder, 'cpu')), 'cpu'


def blocks(flags, size, switch):
    pre, post = [], []
    for b in range(len(flags) // size):
        lo, hi = b * size, (b + 1) * size
        if hi <= switch:
            pre.append(flags[lo:hi].mean())
        elif lo >= switch:
            post.append(flags[lo:hi].mean())
    return np.array(pre), np.array(post)


def verdict(pre, post, ln):
    m0 = pre.mean()
    if np.any(np.abs(pre - m0) > ln):
        return 'unstable', m0, float(np.mean(np.abs(post - m0) > ln))
    share = float(np.mean(np.abs(post - m0) > ln))
    return ('sees the change' if share >= 0.5 else 'does not see the change'), m0, share


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'])
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)
    if a.device is None:
        try:
            import torch
            a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            a.device = 'cpu'
    out = OUT / ('control_random' if a.embedder == 'random' else '')
    (out / 'features').mkdir(parents=True, exist_ok=True)

    d = configs.DATASETS['case2']
    t = ds.load_temporal_triples(ROOT / 'data' / d['file'])
    initial_kg, init, stream, _ = ds.build_initial_knowledge_graph(t, d['cutoff_year'], d['target_type'])
    ref = init.iloc[:-min(250, len(init) - 1)]
    switch = int(np.argmax(stream['event_year'].to_numpy() >= d['switch_year']))   # 0-based: first triple of 2022
    log(f'reference {len(ref)}, stream {len(stream)}, change at stream triple {switch + 1}; device {a.device}')
    ref_rows, s_rows = ref.to_dict('records'), stream.to_dict('records')

    rows, block_rows = [], []
    for model in a.models:
        t0 = time.time()
        state, used = embed(initial_kg, stream, model, a.embedder, a.device)
        log(f'{model}: embedding trained on {used} in {(time.time() - t0) / 60:.1f} min')
        for comp in COMPOSITIONS:
            X_ref, X_s = ds.rows_to_features(ref_rows, state, comp), ds.rows_to_features(s_rows, state, comp)
            np.savez_compressed(out / 'features' / f'{model}__{comp}.npz', reference=X_ref, stream=X_s)
            for k in PCA_K:
                flag = make_svm(X_ref, k)
                in_rate = float(flag(X_ref).mean())
                perm = np.random.default_rng(SEED).permutation(len(X_ref))
                h = len(X_ref) // 2
                ho_rate = float(make_svm(X_ref[perm[:h]], k)(X_ref[perm[h:]]).mean())
                flags = flag(X_s)
                pre, post = blocks(flags, 250, switch)
                v, m0, share = verdict(pre, post, line(250))
                pre100, post100 = blocks(flags, 100, switch)
                v100, m100, share100 = verdict(pre100, post100, line(100))
                for i, x in enumerate(pre):
                    block_rows.append({'model': model, 'composition': comp, 'pca': k, 'period': 'before', 'block': i + 1, 'outlier_share': float(x)})
                for i, x in enumerate(post):
                    block_rows.append({'model': model, 'composition': comp, 'pca': k, 'period': 'after', 'block': i + 1, 'outlier_share': float(x)})
                row = {'model': model, 'composition': comp, 'pca': k, 'baseline_in_sample': round(in_rate, 3),
                       'baseline_held_out': round(ho_rate, 3), 'pre_blocks': len(pre), 'post_blocks': len(post),
                       'pre_mean': round(float(m0), 3), 'pre_min': round(float(pre.min()), 3), 'pre_max': round(float(pre.max()), 3),
                       'post_mean': round(float(post.mean()), 3), 'post_minus_pre': round(float(post.mean() - m0), 3),
                       'post_share_beyond_line': round(share, 3), 'verdict': v,
                       'verdict_blocks100': v100, 'post_share_beyond_line_100': round(share100, 3)}
                rows.append(row)
                log(f'  {comp:13s} pca {str(k):4s}: baseline {row["baseline_in_sample"]}/{row["baseline_held_out"]}; '
                    f'before {row["pre_mean"]} [{row["pre_min"]}, {row["pre_max"]}], after {row["post_mean"]} '
                    f'(diff {row["post_minus_pre"]}, beyond line {row["post_share_beyond_line"]}) -> {v}  | blocks of 100: {v100}')
                pd.DataFrame(rows).to_csv(out / 'case2_bench_table.csv', index=False)          # saved after every row
                pd.DataFrame(block_rows).to_csv(out / 'case2_bench_blocks.csv', index=False)
        del state
    print('\n' + pd.DataFrame(rows).to_string(index=False))
    log(f'saved {out}')


if __name__ == '__main__':
    main()
