"""Full-data bench: can the detector's statistic (the one-class SVM outlier share) see a topic change on the full data?

Question. On the full data there is no known change, so the alarms of the 40 runs cannot be judged. This check tests the
detector piece on its own, under the best conditions, with the full data's own triples:

  A  no change   random blocks of 250 non-cancer stream triples: how much the outlier share moves by chance
  B  change      random blocks of 250 cancer stream triples vs the non-cancer blocks: does the share differ?
  C  time order  consecutive blocks of 250 stream triples in time order: outlier share next to the cancer share
                 (descriptive only)
  +  baseline    outlier rate of the reference in-sample (the SVM's own triples) and held-out (fit on a random half,
                 measured on the other half), as in the final pipeline

Nothing of the pipeline changes. Same starting graph, same monitored type (Gene - causes_or_contributes_to_condition -
Disease), same reference (monitored triples up to 2010 minus the newest 250), same SVM (StandardScaler + OneClassSVM,
RBF, nu 0.1, gamma 'scale'), same vectors: the first embeddings saved by the final pipeline
(final_pipeline/output/full/first_embedding/<model>__seed42__state.pkl). No stream loop: no re-embeds, refits, resets.
Only stream triples whose subject, relation and object have a vector in that embedding are scored.
Cancer = the disease is a MONDO descendant of cancer (MONDO:0004992), from data/mondo.obo.

Decision rule (fixed before measuring; 0.1002 = ADAZOR's alarm line for 250 triples, alpha 0.025):
  unstable           more than 5% of the A blocks differ from the A mean by more than 0.1002
  sees the change    stable, and at least half of the B (cancer) blocks differ from the A mean by more than 0.1002
  does not see it    stable, and fewer than half do

Run (server, inside tmux, from the repository root):
    cd ~/Thesis && source ~/drift/.venv/bin/activate
    python checks/full_bench_check.py
Smoke test with random stand-in vectors (no saved embedding needed; numbers are meaningless):
    python checks/full_bench_check.py --embedder random --models TransE --compositions hadamard
Output: checks/full_bench/ (table, blocks, figure).
"""
from __future__ import annotations

import argparse
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'phase1'))
import configs  # noqa: E402
import drift_sphere as ds  # noqa: E402

SEED = 42
W = 250
ALPHA = 0.025
LINE = st.norm().ppf(1 - ALPHA / 2) / math.sqrt(2 * W)      # 0.1002, ADAZOR's alarm line for 250 triples
N_BLOCKS = 200
MAX_UNSTABLE = 0.05
MIN_SEEN = 0.5
MODELS = ['TransE', 'TransH', 'DistMult', 'ComplEx', 'RotatE']
COMPOSITIONS = ['hadamard', 'concatenation']
CANCER = 'MONDO:0004992'
EMB_DIR = ROOT / 'final_pipeline' / 'output' / 'full' / 'first_embedding'
OUT = ROOT / 'checks' / 'full_bench'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---- data: the pipeline's own loader and split
def load_full():
    d = configs.DATASETS['full']
    t = ds.load_temporal_triples(ROOT / 'data' / d['file'])
    start, init, stream, _ = ds.build_initial_knowledge_graph(t, d['cutoff_year'], d['target_type'])
    n_win = min(W, len(init) - 1)
    return start, init.iloc[:-n_win].reset_index(drop=True), stream.reset_index(drop=True)


# ---- disease category from MONDO
def cancer_labeller():
    parents = defaultdict(set); cur = None
    for line in (ROOT / 'data' / 'mondo.obo').read_text(encoding='utf-8', errors='ignore').splitlines():
        if line.startswith('[Term]'):
            cur = None
        elif line.startswith('id: MONDO:'):
            cur = line[4:].strip()
        elif cur and line.startswith('is_a: MONDO:'):
            parents[cur].add(line[6:].split()[0])
    memo = {}

    def is_cancer(iri):
        m = re.search(r'MONDO_(\d+)', str(iri))
        if not m:
            return False
        node = 'MONDO:' + m.group(1)
        if node not in memo:
            seen, todo = {node}, [node]
            while todo:
                for p in parents.get(todo.pop(), ()):
                    if p not in seen:
                        seen.add(p); todo.append(p)
            memo[node] = CANCER in seen
        return memo[node]
    return is_cancer


# ---- vectors: the saved first embedding (or random stand-in vectors for the smoke test)
def load_embedding(model, embedder, start):
    if embedder == 'random':
        rng = np.random.default_rng(SEED)
        ents = sorted(set(start['subject'].astype(str)) | set(start['object'].astype(str)))
        rels = sorted(set(start['edge_type'].astype(str)))
        unit = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
        return {'entity_index': {e: i for i, e in enumerate(ents)}, 'entity_matrix': unit(rng.normal(size=(len(ents), 64))),
                'relation_index': {r: i for i, r in enumerate(rels)}, 'relation_matrix': unit(rng.normal(size=(len(rels), 64)))}
    return pickle.loads((EMB_DIR / f'{model}__seed{SEED}__state.pkl').read_bytes())


def features(rows, emb, comp):
    s = emb['entity_matrix'][[emb['entity_index'][str(x)] for x in rows['subject']]]
    p = emb['relation_matrix'][[emb['relation_index'][str(x)] for x in rows['edge_type']]]
    o = emb['entity_matrix'][[emb['entity_index'][str(x)] for x in rows['object']]]
    return s * p * o if comp == 'hadamard' else np.concatenate([s, p, o], axis=1)


def scorable(rows, emb):
    e, r = emb['entity_index'], emb['relation_index']
    return rows['subject'].astype(str).isin(e) & rows['object'].astype(str).isin(e) & rows['edge_type'].astype(str).isin(r)


def fit_ocsvm(X, k='none'):
    """As the final pipeline (build_notebook.py, fit_ocsvm): StandardScaler + OCSVM. With k (idea 1, 25 Nov): the
    features are first standardised and projected on their first k principal components (PCA fitted on X only)."""
    steps = []
    if k != 'none':
        sc = StandardScaler().fit(X)
        steps = [sc, PCA(n_components=int(k), random_state=SEED).fit(sc.transform(X))]
    for s in steps:
        X = s.transform(X)
    scaler = StandardScaler().fit(X)
    return steps, scaler, OneClassSVM(kernel='rbf', nu=0.1, gamma='scale').fit(scaler.transform(X))


def score(model, X):
    steps, scaler, svm = model
    for s in steps:
        X = s.transform(X)
    Z = scaler.transform(X)
    return svm.predict(Z) == -1, svm.decision_function(Z)


def random_blocks(flags, rng):
    return np.array([flags[rng.choice(len(flags), W, replace=False)].mean() for _ in range(N_BLOCKS)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--compositions', nargs='+', default=COMPOSITIONS)
    ap.add_argument('--embedder', default='saved', choices=['saved', 'random'])
    ap.add_argument('--pca', nargs='+', default=['none'], help="idea 1: 'none' and/or numbers of principal components")
    a = ap.parse_args(argv)
    a.pca = [x if x == 'none' else int(x) for x in a.pca]
    out = OUT / ('TEST_random' if a.embedder == 'random' else '') / ('' if a.pca == ['none'] else 'pca')
    out.mkdir(parents=True, exist_ok=True)

    log('loading the full data (pipeline loader and split)')
    start, ref, stream = load_full()
    is_cancer = cancer_labeller()
    stream = stream.assign(cancer=stream['object'].map(is_cancer))
    log(f'reference {len(ref):,} triples; stream {len(stream):,} triples (cancer share {stream["cancer"].mean():.2f})')

    rows, block_rows = [], []
    for model in a.models:
        emb = load_embedding(model, a.embedder, start)
        ok = scorable(stream, emb)
        s = stream[ok].reset_index(drop=True)
        ref_ok = scorable(ref, emb)
        r = ref[ref_ok].reset_index(drop=True)
        log(f'{model}: scorable stream triples {len(s):,} of {len(stream):,} (cancer {int(s["cancer"].sum()):,}, '
            f'non-cancer {int((~s["cancer"]).sum()):,}); reference with vectors {len(r):,} of {len(ref):,}')
        for comp, k in [(c, k) for c in a.compositions for k in a.pca]:
            t0 = time.time()
            X_ref, X_s = features(r, emb, comp), features(s, emb, comp)
            svm = fit_ocsvm(X_ref, k)
            fit_s = time.time() - t0
            in_flags, _ = score(svm, X_ref)
            perm = np.random.default_rng(SEED).permutation(len(r))   # held-out baseline: as make_split in the pipeline
            fit_idx, cal_idx = perm[: len(r) // 2], perm[len(r) // 2:]
            ho_flags, _ = score(fit_ocsvm(X_ref[fit_idx], k), X_ref[cal_idx])
            flags, dec = score(svm, X_s)
            canc = s['cancer'].to_numpy()
            rng = np.random.default_rng(SEED)
            A = random_blocks(flags[~canc], rng)
            B = random_blocks(flags[canc], rng)
            m0 = A.mean()
            unstable_share = float(np.mean(np.abs(A - m0) > LINE))
            seen_share = float(np.mean(np.abs(B - m0) > LINE))
            verdict = ('unstable' if unstable_share > MAX_UNSTABLE else
                       'sees the change' if seen_share >= MIN_SEEN else 'does not see the change')
            # C: consecutive blocks in time order (the stream is already sorted by the pipeline loader)
            nb = len(s) // W
            for b in range(nb):
                sl = slice(b * W, (b + 1) * W)
                block_rows.append({'model': model, 'composition': comp, 'pca': k, 'block': b + 1,
                                   'first_year': int(s['event_year'].iloc[sl.start]), 'last_year': int(s['event_year'].iloc[sl.stop - 1]),
                                   'outlier_share': float(flags[sl].mean()), 'cancer_share': float(canc[sl].mean())})
            cb = pd.DataFrame([x for x in block_rows if x['model'] == model and x['composition'] == comp and x['pca'] == k])
            rho = st.spearmanr(cb['outlier_share'], cb['cancer_share']).correlation if len(cb) > 2 else np.nan
            row = {'model': model, 'composition': comp, 'pca': k, 'reference': len(r), 'scorable_stream': len(s),
                   'baseline_in_sample': round(float(in_flags.mean()), 3), 'baseline_held_out': round(float(ho_flags.mean()), 3),
                   'stream_outlier_share_all': round(float(flags.mean()), 3),
                   'A_mean_noncancer': round(float(m0), 3), 'A_p2.5': round(float(np.percentile(A, 2.5)), 3),
                   'A_p97.5': round(float(np.percentile(A, 97.5)), 3), 'A_share_beyond_line': round(unstable_share, 3),
                   'B_mean_cancer': round(float(B.mean()), 3), 'B_minus_A': round(float(B.mean() - m0), 3),
                   'B_share_beyond_line': round(seen_share, 3), 'verdict': verdict,
                   'C_blocks': nb, 'C_spearman_outlier_vs_cancer': round(float(rho), 3),
                   'decision_mean_noncancer': round(float(dec[~canc].mean()), 4), 'decision_mean_cancer': round(float(dec[canc].mean()), 4),
                   'fit_seconds': round(fit_s, 1)}
            rows.append(row)
            log(f'  {comp} pca {k}: in-sample {row["baseline_in_sample"]}, held-out {row["baseline_held_out"]}; A {row["A_mean_noncancer"]} '
                f'[{row["A_p2.5"]}, {row["A_p97.5"]}], B {row["B_mean_cancer"]} (B-A {row["B_minus_A"]}, beyond line {row["B_share_beyond_line"]}) '
                f'-> {verdict}; C rho {row["C_spearman_outlier_vs_cancer"]} ({nb} blocks); fit {fit_s:.0f} s')
            pd.DataFrame(rows).to_csv(out / 'full_bench_table.csv', index=False)          # saved after every row
            pd.DataFrame(block_rows).to_csv(out / 'full_bench_time_blocks.csv', index=False)
        del emb

    table = pd.DataFrame(rows)
    print('\n' + table.to_string(index=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        blocks = pd.DataFrame(block_rows)
        keys = list(table[['model', 'composition', 'pca']].itertuples(index=False, name=None))
        fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.9 * len(keys)), sharex=True, squeeze=False)
        for ax, (m, c, k) in zip(axes[:, 0], keys):
            b = blocks[blocks.model.eq(m) & blocks.composition.eq(c) & (blocks.pca.astype(str) == str(k))]
            ax.plot(b['block'], b['outlier_share'], color='#2a78d6', lw=1.2, label='outlier share')
            ax.plot(b['block'], b['cancer_share'], color='#b4452f', lw=1.0, ls='--', label='cancer share')
            ax.set_ylim(0, 1); ax.set_title(f'{m} · {c} · pca {k}', fontsize=8, loc='left')
        axes[0, 0].legend(fontsize=7, loc='upper right'); axes[-1, 0].set_xlabel('block of 250 stream triples, in time order')
        plt.tight_layout(); fig.savefig(out / 'full_bench_time_blocks.png', dpi=120)
    except Exception as e:  # the figure is a convenience; the tables are the result
        log(f'figure not written: {e}')
    log(f'saved {out}')


if __name__ == '__main__':
    main()
