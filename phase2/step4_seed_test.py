"""Step 4 (GPU): how different are two trainings of the same graph? (Experiment 3)

Trains every model with several seeds on
  G0 = the cutoff graph (what A0 starts from)
  G1 = G0 + the pre-switch stream triples (a grown graph with NO topic change)
and compares pairs of embeddings, matching entities by label:
  seed_only        G0 seed i  vs G0 seed j          only the random seed differs: the noise floor
  same_seed_grown  G0 seed s  vs G1 seed s          exactly what A0 does on a re-embed (seed 42 every time)
  cross            G0 seed i  vs G1 seed j          graph and seed differ
Measures (frame-free or after alignment):
  raw_cosine        mean cosine of the same entity in the two raw coordinate systems
  aligned_cosine    the same after orthogonal Procrustes (best rotation of A onto B)
  procrustes_resid  ||A R - B|| / ||B|| after alignment
  linear_cka        linear CKA (ignores rotation and scale; Kornblith et al. 2019)
  knn_overlap       share of each entity's 10 nearest neighbours that are the same (Meroue et al. 2026)
  ocsvm_*           an OCSVM fitted on Antoine's reference in frame A, applied to the pre-switch triples
                    in frame A (same frame), in frame B as is (raw) and in frame B rotated onto A (aligned)
                    -> the outlier rate a re-embed alone produces

Output: outputs/seed_test/<dataset>/embeddings/*.npz and pairs.csv.

  python phase2/step4_seed_test.py --models TransE DistMult --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402
import lib_kge  # noqa: E402
import drift_sphere as p1  # noqa: E402


def graphs(dataset):
    initial_kg, initial_stream, future_stream, d = lib_data.load_dataset(dataset)
    g0 = initial_kg[['subject', 'edge_type', 'object']].dropna().astype(str).to_numpy(dtype=object)
    pre = future_stream[future_stream['event_year'] < d['switch_year']]
    g1 = np.vstack([g0, pre[['subject', 'edge_type', 'object']].astype(str).to_numpy(dtype=object)])
    W = settings.WINDOW
    ref = initial_stream.iloc[:-min(W, len(initial_stream) - 1)]   # Antoine's initial reference
    return {'G0': g0, 'G1': g1}, ref, pre


def train_or_load(path, triples, model, seed, device, embedder):
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {'labels': z['labels'].tolist(), 'E': z['E'], 'rel_labels': z['rel_labels'].tolist(), 'R': z['R']}
    if embedder == 'random':
        kge, secs = lib_kge.FrozenKGE.random(triples, seed), 0.0
    else:
        kge, secs = lib_kge.train_pykeen(triples, model, seed, device)
    labels = sorted(kge.entity_to_id, key=kge.entity_to_id.get)
    rel_labels = sorted(kge.relation_to_id, key=kge.relation_to_id.get)
    out = {'labels': labels, 'E': kge.entity_matrix().astype(np.float32), 'rel_labels': rel_labels,
           'R': kge.relation_matrix().astype(np.float32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=np.asarray(labels), E=out['E'], rel_labels=np.asarray(rel_labels), R=out['R'],
                        train_seconds=secs)
    print(f'    trained {path.stem} in {secs:.0f}s', flush=True)
    return out


def procrustes(A, B):
    """Orthogonal R minimising ||A R - B|| (Schonemann 1966): R = U V^T from the SVD of A^T B."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def knn_overlap(A, B, k, sample, rng):
    idx = rng.choice(len(A), size=min(sample, len(A)), replace=False)
    out = []
    def top(M, chunk):
        S = M[chunk] @ M.T
        S[np.arange(len(chunk)), chunk] = -np.inf   # not its own neighbour
        return np.argpartition(-S, k, axis=1)[:, :k]

    for chunk in np.array_split(idx, max(1, len(idx) // 250)):
        out += [len(set(a) & set(b)) / k for a, b in zip(top(A, chunk), top(B, chunk))]
    return float(np.mean(out))


def feats(emb, rows, comp, R=None):
    e_pos = {x: i for i, x in enumerate(emb['labels'])}
    r_pos = {x: i for i, x in enumerate(emb['rel_labels'])}
    E, Rel = emb['E'].astype(np.float64), emb['R'].astype(np.float64)
    if R is not None:
        E, Rel = E @ R, Rel @ R
    s = E[[e_pos[x] for x in rows['subject']]]
    p = Rel[[r_pos[x] for x in rows['edge_type']]]
    o = E[[e_pos[x] for x in rows['object']]]
    return s * p * o if comp == 'hadamard' else np.concatenate([s, p, o], axis=1)


def compare(a, b, ref, test, rng):
    shared = sorted(set(a['labels']) & set(b['labels']))
    pa = {x: i for i, x in enumerate(a['labels'])}
    pb = {x: i for i, x in enumerate(b['labels'])}
    A = a['E'][[pa[x] for x in shared]].astype(np.float64)
    B = b['E'][[pb[x] for x in shared]].astype(np.float64)
    Rm = procrustes(A, B)
    AR = A @ Rm
    out = {'shared_entities': len(shared),
           'raw_cosine': float(np.mean(np.sum(A * B, 1))),
           'aligned_cosine': float(np.mean(np.sum(AR * B, 1) / np.maximum(np.linalg.norm(AR, axis=1), 1e-12))),
           'procrustes_resid': float(np.linalg.norm(AR - B) / np.linalg.norm(B)),
           'linear_cka': linear_cka(A, B),
           'knn_overlap': knn_overlap(A, B, settings.SEED_TEST['knn'], settings.SEED_TEST['knn_sample'], rng)}
    # OCSVM: fit in frame A on the reference; test the pre-switch triples in A, in B raw, in B rotated onto A.
    known = lambda df, emb: df[df['subject'].isin(set(emb['labels'])) & df['object'].isin(set(emb['labels']))  # noqa: E731
                               & df['edge_type'].isin(set(emb['rel_labels']))]
    ref_ok = known(known(ref, a), b)
    test_ok = known(known(test, a), b)
    cfg = {'standardize': settings.OCSVM['standardize'], 'nu': settings.OCSVM['nu'], 'gamma': settings.OCSVM['gamma']}
    for comp in settings.COMPOSITIONS:
        ocsvm, scaler, ref_flags = p1.train_classifier(feats(a, ref_ok, comp), cfg)
        out[f'ocsvm_{comp}_reference_insample'] = float(ref_flags.mean())
        out[f'ocsvm_{comp}_same_frame'] = float(p1.outlier_flags(ocsvm, scaler, feats(a, test_ok, comp)).mean())
        out[f'ocsvm_{comp}_other_frame_raw'] = float(p1.outlier_flags(ocsvm, scaler, feats(b, test_ok, comp)).mean())
        out[f'ocsvm_{comp}_other_frame_aligned'] = float(
            p1.outlier_flags(ocsvm, scaler, feats(b, test_ok, comp, R=Rm.T)).mean())
    out['ocsvm_test_triples'] = len(test_ok)
    return out


def main(argv=None):
    st = settings.SEED_TEST
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', default=st['dataset'], choices=[k for k, v in settings.DATASETS.items() if v['switch_year']])
    p.add_argument('--models', nargs='+', default=settings.MODELS)
    p.add_argument('--seeds', nargs='+', type=int, default=st['seeds'])
    p.add_argument('--device', default='auto')
    p.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'], help='random = smoke test only')
    args = p.parse_args(argv)
    device = lib_kge.pick_device(args.device)
    out_dir = settings.OUT / 'seed_test' / args.dataset
    g, ref, test = graphs(args.dataset)
    print(f'[seed test {args.dataset}] G0 {len(g["G0"]):,} triples, G1 {len(g["G1"]):,}; reference {len(ref)}, '
          f'test {len(test)} triples; seeds {args.seeds}', flush=True)
    rows = []
    rng = np.random.default_rng(0)
    for model in args.models:
        t0 = time.time()
        emb = {(gname, s): train_or_load(out_dir / 'embeddings' / f'{model}__{gname}__seed{s}.npz', triples,
                                         model, s, device, args.embedder)
               for gname, triples in g.items() for s in args.seeds}
        pairs = [('seed_only', ('G0', i), ('G0', j)) for i, j in itertools.combinations(args.seeds, 2)]
        pairs += [('same_seed_grown', ('G0', s), ('G1', s)) for s in args.seeds]
        cross = [(i, j) for i in args.seeds for j in args.seeds if i != j]
        sel = rng.permutation(len(cross))[:st['cross_pairs']]
        pairs += [('cross', ('G0', cross[k][0]), ('G1', cross[k][1])) for k in sel]
        for kind, ka, kb in pairs:
            row = {'dataset': args.dataset, 'model': model, 'pair': kind, 'a': f'{ka[0]}_s{ka[1]}', 'b': f'{kb[0]}_s{kb[1]}'}
            row.update(compare(emb[ka], emb[kb], ref, test, rng))
            rows.append(row)
        print(f'  {model}: {len(pairs)} pairs compared ({(time.time() - t0) / 60:.1f} min)', flush=True)
        pd.DataFrame(rows).to_csv(out_dir / 'pairs.csv', index=False)
    print('wrote', out_dir / 'pairs.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
