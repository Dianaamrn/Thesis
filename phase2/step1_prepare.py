"""Step 1 (GPU): train each frozen embedding once, fold in new entities, save per-triple data.

For one dataset and variant this writes to outputs/features/<dataset>/:
  pool__<variant>.tsv.gz              every reference/stream triple, in stream order (model-independent)
  categories__holdout.npz             MONDO categories per triple (C3), if data/mondo.obo exists
  <model>__<variant>__seed<s>.npz     per triple: Hadamard and concatenation features, model score,
                                      whether a folded-in vector was used; plus timings

Everything later (steps 2-6) only reads these files, so the embedding is trained exactly once per
dataset, model and variant.

  python phase2/step1_prepare.py --dataset case1 --variant holdout full
  python phase2/step1_prepare.py --dataset case1 --model TransE --embedder random --max-stream 300   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402
import lib_kge  # noqa: E402


def prepare_pool(dataset, variant, max_stream=None, overwrite=False):
    path = lib_data.pool_path(dataset, variant)
    train, pool = lib_data.build_variant(dataset, variant)
    if max_stream:
        stream = np.flatnonzero(pool['part'].eq('stream').to_numpy())
        pool = pool.drop(index=stream[max_stream:]).reset_index(drop=True)
    if overwrite or not path.exists():
        pool.to_csv(path, sep='\t', index=False)
    if variant == 'holdout':
        cpath = lib_data.categories_path(dataset)
        if overwrite or not cpath.exists():
            cats = lib_data.mondo_categories(pool['disease'].tolist())
            if cats is None:
                print(f'  [C3] {settings.MONDO["file"]} not found: run step0_get_mondo.py; ontology channel skipped')
            else:
                np.savez_compressed(cpath, matrix=cats['matrix'], labels=np.asarray(cats['labels']),
                                    version=str(cats['version']), root=cats['root'],
                                    row_number=pool['row_number'].to_numpy())
                print(f'  [C3] MONDO {cats["version"]}: {len(cats["labels"]) - 1} categories')
    return train, pool


def prepare_model(dataset, variant, model, train, pool, seed, embedder, device, overwrite=False):
    out = lib_data.kge_features_path(dataset, model, variant, seed)
    if out.exists() and not overwrite:
        print(f'  [skip] {out.name} exists')
        return
    t0 = time.time()
    if embedder == 'random':
        kge, train_s = lib_kge.FrozenKGE.random(train, seed, device), 0.0
    else:
        kge, train_s = lib_kge.train_pykeen(train, model, seed, device)
    t1 = time.time()
    overrides, folded, unscorable = lib_kge.first_appearance_foldin(kge, pool)
    foldin_s = time.time() - t1

    h = pool['subject'].astype(str).tolist()
    r = pool['edge_type'].astype(str).tolist()
    t = pool['object'].astype(str).tolist()
    ok = np.flatnonzero(unscorable == 0)
    d = settings.KGE['real_dim']
    had = np.zeros((len(pool), d), dtype=np.float32)
    cat = np.zeros((len(pool), 3 * d), dtype=np.float32)
    score = np.full(len(pool), np.nan)
    if len(ok):
        sub = lambda xs: [xs[i] for i in ok]  # noqa: E731
        hh, cc = kge.features(sub(h), sub(r), sub(t), overrides)
        had[ok], cat[ok] = hh, cc
        score[ok] = kge.score(sub(h), sub(r), sub(t), overrides)
    if (unscorable == 1).any():
        # Should not happen for these streams (all stream relations exist before the cutoff).
        score[unscorable == 1] = np.nanmin(score)
        print(f'  WARNING: {int(unscorable.sum())} triples could not be embedded (unknown relation); '
              'features set to 0 and score to the minimum')
    np.savez_compressed(out, row_number=pool['row_number'].to_numpy(), hadamard=had, concatenation=cat,
                        score=score, folded=folded, unscorable=unscorable,
                        train_seconds=train_s, foldin_seconds=foldin_s, n_train_triples=len(train),
                        n_entities=len(kge.entity_to_id), n_folded_entities=len(overrides),
                        embedder=embedder, seed=seed)
    print(f'  {model} {variant}: trained {train_s:.0f}s, folded in {len(overrides)} entities '
          f'({foldin_s:.0f}s), {int(folded.sum())}/{len(pool)} triples use a folded vector, '
          f'total {time.time() - t0:.0f}s', flush=True)
    del kge
    if device.startswith('cuda'):
        import torch
        torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(settings.DATASETS))
    p.add_argument('--model', nargs='+', default=['all'], help='model names or "all"')
    p.add_argument('--variant', nargs='+', default=['holdout', 'full'], choices=['holdout', 'full'])
    p.add_argument('--seed', type=int, default=settings.KGE['seed'])
    p.add_argument('--embedder', default='pykeen', choices=['pykeen', 'random'], help='random = smoke test only')
    p.add_argument('--device', default='auto')
    p.add_argument('--max-stream', type=int, default=None, help='keep only the first N stream triples (smoke test)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    models = settings.MODELS if args.model == ['all'] else args.model
    device = lib_kge.pick_device(args.device)
    for variant in args.variant:
        print(f'[{args.dataset} / {variant}] building pool (device={device})', flush=True)
        train, pool = prepare_pool(args.dataset, variant, args.max_stream, args.overwrite)
        print(f'  training graph {len(train):,} triples; pool {len(pool):,} triples '
              f'({int(pool["part"].ne("stream").sum())} reference side, {int(pool["part"].eq("stream").sum())} stream; '
              f'{int(pool["novel"].sum())} with an unseen entity)', flush=True)
        for model in models:
            prepare_model(args.dataset, variant, model, train, pool, args.seed, args.embedder, device, args.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())
