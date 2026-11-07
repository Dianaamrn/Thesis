"""Step 7 (GPU): is the embedding good enough? Link-prediction quality vs training length (Part 2.6).

Phase 1 trains every model for 10 epochs and its MRR is low (0.000-0.070). If the embeddings are
simply undertrained, any conclusion about the drift method could be blamed on that. This script
measures MRR / Hits@10 for one model after 10, 25, 50, 100, 200 epochs (same split and scoring as
phase1/embedding_quality.py: 98% of the cutoff graph to train, 2% held out, filtered ranking).
The key stream comparisons are then repeated with the model trained for QUALITY['quality_epochs']
(see run_phase2.sh).

Output: outputs/quality/<dataset>__<model>.csv

  python phase2/step7_quality.py --dataset case1 --model TransE
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402
import lib_kge  # noqa: E402


def main(argv=None):
    q = settings.QUALITY
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', default=q['dataset'], choices=list(settings.DATASETS))
    p.add_argument('--model', default=q['model'], choices=settings.MODELS)
    p.add_argument('--epochs', nargs='+', type=int, default=q['epoch_grid'])
    p.add_argument('--device', default='auto')
    args = p.parse_args(argv)

    import pykeen.models as pykeen_models
    import torch
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    device = torch.device(lib_kge.pick_device(args.device))
    initial_kg, _, _, _ = lib_data.load_dataset(args.dataset)
    triples = initial_kg[['subject', 'edge_type', 'object']].dropna().astype(str).to_numpy()
    tf = TriplesFactory.from_labeled_triples(triples)
    seed = settings.KGE['seed']
    train, test = tf.split([0.98, 0.02], random_state=seed)
    rng = np.random.default_rng(seed)
    test_mapped = test.mapped_triples
    if len(test_mapped) > q['test_sample']:
        test_mapped = test_mapped[rng.choice(len(test_mapped), q['test_sample'], replace=False)]
    dim = settings.KGE['real_dim'] // 2 if args.model in settings.COMPLEX_MODELS else settings.KGE['real_dim']
    model = getattr(pykeen_models, args.model)(triples_factory=train, embedding_dim=dim, random_seed=seed).to(device)
    loop = SLCWATrainingLoop(model=model, triples_factory=train)
    rows, done, t0 = [], 0, time.time()
    out = settings.OUT / 'quality'
    out.mkdir(parents=True, exist_ok=True)
    for target in sorted(args.epochs):
        # Training continues from where it stopped (num_epochs = total so far), so one model is scored at
        # every length. Without continue_training PyKEEN would re-initialise the weights.
        loop.train(triples_factory=train, num_epochs=target, batch_size=settings.KGE['batch_size'],
                   continue_training=done > 0,
                   use_tqdm=False, use_tqdm_batch=False, pin_memory=False, num_workers=0)
        done = target
        res = RankBasedEvaluator(filtered=True).evaluate(
            model, test_mapped, additional_filter_triples=[train.mapped_triples, test.mapped_triples], use_tqdm=False)
        rows.append({'dataset': args.dataset, 'model': args.model, 'epochs': target,
                     'mrr': float(res.get_metric('both.realistic.inverse_harmonic_mean_rank')),
                     'hits_at_10': float(res.get_metric('both.realistic.hits_at_10')),
                     'random_mrr_approx': float(np.log(tf.num_entities) / tf.num_entities),
                     'minutes': (time.time() - t0) / 60})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / f'{args.dataset}__{args.model}.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
