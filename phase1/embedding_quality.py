"""Link-prediction MRR / Hits@10 on the cutoff KG, per model, with the settings the runs use.

Shows whether each model learned the graph (same dim, learning rate, epochs, batch size as the runs).
Same 96/2/2 split as settings_check.py: trains on 96%, scores a sample of the test 2%, which was
not used to choose the learning rate and epochs (filtered ranking).

    python embedding_quality.py --dataset case1 --models TransE TransH DistMult ComplEx RotatE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402
from drift_sphere import KnowledgeGraph, build_initial_knowledge_graph, load_temporal_triples  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(configs.DATASETS))
    p.add_argument('--models', nargs='+', default=configs.MODELS, choices=configs.MODELS)
    p.add_argument('--preset', default='phase1', choices=list(configs.PRESETS))
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=HERE / 'quality')
    p.add_argument('--test-sample', type=int, default=2000)
    args = p.parse_args()

    import pykeen.models as pykeen_models
    import torch
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    ds = configs.DATASETS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    triples_df = load_temporal_triples(args.data_dir / ds['file'])
    initial_kg, _, _, _ = build_initial_knowledge_graph(triples_df, ds['cutoff_year'], ds['target_type'])
    triples = KnowledgeGraph(initial_kg).triples().astype(str)
    tf = TriplesFactory.from_labeled_triples(triples)
    seed = configs.PRESETS[args.preset]['seed']
    train, validation, test = tf.split([0.96, 0.02, 0.02], random_state=seed)
    rng = np.random.default_rng(seed)
    test_mapped = test.mapped_triples
    if len(test_mapped) > args.test_sample:
        test_mapped = test_mapped[rng.choice(len(test_mapped), args.test_sample, replace=False)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        started = time.time()
        cfg = configs.training_settings(args.preset, args.dataset, model_name)
        dim = cfg['real_dim'] // 2 if model_name in configs.COMPLEX_MODELS else cfg['real_dim']
        model = getattr(pykeen_models, model_name)(triples_factory=train, embedding_dim=dim, random_seed=cfg['seed']).to(device)
        SLCWATrainingLoop(model=model, triples_factory=train, optimizer='Adam',
                          optimizer_kwargs={'lr': cfg['learning_rate']}).train(
            triples_factory=train, num_epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            use_tqdm=False, use_tqdm_batch=False, pin_memory=False, num_workers=0)
        result = RankBasedEvaluator(filtered=True).evaluate(
            model, test_mapped, additional_filter_triples=[train.mapped_triples, validation.mapped_triples, test.mapped_triples],
            use_tqdm=False)
        out = {
            'dataset': args.dataset, 'model': model_name, 'preset': args.preset, 'embedding_dim': dim,
            'learning_rate': cfg['learning_rate'], 'epochs': cfg['epochs'], 'n_train': int(train.num_triples), 'n_test_scored': int(len(test_mapped)),
            'num_entities': int(tf.num_entities),
            'mrr': float(result.get_metric('both.realistic.inverse_harmonic_mean_rank')),
            'hits_at_10': float(result.get_metric('both.realistic.hits_at_10')),
            'random_mrr_approx': float(np.log(tf.num_entities) / tf.num_entities),
            'seconds': time.time() - started,
        }
        (args.out_dir / f'{args.dataset}__{model_name}__{args.preset}.json').write_text(json.dumps(out, indent=2))
        print(json.dumps(out), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
