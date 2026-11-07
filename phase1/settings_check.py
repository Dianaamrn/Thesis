"""Choose the learning rate and number of epochs for every model, with one rule for all models.

Rule (PyKEEN's own early-stopping defaults, with MRR as the measure):
  - train on 98% of the dataset's cutoff KG, keep 2% for validation (same split as embedding_quality.py);
  - every 10 epochs compute the validation MRR (filtered ranking);
  - stop when 2 checks in a row do not improve the best MRR by at least 1%, or at --max-epochs;
  - do this for each learning rate; the chosen setting is the learning rate with the best MRR,
    and the epochs at which that best MRR was reached.
A model passes if its chosen MRR is at least 10 x the MRR of random guessing (ln N / N).
Also logged at every check: the MRR on a sample of the training triples ('fit'), which tells
"cannot learn at all" (fit also near random) from "learns but does not generalise".

    python settings_check.py --dataset case1
    python settings_check.py --dataset full --models ComplEx

Output in settings_check/ (one JSON per dataset/model/learning rate, resumable), and
settings_check/<dataset>__chosen.csv with the chosen settings, pass/fail and the time per epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402
from drift_sphere import KnowledgeGraph, build_initial_knowledge_graph, load_temporal_triples  # noqa: E402

PASS_FACTOR = 10  # chosen MRR must be >= PASS_FACTOR x random MRR


def check_one(model_name, lr, train, test, test_mapped, train_sample, num_entities, args, device):
    import pykeen.models as pykeen_models
    import torch
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop

    preset = configs.PRESETS['phase1']
    dim = preset['real_dim'] // 2 if model_name in configs.COMPLEX_MODELS else preset['real_dim']
    model = getattr(pykeen_models, model_name)(triples_factory=train, embedding_dim=dim,
                                               random_seed=preset['seed']).to(device)
    loop = SLCWATrainingLoop(model=model, triples_factory=train, optimizer='Adam', optimizer_kwargs={'lr': lr})
    evaluator = RankBasedEvaluator(filtered=True)
    curve, best_mrr, best_epoch, bad, done, train_seconds = [], -1.0, 0, 0, 0, 0.0
    while done < args.max_epochs:
        target = min(done + args.frequency, args.max_epochs)
        t0 = time.time()
        # num_epochs is the total; continue_training keeps the weights (otherwise PyKEEN re-initialises them)
        loop.train(triples_factory=train, num_epochs=target, batch_size=preset['batch_size'],
                   continue_training=done > 0, use_tqdm=False, use_tqdm_batch=False,
                   pin_memory=False, num_workers=0)
        train_seconds += time.time() - t0
        done = target
        known = [train.mapped_triples, test.mapped_triples]
        result = evaluator.evaluate(model, test_mapped, additional_filter_triples=known, use_tqdm=False)
        fit = evaluator.evaluate(model, train_sample, additional_filter_triples=known, use_tqdm=False)
        mrr = float(result.get_metric('both.realistic.inverse_harmonic_mean_rank'))
        hits = float(result.get_metric('both.realistic.hits_at_10'))
        fit_mrr = float(fit.get_metric('both.realistic.inverse_harmonic_mean_rank'))
        curve.append({'epoch': done, 'mrr': mrr, 'hits_at_10': hits, 'fit_mrr': fit_mrr})
        print(f'    {model_name} lr={lr:g} epoch {done}: MRR {mrr:.5f}  (training triples {fit_mrr:.5f})', flush=True)
        if np.isfinite(mrr) and mrr > best_mrr * (1 + args.relative_delta):
            best_mrr, best_epoch, bad = mrr, done, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    del model, loop
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    fit_at_best = next((c['fit_mrr'] for c in curve if c['epoch'] == best_epoch), float('nan'))
    return {'model': model_name, 'learning_rate': lr, 'best_mrr': best_mrr, 'best_epoch': best_epoch,
            'fit_mrr_at_best': fit_at_best,
            'stopped_at_epoch': done, 'seconds_per_epoch': train_seconds / max(done, 1),
            'random_mrr': float(np.log(num_entities) / num_entities), 'curve': curve}


def choose(rows):
    """Per model: the learning rate with the best MRR, epochs where that MRR was reached."""
    out = []
    for model_name, d in rows.groupby('model', sort=False):
        best = d.sort_values('best_mrr', ascending=False).iloc[0]
        out.append({'dataset': best['dataset'], 'model': model_name, 'learning_rate': best['learning_rate'],
                    'epochs': int(best['best_epoch']), 'mrr': best['best_mrr'],
                    'fit_mrr': best['fit_mrr_at_best'], 'random_mrr': best['random_mrr'],
                    'times_random': best['best_mrr'] / best['random_mrr'],
                    'passes': bool(best['best_mrr'] >= PASS_FACTOR * best['random_mrr']),
                    'seconds_per_epoch': best['seconds_per_epoch'],
                    'seconds_per_reembed_approx': best['seconds_per_epoch'] * int(best['best_epoch']),
                    'mrr_by_learning_rate': '; '.join(f'{r.learning_rate:g}: {r.best_mrr:.4f} @ {int(r.best_epoch)}'
                                                     for r in d.sort_values('learning_rate').itertuples())})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(configs.DATASETS))
    p.add_argument('--models', nargs='+', default=configs.MODELS, choices=configs.MODELS)
    p.add_argument('--learning-rates', nargs='+', type=float, default=[0.001, 0.01, 0.05])
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--frequency', type=int, default=10, help='epochs between validation checks (PyKEEN default 10)')
    p.add_argument('--patience', type=int, default=2, help='checks without improvement before stopping (PyKEEN default 2)')
    p.add_argument('--relative-delta', type=float, default=0.01, help='minimal relative improvement (PyKEEN default 0.01)')
    p.add_argument('--test-sample', type=int, default=2000)
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=HERE / 'settings_check')
    args = p.parse_args()

    import torch
    from pykeen.triples import TriplesFactory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = configs.PRESETS['phase1']['seed']
    ds = configs.DATASETS[args.dataset]
    triples_df = load_temporal_triples(args.data_dir / ds['file'])
    initial_kg, _, _, _ = build_initial_knowledge_graph(triples_df, ds['cutoff_year'], ds['target_type'])
    tf = TriplesFactory.from_labeled_triples(KnowledgeGraph(initial_kg).triples().astype(str))
    train, test = tf.split([0.98, 0.02], random_state=seed)
    rng = np.random.default_rng(seed)
    test_mapped = test.mapped_triples
    if len(test_mapped) > args.test_sample:
        test_mapped = test_mapped[rng.choice(len(test_mapped), args.test_sample, replace=False)]
    train_sample = train.mapped_triples
    if len(train_sample) > args.test_sample:
        train_sample = train_sample[rng.choice(len(train_sample), args.test_sample, replace=False)]
    print(f'{args.dataset}: cutoff KG {tf.num_triples} triples, {tf.num_entities} entities, '
          f'{len(test_mapped)} validation triples, device {device}', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        for lr in args.learning_rates:
            path = args.out_dir / f'{args.dataset}__{model_name}__lr{lr:g}.json'
            if path.exists():
                print(f'[skip] {path.name}', flush=True)
                continue
            print(f'[start] {args.dataset} {model_name} lr={lr:g}  ({time.strftime("%H:%M:%S")})', flush=True)
            res = check_one(model_name, lr, train, test, test_mapped, train_sample, tf.num_entities, args, device)
            res.update({'dataset': args.dataset, 'num_entities': int(tf.num_entities),
                        'n_train': int(train.num_triples), 'n_validation': int(len(test_mapped)),
                        'max_epochs': args.max_epochs, 'frequency': args.frequency, 'patience': args.patience,
                        'relative_delta': args.relative_delta})
            path.write_text(json.dumps(res, indent=2), encoding='utf-8')
            print(f'[done] {model_name} lr={lr:g}: best MRR {res["best_mrr"]:.5f} at epoch {res["best_epoch"]} '
                  f'(random {res["random_mrr"]:.5f})', flush=True)

    rows = pd.DataFrame([json.loads(f.read_text(encoding='utf-8'))
                         for f in sorted(args.out_dir.glob(f'{args.dataset}__*__lr*.json'))]).drop(columns='curve')
    chosen = choose(rows)
    chosen.to_csv(args.out_dir / f'{args.dataset}__chosen.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(chosen.drop(columns='dataset').to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
