"""Warm start check (version v3): how many epochs does one warm re-embed need?

What it does, per model, on one dataset:
  1. Splits the cutoff KG 96/2/2 as settings_check.py does and trains the model on the 96% with its
     chosen learning rate and epochs (configs.TRAINING). This is the OLD embedding.
  2. Adds the first --block stream triples (default 250, one window) to the training triples.
     This is the GROWN KG.
  3. Builds a model on the grown KG that starts from the old embedding (new entities start at
     PyKEEN's random vector) and trains it over the whole grown KG, for each learning rate
     (default: the chosen one and one fifth of it, as in Wewer et al. 2021). At 0, 1, 2, 3, 5 and
     10 epochs it measures:
       validation_mrr        MRR on validation triples of the cutoff KG: is the old knowledge kept?
       added_mrr             MRR of the added stream triples: are the new triples learned?
       old_cosine            mean cosine between every old entity's vector before and after
                             (L2-normalised vectors, as the features use them)
       reference_outlier_*   share of Antoine's reference triples that the OCSVM fitted under the old
                             embedding calls outliers under the new one (hadamard / concatenation)
       seconds               training time so far
  4. Trains the grown KG from scratch with the chosen settings and measures the same numbers:
     what a re-embed of version v1 gives (skip with --no-scratch).

    python warm_check.py --dataset case1
    python warm_check.py --dataset full --models TransE RotatE

    python warm_check.py --dataset case1 --choose-only   # no training: the choice again from the saved results

Output in warm_check/: one JSON per dataset and model (resumable), <dataset>__table.csv (all
measurements) and <dataset>__chosen.csv (the warm-start epochs and learning rate per model and composition,
which version v3 uses; see choose_warm for the rule).
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
import drift_sphere as p1  # noqa: E402


def state_of(model, tf):
    """The model as the stream code sees it: L2-normalised entity and relation vectors by label."""
    entities = p1.normalize_rows(p1._representation_to_numpy(model.entity_representations))
    relations = p1.normalize_rows(p1._representation_to_numpy(model.relation_representations))
    return p1.EmbeddingState(entity_index=dict(tf.entity_to_id), entity_matrix=entities,
                             relation_index=dict(tf.relation_to_id), relation_matrix=relations,
                             dim=entities.shape[1], kg_rows=int(tf.num_triples), train_seconds=0.0)


def new_model(model_name, tf, device):
    import pykeen.models as pykeen_models
    preset = configs.PRESETS['phase1']
    dim = preset['real_dim'] // 2 if model_name in configs.COMPLEX_MODELS else preset['real_dim']
    return getattr(pykeen_models, model_name)(triples_factory=tf, embedding_dim=dim,
                                              random_seed=preset['seed']).to(device)


def train(loop, tf, total_epochs, continue_training):
    t0 = time.time()
    loop.train(triples_factory=tf, num_epochs=total_epochs, batch_size=configs.PRESETS['phase1']['batch_size'],
               continue_training=continue_training, use_tqdm=False, use_tqdm_batch=False,
               pin_memory=False, num_workers=0)
    return time.time() - t0


def check_model(model_name, data, args, device):
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.training import SLCWATrainingLoop

    train_tf, grown_tf = data['train_tf'], data['grown_tf']
    chosen = configs.training_settings('phase1', args.dataset, model_name)
    base_lr = chosen['learning_rate']
    base_epochs = args.base_epochs or chosen['epochs']
    evaluator = RankBasedEvaluator(filtered=True)
    ocsvm_cfg = {k: configs.PRESETS['phase1'][k] for k in ('standardize', 'nu', 'gamma')}

    # 1. The old embedding.
    base = new_model(model_name, train_tf, device)
    base_seconds = train(SLCWATrainingLoop(model=base, triples_factory=train_tf, optimizer='Adam',
                                           optimizer_kwargs={'lr': base_lr}), train_tf, base_epochs, False)
    base_state = state_of(base, train_tf)
    raw = p1._model_weights(base, train_tf)
    old_labels = raw['entity_labels']
    old_vectors = base_state.entity_matrix
    reference = {}
    for comp in configs.COMPOSITIONS:
        X = p1.rows_to_features(data['reference_rows'], base_state, comp)
        ocsvm, scaler, flags = p1.train_classifier(X, ocsvm_cfg)
        reference[comp] = (ocsvm, scaler, float(flags.mean()))
    base_mrr = float(evaluator.evaluate(base, data['valid_in_train_ids'], additional_filter_triples=data['known_train_ids'],
                                        use_tqdm=False).get_metric('both.realistic.inverse_harmonic_mean_rank'))
    del base
    print(f'  {model_name}: old embedding trained ({base_epochs} epochs, {base_seconds:.0f}s), '
          f'validation MRR {base_mrr:.4f}', flush=True)

    def measure(model, start, lr, epochs, seconds):
        state = state_of(model, grown_tf)
        new_rows = [state.entity_index[x] for x in old_labels]
        row = {'model': model_name, 'start': start, 'learning_rate': lr, 'epochs': epochs, 'seconds': seconds,
               'validation_mrr': float(evaluator.evaluate(
                   model, data['valid_ids'], additional_filter_triples=data['known_ids'],
                   use_tqdm=False).get_metric('both.realistic.inverse_harmonic_mean_rank')),
               'added_mrr': float(evaluator.evaluate(
                   model, data['added_ids'], additional_filter_triples=data['known_ids'],
                   use_tqdm=False).get_metric('both.realistic.inverse_harmonic_mean_rank')),
               'old_cosine': float(np.mean(np.sum(old_vectors * state.entity_matrix[new_rows], axis=1)))}
        for comp in configs.COMPOSITIONS:
            ocsvm, scaler, in_sample = reference[comp]
            X = p1.rows_to_features(data['reference_rows'], state, comp)
            row[f'reference_outlier_{comp}'] = float(p1.outlier_flags(ocsvm, scaler, X).mean())
            row[f'reference_outlier_{comp}_old_embedding'] = in_sample
        print(f'    {start:7s} lr={lr:g} epochs={epochs:3d}: validation MRR {row["validation_mrr"]:.4f}  '
              f'added MRR {row["added_mrr"]:.4f}  old cosine {row["old_cosine"]:.3f}  reference outliers '
              f'{row["reference_outlier_hadamard"]:.2f} / {row["reference_outlier_concatenation"]:.2f}', flush=True)
        return row

    rows = []
    # 3. Warm start, epoch by epoch, for each learning rate.
    for lr in (args.learning_rates or [base_lr, base_lr / 5]):
        model = new_model(model_name, grown_tf, device)
        p1._copy_previous_weights(model, grown_tf, raw)
        model.reset_parameters_ = lambda *a, _m=model, **k: _m   # keep the copied weights (see drift_sphere.py)
        rows.append(measure(model, 'warm', lr, 0, 0.0))
        loop = SLCWATrainingLoop(model=model, triples_factory=grown_tf, optimizer='Adam', optimizer_kwargs={'lr': lr})
        seconds, done = 0.0, 0
        for target in args.epoch_grid:
            seconds += train(loop, grown_tf, target, done > 0)
            done = target
            rows.append(measure(model, 'warm', lr, target, seconds))
        del model, loop
    # 4. From scratch on the grown KG (a version v1 re-embed).
    if not args.no_scratch:
        model = new_model(model_name, grown_tf, device)
        seconds = train(SLCWATrainingLoop(model=model, triples_factory=grown_tf, optimizer='Adam',
                                          optimizer_kwargs={'lr': base_lr}), grown_tf, base_epochs, False)
        rows.append(measure(model, 'scratch', base_lr, base_epochs, seconds))
        del model
    if device.type == 'cuda':
        import torch
        torch.cuda.empty_cache()
    return {'dataset': args.dataset, 'model': model_name, 'base_learning_rate': base_lr, 'base_epochs': base_epochs,
            'base_seconds': base_seconds, 'base_validation_mrr': base_mrr, 'block': int(len(data['added_ids'])),
            'new_entities_in_block': data['new_entities'], 'rows': rows}


KEEP_OLD_MRR = 0.95        # (a) the old triples' validation MRR keeps at least 95% of its value before the update
MAX_REFERENCE_RISE = 0.05  # (b) the old OCSVM's outlier rate on the reference rises by at most 0.05 (half the 0.10 alarm gap)
TIE = 0.01                 # (c) added-triple MRRs within 1% of the best count as equal; then fewer epochs, then lower rate


def choose_warm(table):
    """Warm-start epochs and learning rate per model AND composition, with criteria fixed before any
    measurement was seen.

    Candidates: every warm row with at least 1 epoch (learning rates: the chosen one and one fifth of it,
    Wewer et al. 2021; epochs 1, 2, 3, 5, 10). A candidate passes if
      (a) it keeps the old knowledge: validation MRR on the cutoff KG's validation triples >= 95% of the
          same model before the update (the 0-epoch row: old vectors, new entities random);
      (b) it keeps the map: the outlier rate that the OCSVM fitted under the old embedding gives the reference
          triples under the updated vectors is at most 0.05 above its rate under the old embedding (half of
          Antoine's alarm gap of about 0.10, so the update alone cannot raise an alarm);
    and among the passing candidates
      (c) it learns the new triples: the highest MRR on the added stream triples; candidates within 1% of that
          count as equal, and the one with fewer epochs (then the lower learning rate) is taken.
    If no candidate passes (a) and (b), the gentlest update is taken (1 epoch, lower learning rate) and the
    row says so (passes = False).

    Changed on 19 Nov 2026, after the first measurements (CASE1, CASE2), with Diana's agreement: (b) was first
    required for both compositions at once, giving one setting per model. With Hadamard no setting passed (b)
    for any model, so concatenation also got the gentlest update, although longer updates pass (b) for it.
    Now (b) is applied to each composition on its own; the criteria are unchanged."""
    out = []
    for model, d in table.groupby('model', sort=False):
        warm = d[d['start'].eq('warm')]
        before = warm[warm['epochs'].eq(0)].iloc[0]
        scratch = d[d['start'].eq('scratch')]
        for comp in configs.COMPOSITIONS:
            cand = warm[warm['epochs'] >= 1].copy()
            cand['keeps_old'] = cand['validation_mrr'] >= KEEP_OLD_MRR * before['validation_mrr']
            cand['reference_rise'] = cand[f'reference_outlier_{comp}'] - cand[f'reference_outlier_{comp}_old_embedding']
            cand['keeps_map'] = cand['reference_rise'] <= MAX_REFERENCE_RISE
            ok = cand[cand['keeps_old'] & cand['keeps_map']]
            if len(ok):
                best = ok['added_mrr'].max()
                pick = ok[ok['added_mrr'] >= (1 - TIE) * best].sort_values(['epochs', 'learning_rate']).iloc[0]
                passes, reason = True, 'passes (a) and (b); best added-triple MRR (c)'
            else:
                pick = cand.sort_values(['epochs', 'learning_rate']).iloc[0]
                passes, reason = False, 'no candidate passes (a) and (b): gentlest update taken'
            out.append({'dataset': pick['dataset'], 'model': model, 'composition': comp,
                        'warm_epochs': int(pick['epochs']), 'warm_learning_rate': float(pick['learning_rate']),
                        'passes': passes, 'reason': reason,
                        'validation_mrr_before': float(before['validation_mrr']),
                        'validation_mrr_after': float(pick['validation_mrr']),
                        'added_mrr_before': float(before['added_mrr']), 'added_mrr': float(pick['added_mrr']),
                        'reference_outlier_before': float(pick[f'reference_outlier_{comp}_old_embedding']),
                        'reference_outlier_after': float(pick[f'reference_outlier_{comp}']),
                        'reference_rise': float(pick['reference_rise']),
                        'old_cosine': float(pick['old_cosine']), 'seconds': float(pick['seconds']),
                        'scratch_added_mrr': float(scratch['added_mrr'].iloc[0]) if len(scratch) else np.nan,
                        'scratch_validation_mrr': float(scratch['validation_mrr'].iloc[0]) if len(scratch) else np.nan,
                        'scratch_reference_outlier': float(scratch[f'reference_outlier_{comp}'].iloc[0]) if len(scratch) else np.nan,
                        'scratch_seconds': float(scratch['seconds'].iloc[0]) if len(scratch) else np.nan,
                        'candidates_passing': int(len(ok)), 'candidates': int(len(cand))})
    return pd.DataFrame(out)


def load_chosen(dataset, out_dir=None):
    """The warm-start settings per (model, composition) that version v3 uses.

    Read from <dataset>__chosen.csv. A file written before the per-composition rule (no 'composition' column)
    is not used: the choice is made again from <dataset>__table.csv (nothing is retrained). None if neither
    file exists."""
    out_dir = Path(out_dir) if out_dir is not None else HERE / configs.WARM_CHECK_DIR
    chosen_path = out_dir / f'{dataset}__chosen.csv'
    table_path = out_dir / f'{dataset}__table.csv'
    if chosen_path.exists():
        chosen = pd.read_csv(chosen_path)
        if 'composition' in chosen.columns:
            return chosen
    if table_path.exists():
        return choose_warm(pd.read_csv(table_path))
    return None


def write_tables(dataset, out_dir):
    """<dataset>__table.csv from the per-model JSONs, and <dataset>__chosen.csv from choose_warm."""
    tables = []
    for f in sorted(out_dir.glob(f'{dataset}__*.json')):
        res = json.loads(f.read_text(encoding='utf-8'))
        t = pd.DataFrame(res['rows'])
        t.insert(0, 'dataset', res['dataset'])
        t['old_embedding_validation_mrr'] = res['base_validation_mrr']
        tables.append(t)
    if not tables:
        raise SystemExit(f'no {dataset}__<model>.json in {out_dir}: run warm_check.py --dataset {dataset} first')
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_dir / f'{dataset}__table.csv', index=False)
    show = ['model', 'start', 'learning_rate', 'epochs', 'validation_mrr', 'added_mrr', 'old_cosine',
            'reference_outlier_hadamard', 'reference_outlier_concatenation', 'seconds']
    chosen = choose_warm(table)
    chosen.to_csv(out_dir / f'{dataset}__chosen.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 500):
        print(table[show].round(4).to_string(index=False))
        print('\nchosen for version v3, per model and composition (warm_check.choose_warm):')
        print(chosen[['model', 'composition', 'warm_epochs', 'warm_learning_rate', 'passes', 'validation_mrr_before',
                      'validation_mrr_after', 'added_mrr_before', 'added_mrr', 'reference_outlier_before',
                      'reference_outlier_after']].round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(configs.DATASETS))
    p.add_argument('--models', nargs='+', default=configs.MODELS, choices=configs.MODELS)
    p.add_argument('--block', type=int, default=250, help='stream triples added to the KG (default one window)')
    p.add_argument('--epoch-grid', nargs='+', type=int, default=[1, 2, 3, 5, 10])
    p.add_argument('--learning-rates', nargs='+', type=float, default=None,
                   help='warm learning rates (default: the chosen one and one fifth of it)')
    p.add_argument('--no-scratch', action='store_true')
    p.add_argument('--base-epochs', type=int, default=None, help='smoke test only: epochs of the old embedding')
    p.add_argument('--test-sample', type=int, default=2000)
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--out-dir', type=Path, default=HERE / 'warm_check')
    p.add_argument('--choose-only', action='store_true',
                   help='no training: rebuild the table and the choice from the saved per-model JSONs')
    args = p.parse_args()
    args.epoch_grid = sorted(set(args.epoch_grid))
    if args.choose_only:
        write_tables(args.dataset, args.out_dir)
        return 0

    import torch
    from pykeen.triples import TriplesFactory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = configs.PRESETS['phase1']['seed']
    ds = configs.DATASETS[args.dataset]
    window = configs.PRESETS['phase1']['window_size']
    triples_df = p1.load_temporal_triples(args.data_dir / ds['file'])
    initial_kg, initial_stream, future_stream, _ = p1.build_initial_knowledge_graph(
        triples_df, ds['cutoff_year'], ds['target_type'])
    del triples_df
    tf = TriplesFactory.from_labeled_triples(p1.KnowledgeGraph(initial_kg).triples().astype(str))
    train_tf, valid_tf, test_tf = tf.split([0.96, 0.02, 0.02], random_state=seed)   # same split as settings_check.py

    block = future_stream.head(args.block)[['subject', 'edge_type', 'object']].astype(str).to_numpy()
    grown = np.vstack([tf.label_triples(train_tf.mapped_triples), block])
    grown_tf = TriplesFactory.from_labeled_triples(grown)

    def ids(labeled):   # labelled triples -> ids of the grown KG
        return TriplesFactory.from_labeled_triples(np.asarray(labeled, dtype=str), entity_to_id=grown_tf.entity_to_id,
                                                   relation_to_id=grown_tf.relation_to_id).mapped_triples

    rng = np.random.default_rng(seed)
    valid_sample = valid_tf.mapped_triples
    if len(valid_sample) > args.test_sample:
        valid_sample = valid_sample[rng.choice(len(valid_sample), args.test_sample, replace=False)]
    valid_labeled = tf.label_triples(valid_sample)
    reference_rows = initial_stream.iloc[:-min(window, len(initial_stream) - 1)].to_dict('records')
    data = {
        'train_tf': train_tf, 'grown_tf': grown_tf, 'reference_rows': reference_rows,
        'valid_in_train_ids': valid_sample,                       # train_tf shares tf's ids
        'known_train_ids': [train_tf.mapped_triples, valid_tf.mapped_triples, test_tf.mapped_triples],
        'valid_ids': ids(valid_labeled), 'added_ids': ids(block),
        'known_ids': [grown_tf.mapped_triples, ids(tf.label_triples(valid_tf.mapped_triples)),
                      ids(tf.label_triples(test_tf.mapped_triples))],
        'new_entities': int(grown_tf.num_entities - tf.num_entities),
    }
    print(f'{args.dataset}: cutoff KG {tf.num_triples} triples, {tf.num_entities} entities; added {len(block)} stream '
          f'triples with {data["new_entities"]} new entities; reference {len(reference_rows)} triples; '
          f'{len(valid_sample)} validation triples; device {device}', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        path = args.out_dir / f'{args.dataset}__{model_name}.json'
        if path.exists():
            print(f'[skip] {path.name}', flush=True)
            continue
        print(f'[start] {args.dataset} {model_name}  ({time.strftime("%H:%M:%S")})', flush=True)
        path.write_text(json.dumps(check_model(model_name, data, args, device), indent=2), encoding='utf-8')

    write_tables(args.dataset, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
