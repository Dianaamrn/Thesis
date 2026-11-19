"""Correctness checks: does the code do what Antoine's method says, step by step?

Each check states what the method requires, measures it on the code's own output, and passes or fails.
  1  the alarm threshold: z(1 - alpha/2) / sqrt(2n) = 0.1002 for n = 250, alpha = 0.025, and our detector
     gives the same answer as adazor's ZTestDriftDetector
  2  the data: starting graph = every triple up to the cutoff plus undated ones (all types); stream = one type;
     reference = the monitored triples before the last 250; counts as documented
  3  the stream loop, on the real data with a random stand-in embedding (fast, no training), for Rules 1, 2, 4:
     window always 250; every alarm = |window rate - reference rate| > threshold; after an alarm 225 untested
     refill triples, then the new reference (250) is the window; a re-embed exactly at the triples that bring an
     unseen entity or relation (never during a refill); the graph grows only by the stream triples;
     Rule 1 retrains iff ratio > k, Rule 2 at every re-embed, Rule 4 every X triples
  4  the sphere rule's geometry: a point inside the box has ratio <= 1, a corner has ratio 1
  5  the features: Hadamard = s * p * o, concatenation = [s, p, o], every vector has length 1
  6  the warm start, with real PyKEEN on a tiny graph: old vectors are copied exactly, a new entity is not,
     and one warm epoch keeps old vectors close (cosine) while a from-scratch training does not
  7  Antoine's own settings reproduce his alarms (when runs_antoine_check exists: run_grid.py --grid antoine)

    python checks.py                      # all checks; writes checks/checks.csv
    python checks.py --skip-pykeen        # without check 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402
import drift_sphere as ds  # noqa: E402

# Counts measured with this loader on the data files (strategy report 2026-11-16 and the v1 runs).
EXPECTED_DATA = {
    'case1': {'initial_kg_rows': 330740, 'initial_stream_rows': 506, 'initial_reference_size': 256, 'future_rows': 763},
    'case2': {'initial_kg_rows': 369719, 'initial_stream_rows': 476, 'initial_reference_size': 226, 'future_rows': 3829},
    'full': {'initial_kg_rows': 1428493, 'initial_stream_rows': 23186, 'initial_reference_size': 22936, 'future_rows': 13294},
}

RESULTS = []


def record(check, what, expected, got, passed, detail=''):
    RESULTS.append({'check': check, 'what': what, 'expected': str(expected), 'got': str(got),
                    'passed': bool(passed), 'detail': detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {check} {what}: expected {expected}, got {got} {detail}", flush=True)


# 1 ---------------------------------------------------------------------------------------------
def check_threshold(adazor_src):
    import scipy.stats as st
    thr = st.norm().ppf(1 - 0.025 / 2) / math.sqrt(2 * 250)
    record('1', 'alarm threshold for n = 250, alpha = 0.025', 0.1002, round(thr, 4), round(thr, 4) == 0.1002)
    ours = ds.FallbackZTestDriftDetector(alpha=0.025)
    theirs, source = ds.make_detector('ztest', 0.025, adazor_src)
    rng = np.random.default_rng(0)
    disagree = 0
    for _ in range(2000):
        ref = rng.random(int(rng.integers(50, 500))) < rng.uniform(0.05, 0.3)
        win = rng.random(250) < rng.uniform(0.0, 0.5)
        ours.reset(ref)
        theirs.reset(ref)
        disagree += ours.detect(win) != theirs.detect(win)
    record('1', f'our detector agrees with {source} ZTestDriftDetector on 2000 random windows', 0, disagree, disagree == 0)


# 2 ---------------------------------------------------------------------------------------------
def check_data(data_dir, datasets):
    loaded = {}
    for name in datasets:
        d = configs.DATASETS[name]
        path = data_dir / d['file']
        if not path.exists():
            record('2', f'{name}: data file', 'present', 'missing', False, str(path))
            continue
        triples = ds.load_temporal_triples(path)
        initial_kg, initial_stream, future_stream, lookup = ds.build_initial_knowledge_graph(
            triples, cutoff_year=d['cutoff_year'], target_type=d['target_type'])
        got = {'initial_kg_rows': len(initial_kg), 'initial_stream_rows': len(initial_stream),
               'initial_reference_size': len(initial_stream) - min(250, len(initial_stream) - 1),
               'future_rows': len(future_stream)}
        exp = EXPECTED_DATA[name]
        record('2', f'{name}: graph rows / monitored before cutoff / reference / stream', exp, got, got == exp)
        later = triples['event_year'] > d['cutoff_year']
        in_kg = initial_kg['event_year'].isna() | (initial_kg['event_year'] <= d['cutoff_year'])
        record('2', f'{name}: starting graph = undated + dated up to {d["cutoff_year"]}, all triple types',
               f'{int((~later).sum())} rows, {triples.loc[~later, "type"].nunique()} types',
               f'{len(initial_kg)} rows, {initial_kg["type"].nunique()} types',
               len(initial_kg) == int((~later).sum()) and bool(in_kg.all()))
        types = set(future_stream['type']) | set(initial_stream['type'])
        record('2', f'{name}: monitored stream = one triple type', d['target_type'], ' | '.join(sorted(types)),
               types == {d['target_type']})
        years = future_stream['event_year']
        record('2', f'{name}: stream is in time order and after the cutoff',
               f'years > {d["cutoff_year"]}, non-decreasing', f'{years.min():g} to {years.max():g}',
               bool((years > d['cutoff_year']).all() and years.is_monotonic_increasing))
        loaded[name] = (initial_kg, initial_stream, future_stream, lookup)
    return loaded


# 3 ---------------------------------------------------------------------------------------------
def run_random(dataset, policy, data_dir, out_dir, adazor_src, k=1.0, every=None):
    name = f'{dataset}__{policy}{every or ""}__check'
    argv = ['--dataset', dataset, '--model', 'TransE', '--composition', 'hadamard', '--embedder', 'random',
            '--retrain-policy', policy, '--sphere-k', str(k), '--device', 'cpu', '--out-dir', str(out_dir),
            '--run-name', name, '--data-dir', str(data_dir), '--adazor-src', str(adazor_src), '--overwrite',
            '--no-drift-windows']
    if every:
        argv += ['--retrain-every', str(every)]
    ds.main(argv)
    run = out_dir / name
    read = lambda f: pd.read_csv(run / f, sep='\t') if (run / f).stat().st_size > 1 else pd.DataFrame()
    return {'summary': json.loads((run / 'summary.json').read_text()), 'history': read('window_history.tsv.gz'),
            'trace': read('step_trace.tsv.gz'), 'shifts': read('embedding_shift_metrics.tsv'), 'drifts': read('drifts.tsv')}


def check_loop(r, label, initial_kg, initial_stream, future_stream, lookup, policy, k=1.0, every=None):
    s, h, t, sh = r['summary'], r['history'], r['trace'], r['shifts']
    window = 250
    record('3', f'{label}: every tested window has 250 triples', window, sorted(set(h['window_size'])),
           set(h['window_size']) == {window})
    import scipy.stats as st
    gap = st.norm().ppf(1 - 0.025 / 2) / math.sqrt(2 * window)
    expected_alarm = (h['current_outlier_fraction'] - h['reference_outlier_fraction']).abs() > gap
    wrong = int((expected_alarm != h['drift_detected'].astype(bool)).sum())
    record('3', f'{label}: alarm exactly when |window rate - reference rate| > 0.1002', 0, f'{wrong} wrong of {len(h)}', wrong == 0)
    total = int(s['processed_future_triples'])
    bad_refill, bad_resume = [], []
    refill_idx = set(t.loc[t['refill'].astype(bool), 'stream_index']) if len(t) else set()
    tested_idx = set(h['stream_index'])
    for a in r['drifts']['stream_index'] if len(r['drifts']) else []:
        want = set(range(a + 1, min(a + 225, total) + 1))
        if not want <= refill_idx:
            bad_refill.append(int(a))
        if a + 226 <= total and (a + 226) not in tested_idx:
            bad_resume.append(int(a))
    record('3', f'{label}: after each alarm the next 225 triples are untested refill, then testing resumes',
           '0 alarms wrong', f'{len(bad_refill)} refill / {len(bad_resume)} resume wrong of {len(r["drifts"])} alarms',
           not bad_refill and not bad_resume)
    after = h[h['stream_index'] > (r['drifts']['stream_index'].min() if len(r['drifts']) else np.inf)]
    before = h[h['stream_index'] <= (r['drifts']['stream_index'].min() if len(r['drifts']) else np.inf)]
    ok_ref = set(before['reference_size']) <= {len(initial_stream) - 250} and set(after['reference_size']) <= {250}
    record('3', f'{label}: reference = monitored triples before the first window, then the 250-triple window after an alarm',
           f'{len(initial_stream) - 250}, then 250', f'{sorted(set(before["reference_size"]))}, then {sorted(set(after["reference_size"]))}', ok_ref)
    # Re-embeds: recomputed independently from the data.
    nodes, rels = set(lookup['nodes']), set(lookup['edge_types'])
    expected = set()
    for i, row in enumerate(future_stream.head(total).itertuples(index=False), start=1):
        unseen = row.subject not in nodes or row.object not in nodes or row.edge_type not in rels
        if unseen and i not in refill_idx:
            expected.add(i)
        nodes.update([row.subject, row.object])
        rels.add(row.edge_type)
    got = set(sh.loc[sh['trigger_reason'].eq('unseen_element'), 'stream_index']) if len(sh) else set()
    record('3', f'{label}: re-embed exactly at the triples with an unseen entity or relation (not in a refill)',
           f'{len(expected)} triples', f'{len(got)} triples, {len(expected ^ got)} differ', expected == got)
    growth = (sh['embedding_kg_rows'] == len(initial_kg) + sh['stream_index']).all() if len(sh) else True
    record('3', f'{label}: the graph grows only by the stream triples (rows = starting graph + stream position)',
           True, bool(growth), bool(growth))
    if policy == 'always' and len(sh):
        u = sh[sh['trigger_reason'].eq('unseen_element')]
        record('3', f'{label}: Rule 2 retrains the OCSVM at every new-entity re-embed', len(u),
               int(u['retrained_ocsvm'].sum()), bool(u['retrained_ocsvm'].all()))
    if policy == 'sphere' and len(sh):
        u = sh[sh['trigger_reason'].eq('unseen_element')]
        want = u['center_radius_ratio'] > k
        wrong = int((want != u['retrained_ocsvm'].astype(bool)).sum())
        record('3', f'{label}: Rule 1/3 retrains iff ratio > k (k = {k:g}); {int(want.sum())} of {len(u)} above k',
               0, f'{wrong} wrong', wrong == 0)
    if policy == 'every':
        rows = t[~t['refill'].astype(bool)].sort_values('stream_index')
        count, want, alarms = 0, set(), set(r['drifts']['stream_index']) if len(r['drifts']) else set()
        for idx in rows['stream_index']:
            count += 1
            if count == every:
                want.add(int(idx))
                count = 0
            if idx in alarms:
                count = 0
        got = set(rows.loc[rows['scheduled_retrain'].astype(bool), 'stream_index'])
        record('3', f'{label}: Rule 4 reaches its schedule every {every} triples (refill not counted, reset at an alarm)',
               f'{len(want)} times', f'{len(got)} times, {len(want ^ got)} differ', want == got)


# 4, 5 ------------------------------------------------------------------------------------------
def check_geometry():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 64))
    box = ds.boundary_from_min_max(X.min(0), X.max(0))
    inside = rng.uniform(X.min(0), X.max(0), size=(10000, 64))
    ratios = np.array([ds.compute_center_radius_metrics(x, box, 1.0)['center_radius_ratio'] for x in inside])
    record('4', 'any point inside the box has sphere ratio <= 1', '<= 1', round(float(ratios.max()), 3), ratios.max() <= 1 + 1e-12)
    corner = ds.compute_center_radius_metrics(X.max(0), box, 1.0)['center_radius_ratio']
    record('4', 'a corner of the box has ratio exactly 1', 1.0, round(corner, 6), abs(corner - 1) < 1e-9)
    ent = ds.normalize_rows(rng.normal(size=(5, 8)))
    rel = ds.normalize_rows(rng.normal(size=(2, 8)))
    state = ds.EmbeddingState({f'e{i}': i for i in range(5)}, ent, {'r0': 0, 'r1': 1}, rel, 8, 0, 0.0)
    h = ds.compose(['e0'], ['r1'], ['e3'], state, 'hadamard')[0]
    c = ds.compose(['e0'], ['r1'], ['e3'], state, 'concatenation')[0]
    record('5', 'Hadamard = s * p * o', True, np.allclose(h, ent[0] * rel[1] * ent[3]), np.allclose(h, ent[0] * rel[1] * ent[3]))
    ok = np.allclose(c, np.concatenate([ent[0], rel[1], ent[3]]))
    record('5', 'concatenation = [s, p, o] (192 numbers for 64-number vectors)', True, ok, ok)
    norms = np.linalg.norm(ent, axis=1)
    record('5', 'vectors are scaled to length 1', 1.0, round(float(norms.min()), 6), np.allclose(norms, 1))


# 6 ---------------------------------------------------------------------------------------------
def check_warm_start():
    rng = np.random.default_rng(2)
    ents = [f'e{i}' for i in range(60)]
    rels = ['r0', 'r1', 'r2']
    base = pd.DataFrame({'subject': rng.choice(ents, 600), 'edge_type': rng.choice(rels, 600),
                         'object': rng.choice(ents, 600)})
    for model in ('TransE', 'ComplEx'):
        cfg = {'embedder': 'pykeen', 'model': model, 'real_dim': 16, 'seed': 42, 'device': 'cpu', 'batch_size': 128,
               'learning_rate': 0.01, 'epochs': 20, 'embed_update': 'warm', 'warm_epochs': 1, 'warm_learning_rate': 0.002}
        kg = ds.KnowledgeGraph(base)
        old = ds.generate_embeddings_from_kg(kg, cfg)
        kg.add({'subject': 'e0', 'edge_type': 'r0', 'object': 'new_entity'})
        # Exact copy: build the new model and copy, without training.
        import pykeen.models as pm
        from pykeen.triples import TriplesFactory
        tf = TriplesFactory.from_labeled_triples(kg.triples().astype(str))
        m = getattr(pm, model)(triples_factory=tf, embedding_dim=8 if model == 'ComplEx' else 16, random_seed=42)
        ds._copy_previous_weights(m, tf, old.raw)
        new_raw = ds._model_weights(m, tf)
        same = all(np.array_equal(new_raw['entity'][0][tf.entity_to_id[e]], old.raw['entity'][0][old.raw['entity_labels'].index(e)])
                   for e in ents)
        record('6', f'{model}: warm start copies every old vector exactly', True, same, same)
        new_row = new_raw['entity'][0][tf.entity_to_id['new_entity']]
        differs = not any(np.array_equal(new_row, row) for row in old.raw['entity'][0])
        record('6', f'{model}: the new entity starts from its own (random) vector', True, differs, differs)
        warm = ds.generate_embeddings_from_kg(kg, cfg, previous=old)
        scratch = ds.generate_embeddings_from_kg(kg, {**cfg, 'embed_update': 'scratch'})
        cos = lambda a, b: float(np.mean([a.entity_matrix[a.entity_index[e]] @ b.entity_matrix[b.entity_index[e]] for e in ents]))
        cw, cs = cos(old, warm), cos(old, scratch)
        record('6', f'{model}: old vectors after 1 warm epoch stay close; from scratch they do not (mean cosine)',
               'warm > 0.9, scratch < 0.5', f'warm {cw:.3f}, scratch {cs:.3f}', cw > 0.9 and cs < 0.5)


# 7 ---------------------------------------------------------------------------------------------
def check_antoine(runs_dir):
    a = configs.ANTOINE_CHECK
    for comp in a['compositions']:
        name = configs.run_id(a['dataset'], a['model'], comp, 'antoine', a['policy'], 1.0, version='v1')
        path = runs_dir / name / 'drifts.tsv'
        if not (runs_dir / name / 'summary.json').exists():
            record('7', f"Antoine's settings, {comp}: alarm file rows", a['alarm_rows'], 'not run yet', False,
                   '(python run_grid.py --grid antoine)')
            continue
        rows = pd.read_csv(path, sep='\t')['row_number'].tolist() if path.stat().st_size > 1 else []
        record('7', f"Antoine's settings, {comp}: alarm file rows as in Antoine's own run", a['alarm_rows'], rows,
               rows == a['alarm_rows'])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    p.add_argument('--adazor-src', type=Path, default=HERE.parent / 'adazor' / 'src')
    p.add_argument('--datasets', nargs='+', default=configs.ALL_DATASETS)
    p.add_argument('--loop-datasets', nargs='+', default=['case1'], help='datasets for check 3 (random embedding)')
    p.add_argument('--skip-pykeen', action='store_true')
    p.add_argument('--out', type=Path, default=HERE / 'checks')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    check_threshold(args.adazor_src)
    loaded = check_data(args.data_dir, args.datasets)
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.loop_datasets:
            if name not in loaded:
                continue
            kg, stream0, stream, lookup = loaded[name]
            k = 1.0
            for policy, every in (('always', None), ('sphere', None), ('every', 10)):
                r = run_random(name, policy, args.data_dir, Path(tmp), args.adazor_src, k=k, every=every)
                check_loop(r, f'{name}, rule {policy}{every or ""}', kg, stream0, stream, lookup, policy, k=k, every=every)
                if policy == 'always' and len(r['shifts']):
                    # The sphere run uses the always run's median ratio as k, so both branches (retrain, keep) occur.
                    u = r['shifts'][r['shifts']['trigger_reason'].eq('unseen_element')]
                    k = float(u['center_radius_ratio'].median())
    check_geometry()
    if not args.skip_pykeen:
        check_warm_start()
    check_antoine(HERE / configs.ANTOINE_CHECK['runs_dir'])

    table = pd.DataFrame(RESULTS)
    table.to_csv(args.out / 'checks.csv', index=False)
    n_fail = int((~table['passed']).sum())
    print(f'\n{len(table)} checks, {n_fail} failed -> {args.out / "checks.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
