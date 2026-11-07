"""Step 5 (CPU): period-level graph distances -- channel C6 (Experiment 10).

A 250-triple window is a drop in a graph of 330,000 triples, so whole-graph distances are used on a
slower clock: one graph per publication year of the monitored triples, compared year to year. The series
covers the reference period before the cutoff AND the stream after it (the same triple types throughout),
so the change-point test has enough years on both sides of a switch: with only the stream, CASE1 has a
single year-to-year comparison before its 2005 switch and the test cannot place a change there.
  disease_jaccard   1 - Jaccard overlap of the diseases studied in year y and year y-1
  wl_distance       total-variation distance between Weisfeiler-Lehman (1 iteration) label histograms
                    of the two yearly graphs; node label = node type (Gene, miRNA, Disease), relation
                    types included -> "which kinds of entities are linked by which relations"
                    (Shervashidze et al. 2011, typed adaptation)
  deltacon          1 - DeltaCon similarity (Koutra et al. 2013) between the cumulative stream graphs up to
                    y-1 and up to y, on their union node set (approximate version with random node groups)
For each series a single change-point test: the year that best splits the series into two levels,
with a permutation p-value.

Output: outputs/graph_distance/<dataset>/series.csv, changepoints.csv

  python phase2/step5_graph_distance.py --dataset case1
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402


def node_type(uri):
    u = str(uri)
    if 'MONDO_' in u:
        return 'Disease'
    if 'ncbi.nlm.nih.gov/gene' in u:
        return 'Gene'
    if 'mirbase' in u.lower() or 'MIMAT' in u or 'MI0' in u:
        return 'miRNA'
    m = re.search(r'/obo/([A-Za-z]+)_', u)
    return m.group(1) if m else 'other'


def wl_histogram(df):
    """1-iteration WL labels: (own type, sorted multiset of (relation, direction, neighbour type))."""
    neigh = {}
    for s, r, o in zip(df['subject'], df['edge_type'], df['object']):
        neigh.setdefault(s, []).append((r, 'out', node_type(o)))
        neigh.setdefault(o, []).append((r, 'in', node_type(s)))
    return Counter((node_type(n), tuple(sorted(v))) for n, v in neigh.items())


def tv_distance(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2:
        return np.nan
    keys = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in keys)


def deltacon(edges1, edges2, groups, iterations, rng):
    """Approximate DeltaCon similarity between two undirected graphs on the union node set."""
    nodes = sorted(set(edges1.ravel()) | set(edges2.ravel()))
    pos = {x: i for i, x in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return np.nan
    group = rng.integers(0, groups, n)

    def affinities(edges):
        if len(edges) == 0:
            A = sp.csr_matrix((n, n))
        else:
            i = np.array([pos[x] for x in edges[:, 0]])
            j = np.array([pos[x] for x in edges[:, 1]])
            A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
            A = ((A + A.T) > 0).astype(float)
        deg = np.asarray(A.sum(1)).ravel()
        eps = 1.0 / (1.0 + deg.max()) if deg.max() > 0 else 0.5
        E0 = np.zeros((n, groups))
        E0[np.arange(n), group] = 1.0
        # S = [I + eps^2 D - eps A]^-1 E0, by fixed-point iteration (FaBP)
        S = E0.copy()
        for _ in range(iterations):
            S = E0 - (eps ** 2) * deg[:, None] * S + eps * (A @ S)
        return np.sqrt(np.maximum(S, 0))

    d = np.linalg.norm(affinities(edges1) - affinities(edges2))
    return 1.0 - 1.0 / (1.0 + d)


def changepoint(series, permutations, rng):
    """Best single split into two levels (CUSUM-type statistic) with a permutation p-value."""
    x = np.asarray(series, dtype=float)
    ok = ~np.isnan(x)
    x = x[ok]
    T = len(x)
    if T < 4:
        return np.nan, np.nan

    def best(v):
        stats = [abs(v[:t].mean() - v[t:].mean()) * np.sqrt(t * (T - t) / T) for t in range(2, T - 1)]
        k = int(np.argmax(stats))
        return stats[k], k + 2

    s0, t0 = best(x)
    null = np.array([best(rng.permutation(x))[0] for _ in range(permutations)])
    p = (1 + np.sum(null >= s0)) / (permutations + 1)
    return int(np.flatnonzero(ok)[t0]), float(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(settings.DATASETS))
    args = p.parse_args(argv)
    cfg = settings.GRAPH
    rng = np.random.default_rng(0)
    # 'full' pool = every monitored triple: the whole pre-cutoff period ('initial') and the stream
    pool = lib_data.load_pool(args.dataset, 'full')
    df = pool[pool['event_year'].notna()].copy()
    df['event_year'] = df['event_year'].astype(int)
    years = sorted(df['event_year'].unique())
    rows, prev = [], None
    for y in years:
        cur = df[df['event_year'] == y]
        row = {'year': y, 'period': 'reference' if y <= settings.DATASETS[args.dataset]['cutoff_year'] else 'stream',
               'triples': len(cur), 'diseases': cur['disease'].nunique()}
        if prev is not None:
            d1, d2 = set(prev['disease']), set(cur['disease'])
            row['disease_jaccard'] = 1 - len(d1 & d2) / len(d1 | d2) if d1 | d2 else np.nan
            row['wl_distance'] = tv_distance(wl_histogram(prev), wl_histogram(cur))
            before = df[df['event_year'] < y][['subject', 'object']].astype(str).to_numpy()
            upto = df[df['event_year'] <= y][['subject', 'object']].astype(str).to_numpy()
            row['deltacon'] = deltacon(before, upto, cfg['deltacon_groups'], cfg['deltacon_iterations'], rng)
        rows.append(row)
        prev = cur
    series = pd.DataFrame(rows)
    out_dir = settings.OUT / 'graph_distance' / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_dir / 'series.csv', index=False)
    cps = []
    for col in ('disease_jaccard', 'wl_distance', 'deltacon'):
        if col not in series:
            continue
        k, pval = changepoint(series[col].to_numpy(), cfg['permutations'], rng)
        cps.append({'dataset': args.dataset, 'measure': col,
                    'changepoint_year': int(series['year'].iloc[k]) if k == k else np.nan, 'p_value': pval,
                    'switch_year': settings.DATASETS[args.dataset]['switch_year']})
    pd.DataFrame(cps).to_csv(out_dir / 'changepoints.csv', index=False)
    print(series.round(3).to_string(index=False))
    print(pd.DataFrame(cps).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
