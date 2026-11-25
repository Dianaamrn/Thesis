"""Read-only data profile of CASE1, CASE2 and full, using the pipeline's own loader and split (phase1/drift_sphere.py)."""
import sys, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / 'phase1'))
import configs, drift_sphere as ds

# ---- MONDO: ancestors, to label diseases as cancer / infectious
par = defaultdict(set); name = {}; cur = None
for line in (ROOT / 'data' / 'mondo.obo').read_text(encoding='utf-8', errors='ignore').splitlines():
    if line.startswith('[Term]'): cur = None
    elif line.startswith('id: MONDO:'): cur = line[4:].strip()
    elif cur and line.startswith('name: '): name[cur] = line[6:].strip()
    elif cur and line.startswith('is_a: MONDO:'): par[cur].add(line[6:].split()[0])
anc_cache = {}
def anc(m):
    if m in anc_cache: return anc_cache[m]
    seen, st = set(), [m]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in seen: seen.add(p); st.append(p)
    anc_cache[m] = seen; return seen
CANCER, INFECT = 'MONDO:0004992', 'MONDO:0005550'
print('MONDO roots:', name.get(CANCER), '|', name.get(INFECT))
def cat(iri):
    m = re.search(r'MONDO_(\d+)', str(iri))
    if not m: return 'not MONDO'
    i = 'MONDO:' + m.group(1); a = anc(i) | {i}
    c, f = CANCER in a, INFECT in a
    return 'both' if c and f else 'cancer' if c else 'infectious' if f else 'other'

def share(series):
    c = Counter(series); n = sum(c.values())
    return ', '.join(f'{k} {v/n:.2f}' for k, v in sorted(c.items(), key=lambda kv: -kv[1])) + f'  (n={n})'

for key in ['case1', 'case2', 'full']:
    d = configs.DATASETS[key]
    print('\n' + '=' * 100 + f'\n{key}: file {d["file"]}, cutoff {d["cutoff_year"]}, monitored {d["target_type"]}, switch {d["switch_year"]}')
    t = ds.load_temporal_triples(ROOT / 'data' / d['file'])
    kg, init_stream, stream, lookup = ds.build_initial_knowledge_graph(t, d['cutoff_year'], d['target_type'])
    kg = pd.DataFrame(kg) if not isinstance(kg, pd.DataFrame) else kg
    yr = t['event_year']
    print(f'file rows {len(t):,}; triple types {t["type"].nunique()}; undated rows {yr.isna().sum():,}; years {int(yr.min())}-{int(yr.max())}')
    print(f'starting graph rows {len(kg):,}; undated in it {kg["event_year"].isna().sum():,}; types {kg["type"].nunique()}')
    init = pd.DataFrame(init_stream) if not isinstance(init_stream, pd.DataFrame) else init_stream
    stream = pd.DataFrame(stream) if not isinstance(stream, pd.DataFrame) else stream
    W = 250; ref, win = init.iloc[:-W], init.iloc[-W:]
    sy = stream['event_year']
    parts = {'reference': ref, 'first window': win}
    if d['switch_year']:
        parts['stream before switch'] = stream[sy < d['switch_year']]; parts['stream from switch'] = stream[sy >= d['switch_year']]
    else:
        parts['stream'] = stream
    print('monitored: ' + ', '.join(f'{k} {len(v):,}' for k, v in parts.items()))
    iy = init['event_year']
    print(f'  monitored up to cutoff: undated {iy.isna().sum()}, years {iy.min():.0f}-{iy.max():.0f}; reference years {ref["event_year"].min():.0f}-{ref["event_year"].max():.0f}, first window years {win["event_year"].min():.0f}-{win["event_year"].max():.0f}')
    # entities
    start_nodes = set(kg['subject']) | set(kg['object'])
    for col, lab in (('subject', 'subjects'), ('object', 'objects (diseases)')):
        sets = {k: set(v[col]) for k, v in parts.items()}
        ks = list(sets)
        line = ', '.join(f'{k} {len(s)}' for k, s in sets.items())
        print(f'  unique {lab}: {line}')
        ov = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ov.append(f'{ks[i][:10]}&{ks[j][:10]} {len(sets[ks[i]] & sets[ks[j]])}')
        print('     overlaps: ' + '; '.join(ov))
        for k in ks[2:]:
            s = sets[k]; known_mon = set(init[col]);
            print(f'     {k}: in starting graph at all {len(s & start_nodes)}/{len(s)}; of those, only via other types (not in monitored before cutoff) {len((s & start_nodes) - known_mon)}')
    # re-embed triggers (sequential first occurrence of an unseen subject/object), per period
    seen = set(start_nodes); trig = []
    for s, o in zip(stream['subject'], stream['object']):
        new = s not in seen or o not in seen
        trig.append(new); seen.add(s); seen.add(o)
    stream = stream.assign(new_entity=trig)
    if d['switch_year']:
        sy = stream['event_year']
        for k, m in (('before switch', sy < d['switch_year']), ('from switch', sy >= d['switch_year'])):
            print(f'  stream {k}: triples with an unseen subject or object at arrival {int(stream.loc[m, "new_entity"].sum())}/{int(m.sum())} ({stream.loc[m, "new_entity"].mean():.2f})')
    else:
        print(f'  stream: triples with an unseen subject or object at arrival {int(stream["new_entity"].sum())}/{len(stream)} ({stream["new_entity"].mean():.2f})')
    # degree in the starting graph of the stream's entities (how well they are trained in the first embedding)
    deg = Counter(kg['subject']) + Counter(kg['object'])
    for k, v in parts.items():
        ds_ = np.array([deg.get(x, 0) for x in v['object']]); ss_ = np.array([deg.get(x, 0) for x in v['subject']])
        print(f'  starting-graph degree, {k}: subject median {np.median(ss_):.0f}, object median {np.median(ds_):.0f}, objects with degree 0 {np.mean(ds_ == 0):.2f}')
    # dates
    ts = stream['timestamp'].astype(str)
    c = ts.value_counts()
    print(f'  stream dates: distinct {len(c)}, triples per date median {c.median():.0f}, max {c.max()} (on {c.index[0]}), share of triples on dates with >=50 triples {c[c >= 50].sum() / len(stream):.2f}')
    print('  stream per year: ' + ', '.join(f'{int(y)} {n}' for y, n in stream['event_year'].value_counts().sort_index().items()))
    # disease categories
    for k, v in parts.items():
        print(f'  disease category, {k}: {share(v["object"].map(cat))}')
    if not d['switch_year']:
        sy = stream['event_year']
        per = stream.assign(c=stream['object'].map(cat), y=sy).groupby('y')['c'].value_counts(normalize=True).unstack().fillna(0).round(2)
        print('  full stream, category share per year:\n' + per.to_string())
