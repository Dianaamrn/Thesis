"""Writes the two Phase 1 notebooks (plain JSON, no nbformat needed):

  checks.ipynb   is the code doing what Antoine's method says? (checks.py, plus Antoine's own settings)
  results.ipynb  every version x rule x dataset, side by side, to choose one version with the professor

    python notebooks/build_notebooks.py
Execute on the server after the runs (run_all.sh does it):
    jupyter nbconvert --to notebook --execute --inplace notebooks/checks.ipynb notebooks/results.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n')}


def notebook(cells):
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


SETUP = """
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
HERE = Path.cwd() if (Path.cwd() / 'results_helpers.py').exists() else Path.cwd() / 'notebooks'
sys.path.insert(0, str(HERE))
import results_helpers as rh
"""

CHECKS = [
    md("""
# Correctness checks: does our code do what Antoine's method says?

Before comparing any variation, the code is checked against the method, step by step. Each check states what
the method requires, measures it on the code's own output, and passes or fails (`checks.py`).

| # | What the method requires | How it is checked |
|---|---|---|
| 1 | Alarm when the window's outlier rate differs from the reference's by more than z(1-α/2)/√(2n) (ADAZOR) | the threshold for n = 250, α = 0.025 is 0.1002; our detector agrees with adazor's `ZTestDriftDetector` on 2,000 random windows |
| 2 | Starting graph = every triple up to the cutoff plus the undated ones, all types; stream = one type, in time order; reference = monitored triples before the last 250 | counts read with the run code's own loader, on all three data files |
| 3 | Window of 250; after an alarm keep 25, refill 225 untested, re-embed, new reference = window; re-embed at every triple with an unseen entity or relation; the graph grows by the stream; Rules 1, 2, 4 | runs on the real CASE1 stream with a random stand-in embedding (no training, so fast), every step compared with what it should be, recomputed independently from the data |
| 4 | Sphere rule: ratio = distance to the box centre / half the box diagonal | a point inside the box has ratio ≤ 1, a corner exactly 1 |
| 5 | Features: Hadamard s·p·o, concatenation [s, p, o], unit vectors | on known vectors |
| 6 | Warm start: old vectors kept, new entity random, a few epochs over the whole graph | real PyKEEN on a tiny graph, TransE and ComplEx: exact copy, then cosine of old vectors after one warm epoch vs from scratch |
| 7 | Our port reproduces Antoine's own run | his settings (`--preset antoine`: 32 dims, 2 epochs, lr 0.001, γ 0.5, no scaling) on CASE1 with TransE give his alarms: concatenation at file rows 2295, 2857, 3430, 37769; Hadamard one alarm on 2004-09-02 |

How each part of our code maps to Antoine's notebook (`reference/antoine_sphere_rule_edition2.ipynb`):

| Antoine's notebook | Our code (`drift_sphere.py`) | Difference |
|---|---|---|
| cell 1: `ZTestDriftDetector` (adazor) with fallback | `make_detector`, `FallbackZTestDriftDetector` | none |
| cell 2: `load_temporal_triples`, `sort_temporal_triples`, `build_initial_knowledge_graph`, `triple_has_unseen_elements`, `update_knowledge_graph` | same names | the graph is appended to instead of re-concatenated (same rows, same order) |
| cell 3: `generate_transe_embeddings_from_kg` | `generate_embeddings_from_kg` | 5 models instead of TransE; learning rate and epochs from the version; warm start (v3) |
| cell 4: `create_hadamard_features`, `create_concatenation_features`, `train_classifier`, `retrain_reference_model`, box and ratio functions | `compose`, `train_classifier`, `retrain_reference_model`, `boundary_from_min_max`, `compute_center_radius_metrics` | vectorised (same numbers); ratio limit k is a setting (Antoine: 1) |
| cell 5: `process_triple_for_features`, `rebuild_reference_after_drift`, `run_dynamic_ocsvm_ztest_stream_keep10` | same names, `run_stream` | Rules 2, 3, 4 added; logging of the untested refill triples (logging only) |
"""),
    code(SETUP + """
import subprocess
RUN_CHECKS = True   # False: only read checks/checks.csv from the last run
if RUN_CHECKS:
    out = subprocess.run([sys.executable, str(rh.PHASE1 / 'checks.py')], cwd=rh.PHASE1, capture_output=True, text=True)
    print(out.stdout[-3000:] if out.returncode == 0 else out.stdout[-3000:] + out.stderr[-3000:])
"""),
    code("""
c = rh.read_csv(rh.paths()['checks'])
print(f"{len(c)} checks: {int(c['passed'].sum())} passed, {int((~c['passed']).sum())} failed")
c[['check', 'what', 'expected', 'got', 'passed']]
"""),
    code("""
c[~c['passed']]   # anything listed here needs attention before the results are read
"""),
]

RESULTS = [
    md("""
# Phase 1 results: every version, every rule, every dataset

**What varies.** How the graph is re-embedded when a new entity arrives (the *version*), and when the OCSVM is
fitted again (the *rule*). Everything else is the same in every run (64 numbers per vector, window 250,
OCSVM ν = 0.1, γ 'scale', α = 0.025, keep 10% after an alarm, seed 42).

| Version | Re-embedding |
|---|---|
| scratch, 2 / 10 / 20 epochs | a new model on the grown graph, the same number of epochs for every model (each model's chosen learning rate) |
| scratch, MRR-chosen epochs (v1) | a new model on the grown graph, epochs chosen per model by validation MRR (20-100) |
| warm start (v3) | the previous vectors, a random vector for the new entity, a few epochs over the whole graph (chosen by `warm_check.py`) |

| Rule | The OCSVM is fitted again (on the same reference triples, with the current vectors) |
|---|---|
| R1 sphere k=1 | at a new entity, if the new triple is outside Antoine's sphere (ratio > 1) |
| R2 always | at every new entity |
| R3 sphere ~10/50/90% | Antoine's sphere with a smaller limit, set so it would fire at about 10/50/90% of new entities |
| R4 every 250/50/10 | after every X stream triples |

After an alarm every rule does Antoine's procedure: keep 25, refill 225 untested, re-embed, new reference.

**Data.** CASE1 (Gene-Disease, graph up to 2002, change 2005: cancer to infectious), CASE2 (miRNA-Disease,
graph up to 2018, change 2022: infectious to cancer), full data (Gene-Disease, graph up to 2010, no known change).
The change is placed at the first stream triple of the change year (CASE1: stream triple 567), whether or
not a run tested it.

**How to read the tables.** One run = one model with one composition. Where 10 runs are summarised (5 models
x 2 compositions), the table gives counts of runs, or the median with the lowest and highest value.
"""),
    code(SETUP + """
t = rh.load()
print(f'{len(t)} finished runs')
"""),
    md("## 1. What has been run (finished / planned)"),
    code("rh.run_overview(t)"),
    md("## 2. Correctness checks (details in `checks.ipynb`)"),
    code("""
c = rh.read_csv(rh.paths()['checks'])
print(f"{len(c)} checks, {int(c['passed'].sum())} passed" if len(c) else 'checks not run yet')
c[~c['passed']] if len(c) else None
"""),
    md("""
## 3. Embedding quality

Test MRR (filtered link prediction on 2% of the starting graph held out from training) per model and version.
Random guessing is about 0.0001-0.0002. The warm start starts from the v1 embedding.
"""),
    code("rh.quality_table()"),
    md("""
### Warm start: how many epochs per update

`warm_check.py` adds one window of stream triples (250) to the graph, continues training from the old vectors
for 1, 2, 3, 5, 10 epochs at the chosen learning rate and one fifth of it (Wewer et al. 2021), and measures
(a) the old triples' validation MRR, (b) how the old OCSVM judges the reference triples under the updated
vectors, (c) the MRR of the added triples. The rule fixed before the measurements: pass (a) >= 95% of before
and (b) a rise of at most 0.05, then the best (c), then the fewest epochs. `passes = False` means no setting
passed and the gentlest one was taken.
"""),
    code("""
for dataset, table in rh.warm_tables().items():
    print(dataset); display(table)
"""),
]

for case, change in (('case1', '2005 (cancer to infectious disease)'), ('case2', '2022 (infectious disease to cancer)')):
    RESULTS += [
        md(f"## {4 if case == 'case1' else 5}. {case.upper()}: the change is in {change}"),
        md("### Summary over the 10 runs of each version and rule"),
        code(f"rh.case_summary(t, '{case}')"),
        md("### Every run: alarms before the change, alarms after it, delay"),
        code(f"""
rh.heatmap(t, '{case}', 'alarms_pre', 'alarms before the change')
rh.heatmap(t, '{case}', 'alarms_post', 'alarms after the change', cmap='Blues')
rh.heatmap(t, '{case}', 'detection_delay', 'delay of the first alarm after the change (stream triples)', cmap='Purples')
"""),
        md("### Where the alarms fall relative to the change"),
        code(f"rh.alarm_offsets_grid('{case}')"),
        md("""
### Outlier rate over the stream (TransE, Hadamard; the other runs are in `analysis/figures/`)

Blue line = window outlier rate at tested triples, grey = reference rate, red = alarms, orange = peaks (numbered by
height, with their date), light blue = untested refill triples as the OCSVM in use at the alarm sees them,
ticks at the bottom = re-embeds (green = OCSVM retrained), dashed = the change.
"""),
        code(f"rh.show_run_figures('{case}', rules=('sphere1', 'always', 'every50'))"),
        md("### Peaks: highest and second-highest, and their distance to the change"),
        code(f"rh.peak_distance_table(t, '{case}')"),
        md("### Peak dates over all runs (for finding the papers: `analysis/literature/`)"),
        code(f"rh.peak_dates('{case}')"),
    ]

RESULTS += [
    md("""
## 6. Full data (no known change): where are the alarms and the peaks?

The v1 runs with Rules 1 and 2 are the earlier ones (5 to 27 hours each); v1 is not run with Rules 3 and 4
on the full data. The 10-epoch runs of TransE, TransH, DistMult and RotatE with Rules 1 and 2 are the earlier
10-epoch runs (same settings, `imported_from` in their summary).
"""),
    code("rh.full_alarms_by_year(t)"),
    code("rh.heatmap(t, 'full', 'num_detected_drifts', 'number of alarms')"),
    code("rh.show_run_figures('full', rules=('sphere1', 'always'))"),
    md("### Peak dates on the full data, ordered by how many runs have a peak there"),
    code("rh.peak_dates('full', top=50)"),
    md("""
## 7. Sensitivity of the detector settings (one setting changed at a time)

Version scratch 20 epochs, Rule 2, both cases, 10 runs per setting. `base` = the Phase 1 values.
`nu`: OCSVM ν; `win`: window size; `alpha`: test level; `keep`: share of the window kept after an alarm;
`seed`: random seed of the embedding.
"""),
    code("rh.sensitivity_table()"),
    md("## 8. Cost"),
    code("rh.cost_table(t)"),
    md("""
## 9. Choosing a version

Questions to answer from the tables above, for each version and rule:
1. Does it stay quiet before the change? (alarms before the change; CASE1 and CASE2)
2. Does it raise an alarm after the change, and how soon? (alarms after, delay)
3. Does it test the stream at all around the change? (share tested, runs that tested the change triple)
4. Do the models and compositions agree? (spread of the 10 runs)
5. What does it cost? (run time, re-embeds)
6. On the full data, do the peaks fall on dates that can be traced to papers?
"""),
]


def main():
    (HERE / 'checks.ipynb').write_text(json.dumps(notebook(CHECKS), indent=1), encoding='utf-8')
    (HERE / 'results.ipynb').write_text(json.dumps(notebook(RESULTS), indent=1), encoding='utf-8')
    print('wrote', HERE / 'checks.ipynb', 'and', HERE / 'results.ipynb')


if __name__ == '__main__':
    main()
