"""Writes the Phase 2 notebooks (run after editing this file: python build_notebooks.py).

  part1b_evidence_antoine.ipynb   Part 1: evaluating Antoine's method, with evidence
  part2_other_methods.ipynb       Part 2: what other researchers do, tested on our data
(part1a_audit_antoine.ipynb, the line-by-line audit of Antoine's code, is written separately.)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
import results_helpers as R
pd.set_option('display.max_columns', 40, 'display.width', 220)
print('results folder:', R.RESULTS)
R.settings_summary()
"""


def notebook(cells):
    out = []
    for kind, text in cells:
        src = text.strip('\n').splitlines(keepends=True)
        if kind == 'md':
            out.append({'cell_type': 'markdown', 'metadata': {}, 'source': src})
        else:
            out.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src})
    return {'cells': out, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                      'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


PART1 = [
    ('md', """
# Phase 2, Part 1: Evaluating Antoine's method, with evidence

**Question.** Why does Antoine's method (re-embed from scratch on every new entity, OCSVM, ADAZOR Z-test,
sphere rule) give alarms that are the same for every embedding model, and does it detect real drift?

The line-by-line audit of the code (12 problems, with small demonstrations) is in
`part1a_audit_antoine.ipynb`. This notebook shows the **evidence from the runs**.

| Section | Claim | Evidence |
|---|---|---|
| 1.1 | The sphere rule (almost) never fires | geometry + every ratio logged in Phase 1 + the sphere-size sweep |
| 1.2 | Embeddings retrained from scratch cannot be compared directly | controlled test: same graph with several seeds, an identical rerun, a grown graph, a warm start; literature |
| 1.3 | The alarms come from re-embedding | the same method with the embedding held fixed, or aligned |
| 1.4 | The method fires when nothing changes | streams with no drift |
| 1.5 | Its statistics are biased | reference vs new-triple outlier rates; fixing the statistics step by step |

All settings: `phase2/settings.py` (embedding as in Phase 1: 64 dims, 10 epochs, seed 42; window 250;
nu 0.1; alpha 0.025). Results are produced on the server by `phase2/run_phase2.sh`, which also executes
this notebook, so the outputs below are the real runs.
"""),
    ('code', SETUP),
    ('md', """
## Hypotheses, written before the runs

| # | Hypothesis | Supported if | Refuted if |
|---|---|---|---|
| H1.1 | The sphere rule cannot fire in practice | every logged ratio is below 1; the rule only retrains once k is shrunk far below 1 | ratios near or above 1 occur |
| H1.2 | Two from-scratch trainings are not comparable in raw coordinates, but are after alignment | raw cosine between the same entity in two trainings is near 0 while aligned cosine / neighbour overlap are clearly higher; an OCSVM applied to a retrained embedding flags far more than nu with no drift | raw coordinates agree (cosine near 1) |
| H1.3 | Re-embedding causes most alarms | the fixed-embedding (or aligned) version has clearly fewer alarms than Antoine's on the same stream | similar alarm counts |
| H1.4 | Antoine's method raises false alarms with no drift | most no-drift streams get at least one alarm | no-drift streams stay mostly silent |
| H1.5 | The reference rate is biased low | outlier rate on new triples > reference rate in (almost) every run; a held-out reference removes most no-drift alarms | rates equal; no change |
"""),
    ('md', """
## 1.1 The sphere rule (almost) never fires

**Why it can't fire, from the method itself.** The rule draws a box around all graph triple features and
retrains the OCSVM only if a new triple is further from the box centre than half the box diagonal
(ratio > k, k = 1). The furthest any point *inside* a box can be from its centre is a corner, exactly half
a diagonal away. So the ratio exceeds 1 only for a triple *outside* the box of all existing triples.
New triples are built from the same normalised vectors and fall inside. (Proof and demonstration:
`part1a_audit_antoine.ipynb`, problem 1.)

**Evidence from the runs.** Every ratio logged at a Phase 1 re-embed under Antoine's rule:
"""),
    ('code', "R.sphere_ratios()"),
    ('md', """
**The sphere-size sweep.** The radius was shrunk so that the rule would retrain on about 10%, 50% and 90%
of re-embeds (k chosen from the ratios above), plus "always retrain". If the alarms come from the
change of embedding, retraining the OCSVM more often should lower false alarms and re-embed-driven alarms.
"""),
    ('code', "R.sphere_sweep()"),
    ('md', """
## 1.2 Embeddings retrained from scratch cannot be compared directly

**Why, from the method itself.** A KG embedding is trained only to get *relationships* right (TransE:
subject + relation = object). Rotating every vector by the same amount leaves every distance, and so the
loss, unchanged. Training has no reason to prefer one orientation, and random initialisation, triple
order and negative sampling decide which one it lands in. DistMult, ComplEx and RotatE have their own
versions of this freedom (sign flips, coordinate swaps, phase rotations).

**What other researchers report** (short quotes; items marked [check against PDF] were read from arXiv HTML and must be checked against the published PDF before citing):
"""),
    ('code', "R.literature()"),
    ('md', """
**Controlled test (step 4).** Each model is trained on the cutoff graph G0 and on G1 = G0 + the pre-switch
triples (more data, **no topic change**). Pairs compared, entities matched by label:

| Pair | What it isolates |
|---|---|
| same graph, same seed, trained twice | is training even repeatable? |
| same graph, other seed | only randomness differs: the noise floor |
| grown graph, from scratch, same seed | **exactly what Antoine's method does at every re-embed** |
| grown graph, warm start | training continued from the old model instead of from zero |
| grown graph, other seed | data and randomness differ |

*Raw cosine* compares coordinates directly. *Aligned cosine* first rotates one embedding onto the other
(orthogonal Procrustes, the method of Hamilton et al. 2016). *Shared nearest neighbours* ignores rotation
altogether.
"""),
    ('code', "R.seed_pairs()"),
    ('md', """
**What this does to the OCSVM, with no drift at all.** An OCSVM is fitted on Antoine's reference in the
old embedding and applied to new pre-switch triples: in the same embedding, in the retrained embedding as
it is (what Antoine's method does), and in the retrained embedding rotated back. The gap between
"same embedding" and "new embedding" is the outlier rate that re-embedding alone creates.
"""),
    ('code', "R.seed_ocsvm()"),
    ('md', """
## 1.3 The alarms come from re-embedding

Antoine's method on the real streams, in three versions that differ **only** in how the embedding is
handled: retrained from scratch (Antoine; Phase 1 runs), retrained and then aligned onto the previous
embedding, and trained once and held fixed (new entities get a folded-in vector). Same reference, OCSVM,
Z-test, sphere rule and refill rule.
"""),
    ('code', "R.stream_real()"),
    ('md', """
## 1.4 Antoine's method fires when nothing changes

Streams of pre-switch triples only (one topic) in random order: every alarm is false. The same streams
are used for all three versions.
"""),
    ('code', "R.stream_no_drift()"),
    ('md', """
## 1.5 Antoine's statistics are biased

**The reference is checked on its own training data.** The OCSVM's reference outlier rate is measured on
the triples it was trained on (about nu = 10% of them lie outside its boundary by construction); fresh
triples from the same distribution fall outside more often. If so, the outlier rate on new triples should
exceed the reference rate in nearly every run:
"""),
    ('code', "R.insample_bias()"),
    ('md', """
**Fixing the statistics one step at a time** (same fixed embedding and OCSVM; no-drift and injected-switch
streams, 100 each): Antoine's statistics -> a fair reference (rate measured on held-out triples,
conformal p-values, Bates et al. 2023) -> plus a threshold set for repeated tests on overlapping windows
(Cobb et al. 2022).
"""),
    ('code', "R.statistics_steps()"),
    ('md', """
## What is not covered here

- Gradual or subtle drift, and no-drift streams in real time order: next steps, if time allows.
- The methods that retrain are run on fewer models and repeats than the cheap ones (a training per new
  entity); see the `streams` column in each table.
"""),
]

PART2 = [
    ('md', """
# Phase 2, Part 2: What other researchers do, tested on our data

**Question.** If Antoine's method is not reliable, what do researchers who monitor changing graphs do,
and how well does it work on our streams?

Tested mainly on **CASE1 and CASE2**: each has exactly one known drift (cancer <-> infectious disease), so
every method is scored the same way on 100 streams with no drift and 100 streams with the switch at a
random position. Methods that do not retrain also run on the whole full stream.

| Section | What others do | Source |
|---|---|---|
| 2.1 | retrain, then align the new embedding onto the old one | Hamilton et al. 2016; Hu et al. 2022 |
| 2.2 | keep the embedding fixed, give new entities their own vector (fold-in) | Hamaguchi et al. 2017; Zambon et al. 2018 |
| 2.3 | fair reference + threshold calibrated for repeated tests | Bates et al. 2023; Cobb et al. 2022 |
| 2.4 | measure change without an embedding: triple types, new entities, MONDO categories, plausibility, MMD | Rabanser et al. 2019; Polleres et al. 2023; Crichton et al. 2018 |
| 2.5 | graph distances year by year | Wills & Meyer 2020; Koutra et al. 2013 (DeltaCon) |
| 2.6 | compare everything fairly: threshold-free, over all thresholds, real streams, embedding quality | Hinder et al. 2024; Komorniczak 2026 |
"""),
    ('code', SETUP),
    ('md', """
## Hypotheses, written before the runs

| # | Hypothesis | Supported if | Refuted if |
|---|---|---|---|
| H2.1 | Retraining + alignment keeps the professor's procedure and removes most re-embed alarms | fewer false alarms than Antoine's version at a similar detection rate | no reduction |
| H2.2 | A fixed embedding with fold-in also removes them | same as above | no reduction, or the switch is missed |
| H2.3 | Fair reference + calibrated threshold give the intended false-alarm rate | no-drift false-alarm share close to the calibration target | far above target |
| H2.4 | Simple counts detect the cancer -> infectious switch | high detection, few false alarms | low detection |
| H2.5 | Yearly graph distances place the change point at the switch year | change point = switch year, small p-value | elsewhere |
| H2.6 | The conclusions hold with a better-trained embedding | same ordering of methods at 100 epochs | ordering changes |
"""),
    ('md', """
## 2.1 Retrain, then align (keeps the professor's retraining)

At every re-embed the new embedding is rotated onto the previous one using the entities they share
(orthogonal Procrustes) before anything is computed with it. Everything else is Antoine's method.
How far apart consecutive embeddings were, before and after the rotation:
"""),
    ('code', "R.alignment_log()"),
    ('md', """
## 2.1 and 2.2 Antoine vs retrain + align vs fixed embedding, on a known switch

Same streams for all three. *switch_detected* = share of streams with an alarm after the switch;
*median_delay* = triples between the switch and that alarm; *false_alarms_before_switch* = mean number.
(Real streams: Part 1, section 1.3. No-drift streams: section 1.4.)
"""),
    ('code', "R.stream_inject()"),
    ('md', """
## 2.3 Fair reference and calibrated threshold

Same fixed embedding and OCSVM, statistics changed one step at a time.
"""),
    ('code', "R.statistics_steps()"),
    ('md', """
## 2.4 Measuring change without an embedding (and with it)

Each measure describes a window of 250 triples and compares it with a fixed reference of 500 triples.
Calibrated measures allow at most a 10% chance of any false alarm in 1,000 no-drift triples (dashed line:
what that means for the no-drift stream length).
"""),
    ('code', "R.channels()"),
    ('md', "**The real streams**, statistic and alarms over time:"),
    ('code', "R.real_series('case1')"),
    ('code', "R.real_series('case2')"),
    ('code', "R.real_series('full')"),
    ('md', """
## 2.5 Graph distances year by year

A 250-triple window is a drop in a graph of 330,000 triples, so whole-graph distances are compared year by
year, with a change-point test on each series. The series covers the reference period before the cutoff and
the stream after it (same triple types), so the test has enough years on both sides of the switch. Years
with very few triples give noisy distances; the `triples` column shows how many each year has.
"""),
    ('code', "R.graph()"),
    ('md', """
## 2.6 Fair comparison

**Threshold-free.** For each measure: the probability that its statistic within 500 triples after the
switch exceeds its maximum over a whole no-drift stream (1 = perfect separation, 0.5 = chance). This does
not depend on any threshold.
"""),
    ('code', "R.separation()"),
    ('md', "**Over all thresholds**: false alarms with no drift vs switches detected within 500 triples."),
    ('code', "R.curves()"),
    ('md', """
**Embedding quality.** Phase 1's MRR is low. If conclusions only held for weak embeddings they would not be
convincing, so MRR is measured against training length, and the key comparisons are repeated with TransE
trained for 100 epochs:
"""),
    ('code', "R.quality()"),
    ('code', "R.stream_real(epochs=100)"),
    ('code', "R.stream_no_drift(epochs=100)"),
    ('code', "R.stream_inject(epochs=100)"),
    ('md', """
## What is not covered here

- Warm-start retraining inside the stream (warm start is measured in Part 1, section 1.2).
- Gradual and subtle drift, a positive control with artificial drift, no-drift streams in real time order,
  repeating everything with several seeds.
"""),
]


for name, cells in (('part1b_evidence_antoine.ipynb', PART1), ('part2_other_methods.ipynb', PART2)):
    (HERE / name).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', HERE / name)
