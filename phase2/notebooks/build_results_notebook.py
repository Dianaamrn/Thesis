"""Writes phase2_results.ipynb (run once after editing this file: python build_results_notebook.py)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': text.strip('\n').splitlines(keepends=True)})


md("""
# Phase 2: is the drift in the data, or in the method?

**Question.** In Phase 1, Antoine's method (re-embed from scratch on every new entity, OCSVM, ADAZOR
Z-test, sphere rule) raised about one alarm per re-embed, the same for all five embedding models. Phase 2
asks how much of that comes from the method, and what a sound monitor looks like instead.

**Approach** (literature: `reports/Graph change measurement in streams.md`; plain-language version:
`reports/Phase 2 research explained.md`):
1. **Freeze the reference.** Train each embedding once. New entities get a *folded-in* vector fitted on the
   triple that introduces them; old vectors never move.
2. **Measure change on several channels**, each with its own test, against a fixed reference.
3. **Calibrate every threshold** to the same false-alarm budget on simulated no-drift streams.
4. **Evaluate on no-drift controls and on the known switch**, repeated many times.

| Method | What it is |
|---|---|
| **A0** | Phase 1 as run: re-embed on every new entity, in-sample reference rate, fixed z/sqrt(2n) threshold |
| **A1** | A0 with one change: the embedding is frozen (fold-in instead of re-embed) |
| **A2** | A1 + conformal reference rate + calibrated threshold (channel C5, last variant) |

| Channel | What it looks at | Test |
|---|---|---|
| C1 | mix of triple types | G statistic |
| C2 | share of triples that bring an entity for the first time | rate increase |
| C3 | MONDO disease categories (one pinned release) | binomial G per category |
| C4 | frozen-KGE plausibility scores; triple vectors | Kolmogorov-Smirnov; MMD |
| C5 | OCSVM outlier rate (ADAZOR) in three variants | Z-test statistic |
| C6 | yearly graphs: diseases, typed WL patterns, DeltaCon | change-point test |

All settings are in `phase2/settings.py` (same embedding settings as Phase 1: 64 dims, 10 epochs, seed 42;
window 250; nu 0.1; alpha 0.025).

**How to run.** The results come from `phase2/run_phase2.sh` on the server. This notebook only reads
`phase2/outputs/results/`; sections whose results are not there yet print a note.
""")

code("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
import results_helpers as R
pd.set_option('display.max_columns', 40, 'display.width', 200)
print('results folder:', R.RESULTS)
for ds, m in R.meta().items():
    print(f"{ds}: MONDO release {m.get('mondo_version')}; embedder {m.get('embedder')}; "
          f"calibration {m.get('calibration')}; Monte Carlo {m.get('monte_carlo')}")
""")

md("""
## 1. Experiment 2: does freezing the embedding remove the alarms?

A0 and A1 are the same method on the same real stream; the only difference is that A1 never re-embeds.
If most A0 alarms disappear in A1, they were caused by re-embedding, not by the data.

**How to read.** `alarms` = drift alarms on the stream; `alarms_pre` / `alarms_post` = before / after the
switch year; `re-embeds` = A0's from-scratch retrainings (always 0 for A1); `alarms within a window of a
re-embed` = share of A0 alarms that come less than one window after a re-embed.
""")
code("""
R.plot_exp2()
R.exp2_table()
""")

md("""
## 2. Experiment 1: what happens when nothing changes?

**No-drift control streams**: the pre-switch triples (one topic) in random order. Every alarm here is a
false alarm. A0 and A1 see exactly the same control streams.
""")
code("""
R.plot_exp1_frozen()
R.exp1_frozen_table()
""")

md("""
The same question for every channel. Calibrated channels were set to have at most 10% chance of any false
alarm in 1,000 no-drift triples; the dashed line is what that means for the control stream length. A
channel far to the right of the line is miscalibrated; the two *fixed-threshold* variants are not
calibrated at all.
""")
code("""
R.plot_exp1_channels()
R.exp1_channel_table()
""")

md("""
## 3. Experiment 3: how different are two trainings of the same graph?

Each model is trained with 5 seeds on the cutoff graph (G0) and on the cutoff graph plus the pre-switch
triples (G1, a grown graph with no topic change). Entities are matched by label.

- **same graph, other seed**: only the random seed differs. This is the noise floor.
- **grown graph, same seed**: exactly what A0 does at a re-embed (seed 42 every time).
- **raw cosine** compares coordinates directly; **CKA** and **neighbour overlap** ignore rotations.

The second figure applies an OCSVM fitted on the reference in one embedding to new triples in the same
embedding, in the re-trained embedding as is, and in the re-trained embedding rotated back
(orthogonal Procrustes). The gap between "same embedding" and "re-trained (raw)" is the outlier rate a
re-embed alone creates.
""")
code("""
R.plot_exp3()
R.exp3_table()
""")

md("""
## 4. Experiments 11 and 12: fixing the reference rate and the threshold, one step at a time

The three C5 variants use the same frozen embedding and the same OCSVM:
1. in-sample reference rate + fixed z/sqrt(2n) threshold (A1's statistics),
2. **conformal** reference rate (held-out calibration half) + fixed threshold,
3. conformal reference rate + **calibrated** threshold (A2).

Step 1 -> 2 shows the effect of the in-sample bias; step 2 -> 3 the effect of calibrating for repeated
tests on overlapping windows.
""")
code("""
R.calibration_table()
""")

md("""
## 5. Experiments 5-9: which channel sees the switch?

**Switch injected at a random position**: shuffled pre-switch triples, then post-switch triples, repeated
with the switch at many positions. For each channel: share of no-drift streams with a false alarm,
share of injected switches detected, and the median delay (triples after the switch until the alarm).
A good channel is on the left in the first panel, on the right in the second and on the left in the third.
""")
code("""
R.plot_channels()
""")
code("""
R.channel_table('case1')
""")
code("""
R.channel_table('case2')
""")

md("""
## 6. The real stream, in time order

Each channel's statistic along the real stream, with its threshold and alarms. The reference is the 500
most recent pre-cutoff triples. Alarms before the switch are false alarms *with respect to the injected
topic switch*; the real stream may also change for other reasons between the reference period and 2005.
""")
code("""
R.plot_real_series('case1', model='TransE', composition='hadamard')
""")
code("""
R.plot_real_series('case2', model='TransE', composition='hadamard')
""")
code("""
R.plot_real_series('full', model='TransE', composition='hadamard')
""")

md("""
## 7. Experiment 10: how much does the graph change from year to year?

Whole-graph distances on a slower clock: one graph per publication year of the monitored stream.
The orange line is the change point found by the test (p-value in the table).
""")
code("""
R.plot_graph()
R.graph_tables()[1]
""")

md("""
## 8. What is not in this notebook

- Experiment 4 (warm-start retraining) and Experiment 13 (MIDAS burst baseline) from the report: lower
  priority, not run.
- Fold-in uses only the triple that introduces an entity. In the CASE streams almost every disease is
  new after the cutoff, so most monitored triples use at least one folded-in vector; the share is
  printed by `step1_prepare.py`.
- Quotes and claims from the literature: see section (d) of the report for what still has to be checked
  against the PDFs.
""")

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = Path(__file__).resolve().parent / 'phase2_results.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', out)
