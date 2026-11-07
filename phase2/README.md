# Phase 2: is the drift in the data, or in the method?

Phase 1 showed that Antoine's method raises about one alarm per re-embed, identically for all five
models. Phase 2 freezes the reference, measures change on several channels with calibrated thresholds,
and compares everything with Antoine's method on no-drift and known-drift streams.

Background: `reports/Phase 2 research explained.md` (plain language), `reports/Graph change measurement in
streams.md` (literature, section c = experiments), `notebooks/audit_antoine_baseline.ipynb` (12 problems in
the baseline).

## Run it (server, inside tmux, from the repository root)

```bash
bash phase2/run_phase2.sh 1      # cases: frozen embeddings, A1, channels, graph distances -> core results
bash phase2/run_phase2.sh 2      # seed test + A0 on no-drift controls (GPU)
bash phase2/run_phase2.sh 3      # full data, real order
```

Each chunk ends with `step6_analyze.py` and writes `~/phase2_results.tgz` (the tables for the notebook).
Restarting skips finished pieces. Progress: `tail -f phase2/outputs/run_phase2.log`.
Phase 1's block A runs (`phase1/runs/`) are read for the A0-vs-A1 comparison; nothing else from Phase 1 is
needed, so Phase 2 can run before Phase 1 finishes (the A0 rows just fill in later: rerun step 6).

To see the results: copy `~/phase2_results.tgz` to the laptop, unpack it in the repository root, open
`phase2/notebooks/phase2_results.ipynb` and run all cells.

## Files

| File | What it does | Where it runs |
|---|---|---|
| `settings.py` | every setting (same embedding settings as Phase 1) | |
| `lib_data.py` | stream, reference pool, no-drift / injected-switch splits, MONDO categories | |
| `lib_kge.py` | train a frozen KGE once, fold in new entities, score and featurise triples | |
| `lib_detect.py` | channels, window statistics, conformal p-values, calibration, alarm rule | |
| `step0_get_mondo.py` | download the MONDO ontology once (`data/mondo.obo`) | CPU |
| `step1_prepare.py` | train each embedding once, fold-in, save per-triple features and scores | GPU |
| `step2_frozen.py` | A1 (Antoine's method with a frozen embedding); A0 on no-drift controls | CPU / GPU for A0 |
| `step3_channels.py` | channels C1-C5 on real, no-drift and injected-switch streams | CPU |
| `step4_seed_test.py` | how different are two trainings of the same graph | GPU |
| `step5_graph_distance.py` | yearly graph distances (C6) | CPU |
| `step6_analyze.py` | collect everything into `outputs/results/` | CPU |
| `run_phase2.sh` | all of the above, in three chunks | |
| `notebooks/phase2_results.ipynb` | the results, for the professor (built by `build_results_notebook.py`; figures and tables in `results_helpers.py`) | laptop |
| `notebooks/audit_antoine_baseline.ipynb` | what is wrong in the baseline | laptop or server |

## What each experiment answers

| Exp | Question | Script | Notebook section |
|---|---|---|---|
| 2 | Do A0's alarms disappear when the embedding is frozen (A1)? | step 2 | 1 |
| 1 | How often does each method / channel fire when nothing changes? | steps 2, 3 | 2 |
| 3 | How different are two trainings of the same graph; what does a re-embed do to the OCSVM? | step 4 | 3 |
| 11, 12 | What do the conformal reference rate and the calibrated threshold each fix? | step 3 | 4 |
| 5-9 | Which channel detects the switch, how fast, with how many false alarms? | step 3 | 5, 6 |
| 10 | Do yearly graph distances locate the switch? | step 5 | 7 |

Not implemented (lower priority in the report): Exp 4 warm start, Exp 13 MIDAS burst baseline.

## Design decisions (and why)

- **Frozen embedding, trained once** with the Phase 1 settings (64 dims, 10 epochs, seed 42).
- **Fold-in**: a new entity gets its own vector, fitted (300 Adam steps, the model's own score function,
  16 random negatives per side) on the triple where it first appears; every old vector stays exactly
  where it was (tested). In the stream's real order this is what an online system can do.
- **Two variants of the frozen setup.** `full` = trained on the whole cutoff graph, used by A1 so that A0 -> A1
  changes only the re-embedding. `holdout` = trained without the 1,000 most recent monitored triples, which
  form the reference pool, so reference and stream triples are equally unseen by the embedding; used by
  the channels and A2.
- **Reference** = 500 triples, shuffled, split 250 to fit / 250 to calibrate (conformal p-values and
  threshold simulation).
- **Calibrated thresholds**: at most 10% chance of any false alarm in 1,000 no-drift triples, from 200
  simulated streams resampled from the calibration half (Cobb et al. 2022). Fixed thresholds =
  ADAZOR's z/sqrt(2n), as in Phase 1.
- **Alarm rule**: test once a full window of 250 is available; after an alarm the window starts empty.
- **No-drift controls**: the pre-switch triples in random order. **Injected switch**: shuffled pre-switch
  triples, then up to 1,000 post-switch triples, switch at a random position. 100 of each per dataset.
- **New-entity channel (C2)** counts triples that bring an entity for the first time (A0's re-embed
  trigger), one-sided.
- **MONDO categories (C3)**: two levels below "human disease" (e.g. cancer or benign tumor; disease of
  primarily extrinsic mechanism), from one downloaded release whose version is saved with the results.

Something to know when reading the results: in the CASE streams almost every disease first appears
after the cutoff (diseases occur almost only in the monitored triple types). So most monitored triples
use at least one folded-in vector, and that is also why A0 re-embeds so often.

## Smoke test (laptop, no GPU, minutes)

```bash
export PHASE2_OUT=/tmp/p2out    # keeps test outputs away from real ones
python phase2/step1_prepare.py --dataset case1 --model TransE --embedder random
python phase2/step2_frozen.py --dataset case1 --models TransE --a1-repeats 2
python phase2/step3_channels.py --dataset case1 --models TransE --control 2 --inject 2
python phase2/step4_seed_test.py --models TransE --seeds 42 43 --embedder random
python phase2/step5_graph_distance.py --dataset case1
python phase2/step6_analyze.py
```
