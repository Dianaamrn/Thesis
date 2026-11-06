# Phase 1: Antoine's sphere rule with 5 embedding models

Port of `biomedical_ocsvm_ztest_experiment (6).ipynb` (Antoine). The stream loop, the sphere
(center-radius) retrain rule, keep-10% after drift and the Z-test are his.
Run everything with `bash run_overnight.sh` (inside tmux).

## What runs

One universal configuration (`phase1` preset) for every run: 64 real numbers per vector
(ComplEx/RotatE: 32 complex), 10 epochs, batch 2048, seed 42, L2-normalised vectors,
StandardScaler fitted on the reference, OCSVM RBF nu 0.1 gamma 'scale', window 250,
alpha 0.025, keep 10% after drift, adazor `ZTestDriftDetector`. Training on the GPU.

| Block | Retrain rule on a new entity | Datasets | Runs |
|---|---|---|---|
| A | Antoine's sphere rule, ratio > 1 | case1, case2, full | 30 |
| B | always retrain (reference) | case1, case2, full | 30 |
| C | sphere rule with k set from block A's ratios so it retrains ~10/50/90% of new entities | case1, case2 (all models); full_short (TransE) | 60 + 6 |
| A/B short | sphere k=1 and always on full_short (TransE) | full_short | 4 |
| MRR | link-prediction quality per model at the cutoff | case1, case2, full | 15 trainings |

After a drift alarm every rule re-embeds and retrains (Antoine's code).

| Dataset | File | Stream | Cutoff | Switch |
|---|---|---|---|---|
| `case1` | filtered_drifting_edgesCASE1.tsv | 11 disease types | 2002 | 2005 |
| `case2` | filtered_drifting_edgesCASE2.tsv | 11 disease types | 2018 | 2022 |
| `full` | drifting_edges_timestamp.tsv | Gene-Disease | 2010 | none |
| `full_short` | same | Gene-Disease, first 2,000 triples | 2010 | none |

## Outputs

- `runs/<run>/`: summary.json, config.json, drifts.tsv, window_history.tsv.gz, step_trace.tsv.gz,
  embedding_shift_metrics.tsv (one row per re-embed, with the sphere ratio), progress.json
- `runs/_sweep_plan.csv`: the k values chosen for block C
- `quality/`: MRR per dataset and model
- `analysis/summary_table.csv|.md`: one row per run (alarms, re-embeds, retrains, false alarms,
  delay, peak, density, MRR); `analysis/alarm_offsets.csv`
- `analysis/figures/`: one outlier-rate plot per run, alarm positions vs the switch,
  retrain share vs false alarms / delay per policy

## Files

| File | What it does |
|---|---|
| `configs.py` | datasets, presets, grids |
| `drift_sphere.py` | one run: `--dataset --model --composition --retrain-policy` |
| `run_grid.py` | runs a grid N at a time, skips finished runs, builds the sphere sweep |
| `run_overnight.sh` | all of Phase 1, unattended |
| `analyze.py` | tables and plots |
| `embedding_quality.py` | MRR / Hits@10 |
