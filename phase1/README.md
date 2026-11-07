# Phase 1: Antoine's sphere rule vs always retraining, with 5 embedding models

Port of `biomedical_ocsvm_ztest_experiment (6).ipynb` (Antoine). The stream loop, the sphere
(center-radius) retrain rule, keep-10% after drift and the Z-test are his. As in his notebook, the KG is
built from every triple up to the cutoff year, and from then on one triple type is monitored:
**Gene - causes_or_contributes_to_condition - Disease**.

## How to run (server, inside tmux)

1. `python settings_check.py --dataset case1` (then `full`): chooses the learning rate and
   epochs of every model (see below). Copy the chosen values into `TRAINING` in `configs.py`.
2. `bash run_phase1.sh`: all runs, MRR, analysis, then `~/phase1_results.tgz`.

## Embedding training: one rule for all models

Every model is trained with the same procedure, PyKEEN's early stopping on validation MRR:
split the cutoff KG 96/2/2 (train / validation / test), check the validation MRR every 10 epochs, stop after 2 checks
without a 1% improvement (max 100 epochs), for learning rates 0.001, 0.01 and 0.05. Each model keeps
the learning rate with the best MRR and the epochs where it was reached. A model passes if its MRR is
at least 10 x random guessing. The reported MRR (`embedding_quality.py`) uses the test part, never used for choosing. Every re-embed in a run then uses these fixed values.

Why: with one fixed setting for all models (PyKEEN's default learning rate 0.001, 10 epochs),
ComplEx learned nothing (MRR at the level of random guessing). Those earlier runs are archived
outside the repository (`phase1_old_lr0001.tgz`).

## What runs (`run_phase1.sh`)

Other settings, the same for every run: 64 real numbers per vector (ComplEx/RotatE: 32 complex),
batch 2048, seed 42, L2-normalised vectors, StandardScaler fitted on the reference, OCSVM RBF nu 0.1
gamma 'scale', window 250, alpha 0.025, keep 10% after drift, adazor `ZTestDriftDetector`, GPU.

| Grid | Retrain rule on a new entity | Datasets | Runs |
|---|---|---|---|
| `sphere` | Antoine's sphere rule, ratio > 1 | case1, full | 20 |
| `always` | always retrain the OCSVM | case1, full | 20 |
| `sweep` | sphere rule with k set from the `sphere` runs' ratios so it retrains ~10/50/90% of new entities | case1 | 30 |
| MRR | link-prediction quality per model, with the run settings | case1, full | 10 trainings |

Each grid is 5 models (TransE, TransH, DistMult, ComplEx, RotatE) x 2 compositions (hadamard,
concatenation). After a drift alarm every rule re-embeds and retrains (Antoine's code).

| Dataset | File | Stream | Cutoff | Switch |
|---|---|---|---|---|
| `case1` | filtered_drifting_edgesCASE1.tsv | Gene-Disease | 2002 | 2005 (cancer -> infectious disease) |
| `full` | drifting_edges_timestamp.tsv | Gene-Disease | 2010 | none |

CASE2 (infectious -> cancer, 2022) is not part of Phase 1.

## Outputs

- `settings_check/`: MRR curve per dataset/model/learning rate; `<dataset>__chosen.csv`
- `runs/<run>/`: summary.json, config.json, drifts.tsv, window_history.tsv.gz, step_trace.tsv.gz,
  embedding_shift_metrics.tsv (one row per re-embed, with the sphere ratio), progress.json
- `runs/_sweep_plan.csv`: the k values chosen for the sweep
- `quality/`: MRR per dataset and model
- `analysis/summary_table.csv|.md`: one row per run (alarms, re-embeds, retrains, false alarms,
  delay, peak, density, MRR); `analysis/alarm_offsets.csv`
- `analysis/figures/`: one outlier-rate plot per run, alarm positions vs the switch,
  retrain share vs false alarms / delay per rule

## Files

| File | What it does |
|---|---|
| `configs.py` | datasets, settings, learning rate + epochs per model (`TRAINING`), grids |
| `settings_check.py` | chooses learning rate + epochs per model (early stopping on MRR) |
| `drift_sphere.py` | one run: `--dataset --model --composition --retrain-policy` |
| `run_grid.py` | runs a grid N at a time, skips finished runs, builds the sphere sweep |
| `run_phase1.sh` | all of Phase 1 |
| `analyze.py` | tables, peak analysis and plots |
| `embedding_quality.py` | MRR / Hits@10 with the run settings |
