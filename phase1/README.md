# Antoine's method in three versions, with four retraining rules and 5 embedding models

Port of `biomedical_ocsvm_ztest_experiment (6).ipynb` (Antoine). The stream loop, the sphere
(center-radius) retrain rule, keep-10% after drift and the Z-test are his. As in his notebook, the KG is
built from every triple up to the cutoff year, and from then on one triple type is monitored.

## Versions: how the graph is re-embedded

| Version | Re-embedding | Runs folder |
|---|---|---|
| `v1` | from scratch, learning rate and epochs chosen by validation MRR (`configs.TRAINING`) | `runs/` |
| `v2` | from scratch, 20 epochs for every model (learning rate as in v1) | `runs_v2/` |
| `v3` | warm start: the first embedding as in v1; every later re-embed starts from the previous vectors (a new entity from PyKEEN's random vector) and trains a few epochs over the whole graph (`configs.WARM`) | `runs_v3/` |

Antoine's own setting (from scratch, 2 epochs) is in `runs_2epochs/` for CASE1.

## Rules: when the OCSVM is retrained

| Rule | Grid name | Retrains the OCSVM |
|---|---|---|
| 1 | `sphere` | at a new entity, if Antoine's ratio is above 1 |
| 2 | `always` | at every new entity |
| 3 | `sweep` | sphere rule with smaller sizes: per dataset and composition, three sizes taken from the ratios of the version's always runs, so the rule retrains at about 10%, 50%, 90% of new entities |
| 4 | `every` | after every X stream triples (X = 10, 50, 250), on the same reference triples |

After a drift alarm every rule re-embeds and retrains (Antoine's code). In Rule 4, if no re-embed
happened since the last fit, the same triples and vectors would give the same OCSVM, so the fit is not
repeated (`num_scheduled_retrains` counts the schedule, `num_scheduled_refits` the real fits).

## Datasets

| Dataset | File | Monitored type | Cutoff | Switch |
|---|---|---|---|---|
| `case1` | filtered_drifting_edgesCASE1.tsv | Gene - causes_or_contributes_to_condition - Disease | 2002 | 2005 (cancer -> infectious disease) |
| `case2` | filtered_drifting_edgesCASE2.tsv | miRNA - causes_or_contributes_to_condition - Disease | 2018 | 2022 (infectious disease -> cancer) |
| `full` | drifting_edges_timestamp.tsv | Gene - causes_or_contributes_to_condition - Disease | 2010 | none |

CASE2 has only 235 Gene-Disease triples before 2022, too few for a reference plus a window of 250,
so its monitored type is miRNA-Disease (476 triples up to the cutoff, 1,020 before the switch, 2,809 after).

Which rule runs on which dataset per version: `configs.PLAN`. On the full data v1 keeps Rules 1 and 2
(finished); Rules 3 and 4 run there with v3, and with v2 only when asked (`--datasets full`).

## How to run (server, inside tmux)

1. `python settings_check.py --dataset <dataset>`: learning rate and epochs per model (done for case1,
   case2, full; values are in `configs.TRAINING`).
2. `python warm_check.py --dataset <dataset>`: measurements for choosing the warm-start epochs and
   learning rate. Put the chosen values into `configs.WARM` (needed for v3 only).
3. `bash run_versions.sh v1` (or `v2`, `v3`): all planned runs of the version, MRR, analysis, then
   `~/results_<version>.tgz`. Finished runs are skipped.

Single pieces: `python run_grid.py --version v3 --grid sphere always --datasets case1 --parallel 3`,
`python drift_sphere.py --dataset case1 --model TransE --version v3 --retrain-policy every --retrain-every 50`.

## Settings that are the same everywhere

64 real numbers per vector (ComplEx/RotatE: 32 complex), batch 2048, seed 42, L2-normalised vectors,
StandardScaler fitted on the reference, OCSVM RBF nu 0.1 gamma 'scale', window 250, alpha 0.025,
keep 10% after drift, adazor `ZTestDriftDetector`, GPU.

Embedding training rule (v1, and the first embedding of v3): PyKEEN early stopping on validation MRR.
The cutoff KG is split 96/2/2 (train / validation / test); validation MRR every 10 epochs; stop after
2 checks without a 1% improvement (max 100 epochs); learning rates 0.001, 0.01, 0.05. Each model keeps
the learning rate with the best MRR and the epochs where it was reached. The reported MRR
(`embedding_quality.py`) uses the test part.

## Outputs

- `settings_check/`: MRR curve per dataset/model/learning rate; `<dataset>__chosen.csv`
- `warm_check/`: warm-start measurements per dataset and model; `<dataset>__table.csv`
- `<runs folder>/<run>/`: summary.json, config.json, drifts.tsv, window_history.tsv.gz, step_trace.tsv.gz,
  embedding_shift_metrics.tsv (one row per re-embed, with the sphere ratio), progress.json
- `<runs folder>/_sweep_plan.csv`: the sphere sizes chosen for Rule 3
- `quality/`: MRR per dataset and model (`__phase1` = v1 settings, `__v2` = 20 epochs)
- `analysis/summary_table.csv|.md`: one row per run (alarms, re-embeds, retrains, false alarms, delay,
  peaks, density, MRR); `analysis/version_comparison.csv|.md`: the versions side by side
- `analysis/peaks/<run>.csv`, `analysis/peaks_all.csv`: one row per peak (date, file row, triple,
  disease name and group, triples with the same date, distance to the switch and to the highest peak)
- `analysis/figures/`: one outlier-rate plot per run with alarms and peaks, alarm positions vs the switch,
  retrain share vs false alarms / delay per rule

## Files

| File | What it does |
|---|---|
| `configs.py` | datasets, settings, versions, rules, plan, learning rate + epochs (`TRAINING`), warm start (`WARM`) |
| `settings_check.py` | chooses learning rate + epochs per model (early stopping on MRR) |
| `warm_check.py` | measures a warm re-embed at 1, 2, 3, 5, 10 epochs against a re-embed from scratch |
| `drift_sphere.py` | one run: `--dataset --model --composition --version --retrain-policy` |
| `run_grid.py` | runs the rules of one version N at a time, skips finished runs, chooses the Rule 3 sizes |
| `run_versions.sh` | everything planned for one version |
| `analyze.py` | tables, peak tables, plots, version comparison |
| `embedding_quality.py` | MRR / Hits@10 with the settings of v1 or v2 |
| `case1_evidence.ipynb` | evidence notebook on CASE1 (v1 runs) |
