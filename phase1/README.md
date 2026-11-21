# Antoine's method: five ways to re-embed, four retraining rules, 5 embedding models, 3 datasets

Port of `biomedical_ocsvm_ztest_experiment (6).ipynb` (Antoine). The stream loop, the sphere
(center-radius) retrain rule, keep-10% after drift and the Z-test are his. As in his notebooks, the graph is
built from every triple (all types) up to the cutoff year plus the undated ones, and from then on one triple
type is monitored. The full data is used whole, as in his first notebook (cutoff 2010, Gene-Disease).
The plan with the reasons and sources for every choice: `reports/Experiment plan 2026-11-19.md`.

## Versions: how the graph is re-embedded

| Version | Re-embedding | Runs folder |
|---|---|---|
| `v1` | from scratch, learning rate and epochs chosen by validation MRR (`configs.TRAINING`) | `runs/` |
| `v2_e2` | from scratch, 2 epochs for every model (each model's chosen learning rate) | `runs_v2_e2/` |
| `v2_e10` | from scratch, 10 epochs for every model | `runs_v2_e10/` |
| `v2_e20` | from scratch, 20 epochs for every model | `runs_v2_e20/` |
| `v3` | warm start: the first embedding as in v1; every later re-embed starts from the previous vectors (a new entity from PyKEEN's random vector) and trains a few epochs over the whole graph; epochs and learning rate per model and composition from `warm_check/<dataset>__chosen.csv` | `runs_v3/` |

Antoine's own setting (from scratch, 2 epochs, learning rate 0.001 for every model) is in `runs_2epochs/` for CASE1,
and his exact settings (32 dims, gamma 0.5, no scaling) are the correctness check: `runs_antoine_check/cpu/` (on a CPU,
like his own run; graded) and `runs_antoine_check/` (on the GPU, like every experiment run; reported as a finding).

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

Which rule runs on which dataset per version: `configs.PLAN` (everything everywhere, except v1 on the full
data: only Rules 1 and 2, already finished).

## How to run (server, inside tmux)

Everything, in order (checks, warm-start check, MRR, cases, analysis, full data, sensitivity, analysis):

    tmux new -s all
    cd ~/Thesis/phase1 && bash run_all.sh all

Or stage by stage: `bash run_all.sh checks|warm|quality|cases|import|full|sensitivity|analysis`.
Finished runs are skipped, so any stage can be restarted. `import` reuses the earlier 10-epoch full-data runs
from `~/phase1_old_lr0001.tgz` (TransE, TransH, DistMult, RotatE; checked setting by setting).

Single pieces: `python run_grid.py --version v3 --grid sphere always --datasets case1 --parallel 3`,
`python drift_sphere.py --dataset case1 --model TransE --version v3 --retrain-policy every --retrain-every 50`.

## Settings that are the same everywhere

64 real numbers per vector (ComplEx/RotatE: 32 complex), batch 2048, seed 42, L2-normalised vectors,
StandardScaler fitted on the reference, OCSVM RBF nu 0.1 gamma 'scale', window 250, alpha 0.025,
keep 10% after drift, adazor `ZTestDriftDetector`, GPU. The sensitivity runs change one of nu, window,
alpha, keep fraction, seed at a time (`configs.SENSITIVITY`), on the cases, version `v2_e20`, Rule 2.

Embedding training rule (v1, and the first embedding of v3): PyKEEN early stopping on validation MRR.
The cutoff KG is split 96/2/2 (train / validation / test); validation MRR every 10 epochs; stop after
2 checks without a 1% improvement (max 100 epochs); learning rates 0.001, 0.01, 0.05. Each model keeps
the learning rate with the best MRR and the epochs where it was reached. The reported MRR
(`embedding_quality.py`) uses the test part.

Warm start rule (v3, `warm_check.choose_warm`, criteria fixed before any measurement): after adding one window
of stream triples, candidates are 1, 2, 3, 5, 10 epochs at the chosen learning rate and one fifth of it. A
candidate must keep >= 95% of the old triples' validation MRR and raise the old OCSVM's outlier rate on the
reference by at most 0.05; among those, the best MRR on the added triples, then the fewest epochs. If none
passes, the gentlest update (1 epoch, lower rate) is taken and marked `passes = False`. The choice is made per
model and composition (since 19 Nov: with Hadamard no setting kept the map, which first gave concatenation
the gentlest update too). `python warm_check.py --dataset <d> --choose-only` redoes the choice from the saved
measurements without training.

## Outputs

- `checks/checks.csv`: the correctness checks (`checks.py`)
- `settings_check/`: MRR curve per dataset/model/learning rate; `<dataset>__chosen.csv`
- `warm_check/`: warm-start measurements per dataset and model; `<dataset>__table.csv`, `<dataset>__chosen.csv`
- `<runs folder>/<run>/`: summary.json, config.json, drifts.tsv, window_history.tsv.gz, step_trace.tsv.gz,
  embedding_shift_metrics.tsv (one row per re-embed, with the sphere ratio), progress.json
  step_trace has one row per stream triple. Rows with `refill = True` are the triples that refill the
  window after an alarm; the method does not test them. Their `shadow_*` columns record what the OCSVM in
  use at the alarm says about them (logging only: the alarms are the same with or without it).
- `<runs folder>/_sweep_plan.csv`: the sphere sizes chosen for Rule 3
- `quality/`: MRR per dataset and model (`__phase1` = v1 settings, `__v2e2|__v2e10|__v2e20` = fixed epochs)
- `analysis/summary_table.csv|.md`: one row per run (alarms before/after the change, delay, whether the change
  triple was tested, share of the stream tested, re-embeds, retrains, peaks, MRR, run time);
  `analysis/version_comparison.csv|.md`: the versions side by side
- `analysis/peaks/<run>.csv`, `analysis/peaks_all.csv`: one row per peak (date, file row, triple,
  disease name and group, triples with the same date, distance to the switch and to the highest peak)
- `analysis/literature/<dataset>__peak_dates.csv` (one row per peak date over all runs) and
  `<dataset>__peak_date_triples.csv` (every monitored triple of that date): for finding the papers
- `analysis/figures/`: one outlier-rate plot per run with alarms, peaks and the untested refill stretches
  (light blue); alarm positions vs the switch per version and rule; retrain share vs false alarms / delay
- `analysis_sensitivity/`: the same for the sensitivity runs
- `notebooks/checks.ipynb`, `notebooks/results.ipynb`: executed on the server by `run_all.sh analysis`

The change ("switch") is placed at the first stream triple of the switch year in the data (CASE1: stream
triple 567), whether or not a run tested it (`switch_tested`).

## Files

| File | What it does |
|---|---|
| `configs.py` | datasets, settings, versions, rules, plan, learning rate + epochs (`TRAINING`), sensitivity, Antoine check |
| `checks.py` | correctness checks of the code against the method |
| `settings_check.py` | chooses learning rate + epochs per model (early stopping on MRR) |
| `warm_check.py` | measures a warm re-embed at 1, 2, 3, 5, 10 epochs against a re-embed from scratch; chooses v3's settings |
| `drift_sphere.py` | one run: `--dataset --model --composition --version --retrain-policy` |
| `run_grid.py` | runs a grid (rules of one version, `antoine`, `sensitivity`) N at a time, skips finished runs, chooses the Rule 3 sizes |
| `run_all.sh` | every experiment in order, stage by stage |
| `import_old_runs.py` | reuses the earlier 10-epoch full-data runs as `v2_e10` |
| `analyze.py` | tables, peak tables, literature tables, plots, version comparison |
| `embedding_quality.py` | MRR / Hits@10 with the settings of a from-scratch version |
| `notebooks/build_notebooks.py` | writes `checks.ipynb` and `results.ipynb` (`results_helpers.py` draws the tables and figures) |
| `case1_evidence.ipynb` | evidence notebook on CASE1 (v1 runs) |
