# Checks

Every check behind the pipeline's decisions, in one place.

| File | What |
|---|---|
| `decision_checks.ipynb` | **The notebook to open.** One section per decision: Decision, Observation, Literature, Check, Conclusion. Small checks are computed in it; big checks are read from their saved outputs. |
| `build_decision_checks.py` | Writes `decision_checks.ipynb` (edit the checks here, then rebuild). |
| `cache/` | Embeddings trained by the notebook (local only, not in git). |
| `executed/` | Executed copies of the notebook (local only, not in git). |

## How to run (server, inside tmux, from the repository root)

```
cd ~/Thesis && source ~/drift/.venv/bin/activate
python checks/build_decision_checks.py
jupyter nbconvert --to notebook --execute checks/decision_checks.ipynb --output-dir checks/executed --ExecutePreprocessor.timeout=-1
```
Then copy `checks/executed/decision_checks.ipynb` to the laptop.
Options: `CHECKS_FULL=1` also reads the full-data step traces (slow); `CHECKS_TEST=1` is a smoke test with a random
stand-in embedding (no training; numbers are meaningless).

## Small checks (computed in the notebook)

| Id | Check | Section | Time |
|---|---|---|---|
| SC1 | Dataset sizes, undated triples, stream entities known only through undated triples | 2 | ~1–2 min (reads the three data files) |
| SC2 | Reverse pairs in the starting graph | 2 | < 1 min |
| SC3 | Re-embedding from scratch changes the coordinate system (two TransE trainings on CASE1, seeds 42/43) | 4 | a few minutes on the GPU (cached after the first run) |
| SC4 | In-sample vs out-of-sample baseline outlier rate (no-drift test, 100 repeats) | 5 | ~1 min after SC3 |
| SC5 | Position of the first alarm in every run | 6 | seconds |
| SC6 | Sphere ratios of every re-embed | 7 | seconds |
| SC7 | Direction of every alarm (more / fewer outliers) | 8 | seconds |
| SC8 | Share of the stream tested; shadow-logged alarms inside the refill | 10 | seconds (cases) |

## Big checks (own scripts; the notebook reads their outputs)

The scripts stay in `phase1/` for now: `run_all.sh` calls them and `configs.py` imports `warm_check.py`, and the server is
running that code. They move into this folder in the agreed repository restructure, after the runs finish.

| Id | Check | Script | Output read by the notebook | Section |
|---|---|---|---|---|
| BC1 | Correctness of the code (48 checks, incl. Antoine's own alarms on CPU) | `phase1/checks.py` | `phase1/checks/checks.csv` | 1 |
| BC2 | Learning rate and epochs per model (validation MRR) | `phase1/settings_check.py` | `phase1/settings_check/<dataset>__chosen.csv` | 3 |
| BC3 | Warm-start updates (1–10 epochs, lr or lr/5) | `phase1/warm_check.py` | `phase1/warm_check/<dataset>__table.csv` | 3 |
| BC4 | Link-prediction quality per version (test MRR) | `phase1/embedding_quality.py` | `phase1/quality/` | — |
| BC5 | Antoine's exact settings (CPU and GPU) | `run_grid.py --grid antoine` | `phase1/runs_antoine_check/` | 1 (via BC1) |
| BC6 | Sensitivity of the detector settings (ν, window, α, keep, seed) | `run_grid.py --grid sensitivity` | `phase1/runs_sensitivity/` | 9 |
| BC7 | Earlier evidence on CASE1 (experiments A, B, C) | `phase1/case1_evidence.ipynb` | — (SC3 and SC4 repeat A and B) | 4, 5 |

## Checks still to be run (section 11)

| Id | Check | Needs |
|---|---|---|
| PC1 | No-drift control on the full data (shuffled stream) | a code option |
| PC2 | Out-of-sample baseline inside the stream | a code option |
| PC3 | First window = training window (as in ADAZOR) | a code option |
| PC4 | Embedding dimension 128 | no new code (`--real-dim 128`) |
| PC5 | Sensitivity on the full data | no new code (`--grid sensitivity --datasets full`) |
