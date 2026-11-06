# Drift detection on a biomedical knowledge-graph stream

Master's thesis, Diana Amirian (supervisor: Prof. Malchiodi). Builds on Antoine Nasra's thesis
and the ADAZOR drift detectors.

A stream of biomedical triples (from papers) arrives over time. Each triple is embedded with a
knowledge-graph embedding model trained on the graph known so far, composed into a feature vector
(Hadamard product or concatenation), and scored by a one-class SVM trained on a reference window.
A Z-test on the share of outliers in the current window signals drift.

## Repository

| Folder | What |
|---|---|
| `phase1/` | Phase 1 experiments: Antoine's sphere retrain rule with 5 embedding models (TransE, TransH, DistMult, ComplEx, RotatE), Hadamard and concatenation, on CASE1, CASE2 and the full stream; always-retrain reference; sphere-size sweep; MRR; peak analysis. See `phase1/README.md`. |
| `adazor/` | ADAZOR package (drift detectors), by Prof. Malchiodi |
| `reference/` | Notebooks the code is based on, unchanged: `antoine_sphere_rule_edition2.ipynb` (Antoine, sphere rule, the Phase 1 baseline) and `edition1_dynamic_embeddings.ipynb` (first edition, mean-distance rule) |
| `data/` | Not in the repository (too large). See below. |

Earlier work (September-October) is in a separate repository:
[adazor-biomedical-drift](https://github.com/Dianaamrn/adazor-biomedical-drift).

## Data

Put these files in `data/` (they are ignored by git):

| File | Size | What |
|---|---|---|
| `drifting_edges_timestamp.tsv` | 253 MB | full biomedical stream (1.72M triples) |
| `filtered_drifting_edgesCASE1.tsv` | 69 MB | CASE1: cancer before 2005, infectious disease from 2005 |
| `filtered_drifting_edgesCASE2.tsv` | 69 MB | CASE2: infectious disease before 2022, cancer from 2022 |

## Running on the GPU server

```
git clone https://github.com/Dianaamrn/Thesis.git ~/Thesis
mkdir -p ~/Thesis/data      # then copy the three data files here
source ~/drift/.venv/bin/activate
cd ~/Thesis/phase1 && bash run_overnight.sh     # inside tmux
```

Requirements: Python 3.12, pykeen 1.11, torch 2.6 (CUDA), scikit-learn, pandas, numpy, scipy.
