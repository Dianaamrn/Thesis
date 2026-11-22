# Final pipeline notebook — outline (draft, 22 November 2026)

The story of the notebook from top to bottom: headings and the text of each section, no code yet.
Every section says what it shows and where its numbers come from:
**[live]** = computed when the notebook runs · **[loaded]** = read from saved results of scripts run in tmux.
Open points are marked **TO DECIDE**.

---

## 0. Introduction

**Title.** Concept drift detection on a biomedical knowledge-graph stream: the final pipeline.

**Text.** A stream of biomedical triples arrives in time order. The knowledge graph known so far is embedded with a
knowledge-graph embedding model; every monitored triple becomes a feature vector; a one-class SVM fitted on a reference
window flags outliers; the ADAZOR Z-test (Zanaboni, Malchiodi, Nasra 2026) compares the outlier share of a sliding window
with the reference share. The pipeline extends Antoine Nasra's notebook (sphere rule) with five embedding models, two
compositions, four retraining rules and a warm-start re-embedding.

**The final configuration (one table).**
| Part | Choice |
|---|---|
| Re-embedding | warm start: first embedding with MRR-chosen epochs, every update a few epochs from the previous vectors |
| Retraining rule | 4 rules: R1 sphere (k = 1), R2 always, R3 smaller sphere (**TO DECIDE**: 50%), R4 every X triples (**TO DECIDE**: 250) |
| Embedding model | TransE, TransH, DistMult, ComplEx, RotatE |
| Composition | Hadamard, concatenation |
| Detector | OCSVM (RBF, ν = 0.1, γ = 'scale'), StandardScaler, window 250, α = 0.025, keep 10% after an alarm |
| Datasets | CASE1 (cutoff 2002, change 2005), CASE2 (cutoff 2018, change 2022), FULL (cutoff 2010) |

5 models × 2 compositions × 4 rules = 40 runs per dataset.

**Map of the notebook.** Part 1 the pipeline · Part 2 why warm start · Part 3 the checks behind each choice · Part 4 results.

---

# Part 1 — The pipeline, step by step

Each step shows the real function from `phase1/drift_sphere.py` (printed in the cell), then applies it to CASE1 and shows
the result. Same order as Antoine's notebook.

### 1.1 Data
Load the CASE1 file and sort it: undated rows first, then by year, date and file row.
*Shows:* number of rows, columns, the first and last dated rows, the number of undated rows. **[live]**

### 1.2 Starting graph and stream
Starting graph = every triple dated up to the cutoff plus every undated triple, all types. Stream = the monitored type
(Gene–causes_or_contributes_to_condition–Disease) after the cutoff, in time order.
*Shows:* starting-graph size, number of types, monitored triples up to the cutoff, stream size, the switch position
(first triple of 2005). **[live]**

### 1.3 First embedding
TransE trained on the whole starting graph with the learning rate and epochs chosen by validation MRR (Part 3.1).
Vectors are scaled to length 1. ComplEx and RotatE use 32 complex dimensions = 64 real numbers.
*Shows:* training time, number of entities and relations, vector length. **[live]**

### 1.4 Triple → feature vector
Hadamard: s ⊙ p ⊙ o (64 numbers). Concatenation: [s, p, o] (192 numbers).
*Shows:* one triple and its two feature vectors (first few numbers). **[live]**

### 1.5 Reference, first window, one-class SVM
The monitored triples up to the cutoff: the newest 250 = the first window, the rest = the reference (256 on CASE1).
StandardScaler and OCSVM fitted on the reference; baseline outlier rate = share of reference triples outside the boundary.
*Shows:* reference and window sizes, baseline outlier rate (ν = 0.1 is its upper bound on the training points). **[live]**

### 1.6 The Z-test
Alarm when |window outlier share − baseline outlier rate| > z(1 − α/2) / √(2n) = 0.1002 for n = 250, α = 0.025
(ADAZOR, Algorithm 2).
*Shows:* the threshold, computed; ADAZOR's detector and ours agree. **[live]**

### 1.7 The stream loop
For every stream triple: (1) new entity or relation? → warm re-embed of the whole graph; (2) the retraining rule decides
whether the OCSVM is refitted on the same reference triples with the new vectors; (3) the window slides by one;
(4) the window is scored and tested. After an alarm: keep the newest 25 triples, add the next 225 without testing them,
re-embed, these 250 become the new reference, refit the OCSVM.
*Shows:* the code of the loop; a small diagram of the four rules. **[live]**

### 1.8 One complete run, executed here
CASE1, TransE, **TO DECIDE** (proposal: Hadamard and concatenation), warm start, Rule 2.
*Shows:* run time, number of re-embeds and alarms; the outlier share of every tested window with the baseline, the
threshold band, the alarms, the re-embeds and the switch (the plot Antoine's notebook makes). **[live, a few minutes]**

---

# Part 2 — Why warm start

### 2.1 What a from-scratch re-embed does
The same CASE1 graph trained twice (seeds 42 and 43). The OCSVM fitted under the first training scores the same triples
under the second. *Shows:* mean cosine of the same entity in the two trainings; outlier shares before and after. **[live]**
*Literature:* independently trained embeddings live in different coordinate systems (Gursoy et al. 2021; Hu et al. 2022);
the seed alone changes KG embeddings (Méroué et al. 2026).

### 2.2 What it does to the alarms (case runs)
From scratch, Rules 1 and 4 alarm at the same stream triples in almost every run, whatever the model, composition or
number of epochs: every re-embed without a refit raises the window's outlier share to about 1.0. Warm start removes most
of these alarms.
*Shows:* the shared alarm positions; per version and rule, re-embeds without a refit and how many of them alarmed; the
same for warm start. **[loaded: the 800 case runs]**

### 2.3 Conclusion
A re-embed must keep the coordinates (warm start) or be followed by a refit (Rule 2). The final pipeline uses warm start
and keeps all four rules to compare.

---

# Part 3 — The checks behind each choice

### 3.1 First embedding: learning rate and epochs (validation MRR)
Each model trained at learning rates 0.001, 0.01, 0.05 with early stopping on validation MRR (max 100 epochs).
*Shows:* MRR per model and learning rate, chosen setting, "still improving at 100", for all three datasets. **[loaded]**
*Literature:* early stopping on validation MRR (Ruffinelli et al. 2020; Ali et al. 2020); different best learning rate
per model (Bonner et al. 2021); MRR does not measure stability (Méroué et al. 2026).
*Note:* reverse pairs make MRR optimistic (Part 3.6).

### 3.2 Every update: how many epochs
On CASE1, one window of stream triples is added to the starting graph and the previous embedding is trained further for
1, 2, 3, 5, 10 epochs, at the first learning rate and at one fifth of it. Three measures, with limits fixed before
measuring:
1. old knowledge kept — validation MRR of the old graph ≥ 95% of before (catastrophic forgetting);
2. coordinates kept — the old OCSVM's outlier share on the reference rises by ≤ 0.05;
3. new knowledge — MRR on the added triples.
Among the settings that pass 1 and 2, the best on 3, then the fewest epochs.
*Shows:* the table per model and composition, and the chosen setting. **[live on CASE1; CASE2 and FULL loaded]**
*Literature:* update learning rate about one fifth of the first (Wewer et al. 2021); fine-tuning on new facts disrupts old
knowledge (Cui et al. 2022); multiplicative models forget most (Pons et al. 2026); papers update whole snapshots with
10–200 epochs (Wewer 2021; Pons 2025), our updates add one entity at a time.
*Finding to report:* with Hadamard no setting keeps the coordinates on CASE1.

### 3.3 The code implements the method
48 correctness checks, including Antoine's own alarms reproduced with his settings on a CPU. **[loaded]**

### 3.4 Baseline outlier rate: in-sample vs out-of-sample
No-drift test on the CASE1 triples before the cutoff (one embedding, no re-embedding), 100 random splits: alarm rate with
the baseline measured on the OCSVM's training data (ADAZOR) vs on held-out data. **[live]**
*Literature:* ADAZOR Algorithm 1 (baseline on the training window); ν bounds training outliers only (Schölkopf et al.
2001); held-out calibration (Bates et al. 2023; Vovk et al. 2021; Lopez-Paz & Oquab 2016; Rabanser et al. 2018).
*Related observation:* every run alarms at the first test; the first window is new to the OCSVM (in ADAZOR it is the
training window). **[loaded]**

### 3.5 The sphere rule
Logged sphere ratio of every re-embed, from scratch and warm start: the box spans the whole graph, so the ratio stays far
below 1. **[loaded]**

### 3.6 Data facts
Undated triples (share of the starting graph; stream entities known only through them — possible temporal leakage);
reverse pairs (share of the starting graph; none in the monitored type). **[live]**
*Literature:* TGB 2.0 static relations (Gastinger et al. 2024); test leakage from inverse triples (Ali et al. 2020;
Rossi et al. 2021); inverse triples in training (Lacroix et al. 2018).

### 3.7 Untested triples after an alarm
Share of the stream tested; from the shadow logging, how often the detector would have alarmed inside a refill. **[loaded]**

### 3.8 Detector settings: sensitivity
One setting changed at a time (ν, window, α, keep, seed) on the cases. **[loaded]**

### 3.9 Not yet checked
No-drift control on the full data (shuffled stream); out-of-sample baseline inside the stream; dimension 128;
coordinates over many warm updates in a row. Listed with what each would need.

---

# Part 4 — Results of the final pipeline

For each dataset the same blocks; all **[loaded]** from the 40 runs.

### 4.1 Overview
Per model, composition and rule: alarms, re-embeds, refits, share of the stream tested, run time.

### 4.2 Outlier share over the stream
One plot per run (as in 1.8): window outlier share, baseline, threshold band, alarms, re-embeds, switch.
Selectable or as a grid per rule.

### 4.3 Where the alarms pile up
Density of alarms over time (all 40 runs together, then per rule): a histogram by date and by stream position.

### 4.4 Peaks
The peaks of the outlier share: date, stream position, height; how many runs share each peak date.

### 4.5 What is behind each peak
For every peak date: the monitored triples of that date (gene, disease name from MONDO), to look up the papers.

### 4.6 Cases: the known change
CASE1 and CASE2 only: alarms before and after the change, delay of the first alarm after it, triples tested after it.

### 4.7 Full data
Alarms per year; the peak dates and their triples. (Pending until the 40 full-data runs have finished.)

---

# Part 5 — Summary
The final configuration, what each check showed, what the results show, and the limitations (in-sample baseline,
untested refills, undated triples, reverse pairs, 64 dimensions, CPU/GPU differences, many tests in a row).

---

**How it runs.** On the server, inside tmux: `jupyter nbconvert --execute`, outputs saved under every cell; then copied to
the laptop. Expected time (estimate): 20–60 minutes, mostly 1.8 and 3.2. The 40 runs per dataset are run before, as
scripts; Part 4 reads them.
