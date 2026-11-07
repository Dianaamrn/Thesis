"""Phase 2 settings (settings.py; Phase 1 has its own configs.py). Every number used by the Phase 2 scripts is set here.

Phase 2 replaces "re-embed on every new entity" with a fixed (frozen) reference and asks how much
of the Phase 1 alarm pattern came from the method itself. See reports/Phase 2 research explained.md.

Experiments (numbers follow reports/Graph change measurement in streams.md, section c):
  Exp 1   no-drift control streams: how often does each detector fire when nothing changes?
  Exp 2   frozen embedding vs re-embedding: A0 (Phase 1) vs A1 (same method, embedding frozen)
  Exp 3   seed / alignment test: how different are two trainings of the same graph?
  Exp 5-9 monitoring channels C1-C5 on the known switch (real order + switch injected at random points)
  Exp 10  period-level graph distances (C6)
  Exp 11  calibration: fixed z/sqrt(2n) threshold vs threshold calibrated to a false-alarm budget
  Exp 12  conformal reference rate for the OCSVM (A1 -> A2)
Not implemented here (lower priority in the report): Exp 4 warm start, Exp 13 MIDAS burst baseline.
"""
import os
import sys
from pathlib import Path

PHASE2 = Path(__file__).resolve().parent
REPO = PHASE2.parent
sys.path.insert(0, str(REPO / 'phase1'))
import configs as phase1_configs  # noqa: E402

DATA_DIR = REPO / 'data'
# Everything the scripts write (gitignored). PHASE2_OUT redirects it, e.g. for smoke tests.
OUT = Path(os.environ.get('PHASE2_OUT', PHASE2 / 'outputs'))
MODELS_DIR = OUT / 'models'              # trained frozen KGEs
FEATURES_DIR = OUT / 'features'          # per-triple features, scores, novelty (step 1)
RESULTS_DIR = OUT / 'results'            # tables and figures for the notebook (step 6)

MODELS = phase1_configs.MODELS           # TransE, TransH, DistMult, ComplEx, RotatE
COMPLEX_MODELS = phase1_configs.COMPLEX_MODELS
COMPOSITIONS = phase1_configs.COMPOSITIONS  # hadamard, concatenation

# Datasets: same files, cutoffs, streams and switch years as Phase 1.
# holdout = number of the most recent pre-cutoff stream triples kept OUT of the frozen embedding's
#           training graph. They form the reference pool, so the reference and the stream are both
#           triples the embedding has never seen (no in-sample advantage for the reference).
DATASETS = {}
for _name in ('case1', 'case2', 'full'):
    _d = dict(phase1_configs.DATASETS[_name])
    _d['holdout'] = 1000
    DATASETS[_name] = _d
# The full stream has no known switch: only the real-order run is done on it (no controls).

# Embedding: identical to the Phase 1 universal configuration.
KGE = {k: phase1_configs.PRESETS['phase1'][k] for k in ('real_dim', 'epochs', 'batch_size', 'seed')}

# Fold-in: a new entity gets its own vector, fitted on the triple that introduced it, with every
# old vector frozen (Hamaguchi et al. 2017 idea; "hard fold-in" in the report).
FOLDIN = {'steps': 300, 'lr': 0.05, 'negatives': 16, 'seed': 0}

# Detector settings shared by every channel (Phase 1 values where they exist).
WINDOW = 250
REFERENCE = {'size': 500, 'fit': 250}     # reference = 500 triples: 250 to fit, 250 to calibrate
OCSVM = {'nu': 0.1, 'gamma': 'scale', 'standardize': True}
ALPHA = 0.025                             # Phase 1 alpha, used by the fixed z/sqrt(2n) threshold
KEEP_FRACTION_AFTER_DRIFT = 0.10          # A1 only (same as A0)

# Calibrated thresholds: at most BETA chance of any false alarm in HORIZON triples of no-drift data,
# found by simulating no-drift streams resampled from the calibration half of the reference.
CALIBRATION = {'horizon': 1000, 'beta': 0.10, 'n_boot': 200}

# Monte Carlo: number of no-drift control streams and of injected-switch streams per dataset.
MONTE_CARLO = {'control': 100, 'inject': 100, 'post_len': 1000, 'seed': 0}

# Monitoring channels (C1-C5). stride = test every `stride` triples (1 = every triple).
CHANNELS = {
    'C1_types':        {'per_model': False, 'stride': 1,
                        'what': 'mix of triple types (head type - relation - tail type)', 'test': 'G statistic'},
    'C2_novelty':      {'per_model': False, 'stride': 1,
                        'what': 'share of triples that bring an entity for the first time',
                        'test': 'rate increase (one-sided)'},
    'C3_ontology':     {'per_model': False, 'stride': 1,
                        'what': 'MONDO disease categories (pinned release)', 'test': 'binomial G statistic per category'},
    'C4_plausibility': {'per_model': True, 'per_composition': False, 'stride': 5,
                        'what': 'frozen-KGE score of each triple', 'test': 'Kolmogorov-Smirnov distance'},
    'C4_mmd':          {'per_model': True, 'per_composition': True, 'stride': 25, 'pca': 16, 'n_boot': 100,
                        'what': 'triple feature vectors', 'test': 'MMD (RBF, median bandwidth)'},
    'C5_insample_fixed':     {'per_model': True, 'per_composition': True, 'stride': 1,
                              'what': 'OCSVM outlier rate, in-sample reference rate, z/sqrt(2n) threshold (as A0/A1)',
                              'test': 'ADAZOR Z-test'},
    'C5_conformal_fixed':    {'per_model': True, 'per_composition': True, 'stride': 1,
                              'what': 'OCSVM outlier rate, conformal reference rate, z/sqrt(2n) threshold',
                              'test': 'ADAZOR Z-test'},
    'C5_conformal_calibrated': {'per_model': True, 'per_composition': True, 'stride': 1,
                                'what': 'OCSVM outlier rate, conformal reference rate, calibrated threshold (A2)',
                                'test': 'ADAZOR statistic, calibrated'},
}
# Every channel except the two "fixed" ones uses a calibrated threshold.
FIXED_THRESHOLD_CHANNELS = {'C5_insample_fixed', 'C5_conformal_fixed'}

# Ontology (C3). Download once:  python phase2/step0_get_mondo.py
# The release actually used is read from the file's data-version line and saved with the results.
MONDO = {
    'file': DATA_DIR / 'mondo.obo',
    'url': 'http://purl.obolibrary.org/obo/mondo.obo',
    'root': 'MONDO:0700096',          # "human disease"
    'depth': 2,                        # categories = terms 2 levels below the root (e.g. 'cancer or benign tumor')
    'fallback_root': 'MONDO:0000001',  # "disease", for older releases without MONDO:0700096
}

# Exp 3: seed / alignment test.
SEED_TEST = {'dataset': 'case1', 'seeds': [42, 43, 44, 45, 46], 'knn': 10, 'knn_sample': 2000,
             'cross_pairs': 10}

# Exp 10: period-level graph distances (C6).
GRAPH = {'deltacon_groups': 10, 'deltacon_iterations': 30, 'permutations': 999}


def feature_dim(composition):
    return KGE['real_dim'] * (3 if composition == 'concatenation' else 1)
