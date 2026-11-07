"""Phase 2 settings (settings.py; Phase 1 has its own configs.py). Every number used by Phase 2 is set here.

Phase 2 has two parts (see phase2/README.md and the notebooks in phase2/notebooks/):

PART 1  Evaluating Antoine's method, with evidence
  1.1 the sphere rule (almost) never fires: geometry + Phase 1 ratios + the sphere-size sweep
  1.2 embeddings retrained from scratch cannot be compared directly: seed test (several seeds, a
      grown graph, an identical rerun, a warm start), with and without alignment
  1.3 the alarms come from re-embedding: the same method with the embedding held fixed
  1.4 the method fires when nothing changes: no-drift streams
  1.5 its statistics are biased: fair (held-out) reference, then a threshold set for repeated tests
PART 2  What other researchers do, tested on the cases (one known drift each)
  2.1 retrain, then align the new embedding onto the old one (orthogonal Procrustes)
  2.2 fixed embedding + fold-in for new entities
  2.3 fair reference + calibrated threshold
  2.4 simple measures of change: triple types, new entities, MONDO categories, plausibility, MMD
  2.5 graph distances year by year
  2.6 every method judged the same way: no-drift and injected-switch streams, threshold-free comparison,
      peak analysis; plus an embedding-quality check (TransE trained until MRR is reasonable)
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
KEEP_FRACTION_AFTER_DRIFT = 0.10          # Antoine's refill rule (stream methods)

# Calibrated thresholds: at most BETA chance of any false alarm in HORIZON triples of no-drift data,
# found by simulating no-drift streams resampled from the calibration half of the reference. For the
# conformal OCSVM the simulation also draws the detector's no-drift flag rate from its Beta distribution
# (calibration-sample variation; lib_detect.conformal_flag_sampler).
CALIBRATION = {'horizon': 1000, 'beta': 0.10, 'n_boot': 200}

# Monte Carlo: number of no-drift control streams and of injected-switch streams per dataset.
# delay_grid: delays (triples after the switch) at which the running maximum of each statistic is saved,
# for the threshold-free comparison.
MONTE_CARLO = {'control': 100, 'inject': 100, 'post_len': 1000, 'seed': 0, 'delay_grid': list(range(0, 1001, 25))}

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
                              'what': 'OCSVM outlier rate, in-sample reference rate, z/sqrt(2n) threshold (as Antoine)',
                              'test': 'ADAZOR Z-test'},
    'C5_conformal_fixed':    {'per_model': True, 'per_composition': True, 'stride': 1,
                              'what': 'OCSVM outlier rate, conformal reference rate, z/sqrt(2n) threshold',
                              'test': 'ADAZOR Z-test'},
    'C5_conformal_calibrated': {'per_model': True, 'per_composition': True, 'stride': 1,
                                'what': 'OCSVM outlier rate, conformal reference rate, calibrated threshold',
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

# 1.2: seed / comparability test. warm_epochs = epochs of continued training for the warm start.
SEED_TEST = {'dataset': 'case1', 'seeds': [42, 43, 44, 45, 46], 'knn': 10, 'knn_sample': 2000,
             'cross_pairs': 10, 'warm_epochs': 10}

# 1.3, 1.4, 2.1, 2.2: Antoine's stream loop with three ways of handling the embedding.
#   original  re-embed from scratch on every new entity (Antoine, as in Phase 1)
#   aligned   re-embed from scratch, then rotate the new embedding onto the previous one
#   frozen    embedding trained once, new entities folded in
# Methods that retrain are slow (a training per new entity), so they get fewer repeats and models.
STREAM = {
    'methods': ['original', 'aligned', 'frozen'],
    'retrain_models': ['TransE'],          # no-drift / injected repeats of the retraining methods use one model
    'retrain_compositions': ['hadamard'],  #   and one composition (Phase 1: all models and compositions behave alike)
    'retrain_repeats': 10,                 # no-drift and injected-switch streams each, per retraining method
    'frozen_repeats': 20,                  # same, for the frozen method (cheap), all models
}

# 2.6: embedding-quality check. MRR is measured at several training lengths; the key comparisons are
# repeated with TransE trained for quality_epochs.
QUALITY = {'dataset': 'case1', 'model': 'TransE', 'epoch_grid': [10, 25, 50, 100, 200], 'quality_epochs': 100,
           'test_sample': 2000}

# 2.5: graph distances year by year (C6).
GRAPH = {'deltacon_groups': 10, 'deltacon_iterations': 30, 'permutations': 999}


def feature_dim(composition):
    return KGE['real_dim'] * (3 if composition == 'concatenation' else 1)


def epochs_suffix(epochs=None):
    """File/run name suffix for a non-default training length ('' for the Phase 1 default)."""
    return '' if epochs in (None, KGE['epochs']) else f'__e{epochs}'
