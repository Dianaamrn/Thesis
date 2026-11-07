"""Phase 1 settings: datasets, embedding training settings and run grids.

Everything that differs between runs lives here, so the experiment script itself
stays a straight port of Antoine's notebook
(biomedical_ocsvm_ztest_experiment (6).ipynb).

Phase 1 (preset 'phase1'):
  sphere   Antoine's sphere rule (k = 1)         5 models x 2 compositions x {case1, case2, full}  30 runs
  always   always retrain the OCSVM on a new entity  5 models x 2 compositions x {case1, case2, full}  30 runs
  sweep    sphere size on case1: k chosen from the 'sphere' runs' logged ratios so the rule
           retrains on ~10%, ~50%, ~90% of new entities      5 models x 2 compositions x 3       30 runs
  plus MRR per model and dataset, and the peak analysis (analyze.py).
After a drift alarm every policy re-embeds and retrains (Antoine's code).

Embedding training: every model is trained with the same procedure, PyKEEN early stopping on
validation MRR (settings_check.py). That gives each model and dataset its own learning rate and
number of epochs, stored in TRAINING below. Every re-embed in a run uses those fixed values.
"""

GENE_DISEASE = 'Gene - causes_or_contributes_to_condition - Disease'

# Disease-related triple types (Diana's list). Types missing from a file simply match nothing.
DISEASE_TYPES = [
    'Disease - has_material_basis_in_germline_mutation_in - Gene',
    'Disease - has_material_basis_in_germline_mutation_in - miRNA',
    'Disease - has_material_basis_in_somatic_mutation_in - miRNA',
    'Gene - causes_or_contributes_to_condition - Disease',
    'Gene - is_causal_germline_mutation_in - Disease',
    'miRNA - causes_or_contributes_to_condition - Disease',
    'miRNA - expressed_in - Disease',
    'miRNA - is_causal_germline_mutation_in - Disease',
    'miRNA - is_causal_somatic_mutation_in - Disease',
    'miRNA - over_expressed_in - Disease',
    'miRNA - under_expressed_in - Disease',
]

# switch_year = first year of the injected drift (None = no known drift).
# max_stream = stop after this many stream triples (None = whole stream).
DATASETS = {
    # The drift in the cases is defined on diseases (cancer <-> infectious), so the stream is
    # every disease-related triple type. Same stream definition for both cases.
    'case1': {
        'file': 'filtered_drifting_edgesCASE1.tsv',
        'cutoff_year': 2002,
        'target_type': DISEASE_TYPES,
        'switch_year': 2005,
        'max_stream': None,
        'note': 'CASE1: cancer before 2005, infectious disease from 2005. Cutoff as in Antoine.',
    },
    'case2': {
        'file': 'filtered_drifting_edgesCASE2.tsv',
        'cutoff_year': 2018,
        'target_type': DISEASE_TYPES,
        'switch_year': 2022,
        'max_stream': None,
        'note': 'CASE2: infectious disease before 2022, cancer from 2022 (~70 disease triples up to 2002, so cutoff 2018).',
    },
    'full': {
        'file': 'drifting_edges_timestamp.tsv',
        'cutoff_year': 2010,
        'target_type': GENE_DISEASE,
        'switch_year': None,
        'max_stream': None,
        'note': 'Full biomedical stream, Gene-Disease, cutoff 2010 (no known drift).',
    },
}

MODELS = ['TransE', 'TransH', 'DistMult', 'ComplEx', 'RotatE']
COMPLEX_MODELS = {'ComplEx', 'RotatE'}
COMPOSITIONS = ['hadamard', 'concatenation']

# real_dim = number of real values per entity vector. Complex models get real_dim // 2
# complex dimensions, so every model produces vectors of the same length.
PRESETS = {
    # Phase 1. Learning rate and epochs come from TRAINING (per dataset and model).
    'phase1': {
        'real_dim': 64,
        'learning_rate': None,
        'epochs': None,
        'batch_size': 2048,
        'seed': 42,
        'window_size': 250,
        'nu': 0.1,
        'gamma': 'scale',
        'alpha': 0.025,
        'standardize': True,
        'keep_fraction_after_drift': 0.10,
    },
    # For reference: exactly the notebook's constants (PyKEEN's default Adam learning rate 0.001).
    'antoine': {
        'real_dim': 32,
        'learning_rate': 0.001,
        'epochs': 2,
        'batch_size': 2048,
        'seed': 42,
        'window_size': 250,
        'nu': 0.1,
        'gamma': 0.5,
        'alpha': 0.025,
        'standardize': False,
        'keep_fraction_after_drift': 0.10,
    },
}

# Learning rate and epochs per (dataset, model), chosen by settings_check.py on the dataset's
# cutoff KG: best validation MRR over the learning rates tried, epochs = epoch of that best MRR.
# The cutoff KG does not depend on the stream, so one entry per dataset file is enough.
# Empty until the settings check has been run: drift_sphere.py refuses to start without it.
TRAINING = {
    # ('case1', 'TransE'): {'learning_rate': ..., 'epochs': ...},
}


def training_settings(preset, dataset, model):
    """real_dim, learning_rate, epochs, batch_size, seed for one model on one dataset."""
    cfg = {k: PRESETS[preset][k] for k in ('real_dim', 'learning_rate', 'epochs', 'batch_size', 'seed')}
    if preset == 'phase1':
        chosen = TRAINING.get((dataset, model))
        if chosen is None:
            raise ValueError(f'no learning rate / epochs for ({dataset!r}, {model!r}) in configs.TRAINING: '
                             'run settings_check.py first and fill in TRAINING')
        cfg.update(chosen)
    return cfg


# Sphere sweep: k is set so that, in the sphere (k = 1) run, the rule would have retrained on this
# share of new-entity re-embeds. Tag -> share.
SWEEP_TARGETS = {'sphereR10': 0.10, 'sphereR50': 0.50, 'sphereR90': 0.90}


# Grid entry: (dataset, model, composition, preset, retrain_policy, sphere_k, tag)
def _grid(datasets, models=MODELS, compositions=COMPOSITIONS, preset='phase1', policy='sphere', k=1.0):
    return [(d, m, c, preset, policy, k, None) for d in datasets for m in models for c in compositions]


GRIDS = {
    'sphere': _grid(['case1', 'case2', 'full']),
    'always': _grid(['case1', 'case2', 'full'], policy='always'),
    # 'sweep' is built at run time from the finished case1 'sphere' runs (see run_grid.py).
}
DYNAMIC_SWEEPS = {
    'sweep': {'datasets': ['case1'], 'models': MODELS},
}


def run_id(dataset, model, composition, preset, policy, k, tag=None):
    if tag:
        policy_tag = tag
    elif policy == 'sphere':
        policy_tag = f'sphere{k:g}'
    else:
        policy_tag = policy
    return f'{dataset}__{model}__{composition}__{policy_tag}__{preset}'
