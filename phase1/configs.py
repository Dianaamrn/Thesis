"""Phase 1 settings: datasets, presets and run grids.

Everything that differs between runs lives here, so the experiment script itself
stays a straight port of Antoine's notebook
(biomedical_ocsvm_ztest_experiment (6).ipynb).

Phase 1 (one universal config, preset 'phase1', 10 epochs for every model):
  Block A  Antoine's sphere rule (k = 1)      5 models x 2 compositions x {case1, case2, full}  30 runs
  Block B  always retrain on a new entity      5 models x 2 compositions x {case1, case2, full}  30 runs
  Block C  sphere-size sweep, k chosen from block A's logged ratios so the rule retrains on
           ~10%, ~50%, ~90% of new entities     cases: 5 x 2 x 2 x 3 = 60 runs
                                                short full-data test (first 2,000 triples), TransE:
                                                A + B (4 runs) + sweep (6 runs)
  plus MRR per model and dataset, and the peak analysis.
After a drift alarm every policy re-embeds and retrains (Antoine's code).
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
    'full_short': {
        'file': 'drifting_edges_timestamp.tsv',
        'cutoff_year': 2010,
        'target_type': GENE_DISEASE,
        'switch_year': None,
        'max_stream': 2000,
        'note': 'First 2,000 triples of the full stream (sphere-size test).',
    },
    # Extra, not in the Phase 1 run: Antoine's notebook stream (Gene-Disease only).
    'case1_gene': {
        'file': 'filtered_drifting_edgesCASE1.tsv',
        'cutoff_year': 2002,
        'target_type': GENE_DISEASE,
        'switch_year': 2005,
        'max_stream': None,
        'note': "Antoine's notebook stream (Gene-Disease only).",
    },
}

MODELS = ['TransE', 'TransH', 'DistMult', 'ComplEx', 'RotatE']
COMPLEX_MODELS = {'ComplEx', 'RotatE'}
COMPOSITIONS = ['hadamard', 'concatenation']

# real_dim = number of real values per entity vector. Complex models get real_dim // 2
# complex dimensions, so every model produces vectors of the same length.
PRESETS = {
    # Universal Phase 1 configuration: identical for every model and dataset.
    'phase1': {
        'real_dim': 64,
        'epochs': 10,
        'batch_size': 2048,
        'seed': 42,
        'window_size': 250,
        'nu': 0.1,
        'gamma': 'scale',
        'alpha': 0.025,
        'standardize': True,
        'keep_fraction_after_drift': 0.10,
    },
    # Extra: exactly the notebook's constants (for TransE this reproduces Antoine's run).
    'antoine': {
        'real_dim': 32,
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

# Sphere sweep: k is set so that, in the block A run, the rule would have retrained on this
# share of new-entity re-embeds. Tag -> share.
SWEEP_TARGETS = {'sphereR10': 0.10, 'sphereR50': 0.50, 'sphereR90': 0.90}


# Grid entry: (dataset, model, composition, preset, retrain_policy, sphere_k, tag)
def _grid(datasets, models=MODELS, compositions=COMPOSITIONS, preset='phase1', policy='sphere', k=1.0):
    return [(d, m, c, preset, policy, k, None) for d in datasets for m in models for c in compositions]


GRIDS = {
    # Phase 1
    'A_full': _grid(['full']),
    'B_full': _grid(['full'], policy='always'),
    'A_cases': _grid(['case1', 'case2']),
    'B_cases': _grid(['case1', 'case2'], policy='always'),
    'A_full_short': _grid(['full_short'], models=['TransE']),
    'B_full_short': _grid(['full_short'], models=['TransE'], policy='always'),
    # Sphere sweeps are built at run time from block A results (see run_grid.py):
    #   'C_cases'      <- block A on case1, case2 (all models)
    #   'C_full_short' <- block A on full_short (TransE)
    # Extras (not in the Phase 1 run)
    'pilot': [('case1', 'TransE', 'hadamard', 'phase1', 'sphere', 1.0, None)],
    'antoine_check': _grid(['case1_gene'], models=['TransE'], preset='antoine'),
    'case1_gene': _grid(['case1_gene']),
}
DYNAMIC_SWEEPS = {
    'C_cases': {'datasets': ['case1', 'case2'], 'models': MODELS},
    'C_full_short': {'datasets': ['full_short'], 'models': ['TransE']},
}


def run_id(dataset, model, composition, preset, policy, k, tag=None):
    if tag:
        policy_tag = tag
    elif policy == 'sphere':
        policy_tag = f'sphere{k:g}'
    else:
        policy_tag = policy
    return f'{dataset}__{model}__{composition}__{policy_tag}__{preset}'
