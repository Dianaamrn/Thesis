"""Phase 1 settings: datasets, embedding training settings and run grids.

Everything that differs between runs lives here, so the experiment script itself
stays a straight port of Antoine's notebook
(biomedical_ocsvm_ztest_experiment (6).ipynb).

As in the notebook, the KG is built from every triple up to the cutoff year, and from then on
ONE triple type is monitored: Gene - causes_or_contributes_to_condition - Disease on case1 and full,
miRNA - causes_or_contributes_to_condition - Disease on case2 (too few Gene-Disease triples there).

Three versions (how the graph is re-embedded: VERSIONS below) x four retraining rules (RULES below),
5 models x 2 compositions, on case1, case2 and full; plus MRR per model and dataset, and the peak
analysis (analyze.py). After a drift alarm every rule re-embeds and retrains (Antoine's code).

Embedding training: every model is trained with the same procedure, PyKEEN early stopping on
validation MRR (settings_check.py). That gives each model and dataset its own learning rate and
number of epochs, stored in TRAINING below. Every re-embed in a run uses those fixed values.
"""

GENE_DISEASE = 'Gene - causes_or_contributes_to_condition - Disease'
MIRNA_DISEASE = 'miRNA - causes_or_contributes_to_condition - Disease'

# switch_year = first year of the injected drift (None = no known drift).
# max_stream = stop after this many stream triples (None = whole stream).
DATASETS = {
    # Monitored stream = one triple type (as in Antoine's notebook).
    'case1': {
        'file': 'filtered_drifting_edgesCASE1.tsv',
        'cutoff_year': 2002,
        'target_type': GENE_DISEASE,
        'switch_year': 2005,
        'max_stream': None,
        'note': "CASE1: cancer before 2005, infectious disease from 2005. Antoine's cutoff and stream type.",
    },
    # CASE2 has only 235 Gene-Disease triples before the 2022 switch, too few for a reference plus a
    # window of 250. Its monitored type is miRNA-Disease (Diana, 2026-11-16): still one triple type.
    'case2': {
        'file': 'filtered_drifting_edgesCASE2.tsv',
        'cutoff_year': 2018,
        'target_type': MIRNA_DISEASE,
        'switch_year': 2022,
        'max_stream': None,
        'note': 'CASE2: infectious disease before 2022, cancer from 2022. Stream = miRNA-Disease, cutoff 2018.',
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
# Results of settings_check.py on the server, 2026-11-07 (validation MRR, x = times random guessing):
TRAINING = {
    ('case1', 'TransE'):   {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.106,  670x (still improving at 100)
    ('case1', 'TransH'):   {'learning_rate': 0.001, 'epochs': 20},   # MRR 0.078,  490x
    ('case1', 'DistMult'): {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.107,  675x (still improving at 100)
    ('case1', 'ComplEx'):  {'learning_rate': 0.01,  'epochs': 100},  # MRR 0.179, 1130x (still improving at 100)
    ('case1', 'RotatE'):   {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.174, 1095x (still improving at 100)
    # case2: settings_check.py on the cutoff-2018 KG (settings_check/case2__chosen.csv). The cutoff KG
    # does not depend on the monitored type, so these hold for the miRNA-Disease stream.
    ('case2', 'TransE'):   {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.101,  649x (still improving at 100)
    ('case2', 'TransH'):   {'learning_rate': 0.001, 'epochs': 70},   # MRR 0.085,  549x
    ('case2', 'DistMult'): {'learning_rate': 0.001, 'epochs': 50},   # MRR 0.114,  734x
    ('case2', 'ComplEx'):  {'learning_rate': 0.01,  'epochs': 100},  # MRR 0.184, 1185x (still improving at 100)
    ('case2', 'RotatE'):   {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.182, 1171x (still improving at 100)
    ('full', 'TransE'):    {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.107,  909x (still improving at 100)
    ('full', 'TransH'):    {'learning_rate': 0.001, 'epochs': 30},   # MRR 0.037,  318x
    ('full', 'DistMult'):  {'learning_rate': 0.001, 'epochs': 50},   # MRR 0.052,  440x
    ('full', 'ComplEx'):   {'learning_rate': 0.01,  'epochs': 80},   # MRR 0.109,  930x
    ('full', 'RotatE'):    {'learning_rate': 0.001, 'epochs': 100},  # MRR 0.140, 1193x (still improving at 100)
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


# ---------------------------------------------------------------------------
# Versions = how the graph is re-embedded (strategy of 2026-11-16). Every version is run with the
# four retraining rules below.
#   v1  from scratch, learning rate and epochs from TRAINING (chosen by validation MRR)
#   v2  from scratch, 20 epochs for every model (learning rate from TRAINING): the coldest start
#   v3  warm start: the first embedding as in v1; every later re-embed starts from the previous
#       vectors (a new entity starts at PyKEEN's random vector) and trains WARM epochs over the whole KG
# runs_dir = folder of the version's runs; tag = last part of the run name.
# ---------------------------------------------------------------------------
VERSIONS = {
    'v1': {'embed_update': 'scratch', 'epochs': None, 'runs_dir': 'runs', 'tag': 'phase1'},
    'v2': {'embed_update': 'scratch', 'epochs': 20, 'runs_dir': 'runs_v2', 'tag': 'v2'},
    'v3': {'embed_update': 'warm', 'epochs': None, 'runs_dir': 'runs_v3', 'tag': 'v3'},
}

# Warm start (v3): epochs and learning rate of one warm re-embed, per (dataset, model), chosen from
# the measurements of warm_check.py. Empty until the check has been run: v3 refuses to start without it
# (or without --warm-epochs and --warm-learning-rate on the command line).
WARM = {
}


def warm_settings(dataset, model):
    chosen = WARM.get((dataset, model))
    if chosen is None:
        raise ValueError(f'no warm-start epochs / learning rate for ({dataset!r}, {model!r}) in configs.WARM: '
                         'run warm_check.py first and fill in WARM')
    return dict(chosen)


# Retraining rules (when the OCSVM is fitted again; after a drift alarm every rule re-embeds and refits):
#   sphere  Rule 1: Antoine's sphere rule, ratio > 1, at a new entity
#   always  Rule 2: at every new entity
#   sweep   Rule 3: sphere rule with smaller sizes. k is set per dataset and composition from the
#           ratios logged by the version's 'always' runs (5 models pooled), so that the rule would
#           retrain on this share of new-entity re-embeds. Tag -> share.
#   every   Rule 4: after every X stream triples, on the same reference triples. Tag every<X>.
SWEEP_TARGETS = {'sphereR10': 0.10, 'sphereR50': 0.50, 'sphereR90': 0.90}
EVERY_X = [10, 50, 250]
RULES = ['sphere', 'always', 'sweep', 'every']

# Which datasets each version runs each rule on by default (decided 2026-11-16). On the full data v1
# keeps the two finished rules; v2 gets Rules 3 and 4 there only if time allows (ask for it with
# --datasets full).
ALL_DATASETS = ['case1', 'case2', 'full']
CASES = ['case1', 'case2']
PLAN = {
    'v1': {'sphere': ALL_DATASETS, 'always': ALL_DATASETS, 'sweep': CASES, 'every': CASES},
    'v2': {'sphere': ALL_DATASETS, 'always': ALL_DATASETS, 'sweep': CASES, 'every': CASES},
    'v3': {'sphere': ALL_DATASETS, 'always': ALL_DATASETS, 'sweep': ALL_DATASETS, 'every': ALL_DATASETS},
}


def static_entries(rule, datasets):
    """Runs of a rule that need nothing from other runs. One dict per run."""
    out = []
    for d in datasets:
        for m in MODELS:
            for c in COMPOSITIONS:
                if rule in ('sphere', 'always'):
                    out.append({'dataset': d, 'model': m, 'composition': c, 'policy': rule, 'k': 1.0,
                                'tag': None, 'retrain_every': None})
                elif rule == 'every':
                    out += [{'dataset': d, 'model': m, 'composition': c, 'policy': 'every', 'k': 1.0,
                             'tag': None, 'retrain_every': x} for x in EVERY_X]
                else:
                    raise ValueError(rule)
    return out


def run_id(dataset, model, composition, preset, policy, k, tag=None, version='v1', retrain_every=None):
    if tag:
        policy_tag = tag
    elif policy == 'sphere':
        policy_tag = f'sphere{k:g}'
    elif policy == 'every':
        policy_tag = f'every{int(retrain_every)}'
    else:
        policy_tag = policy
    last = preset if version == 'v1' else VERSIONS[version]['tag']
    return f'{dataset}__{model}__{composition}__{policy_tag}__{last}'
