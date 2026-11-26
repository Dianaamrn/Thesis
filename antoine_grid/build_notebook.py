"""Writes antoine_grid/antoine_settings_grid.ipynb: Antoine's exact settings on the full data, with 4 embedding models
and the 4 retraining rules.

    python antoine_grid/build_notebook.py

Run on the server, inside tmux (outputs saved under every cell):
    cd ~/Thesis && source ~/drift/.venv/bin/activate
    jupyter nbconvert --to notebook --execute antoine_grid/antoine_settings_grid.ipynb \
        --output antoine_settings_grid_unscaled --output-dir antoine_grid/executed --ExecutePreprocessor.timeout=-1

Environment variables:
    AG_SCALING      unscaled (Antoine: no scaler, gamma 0.5; default) | scaled (scaler, gamma 'scale')
    AG_COMPS        compositions, comma separated (default hadamard)
    AG_GPU_SLOTS    runs at the same time on the GPU (default 2)
    AG_CPU_SLOTS    runs at the same time on the CPU (default 0)
    AG_CPU_THREADS  threads per CPU run (default 2)
    AG_TEST         1 = smoke test (random stand-in embedding, first 600 stream triples)
Finished runs are skipped, so the notebook can be executed again (e.g. after runs were added from another machine):
it then only analyses.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = []


def md(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')})


def code(text):
    CELLS.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n')})


md(r"""
# Antoine's settings on the full data: four embedding models, four retraining rules

**Question.** Antoine's notebook (TransE, Hadamard, no feature scaling, γ = 0.5) gives 26 drifts on the full data; the same
notebook with feature scaling (StandardScaler, γ = 'scale') gives 54, as do our runs. Is the halving specific to TransE and
Antoine's sphere rule, or does it hold for every embedding model and retraining rule? And do the runs alarm on the same
dates?

**What stays exactly as in Antoine's notebook**

| Setting | Value |
|---|---|
| Data | full stream, cutoff 2010, monitored type Gene – causes_or_contributes_to_condition – Disease |
| Re-embedding | from scratch (cold start) at every new entity and after every alarm |
| Embedding | 32 dimensions, 2 epochs, learning rate 0.001, batch 2048, seed 42 |
| Features | Hadamard s ⊙ p ⊙ o (concatenation optional) |
| One-class SVM | ν = 0.1, γ = 0.5, no feature scaling (the `scaled` execution: StandardScaler, γ = 'scale') |
| Test | window 250, α = 0.025 (alarm line 0.1002), keep 10% after an alarm |

**What changes**

- **Embedding model:** TransE (Antoine's), TransH, DistMult, RotatE. ComplEx is left out: at 2 epochs it does not learn
  (test MRR 0.00007, below random guessing).
- **Retraining rule** (when the SVM is refitted between alarms): R1 sphere k = 1 (Antoine's) · R2 at every re-embed ·
  R3 smaller sphere, k set from the R2 runs so it refits at about 50% of re-embeds · R4 every 250 stream triples.

The code of every run is `phase1/drift_sphere.py` with the preset `antoine` (Antoine's constants); on CASE1, on the CPU,
this code reproduces the alarms of Antoine's own notebook exactly (correctness check 7, 19 November).

TransE + R1 + Hadamard is Antoine's notebook itself (26 drifts on Diana's PC, CPU, 26 November). The runs here use the
server's GPU; with these settings GPU and CPU can give slightly different Hadamard alarms (seen on CASE1).
""")

md("## 0. Settings")
code(r"""
import os, sys, json, time, subprocess, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)

ROOT = Path.cwd()
while not (ROOT / 'phase1').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
SCALING = os.environ.get('AG_SCALING', 'unscaled')
COMPOSITIONS = os.environ.get('AG_COMPS', 'hadamard').split(',')
GPU_SLOTS = int(os.environ.get('AG_GPU_SLOTS', '2'))
CPU_SLOTS = int(os.environ.get('AG_CPU_SLOTS', '0'))
TEST = os.environ.get('AG_TEST') == '1'
MODELS = ['TransE', 'TransH', 'DistMult', 'RotatE']
RULES = {'R1_sphere': {'policy': 'sphere', 'k': 1.0},
         'R2_always': {'policy': 'always'},
         'R3_sphere50': {'policy': 'sphere', 'share': 0.5},
         'R4_every250': {'policy': 'every', 'x': 250}}
SCALE_ARGS = {'unscaled': ['--standardize', 'no', '--gamma', '0.5'], 'scaled': ['--standardize', 'yes', '--gamma', 'scale']}[SCALING]
OUT = ROOT / 'antoine_grid' / ('runs_TEST' if TEST else 'runs')
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
print(f'scaling {SCALING} | compositions {COMPOSITIONS} | GPU slots {GPU_SLOTS}, CPU slots {CPU_SLOTS} | test {TEST}\noutput {OUT}')
""")

md(r"""
## 1. Running one run

`run_one` starts `phase1/drift_sphere.py` for one model, composition and rule, with Antoine's preset and the scaling of this
execution, and writes the run's log next to its results. A run that is already finished is skipped. If a GPU run fails (e.g. for
lack of memory: the GPU is shared), it is repeated on the CPU and marked; CPU runs do not see the GPU at all.
""")
code(r"""
def run_name(model, comp, rule):
    return f'full__{model}__{comp}__{rule}__antoine_{SCALING}'


def command(model, comp, rule, k, device):
    r = RULES[rule]
    cmd = [PY, str(ROOT / 'phase1' / 'drift_sphere.py'), '--dataset', 'full', '--model', model, '--composition', comp,
           '--preset', 'antoine', '--version', 'v1', '--retrain-policy', r['policy'], '--device', device,
           '--out-dir', str(OUT), '--run-name', run_name(model, comp, rule), '--no-drift-windows', *SCALE_ARGS]
    if r['policy'] == 'sphere':
        cmd += ['--sphere-k', str(k)]
    if r['policy'] == 'every':
        cmd += ['--retrain-every', str(r['x'])]
    if TEST:
        cmd += ['--embedder', 'random', '--max-stream', '600']
    return cmd


CPU_THREADS = os.environ.get('AG_CPU_THREADS', '2')
ENV = {'cuda': dict(os.environ),   # CPU runs do not see the GPU at all, and use a few threads each
       'cpu': {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': CPU_THREADS, 'MKL_NUM_THREADS': CPU_THREADS}}
LOCK = threading.Lock()
SLOTS = ['cuda'] * GPU_SLOTS + ['cpu'] * CPU_SLOTS
FREE = list(SLOTS)


def run_one(job):
    model, comp, rule, k = job
    d = OUT / run_name(model, comp, rule)
    if (d / 'summary.json').exists():
        return model, comp, rule, 'already finished'
    with LOCK:
        device = FREE.pop(0)
    try:
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(d / 'run.log', 'w') as log:
            rc = subprocess.call(command(model, comp, rule, k, device), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=ENV[device])
        if rc != 0 and device == 'cuda':   # e.g. out of GPU memory (the GPU is shared): repeat on the CPU
            (d / 'run_gpu_failed.log').write_text((d / 'run.log').read_text(errors='ignore'))
            with open(d / 'run.log', 'w') as log:
                rc = subprocess.call(command(model, comp, rule, k, 'cpu'), stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=ENV['cpu'])
            device = 'cpu (after a GPU failure)'
        status = f'{"ok" if rc == 0 else "FAILED rc=" + str(rc)} on {device}, {(time.time() - t0) / 60:.1f} min'
        print(f'{time.strftime("%H:%M")} {run_name(model, comp, rule)}: {status}', flush=True)
        return model, comp, rule, status
    finally:
        with LOCK:
            FREE.append(device.split(' ')[0])


def run_jobs(jobs):
    with ThreadPoolExecutor(max_workers=len(SLOTS)) as ex:
        return list(ex.map(run_one, jobs))
""")

md(r"""
## 2. The runs

Order: R2 first (R3's sphere size is computed from R2's logged sphere ratios), then R1 and R4, then R3. Within each step the
runs share the GPU slots. Each finished run prints its time.
""")
code(r"""
t_all = time.time()
done = run_jobs([(m, c, 'R2_always', None) for c in COMPOSITIONS for m in MODELS])
""")
code(r"""
def sphere_k(comp, share):
    # k such that the sphere rule would refit at `share` of the new-entity re-embeds of the R2 runs (models pooled)
    ratios = []
    for m in MODELS:
        f = OUT / run_name(m, comp, 'R2_always') / 'embedding_shift_metrics.tsv'
        if f.exists() and f.stat().st_size > 1:
            s = pd.read_csv(f, sep='\t')
            ratios += s.loc[s['trigger_reason'].eq('unseen_element'), 'center_radius_ratio'].replace([np.inf, -np.inf], np.nan).dropna().tolist()
    return float(np.quantile(ratios, 1 - share)) if ratios else 1.0, len(ratios)


K3 = {c: sphere_k(c, RULES['R3_sphere50']['share']) for c in COMPOSITIONS}
print('R3 sphere size per composition (k, number of R2 ratios):', K3)
done += run_jobs([(m, c, r, 1.0) for r in ('R1_sphere', 'R4_every250') for c in COMPOSITIONS for m in MODELS])
done += run_jobs([(m, c, 'R3_sphere50', K3[c][0]) for c in COMPOSITIONS for m in MODELS])
print(f'all runs of this execution done in {(time.time() - t_all) / 3600:.2f} h')
pd.DataFrame(done, columns=['model', 'composition', 'rule', 'status'])
""")

md(r"""
## 3. Results: drifts per model and rule

Per run: drifts (alarms), re-embeds, SVM refits by the rule between alarms, tested triples, run time. The maximum possible
number of drifts on this stream is about 58: after every alarm the next 225 triples are not tested.
""")
code(r"""
def load(name):
    d = OUT / name
    if not (d / 'summary.json').exists():
        return None
    s = json.loads((d / 'summary.json').read_text())
    s['device'] = json.loads((d / 'config.json').read_text()).get('device', '')
    dr = pd.read_csv(d / 'drifts.tsv', sep='\t') if (d / 'drifts.tsv').stat().st_size > 1 else pd.DataFrame()
    wh = pd.read_csv(d / 'window_history.tsv.gz', sep='\t')
    st = pd.read_csv(d / 'step_trace.tsv.gz', sep='\t')
    return s, dr, wh, st


RUNS, rows = {}, []
for c in COMPOSITIONS:
    for m in MODELS:
        for r in RULES:
            x = load(run_name(m, c, r))
            if x is None:
                continue
            s, dr, wh, st = x
            RUNS[(m, c, r)] = x
            refits = int(s.get('num_threshold_retrains', 0) or 0) + int(s.get('num_scheduled_refits', 0) or 0)
            if r == 'R2_always':
                refits = int(s.get('num_unseen_refresh_events', 0) or 0)
            up = int((dr['current_outlier_fraction'] > dr['reference_outlier_fraction']).sum()) if len(dr) else 0
            rows.append({'model': m, 'composition': c, 'rule': r, 'drifts': s['num_detected_drifts'], 'up': up,
                         'down': s['num_detected_drifts'] - up, 're-embeds': s['num_embedding_refresh_events'],
                         'refits by rule': refits, 'tested': s['tested_steps'],
                         'mean window share': round(float(wh['current_outlier_fraction'].mean()), 3),
                         'mean baseline': round(float(wh['reference_outlier_fraction'].mean()), 3),
                         'k': s.get('sphere_k'), 'device': s.get('device', ''), 'minutes': round(s['runtime_seconds'] / 60, 1)})
RESULTS = pd.DataFrame(rows)
RESULTS
""")
code(r"""
if len(RESULTS):
    show = RESULTS.pivot_table(index=['composition', 'model'], columns='rule', values='drifts')
    print('drifts per model and rule'); display(show)
""")

md(r"""
## 4. Outlier share over the stream

Window outlier share (blue) and baseline outlier rate (orange) at every tested triple; alarms in black. One panel per run.
""")
code(r"""
keys = sorted(RUNS)
if keys:
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 1.6 * len(keys)), sharex=True, squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        s, dr, wh, st = RUNS[key]
        ax.plot(wh['stream_index'], wh['current_outlier_fraction'], lw=0.7, color='#2a78d6')
        ax.plot(wh['stream_index'], wh['reference_outlier_fraction'], lw=0.7, color='#eb6834')
        for i in (dr['stream_index'] if len(dr) else []):
            ax.axvline(i, color='k', lw=0.5, ymin=0.85)
        ax.set_ylim(0, 1); ax.set_title(' · '.join(key) + f'  ({len(dr)} drifts)', fontsize=8, loc='left')
    axes[-1, 0].set_xlabel('stream triple'); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Where the drifts fall

Drift dates shared by several runs, and a comparison with Antoine's notebook run on the PC (26 drifts, CPU) and with the 40
final-pipeline runs, when their files are present.
""")
code(r"""
alarm_rows = []
for (m, c, r), (s, dr, wh, st) in RUNS.items():
    for _, a in dr.iterrows():
        alarm_rows.append({'model': m, 'composition': c, 'rule': r, 'stream_index': a['stream_index'], 'row_number': a['row_number'],
                           'date': pd.to_datetime(pd.to_numeric(a['timestamp'], errors='coerce'), unit='s').date()})
ALARMS = pd.DataFrame(alarm_rows)
if len(ALARMS):
    shared = ALARMS.groupby('date').agg(runs=('model', 'size'), models=('model', lambda x: ', '.join(sorted(set(x)))),
                                        rules=('rule', lambda x: ', '.join(sorted(set(x))))).sort_values('runs', ascending=False)
    display(shared.head(25))
ref = ROOT / 'results' / '2026-11-25_full_final_alarms'
for label, f in [('Antoine notebook, PC, unscaled (26)', ref / 'antoine_notebook7_full_rerun_2026-11-26_hadamard_drifts.csv'),
                 ('Antoine notebook, PC, scaled (54)', ref / 'antoine_notebook7_full_SCALED_gamma_scale_hadamard_drifts.csv')]:
    if f.exists() and len(ALARMS):
        other = set(pd.to_datetime(pd.read_csv(f)['date']).dt.date)
        print(label + ':')
        for key in sorted(RUNS):
            mine = set(ALARMS[(ALARMS.model == key[0]) & (ALARMS.composition == key[1]) & (ALARMS.rule == key[2])]['date'])
            print(f'   {" · ".join(key):40s} {len(mine):3d} drift dates, {len(mine & other):3d} shared')
""")

md("## 6. Summary")
code(r"""
if len(RESULTS):
    print(f'scaling: {SCALING}')
    print(RESULTS.groupby(['composition', 'rule'])['drifts'].agg(['min', 'median', 'max']).to_string())
    print('\nper model:'); print(RESULTS.groupby(['composition', 'model'])['drifts'].agg(['min', 'median', 'max']).to_string())
""")


def main():
    nb = {'cells': CELLS, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out = HERE / 'antoine_settings_grid.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', out, f'({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
