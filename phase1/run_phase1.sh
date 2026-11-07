#!/usr/bin/env bash
# Phase 1, complete clean run. Start inside tmux:   bash run_phase1.sh
# Needs configs.TRAINING filled in (from settings_check.py); stops at once if it is not.
#   Background: full data, sphere + always (20 runs, 3 at a time)
#   Foreground: case1 + case2, sphere + always (40) -> MRR -> sphere sweep on case1 (30)
#   Then: analysis + ~/phase1_results.tgz
# Safe to restart: finished runs are skipped.
# Logs: phase1.log (this script), full.log (full data), runs/_logs/<run>.log (each run).
set -u
cd "$(dirname "$0")"
source ~/drift/.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec > >(tee -a phase1.log) 2>&1

step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }
mkdir -p quality analysis

step "check: learning rate and epochs filled in for every dataset and model"
python -c "
import configs
for d in ('case1', 'case2', 'full'):
    for m in configs.MODELS:
        s = configs.training_settings('phase1', d, m)
        print(f'  {d:6s} {m:9s} learning rate {s[\"learning_rate\"]:g}  epochs {s[\"epochs\"]}')
" || { echo "configs.TRAINING is incomplete: nothing started"; exit 1; }

step "full data: sphere + always, 20 runs, in the background (log: full.log)"
python run_grid.py --grid sphere always --datasets full --parallel 3 --threads 1 >> full.log 2>&1 &
FULL_PID=$!

step "case1 + case2: sphere + always, 40 runs"
python run_grid.py --grid sphere always --datasets case1 case2 --parallel 3 --threads 1

step "MRR per model with the run settings: case1, case2, full"
for d in case1 case2 full; do
  ls quality/${d}__RotatE__phase1.json >/dev/null 2>&1 || python embedding_quality.py --dataset $d
done

step "sphere size sweep on case1 (k from the sphere runs' ratios: ~10/50/90% retrain), 30 runs"
python run_grid.py --grid sweep --parallel 3 --threads 1

step "waiting for the full data"
wait $FULL_PID

step "analysis (tables, peak analysis, figures)"
python analyze.py --runs runs --out analysis --quality quality
tar czf ~/phase1_results.tgz --exclude='drift_windows.tsv.gz' runs analysis quality settings_check phase1.log full.log 2>/dev/null
step "ALL DONE -> ~/phase1_results.tgz"
