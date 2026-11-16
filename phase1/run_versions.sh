#!/usr/bin/env bash
# All planned runs of one version. Start inside tmux:
#     bash run_versions.sh v1        (or v2, v3)
#     bash run_versions.sh v3 "case1 case2"     only these datasets
# Versions (configs.VERSIONS): v1 from scratch with the chosen epochs, v2 from scratch with 20 epochs,
# v3 warm start (needs configs.WARM filled in from warm_check.py).
# Order: the cases first, then the full data. In each: Rule 1 (sphere) and Rule 2 (always), then
# Rule 3 (smaller spheres, sizes from the always runs) and Rule 4 (every X triples).
# Which rule runs on which dataset: configs.PLAN. Safe to restart: finished runs are skipped.
# Logs: versions_<version>.log (this script), <runs folder>/_logs/<run>.log (each run).
set -u
cd "$(dirname "$0")"
source ~/drift/.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V=${1:?usage: bash run_versions.sh v1|v2|v3 [\"datasets\"]}
ONLY=${2:-}
exec > >(tee -a "versions_$V.log") 2>&1

step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }
mkdir -p quality analysis

step "check: settings filled in for version $V"
python -c "
import configs
datasets = '$ONLY'.split() or configs.ALL_DATASETS
for d in datasets:
    for m in configs.MODELS:
        s = configs.training_settings('phase1', d, m)
        line = f'  {d:6s} {m:9s} learning rate {s[\"learning_rate\"]:g}  epochs {configs.VERSIONS[\"$V\"][\"epochs\"] or s[\"epochs\"]}'
        if configs.VERSIONS['$V']['embed_update'] == 'warm':
            w = configs.warm_settings(d, m)
            line += f'  warm: {w[\"warm_epochs\"]} epochs, learning rate {w[\"warm_learning_rate\"]:g}'
        print(line)
" || { echo "settings incomplete (configs.TRAINING / configs.WARM): nothing started"; exit 1; }

if [ -n "$ONLY" ]; then
  step "$V on $ONLY: Rule 1 (sphere) and Rule 2 (always)"
  python run_grid.py --version "$V" --grid sphere always --datasets $ONLY --parallel 3 --threads 1
  step "$V on $ONLY: Rule 3 (smaller spheres) and Rule 4 (every X triples)"
  python run_grid.py --version "$V" --grid sweep every --datasets $ONLY --parallel 3 --threads 1
else
  step "$V, cases: Rule 1 (sphere) and Rule 2 (always)"
  python run_grid.py --version "$V" --grid sphere always --datasets case1 case2 --parallel 3 --threads 1
  step "$V, cases: Rule 3 (smaller spheres) and Rule 4 (every X triples)"
  python run_grid.py --version "$V" --grid sweep every --datasets case1 case2 --parallel 3 --threads 1
  step "$V, full data: Rule 1 (sphere) and Rule 2 (always)"
  python run_grid.py --version "$V" --grid sphere always --datasets full --parallel 3 --threads 1
  step "$V, Rules 3 and 4 on every dataset planned for $V (finished runs are skipped)"
  python run_grid.py --version "$V" --grid sweep every --parallel 3 --threads 1
fi

if [ "$V" != "v3" ]; then
  step "MRR per model with the settings of $V"
  for d in ${ONLY:-case1 case2 full}; do
    python embedding_quality.py --dataset $d --version "$V"
  done
fi

step "analysis of every version that has runs (tables, peak tables, figures)"
python analyze.py --out analysis --quality quality
RUNS=$(python -c "import configs; print(configs.VERSIONS['$V']['runs_dir'])")
tar czf ~/results_$V.tgz --exclude='drift_windows.tsv.gz' "$RUNS" analysis quality settings_check warm_check "versions_$V.log" 2>/dev/null
step "ALL DONE -> ~/results_$V.tgz"
