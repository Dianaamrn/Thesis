#!/usr/bin/env bash
# Every Phase 1 experiment, in order, on the server. Start inside tmux:
#     tmux new -s all
#     cd ~/Thesis/phase1 && bash run_all.sh all
# Or one stage at a time (same order as 'all'):
#     bash run_all.sh checks        correctness checks + Antoine's own settings on CASE1 (2 runs, ~1 h)
#     bash run_all.sh warm          warm-start check on case1, case2, full -> warm_check/<dataset>__chosen.csv
#     bash run_all.sh quality       test MRR of every from-scratch version on every dataset
#     bash run_all.sh cases         all versions x all rules on CASE1 and CASE2
#     bash run_all.sh import        reuse the earlier 10-epoch full-data runs (needs OLD_ARCHIVE, see below)
#     bash run_all.sh full          all versions x all rules on the full data (the long part: days)
#     bash run_all.sh sensitivity   detector settings one at a time, on the cases
#     bash run_all.sh analysis      tables, figures, peak dates, executed notebooks (notebooks/executed/),
#                                   everything packed in ~/results_phase1.tgz
# Safe to restart: finished runs are skipped. 'analysis' also runs after 'cases', so the case results can
# be looked at while the full data is still running.
# Logs: run_all.log (this script), <runs folder>/_logs/<run>.log (each run).
set -u
cd "$(dirname "$0")"
source ~/drift/.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYSTOW_HOME="$PWD/.pystow"
STAGE=${1:?usage: bash run_all.sh all|checks|warm|quality|cases|import|full|sensitivity|analysis}
PARALLEL=${PARALLEL:-3}                                # runs at the same time on the GPU
OLD_ARCHIVE=${OLD_ARCHIVE:-$HOME/phase1_old_lr0001.tgz} # the earlier runs (10 epochs), for 'import'
exec > >(tee -a run_all.log) 2>&1

step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }
grid() { python run_grid.py --parallel "$PARALLEL" --threads 1 "$@"; }
# Cheapest versions first, so results arrive early; v1 (MRR-chosen epochs) is the slowest from-scratch version.
CASE_VERSIONS="v2_e2 v2_e10 v2_e20 v3 v1"
FULL_VERSIONS="v2_e2 v3 v2_e10 v2_e20"

stage_checks() {
  step "correctness: Antoine's own settings on CASE1 (TransE, sphere rule, both compositions), GPU"
  grid --grid antoine
  step "correctness: the same on the CPU, the hardware of Antoine's own run (graded by check 7)"
  python run_grid.py --grid antoine --out-dir runs_antoine_check/cpu --parallel 1 -- --device cpu
  step "correctness checks (writes checks/checks.csv)"
  python checks.py
}

stage_warm() {
  for d in case1 case2 full; do
    step "warm-start check on $d"
    python warm_check.py --dataset $d
  done
}

stage_quality() {
  mkdir -p quality
  for v in v1 v2_e2 v2_e10 v2_e20; do
    for d in case1 case2 full; do
      if ls quality/${d}__*__$(python -c "import configs; v='$v'; print('phase1' if v=='v1' else configs.VERSIONS[v]['tag'])").json >/dev/null 2>&1; then
        echo "[skip] MRR $v $d"; continue
      fi
      step "MRR of every model, $v, $d"
      python embedding_quality.py --dataset $d --version $v
    done
  done
}

stage_cases() {
  for v in $CASE_VERSIONS; do
    step "$v, cases: Rule 1 (sphere) and Rule 2 (always)"
    grid --version $v --grid sphere always --datasets case1 case2
    step "$v, cases: Rule 3 (smaller spheres, sizes from this version's always runs) and Rule 4 (every X)"
    grid --version $v --grid sweep every --datasets case1 case2
  done
}

stage_import() {
  if [ -f "$OLD_ARCHIVE" ]; then
    step "reuse the earlier 10-epoch full-data runs from $OLD_ARCHIVE"
    python import_old_runs.py --archive "$OLD_ARCHIVE" --apply
  else
    echo "no $OLD_ARCHIVE: the 10-epoch full-data runs will be run again (set OLD_ARCHIVE=... to reuse them)"
  fi
}

stage_full() {
  for v in $FULL_VERSIONS; do
    step "$v, full data: Rule 1 (sphere) and Rule 2 (always)"
    grid --version $v --grid sphere always --datasets full
  done
  for v in $FULL_VERSIONS; do
    step "$v, full data: Rule 3 (smaller spheres) and Rule 4 (every X)"
    grid --version $v --grid sweep every --datasets full
  done
}

stage_sensitivity() {
  step "sensitivity: one detector setting at a time (configs.SENSITIVITY), cases"
  grid --grid sensitivity
}

stage_analysis() {
  step "analysis: every version"
  python analyze.py --out analysis --quality quality
  if [ -d runs_sensitivity ]; then
    step "analysis: sensitivity runs (with the base runs of the same version)"
    python analyze.py --runs runs_sensitivity "$(python -c 'import configs; print(configs.VERSIONS[configs.SENSITIVITY_VERSION]["runs_dir"])')" \
      --out analysis_sensitivity --quality quality --no-plots
  fi
  step "notebooks"
  python notebooks/build_notebooks.py
  mkdir -p notebooks/executed
  # Executed copies go to notebooks/executed/ (not in git), so the server's clone stays clean for git pull.
  (cd notebooks && jupyter nbconvert --to notebook --execute --output-dir executed \
     --ExecutePreprocessor.timeout=3600 checks.ipynb results.ipynb) \
    || echo "notebook execution failed: open notebooks/results.ipynb and run it by hand"
  tar czf ~/results_phase1.tgz --exclude='drift_windows.tsv.gz' \
    analysis analysis_sensitivity checks quality settings_check warm_check notebooks \
    $(python -c "import configs; print(' '.join(v['runs_dir'] for v in configs.VERSIONS.values()))") \
    runs_sensitivity runs_antoine_check run_all.log 2>/dev/null
  step "-> ~/results_phase1.tgz"
}

case "$STAGE" in
  all)
    stage_checks; stage_warm; stage_quality; stage_cases; stage_analysis
    stage_import; stage_full; stage_sensitivity; stage_analysis ;;
  checks|warm|quality|cases|import|full|sensitivity|analysis) "stage_$STAGE" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
step "DONE: $STAGE"
