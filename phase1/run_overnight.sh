#!/usr/bin/env bash
# Phase 1, unattended. Start inside tmux:
#   bash run_overnight.sh
# One universal config (preset 'phase1': 64 dims, 10 epochs, scaling, gamma 'scale').
#   Track 1 (background): full data, block A (sphere k=1) + block B (always retrain), 20 runs
#   Track 2: cases A + B (40) -> MRR -> sphere sweep on the cases (60)
#            -> short full-data test A + B + sweep (10) -> analysis + ~/phase1_partial.tgz
# When track 1 also finishes: final analysis + ~/phase1_results.tgz
# Safe to restart: finished runs are skipped.
# Logs: overnight.log (track 2), full_grid.log (track 1), runs/_logs/<run>.log (each run).
set -u
cd "$(dirname "$0")"
source ~/drift/.venv/bin/activate
exec > >(tee -a overnight.log) 2>&1

step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }
archive() { tar czf "$1" --exclude='drift_windows.tsv.gz' runs analysis quality overnight.log full_grid.log 2>/dev/null; }
mkdir -p quality analysis

step "track 1 started in background: full data, block A + B, 20 runs (log: full_grid.log)"
python run_grid.py --grid A_full B_full --parallel 3 --threads 1 >> full_grid.log 2>&1 &
FULL_PID=$!

step "2.1 cases: block A (sphere k=1) + block B (always), 5 models x 2 compositions x 2 cases = 40 runs"
python run_grid.py --grid A_cases B_cases --parallel 3 --threads 1

step "2.2 MRR per model: case1, case2, full"
for d in case1 case2 full; do
  ls quality/${d}__RotatE__phase1.json >/dev/null 2>&1 || python embedding_quality.py --dataset $d
done

step "2.3 sphere sweep on the cases (k from block A ratios: ~10/50/90% retrain), 60 runs"
python run_grid.py --grid C_cases --parallel 3 --threads 1

step "2.4 short full-data test (first 2,000 triples), TransE: A + B, then the sweep"
python run_grid.py --grid A_full_short B_full_short --parallel 3 --threads 1
python run_grid.py --grid C_full_short --parallel 3 --threads 1

step "2.5 analysis (cases + short test; full data may still be running)"
python analyze.py --runs runs --out analysis --quality quality
archive ~/phase1_partial.tgz
step "partial results -> ~/phase1_partial.tgz ; waiting for the full-data track"

wait $FULL_PID
step "track 1 done: full data"
python analyze.py --runs runs --out analysis --quality quality
archive ~/phase1_results.tgz
step "ALL DONE -> ~/phase1_results.tgz"
