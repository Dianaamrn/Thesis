#!/usr/bin/env bash
# Phase 2 on the server, in three chunks. Start inside tmux, from the repository root:
#   bash phase2/run_phase2.sh 1      # cases: the core results (~1-2 h)
#   bash phase2/run_phase2.sh 2      # seed test + A0 on no-drift controls (GPU, several hours)
#   bash phase2/run_phase2.sh 3      # full data, real order (longest)
#   bash phase2/run_phase2.sh all    # 1, then 2, then 3
# Safe to restart: finished pieces are skipped. Results: phase2/outputs/, archive ~/phase2_results.tgz.
# Log: phase2/outputs/run_phase2.log. Progress: tail -f phase2/outputs/run_phase2.log
set -u
cd "$(dirname "$0")/.."
source ~/drift/.venv/bin/activate
mkdir -p phase2/outputs
exec > >(tee -a phase2/outputs/run_phase2.log) 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="python -u"
step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }
finish() {
  step "analysis + archive"
  $PY phase2/step6_analyze.py --phase1-runs phase1/runs
  tar czf ~/phase2_results.tgz phase2/outputs/results phase2/outputs/run_phase2.log 2>/dev/null
  echo "archive: ~/phase2_results.tgz"
}

chunk1() {
  step "1.0 MONDO ontology (once)"
  $PY phase2/step0_get_mondo.py
  for d in case1 case2; do
    step "1.1 $d: train each frozen embedding once + fold-in (5 models x 2 variants, GPU)"
    $PY phase2/step1_prepare.py --dataset $d --variant holdout full
    step "1.2 $d: A1 (Antoine's method, embedding frozen) on the real stream and 20 no-drift controls"
    $PY phase2/step2_frozen.py --dataset $d --what A1_real A1_control --a1-repeats 20
    step "1.3 $d: channels C1-C5 on real, 100 no-drift and 100 injected-switch streams (CPU)"
    $PY phase2/step3_channels.py --dataset $d --workers 6
    step "1.4 $d: yearly graph distances (C6)"
    $PY phase2/step5_graph_distance.py --dataset $d
  done
  finish
}

chunk2() {
  step "2.1 seed / alignment test on case1 (5 models x 5 seeds x 2 graphs, GPU)"
  $PY phase2/step4_seed_test.py --dataset case1
  for d in case1 case2; do
    step "2.2 $d: A0 (Phase 1 code, re-embeds) on the same no-drift controls, 3 per model x composition (GPU)"
    $PY phase2/step2_frozen.py --dataset $d --what A0_control --a0-repeats 3
  done
  finish
}

chunk3() {
  step "3.1 full: frozen embeddings + fold-in (5 models x 2 variants, GPU)"
  $PY phase2/step1_prepare.py --dataset full --variant holdout full
  step "3.2 full: A1 on the real stream"
  $PY phase2/step2_frozen.py --dataset full --what A1_real
  step "3.3 full: channels on the real stream"
  $PY phase2/step3_channels.py --dataset full --scenarios real --workers 6
  step "3.4 full: yearly graph distances"
  $PY phase2/step5_graph_distance.py --dataset full
  finish
}

case "${1:-}" in
  1) chunk1 ;;
  2) chunk2 ;;
  3) chunk3 ;;
  all) chunk1; chunk2; chunk3 ;;
  *) echo "usage: bash phase2/run_phase2.sh 1|2|3|all"; exit 1 ;;
esac
step "DONE (chunk ${1})"
