#!/usr/bin/env bash
# Phase 2 on the server. Start inside tmux, from the repository root:
#   bash phase2/run_phase2.sh 1      # cases, fast methods: fixed embedding, all measures, graph distances (~1-2 h)
#   bash phase2/run_phase2.sh 2      # GPU, overnight: comparability test, retrain + align, Antoine on
#                                    #   no-drift / injected streams, embedding-quality check
#   bash phase2/run_phase2.sh 3      # full data, fast methods only (~1-3 h)
# Safe to restart: finished runs are skipped. Every chunk ends by analysing everything that exists and
# executing the notebooks, so the latest results are always in:
#   phase2/outputs/notebooks/*.ipynb   (open them anywhere: the outputs are saved inside)
#   ~/phase2_results.tgz               (tables + executed notebooks, to download)
# Log: phase2/outputs/run_phase2.log      Progress: tail -f phase2/outputs/run_phase2.log
set -u
cd "$(dirname "$0")/.."
source ~/drift/.venv/bin/activate
mkdir -p phase2/outputs
exec > >(tee -a phase2/outputs/run_phase2.log) 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="python -u"
step() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

notebooks() {
  step "executing the notebooks (results saved inside them)"
  python -c "import nbconvert, ipykernel" 2>/dev/null || pip install -q nbconvert ipykernel
  mkdir -p phase2/outputs/notebooks
  for nb in part1a_audit_antoine part1b_evidence_antoine part2_other_methods; do
    jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=3600 \
      --output-dir phase2/outputs/notebooks "phase2/notebooks/$nb.ipynb" && echo "  $nb done"
  done
}
finish() {
  step "analysis"
  $PY phase2/step6_analyze.py --phase1 phase1
  notebooks
  tar czf ~/phase2_results.tgz phase2/outputs/results phase2/outputs/notebooks phase2/outputs/run_phase2.log 2>/dev/null
  echo "archive: ~/phase2_results.tgz"
}

chunk1() {
  step "1.0 MONDO ontology (once)"
  $PY phase2/step0_get_mondo.py
  for d in case1 case2; do
    step "1.1 $d: train each embedding once + fold-in (5 models x 2 variants, GPU)"
    $PY phase2/step1_prepare.py --dataset $d --variant holdout full
    step "1.2 $d: Antoine's loop with the embedding held fixed: real, 20 no-drift, 20 injected (all models)"
    $PY phase2/step2_stream_methods.py --dataset $d --methods frozen
    step "1.3 $d: every measure on real, 100 no-drift and 100 injected-switch streams"
    $PY phase2/step3_channels.py --dataset $d --workers 6
    step "1.4 $d: graph distances year by year"
    $PY phase2/step5_graph_distance.py --dataset $d
  done
  finish
}

chunk2() {
  step "2.1 comparability test on case1: 5 models x (5 seeds x 2 graphs + rerun + 5 warm starts), GPU"
  $PY phase2/step4_seed_test.py --dataset case1
  step "2.2 embedding quality: TransE MRR after 10, 25, 50, 100, 200 epochs"
  $PY phase2/step7_quality.py --dataset case1 --model TransE
  for d in case1 case2; do
    step "2.3 $d: retrain + align on the real stream, all models x compositions (3 at a time)"
    $PY phase2/step2_stream_methods.py --dataset $d --methods aligned --scenarios real --parallel 3
    step "2.4 $d: Antoine and retrain + align on 10 no-drift and 10 injected streams (TransE, Hadamard; 3 at a time)"
    $PY phase2/step2_stream_methods.py --dataset $d --methods original aligned --scenarios control inject \
        --models TransE --compositions hadamard --repeats 10 --parallel 3
  done
  step "2.5 quality check: TransE trained 100 epochs on case1 -- fixed embedding (all streams), Antoine and retrain + align (real)"
  $PY phase2/step1_prepare.py --dataset case1 --model TransE --variant full --epochs 100
  $PY phase2/step2_stream_methods.py --dataset case1 --methods frozen --models TransE --compositions hadamard --epochs 100
  $PY phase2/step2_stream_methods.py --dataset case1 --methods original aligned --scenarios real --models TransE \
      --compositions hadamard --epochs 100 --parallel 2
  finish
}

chunk3() {
  step "3.1 full: train each embedding once + fold-in (5 models x 2 variants, GPU)"
  $PY phase2/step1_prepare.py --dataset full --variant holdout full
  step "3.2 full: Antoine's loop with the embedding held fixed, real stream"
  $PY phase2/step2_stream_methods.py --dataset full --methods frozen --scenarios real
  step "3.3 full: every measure on the real stream"
  $PY phase2/step3_channels.py --dataset full --scenarios real --workers 6
  step "3.4 full: graph distances year by year"
  $PY phase2/step5_graph_distance.py --dataset full
  finish
}

case "${1:-}" in
  1) chunk1 ;;
  2) chunk2 ;;
  3) chunk3 ;;
  notebooks) step "analysis + notebooks only"; finish ;;
  *) echo "usage: bash phase2/run_phase2.sh 1|2|3|notebooks"; exit 1 ;;
esac
step "DONE (${1})"
