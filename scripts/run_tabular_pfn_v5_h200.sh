#!/usr/bin/env bash
#
# Single-command H200 launcher for SparseTabularCausalPFN v5 (Universal High-Order Epistasis &
# Dense-Regime Architecture, P = 20..3,500, N_ctx = 80..8,000, 540,000 unique studies across all
# 7 causal DGP families):
#
#   1. De-aliased 7-Family Universal Prior (firewall-clean vs oci_bench_complex S7-S12b):
#      - sparse_oci (16%): sparse & upgraded dense (12-42 confounders, 4-26 modifiers) all-evidence tables
#      - reequation_calc (10%): OCI engine.R / reequation.py + C x C & C x M interactions
#      - nemotron_design (16%): 4,000 train-split Nemotron oncology designs + clinical interactions
#      - tree_subgroup (16%): depth 2-4 biomarker & cross-role C x M decision-tree conjunctions
#      - causal_scm (14%): layered nonlinear DAG/MLP with joint C x M HTE + observed & latent M-bias colliders
#      - interacting_clinical (14%): upgraded with 38% pure-epistatic crossover HTE (50% strictly pure
#        M1 x M2 with zero main effect on T or Y!) + 50% correlated cosine-MaxSim embedding blocks
#      - universal_nonparametric (14%): localized Gaussian/Laplace RBF kernel mixtures, 1-to-3-way MARS
#        hinge splines, 3-layer Swish residual neural fields, pure 2-/3-way orthogonal epistasis, and
#        ultra-dense (15-42 confounders + 10-32 modifiers) nonlinear regimes
#      - Dense confounding probability doubled: --dense-conf-prob 0.30 (vs v4's 0.15)
#
#   2. v5 Universal Architecture (--use-v3-solver-basis --use-v4-interactions --use-v5-epistasis):
#      - Memory-bounded exact tiled all-pairs symmetric Robinson epistasis scan over ALL P*(P-1)/2
#        pairs (up to 6.1M pairs at P=3,500!) with smooth C^inf orthogonal deduplication
#        sqrt(1 - corr^2) and Bonferroni excess calibration (ROB_STAT_DIM = 20)
#      - Direct discovery & injection of top epi_pair_k=32 global pure M1 x M2 epistatic pairs into
#        the closed-form ARD Robinson R-learner alongside 190 top-node pairs (pair_top_k=20)
#      - Expanded dense-regime capacity: top_k_nuis=160, top_k_cate=128, attn_slots=64 (with detached
#        treatment-propensity slot boost so dense confounders enter residual Two-Way TabPFN cleanly
#        while keeping exact zero prop_loss gradient into g_er / g_per!)
#      - Auto gradient checkpointing above N_ctx=1,200 inside 8-layer bf16 SDPA study_layers
#
#   3. Automatic Post-Training Evaluation:
#      - Full held-out 7-family stress tests across P = 50..3,500, N = 4,000, and all 7 families
#      - Full Option A evaluation suite (Stage-2 + Full-LLM cohorts + 47 Complex Full-LLM cohorts +
#        51 Complex Tabular S7-S12b multi-cancer stress cells) using lean sent_256 default!
#
# Usage:
#   ./scripts/run_tabular_pfn_v5_h200.sh
#   NUM_WORKERS=16 ./scripts/run_tabular_pfn_v5_h200.sh            # dedicated H200 host
#   NUM_WORKERS=8 ./scripts/run_tabular_pfn_v5_h200.sh             # sharing host RAM with Option B
#   CUDA_VISIBLE_DEVICES=0 STEPS=45000 ./scripts/run_tabular_pfn_v5_h200.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${HOME}/thisenv/bin/python" ]]; then
    PYTHON="${HOME}/thisenv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    PYTHON="$(command -v python3)"
  fi
fi

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_v5_h200}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-45000}"
NUM_WORKERS="${NUM_WORKERS:-12}"

echo "=== SparseTabularCausalPFN v5 (Universal Epistasis & Dense-Regime Architecture) ==="
echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# Auto-resume cleanly if checkpoint.pt already exists in RUN_DIR
CMD_MODE="train"
if [[ -f "${RUN_DIR}/checkpoint.pt" ]]; then
  echo "Found existing ${RUN_DIR}/checkpoint.pt — resuming to step ${STEPS}..."
  CMD_MODE="resume"
fi

if [[ "${CMD_MODE}" == "resume" ]]; then
  "${PYTHON}" -m oci.tabular_pfn.cli resume \
    --run-dir "${RUN_DIR}" \
    --steps "${STEPS}" \
    --min-context-size 80 \
    --context-size 8000 \
    --query-size 128 \
    --grad-accum-steps 12 \
    --lr 2.5e-4 \
    --warmup-steps 800 \
    --save-every 1000 \
    --eval-every 1000 \
    --eval-studies 64 \
    --eval-context-size 800 \
    --eval-num-features 500 \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
else
  "${PYTHON}" -m oci.tabular_pfn.cli train \
    --run-dir "${RUN_DIR}" \
    --steps "${STEPS}" \
    --min-context-size 80 \
    --context-size 8000 \
    --query-size 128 \
    --min-features 20 \
    --max-features 3500 \
    --family-weights 0.16,0.10,0.16,0.16,0.14,0.14,0.14 \
    --null-hte-prob 0.25 \
    --instrument-prob 0.70 \
    --unmeasured-conf-prob 0.25 \
    --extraction-noise-prob 0.40 \
    --fragmented-dup-prob 0.55 \
    --collider-prob 0.45 \
    --dense-conf-prob 0.30 \
    --mnar-prob 0.50 \
    --rd-scale-prob 0.25 \
    --rct-prob 0.08 \
    --max-treatment-logit-std 3.2 \
    --col-dim 320 \
    --study-width 320 \
    --study-layers 8 \
    --col-attn-layers 4 \
    --heads 8 \
    --top-k-nuis 160 \
    --top-k-cate 128 \
    --attn-slots 64 \
    --pair-top-k 20 \
    --epi-pair-k 32 \
    --ridge-grid-size 9 \
    --decouple-rlearner-propensity \
    --use-v3-solver-basis \
    --use-v4-interactions \
    --use-v5-epistasis \
    --grad-accum-steps 12 \
    --lr 2.5e-4 \
    --warmup-steps 800 \
    --save-every 1000 \
    --eval-every 1000 \
    --eval-studies 64 \
    --eval-context-size 800 \
    --eval-num-features 500 \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}"
fi

# ==============================================================================
# HELD-OUT STRESS TESTS ACROSS P = 50, 200, 500, 2000, 3500 (N_ctx = 800)
# ==============================================================================
for P in 50 200 500 2000 3500; do
  echo "=== Evaluating v5 held-out 7-family mixture P=${P} (N_ctx=800, 128 studies) ==="
  "${PYTHON}" -m oci.tabular_pfn.cli evaluate \
    --run-dir "${RUN_DIR}" \
    --studies 128 \
    --context-size 800 \
    --query-size 128 \
    --num-features "${P}" \
    --num-workers "${NUM_WORKERS}" \
    --seed 700001 \
    --device "${DEVICE}" | tee "${RUN_DIR}/eval_n800_p${P}.json"
done

# Large-cohort N_ctx = 4,000 evaluation at P = 2,000
echo "=== Evaluating v5 held-out large-N P=2000 (N_ctx=4000, 64 studies) ==="
"${PYTHON}" -m oci.tabular_pfn.cli evaluate \
  --run-dir "${RUN_DIR}" \
  --studies 64 \
  --context-size 4000 \
  --query-size 128 \
  --num-features 2000 \
  --num-workers "${NUM_WORKERS}" \
  --seed 750001 \
  --device "${DEVICE}" | tee "${RUN_DIR}/eval_n4000_p2000.json"

# Family-specific held-out checks across all 7 families (P = 500 and P = 2,000)
for FAM in sparse_oci nemotron_design reequation_calc tree_subgroup causal_scm interacting_clinical universal_nonparametric; do
  for P in 500 2000; do
    echo "=== Evaluating v5 held-out family=${FAM} P=${P} (N_ctx=800, 96 studies) ==="
    "${PYTHON}" -m oci.tabular_pfn.cli evaluate \
      --run-dir "${RUN_DIR}" \
      --studies 96 \
      --context-size 800 \
      --query-size 128 \
      --num-features "${P}" \
      --family "${FAM}" \
      --num-workers "${NUM_WORKERS}" \
      --seed 800001 \
      --device "${DEVICE}" | tee "${RUN_DIR}/eval_n800_${FAM}_p${P}.json"
  done
done

# ==============================================================================
# FULL OPTION-A & COMPLEX OUT-OF-FAMILY BENCHMARK SUITE (STAGE-2 + LLM + S7-S12b)
# ==============================================================================
echo "=== Evaluating v5 on Stage-2, Full-LLM cohorts, Complex Full-LLM, and S7-S12b ==="
"${PYTHON}" "${REPO_ROOT}/scripts/eval_option_a_v4.py" \
  --tabular-v4-dir "${RUN_DIR}" \
  --backbone-path "${RUN_ROOT}/tinyoncbert_6l_384d" \
  --run5-dir "${RUN_ROOT}/longitudinal_tinyoncbert_v5" \
  --output-dir "${RUN_DIR}/eval_option_a" \
  --device "${DEVICE}"
