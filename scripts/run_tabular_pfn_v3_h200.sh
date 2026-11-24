#!/usr/bin/env bash
#
# Single-command H200 launcher for SparseTabularCausalPFN v3 (18.88M params, P = 20..3,500,
# N_ctx = 80..8,000, 360,000 unique studies across all 5 causal DGP families):
#   1. 5-Family Prior: sparse_oci (30%) + reequation_calc (15%) + nemotron_design (20%)
#      + tree_subgroup (18%, depth 2-4 clinical biomarker rules)
#      + causal_scm (17%, layered nonlinear DAG/MLP + observed & latent M-bias colliders)
#   2. Real Stage-1 LLM pathologies: fragmented partial-coverage synonym extraction clusters
#      (--fragmented-dup-prob 0.55), heavy-tailed Student-t labs, zero-inflated counts,
#      dense weak-confounding regimes (--dense-conf-prob 0.15), MNAR, unmeasured confounding
#   3. v3 Architecture (--use-v3-solver-basis --decouple-rlearner-propensity):
#      - 4-channel ARD basis (X, X^2, tanh(1.5 X), smooth variance-gated missingness M_c)
#      - 6-d pairwise-complete co-observed correlation + overlap bias in column attention
#      - Higher capacity (col_dim=256, 4 col layers, study_width=256, 6 study layers, 8 heads,
#        32 attn slots, top_k_nuis=128, top_k_cate=96, 9-point LOO-CV grid, bf16 FlashAttn SDPA)
#
# Measured peak VRAM: 38.2 GiB at worst-case N_ctx=8,000, P=3,500 (9.8 GiB typical at N=1,800, P=1,000).
#
# Usage:
#   ./scripts/run_tabular_pfn_v3_h200.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=20 ./scripts/run_tabular_pfn_v3_h200.sh
#   STEPS=40000 ./scripts/run_tabular_pfn_v3_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_v3_h200}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# ==============================================================================
# TRAINING: LOG-UNIFORM N_ctx = 80..8,000, P = 20..3,500 ACROSS 5 CAUSAL FAMILIES
# ==============================================================================
"${PYTHON}" -m oci.tabular_pfn.cli train \
  --run-dir "${RUN_DIR}" \
  --steps "${STEPS}" \
  --min-context-size 80 \
  --context-size 8000 \
  --query-size 128 \
  --min-features 20 \
  --max-features 3500 \
  --family-weights 0.30,0.15,0.20,0.18,0.17 \
  --null-hte-prob 0.25 \
  --instrument-prob 0.70 \
  --unmeasured-conf-prob 0.25 \
  --extraction-noise-prob 0.40 \
  --fragmented-dup-prob 0.55 \
  --collider-prob 0.45 \
  --dense-conf-prob 0.15 \
  --mnar-prob 0.50 \
  --rd-scale-prob 0.25 \
  --rct-prob 0.08 \
  --max-treatment-logit-std 3.0 \
  --col-dim 256 \
  --study-width 256 \
  --study-layers 6 \
  --col-attn-layers 4 \
  --heads 8 \
  --top-k-nuis 128 \
  --top-k-cate 96 \
  --attn-slots 32 \
  --ridge-grid-size 9 \
  --decouple-rlearner-propensity \
  --use-v3-solver-basis \
  --grad-accum-steps 12 \
  --lr 3e-4 \
  --warmup-steps 500 \
  --save-every 1000 \
  --eval-every 1000 \
  --eval-studies 64 \
  --eval-context-size 800 \
  --eval-num-features 500 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# ==============================================================================
# HELD-OUT STRESS TESTS ACROSS P = 50, 200, 500, 2000, 3500 (N_ctx = 800)
# ==============================================================================
for P in 50 200 500 2000 3500; do
  echo "=== Evaluating v3 held-out 5-family mixture P=${P} (N_ctx=800, 128 studies) ==="
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
echo "=== Evaluating v3 held-out large-N P=2000 (N_ctx=4000, 64 studies) ==="
"${PYTHON}" -m oci.tabular_pfn.cli evaluate \
  --run-dir "${RUN_DIR}" \
  --studies 64 \
  --context-size 4000 \
  --query-size 128 \
  --num-features 2000 \
  --num-workers "${NUM_WORKERS}" \
  --seed 750001 \
  --device "${DEVICE}" | tee "${RUN_DIR}/eval_n4000_p2000.json"

# Family-specific held-out checks across all 5 families (P = 500 and P = 2,000)
for FAM in sparse_oci nemotron_design reequation_calc tree_subgroup causal_scm; do
  for P in 500 2000; do
    echo "=== Evaluating v3 held-out family=${FAM} P=${P} (N_ctx=800, 96 studies) ==="
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
