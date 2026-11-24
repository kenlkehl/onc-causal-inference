#!/usr/bin/env bash
#
# Single-command launcher for SparseTabularCausalPFN v2 (7.38M params, P = 20..2,000,
# N = 80..3,000) across all 3 OCI DGP families (sparse_oci, reequation_calc, nemotron_design):
#   - Decoupled outcome-adaptive R-learner propensity (--decouple-rlearner-propensity) so
#     prop_loss never pulls pure Stage-1 instruments into Robinson e_r_loo
#   - Deeper/wider residual Two-Way TabPFN (4 layers, width 192, 6 heads, 24 slots, bf16 SDPA)
#   - Deeper correlation-biased column attention (3 layers, col_dim 192) & wider recall
#     (top_k_nuis=80, top_k_cate=64)
#
# VRAM: ~6-7 GiB worst-case peak at N_ctx=3,000, P=2,000 (runs comfortably on 1x RTX A6000 or H200).
#
# Usage:
#   ./scripts/run_tabular_pfn_v2_h200.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=8 ./scripts/run_tabular_pfn_v2_h200.sh
#   STEPS=12000 ./scripts/run_tabular_pfn_v2_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_v2_p2000}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-15000}"
NUM_WORKERS="${NUM_WORKERS:-12}"

echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# ==============================================================================
# TRAINING: LOG-UNIFORM N_ctx = 80..3,000, P = 20..2,000 ACROSS 3 OCI DGP FAMILIES
# ==============================================================================
"${PYTHON}" -m oci.tabular_pfn.cli train \
  --run-dir "${RUN_DIR}" \
  --steps "${STEPS}" \
  --min-context-size 80 \
  --context-size 3000 \
  --query-size 64 \
  --min-features 20 \
  --max-features 2000 \
  --null-hte-prob 0.25 \
  --instrument-prob 0.70 \
  --unmeasured-conf-prob 0.25 \
  --extraction-noise-prob 0.35 \
  --mnar-prob 0.50 \
  --rd-scale-prob 0.25 \
  --rct-prob 0.08 \
  --col-dim 192 \
  --study-width 192 \
  --study-layers 4 \
  --col-attn-layers 3 \
  --heads 6 \
  --top-k-nuis 80 \
  --top-k-cate 64 \
  --attn-slots 24 \
  --ridge-grid-size 7 \
  --decouple-rlearner-propensity \
  --grad-accum-steps 8 \
  --lr 4e-4 \
  --warmup-steps 300 \
  --save-every 500 \
  --eval-every 500 \
  --eval-studies 64 \
  --eval-context-size 800 \
  --eval-num-features 500 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# ==============================================================================
# HELD-OUT STRESS TESTS ACROSS P = 50, 200, 500, 2000 (N_ctx = 800, 128 STUDIES)
# ==============================================================================
for P in 50 200 500 2000; do
  echo "=== Evaluating v2 held-out mixture P=${P} (N_ctx=800, 128 studies) ==="
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

# Family-specific held-out checks (unseen Nemotron oncology designs & OCI reequation/calculator DGPs)
for FAM in nemotron_design reequation_calc; do
  for P in 500 2000; do
    echo "=== Evaluating v2 held-out family=${FAM} P=${P} (N_ctx=800, 96 studies) ==="
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
