#!/usr/bin/env bash
#
# Single-command H200 launcher for SparseTabularCausalPFN (P = 20..2,000, N = 80..3,000)
# targeting high-dimensional sparse all-evidence Stage-2 candidate feature tables,
# followed by held-out stress-test benchmarks at P = 50, 200, 500, and 2,000.
#
# Usage:
#   ./scripts/run_tabular_pfn_h200.sh
#   STEPS=6000 RUN_ROOT=/tmp/tabular_pfn ./scripts/run_tabular_pfn_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_p2000}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-12000}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# ==============================================================================
# TRAINING: LOG-UNIFORM N_ctx = 80..3,000, P = 20..2,000 ACROSS 3 OCI DGP FAMILIES
# (50% sparse_oci + 25% reequation_calc + 25% nemotron_design train-split)
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
  --col-dim 128 \
  --study-width 128 \
  --study-layers 2 \
  --col-attn-layers 2 \
  --heads 4 \
  --top-k-nuis 64 \
  --top-k-cate 48 \
  --attn-slots 16 \
  --ridge-grid-size 7 \
  --grad-accum-steps 8 \
  --lr 5e-4 \
  --warmup-steps 200 \
  --save-every 500 \
  --eval-every 500 \
  --eval-studies 48 \
  --eval-context-size 800 \
  --eval-num-features 500 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# ==============================================================================
# HELD-OUT STRESS TESTS ACROSS P = 50, 200, 500, 2000 (N_ctx = 800, 128 STUDIES)
# ==============================================================================
for P in 50 200 500 2000; do
  echo "=== Evaluating held-out mixture P=${P} (N_ctx=800, 128 studies) ==="
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
    echo "=== Evaluating held-out family=${FAM} P=${P} (N_ctx=800, 96 studies) ==="
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
