#!/usr/bin/env bash
#
# Fast ~2-hour H200 warm-start fine-tune (v3.1) from finished v3 best_checkpoint.pt (step 25,000)
# on the de-aliased heavy-missingness & noise-synonym-matched 5-family prior:
#   - Erases v3's miss_rate > 0.55 screener shortcut (where fragmented active columns
#     previously exceeded noise max_missing_rate=0.45) while preserving all 360k studies
#     of v3 causal learning.
#   - Trains noise columns across 0-95% missingness + matched noise fragmented synonym clusters.
#   - Saves to artifacts/tabular_causal_pfn_v3_1_h200 (leaving v3 intact) and runs full evals.
#
# Usage:
#   ./scripts/run_tabular_pfn_v3_1_finetune_h200.sh
#   CUDA_VISIBLE_DEVICES=0 STEPS=5000 ./scripts/run_tabular_pfn_v3_1_finetune_h200.sh
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
INIT_FROM="${INIT_FROM:-${RUN_ROOT}/tabular_causal_pfn_v3_h200/best_checkpoint.pt}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_v3_1_h200}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4000}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "Using PYTHON=${PYTHON}"
echo "Using INIT_FROM=${INIT_FROM}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

"${PYTHON}" -m oci.tabular_pfn.cli train \
  --run-dir "${RUN_DIR}" \
  --init-from "${INIT_FROM}" \
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
  --grad-accum-steps 12 \
  --lr 1e-4 \
  --warmup-steps 200 \
  --save-every 500 \
  --eval-every 500 \
  --eval-studies 64 \
  --eval-context-size 800 \
  --eval-num-features 500 \
  --num-workers "${NUM_WORKERS}" \
  --seed 2026 \
  --device "${DEVICE}"

for P in 50 200 500 2000 3500; do
  echo "=== Evaluating v3.1 held-out 5-family mixture P=${P} (N_ctx=800, 128 studies) ==="
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

echo "=== Evaluating v3.1 held-out large-N P=2000 (N_ctx=4000, 64 studies) ==="
"${PYTHON}" -m oci.tabular_pfn.cli evaluate \
  --run-dir "${RUN_DIR}" \
  --studies 64 \
  --context-size 4000 \
  --query-size 128 \
  --num-features 2000 \
  --num-workers "${NUM_WORKERS}" \
  --seed 750001 \
  --device "${DEVICE}" | tee "${RUN_DIR}/eval_n4000_p2000.json"
