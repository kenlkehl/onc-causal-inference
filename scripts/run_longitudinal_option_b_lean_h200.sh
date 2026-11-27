#!/usr/bin/env bash
#
# Lean Option B (v6.1 / v7): Warm-started hybrid multimodal fine-tuning using the default lean
# Option-A input structure (sent_256 = 220d structured summary + 256d frozen TinyOncBERT SentKM
# = 476 unmixed explicit columns) + 24 trainable StructuredTemporalEncoder columns = 500 total!
#
# Avoids collinear feature duplication (nadir/peak + latest-SentKM + LexSVD) that diluted zero-shot
# screening on full-LLM cohorts at N=800, while letting the 24 learned temporal columns carry
# genuinely orthogonal trajectory dynamics.
#
# Usage:
#   ./scripts/run_longitudinal_option_b_lean_h200.sh
#   TABULAR_INIT_DIR=artifacts/tabular_causal_pfn_v5_h200 RUN_DIR=artifacts/longitudinal_option_b_v7 ./scripts/run_longitudinal_option_b_lean_h200.sh
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
TABULAR_INIT_DIR="${TABULAR_INIT_DIR:-${RUN_ROOT}/tabular_causal_pfn_v5_h200}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_option_b_lean}"
BACKBONE_PATH="${BACKBONE_PATH:-${RUN_ROOT}/tinyoncbert_6l_384d}"
STRUCTURED_INIT_DIR="${STRUCTURED_INIT_DIR:-${RUN_ROOT}/longitudinal_tinyoncbert_v5}"
PARAPHRASE_BANK="${PARAPHRASE_BANK:-${REPO_ROOT}/synthetic_data/designs/nemotron_5000_paraphrases.json.gz}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-1200}"
NUM_WORKERS="${NUM_WORKERS:-6}"

echo "=== Lean Option B (sent_256 + 24 learned temporal = 500 cols) ==="
echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using TABULAR_INIT_DIR=${TABULAR_INIT_DIR}"
echo "Using BACKBONE_PATH=${BACKBONE_PATH}"
echo "Using STRUCTURED_INIT_DIR=${STRUCTURED_INIT_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

"${PYTHON}" -m oci.longitudinal.option_b train \
  --run-dir "${RUN_DIR}" \
  --tabular-init-dir "${TABULAR_INIT_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --structured-init-dir "${STRUCTURED_INIT_DIR}" \
  --paraphrase-bank "${PARAPHRASE_BANK}" \
  --feature-preset sent_256 \
  --learned-temporal-dim 24 \
  --event-width 128 \
  --event-layers 2 \
  --event-heads 4 \
  --temporal-chunk-size 1024 \
  --steps "${STEPS}" \
  --grad-accum-steps 8 \
  --multimodal-prob 0.5 \
  --min-context-size 100 \
  --context-size 5000 \
  --query-size 64 \
  --tabular-min-context 80 \
  --tabular-max-context 4000 \
  --tabular-query-size 128 \
  --tabular-min-features 20 \
  --tabular-max-features 2500 \
  --temporal-lr 2e-4 \
  --tabular-lr 3e-5 \
  --weight-decay 1e-2 \
  --tabular-weight-decay 1e-4 \
  --anchor-weight 1e-4 \
  --warmup-steps 30 \
  --save-every 100 \
  --eval-every 100 \
  --eval-studies 32 \
  --eval-context-size 5000 \
  --eval-query-size 64 \
  --tabular-eval-studies 32 \
  --tabular-eval-context 800 \
  --tabular-eval-features 500 \
  --seed 42 \
  --precision auto \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

echo "=== Running 1,000-study held-out paraphrased evaluation ==="
"${PYTHON}" -m oci.longitudinal.option_b evaluate \
  --run-dir "${RUN_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --studies 1000 \
  --context-size 5000 \
  --query-size 64 \
  --seed 42 \
  --precision auto \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --output "${RUN_DIR}/eval_n5000_1000studies.json"

echo "=== Running full Option A vs Option B vs Run 5 benchmark suite ==="
"${PYTHON}" "${REPO_ROOT}/scripts/eval_option_a_v4.py" \
  --tabular-v4-dir "${TABULAR_INIT_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --run5-dir "${STRUCTURED_INIT_DIR}" \
  --option-b-run-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/eval_full_suite" \
  --device "${DEVICE}"
