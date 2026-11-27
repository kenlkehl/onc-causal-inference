#!/usr/bin/env bash
#
# Lean Option B (v7): Warm-started hybrid multimodal fine-tuning built on Tabular v5 +
# lean Option-A input structure (sent_256 = 220d structured summary + 256d frozen TinyOncBERT
# SentKM = 476 unmixed explicit columns) + 24 trainable StructuredTemporalEncoder columns = 500 total.
#
# When artifacts/mmai_v23_option_b_cache/manifest.json is present (built via
# ./scripts/prepare_mmai_v23_cache_h200.sh), automatically enables 3-way episode mixing:
#   - 35% (3/8 micro-steps): TinyOncBERT-disjoint MMAI v23 train_unseen (~14-note pan-cancer charts)
#   - 25% (2/8 micro-steps): nemotron_5000 5-stream longitudinal simulator (labs/vitals/PROs/notes)
#   - 40% (3/8 micro-steps): pure-tabular Tabular v5 replay (with full role + screener supervision)
# and evaluates zero-shot on held-out v23 validation during training + v23 test at completion.
#
# Usage:
#   ./scripts/run_longitudinal_option_b_lean_h200.sh
#   V23_PROB=0 ./scripts/run_longitudinal_option_b_lean_h200.sh   # 2-way ablation without v23
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_option_b_v7}"
BACKBONE_PATH="${BACKBONE_PATH:-${RUN_ROOT}/tinyoncbert_6l_384d}"
STRUCTURED_INIT_DIR="${STRUCTURED_INIT_DIR:-${RUN_ROOT}/longitudinal_tinyoncbert_v5}"
PARAPHRASE_BANK="${PARAPHRASE_BANK:-${REPO_ROOT}/synthetic_data/designs/nemotron_5000_paraphrases.json.gz}"
V23_CACHE_DIR="${V23_CACHE_DIR:-${RUN_ROOT}/mmai_v23_option_b_cache}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-1200}"
NUM_WORKERS="${NUM_WORKERS:-3}"

V23_ARGS=()
USE_V23=0
if [[ -f "${V23_CACHE_DIR}/manifest.json" ]] && [[ "${V23_PROB:-0.35}" != "0" ]] && [[ "${V23_PROB:-0.35}" != "0.0" ]]; then
  USE_V23=1
  V23_PROB="${V23_PROB:-0.35}"
  MULTIMODAL_PROB="${MULTIMODAL_PROB:-0.25}"
  V23_ARGS=(
    --v23-cache-dir "${V23_CACHE_DIR}"
    --v23-prob "${V23_PROB}"
    --v23-train-split "${V23_TRAIN_SPLIT:-train_unseen}"
    --v23-eval-split "${V23_EVAL_SPLIT:-validation}"
    --v23-eval-studies "${V23_EVAL_STUDIES:-32}"
    --v23-eval-context "${V23_EVAL_CONTEXT:-800}"
    --v23-min-context "${V23_MIN_CONTEXT:-120}"
    --v23-max-context "${V23_MAX_CONTEXT:-4000}"
    --v23-query-size "${V23_QUERY_SIZE:-128}"
    --v23-notes-only-prob "${V23_NOTES_ONLY_PROB:-0.55}"
    --v23-within-cancer-prob "${V23_WITHIN_CANCER_PROB:-0.65}"
  )
else
  V23_PROB="0.0"
  MULTIMODAL_PROB="${MULTIMODAL_PROB:-0.5}"
fi

echo "=== Lean Option B v7 (sent_256 + 24 learned temporal = 500 cols) ==="
echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using TABULAR_INIT_DIR=${TABULAR_INIT_DIR}"
echo "Using BACKBONE_PATH=${BACKBONE_PATH}"
echo "Using STRUCTURED_INIT_DIR=${STRUCTURED_INIT_DIR}"
echo "Using USE_V23=${USE_V23} V23_CACHE_DIR=${V23_CACHE_DIR} V23_PROB=${V23_PROB} MULTIMODAL_PROB=${MULTIMODAL_PROB}"
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
  --multimodal-prob "${MULTIMODAL_PROB}" \
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
  --device "${DEVICE}" \
  "${V23_ARGS[@]}"

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

if [[ -f "${V23_CACHE_DIR}/manifest.json" ]]; then
  echo "=== Running strictly held-out MMAI v23 TEST-split evaluation (Tabular init vs Option B v7) ==="
  "${PYTHON}" -m oci.longitudinal.mmai_v23 evaluate \
    --cache-dir "${V23_CACHE_DIR}" \
    --split test \
    --tabular-run-dir "${TABULAR_INIT_DIR}" \
    --option-b-run-dir "${RUN_DIR}" \
    --backbone-path "${BACKBONE_PATH}" \
    --studies 64 \
    --context-size 800 \
    --query-size 128 \
    --sent-km-k 256 \
    --device "${DEVICE}" \
    --output "${RUN_DIR}/eval_v23_test.json"
fi

echo "=== Running full Option A vs Option B vs Run 5 benchmark suite ==="
"${PYTHON}" "${REPO_ROOT}/scripts/eval_option_a_v4.py" \
  --tabular-v4-dir "${TABULAR_INIT_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --run5-dir "${STRUCTURED_INIT_DIR}" \
  --option-b-run-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/eval_full_suite" \
  --device "${DEVICE}"
