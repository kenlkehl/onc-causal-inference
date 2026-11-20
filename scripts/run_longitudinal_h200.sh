#!/usr/bin/env bash
#
# Full TinyBERT longitudinal CausalPFN training and evaluation run up to
# N=5,000 context patients per study on an H200 GPU.
#
# Active by default:
#   1. Phase 1 (steps 1..500): frozen TinyBERT, N_ctx = 500..5000, query = 64
#   2. Phase 2 (steps 501..1500): unfrozen TinyBERT (--note-grad-patients 128)
#   3. Full held-out evaluation across all 1,000 held-out designs at N_ctx = 5000
#
# Optional commented sections below cover a quick smoke test, extending steps,
# LLM-rendered cohort evaluation, and unlabeled real-EHR prediction.
#
# Usage:
#   ./scripts/run_longitudinal_h200.sh
#   PYTHON=/path/to/python RUN_ROOT=/tmp/longitudinal NUM_WORKERS=16 ./scripts/run_longitudinal_h200.sh
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

# Tip: override RUN_ROOT=/tmp/longitudinal on local NVMe if /data1 is slow for checkpoints.
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_tinybert_n5000}"
DEVICE="${DEVICE:-cuda}"
NOTE_MODEL="${NOTE_MODEL:-huawei-noah/TinyBERT_General_4L_312D}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "Using PYTHON=${PYTHON}"
echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using NOTE_MODEL=${NOTE_MODEL}"
echo "Using NUM_WORKERS=${NUM_WORKERS}"
echo "Using DEVICE=${DEVICE}"

# ==============================================================================
# 1. FULL TRAINING + HELD-OUT EVAL AT N_CTX UP TO 5,000 — ACTIVE BY DEFAULT
# ==============================================================================

# Phase 1 (steps 1..500): freeze TinyBERT to rapidly warm up the cross-modal
# patient encoder and two-way CausalPFN study layers across N_ctx = 500..5000.
"${PYTHON}" -m oci.longitudinal.cli train \
  --run-dir "${RUN_DIR}" \
  --note-model "${NOTE_MODEL}" \
  --steps 500 \
  --save-every 100 \
  --min-context-size 500 \
  --context-size 5000 \
  --query-size 64 \
  --grad-accum-steps 4 \
  --note-batch-size 256 \
  --freeze-note-backbone \
  --num-workers "${NUM_WORKERS}" \
  --lr 3e-4 \
  --note-lr 3e-5 \
  --lr-scheduler cosine \
  --warmup-steps 50 \
  --precision bf16 \
  --eval-every 100 \
  --eval-studies 32 \
  --device "${DEVICE}"

# Phase 2 (steps 501..1500): unfreeze TinyBERT end-to-end with partial-gradient
# context encoding (128 random context patients + 64 queries backprop through
# TinyBERT; remaining up to 4,872 context patients encode in 1,024-note no_grad batches).
"${PYTHON}" -m oci.longitudinal.cli resume \
  --run-dir "${RUN_DIR}" \
  --steps 1500 \
  --save-every 100 \
  --no-freeze-note-backbone \
  --note-grad-patients 128 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# Full held-out evaluation across all 1,000 held-out designs at N_ctx = 5,000.
"${PYTHON}" -m oci.longitudinal.cli evaluate \
  --run-dir "${RUN_DIR}" \
  --studies 1000 \
  --context-size 5000 \
  --query-size 64 \
  --note-batch-size 512 \
  --num-workers "${NUM_WORKERS}" \
  --precision bf16 \
  --seed 42 \
  --device "${DEVICE}" | tee "${RUN_DIR}/eval_n5000_1000studies.json"


# ==============================================================================
# 2. OPTIONAL: QUICK N=5,000 SMOKE TEST (uncomment & comment out Section 1)
# ==============================================================================
# SMOKE_DIR="${RUN_ROOT}/longitudinal_tinybert_n5000_smoke"
#
# "${PYTHON}" -m oci.longitudinal.cli train \
#   --run-dir "${SMOKE_DIR}" \
#   --note-model "${NOTE_MODEL}" \
#   --steps 20 \
#   --save-every 10 \
#   --min-context-size 500 \
#   --context-size 5000 \
#   --query-size 64 \
#   --grad-accum-steps 2 \
#   --note-batch-size 256 \
#   --note-grad-patients 128 \
#   --num-workers "${NUM_WORKERS}" \
#   --lr 3e-4 \
#   --note-lr 3e-5 \
#   --lr-scheduler cosine \
#   --warmup-steps 5 \
#   --precision bf16 \
#   --eval-every 20 \
#   --eval-studies 16 \
#   --device "${DEVICE}"


# ==============================================================================
# 3. OPTIONAL: EXTEND TRAINING FURTHER (uncomment to resume past step 1,500)
# ==============================================================================
# "${PYTHON}" -m oci.longitudinal.cli resume \
#   --run-dir "${RUN_DIR}" \
#   --steps 3000 \
#   --save-every 100 \
#   --no-freeze-note-backbone \
#   --note-grad-patients 128 \
#   --num-workers "${NUM_WORKERS}" \
#   --device "${DEVICE}"


# ==============================================================================
# 4. OPTIONAL: LLM-RENDERED HELD-OUT COHORT GENERATION & EVALUATION
#    (requires an OpenAI-compatible vLLM endpoint serving LLM_MODEL)
# ==============================================================================
# LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1}"
# LLM_MODEL="${LLM_MODEL:-MODEL_ID}"
# LLM_EVAL_DIR="${RUN_ROOT}/eval_llm_bladder_q01"
#
# "${PYTHON}" -m synthetic_data.cli longitudinal \
#   --design-bank synthetic_data/designs/nemotron_5000.json.gz \
#   --design-id bladder_q01_v001 \
#   --dgp-seed 101 \
#   --seed 42 \
#   --dataset-size 5100 \
#   --note-style llm \
#   --api-url "${LLM_URL}" \
#   --model "${LLM_MODEL}" \
#   --num-workers 32 \
#   --output-dir "${LLM_EVAL_DIR}"
#
# "${PYTHON}" -m oci.longitudinal.cli evaluate \
#   --run-dir "${RUN_DIR}" \
#   --eval-cohort "${LLM_EVAL_DIR}/dataset.parquet" \
#   --studies 16 \
#   --context-size 5000 \
#   --query-size 64 \
#   --note-batch-size 512 \
#   --precision bf16 \
#   --seed 42 \
#   --device "${DEVICE}" | tee "${RUN_DIR}/eval_llm_bladder_q01.json"


# ==============================================================================
# 5. OPTIONAL: UNLABELED INFERENCE (REAL EHR OR EXTERNAL PARQUET)
# ==============================================================================
# "${PYTHON}" -m oci.longitudinal.cli predict \
#   --run-dir "${RUN_DIR}" \
#   --context /path/to/labeled_context.parquet \
#   --queries /path/to/unlabeled_queries.parquet \
#   --output "${RUN_ROOT}/predictions.parquet" \
#   --batch-size 64 \
#   --note-batch-size 512 \
#   --precision bf16 \
#   --device "${DEVICE}"
