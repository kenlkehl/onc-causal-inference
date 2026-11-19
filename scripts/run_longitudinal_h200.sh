#!/usr/bin/env bash
#
# Run the longitudinal multimodal CausalPFN workflow on an H200 GPU.
#
# By default, only Section 1 (the quick smoke test) is active. Uncomment
# Sections 2-6 below as needed for the two-phase large-context prototype run
# (N_ctx = 128..512), full 1,000-design held-out evaluation, LLM-rendered
# cohort evaluation, or unlabeled predict.
#
# Usage:
#   ./scripts/run_longitudinal_h200.sh
#   PYTHON=/path/to/python RUN_ROOT=/tmp/longitudinal ./scripts/run_longitudinal_h200.sh
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
DEVICE="${DEVICE:-cuda}"

echo "Using PYTHON=${PYTHON}"
echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using RUN_ROOT=${RUN_ROOT}"
echo "Using DEVICE=${DEVICE}"

# ==============================================================================
# 1. QUICK SMOKE TEST (~3-5 min on H200) — ACTIVE BY DEFAULT
#    Exercises variable large context (64-256), partial-grad note backbone,
#    bf16 autocast, background CPU episode prefetching, and baseline eval.
# ==============================================================================
SMOKE_DIR="${RUN_ROOT}/longitudinal_smoke"

"${PYTHON}" -m oci.longitudinal.cli train \
  --run-dir "${SMOKE_DIR}" \
  --steps 100 \
  --save-every 50 \
  --min-context-size 64 \
  --context-size 256 \
  --query-size 32 \
  --grad-accum-steps 4 \
  --note-batch-size 64 \
  --note-grad-patients 48 \
  --num-workers 4 \
  --lr-scheduler cosine \
  --warmup-steps 10 \
  --precision bf16 \
  --eval-every 50 \
  --eval-studies 32 \
  --device "${DEVICE}"

"${PYTHON}" -m oci.longitudinal.cli evaluate \
  --run-dir "${SMOKE_DIR}" \
  --studies 64 \
  --context-size 256 \
  --query-size 32 \
  --note-batch-size 128 \
  --num-workers 4 \
  --precision bf16 \
  --seed 42 \
  --device "${DEVICE}" | tee "${SMOKE_DIR}/eval_smoke_64.json"


# ==============================================================================
# 2. FULL H200 PROTOTYPE RUN — TWO-PHASE LARGE-CONTEXT TRAINING (uncomment to run)
#    Phase 1: Freeze LFM2.5-230M and rapidly train the cross-modal & two-way
#             CausalPFN layers across N_ctx = 128..512 (8 studies/step).
#    Phase 2: Unfreeze LFM2.5-230M via resume with --note-grad-patients 64 so
#             the backbone adapts end-to-end while keeping full 128..512 context.
# ==============================================================================
# PROTO_DIR="${RUN_ROOT}/longitudinal_h200_v1"
#
# # Phase 1 (steps 1..1000, frozen note backbone)
# "${PYTHON}" -m oci.longitudinal.cli train \
#   --run-dir "${PROTO_DIR}" \
#   --steps 1000 \
#   --save-every 100 \
#   --min-context-size 128 \
#   --context-size 512 \
#   --query-size 32 \
#   --grad-accum-steps 8 \
#   --note-batch-size 64 \
#   --freeze-note-backbone \
#   --num-workers 8 \
#   --lr 3e-4 \
#   --note-lr 1e-5 \
#   --lr-scheduler cosine \
#   --warmup-steps 100 \
#   --precision bf16 \
#   --eval-every 200 \
#   --eval-studies 64 \
#   --device "${DEVICE}"
#
# # Phase 2 (steps 1001..2500, unfreeze note backbone with partial-grad context)
# "${PYTHON}" -m oci.longitudinal.cli resume \
#   --run-dir "${PROTO_DIR}" \
#   --steps 2500 \
#   --save-every 100 \
#   --no-freeze-note-backbone \
#   --note-grad-patients 64 \
#   --num-workers 8 \
#   --device "${DEVICE}"


# ==============================================================================
# 3. EXTEND / RESUME PROTOTYPE RUN FURTHER (uncomment if interrupted or extending)
# ==============================================================================
# PROTO_DIR="${RUN_ROOT}/longitudinal_h200_v1"
#
# "${PYTHON}" -m oci.longitudinal.cli resume \
#   --run-dir "${PROTO_DIR}" \
#   --steps 4000 \
#   --save-every 100 \
#   --num-workers 8 \
#   --device "${DEVICE}"


# ==============================================================================
# 4. FULL HELD-OUT EVALUATION (all 1,000 held-out designs + baselines at N=512)
# ==============================================================================
# PROTO_DIR="${RUN_ROOT}/longitudinal_h200_v1"
#
# "${PYTHON}" -m oci.longitudinal.cli evaluate \
#   --run-dir "${PROTO_DIR}" \
#   --studies 1000 \
#   --context-size 512 \
#   --query-size 32 \
#   --note-batch-size 128 \
#   --num-workers 8 \
#   --precision bf16 \
#   --seed 42 \
#   --device "${DEVICE}" | tee "${PROTO_DIR}/eval_1000.json"


# ==============================================================================
# 5. OPTIONAL: LLM-RENDERED HELD-OUT COHORT GENERATION & EVALUATION
#    (requires an OpenAI-compatible vLLM endpoint serving LLM_MODEL)
# ==============================================================================
# PROTO_DIR="${RUN_ROOT}/longitudinal_h200_v1"
# LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1}"
# LLM_MODEL="${LLM_MODEL:-MODEL_ID}"
# LLM_EVAL_DIR="${RUN_ROOT}/eval_llm_bladder_q01"
#
# "${PYTHON}" -m synthetic_data.cli longitudinal \
#   --design-bank synthetic_data/designs/nemotron_5000.json.gz \
#   --design-id bladder_q01_v001 \
#   --dgp-seed 101 \
#   --seed 42 \
#   --dataset-size 600 \
#   --note-style llm \
#   --api-url "${LLM_URL}" \
#   --model "${LLM_MODEL}" \
#   --num-workers 16 \
#   --output-dir "${LLM_EVAL_DIR}"
#
# "${PYTHON}" -m oci.longitudinal.cli evaluate \
#   --run-dir "${PROTO_DIR}" \
#   --eval-cohort "${LLM_EVAL_DIR}/dataset.parquet" \
#   --studies 16 \
#   --context-size 512 \
#   --query-size 64 \
#   --note-batch-size 128 \
#   --precision bf16 \
#   --seed 42 \
#   --device "${DEVICE}" | tee "${PROTO_DIR}/eval_llm_bladder_q01.json"


# ==============================================================================
# 6. OPTIONAL: UNLABELED INFERENCE (REAL EHR OR EXTERNAL PARQUET)
# ==============================================================================
# PROTO_DIR="${RUN_ROOT}/longitudinal_h200_v1"
#
# "${PYTHON}" -m oci.longitudinal.cli predict \
#   --run-dir "${PROTO_DIR}" \
#   --context /path/to/labeled_context.parquet \
#   --queries /path/to/unlabeled_queries.parquet \
#   --output "${RUN_ROOT}/predictions.parquet" \
#   --batch-size 32 \
#   --note-batch-size 128 \
#   --precision bf16 \
#   --device "${DEVICE}"
