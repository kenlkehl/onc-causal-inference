#!/usr/bin/env bash
#
# Single-stage TinyBERT longitudinal hybrid R-learner + study-adaptive lexical SVD
# + per-slot temporal note pooling training across log-uniform N_ctx = 100..10,000
# and final 1,000-study N=5,000 evaluation on an H200 GPU.
#
# Key upgrades from Run 1 (Hybrid) & Run 2 (Wide Hybrid) findings:
#   - No Stage 1/2 split: trains on log-uniform N_ctx = 100..10,000 from step 1
#     (median 1,000, mean ~2,150) so effect-modifier gradients flow from the start
#     while calibrating evenly across every cohort-size order of magnitude.
#   - --per-slot-note-temporal: preserves per-slot longitudinal progression
#     (max, latest, delta, and timestamp-conditioned attention across visits) for
#     all 16 note concept slots instead of only mean-collapsing before EventTransformer.
#   - --solver-dim 256 --lexical-dim 128 --ridge-grid-size 7: combines the wide
#     neural + study-adaptive lexical SVD capacity with a wider upper Ridge penalty
#     ceiling (~22,000) to prevent 384-dim CATE over-spread at large N.
#   - --eval-context-size 5000: keeps mid-training evals fast and directly
#     apples-to-apples with Run 1 & Run 2.
#
# Usage:
#   ./scripts/run_longitudinal_single_stage_h200.sh
#   RUN_ROOT=/tmp/longitudinal NUM_WORKERS=16 ./scripts/run_longitudinal_single_stage_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_tinybert_single_stage}"
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
# SINGLE-STAGE TRAINING (steps 1..1200): LOG-UNIFORM N_ctx = 100..10,000
# ==============================================================================
"${PYTHON}" -m oci.longitudinal.cli train \
  --run-dir "${RUN_DIR}" \
  --note-model "${NOTE_MODEL}" \
  --note-slots 16 \
  --per-slot-note-temporal \
  --solver-dim 256 \
  --lexical-dim 128 \
  --ridge-grid-size 7 \
  --steps 1200 \
  --save-every 100 \
  --min-context-size 100 \
  --context-size 10000 \
  --context-size-sampling log_uniform \
  --query-size 64 \
  --grad-accum-steps 8 \
  --note-batch-size 256 \
  --note-grad-patients 128 \
  --num-workers "${NUM_WORKERS}" \
  --lr 4e-4 \
  --note-lr 2e-5 \
  --lr-scheduler cosine \
  --warmup-steps 30 \
  --precision bf16 \
  --eval-every 100 \
  --eval-studies 32 \
  --eval-context-size 5000 \
  --device "${DEVICE}"

# ==============================================================================
# FINAL EVALUATION: 1,000 HELD-OUT DESIGNS AT N_ctx = 5,000 (APPLES-TO-APPLES)
# ==============================================================================
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
