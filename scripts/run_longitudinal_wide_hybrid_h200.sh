#!/usr/bin/env bash
#
# Wide-capacity two-stage curriculum TinyBERT longitudinal hybrid R-learner +
# study-adaptive lexical SVD + residual CausalPFN training and N=5,000 eval
# on an H200 GPU.
#
# Upgrades over scripts/run_longitudinal_curriculum_h200.sh:
#   - --solver-dim 256  (128 note + 128 structured neural solver dims; removes
#     the 32-dim note projection bottleneck observed in step-450 probes)
#   - --note-slots 16   (2x more learned multi-query token-attention slots)
#   - --lexical-dim 128 (study-adaptive unsupervised SVD over hashed note
#     1-3grams, preserving 27-70 note findings even on unseen cancers)
#
# Usage:
#   ./scripts/run_longitudinal_wide_hybrid_h200.sh
#   RUN_ROOT=/tmp/longitudinal NUM_WORKERS=16 ./scripts/run_longitudinal_wide_hybrid_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_tinybert_wide_hybrid}"
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
# STAGE 1 (steps 1..600): REPRESENTATION & SOLVER LEARNING
#   Unfrozen TinyBERT + 16 note slots + 256-d neural solver + 128-d lexical SVD
#   N_ctx = 64..256 + meta-batch = 32 studies/update (19,200 studies)
# ==============================================================================
"${PYTHON}" -m oci.longitudinal.cli train \
  --run-dir "${RUN_DIR}" \
  --note-model "${NOTE_MODEL}" \
  --note-slots 16 \
  --solver-dim 256 \
  --lexical-dim 128 \
  --steps 600 \
  --save-every 50 \
  --min-context-size 64 \
  --context-size 256 \
  --query-size 32 \
  --grad-accum-steps 32 \
  --note-batch-size 256 \
  --note-grad-patients 64 \
  --num-workers "${NUM_WORKERS}" \
  --lr 6e-4 \
  --note-lr 3e-5 \
  --lr-scheduler cosine \
  --warmup-steps 30 \
  --precision bf16 \
  --eval-every 50 \
  --eval-studies 32 \
  --device "${DEVICE}"

# ==============================================================================
# STAGE 2 (steps 601..1200): SCALE TO LARGE COHORTS (N_ctx = 500..5000)
# ==============================================================================
"${PYTHON}" -m oci.longitudinal.cli resume \
  --run-dir "${RUN_DIR}" \
  --steps 1200 \
  --save-every 100 \
  --min-context-size 500 \
  --context-size 5000 \
  --query-size 64 \
  --grad-accum-steps 8 \
  --note-grad-patients 128 \
  --lr 2e-4 \
  --note-lr 1e-5 \
  --eval-every 100 \
  --eval-studies 32 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# ==============================================================================
# STAGE 3: FULL 1,000-DESIGN HELD-OUT EVALUATION AT N_ctx = 5,000
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
