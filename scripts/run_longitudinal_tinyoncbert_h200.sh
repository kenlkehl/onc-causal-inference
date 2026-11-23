#!/usr/bin/env bash
#
# Run 5: Next-gen TinyOncBERT longitudinal causal foundation model with:
#   - TinyOncBERT (6L/384D, PubMedBERT vocab, negation-preserving Qwen3 distillation)
#   - --paraphrase-bank (Nemotron paraphrased multi-section clinical notes)
#   - --sentence-concept-dim 128 (study-adaptive spherical k-means sentence MaxSim)
#   - --lexical-dim 128 (study-adaptive 1-3gram SVD)
#   - --block-ridge (2D LOO-CV Ridge independently shrinking text vs. structured blocks
#     for propensity, outcome, and CATE so noisy text never dilutes structured signals)
#   - Single-stage log-uniform N_ctx = 100..10,000 + --per-slot-note-temporal
#
# Usage:
#   ./scripts/run_longitudinal_tinyoncbert_h200.sh
#   RUN_ROOT=/tmp/longitudinal NUM_WORKERS=16 ./scripts/run_longitudinal_tinyoncbert_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_tinyoncbert_v5}"
DEVICE="${DEVICE:-cuda}"
NOTE_MODEL="${NOTE_MODEL:-${REPO_ROOT}/artifacts/tinyoncbert_6l_384d}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PARAPHRASE_BANK="${PARAPHRASE_BANK:-${REPO_ROOT}/synthetic_data/designs/nemotron_5000_paraphrases.json.gz}"

echo "Using PYTHON=${PYTHON}"
echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using NOTE_MODEL=${NOTE_MODEL}"
echo "Using PARAPHRASE_BANK=${PARAPHRASE_BANK}"
echo "Using NUM_WORKERS=${NUM_WORKERS}"
echo "Using DEVICE=${DEVICE}"

# ==============================================================================
# SINGLE-STAGE TRAINING (steps 1..1200): LOG-UNIFORM N_ctx = 100..10,000
# ==============================================================================
"${PYTHON}" -m oci.longitudinal.cli train \
  --run-dir "${RUN_DIR}" \
  --note-model "${NOTE_MODEL}" \
  --paraphrase-bank "${PARAPHRASE_BANK}" \
  --note-slots 16 \
  --per-slot-note-temporal \
  --solver-dim 256 \
  --lexical-dim 128 \
  --sentence-concept-dim 128 \
  --ridge-grid-size 7 \
  --block-ridge \
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
# FINAL EVALUATION: 1,000 HELD-OUT DESIGNS AT N_ctx = 5,000 (PARAPHRASED NOTES)
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
