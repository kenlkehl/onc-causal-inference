#!/usr/bin/env bash
#
# Single-command H200 launcher for TinyOncBERT (6L, 384D) distillation & MLM
# pretraining on ksg-dfci/mmai-synthetic-1226 synthetic oncology clinical notes.
#
# Usage:
#   ./scripts/run_tinyoncbert_h200.sh
#   OUTPUT_DIR=/tmp/tinyoncbert_6l_384d EPOCHS=3 ./scripts/run_tinyoncbert_h200.sh
#   COMPILE=1 ./scripts/run_tinyoncbert_h200.sh --eval-every 250
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
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

PARQUET="${PARQUET:-hf://datasets/ksg-dfci/mmai-synthetic-1226/patients/all_synthetic_notes.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/tinyoncbert_6l_384d}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/work}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
TEACHER_POOLING="${TEACHER_POOLING:-auto}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1024}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext}"
SVD_INIT_MODEL="${SVD_INIT_MODEL:-NeuML/pubmedbert-base-embeddings}"
NUM_SENTENCES="${NUM_SENTENCES:-2000000}"
NUM_WINDOWS="${NUM_WINDOWS:-250000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-4}"
WINDOW_MLM_EVERY="${WINDOW_MLM_EVERY:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
COMPILE="${COMPILE:-0}"

echo "Using PYTHON=${PYTHON}"
echo "Using REPO_ROOT=${REPO_ROOT}"
echo "Using PARQUET=${PARQUET}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using WORK_DIR=${WORK_DIR}"
echo "Using TEACHER_MODEL=${TEACHER_MODEL} (pooling=${TEACHER_POOLING})"
echo "Using TOKENIZER_MODEL=${TOKENIZER_MODEL}"
echo "Using SVD_INIT_MODEL=${SVD_INIT_MODEL:-<none>}"
echo "Using NUM_SENTENCES=${NUM_SENTENCES} NUM_WINDOWS=${NUM_WINDOWS}"
echo "Using BATCH_SIZE=${BATCH_SIZE} WINDOW_BATCH_SIZE=${WINDOW_BATCH_SIZE} EPOCHS=${EPOCHS} DEVICE=${DEVICE}"

EXTRA_ARGS=()
if [[ "${COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=(--compile)
fi

"${PYTHON}" -m oci.longitudinal.tinyoncbert \
  --parquet "${PARQUET}" \
  --work-dir "${WORK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-sentences "${NUM_SENTENCES}" \
  --num-windows "${NUM_WINDOWS}" \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-pooling "${TEACHER_POOLING}" \
  --teacher-batch-size "${TEACHER_BATCH_SIZE}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --svd-init-model "${SVD_INIT_MODEL}" \
  --num-hidden-layers 6 \
  --hidden-size 384 \
  --num-attention-heads 6 \
  --intermediate-size 1536 \
  --max-position-embeddings 512 \
  --lambda-emb 1.0 \
  --lambda-rel 0.5 \
  --lambda-neg 0.5 \
  --lambda-pol 0.1 \
  --lambda-mlm 0.25 \
  --neg-margin 0.25 \
  --window-mlm-every "${WINDOW_MLM_EVERY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --window-batch-size "${WINDOW_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --grad-clip 1.0 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
