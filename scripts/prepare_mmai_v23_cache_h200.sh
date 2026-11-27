#!/usr/bin/env bash
#
# One-time H200 pre-embedding & clinical factor extraction cache builder for MMAI v23
# (/ksg/kehl_mm_data/mmai/v23/data/no_phi).
#
# Enforces EXACT patient-disjointness from TinyOncBERT's training curation cutoff
# (artifacts/tinyoncbert_6l_384d/work/curated/manifest.json, notes_read=564,224):
#   - train_unseen : 37,631 v23 train patients (0 notes ever read by TinyOncBERT)
#   - validation   :  8,470 v23 validation patients (held-out Option B monitoring)
#   - test         :  8,340 v23 test patients (strictly held-out final zero-shot eval)
#   - train_seen   : 33,449 TinyOncBERT-seen train patients (EXCLUDED by default)
#
# Uses balanced multi-note sentence extraction (cap 320 clauses across all ~14 notes/patient)
# and streams all_synthetic_notes.parquet in a single pass (<2 GB peak RAM, ~5-7 min on 1x H200,
# ~11-12 GB fp16 memmap cache on disk).
#
# Usage:
#   ./scripts/prepare_mmai_v23_cache_h200.sh
#   OUTPUT_DIR=artifacts/mmai_v23_option_b_cache DEVICE=cuda ./scripts/prepare_mmai_v23_cache_h200.sh
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

V23_ROOT="${V23_ROOT:-/ksg/kehl_mm_data/mmai/v23/data/no_phi}"
NOTES_PARQUET="${NOTES_PARQUET:-${V23_ROOT}/synthetic_dataset_n5000_screened_20261106/patient_summaries/all_synthetic_notes.parquet}"
META_PARQUET="${META_PARQUET:-${V23_ROOT}/synthetic_dataset_n5000_screened_20261106/patient_summaries/synthetic_patient_metadata.parquet}"
EXTRACTION_SOURCE="${EXTRACTION_SOURCE:-${V23_ROOT}/dashboard_synthetic/extraction_records}"
BACKBONE_PATH="${BACKBONE_PATH:-${REPO_ROOT}/artifacts/tinyoncbert_6l_384d}"
TINYONCBERT_MANIFEST="${TINYONCBERT_MANIFEST:-${BACKBONE_PATH}/work/curated/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/mmai_v23_option_b_cache}"
MAX_SENTENCES="${MAX_SENTENCES:-320}"
SENT_BATCH_SIZE="${SENT_BATCH_SIZE:-1024}"
FLUSH_PATIENTS="${FLUSH_PATIENTS:-256}"
NUM_IO_THREADS="${NUM_IO_THREADS:-16}"
DEVICE="${DEVICE:-cuda}"

echo "=== Preparing TinyOncBERT-disjoint MMAI v23 Option B cache ==="
echo "Using PYTHON=${PYTHON}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using NOTES_PARQUET=${NOTES_PARQUET}"
echo "Using EXTRACTION_SOURCE=${EXTRACTION_SOURCE}"
echo "Using BACKBONE_PATH=${BACKBONE_PATH}"
echo "Using TINYONCBERT_MANIFEST=${TINYONCBERT_MANIFEST}"
echo "Using MAX_SENTENCES=${MAX_SENTENCES} SENT_BATCH_SIZE=${SENT_BATCH_SIZE} DEVICE=${DEVICE}"

"${PYTHON}" -m oci.longitudinal.mmai_v23 prepare \
  --output-dir "${OUTPUT_DIR}" \
  --notes-parquet "${NOTES_PARQUET}" \
  --meta-parquet "${META_PARQUET}" \
  --extraction-source "${EXTRACTION_SOURCE}" \
  --backbone-path "${BACKBONE_PATH}" \
  --tinyoncbert-manifest "${TINYONCBERT_MANIFEST}" \
  --max-sentences-per-patient "${MAX_SENTENCES}" \
  --sent-batch-size "${SENT_BATCH_SIZE}" \
  --flush-patients "${FLUSH_PATIENTS}" \
  --num-io-threads "${NUM_IO_THREADS}" \
  --device "${DEVICE}"
