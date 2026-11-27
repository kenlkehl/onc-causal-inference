#!/usr/bin/env bash
#
# Option A (Zero-Shot Composition) full evaluation suite for finished Tabular PFN (v4 / v5):
#   - Default primary input structure: lean sent_256 (476 unmixed columns = 220d structured
#     summary + 256d context-fitted frozen TinyOncBERT SentKM), benchmarked alongside struct_only,
#     sent_256_lex_128, option_a_temp128, and option_a_full ablations.
#   - Head-to-head vs Longitudinal Run 5 (step 1200) across:
#       Suite 1: Real OCI Stage-2 extracted catalogs (S0/n1000/seed101 & seed102, raw unmasked)
#       Suite 2: 16 Full-LLM realistic cohorts (N_ctx = 800 and N_ctx = 200)
#       Suite 3: 44 complex out-of-family Full-LLM cohorts (epistatic_s8, nonparametric_s10,
#                composite_var_width, dense_50_50)
#       Suite 4: Complex tabular oracle stress cohorts (S7..S12b + 5 oncology contexts)
#
# Safe to run simultaneously alongside Option B training (uses ~8-12 GiB VRAM on 1 GPU, ~10-15 min).
#
# Usage:
#   ./scripts/run_option_a_eval_h200.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_option_a_eval_h200.sh
#   OPTION_B_RUN_DIR=artifacts/longitudinal_option_b_v6 ./scripts/run_option_a_eval_h200.sh
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

TABULAR_V4_DIR="${TABULAR_V4_DIR:-${REPO_ROOT}/artifacts/tabular_causal_pfn_v4_h200}"
BACKBONE_PATH="${BACKBONE_PATH:-${REPO_ROOT}/artifacts/tinyoncbert_6l_384d}"
RUN5_DIR="${RUN5_DIR:-${REPO_ROOT}/artifacts/longitudinal_tinyoncbert_v5}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/eval_option_a_v4}"
DEVICE="${DEVICE:-cuda}"

echo "=== Option A Zero-Shot Evaluation Suite ==="
echo "Using PYTHON=${PYTHON}"
echo "Using TABULAR_V4_DIR=${TABULAR_V4_DIR}"
echo "Using BACKBONE_PATH=${BACKBONE_PATH}"
echo "Using RUN5_DIR=${RUN5_DIR}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using DEVICE=${DEVICE}"

CMD=(
  "${PYTHON}" "${REPO_ROOT}/scripts/eval_option_a_v4.py"
  --tabular-v4-dir "${TABULAR_V4_DIR}"
  --backbone-path "${BACKBONE_PATH}"
  --run5-dir "${RUN5_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

if [[ -n "${OPTION_B_RUN_DIR:-}" ]]; then
  echo "Including Option B checkpoint: ${OPTION_B_RUN_DIR}"
  CMD+=(--option-b-run-dir "${OPTION_B_RUN_DIR}")
fi

"${CMD[@]}" "$@"
