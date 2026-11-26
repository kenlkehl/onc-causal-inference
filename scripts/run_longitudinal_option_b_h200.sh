#!/usr/bin/env bash
#
# Option B (v6): Warm-started hybrid multimodal fine-tuning of SparseTabularCausalPFN v4:
#   - Frozen TinyOncBERT (6L/384D) — preserves full-LLM generalization & avoids 10x BERT backprop
#   - All explicit Option-A columns kept unmixed (1,077 dims: 220 summary + 153 NaN nadir/peak +
#     64 ICD SVD + 256 all-note SentKM + 256 latest-note SentKM + 128 LexSVD)
#   - Appends 24 trainable temporal columns from StructuredTemporalEncoder (labs/vitals/encounters/PROs
#     EventTransformer + CrossModalTimeline, warm-started from Longitudinal Run 5!)
#   - Warm-starts SparseTabularCausalPFN head from finished Tabular v4 (best_checkpoint.pt) with
#     gentle differential LRs (temporal_lr=2e-4, tabular_lr=3e-5) + L2-SP anchor (1e-4)
#   - 50/50 mixed episodes:
#       * 50% longitudinal multimodal (nemotron_5000 split=="train" + paraphrases, N_ctx=100..5000)
#       * 50% pure-tabular replay (Tabular v4's exact 6-family prior on design_split=="train",
#         retaining full 6-class role + screener supervision -> zero tabular forgetting!)
#   - Automatically evaluates step 0, every 100 steps (32 paraphrased longitudinal + 32 tabular guard),
#     runs the 1,000-study held-out evaluation (eval_n5000_1000studies.json), and runs the complete
#     Option A vs Option B vs Run 5 head-to-head benchmark suite upon completion!
#
# Usage:
#   ./scripts/run_longitudinal_option_b_h200.sh
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_longitudinal_option_b_h200.sh
#   NUM_WORKERS=10 STEPS=1200 ./scripts/run_longitudinal_option_b_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/longitudinal_option_b_v6}"
TABULAR_V4_DIR="${TABULAR_V4_DIR:-${RUN_ROOT}/tabular_causal_pfn_v4_h200}"
BACKBONE_PATH="${BACKBONE_PATH:-${RUN_ROOT}/tinyoncbert_6l_384d}"
STRUCTURED_INIT_DIR="${STRUCTURED_INIT_DIR:-${RUN_ROOT}/longitudinal_tinyoncbert_v5}"
PARAPHRASE_BANK="${PARAPHRASE_BANK:-${REPO_ROOT}/synthetic_data/designs/nemotron_5000_paraphrases.json.gz}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-1200}"
NUM_WORKERS="${NUM_WORKERS:-6}"

echo "=== Option B (v6) Hybrid Warm-Started Multimodal Fine-Tuning ==="
echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using TABULAR_V4_DIR=${TABULAR_V4_DIR}"
echo "Using BACKBONE_PATH=${BACKBONE_PATH}"
echo "Using STRUCTURED_INIT_DIR=${STRUCTURED_INIT_DIR}"
echo "Using PARAPHRASE_BANK=${PARAPHRASE_BANK}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# ==============================================================================
# STAGE 1: 50/50 MIXED MULTIMODAL + TABULAR REPLAY FINE-TUNING (1,200 STEPS)
# ==============================================================================
"${PYTHON}" -m oci.longitudinal.option_b train \
  --run-dir "${RUN_DIR}" \
  --tabular-init-dir "${TABULAR_V4_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --structured-init-dir "${STRUCTURED_INIT_DIR}" \
  --paraphrase-bank "${PARAPHRASE_BANK}" \
  --feature-preset option_a_full \
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

# ==============================================================================
# STAGE 2: FINAL 1,000-STUDY HELD-OUT PARAPHRASED EVALUATION (EXACT RUN 1-5 PROTOCOL)
# ==============================================================================
echo "=== Running 1,000-study held-out paraphrased evaluation for Option B (v6) ==="
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

# ==============================================================================
# STAGE 3: FULL HEAD-TO-HEAD BENCHMARK (STAGE-2 + 16 LLM + 44 COMPLEX LLM + COMPLEX TABULAR)
# ==============================================================================
echo "=== Running full Option A vs Option B (v6) vs Run 5 benchmark suite ==="
"${PYTHON}" "${REPO_ROOT}/scripts/eval_option_a_v4.py" \
  --tabular-v4-dir "${TABULAR_V4_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --run5-dir "${STRUCTURED_INIT_DIR}" \
  --option-b-run-dir "${RUN_DIR}" \
  --output-dir "${RUN_DIR}/eval_full_suite" \
  --device "${DEVICE}"
