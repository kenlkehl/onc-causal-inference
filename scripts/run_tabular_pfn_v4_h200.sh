#!/usr/bin/env bash
#
# Single-command H200 launcher for SparseTabularCausalPFN v4 (34.60M params, P = 20..3,500,
# N_ctx = 80..8,000, 480,000 unique studies across all 6 causal DGP families):
#   1. De-aliased 6-Family Interacting + Embedding Hybrid Prior:
#      - sparse_oci (20%): high-P sparse all-evidence tables + C x C, C x M, M x M interactions (20% pure-epistatic)
#      - reequation_calc (12%): OCI engine.R / reequation.py + C x C & C x M interactions
#      - nemotron_design (18%): 4,000 train-split Nemotron oncology designs + clinical interactions
#      - tree_subgroup (18%): depth 2-4 biomarker & cross-role C x M decision-tree conjunctions
#      - causal_scm (16%): layered nonlinear DAG/MLP with joint C x M HTE + observed & latent M-bias colliders
#      - interacting_clinical (16%): interaction-heavy clinical DGP (30% mathematically pure-epistatic
#        orthogonal crossover HTE!) + 50% correlated cosine-MaxSim embedding blocks (calibrating
#        directly to frozen TinyOncBERT SentKM multimodal zero-shot geometry)
#      - Verified zero missingness-rate shortcut across all 6 families (miss_rate AUROC ~0.51-0.53)
#   2. v4 Interaction Architecture (--use-v3-solver-basis --use-v4-interactions --decouple-rlearner-propensity):
#      - Round-2 O(8 * N * P) exact centered confounder-modulated Robinson scan rescuing pure C x M epistasis
#      - Symmetric C^inf variance-weighted pairwise interaction basis inside the closed-form ARD LOO-CV
#        solver (pair_top_k=16 -> 120 unordered pairs entering A_e, A_er [zero prop_loss grad!], A_m, and A_tau)
#      - Scaled capacity: col_dim=320, study_width=320, 8 study layers, 4 col-attn layers, 8 heads,
#        40 attn slots, top_k_nuis=128, top_k_cate=96, 9-point LOO-CV grid, auto gradient checkpointing
#        above N_ctx=4,000, bf16 FlashAttn SDPA
#
# Verified peak VRAM (RTX A6000 / H200): 18.68 GiB typical (N=1,800, P=1,000);
# 24.19 GiB worst-case (N_ctx=8,000, P=3,500 with auto gradient checkpointing).
#
# Host RAM note:
#   - On a dedicated H200 host, NUM_WORKERS=16 (or 20) saturates the GPU.
#   - If sharing the same host RAM with Longitudinal Run 5, use NUM_WORKERS=6.
#
# Usage:
#   ./scripts/run_tabular_pfn_v4_h200.sh
#   NUM_WORKERS=16 ./scripts/run_tabular_pfn_v4_h200.sh            # dedicated H200 host
#   NUM_WORKERS=6 ./scripts/run_tabular_pfn_v4_h200.sh             # sharing host RAM with Run 5
#   CUDA_VISIBLE_DEVICES=0 STEPS=50000 ./scripts/run_tabular_pfn_v4_h200.sh
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
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/tabular_causal_pfn_v4_h200}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-40000}"
NUM_WORKERS="${NUM_WORKERS:-12}"

echo "Using PYTHON=${PYTHON}"
echo "Using RUN_DIR=${RUN_DIR}"
echo "Using STEPS=${STEPS} NUM_WORKERS=${NUM_WORKERS} DEVICE=${DEVICE}"

# ==============================================================================
# TRAINING: LOG-UNIFORM N_ctx = 80..8,000, P = 20..3,500 ACROSS 6 CAUSAL FAMILIES
# ==============================================================================
"${PYTHON}" -m oci.tabular_pfn.cli train \
  --run-dir "${RUN_DIR}" \
  --steps "${STEPS}" \
  --min-context-size 80 \
  --context-size 8000 \
  --query-size 128 \
  --min-features 20 \
  --max-features 3500 \
  --family-weights 0.20,0.12,0.18,0.18,0.16,0.16 \
  --null-hte-prob 0.25 \
  --instrument-prob 0.70 \
  --unmeasured-conf-prob 0.25 \
  --extraction-noise-prob 0.40 \
  --fragmented-dup-prob 0.55 \
  --collider-prob 0.45 \
  --dense-conf-prob 0.15 \
  --mnar-prob 0.50 \
  --rd-scale-prob 0.25 \
  --rct-prob 0.08 \
  --max-treatment-logit-std 3.2 \
  --col-dim 320 \
  --study-width 320 \
  --study-layers 8 \
  --col-attn-layers 4 \
  --heads 8 \
  --top-k-nuis 128 \
  --top-k-cate 96 \
  --attn-slots 40 \
  --pair-top-k 16 \
  --ridge-grid-size 9 \
  --decouple-rlearner-propensity \
  --use-v3-solver-basis \
  --use-v4-interactions \
  --grad-accum-steps 12 \
  --lr 2.5e-4 \
  --warmup-steps 600 \
  --save-every 1000 \
  --eval-every 1000 \
  --eval-studies 64 \
  --eval-context-size 800 \
  --eval-num-features 500 \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

# ==============================================================================
# HELD-OUT STRESS TESTS ACROSS P = 50, 200, 500, 2000, 3500 (N_ctx = 800)
# ==============================================================================
for P in 50 200 500 2000 3500; do
  echo "=== Evaluating v4 held-out 6-family mixture P=${P} (N_ctx=800, 128 studies) ==="
  "${PYTHON}" -m oci.tabular_pfn.cli evaluate \
    --run-dir "${RUN_DIR}" \
    --studies 128 \
    --context-size 800 \
    --query-size 128 \
    --num-features "${P}" \
    --num-workers "${NUM_WORKERS}" \
    --seed 700001 \
    --device "${DEVICE}" | tee "${RUN_DIR}/eval_n800_p${P}.json"
done

# Large-cohort N_ctx = 4,000 evaluation at P = 2,000
echo "=== Evaluating v4 held-out large-N P=2000 (N_ctx=4000, 64 studies) ==="
"${PYTHON}" -m oci.tabular_pfn.cli evaluate \
  --run-dir "${RUN_DIR}" \
  --studies 64 \
  --context-size 4000 \
  --query-size 128 \
  --num-features 2000 \
  --num-workers "${NUM_WORKERS}" \
  --seed 750001 \
  --device "${DEVICE}" | tee "${RUN_DIR}/eval_n4000_p2000.json"

# Family-specific held-out checks across all 6 families (P = 500 and P = 2,000)
for FAM in sparse_oci nemotron_design reequation_calc tree_subgroup causal_scm interacting_clinical; do
  for P in 500 2000; do
    echo "=== Evaluating v4 held-out family=${FAM} P=${P} (N_ctx=800, 96 studies) ==="
    "${PYTHON}" -m oci.tabular_pfn.cli evaluate \
      --run-dir "${RUN_DIR}" \
      --studies 96 \
      --context-size 800 \
      --query-size 128 \
      --num-features "${P}" \
      --family "${FAM}" \
      --num-workers "${NUM_WORKERS}" \
      --seed 800001 \
      --device "${DEVICE}" | tee "${RUN_DIR}/eval_n800_${FAM}_p${P}.json"
  done
done
