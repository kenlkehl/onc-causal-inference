#!/usr/bin/env bash

# Run or resume the complete Stage 1 -> 2 workflow on the bundled
# one-confounder/one-effect-modifier NSCLC cohort.
#
# Usage:
#   ./run_one_conf_one_mod.sh
#   GPU_COUNT=2 ./run_one_conf_one_mod.sh
#   PHYSICAL_GPUS=1,3 STAGE2_ENDPOINT=http://127.0.0.1:8010/v1 \
#     STAGE2_EXTRACTION_ENDPOINT=http://127.0.0.1:8020/v1 ./run_one_conf_one_mod.sh
#   GPU_COUNT=8 STAGE2_VLLM_SERVERS=8 STAGE2_MODEL=google/gemma-4-31B-it \
#     STAGE2_EXTRACTION_ENDPOINT=http://127.0.0.1:8020/v1 ./run_one_conf_one_mod.sh
#   STAGE1_ARCHITECTURES=bow_nuisance,tfidf_topics ./run_one_conf_one_mod.sh
#   STAGE2_CONSOLIDATION_MAX_ROUNDS=12 ./run_one_conf_one_mod.sh
#   STAGE2_ENDPOINT= ./run_one_conf_one_mod.sh  # Stage 1 only
#   ./run_one_conf_one_mod.sh /persistent/results/my_run
#   OCI_RUN_CONFIG=/private/saved.json STAGE2_ONLY=1 \
#     STAGE2_SELECTION_MODE=independent_tasks OCI_PREFLIGHT_ONLY=1 ./run_one_conf_one_mod.sh
#   See docs/stage2_independent_tasks.md for archived-source migration and resume.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Runtime defaults for the researcher-facing wrapper. Callers may override any
# of these through the corresponding environment variable.
export MIN_FREE_GPU_GB="${MIN_FREE_GPU_GB:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Saved-run launches inherit science and model settings from the preserved config.
if [[ -z "${OCI_RUN_CONFIG:-}" && "${STAGE2_ONLY:-0}" != "1" && "${STAGE2_RESELECT:-0}" != "1" && "${OCI_PREFLIGHT_ONLY:-0}" != "1" ]]; then
    export STAGE2_MODEL="${STAGE2_MODEL:-RedHatAI/Gemma-4-31B-IT-FP8-Dynamic}"
    export STAGE2_EXTRACTION_MODEL="${STAGE2_EXTRACTION_MODEL:-google/gemma-4-e4b-it}"
    # Match the extraction server window and reserve room for token-count differences.
    # The extractor pages long records and caps repair output within this budget.
    export STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS="${STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS:-128000}"
    export STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS="${STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS:-4096}"
    # Bound load on a single reasoning server and allow slow generations to finish.
    # All values remain overridable when targeting a faster server or replica pool.
    export STAGE2_WORKERS="${STAGE2_WORKERS:-4}"
    # Shared by interpretation and extraction, including thinking-enabled repairs.
    # Allow three full 30-minute attempts plus backoff and response repair time.
    export STAGE2_REQUEST_ATTEMPT_TIMEOUT="${STAGE2_REQUEST_ATTEMPT_TIMEOUT:-1800}"
    export STAGE2_REQUEST_TIMEOUT="${STAGE2_REQUEST_TIMEOUT:-6000}"

    # Stage 2 ontology preset for this example. Callers may override any setting
    # through the corresponding environment variable.
    export STAGE2_CONSOLIDATION_BATCH_SIZE="${STAGE2_CONSOLIDATION_BATCH_SIZE:-20}"
    export STAGE2_CONSOLIDATION_ALPHABETICAL_ROUNDS="${STAGE2_CONSOLIDATION_ALPHABETICAL_ROUNDS:-5}"
    export STAGE2_CONSOLIDATION_MAX_ROUNDS="${STAGE2_CONSOLIDATION_MAX_ROUNDS:-55}"
    export STAGE2_OPERATIONALIZATION_MAX_PROMPT_CHARS="${STAGE2_OPERATIONALIZATION_MAX_PROMPT_CHARS:-640000}"
    export STAGE2_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS="${STAGE2_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS:-3}"
    export STAGE2_MAX_ONTOLOGY_REFINEMENT_ROUNDS="${STAGE2_MAX_ONTOLOGY_REFINEMENT_ROUNDS:-2}"

fi

exec "${repo_root}/scripts/run_synthetic_all_evidence.sh" \
    synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet \
    one_conf_one_mod_nsclc_full \
    "$@"
