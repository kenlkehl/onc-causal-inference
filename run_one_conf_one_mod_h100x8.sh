#!/usr/bin/env bash

# One-confounder/one-modifier workflow for an eight-H100 (80 GB each) Linux VM.
# OCI owns both vLLM pools, including startup, readiness, switching, and cleanup.
# Usage: ./run_one_conf_one_mod_h100x8.sh [OUTPUT_DIR]
# Re-run with the same output directory to resume compatible checkpoints.
# Explicit saved Stage 2 launches retain their saved model and serving settings.

# Extraction defaults to cached ColBERT retrieval. Configure STAGE2_COLBERT_*
# or select STAGE2_EXTRACTION_CONTEXT_STRATEGY=full_record for fresh legacy runs.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# > 1 )); then
    echo "Usage: $0 [OUTPUT_DIR]" >&2
    exit 2
fi

if [[ -z "${OCI_RUN_CONFIG:-}" && "${STAGE2_ONLY:-0}" != "1" && "${STAGE2_RESELECT:-0}" != "1" && "${OCI_PREFLIGHT_ONLY:-0}" != "1" ]]; then
    # These are logical CUDA indices after any CUDA_VISIBLE_DEVICES mapping.
    # PHYSICAL_GPUS remains available for explicitly selecting eight host GPUs.
    if [[ -z "${PHYSICAL_GPUS:-}" ]]; then
        export GPU_COUNT="${GPU_COUNT:-8}"
    fi
    export STAGE2_MODEL="${STAGE2_MODEL:-RedHatAI/gemma-4-31B-it-FP8-dynamic}"
    export STAGE2_EXTRACTION_MODEL="${STAGE2_EXTRACTION_MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic}"
    export STAGE2_WORKERS="${STAGE2_WORKERS:-32}"
    export STAGE2_EXTRACTION_WORKERS="${STAGE2_EXTRACTION_WORKERS:-32}"

    # Each model initially runs eight single-GPU replicas across the GPU union.
    # Rapid switching falls back to four replicas per model on disjoint GPUs.
    export STAGE2_VLLM_GPUS="${STAGE2_VLLM_GPUS:-cuda:0,cuda:1,cuda:2,cuda:3}"
    export STAGE2_VLLM_GPUS_PER_SERVER="${STAGE2_VLLM_GPUS_PER_SERVER:-1}"
    export STAGE2_EXTRACTION_VLLM_GPUS="${STAGE2_EXTRACTION_VLLM_GPUS:-cuda:4,cuda:5,cuda:6,cuda:7}"
    export STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER="${STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER:-1}"
    export STAGE2_VLLM_RAPID_SWITCH_SECONDS="${STAGE2_VLLM_RAPID_SWITCH_SECONDS:-900}"
    export STAGE2_VLLM_BASE_PORT="${STAGE2_VLLM_BASE_PORT:-8010}"
    export STAGE2_EXTRACTION_VLLM_BASE_PORT="${STAGE2_EXTRACTION_VLLM_BASE_PORT:-8110}"
    export STAGE2_VLLM_INTERNAL_PORT_BASE="${STAGE2_VLLM_INTERNAL_PORT_BASE:-20000}"
    export STAGE2_EXTRACTION_VLLM_INTERNAL_PORT_BASE="${STAGE2_EXTRACTION_VLLM_INTERNAL_PORT_BASE:-30000}"

    # FP8 is read from each checkpoint's quantization config. Managed Gemma
    # servers automatically use --language-model-only and the gemma4 parser.
    # Match the inherited extraction context budget to the served model window.
    vllm_extra_args='["--gpu-memory-utilization","0.90","--max-model-len","128000","--max-num-seqs","32"]'
    export STAGE2_VLLM_EXTRA_ARGS_JSON="${STAGE2_VLLM_EXTRA_ARGS_JSON:-${vllm_extra_args}}"
    export STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON="${STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON:-${vllm_extra_args}}"
fi

if (( $# == 0 )) && [[ -z "${OCI_RUN_CONFIG:-}" ]]; then
    output_name="one_conf_one_mod_nsclc_h100x8_full"
    if [[ "${DISABLE_HTR:-0}" == "1" ]]; then
        output_name+="_no_htr"
    fi
    set -- "${repo_root}/artifacts/research_all_evidence/${output_name}"
fi

# Reuse the original scientific preset, dependency setup, and resume adapter.
exec "${repo_root}/run_one_conf_one_mod.sh" "$@"
