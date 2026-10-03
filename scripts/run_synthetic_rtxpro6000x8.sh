#!/usr/bin/env bash

# Shared hardware/serving preset for the RTX PRO 6000 Blackwell x8 quickstarts.
# Delegate scientific settings, environment setup, and resumes to the cohort
# wrappers. Explicit saved Stage 2 launches retain their preserved settings.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 {one_conf_one_mod|five_conf_five_mod} [OUTPUT_DIR]" >&2
    exit 2
fi
cohort="$1"
shift
case "$cohort" in
    one_conf_one_mod|five_conf_five_mod) ;;
    *) echo "Unknown synthetic cohort: $cohort" >&2; exit 2 ;;
esac

if [[ -z "${OCI_RUN_CONFIG:-}" && "${STAGE2_ONLY:-0}" != "1" && "${STAGE2_RESELECT:-0}" != "1" && "${OCI_PREFLIGHT_ONLY:-0}" != "1" ]]; then
    # GPU indices refer to the logical namespace after CUDA_VISIBLE_DEVICES.
    if [[ -z "${PHYSICAL_GPUS:-}" ]]; then
        export GPU_COUNT="${GPU_COUNT:-8}"
    fi
    export STAGE2_MODEL="${STAGE2_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
    # NVIDIA omits IT from this repository name; the checkpoint is instruction tuned.
    export STAGE2_EXTRACTION_MODEL="${STAGE2_EXTRACTION_MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
    export STAGE2_WORKERS="${STAGE2_WORKERS:-32}"
    export STAGE2_EXTRACTION_WORKERS="${STAGE2_EXTRACTION_WORKERS:-32}"
    export STAGE2_REQUEST_ATTEMPT_TIMEOUT="${STAGE2_REQUEST_ATTEMPT_TIMEOUT:-1800}"
    export STAGE2_REQUEST_TIMEOUT="${STAGE2_REQUEST_TIMEOUT:-6000}"
    export STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS="${STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS:-262144}"
    export STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS="${STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS:-4096}"

    # Alternate eight single-GPU replicas, with a four/four concurrent fallback.
    # Keep TP=1 for the NVIDIA 26B A4B checkpoint's supported vLLM layout.
    export STAGE2_VLLM_GPUS="${STAGE2_VLLM_GPUS:-cuda:0,cuda:1,cuda:2,cuda:3}"
    export STAGE2_VLLM_GPUS_PER_SERVER="${STAGE2_VLLM_GPUS_PER_SERVER:-1}"
    export STAGE2_EXTRACTION_VLLM_GPUS="${STAGE2_EXTRACTION_VLLM_GPUS:-cuda:4,cuda:5,cuda:6,cuda:7}"
    export STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER="${STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER:-1}"
    export STAGE2_VLLM_RAPID_SWITCH_SECONDS="${STAGE2_VLLM_RAPID_SWITCH_SECONDS:-900}"
    export STAGE2_VLLM_BASE_PORT="${STAGE2_VLLM_BASE_PORT:-8010}"
    export STAGE2_EXTRACTION_VLLM_BASE_PORT="${STAGE2_EXTRACTION_VLLM_BASE_PORT:-8110}"
    export STAGE2_VLLM_INTERNAL_PORT_BASE="${STAGE2_VLLM_INTERNAL_PORT_BASE:-20000}"
    export STAGE2_EXTRACTION_VLLM_INTERNAL_PORT_BASE="${STAGE2_EXTRACTION_VLLM_INTERNAL_PORT_BASE:-30000}"

    # vLLM reads ModelOpt/NVFP4 quantization from each checkpoint's metadata.
    # Managed Gemma servers supply --language-model-only and the gemma4 parser.
    # Use the checkpoints' native 256K window: prompt and output share this
    # budget, including the primary role's 100,000-token output allowance.
    vllm_extra_args='["--gpu-memory-utilization","0.90","--max-model-len","262144","--max-num-seqs","32"]'
    export STAGE2_VLLM_EXTRA_ARGS_JSON="${STAGE2_VLLM_EXTRA_ARGS_JSON:-${vllm_extra_args}}"
    export STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON="${STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON:-${vllm_extra_args}}"
fi

if (( $# == 0 )) && [[ -z "${OCI_RUN_CONFIG:-}" ]]; then
    output_name="${cohort}_nsclc_rtxpro6000x8_full"
    if [[ "${DISABLE_HTR:-0}" == "1" ]]; then
        output_name+="_no_htr"
    fi
    set -- "${repo_root}/artifacts/research_all_evidence/${output_name}"
fi

exec "${repo_root}/run_${cohort}.sh" "$@"
